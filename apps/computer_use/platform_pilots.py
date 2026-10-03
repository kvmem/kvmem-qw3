"""Owner-only cross-device integration pilots, independent checks and video.

Administrative initialization and result reads are outside Computer MCP.
These short development tasks are not OSWorld scores or broad acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from osworld_linux.runner import request, save
from osworld_windows.source import ROOT


ANDROID_PROMPT = ("In Android Settings, set the screen timeout to 5 minutes. Verify the selected setting "
                  "after closing the choice dialog. Do not change any other settings.")
LINUX_PROMPT = ("Create and save /home/user/Desktop/qw3-pilot-note.txt in the text editor. The complete file "
                "must contain exactly two lines: first line Local computer use; second line QW3 Windows Linux Android. "
                "Include a newline after each line, including the final line. Verify the saved filename and both lines "
                "in the editor. Do not use a terminal.")
LINUX_EXPECTED = b"Local computer use\nQW3 Windows Linux Android\n"


def owned_container(name, label, value):
    info = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]
    if info["Config"].get("Labels", {}).get(label) != value:
        raise ValueError("Device container owner does not match this task")
    return info


def android_command(arguments, **kwargs):
    sdk = ROOT / "apps/computer_use/.runtime/android/sdk/platform-tools/adb"
    return subprocess.run(["docker", "exec", "qw3-computer-android", str(sdk),
                           "-s", "emulator-5556", *arguments], check=True, **kwargs)


def android_system_settings():
    raw = android_command(["shell", "settings", "list", "system"], capture_output=True, text=True).stdout
    return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)


def linux_execute(code):
    result = request("http://127.0.0.1:18126", "/execute",
                     {"command": ["python", "-c", code], "shell": False})
    if result.get("returncode") != 0 or result.get("status") != "success":
        raise RuntimeError("Pilot initialization/result read failed: " + str(result))
    return result


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    state = request(args.app, "/api/state")
    if state["status"] not in {"idle", "completed", "failed", "stopped"} or state["uncertain_actions"]:
        raise ValueError("Finish the existing task and reconcile uncertain input before a pilot")
    android = args.case == "android_settings"
    info = owned_container("qw3-computer-android" if android else "qw3-computer-linux",
                           "qw3.goal", "cross-device-20261003")
    desired = "android-emulator" if android else "linux-vm"
    state = request(args.app, "/api/device", {"device_id": desired})
    seconds = 180 if android else 360
    prompt = ANDROID_PROMPT if android else LINUX_PROMPT
    manifest = {"kind": "integration-pilot-not-formal", "case": args.case, "prompt": prompt,
                "deadline_seconds": seconds, "container_id": info["Id"], "app_runtime": str(args.runtime.resolve()),
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "no_model_access_to_initializer_or_grader": True}
    save(directory / "pilot-manifest.json", manifest)
    video = None
    task_id = None
    recording = False
    try:
        if android:
            metadata = json.loads((ROOT / "apps/computer_use/.runtime/android/emulator-container-manifest.json").read_text())
            if metadata["container_id"] != info["Id"]:
                raise ValueError("Android emulator identity changed")
            android_command(["shell", "settings", "put", "system", "screen_off_timeout", "2147483647"], capture_output=True)
            android_command(["shell", "am", "force-stop", "com.android.settings"], capture_output=True)
            android_command(["shell", "am", "start", "-a", "android.settings.SETTINGS"], capture_output=True)
            time.sleep(2)
            before_settings = android_system_settings()
            save(directory / "initial-system-settings.json", before_settings)
            remote = "/sdcard/qw3-pilot-" + str(int(time.time() * 1000)) + ".mp4"
            command = ["docker", "exec", "qw3-computer-android",
                       str(ROOT / "apps/computer_use/.runtime/android/sdk/platform-tools/adb"), "-s", "emulator-5556",
                       "shell", "screenrecord", "--bit-rate", "2000000", "--time-limit", "180", remote]
            with (directory / "recording.log").open("wb") as log:
                video = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        else:
            # Preserve the previous task artifact before removing that exact file.
            code = ("from pathlib import Path; import json,base64; p=Path('/home/user/Desktop/qw3-pilot-note.txt'); "
                    "print(json.dumps({'exists':p.exists(),'data':base64.b64encode(p.read_bytes()).decode() if p.exists() else None}))")
            previous = json.loads(linux_execute(code)["output"])
            save(directory / "previous-artifact.json", previous)
            if previous["exists"]:
                import base64
                digest = hashlib.sha256(base64.b64decode(previous["data"])).hexdigest()
                code = ("from pathlib import Path; import hashlib; p=Path('/home/user/Desktop/qw3-pilot-note.txt'); "
                        "assert hashlib.sha256(p.read_bytes()).hexdigest()==" + repr(digest) + "; p.unlink()")
                linux_execute(code)
            # A fresh test document, without an old modified buffer.
            linux_execute("import subprocess,time; subprocess.run(['pkill','-TERM','-x','gedit'],check=False); time.sleep(1); "
                          "subprocess.Popen(['gedit'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)")
            request("http://127.0.0.1:18126", "/start_recording", {}, timeout=30)
            recording = True
        started = time.monotonic()
        state = request(args.app, "/api/task", {"prompt": prompt})
        task_id = state["id"]
        manifest.update(task_id=task_id, started_at=time.time(), context_window=state.get("model_context_window"))
        save(directory / "pilot-manifest.json", manifest)
        with (directory / "progress.jsonl").open("w") as log:
            while True:
                state = request(args.app, "/api/state")
                if state["id"] != task_id:
                    raise RuntimeError("Pilot task was replaced by another owner")
                log.write(json.dumps({"at": time.time(), "status": state["status"], "tools": state["tool_calls_used"]}) + "\n")
                log.flush()
                if state["status"] in {"completed", "failed", "paused", "stopped"}:
                    break
                if time.monotonic() - started >= seconds:
                    state = request(args.app, "/api/control", {"command": "stop"})
                    break
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader"], capture_output=True, text=True)
                with (directory / "gpu.csv").open("a") as f:
                    f.write(gpu.stdout)
                time.sleep(2)
        result = {"kind": "integration-pilot-not-formal", "app_status": state["status"],
                  "seconds": time.monotonic() - started, "tool_calls": state["tool_calls_used"],
                  "turn_error": state.get("turn_error"), "uncertain_actions": len(state["uncertain_actions"])}
        if android:
            value = android_command(["shell", "settings", "get", "system", "screen_off_timeout"],
                                    capture_output=True, text=True).stdout.strip()
            result.update(independent_setting_value=value, artifact_passed=value == "300000")
            after_settings = android_system_settings()
            save(directory / "final-system-settings.json", after_settings)
            others = {key: {"before": before_settings.get(key), "after": after_settings.get(key)}
                      for key in set(before_settings) | set(after_settings)
                      if key != "screen_off_timeout" and before_settings.get(key) != after_settings.get(key)}
            result.update(other_system_setting_changes=others, artifact_passed=value == "300000" and not others)
            video.wait(timeout=max(10, 195 - (time.monotonic() - started)))
            with (directory / "continuous.mp4").open("wb") as f:
                android_command(["exec-out", "cat", remote], stdout=f)
            result["video_collection_passed"] = video.returncode == 0
        else:
            import base64
            artifact = json.loads(linux_execute(
                "from pathlib import Path; import json,base64; p=Path('/home/user/Desktop/qw3-pilot-note.txt'); "
                "print(json.dumps({'exists':p.exists(),'data':base64.b64encode(p.read_bytes()).decode() if p.exists() else None}))")["output"])
            data = base64.b64decode(artifact["data"]) if artifact["exists"] else None
            save(directory / "independent-artifact.json", artifact)
            result.update(artifact_passed=data == LINUX_EXPECTED,
                          artifact_bytes=len(data) if data is not None else None)
            import urllib.request
            req = urllib.request.Request("http://127.0.0.1:18126/end_recording", data=b"{}",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=45) as r:
                (directory / "continuous.mp4").write_bytes(r.read())
            recording = False
            result["video_collection_passed"] = True
        result["unassisted_passed"] = result["artifact_passed"] and state["status"] == "completed" and not result["uncertain_actions"]
        result["video_bytes"] = (directory / "continuous.mp4").stat().st_size
        save(directory / "task.json", state)
        for name in ["codex-events.jsonl"]:
            source = args.runtime / name
            if source.exists():
                shutil.copy2(source, directory / name)
        save(directory / "integration-result.json", result)
        return result
    except Exception as exc:
        failed = {"kind": "integration-pilot-not-formal", "status": "infrastructure-failed",
                  "error": type(exc).__name__ + ": " + str(exc), "unassisted_passed": False}
        if task_id:
            try:
                current = request(args.app, "/api/state")
                if current.get("id") == task_id and current["status"] in {"starting", "running"}:
                    current = request(args.app, "/api/control", {"command": "stop"})
                save(directory / "task-at-failure.json", current)
            except Exception:
                failed["owned_task_stop_unconfirmed"] = True
        save(directory / "integration-result.json", failed)
        return failed
    finally:
        if recording:
            # The native recorder stops on the bounded VM endpoint. Preserve
            # failure details; a partial recording is not a passed task.
            try:
                import urllib.request
                req = urllib.request.Request("http://127.0.0.1:18126/end_recording", data=b"{}",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=45) as r:
                    (directory / "interrupted.mp4").write_bytes(r.read())
            except Exception as exc:
                save(directory / "recording-error.json", {"error": type(exc).__name__})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["android_settings", "linux_text"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--app", default="http://127.0.0.1:18205")
    args = p.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
