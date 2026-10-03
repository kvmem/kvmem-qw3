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


def fixture_identity(directory):
    from browser_pilot import identity
    # Scheduler interludes all use a nested "pilot" evidence directory. Its
    # parent is the attempt identity; using only "pilot" creates ambiguous
    # contacts and colliding browser fixture folders in later repetitions.
    value=directory.parent.name+'-'+directory.name if directory.name=='pilot' else directory.name
    return identity(value)


def run(args):
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    identity_value=fixture_identity(directory)
    state = request(args.app, "/api/state")
    if state["status"] not in {"idle", "completed", "failed", "stopped"} or state["uncertain_actions"]:
        raise ValueError("Finish the existing task and reconcile uncertain input before a pilot")
    android = args.case.startswith('android_')
    info = owned_container("qw3-computer-android" if android else "qw3-computer-linux",
                           "qw3.goal", "cross-device-20261003")
    desired = "android-emulator" if android else "linux-vm"
    state = request(args.app, "/api/device", {"device_id": desired})
    defaults={'android_settings':180,'android_contacts':600,'linux_text':360,
              'android_browser':900,'linux_browser':900,'linux_calc':900}
    requested_seconds=getattr(args,'seconds',None)
    seconds=requested_seconds if requested_seconds is not None else defaults[args.case]
    if type(seconds) is not int or not 60<=seconds<=1800:
        raise ValueError('Pilot deadline must be60..1800seconds')
    prompt = ANDROID_PROMPT if android else LINUX_PROMPT
    manifest = {"kind": "integration-pilot-not-formal", "case": args.case, "prompt": prompt,
                "deadline_seconds": seconds, "container_id": info["Id"], "app_runtime": str(args.runtime.resolve()),
                "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'video_mode':args.video_mode if android else getattr(args,'linux_video_mode','official'),
                'model_output_tokens':state.get('model_output_tokens',4096),
                "no_model_access_to_initializer_or_grader": True}
    manifest['fixture_sources']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ['browser_pilot.py','browser_pilot_server.js','calc_pilot.py','android_contacts_pilot.py']}
    manifest['recording_sources']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                 for name in ['linux_recording.py','vm/record_console.py','vm/vnc_admin.py','live_android_recording.py']}
    save(directory / "pilot-manifest.json", manifest)
    video = None
    fixture = None
    browser_fixture = None
    task_id = None
    recording = False
    try:
        if android:
            metadata = json.loads((ROOT / "apps/computer_use/.runtime/android/emulator-container-manifest.json").read_text())
            if metadata["container_id"] != info["Id"]:
                raise ValueError("Android emulator identity changed")
            if args.case=='android_settings':
                android_command(["shell", "settings", "put", "system", "screen_off_timeout", "2147483647"], capture_output=True)
                android_command(["shell", "am", "force-stop", "com.android.settings"], capture_output=True)
                android_command(["shell", "am", "start", "-a", "android.settings.SETTINGS"], capture_output=True)
                time.sleep(2)
                before_settings = android_system_settings()
                save(directory / "initial-system-settings.json", before_settings)
            elif args.case=='android_contacts':
                from android_contacts_pilot import initialize
                fixture,prompt=initialize(android_command,identity_value)
                save(directory/'contacts-fixture.json',fixture)
                manifest['prompt']=prompt
                save(directory/'pilot-manifest.json',manifest)
                android_command(['shell','am','force-stop','com.google.android.contacts'],capture_output=True)
                android_command(['shell','am','start','-a','android.intent.action.MAIN',
                                 '-c','android.intent.category.LAUNCHER','-p','com.google.android.contacts'],capture_output=True)
            else:
                from browser_pilot import initialize
                browser_fixture,prompt=initialize('android',identity_value,directory)
                manifest['prompt']=prompt
                save(directory/'pilot-manifest.json',manifest)
                android_command(['shell','am','start','-a','android.intent.action.VIEW',
                                 '-d',browser_fixture['url'],'-p','com.android.chrome'],capture_output=True)
            if args.video_mode=='live':
                from live_android_recording import LiveAndroidRecording as Recorder
            else:
                from android_recording import AndroidRecording as Recorder
            video=Recorder(['docker','exec','qw3-computer-android',
                str(ROOT/'apps/computer_use/.runtime/android/sdk/platform-tools/adb'),'-s','emulator-5556'],directory)
            video.start()
        else:
            if args.case=='linux_browser':
                from browser_pilot import initialize
                browser_fixture,prompt=initialize('linux',identity_value,directory,linux_execute)
                manifest['prompt']=prompt
                save(directory/'pilot-manifest.json',manifest)
                linux_execute('import subprocess;subprocess.Popen('+repr(['google-chrome','--new-window',browser_fixture['url']])+
                              ',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)')
            elif args.case=='linux_calc':
                from calc_pilot import initialize
                fixture,prompt=initialize(linux_execute,directory)
                save(directory/'calc-fixture.json',fixture)
                manifest['prompt']=prompt
                save(directory/'pilot-manifest.json',manifest)
            else:
                initialize_linux_text(directory)
            if getattr(args,'linux_video_mode','official')=='vnc':
                from linux_recording import LinuxRecording
                video=LinuxRecording(directory,info['Id'])
                video.start()
            else:
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
        if android and args.case=='android_settings':
            value = android_command(["shell", "settings", "get", "system", "screen_off_timeout"],
                                    capture_output=True, text=True).stdout.strip()
            result.update(independent_setting_value=value, artifact_passed=value == "300000")
            after_settings = android_system_settings()
            save(directory / "final-system-settings.json", after_settings)
            others = {key: {"before": before_settings.get(key), "after": after_settings.get(key)}
                      for key in set(before_settings) | set(after_settings)
                      if key != "screen_off_timeout" and before_settings.get(key) != after_settings.get(key)}
            result.update(other_system_setting_changes=others, artifact_passed=value == "300000" and not others)
        elif args.case=='android_contacts':
            from android_contacts_pilot import grade
            independent=grade(android_command,fixture)
            save(directory/'independent-contacts.json',independent)
            result['artifact_passed']=independent['passed']
        elif browser_fixture:
            from browser_pilot import grade
            independent=grade(browser_fixture,linux_execute)
            save(directory/'independent-browser.json',independent)
            result['artifact_passed']=independent['passed']
        elif args.case=='linux_calc':
            from calc_pilot import grade
            independent=grade(linux_execute,fixture,directory)
            save(directory/'independent-workbook.json',independent)
            result['artifact_passed']=independent['passed']
        else:
            import base64
            artifact = json.loads(linux_execute(
                "from pathlib import Path; import json,base64; p=Path('/home/user/Desktop/qw3-pilot-note.txt'); "
                "print(json.dumps({'exists':p.exists(),'data':base64.b64encode(p.read_bytes()).decode() if p.exists() else None}))")["output"])
            data = base64.b64decode(artifact["data"]) if artifact["exists"] else None
            save(directory / "independent-artifact.json", artifact)
            result.update(artifact_passed=data == LINUX_EXPECTED,
                          artifact_bytes=len(data) if data is not None else None)
        # Preserve the independent task result if media collection fails. A
        # missing video is incomplete acceptance evidence, not a fabricated
        # zero score or loss of the saved application data.
        try:
            if android:
                result['video_collection_passed']=video.finish()['passed']
            elif video is not None:
                result['video_collection_passed']=video.finish()['complete']
            else:
                collect_linux_recording(directory/'continuous.mp4')
                recording=False
                result['video_collection_passed']=True
        except Exception as exc:
            result.update(video_collection_passed=False,video_collection_error=type(exc).__name__+': '+str(exc))
        result["unassisted_passed"] = result["artifact_passed"] and state["status"] == "completed" and not result["uncertain_actions"]
        result['accepted']=result['unassisted_passed'] and result['video_collection_passed']
        if (directory/'continuous.mp4').exists():
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
        if video is not None and not video.collected:
            try:
                video.finish('interrupted.mp4')
            except Exception as exc:
                save(directory/'recording-error.json',{'error':type(exc).__name__+': '+str(exc)})
        if recording:
            try:
                collect_linux_recording(directory/'interrupted.mp4')
            except Exception as exc:
                save(directory / "recording-error.json", {"error": type(exc).__name__})
        if browser_fixture:
            try:
                from browser_pilot import stop
                save(directory/'browser-service-stop.json',stop(browser_fixture,linux_execute))
            except Exception as exc:
                save(directory/'browser-service-stop.json',{'stopped':False,'error':type(exc).__name__+': '+str(exc)})


def collect_linux_recording(path):
    import urllib.request
    req=urllib.request.Request('http://127.0.0.1:18126/end_recording',data=b'{}',headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=45) as response:
        data=response.read()
    if len(data)<32 or data[4:8]!=b'ftyp':
        raise ValueError('The recorder did not return an MP4 container')
    path.write_bytes(data)


def initialize_linux_text(directory):
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=["android_settings", "android_contacts", "android_browser", "linux_text", "linux_browser", "linux_calc"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--app", default="http://127.0.0.1:18205")
    p.add_argument('--seconds',type=int,help='Explicit60..1800s task deadline; record longer probes separately from prior pilots')
    p.add_argument('--video-mode',choices=['live','native'],default='live',
                   help='Live1Hz measured captures preserve static waits; native video may omit a stationary tail')
    p.add_argument('--linux-video-mode',choices=['vnc','official'],default='vnc',
                   help='Timestamped read-only VNC sampling; official server recordings can omit long waits')
    args = p.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
