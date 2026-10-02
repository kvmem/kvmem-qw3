"""P0 fixed Windows operations, independent of model decisions.

Restore the owned VM once, then administratively reset neutral application
fixtures before each trial. Only /act performs the operations under test.
The original executor is unchanged; UIA locates known controls for this
infrastructure check and is not credited as model visual grounding.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import aiohttp

from qw3_computer.desktop import Desktop
from osworld_windows.native_run import require_evidence_space, snapshot_qualification
from osworld_windows.source import ROOT, sha256
from windows_suite import command, stop


APP = ROOT / "apps/computer_use"
SHARE = APP / ".runtime/windows/shared"
CATEGORIES = ("launch_focus", "known_control_click", "literal_text", "save_as_preserve_original")
LITERAL = "QW3 local input 0123456789 {}[]%+^\n中文 Café 日本語 🙂\n"
ORIGINAL = "D0 original must remain unchanged.\r\n"
REVISED = "D0 revised copy, saved through the GUI.\r\n"


def save(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


async def wait_json(path, timeout=90):
    async with asyncio.timeout(timeout):
        while not path.is_file():
            await asyncio.sleep(.3)
        return json.loads(path.read_text(encoding="utf-8-sig"))


async def prepare_trial(output, trial_id, category):
    """Setup writes input files only; never write a requested final artifact."""
    source = f"{trial_id}-original.txt"
    initial = "edge" if category == "known_control_click" else "notepad"
    document = rf"C:\Qw3Computer\outputs\{source}" if category == "save_as_preserve_original" else ""
    script = r'''$ErrorActionPreference='Stop'
try {
    # This VM was restored under the exclusive trial lock. These processes
    # belong only to preceding diagnostic fixtures; their evidence is saved.
    Get-Process notepad,msedge,et,wpscloudsvr -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Milliseconds 500
    $root='C:\Qw3Computer'
    $entries=Join-Path $root 'application-entries.json'
    if (Test-Path $entries) {Move-Item -LiteralPath $entries -Destination ($entries+'.diagnostic-backup-TRIAL_ID')}
    $source=Join-Path "$root\outputs" 'SOURCE_FILE'
    if (Test-Path $source) {throw 'Source fixture already exists'}
    [IO.File]::WriteAllText($source, "D0 original must remain unchanged.`r`n", [Text.UTF8Encoding]::new($false))
    if ('INITIAL_APP' -eq 'edge') {
        $app="${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"
        Start-Process $app -ArgumentList @('--no-first-run','--no-default-browser-check','--disable-background-networking','--disable-sync','--force-renderer-accessibility','--user-data-dir=C:\Qw3Computer\BrowserProfile','file:///C:/Qw3Computer/fixture.html')
    } else {
        if ('INITIAL_DOCUMENT') {Start-Process notepad.exe -ArgumentList 'INITIAL_DOCUMENT'}
        else {Start-Process notepad.exe}
    }
    Start-Sleep -Seconds 2
    $result=@{prepared=$true;source_sha256=(Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant();source=$source;initial_app='INITIAL_APP';executor_sha256=(Get-FileHash "$root\executor.ps1" -Algorithm SHA256).Hash.ToLowerInvariant()}
} catch {$result=@{prepared=$false;error=$_.Exception.Message}}
[IO.File]::WriteAllText('\\host.lan\Data\TRIAL_ID-prepared.json', ($result | ConvertTo-Json -Depth 5), [Text.UTF8Encoding]::new($false))
'''.replace("TRIAL_ID", trial_id).replace("SOURCE_FILE", source).replace("INITIAL_APP", initial).replace("INITIAL_DOCUMENT", document)
    path = output / "prepare.ps1"
    path.write_text(script, encoding="utf-8-sig")
    await command(sys.executable, APP / "vm/provision.py", path)
    result = await wait_json(SHARE / f"{trial_id}-prepared.json")
    save(output / "prepared.json", result)
    if not result.get("prepared"):
        raise RuntimeError("Neutral fixture preparation failed: " + result.get("error", "unknown"))
    if result["executor_sha256"] != sha256(APP / "vm/executor.ps1"):
        raise RuntimeError("Guest executor differs from frozen source")
    await asyncio.sleep(1)
    return result


async def trial(desktop, output, trial_id, category, repetition):
    output.mkdir()
    result = dict(id=trial_id, category=category, repetition=repetition, status="running",
                  passed=False, model_used=False, started_monotonic_ns=time.monotonic_ns(),
                  actions=[], phase="preparation")
    save(output / "result.json", result)
    desktop.task_id = trial_id
    counter = 0

    async def act(action):
        nonlocal counter
        counter += 1
        before = desktop.latest
        record = dict(action=action, phase=result["phase"], before=before, started_monotonic_ns=time.monotonic_ns())
        result["actions"].append(record)
        save(output / "result.json", result)
        response = await desktop.act(dict(desktop_id=before["desktop_id"],
            observation_id=before["observation_id"], action_id=f"{trial_id}-{counter}", action=action))
        record.update(ended_monotonic_ns=time.monotonic_ns(), after=desktop.latest,
                      no_progress=response.get("no_progress", False))
        save(output / "result.json", result)
        return response

    async def key(name):
        return await act(dict(kind="key", key=name))

    async def control(name, kind):
        deadline = time.monotonic() + 8
        while True:
            matches = [c for c in desktop.latest.get("accessibility", {}).get("controls", [])
                       if c["name"] == name and c["kind"] == kind]
            if len(matches) == 1:
                return matches[0]
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Expected exactly one visible {name}/{kind}; found {len(matches)}")
            await asyncio.sleep(.3)
            await desktop.observe()

    async def click_control(name, kind):
        target = await control(name, kind)
        return await act(dict(kind="click", x=target["center_x"], y=target["center_y"], button="left", clicks=1))

    async def save_file(filename, save_as=False):
        await key("CTRL+SHIFT+S" if save_as else "CTRL+S")
        await click_control("File name:", "Edit")
        await key("CTRL+A")
        await act(dict(kind="type", text="C:\\Qw3Computer\\outputs\\" + filename))
        await click_control("Save", "Button")
        deadline = time.monotonic() + 8
        while True:
            files = {f["name"]: f for f in (await desktop.request("GET", "/files"))["files"]}
            if filename in files:
                return files
            if time.monotonic() >= deadline:
                raise TimeoutError("Saved artifact not found")
            await asyncio.sleep(.3)

    def check_file(actual, text):
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
        raw = normalized.encode()
        return actual["content"] == normalized and actual["size"] == len(raw) and actual["sha256"] == hashlib.sha256(raw).hexdigest()

    async def wait_foreground(process, title=None):
        deadline = time.monotonic() + 8
        while True:
            front = desktop.latest["foreground"]
            if front["process"].lower() == process and (title is None or title in front["title"]):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("Prepared foreground/document identity not confirmed")
            await asyncio.sleep(.3)
            await desktop.observe()

    try:
        await desktop.control("pause")
        result["preparation"] = await prepare_trial(output, trial_id, category)
        await desktop.control("resume")
        await desktop.observe()
        expected_front = "msedge" if category == "known_control_click" else "notepad"
        await wait_foreground(expected_front)
        if category == "literal_text":
            # Windows Notepad can restore a previously saved document. A
            # process-name match alone is not a blank-document precondition.
            await key("CTRL+N")
            await wait_foreground("notepad", "Untitled")
        if category == "save_as_preserve_original":
            # A launch argument can open a background tab/window. Establish
            # this trial's input through the visible Open dialog, and verify
            # its title before any edit. This remains fixture preparation.
            await key("CTRL+O")
            await click_control("File name:", "Edit")
            await key("CTRL+A")
            await act(dict(kind="type", text=f"C:\\Qw3Computer\\outputs\\{trial_id}-original.txt"))
            await click_control("Open", "Button")
            await wait_foreground("notepad", f"{trial_id}-original")
        result["initial_observation"] = desktop.latest
        result["phase"] = "operation"
        if category == "launch_focus":
            result["focus_checks"] = []
            for app, processes in (("edge", ["msedge"]), ("notepad", ["notepad"]), ("wps_spreadsheets", ["et", "wps"])):
                await act(dict(kind="open_app", app=app))
                at_return = desktop.latest
                await asyncio.sleep(2)
                await desktop.observe()
                def matches(observation):
                    front = observation["foreground"]
                    return front["process"].lower() in processes and (app != "wps_spreadsheets" or "Workbook" in front["title"])
                result["focus_checks"].append(dict(app=app, at_return=at_return["foreground"],
                    settled=desktop.latest["foreground"], at_return_matches=matches(at_return),
                    settled_matches=matches(desktop.latest)))
            result["passed"] = all(x["at_return_matches"] and x["settled_matches"] for x in result["focus_checks"])
        elif category == "known_control_click":
            await click_control("Name", "Edit")
            target = await control("Name", "Edit")
            result["clicked_control"] = target
            result["passed"] = bool(target["focused"])
        elif category == "literal_text":
            await act(dict(kind="type", text=LITERAL))
            result["phase"] = "save_and_grade"
            name = trial_id + "-typed.txt"
            files = await save_file(name)
            result["artifact"] = files[name]
            result["passed"] = check_file(files[name], LITERAL)
        else:
            await key("CTRL+A")
            await act(dict(kind="type", text=REVISED))
            name, source = trial_id + "-copy.txt", trial_id + "-original.txt"
            files = await save_file(name, save_as=True)
            result["artifacts"] = {k: files.get(k) for k in (name, source)}
            result["checks"] = dict(copy_correct=check_file(files[name], REVISED),
                original_unchanged=source in files and check_file(files[source], ORIGINAL),
                distinct_content=source in files and files[name]["sha256"] != files[source]["sha256"])
            result["passed"] = all(result["checks"].values())
        result["status"] = "passed" if result["passed"] else "failed"
    except Exception as exc:
        result.update(status="infra_invalid" if result["phase"] == "preparation" else "failed",
                      error=f"{type(exc).__name__}: {exc}", unresolved_actions=desktop.unresolved())
    finally:
        await desktop.control("pause")
        result.update(ended_monotonic_ns=time.monotonic_ns(), final_observation=desktop.latest)
        save(output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("id", "category", "status")}), flush=True)
    return result


async def run(args):
    output = args.output.resolve()
    if not re.fullmatch(r"[a-zA-Z0-9_-]{8,40}", output.name):
        raise ValueError("Use a new short alphanumeric run identity")
    output.mkdir(parents=True, exist_ok=False)
    lock = open("/tmp/qw3-osworld-vm.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report = dict(kind="local_diagnostic_p0_executor", status="preparing", model_used=False,
                  model_task_success_claimed=False, snapshot=args.snapshot,
                  planned_trials=len(args.categories) * args.repetitions, categories=args.categories,
                  repetitions=args.repetitions, trials=[])
    recorder = None
    desktop = None
    try:
        info = json.loads(await command("docker", "inspect", "qw3-computer-windows"))[0]
        if info["Config"].get("Labels", {}).get("com.qw3.computer-use.owner") != "codex-goal-20260930":
            raise ValueError("VM ownership mismatch")
        storage = Path(next(m["Source"] for m in info["Mounts"] if m["Destination"] == "/storage"))
        report["space"] = require_evidence_space(output, SHARE, storage)
        report["snapshot_qualification"] = snapshot_qualification(APP, args.snapshot)
        token = (SHARE / "executor-token.txt").read_text().strip()
        async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=10)) as client:
            async def health():
                async with client.get("http://127.0.0.1:18082/health", headers={"Authorization": "Bearer " + token}) as r:
                    r.raise_for_status()
                    return await r.json()
            before = await health()
            if not before["paused"]:
                raise ValueError("Executor is active; cannot restore VM")
            report["before_restore_health"] = before
            async with client.get("http://127.0.0.1:18082/observe", headers={"Authorization": "Bearer " + token}) as r:
                r.raise_for_status()
                before_image = await r.json()
            import base64
            (output / "before-restore.png").write_bytes(base64.b64decode(before_image.pop("image")))
            save(output / "before-restore.json", before_image)
            sources = [*sorted((APP / "qw3_computer").glob("*.py")), APP / "vm/executor.ps1", Path(__file__)]
            report["source_commit"] = (await command("git", "rev-parse", "HEAD")).strip()
            report["source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
            report["restored_executor_only"] = True
            save(output / "result.json", report)
            print("restoring clean executor baseline", flush=True)
            await command("docker", "stop", "-t", "120", "qw3-computer-windows")
            archive = await command(sys.executable, APP / "vm/archive_share.py", "shared-archive-" + output.name)
            (output / "prior-share-archive.json").write_text(archive)
            await command(sys.executable, APP / "vm/snapshot.py", "restore", "--name", args.snapshot)
            await command("docker", "start", "qw3-computer-windows")
            async with asyncio.timeout(180):
                while True:
                    try:
                        initial = await health()
                        if not initial["paused"]:
                            raise ValueError("Restored executor is active")
                        break
                    except (aiohttp.ClientError, TimeoutError):
                        await asyncio.sleep(1)
            report["restored_health"] = initial
        env = dict(os.environ, PYTHONPATH=str(APP))
        with (output / "recorder.log").open("wb") as log:
            recorder = await asyncio.create_subprocess_exec(sys.executable, "-m", "vm.record_console",
                "--output", str(output / "video"), "--ffmpeg", str(args.ffmpeg),
                cwd=ROOT, env=env, stdout=log, stderr=log)
        async with asyncio.timeout(30):
            while True:
                if recorder.returncode is not None:
                    raise RuntimeError("Recorder exited during startup")
                manifest = output / "video/recording.json"
                if manifest.is_file() and json.loads(manifest.read_text()).get("started_monotonic_ns"):
                    break
                await asyncio.sleep(.3)
        desktop = Desktop("http://127.0.0.1:18082", token, output / "desktop")
        await desktop.start()
        report["status"] = "running"
        save(output / "result.json", report)
        # Executor readiness can precede the shell's first interactive launch.
        # This fixed setup wait is outside operation latency measurements.
        await asyncio.sleep(8)
        for category in args.categories:
            category_index = CATEGORIES.index(category) + 1
            for repetition in range(1, args.repetitions + 1):
                require_evidence_space(output, SHARE, storage)
                trial_id = f"{output.name}-{category_index}-{repetition}"
                result = await trial(desktop, output / trial_id, trial_id, category, repetition)
                report["trials"].append(result)
                save(output / "result.json", report)
                if desktop.unresolved():
                    raise RuntimeError("Uncertain action retained; reconcile before further live trials")
        report["status"] = "complete"
        report["passed_trials"] = sum(t["passed"] for t in report["trials"])
    except asyncio.CancelledError:
        report.update(status="interrupted", error="Diagnostic cancelled; preserve this attempt and use a new run identity")
        raise
    except Exception as exc:
        report.update(status="halted", error=f"{type(exc).__name__}: {exc}")
    finally:
        if desktop:
            try:
                await desktop.control("pause")
            finally:
                await desktop.close()
        await stop(recorder)
        recording = output / "video/recording.json"
        if recording.is_file():
            report["recording"] = json.loads(recording.read_text())
        report["completed_trials"] = len(report["trials"])
        save(output / "result.json", report)
        lock.close()
    print(json.dumps({k: report.get(k) for k in ("status", "completed_trials", "passed_trials", "error")}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", default="qw3-osworld-client15-20261002")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--categories", choices=CATEGORIES, nargs="+", default=list(CATEGORIES))
    parser.add_argument("--repetitions", type=int, choices=range(1, 4), default=3)
    result = asyncio.run(run(parser.parse_args()))
    raise SystemExit(0 if result["status"] == "complete" else 1)
