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
    fault=getattr(args,'fault',None)
    if fault and args.case not in {'android_browser','linux_browser'}:
        raise ValueError('Recovery fixture conditions apply only to browser pilots')
    if fault:
        import fault_browser_pilot as browser_provider
    else:
        import browser_pilot as browser_provider
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
    if fault:
        manifest.update(fault_condition=fault,fault_sources={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['fault_browser_pilot.py','fault_browser_server.js']})
    dependencies=args.runtime/'device-dependencies-canonical-20261004.json'
    if dependencies.exists():
        # This is a separate dependency upgrade, not a rewrite of the private
        # core manifest or any earlier pilot's declared configuration.
        declaration=json.loads(dependencies.read_text())
        from device_receipt_qualification import owned_gateway
        from qw3_computer.device_registry import load_registry
        item=load_registry(ROOT/'apps/computer_use/.runtime/cross-device-catalogue.json')[desired]
        live=owned_gateway(item)
        declared=next(row for row in declaration['devices'] if row['catalogue_id']==desired)
        entry=Path(__file__).with_name('device_gateway_entry.py')
        if (live['pid']!=declared['pid'] or live['command']!=declared['command']
                or hashlib.sha256(entry.read_bytes()).hexdigest()!=declaration['entry_sha256']):
            raise ValueError('Declared candidate gateway dependency changed')
        manifest['device_dependency_upgrade']={'sha256':hashlib.sha256(dependencies.read_bytes()).hexdigest(),
            'entry_sha256':declaration['entry_sha256'],'gateway_pid':live['pid'],'gateway_device_id':live['device_id']}
    save(directory / "pilot-manifest.json", manifest)
    video = None
    fixture = None
    browser_fixture = None
    task_id = None
    recording = False
    result=None
    deadline_reached=False
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
                options={'condition':fault} if fault else {}
                browser_fixture,prompt=browser_provider.initialize('android',identity_value,directory,**options)
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
                options={'condition':fault} if fault else {}
                browser_fixture,prompt=browser_provider.initialize('linux',identity_value,directory,linux_execute,**options)
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
        event_source=args.runtime/'codex-events.jsonl'
        event_offset=event_source.stat().st_size if event_source.exists() else 0
        manifest['codex_event_offset_bytes']=event_offset
        if fault:
            manifest['fault_armed']=browser_provider.arm(browser_fixture,linux_execute)
            save(directory/'pilot-manifest.json',manifest)
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
                    deadline_reached=True
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
            independent=browser_provider.grade(browser_fixture,linux_execute)
            save(directory/'independent-browser.json',independent)
            result['artifact_passed']=independent['passed']
            if fault:
                response=browser_provider.response_grade(independent,state,deadline_reached=deadline_reached)
                save(directory/'independent-fault-response.json',response)
                result.update(fault_condition=fault,fault_exercised=response['fault_exercised'],
                              fault_response_passed=response['passed'])
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
        if fault:result['fault_accepted']=result['fault_response_passed'] and result['video_collection_passed']
        if (directory/'continuous.mp4').exists():
            result["video_bytes"] = (directory / "continuous.mp4").stat().st_size
        save(directory / "task.json", state)
        for name in ["codex-events.jsonl"]:
            source = args.runtime / name
            if source.exists():
                # Device selection starts a fresh thread, but the app event
                # journal is append-only across sessions. Export this attempt.
                with source.open('rb') as log:
                    log.seek(event_offset)
                    (directory/name).write_bytes(log.read())
        save(directory / "integration-result.json", result)
        return result
    except Exception as exc:
        failed = {"kind": "integration-pilot-not-formal", "status": "infrastructure-failed",
                  "error": type(exc).__name__ + ": " + str(exc), "unassisted_passed": False}
        result=failed
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
                save(directory/'browser-service-stop.json',browser_provider.stop(browser_fixture,linux_execute))
            except Exception as exc:
                save(directory/'browser-service-stop.json',{'stopped':False,'error':type(exc).__name__+': '+str(exc)})
        if task_id and getattr(args,'retire_task',False):
            # Probe-only cleanup: keep the original task.json and durable
            # archive, stop only our terminal task, then reset the idle device
            # entry so the next queued model trial has explicit admission.
            try:
                retirement=retire_task(args.app,task_id,desired)
                save(directory/'task-retirement.json',retirement)
                if result is not None:result['owned_task_retired']=True
            except Exception as exc:
                save(directory/'task-retirement.json',{'task_id':task_id,'stopped':False,'error':type(exc).__name__+': '+str(exc)})
                if result is not None:
                    result.update(owned_task_stop_unconfirmed=True,accepted=False)
                    if fault:result['fault_accepted']=False
            if result is not None:save(directory/'integration-result.json',result)


def retire_task(app,task_id,desired):
    current=request(app,'/api/state')
    if (current.get('id')!=task_id or current['status'] in {'starting','running','verifying'}
            or current.get('uncertain_actions')):
        raise ValueError('Probe task retirement requires its identified reconciled terminal state')
    stopped=request(app,'/api/control',{'command':'stop'})
    if stopped.get('id')!=task_id or stopped['status']!='stopped' or stopped.get('desktop_paused') is not True:
        raise ValueError('Probe task stop not acknowledged')
    reset=request(app,'/api/device',{'device_id':desired})
    if reset['status']!='idle' or reset.get('uncertain_actions'):
        raise ValueError('Probe device reset not acknowledged')
    return {'task_id':task_id,'stopped':True,'idle':True,'archive_preserved':True}


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
    p.add_argument('--fault',choices=['slow_loading','focus_dialog','no_progress'],
                   help='Separate declared browser recovery probe, never an OSWorld score')
    p.add_argument('--retire-task',action='store_true',
                   help='Archive and stop only this probe task, then restore its idle device entry')
    args = p.parse_args()
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
