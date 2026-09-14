"""Wait for exclusive Q8 measurements, export, and restore the authorized suite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SUITE = ROOT / 'benchmark/osworld_v2/claudecode_remaining_20260912'
PAUSE = SUITE / 'throughput_pause_q8mtp_20260913'


def read(path):
    return json.loads(path.read_text())


def verify_completed(pause):
    for name, expected in pause['completed_result_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name


def main():
    state = dict(status='waiting_for_measurements', pid=os.getpid(), started=time.time())
    def save():
        tmp = HERE / 'finish_status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(HERE / 'finish_status.json')
    save()
    try:
        while True:
            measured = read(HERE / 'runs/status.json')
            if measured['status'] == 'complete':
                break
            if measured['status'] != 'running':
                raise RuntimeError('Measurement failed: ' + str(measured.get('error')))
            time.sleep(5)
        with (HERE / 'export.log').open('w') as log:
            subprocess.run([sys.executable, str(HERE / 'export_report.py')],
                           cwd=ROOT, stdout=log, stderr=log, check=True, timeout=120)
        state.update(status='osworld_preflight', updated=time.time())
        save()
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        pause = read(PAUSE / 'pause.json')
        verify_completed(pause)
        task, arm = pause['interrupted_task'], pause['interrupted_arm']
        out = SUITE / 'trials' / task / 'full' / arm
        assert not out.exists(), 'Unexpected resumed arm already exists'
        workspace = Path(read(SUITE / 'trials' / task / 'workspace.json')['path'])
        assert not workspace.exists() or not any(workspace.iterdir())
        with (HERE / 'osworld_preflight.log').open('w') as log:
            subprocess.run(['/usr/bin/python3', str(SUITE / 'run_suite.py'), '--prepare-only'],
                           cwd=ROOT, stdout=log, stderr=log, check=True, timeout=300)
        env = dict(os.environ)
        for key in list(env):
            if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
                env.pop(key)
        launch_path = PAUSE / 'resume_launch.json'
        assert not launch_path.exists()
        with (PAUSE / 'resume.log').open('x') as log:
            child = subprocess.Popen(['/usr/bin/python3', str(SUITE / 'run_suite.py')],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        launch_path.write_text(json.dumps(dict(pid=child.pid, started=time.time(),
            task=task, arm=arm, report=str(HERE / 'REPORT.md')), indent=2) + '\n')
        state.update(status='verifying_osworld_progress', osworld_pid=child.pid)
        save()
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError('Resumed OSWorld exited: ' + str(child.returncode))
            suite = read(SUITE / 'suite_status.json')
            if suite.get('pid') == child.pid and suite.get('status') == 'needs_attention':
                raise RuntimeError(str(suite.get('error')))
            events = []
            api = out / 'api.jsonl'
            if api.exists():
                for line in api.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('status') == 200 and event.get('first_content_ns'):
                        events.append(event)
            if events and suite.get('task') == task and suite.get('arm') == arm:
                verify_completed(pause)
                pause.update(status='resumed', resumed_at=time.time(), resume_pid=child.pid,
                    completed_results_unchanged=True, first_resumed_request_id=events[0].get('id'),
                    comparison_report=str(HERE / 'REPORT.md'))
                (PAUSE / 'pause.json').write_text(json.dumps(pause, indent=2) + '\n')
                state.update(status='complete', finished=time.time(),
                    osworld_progress=dict(task=task, arm=arm, successful_requests=len(events),
                        server_pid=suite.get('server_pid'), driver_pid=suite.get('driver_pid')))
                save()
                return
            time.sleep(5)
        raise TimeoutError('No successful resumed OSWorld request within 15 minutes')
    except BaseException as exc:
        state.update(status='needs_attention', error=repr(exc), traceback=traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
