"""Pause the authorized OSWorld suite after the current arm has finished."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[2] / 'benchmark/osworld_v2/claudecode_remaining_20260912'


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    initial = json.loads((SUITE / 'suite_status.json').read_text())
    assert initial['status'] == 'running' and alive(initial['pid'])
    assert not (HERE / 'pause.json').exists()
    task, arm = initial['task'], initial['arm']
    result_path = SUITE / 'trials' / task / 'full' / arm / 'result.json'
    state = dict(status='waiting_for_arm_end', task=task, arm=arm,
        suite_pid=initial['pid'], started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save(HERE / 'pause.json', state)
    while alive(initial['pid']):
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get('finished_at'):
                os.kill(initial['pid'], signal.SIGTERM)
                state['completed_boundary_result'] = str(result_path)
                break
        time.sleep(.1)
    deadline = time.monotonic() + 120
    while alive(initial['pid']) and time.monotonic() < deadline:
        time.sleep(1)
    assert not alive(initial['pid']), 'Supervisor did not exit'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    preserved = {}
    for path in (SUITE / 'trials').glob('*/full/*/result.json'):
        result = json.loads(path.read_text())
        if result.get('status') == 'complete' and result.get('finished_at'):
            preserved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    state.update(status='paused', completed_result_sha256=preserved,
        paused_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ended_suite_state=json.loads((SUITE / 'suite_status.json').read_text()))
    save(HERE / 'pause.json', state)
    print(json.dumps({k:v for k,v in state.items() if k not in ('completed_result_sha256','ended_suite_state')}), flush=True)


if __name__ == '__main__':
    main()
