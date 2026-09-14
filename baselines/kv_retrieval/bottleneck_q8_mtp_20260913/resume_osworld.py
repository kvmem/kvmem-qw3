"""Resume the original frozen suite and verify a real successful API request."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[2] / 'benchmark/osworld_v2/claudecode_remaining_20260912'


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    pause = json.loads((HERE/'pause.json').read_text())
    assert pause['status']=='paused'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for path, expected in pause['completed_result_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    out=SUITE/'trials'/pause['interrupted_task']/'full'/pause['interrupted_arm']
    assert not out.exists()
    env=dict(os.environ)
    for key in list(env):
        if key.lower().endswith('_proxy'):
            env.pop(key)
    subprocess.run(['/usr/bin/python3',str(SUITE/'run_suite.py'),'--prepare-only'],env=env,check=True,cwd=HERE.parents[2])
    with (HERE/'resume.log').open('x') as log:
        child=subprocess.Popen(['/usr/bin/python3',str(SUITE/'run_suite.py')],cwd=HERE.parents[2],env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    state={'status':'verifying_requests','suite_pid':child.pid,'task':pause['interrupted_task'],
        'arm':pause['interrupted_arm'],'started':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(HERE/'resume_status.json',state)
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        if child.poll() is not None:
            raise RuntimeError('OSWorld supervisor exited during resume')
        path=out/'api.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try: item=json.loads(line)
                except json.JSONDecodeError: continue
                if item.get('status')==200 and item.get('first_content_ns'):
                    for name,expected in pause['completed_result_sha256'].items():
                        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
                    state.update(status='complete',first_request_id=item.get('id'),completed_results_unchanged=True,
                        finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    save(HERE/'resume_status.json',state)
                    pause.update(status='resumed',resume_pid=child.pid,resumed_at=state['finished'])
                    save(HERE/'pause.json',pause)
                    print(json.dumps(state),flush=True)
                    return
        time.sleep(5)
    raise TimeoutError('No successful request observed within 15 minutes')


if __name__=='__main__':main()
