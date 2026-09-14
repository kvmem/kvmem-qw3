"""Restore the authorized frozen suite after all exclusive GPU work exits."""
from common import HERE, ROOT
import hashlib
import json
import os
import subprocess
import time
import traceback

SUITE=ROOT/'benchmark/osworld_v2/claudecode_remaining_20260912'
PAUSE=SUITE/'throughput_pause_external_mtp_20260913'


def read(path):return json.loads(path.read_text())


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def verify_completed(pause):
    for name,expected in pause['completed_result_sha256'].items():
        from pathlib import Path
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name


def main():
    from pathlib import Path
    status=dict(status='preflight',started=time.time(),pid=os.getpid())
    state_path=HERE/'resume_status.json';save(state_path,status)
    try:
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        pause=read(PAUSE/'pause.json');assert pause['status']=='paused'
        verify_completed(pause)
        task,arm=pause['interrupted_task'],pause['interrupted_arm']
        out=SUITE/'trials'/task/'full'/arm
        assert not out.exists()
        workspace=Path(read(SUITE/'trials'/task/'workspace.json')['path'])
        assert not workspace.exists() or not any(workspace.iterdir())
        env=dict(os.environ)
        for k in list(env):
            if k.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(k)
        with (HERE/'osworld_preflight.log').open('w') as log:
            subprocess.run(['/usr/bin/python3',str(SUITE/'run_suite.py'),'--prepare-only'],
                cwd=ROOT,env=env,stdout=log,stderr=log,check=True,timeout=300)
        assert not (PAUSE/'resume_launch.json').exists()
        with (PAUSE/'resume.log').open('x') as log:
            child=subprocess.Popen(['/usr/bin/python3',str(SUITE/'run_suite.py')],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        save(PAUSE/'resume_launch.json',dict(pid=child.pid,started=time.time(),task=task,arm=arm,report=str(HERE/'REPORT.md')))
        status.update(status='verifying_requests',suite_pid=child.pid);save(state_path,status)
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError(f'OSWorld exited {child.returncode}')
            suite=read(SUITE/'suite_status.json')
            if suite.get('pid')==child.pid and suite.get('status')=='needs_attention':
                raise RuntimeError(str(suite.get('error')))
            events=[]
            if (out/'api.jsonl').exists():
                for line in (out/'api.jsonl').read_text().splitlines():
                    try:e=json.loads(line)
                    except json.JSONDecodeError:continue
                    if e.get('status')==200 and e.get('first_content_ns'):events.append(e)
            if events and suite.get('task')==task and suite.get('arm')==arm:
                verify_completed(pause)
                pause.update(status='resumed',resumed_at=time.time(),resume_pid=child.pid,
                    completed_results_unchanged=True,first_resumed_request_id=events[0].get('id'),report=str(HERE/'REPORT.md'))
                save(PAUSE/'pause.json',pause)
                status.update(status='complete',finished=time.time(),task=task,arm=arm,
                    completed_results_unchanged=True,successful_requests=len(events),server_pid=suite.get('server_pid'))
                save(state_path,status);return
            time.sleep(5)
        raise TimeoutError('OSWorld did not produce a successful request within 15 minutes')
    except BaseException as exc:
        status.update(status='needs_attention',error=repr(exc),traceback=traceback.format_exc())
        save(state_path,status);raise


if __name__=='__main__':main()
