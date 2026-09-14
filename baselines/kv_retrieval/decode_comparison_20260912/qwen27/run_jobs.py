"""Exclusive, serial, auditable benchmark jobs; no automatic failure retries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);a=p.parse_args()
    plan=json.loads(a.plan.read_text());out=Path(plan['output']);out.mkdir(exist_ok=True,parents=True)
    state={'status':'running','pid':os.getpid(),'started':time.time(),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'completed':[]}
    child=None
    def save():
        tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
    def stop(sig,_):
        raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,stop)
    env=dict(os.environ,HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    for key in list(env):
        if key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key)
    os.sched_setaffinity(0,set(range(16))|set(range(32,48)))
    state['cpu_affinity']=sorted(os.sched_getaffinity(0))
    try:
        for job in plan['jobs']:
            for filename,expected in plan.get('source_sha256',{}).items():
                assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
            path=Path(job['result'])
            if path.exists():
                assert json.loads(path.read_text())['status']=='complete',f'Inspect incomplete attempt: {path}'
                state['completed'].append(job['name']);save();continue
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
            state.update(current=job['name'],command=job['command'],updated=time.time());save()
            print('START',job['name'],time.time(),flush=True)
            with (out/(job['name']+'.log')).open('w') as log:
                child=subprocess.Popen(job['command'],stdout=log,stderr=log,env=env,start_new_session=True)
                state['child_pid']=child.pid;save()
                code=child.wait(timeout=job.get('timeout_s',2400))
            print('END',job['name'],code,time.time(),flush=True)
            assert code==0 and path.exists() and json.loads(path.read_text())['status']=='complete',f'Failed job: {job["name"]}'
            state['completed'].append(job['name']);save()
        state['status']='complete'
    except BaseException as exc:
        state.update(status='needs_attention',error=repr(exc),traceback=traceback.format_exc())
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    finally:
        state['ended']=time.time();save()


if __name__=='__main__':main()
