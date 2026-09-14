from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parents[1]

def save(state):
    p = HERE / 'native_status.json'
    p.with_suffix('.tmp').write_text(json.dumps(state, indent=2) + '\n')
    p.with_suffix('.tmp').replace(p)

def main():
    assert json.loads((HERE/'pause.json').read_text())['status'] == 'paused'
    assert not (HERE/'native_status.json').exists()
    env = {k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
    os.sched_setaffinity(0, set(range(16)) | set(range(32,48)))
    script = BASE/'decode_comparison_20260912/q8_mtp_20260913/run_native.py'
    state = dict(status='running', pid=os.getpid(), started=time.time(), jobs=[], source_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    save(state)
    try:
        for i,(method,mtp) in enumerate([('qw3',0),('kvmem',0),('kvmem',4),('qw3',4)]):
            name=f'{method}_mtp{mtp}'
            cmd=[str(ROOT/'.venv/bin/python'), '-u', str(script), '--method',method,'--mtp',str(mtp),
                 '--out',str(HERE/'native'/name),'--lengths','8192','16384','32768','65536','131072',
                 '--reps','3','--tokens','128','--port',str(18530+i)]
            (HERE/'native').mkdir(exist_ok=True)
            state.update(current_job=name,command=cmd);save(state)
            with (HERE/f'native_{name}.log').open('x') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log,cwd=ROOT)
                state['child_pid']=proc.pid;save(state)
                code=proc.wait()
            state['jobs'].append(dict(name=name,exit_code=code));save(state)
            if code:raise RuntimeError(f'{name} failed')
        state['status']='complete'
    except BaseException as e:
        state.update(status='error',error=repr(e));raise
    finally:
        state['updated']=time.time();save(state)

if __name__=='__main__':main()
