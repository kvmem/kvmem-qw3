"""Finish the guided-query supplement, audit, then restore the paused suite."""
from pathlib import Path
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent

def save(d):
    p=HERE/'finish_status.json';p.with_suffix('.tmp').write_text(json.dumps(d,indent=2)+'\n');p.with_suffix('.tmp').replace(p)

def main():
    assert not (HERE/'finish_status.json').exists()
    state=dict(status='waiting_traces',pid=os.getpid(),started=time.time(),jobs=[]);save(state)
    env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
    def job(name,cmd):
        state.update(status='running',current_job=name,command=cmd);save(state)
        with (HERE/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
            state['child_pid']=p.pid;save(state);code=p.wait()
        state['jobs'].append(dict(name=name,exit_code=code));save(state)
        if code:raise RuntimeError(name+' failed')
    try:
        while True:
            d=json.loads((HERE/'trace_status.json').read_text())
            if d['status']=='error':raise RuntimeError('Trace campaign failed')
            if d['status']=='complete':break
            time.sleep(5)
        job('guided_query_kvmem_mtp4',[str(HERE.parent/'.venv/bin/python'),'-u',str(HERE/'run_native_query.py'),
            '--method','kvmem','--mtp','4','--guided-query','--lengths','131072','245760','--reps','1',
            '--port','18570','--out',str(HERE/'guided_query_kvmem_mtp4')])
        for name in ['analyze','analyze_traces','audit_results']:
            job('final_'+name,['/usr/bin/python3',str(HERE/(name+'.py'))])
        job('restore_osworld',['/usr/bin/python3',str(HERE/'resume_osworld.py')])
        state['status']='complete'
    except BaseException as e:
        state.update(status='error',error=repr(e));raise
    finally:
        state['updated']=time.time();save(state)

if __name__=='__main__':main()
