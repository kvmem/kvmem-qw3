from pathlib import Path
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON=HERE.parent/'.venv/bin/python'

def save(d):
    p=HERE/'followup_status.json';p.with_suffix('.tmp').write_text(json.dumps(d,indent=2)+'\n');p.with_suffix('.tmp').replace(p)

def main():
    assert not (HERE/'followup_status.json').exists()
    state=dict(status='waiting_native',pid=os.getpid(),started=time.time(),jobs=[]);save(state)
    env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
    def job(name,cmd):
        output=HERE/(name+'.json') if name.startswith('external_') else HERE/name/'result.json'
        if output.exists():
            previous=json.loads(output.read_text())
            assert previous['status']=='complete',(name,previous['status'])
            state['jobs'].append(dict(name=name,exit_code=0,reused_completed_result=str(output)))
            save(state);return
        state.update(status='running',current_job=name,command=cmd);save(state)
        with (HERE/(name+'.log')).open('x') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
            state['child_pid']=p.pid;save(state);code=p.wait()
        state['jobs'].append(dict(name=name,exit_code=code));save(state)
        if code:raise RuntimeError(name+' failed')
    try:
        while True:
            d=json.loads((HERE/'native_status.json').read_text())
            if d['status']=='error':raise RuntimeError('native matrix failed')
            if d['status']=='complete':break
            time.sleep(5)
        for method in ['infllm','infinigen']:
            job('external_'+method,[str(PYTHON),'-u',str(HERE/'run_external.py'),'--method',method,
                '--out',str(HERE/f'external_{method}.json'),'--lengths','8192','16384','32768','65536','131072','245760'])
        for i,(method,mtp) in enumerate([('qw3',0),('kvmem',0),('kvmem',4),('qw3',4)]):
            job(f'query_{method}_mtp{mtp}',[str(PYTHON),'-u',str(HERE/'run_native_query.py'),
                '--method',method,'--mtp',str(mtp),'--out',str(HERE/f'query_{method}_mtp{mtp}'),'--port',str(18540+i)])
        state['status']='complete'
    except BaseException as e:
        state.update(status='error',error=repr(e));raise
    finally:
        state['updated']=time.time();save(state)

if __name__=='__main__':main()
