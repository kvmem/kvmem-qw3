"""Wait for primary measurements, then thread confirmation and Nsight traces."""
from pathlib import Path
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON=HERE.parent/'.venv/bin/python'


def save(data):
    temp=HERE/'followup_status.tmp'
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(HERE/'followup_status.json')


def main():
    assert not (HERE/'followup_status.json').exists()
    state={'status':'waiting_primary','pid':os.getpid(),'started':time.time(),'jobs':[]}
    save(state)
    try:
        while True:
            try:primary=json.loads((HERE/'jobs_status.json').read_text())
            except json.JSONDecodeError:
                time.sleep(1);continue
            if primary['status']=='error':raise RuntimeError('Primary experiment failed')
            if primary['status']=='formal_complete':break
            time.sleep(5)
        job_list=[('thread_confirmation',[str(PYTHON),str(HERE/'check_threads.py')])]
        chosen=json.loads((HERE/'formal_infinigen.json').read_text())['chosen_threads']
        for mtp,mode in ((0,'full'),(4,'full'),(4,'optimized_bookkeeping')):
            name=f'trace_mtp{mtp}_{mode}'
            command=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                '--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-graph-trace=node',
                '--force-overwrite=false','-o',str(HERE/name),str(PYTHON),str(HERE/'run_experiment.py'),
                '--trace-mtp',str(mtp),'--trace-mode',mode,'--threads',str(chosen[str(mtp)]),
                '--out',str(HERE/(name+'.json'))]
            job_list.append((name,command))
        for name,command in job_list:
            state.update(status='running',current_job=name);save(state)
            with (HERE/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=log)
                state['child_pid']=child.pid;save(state)
                code=child.wait()
            state['jobs'].append({'name':name,'exit_code':code});save(state)
            if code:raise RuntimeError(name+' failed')
            if name.startswith('trace_'):
                with (HERE/(name+'_export.log')).open('x') as log:
                    subprocess.run(['nsys','export','--type=sqlite','--output',str(HERE/(name+'.sqlite')),
                        str(HERE/(name+'.nsys-rep'))],stdout=log,stderr=log,check=True)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        state['updated']=time.time();save(state)


if __name__=='__main__':main()
