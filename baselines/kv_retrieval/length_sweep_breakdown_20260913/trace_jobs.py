from pathlib import Path
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
PYTHON=HERE.parent/'.venv/bin/python'

def save(d):
    p=HERE/'trace_status.json';p.with_suffix('.tmp').write_text(json.dumps(d,indent=2)+'\n');p.with_suffix('.tmp').replace(p)

def main():
    assert not (HERE/'trace_status.json').exists()
    state=dict(status='waiting_measurements',pid=os.getpid(),started=time.time(),jobs=[]);save(state)
    env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
    env['NSYS_NVTX_PROFILER_REGISTER_ONLY']='0'
    try:
        while True:
            d=json.loads((HERE/'followup_status.json').read_text())
            if d['status']=='error':raise RuntimeError('Measurement campaign failed')
            if d['status']=='complete':break
            time.sleep(5)
        # Use InfLLM's supplied-query chunk path. Earlier strict causal append
        # results are retained as an adapter diagnostic, not the headline query.
        name='infllm_chunk_query';prefix=HERE/name
        cmd=[str(PYTHON),'-u',str(HERE/'run_external.py'),'--method','infllm','--query-only',
             '--lengths','8192','16384','32768','65536','131072','245760','--out',str(prefix.with_suffix('.json'))]
        state.update(status='running',current_job=name,command=cmd);save(state)
        with prefix.with_suffix('.log').open('x') as log:
            p=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
            state['child_pid']=p.pid;save(state);code=p.wait()
        state['jobs'].append(dict(name=name,exit_code=code));save(state)
        if code:raise RuntimeError(name+' failed')
        for i,(method,mtp) in enumerate((m,t) for m in ['kvmem','qw3','infllm','infinigen'] for t in [0,4]):
            name=f'trace_query_{method}_mtp{mtp}';prefix=HERE/name
            cmd=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none',
                 '--capture-range-end=stop','--cuda-graph-trace=node','--force-overwrite=false','-o',str(prefix)]
            if method in ['qw3','kvmem']:
                cmd+=['--capture-range=nvtx','--nvtx-capture=measured_query',str(PYTHON),str(HERE/'run_native_query.py'),
                      '--method',method,'--mtp',str(mtp),'--lengths','131072','--reps','1','--port',str(18550+i),'--trace','--out',str(prefix)]
            else:
                cmd+=['--capture-range=cudaProfilerApi',str(PYTHON),str(HERE/'run_external.py'),'--method',method,
                      '--lengths','131072','--trace','--trace-query','--trace-mtp',str(mtp),'--out',str(prefix.with_suffix('.json'))]
            state.update(status='running',current_job=name,command=cmd);save(state)
            with prefix.with_suffix('.log').open('x') as log:
                p=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
                state['child_pid']=p.pid;save(state);code=p.wait()
            state['jobs'].append(dict(name=name,exit_code=code));save(state)
            if code:raise RuntimeError(name+' failed')
            with (HERE/(name+'_export.log')).open('x') as log:
                subprocess.run(['nsys','export','--type=sqlite','--output',str(prefix.with_suffix('.sqlite')),str(prefix.with_suffix('.nsys-rep'))],stdout=log,stderr=log,check=True)
        state['status']='complete'
    except BaseException as e:
        state.update(status='error',error=repr(e));raise
    finally:
        state['updated']=time.time();save(state)

if __name__=='__main__':main()
