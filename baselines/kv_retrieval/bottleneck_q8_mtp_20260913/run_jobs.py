"""Serial exclusive GPU jobs; preserve failures for inspection."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import tarfile

HERE=Path(__file__).resolve().parent
PYTHON=HERE.parent/'.venv/bin/python'


def save(data):
    (HERE/'jobs_status.json').write_text(json.dumps(data,indent=2)+'\n')


def main():
    assert not (HERE/'jobs_status.json').exists()
    smoke=json.loads((HERE/'smoke_optimized.json').read_text())
    assert smoke['status']=='complete'
    assert all(row['exact_output_vs_prior_fresh_prefill'] for row in smoke['measurements'])
    assert json.loads((HERE/'pause.json').read_text())['status']=='paused'
    with tarfile.open(HERE/'formal_source.tar.gz','w:gz') as archive:
        for path in HERE.glob('*.py'):
            archive.add(path,arcname=path.name)
    state={'status':'running','pid':os.getpid(),'started':time.time(),'jobs':[],
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    save(state)
    try:
        for method in ('infinigen','infllm'):
            command=[str(PYTHON),str(HERE/'run_experiment.py'),'--method',method,'--out',str(HERE/f'formal_{method}.json')]
            state['current_job']=method;save(state)
            with (HERE/f'formal_{method}.log').open('x') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=log)
                state['child_pid']=proc.pid;save(state)
                code=proc.wait()
            state['jobs'].append({'name':method,'exit_code':code});save(state)
            if code:raise RuntimeError(f'{method} failed')
        state['status']='formal_complete'
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        state['updated']=time.time();save(state)


if __name__=='__main__':main()
