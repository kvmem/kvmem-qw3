"""Run the validated implementations sequentially, retaining every trial."""
from common import HERE, BASE, ROOT
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile
import time


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n')
    temp.replace(path)


def main():
    out = HERE/'formal'
    out.mkdir()
    sources = list(HERE.glob('*.py'))+[HERE/'q8_consistent/build.json',HERE/'q8_consistent/mmvq_q8.cu']
    with tarfile.open(out/'source.tar.gz','w:gz') as tar:
        for f in sources:
            tar.add(f,arcname=str(f.relative_to(HERE)))
    plan = dict(started=time.time(),pid=os.getpid(),status='running',jobs=[],
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
        scope='pure-text decode throughput; authors retrieval in Qwen adapters; not agent task accuracy')
    save(out/'status.json',plan)
    env = dict(os.environ)
    for k in list(env):
        if k.lower() in ('http_proxy','https_proxy','all_proxy'):
            env.pop(k)
    for method in ('infllm','infinigen'):
        command = ['taskset','-c','0-15,32-47',str(BASE/'.venv/bin/python'),str(HERE/'run.py'),
            '--method',method,'--out',str(out/(method+'.json')),'--graphs','--lengths',
            '8192','32768','122880','245760','--tokens','128','--reps','3']
        job = dict(method=method,command=command,started=time.time(),status='running')
        plan['jobs'].append(job)
        with (out/(method+'.log')).open('x') as log:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
            job['pid']=process.pid;save(out/'status.json',plan)
            code=process.wait()
        job.update(ended=time.time(),returncode=code,status='complete' if code==0 else 'error')
        if code:
            plan.update(status='needs_attention',error=f'{method} exited {code}')
            save(out/'status.json',plan)
            return
        save(out/'status.json',plan)
    plan.update(status='complete',finished=time.time())
    save(out/'status.json',plan)


if __name__=='__main__':
    main()
