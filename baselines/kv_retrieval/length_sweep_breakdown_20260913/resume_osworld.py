from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parents[1]

def save(path,d):
    path.with_suffix('.tmp').write_text(json.dumps(d,indent=2)+'\n');path.with_suffix('.tmp').replace(path)

def main():
    pause=json.loads((HERE/'pause.json').read_text());suite=Path(pause['suite'])
    assert pause['status']=='paused'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    for name,h in pause['completed_result_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
    out=suite/'trials'/pause['task']/'full'/pause['arm']
    if pause['interrupted']:assert not out.exists(),str(out)
    env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
    subprocess.run(['/usr/bin/python3',str(suite/'run_suite.py'),'--prepare-only'],cwd=ROOT,env=env,check=True)
    with (HERE/'resume.log').open('x') as log:
        p=subprocess.Popen(['/usr/bin/python3',str(suite/'run_suite.py')],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    d=dict(status='verifying',pid=p.pid,task=pause['task'],arm=pause['arm'],started=time.time());save(HERE/'resume_status.json',d)
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        if p.poll() is not None:raise RuntimeError('Resumed supervisor exited')
        if (out/'api.jsonl').exists():
            for line in (out/'api.jsonl').read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if r.get('status')==200 and r.get('first_content_ns'):
                    for name,h in pause['completed_result_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
                    d.update(status='complete',first_request_id=r.get('id'),completed_result_hashes_unchanged=True,finished=time.time())
                    save(HERE/'resume_status.json',d)
                    pause.update(status='resumed',resume_pid=p.pid);save(HERE/'pause.json',pause)
                    print(json.dumps(d));return
        time.sleep(5)
    raise TimeoutError('No successful resumed request')

if __name__=='__main__':main()
