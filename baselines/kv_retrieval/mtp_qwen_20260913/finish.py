"""Export complete measurements and immediately resume OSWorld."""
from common import HERE, ROOT
import json
import os
import subprocess
import time
import traceback


def main():
    state=dict(status='waiting',pid=os.getpid(),started=time.time())
    def save():
        p=HERE/'finish_status.json';t=p.with_suffix('.tmp')
        t.write_text(json.dumps(state,indent=2)+'\n');t.replace(p)
    save()
    try:
        while True:
            trial=json.loads((HERE/'formal/status.json').read_text())
            if trial['status']=='complete':break
            if trial['status']!='running':raise RuntimeError(str(trial.get('error',trial['status'])))
            time.sleep(5)
        state['status']='exporting';save()
        with (HERE/'export.log').open('w') as log:
            subprocess.run(['/usr/bin/python3',str(HERE/'export_report.py')],cwd=ROOT,
                           stdout=log,stderr=log,check=True,timeout=180)
        state['status']='resuming_osworld';save()
        with (HERE/'resume.log').open('w') as log:
            subprocess.run(['/usr/bin/python3',str(HERE/'resume_osworld.py')],cwd=ROOT,
                           stdout=log,stderr=log,check=True,timeout=1500)
        state.update(status='complete',finished=time.time());save()
    except BaseException as exc:
        state.update(status='needs_attention',error=repr(exc),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
