"""Finish authorized measurements, export, then resume the paused OSWorld suite."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SUITE=ROOT/'benchmark/osworld_v2/claudecode_remaining_20260912'


def read(path):return json.loads(path.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    state={'status':'waiting_for_primary','pid':os.getpid(),'started':time.time()}
    def save():
        tmp=HERE/'pipeline_status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(HERE/'pipeline_status.json')
    def run(name,cmd,timeout):
        state.update(status=name,updated=time.time());save()
        with (HERE/(name+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    save()
    try:
        while True:
            current=read(HERE/'formal/status.json')
            if current['status']=='complete':break
            if current['status']!='running':raise RuntimeError(f'Primary measurement needs attention: {current.get("error")}')
            if not Path(f'/proc/{current["pid"]}').exists():raise RuntimeError('Primary supervisor exited without completion')
            time.sleep(5)
        run('matched_budget_measurement',[sys.executable,str(HERE/'run_jobs.py'),str(HERE/'matched_budget_jobs.json')],3600)
        run('export_primary',[sys.executable,str(HERE/'export_results.py'),'--plan',str(HERE/'formal_jobs.json'),'--out',str(HERE/'export')],180)
        run('export_matched_budget',[sys.executable,str(HERE/'export_results.py'),'--plan',str(HERE/'matched_budget_jobs.json'),'--out',str(HERE/'matched_budget_export')],180)
        run('write_report',[sys.executable,str(HERE/'write_report.py')],180)
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU still occupied'
        pause_path=SUITE/'throughput_pause_20260912/pause.json';pause=read(pause_path)
        for filename,expected in pause['completed_result_sha256'].items():assert sha(Path(filename))==expected,filename
        workspace=Path(read(SUITE/'trials/002/workspace.json')['path'])
        assert not workspace.exists() or not any(workspace.iterdir()),'Fresh interrupted-arm workspace required'
        assert not (SUITE/'trials/002/full/kvmem').exists(),'Inspect existing interrupted arm'
        run('osworld_preflight',['/usr/bin/python3',str(SUITE/'run_suite.py'),'--prepare-only'],300)
        launch_path=SUITE/'resume_launch.json';assert not launch_path.exists(),'A resume launch already exists'
        env=dict(os.environ)
        for key in list(env):
            if key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key)
        with (SUITE/'resume_after_throughput.log').open('x') as log:
            child=subprocess.Popen(['/usr/bin/python3',str(SUITE/'run_suite.py')],cwd=ROOT,env=env,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch={'pid':child.pid,'started':time.time(),'command':['/usr/bin/python3',str(SUITE/'run_suite.py')],
                'comparison_report':str(HERE/'REPORT.md'),'primary_audit':read(HERE/'export/audit.json'),
                'matched_budget_audit':read(HERE/'matched_budget_export/audit.json')}
        launch_path.write_text(json.dumps(launch,indent=2))
        state.update(status='verifying_osworld_progress',osworld_pid=child.pid);save()
        deadline=time.monotonic()+900
        while time.monotonic()<deadline:
            if child.poll() is not None:raise RuntimeError(f'OSWorld resume exited: {child.returncode}')
            current=read(SUITE/'suite_status.json')
            if current.get('pid')==child.pid and current.get('status')=='needs_attention':raise RuntimeError(current.get('error'))
            api=SUITE/'trials/002/full/kvmem/api.jsonl'
            events=[]
            if api.exists():
                for line in api.read_text().splitlines():
                    try:events.append(json.loads(line))
                    except json.JSONDecodeError:pass
            successful=[e for e in events if e.get('status')==200 and e.get('first_content_ns')]
            if successful and current.get('task')=='002' and current.get('arm')=='kvmem':
                for filename,expected in pause['completed_result_sha256'].items():assert sha(Path(filename))==expected,filename
                pause.update(status='resumed',resumed_at=time.time(),resume_pid=child.pid,completed_results_unchanged=True,
                             comparison_report=str(HERE/'REPORT.md'),first_resumed_request_id=successful[0].get('id'))
                pause_path.write_text(json.dumps(pause,indent=2))
                state.update(status='complete',osworld_progress={'task':'002','arm':'kvmem','successful_requests':len(successful),
                             'server_pid':current.get('server_pid'),'driver_pid':current.get('driver_pid')},ended=time.time());save()
                return
            time.sleep(5)
        raise TimeoutError('OSWorld launched but successful resumed API progress not observed within 15 minutes')
    except BaseException as exc:
        state.update(status='needs_attention',error=repr(exc),traceback=traceback.format_exc(),ended=time.time());save();raise


if __name__=='__main__':main()
