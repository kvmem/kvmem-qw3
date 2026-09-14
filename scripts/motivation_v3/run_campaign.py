"""Run the frozen cohort in isolated server batches and build final artifacts."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--collection-only',action='store_true',help='Collect a supplement; analyze its combined primary cohort separately')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).resolve().parent;manifest=json.loads(args.manifest.read_text())
    pinned_worker=args.out/'vision_worker_snapshot.py'
    if not pinned_worker.exists():shutil.copy2(script.parent/'qw3_vision_cpu_worker.py',pinned_worker)
    domains=defaultdict(list)
    for row in manifest['trajectories']:domains[row['domain']].append(row['id'])
    ordered=[]
    while any(domains.values()):
        for rows in domains.values():
            if rows:ordered.append(rows.pop(0))
    status={'started_at':time.time(),'status':'running','planned_trajectories':len(ordered),'batches':[]}
    state=args.out/'campaign.json'
    def save():
        tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(state)
    save()
    try:
        for batch,start in enumerate(range(0,len(ordered),20),1):
            ids=ordered[start:start+20];server_dir=args.out/f'server_{batch:02d}'
            if server_dir.exists():raise FileExistsError('Campaign output must be fresh; retain previous attempts')
            server=subprocess.Popen([sys.executable,str(script/'serve_probe.py'),'--out',str(server_dir),
                '--binary',str(args.binary)],start_new_session=True)
            info={'batch':batch,'ids':ids,'status':'running','started_at':time.time()}
            status['batches'].append(info);save()
            try:
                cmd=[sys.executable,str(script/'collect.py'),'--manifest',str(args.manifest),
                    '--out',str(args.out/'measurements'),'--trace',str(server_dir/'attention.jsonl'),
                    '--input-trace',str(server_dir/'inputs.jsonl')]
                for name in ids:cmd+=['--only',name]
                subprocess.run(cmd,check=True)
                info.update(status='finished',finished_at=time.time());save()
            finally:
                # Signal the wrapper, which records exit and stops only its own child.
                if server.poll() is None:
                    os.kill(server.pid,signal.SIGINT)
                    try:server.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid,signal.SIGTERM);server.wait(timeout=30)
            common=['--manifest',str(args.manifest),'--measurements',str(args.out/'measurements'),
                    '--out',str(args.out/'analysis')]
            if not args.collection_only:
                subprocess.run([sys.executable,str(script/'analyze_new.py')]+common,check=True)
                subprocess.run([sys.executable,str(script/'schedule_diagnostics.py')]+common,check=True)
        status.update(status='finished',finished_at=time.time());save()
    except BaseException as e:
        status.update(status='failed',error=str(e),finished_at=time.time());save();raise


if __name__=='__main__':main()
