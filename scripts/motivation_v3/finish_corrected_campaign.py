"""Finish the existing run, collect one frozen supplement, and audit the primary cohort."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from finalize_campaign import process_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--supplement',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--legacy',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int,required=True)
    args=parser.parse_args();script=Path(__file__).resolve().parent
    state_path=args.primary/'correction_run.json'
    state={'status':'waiting_for_original','started_at':time.time(),
           'wait_pid':args.wait_pid,'goal_complete':False,'human_review_required':True}
    with state_path.open('x') as stream:json.dump(state,stream,indent=2)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    def run(name,*options):
        subprocess.run([sys.executable,str(script/name),*map(str,options)],check=True)
    try:
        supplement_run=args.supplement/'run';supplement_run.mkdir(exist_ok=True)
        worker=args.original/'vision_worker_snapshot.py';pinned=supplement_run/worker.name
        if not pinned.exists():shutil.copy2(worker,pinned)
        assert hashlib.sha256(pinned.read_bytes()).digest()==hashlib.sha256(worker.read_bytes()).digest()
        for name in ['model_provenance.json','environment.json','protocol_snapshot']:
            link=args.primary/name
            if not link.exists():link.symlink_to((args.original/name).resolve())
        identity=process_identity(args.wait_pid);state['process_start_ticks']=identity;save()
        while identity is not None and process_identity(args.wait_pid)==identity:time.sleep(30)
        original=json.loads((args.original/'campaign.json').read_text())
        if original['status']!='finished':raise RuntimeError('Original campaign did not finish; no automatic retry')
        state['status']='auditing_original_primary_subset';save()
        common=['--manifest',args.primary/'manifest.json','--measurements',args.primary/'measurements']
        run('audit_new.py',*common,'--out',args.primary/'pre_supplement_integrity_audit.json')
        state['status']='collecting_supplement';save()
        run('run_campaign.py','--manifest',args.supplement/'manifest.json','--out',supplement_run,
            '--binary',args.binary,'--collection-only')
        supplement=json.loads((supplement_run/'campaign.json').read_text())
        if supplement['status']!='finished':raise RuntimeError('Supplement did not finish')
        (args.primary/'campaign.json').write_text(json.dumps({
            'status':'finished','started_at':original['started_at'],'finished_at':time.time(),
            'original_campaign':str(args.original),'supplement_campaign':str(supplement_run),
            'planned_primary_trajectories':80,'collected_trajectories_including_auxiliary':81,
            'scope':'Collection finished; full corrected-primary audit and human review are separate gates.'
        },indent=2)+'\n')
        state['status']='finalizing_corrected_primary';save()
        run('finalize_campaign.py','--campaign',args.primary,'--manifest',args.primary/'manifest.json',
            '--legacy',args.legacy)
        state.update(status='ready_for_human_review',finished_at=time.time());save()
    except BaseException as error:
        state.update(status='failed',error=str(error),finished_at=time.time());save();raise


if __name__=='__main__':main()
