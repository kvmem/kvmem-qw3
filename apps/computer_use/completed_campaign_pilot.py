"""Run a declared development child after every frozen formal attempt ends.

The old scheduler and its results are never rewritten or impersonated. A missing
terminal receipt, live formal app or unknown child stop prevents another task.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from campaign_interlude import command
from osworld_linux.campaign import jobs, verify_frozen
from osworld_linux.runner import request, save
from osworld_windows.source import ROOT


def completed_campaign(directory):
    directory=directory.resolve()
    plan=json.loads((directory/'plan.json').read_text())
    expected={row['id'] for row in jobs()}
    rows=plan.get('jobs',[])
    if (plan.get('status')!='finished' or len(rows)!=len(expected)
            or {row.get('id') for row in rows}!=expected
            or any(row.get('status')!='finished' for row in rows)):
        raise ValueError('All forty original attempts must be durably finished')
    for row in rows:
        result=json.loads((directory/row['id']/'result.json').read_text())
        if not result.get('finished_at'):
            raise ValueError('Original attempt has no terminal receipt: '+row['id'])
    verify_frozen(plan)
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:argv=command(int(path.name))
        except (OSError,ValueError,UnicodeError):continue
        if 'qw3_computer.app' in argv and '--runtime' in argv:
            runtime=Path(argv[argv.index('--runtime')+1]).resolve()
            if runtime==directory or directory in runtime.parents:
                raise RuntimeError('A formal model app remains alive')
        if 'osworld_linux.runner' in argv and '--output' in argv:
            runtime=Path(argv[argv.index('--output')+1]).resolve()
            if directory in runtime.parents:
                raise RuntimeError('A formal runner remains alive')
    return {'campaign':str(directory),'ended_attempts':len(rows),
            'plan_sha256':hashlib.sha256((directory/'plan.json').read_bytes()).hexdigest()}


def run_completed(args,pilot_command,result_path,terminal,*,watchdog_seconds=None):
    with Path('/tmp/qw3-cross-device-model-queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        completion=completed_campaign(args.campaign)
        state=request(args.app,'/api/state')
        if state['status']!='idle' or state.get('uncertain_actions'):
            raise ValueError('The independent candidate must be confirmed idle')
        args.output.mkdir(parents=True,exist_ok=False)
        receipt={'kind':'development-pilot-after-completed-formal-campaign','formal':False,
                 'started_at':time.time(),'controller_pid':os.getpid(),'status':'preparing',
                 'formal_completion':completion,'model_concurrency':1,
                 'formal_sources_and_budgets_unchanged':True}
        save(args.output/'interlude.json',receipt)
        environment=dict(os.environ,PYTHONPATH=str(ROOT/'apps/computer_use')+':'+
                         str(ROOT/'apps/computer_use/.runtime/media-qualification'))
        try:
            with (args.output/'pilot-launch.log').open('wb') as log:
                child=subprocess.Popen(pilot_command,cwd=ROOT,env=environment,
                                       stdout=log,stderr=subprocess.STDOUT)
                receipt.update(status='running-standalone-pilot',pilot_pid=child.pid,
                               pilot_command=pilot_command,pilot_started_at=time.time())
                save(args.output/'interlude.json',receipt)
                code=child.wait(timeout=args.seconds+240 if watchdog_seconds is None else watchdog_seconds)
            outcome=json.loads(result_path.read_text())
            if not terminal(args,outcome):
                raise RuntimeError('Development task stop/retirement is unconfirmed')
            receipt.update(status='pilot-finished',pilot_exit_code=code,pilot_result=outcome,
                           pilot_finished_at=time.time())
        except Exception as exc:
            # A process timeout/exit is not a desktop stop acknowledgment.
            receipt.update(status='needs-reconciliation',error=type(exc).__name__+': '+str(exc))
        receipt['finished_at']=time.time()
        save(args.output/'interlude.json',receipt)
        return receipt
