"""Run one separately recorded pilot between frozen formal attempts.

Only the owned scheduler is suspended. Its current runner, VM and task budgets
continue normally. No source, task result or frozen configuration is rewritten.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from osworld_linux.runner import request, save
from osworld_linux.campaign import verify_frozen
from osworld_windows.source import ROOT


def command(pid):
    source=Path(f'/proc/{pid}')
    if source.stat().st_uid!=os.getuid():
        raise ValueError('Process owner changed')
    return [p.decode() for p in (source/'cmdline').read_bytes().split(b'\0') if p]


def suspended(pid):
    state=next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('State:'))
    return state.split()[1]=='T'


def scheduler(directory):
    found=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:cmd=command(int(path.name))
        except (OSError,ValueError,UnicodeError):continue
        if 'osworld_linux.campaign' in cmd and '--directory' in cmd and 'run' in cmd and Path(cmd[cmd.index('--directory')+1]).resolve()==directory:
            found.append((int(path.name),cmd))
    if len(found)!=1:
        raise ValueError('Expected exactly one owned frozen scheduler')
    return found[0]


def run(args):
    directory=args.campaign.resolve()
    plan=json.loads((directory/'plan.json').read_text())
    if plan['status']!='running':
        raise ValueError('Frozen scheduler is not running')
    verify_frozen(plan)
    initial=request(args.app,'/api/state')
    if initial['status']!='idle' or initial['uncertain_actions']:
        raise ValueError('Interlude requires an idle independent candidate app')
    args.output.mkdir(parents=True,exist_ok=False)
    pid,cmd=scheduler(directory)
    if suspended(pid):
        raise ValueError('Do not take over an already suspended scheduler')
    receipt={'kind':'separate-development-pilot-between-formal-attempts','started_at':time.time(),
             'scheduler_pid':pid,'scheduler_command':cmd,'campaign':str(directory),'case':args.case,
             'formal_sources_and_budgets_unchanged':True,'status':'preparing','model_concurrency':1}
    save(args.output/'interlude.json',receipt)
    stopped=False
    pilot=None
    safe_to_resume=True
    try:
        if command(pid)!=cmd:raise ValueError('Scheduler process identity changed')
        os.kill(pid,signal.SIGSTOP)
        stopped=True
        deadline=time.monotonic()+5
        while not suspended(pid) and time.monotonic()<deadline:time.sleep(.05)
        if not suspended(pid):raise TimeoutError('Scheduler suspension unconfirmed')
        # Read again after suspension: the first read may have raced the handoff
        # between two attempts. The current child continues unaffected.
        plan=json.loads((directory/'plan.json').read_text())
        active=[job for job in plan['jobs'] if job['status']=='running']
        if len(active)>1:raise ValueError('Multiple formal runners recorded')
        receipt.update(status='waiting-for-current-formal-attempt',suspended_at=time.time(),
                       current_attempt=active[0]['id'] if active else None)
        save(args.output/'interlude.json',receipt)
        if active:
            job=active[0]
            result=directory/job['id']/'result.json'
            child=job['runner_pid']
            deadline=time.monotonic()+3900
            while Path(f'/proc/{child}').exists() and time.monotonic()<deadline:
                status=Path(f'/proc/{child}/status').read_text()
                if '\nState:\tZ' in status:break
                time.sleep(2)
            if Path(f'/proc/{child}').exists() and '\nState:\tZ' not in Path(f'/proc/{child}/status').read_text():
                raise TimeoutError('Current runner did not finish; no pilot started')
            if not result.exists():raise RuntimeError('Current runner has no terminal result')
            receipt['preceding_formal_result']=json.loads(result.read_text())
        verify_frozen(plan)
        if not suspended(pid):raise RuntimeError('Scheduler resumed elsewhere; no pilot started')
        # No formal app may still be serving an agent, including after a
        # crashed runner. Do not start a second model workload in that state.
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:other=command(int(path.name))
            except (OSError,ValueError,UnicodeError):continue
            if 'qw3_computer.app' in other and '--runtime' in other and directory in Path(other[other.index('--runtime')+1]).resolve().parents:
                raise RuntimeError('Formal agent app still alive; no pilot started')
        time.sleep(1)
        receipt.update(status='running-separate-pilot',pilot_started_at=time.time())
        save(args.output/'interlude.json',receipt)
        environment=dict(os.environ,PYTHONPATH=str(ROOT/'apps/computer_use')+':'+str(ROOT/'apps/computer_use/.runtime/media-qualification'))
        safe_to_resume=False
        pilot_command=[str(ROOT/'third_party/OSWorld-V2/.venv/bin/python'),'-m','platform_pilots',
                                '--case',args.case,'--seconds',str(args.seconds),'--app',args.app,
                                '--runtime',str(args.app_runtime.resolve()),'--output',str(args.output/'pilot')]
        pilot=subprocess.Popen(pilot_command,
                               cwd=ROOT,env=environment,stdout=(args.output/'pilot-launch.log').open('wb'),stderr=subprocess.STDOUT)
        receipt.update(pilot_pid=pilot.pid,pilot_command=pilot_command)
        save(args.output/'interlude.json',receipt)
        try:code=pilot.wait(timeout=args.seconds+240)
        except subprocess.TimeoutExpired:
            safe_to_resume=False
            receipt.update(status='needs-reconciliation',error='Pilot watchdog expired; do not resume model queue automatically')
            save(args.output/'interlude.json',receipt)
            raise
        result=args.output/'pilot/integration-result.json'
        if not result.exists():
            safe_to_resume=False
            raise RuntimeError('Pilot ended without durable result; reconcile candidate app before resuming queue')
        outcome=json.loads(result.read_text())
        current=request(args.app,'/api/state')
        if outcome.get('owned_task_stop_unconfirmed') or current['status'] in {'starting','running','verifying'}:
            safe_to_resume=False
            raise RuntimeError('Candidate model task termination unconfirmed')
        safe_to_resume=True
        receipt.update(status='pilot-finished',pilot_exit_code=code,pilot_result=outcome,pilot_finished_at=time.time())
    except Exception as exc:
        receipt.update(error=type(exc).__name__+': '+str(exc))
        if receipt['status']!='needs-reconciliation':receipt['status']='failed'
    finally:
        if stopped and safe_to_resume:
            try:
                if command(pid)!=cmd or not suspended(pid):raise ValueError('Scheduler identity or suspension changed')
                verify_frozen(plan)
                if pilot is not None:time.sleep(1)
                os.kill(pid,signal.SIGCONT)
                receipt['scheduler_resumed_at']=time.time()
            except Exception as exc:receipt['resume_error']=type(exc).__name__+': '+str(exc)
        receipt['finished_at']=time.time()
        save(args.output/'interlude.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--app-runtime',type=Path,required=True)
    parser.add_argument('--app',default='http://127.0.0.1:18205')
    parser.add_argument('--case',choices=['android_contacts','android_settings','android_browser','linux_browser','linux_calc','linux_text'],required=True)
    parser.add_argument('--seconds',type=int,default=600)
    args=parser.parse_args()
    if not 60<=args.seconds<=900:parser.error('Interlude budget must be60..900seconds')
    result=run(args)
    print({k:result.get(k) for k in ['status','error','resume_error','scheduler_resumed_at']},flush=True)
