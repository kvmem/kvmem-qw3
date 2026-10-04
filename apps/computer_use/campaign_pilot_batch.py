"""Run declared development pilots in one existing formal-queue interlude.

No model requests overlap. Each child must retire its own task before the next
one can start; a timeout or unknown stop leaves the formal scheduler suspended.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

import campaign_interlude as interlude
from osworld_linux.runner import request, save
from osworld_windows.source import ROOT


CASES={'linux_text','linux_calc','linux_browser','android_settings','android_contacts','android_browser'}
FAULTS={'slow_loading','focus_dialog','no_progress'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path):
    value=json.loads(path.read_text())
    if value.get('kind')!='declared_serial_development_pilot_batch':
        raise ValueError('Use a declared development protocol')
    trials=value.get('trials',[])
    if not 1<=len(trials)<=20:raise ValueError('Declare 1..20 trials')
    identities=set()
    for row in trials:
        if set(row)-{'id','case','seconds','fault'} or not {'id','case','seconds'}<=set(row):
            raise ValueError('Invalid trial fields')
        identity=row['id']
        if not isinstance(identity,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,45}',identity) or identity in identities:
            raise ValueError('Unsafe or duplicated trial identity')
        identities.add(identity)
        if row['case'] not in CASES or type(row['seconds']) is not int or not 60<=row['seconds']<=900:
            raise ValueError('Unsupported pilot or budget')
        if row.get('fault') and (row['fault'] not in FAULTS or not row['case'].endswith('_browser')):
            raise ValueError('Unsupported fault condition')
    if not value.get('source_sha256') or not value.get('framework_sha256'):
        raise ValueError('Declare helper and candidate framework bytes')
    if (value.get('model_output_tokens')!=16384 or value.get('tool_call_limit')!=100
            or value.get('thinking') is not True or value.get('advisor') is not False):
        raise ValueError('Use the declared local thinking profile without an advisor')
    for relative,expected in value['source_sha256'].items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or digest(source)!=expected:
            raise ValueError('Declared helper changed: '+relative)
    runtime=Path(value['app_runtime']).resolve()
    for filename,expected in value['framework_sha256'].items():
        source=(runtime/'framework/qw3_computer'/filename).resolve()
        if not source.is_relative_to(runtime/'framework/qw3_computer') or digest(source)!=expected:
            raise ValueError('Declared candidate framework changed: '+filename)
    return value


def idle(app):
    state=request(app,'/api/state')
    return state['status']=='idle' and not state.get('uncertain_actions')


def admitted_app(protocol):
    state=request(protocol['app'],'/api/state')
    if state['status']!='idle' or state.get('uncertain_actions'):
        raise RuntimeError('Candidate is not confirmed idle')
    if (state.get('model_output_tokens')!=protocol['model_output_tokens'] or state.get('thinking') is not True
            or state.get('tool_call_limit')!=protocol['tool_call_limit']):
        raise RuntimeError('Candidate model profile differs from the declaration')


def child_terminal(result):
    # Correct files or an "accepted" flag do not authorize a second task.
    return (result.get('owned_task_retired') is True
            and not result.get('owned_task_stop_unconfirmed'))


def worker(protocol_path,output):
    protocol=load_protocol(protocol_path)
    output.mkdir(parents=True,exist_ok=False)
    report={'kind':'serial_development_pilot_batch','formal':False,'started_at':time.time(),
            'protocol_sha256':digest(protocol_path),'status':'running','trials':[],
            'termination_confirmed':False,'owned_task_stop_unconfirmed':True}
    save(output/'result.json',report)
    child=None
    try:
        for row in protocol['trials']:
            load_protocol(protocol_path)
            admitted_app(protocol)
            directory=output/row['id']
            command=[sys.executable,'-m','platform_pilots','--case',row['case'],'--seconds',str(row['seconds']),
                     '--app',protocol['app'],'--runtime',protocol['app_runtime'],'--output',str(directory),
                     '--retire-task']
            if row.get('fault'):command+=['--fault',row['fault']]
            with (output/(row['id']+'-launch.log')).open('wb') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(output/'current-child.json',dict(id=row['id'],pid=child.pid,command=command,started_at=time.time()))
                code=child.wait(timeout=row['seconds']+240)
            result=json.loads((directory/'integration-result.json').read_text())
            report['trials'].append(dict(id=row['id'],exit_code=code,result=result))
            save(output/'result.json',report)
            if not child_terminal(result) or not idle(protocol['app']):
                raise RuntimeError('Child did not durably retire its task; no next pilot started')
            child=None
        report.update(status='completed',termination_confirmed=True,owned_task_stop_unconfirmed=False)
    except Exception as exc:
        report.update(status='needs-reconciliation',error=type(exc).__name__+': '+str(exc))
        # An early failure can safely return the queue only with no live child
        # and a separately acknowledged idle app. A killed process is not proof.
        if child is None:
            try:
                if idle(protocol['app']):
                    report.update(status='halted-before-next-pilot',termination_confirmed=True,
                                  owned_task_stop_unconfirmed=False)
            except Exception:pass
    report['finished_at']=time.time();save(output/'result.json',report)
    return report


def run(args):
    protocol=load_protocol(args.protocol)
    args.app=protocol['app'];args.app_runtime=Path(protocol['app_runtime'])
    args.case='linux_browser';args.seconds=900
    args.bundle=args.engine=args.ffmpeg=args.repair_fixture=args.acceptance_manifest=None
    def specification(_):
        return ([sys.executable,'-m','campaign_pilot_batch','worker','--protocol',str(args.protocol.resolve()),
                 '--output',str(args.output/'batch')],args.output/'batch/result.json')
    def terminal(_,result):
        return (result.get('termination_confirmed') is True
                and not result.get('owned_task_stop_unconfirmed') and idle(args.app))
    # Reuse the proven scheduler ownership, frozen sources and current-runner
    # handoff unchanged. Only the independently executed child is a batch.
    with patch.object(interlude,'pilot_spec',specification),patch.object(interlude,'terminal_pilot',terminal),\
            patch.object(interlude,'pilot_watchdog',lambda _:sum(row['seconds']+240 for row in protocol['trials'])+120):
        return interlude.run(args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['run','worker'])
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--campaign',type=Path)
    args=parser.parse_args()
    if args.operation=='run' and not args.campaign:parser.error('Specify the existing frozen campaign')
    result=worker(args.protocol,args.output) if args.operation=='worker' else run(args)
    print({k:result.get(k) for k in ['status','error','scheduler_resumed_at','termination_confirmed']},flush=True)


if __name__=='__main__':main()
