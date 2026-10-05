"""Restore only retired, zero-trial successor stages after static admission mixed profiles.

The ten completed SDK trials and all original task protocols remain intact.
Historical source admission uses its historical SDK validator; actual terminal
receipts, processes, device leases and engine handoff retain their own checks.
"""
import argparse
import copy
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

import shared_quota_staged_successors as previous

ROOT=previous.ROOT
MODULE='static_handoff_successors'
STAGES=('image','thinking','feedback')
ORIGINAL=ROOT/'apps/computer_use/.runtime/shared-quota-staged-successors-20261005-01/successors.json'
ORIGINAL_SHA='7514e1258693b4e7707f88e58c4d3def02c882fb938aee6e9559c69715698799'
TERMINAL=ROOT/'apps/computer_use/.runtime/native-keyboard-delivery-final-audit-03/result.json'
TERMINAL_SHA='14cf9b1e64eda2e8ce310002f647172804fd5d352296e93e8352587af0c08ccf'
DIAGNOSTIC=ROOT/'apps/computer_use/.runtime/image-admission-terminal-diagnostic-01/result.json'
DIAGNOSTIC_SHA='d97aec37de971d9d38ff9181bc06fcb31f28a7393e6e91dd206d0ea62b698a30'
HELPERS=(MODULE+'.py','tests/test_'+MODULE+'.py')
old=previous.RUNTIME
owned,SafetyHalt=old.owned,old.SafetyHalt
STATIC_SDK=old.FROZEN_VALIDATORS['sdk']
STATIC_IMAGE=old.FROZEN_VALIDATORS['image']


def historical_image(path):
    # No effective memory profile may enter historical source admission.
    # Its actual profile is still validated independently by the original
    # terminal inventory and register_engine calls after this scope exits.
    with patch.object(old.sdk,'validate',STATIC_SDK):
        return STATIC_IMAGE(Path(path))


def qualification(path):
    result=old.read(path)
    if (result.get('kind')!='static_handoff_successors_cpu_qualification'
            or result.get('passed') is not True or result.get('tests_run',0)<18
            or result.get('skipped')!=0 or result.get('model_calls')!=0
            or result.get('device_inputs')!=0):
        raise SafetyHalt('Require complete static admission and retained SDK routing qualification')
    for name in HELPERS:
        old.image.pinned(ROOT/'apps/computer_use'/name,result['source_sha256'][name])


def terminal_scope(*,actual=False):
    old.image.pinned(ORIGINAL,ORIGINAL_SHA);old.image.pinned(TERMINAL,TERMINAL_SHA)
    old.image.pinned(DIAGNOSTIC,DIAGNOSTIC_SHA)
    evidence=old.read(TERMINAL);diagnostic=old.read(DIAGNOSTIC)
    if (evidence.get('kind')!='readonly_actual_terminal_and_halted_six_stage_audit'
            or diagnostic.get('nested_terminal_check_rejected') is not True
            or diagnostic.get('exact_error')!='Original OSWorld source admission or budget changed'):
        raise SafetyHalt('Use only the preserved nested static-admission failure')
    declaration=old.read(ORIGINAL)
    for name,row in evidence['stages'].items():
        old.image.pinned(row['registration']['path'],row['registration']['sha256'])
        old.image.pinned(row['queue']['path'],row['queue']['sha256'])
        registration=old.read(row['registration']['path']);state=old.read(row['queue']['path'])
        if (registration['identity']!=row['identity']['registered']
                or state['pid']!=registration['identity']['pid']
                or state['protocol_sha256']!=registration['protocol_sha256']):
            raise SafetyHalt('Terminal producer declaration changed')
        if name=='sdk':
            if state.get('status')!='completed' or len(state.get('trials',[]))!=10:
                raise SafetyHalt('Retain exactly ten completed SDK trials; never rerun SDK setup')
            p,kp=old.frozen_protocol('sdk',Path(declaration['sdk_protocol']))
            old.image.mixed_terminal_rows(p['trials'],state,Path(declaration['sdk_protocol']).parent,registration['protocol_sha256'])
            if state.get('termination_confirmed') is not True or state.get('new_services_preserved') is not True:
                raise SafetyHalt('Completed SDK handoff is unconfirmed')
            sdk=state
        elif state.get('status')!='halted-no-further-dispatch' or state.get('trials')!=[]:
            raise SafetyHalt('Only retired zero-trial stages may be restored')
        if actual and not old.base.process_retired(registration['identity']):
            raise SafetyHalt('A former producer is still live; no new producer allowed')
    return dict(declaration=declaration,sdk=sdk,evidence=evidence)


def derive(declaration,output,cpu):
    result=copy.deepcopy(declaration)
    result.update(recovery_kind='static-source-admission-only',recovery_parent=str(ORIGINAL),
                  recovery_parent_sha256=ORIGINAL_SHA,recovery_terminal_audit=str(TERMINAL),
                  recovery_terminal_audit_sha256=TERMINAL_SHA,recovery_diagnostic=str(DIAGNOSTIC),
                  recovery_diagnostic_sha256=DIAGNOSTIC_SHA,qualification=str(cpu),
                  completed_sdk_preserved=True,new_sdk_dispatch_allowed=False,
                  scope='Only restore three retired zero-trial stages. Preserve all ten SDK terminal trials and original protocols, tasks, models, budgets, grades and input guards.')
    for stage in STAGES:
        result[stage+'_state_dir']=str(Path(output)/(stage+'-queue'))
        result[stage+'_registration']=str(Path(output)/(stage+'-registration.json'))
    files=[ORIGINAL,TERMINAL,DIAGNOSTIC,Path(cpu),*[ROOT/'apps/computer_use'/name for name in HELPERS]]
    result['files']=copy.deepcopy(declaration['files'])+[dict(path=str(p),sha256=owned.digest(p)) for p in files]
    return result


def validate(path):
    path=Path(path);d=old.read(path)
    scope=terminal_scope()
    if d!=derive(scope['declaration'],path.parent,Path(d['qualification'])):
        raise SafetyHalt('Recovery changed more than static source admission and fresh producer routing')
    qualification(d['qualification'])
    # The original validator retains source inventories, model hashes,
    # predecessor evidence, budgets and all registered protocol checks.
    _,canonical,mr=old.validate(ORIGINAL)
    return d,canonical,mr


def successor_registration(path,stage,declaration,d):
    if stage=='sdk':
        if Path(path)!=Path(d['sdk_registration']) or d['sdk_registration']!=old.read(ORIGINAL)['sdk_registration']:
            raise SafetyHalt('Cannot substitute another SDK producer')
        return old.successor_registration(Path(path),stage,ORIGINAL,old.read(ORIGINAL))
    if stage not in STAGES:raise SafetyHalt('Unsupported recovered stage')
    return BASE_REGISTRATION(path,stage,declaration,d)


def successor_queue(registration,declaration):
    if registration.get('stage')=='sdk':
        expected=old.read(old.read(ORIGINAL)['sdk_registration'])
        if registration!=expected:raise SafetyHalt('SDK terminal producer identity changed')
        return old.successor_queue(registration,ORIGINAL)
    return BASE_QUEUE(registration,declaration)


def prepare(output,cpu):
    output=Path(output).resolve();cpu=Path(cpu).resolve()
    qualification(cpu);scope=terminal_scope(actual=True)
    old.validate(ORIGINAL)
    if output.exists():raise SafetyHalt('Keep existing producer states; use a fresh recovery directory')
    for stage in STAGES:
        protocol=old.frozen_protocol(stage,scope['declaration'][stage+'_protocol'])
        for row in protocol['trials']:
            if (Path(scope['declaration'][stage+'_protocol']).parent/'attempts'/row['id']).exists():
                raise SafetyHalt('Original successor trial already exists; cannot replay it')
    output.mkdir(parents=True,exist_ok=False)
    d=derive(scope['declaration'],output,cpu);path=output/'successors.json'
    old.save(path,d);validate(path)
    return path


def build_namespace():
    module=previous.build_namespace()
    module.__file__=__file__;module.MODULE=MODULE
    module.FROZEN_VALIDATORS=dict(module.FROZEN_VALIDATORS,image=historical_image)
    module.validate=validate;module.qualification=qualification
    return module


q=build_namespace()
BASE_REGISTRATION,BASE_QUEUE=q.successor_registration,q.successor_queue
q.successor_registration,q.successor_queue=successor_registration,successor_queue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    for name in ('output','qualification','declaration'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--stage',choices=STAGES)
    parser.add_argument('--wait-seconds',type=int,default=172800)
    parser.add_argument('--resource-wait-seconds',type=int,default=21600)
    parser.add_argument('--min-free-mib',type=int,default=56000)
    args=parser.parse_args()
    if (not 60<=args.wait_seconds<=172800 or not 60<=args.resource_wait_seconds<=86400 or args.min_free_mib!=56000):
        parser.error('Keep the exact bounded shared-resource profile')
    if args.prepare:
        if args.output is None or args.qualification is None:parser.error('Provide qualification and fresh output')
        print(dict(declaration=str(prepare(args.output,args.qualification)),model_calls=0,device_inputs=0));return
    if args.declaration is None or args.stage is None:parser.error('Provide the immutable declaration and stage')
    args.declaration=args.declaration.resolve();terminal_scope(actual=True)
    d,canonical,mr=validate(args.declaration)
    if sys.executable!=d['controller_python']:raise SafetyHalt('Keep the exact qualified lexical interpreter')
    args.protocol=Path(d[args.stage+'_protocol']);args.state_dir=Path(d[args.stage+'_state_dir'])
    def interrupted(*_):raise SafetyHalt('Recovered waiter interrupted; live work and evidence preserved')
    signal.signal(signal.SIGTERM,interrupted)
    # validate_followup also invokes this historical validator inside the
    # runtime SDK memory scope. Restore that source boundary, not its guards.
    with patch.object(q.old,'IMAGE_VALIDATE',historical_image):
        result={'image':q.run_image,'thinking':q.run_thinking,'feedback':q.run_feedback}[args.stage](args,d,canonical,mr)
    print(dict(status=result['status'],completed=len(result['trials'])))


if __name__=='__main__':main()
