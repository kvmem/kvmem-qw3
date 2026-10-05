"""Resume the frozen memory trials after the exact pre-VM cssselect failure.

Only the runner's Python environment, fresh evidence paths and initial owned
engine registration change. Original failures, eight cases, source packages,
model configurations, graders and budgets remain immutable.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import retired_051_successors as old

owned, read, save, SafetyHalt = old.owned, old.read, old.save, old.SafetyHalt
ROOT = old.image.ROOT
PYTHON = ROOT/'third_party/OSWorld-V2/.venv/bin/python'
HELPERS = ('runner_runtime_continuation.py','tests/test_runner_runtime_continuation.py')


def failure_scope(state, result, manifest, traceback_text, existing_paths):
    if (state.get('status')!='halted-no-further-dispatch' or state.get('trials')!=[]
            or state.get('current_trial')!='035-kvmem-off-r1'
            or result.get('status')!='infrastructure-failed' or result.get('score') is not None
            or not result.get('finished_at') or result.get('accepted') is not False
            or not result.get('error','').startswith('ImportError: cssselect does not seem to be installed.')
            or "ModuleNotFoundError: No module named 'cssselect'" not in traceback_text
            or 'task_035.py' not in traceback_text or manifest.get('initialized_at') is not None
            or existing_paths):
        raise SafetyHalt('Only the exact pre-VM runtime failure can receive continuation admission')


def patch_memory_source(source):
    source=old.patched_memory_source(source)
    anchor='initial_runtime = Path(parent_protocol["engine_runtime"])'
    if source.count(anchor)!=1:raise SafetyHalt('Frozen initial runtime anchor changed')
    source=source.replace(anchor,'initial_runtime = Path(runtime_declaration["preserved_engine_runtime"])',1)
    anchor='"continued_successor_wrapper_sha256": wrapper_sha, "continued_successor_declaration_sha256": followup_sha}'
    if source.count(anchor)!=1:raise SafetyHalt('Frozen continuation state anchor changed')
    return source.replace(anchor,anchor[:-1]+', "runner_runtime_continuation_sha256": runtime_sha, '
        '"qualified_runner_python": runtime_declaration["python"]}',1)


def validate(path, *, require_engine=False):
    d=read(path)
    if (d.get('kind')!='pre_vm_cssselect_runner_runtime_continuation'
            or d.get('original_profiles_changed') is not False or d.get('failed_attempt_overwritten') is not False
            or d.get('python')!=str(PYTHON)):
        raise SafetyHalt('Use the declared original runtime and preserved frozen trials')
    for item in d['files']:old.image.pinned(item['path'],item['sha256'])
    old.image.pinned(PYTHON,d['python_binary_sha256'])
    q=read(d['qualification'])
    if q.get('passed') is not True or q.get('tests_run',0)<10 or q.get('model_calls')!=0 or q.get('device_inputs')!=0:
        raise SafetyHalt('Require completed continuation CPU qualification')
    for name in HELPERS:old.image.pinned(ROOT/'apps/computer_use'/name,q['source_sha256'][name])
    canonical,_,_,_=old.validate_followup(Path(d['original_followup']))
    parent=old.continued_parent(canonical,owned.digest(d['original_followup']),Path(d['original_followup']))
    if parent['status']!='completed' or owned.process_identity(parent['pid']) is not None:
        raise SafetyHalt('All forty predecessors must be terminal and retired')
    for previous in d['retired_controllers']:
        reg=read(previous['registration']);state=read(previous['queue'])
        if (reg['identity']!=previous['identity'] or state.get('pid')!=previous['identity']['pid']
                or state.get('status')!='halted-no-further-dispatch' or state.get('trials')!=[]
                or owned.process_identity(previous['identity']['pid']) is not None):
            raise SafetyHalt('A former controller remains live, dispatched work or changed')
    failed=Path(d['failed_attempt']);state=read(d['retired_controllers'][0]['queue'])
    existing=[name for name in ('vm.json','initial.png','task_instruction.json','app','gateway') if (failed/name).exists()]
    failure_scope(state,read(failed/'result.json'),read(failed/'manifest.json'),(failed/'traceback.txt').read_text(),existing)
    child=state.get('child')
    if not isinstance(child,dict) or owned.process_identity(child['pid']) is not None:
        raise SafetyHalt('The failed trial process still exists or is unconfirmed')
    owned.assert_no_services(failed);owned.assert_vm_retired(failed)
    protocol=Path(d['continued_protocol'])
    if owned.digest(protocol)!=owned.digest(d['original_protocol']):raise SafetyHalt('Frozen memory protocol changed')
    owned.validate_protocol(protocol)
    if read(d['runtime_qualification']).get('executable')!=str(PYTHON):raise SafetyHalt('Qualified venv executable changed')
    if require_engine:
        engine=owned.register_engine(Path(d['preserved_engine_runtime']),read(protocol)['engine_identities']['off'])
        if not owned.same_process(d['preserved_engine_identity'],engine):raise SafetyHalt('Preserved own engine changed')
    return d,canonical


def prepare(args):
    old_f=args.followup.resolve();canonical,_,_,_=old.validate_followup(old_f)
    reg=read(args.memory_controller);state=read(reg['queue']);previous=[]
    for path in (args.memory_controller,args.sdk_controller,args.image_controller,args.thinking_controller,args.feedback_controller):
        r=read(path);previous.append(dict(registration=str(path.resolve()),queue=r['queue'],identity=r['identity']))
    failed=Path(canonical['memory_protocol']).parent/'attempts/035-kvmem-off-r1'
    engine=owned.register_engine(Path(state['current_engine_runtime']),read(canonical['memory_protocol'])['engine_identities']['off'])
    if not owned.same_process(state['last_registered_engine'],engine):raise SafetyHalt('Retained engine owner changed')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    original=Path(canonical['memory_protocol']);clone=out/'protocol.json';clone.write_bytes(original.read_bytes())
    for policy in ('off','on'):
        target=out/policy;target.mkdir();(target/'acceptance.json').write_bytes((original.parent/policy/'acceptance.json').read_bytes())
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'apps/computer_use')+':'+str(ROOT/'apps/computer_use/.runtime/media-qualification'))
    command=[str(PYTHON),'-c',"import json,sys,platform;import cssselect,lxml.cssselect,numpy,av,aiohttp;import osworld_office_candidate_run;import qw3_computer.app,qw3_computer.device_gateway;print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,version=platform.python_version(),imports=['cssselect','lxml.cssselect','numpy','av','aiohttp','osworld_office_candidate_run','qw3_computer.app','qw3_computer.device_gateway'],model_calls=0,device_inputs=0)))"]
    check=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=20,check=True)
    runtime=out/'runtime-qualification.json';runtime.write_text(check.stdout)
    paths=[old_f,original,clone,args.qualification.resolve(),runtime,failed/'result.json',failed/'manifest.json',failed/'traceback.txt',
           Path(state['current_engine_runtime'])/'manifest.json']
    for p in previous:paths.extend((Path(p['registration']),Path(p['queue'])))
    for policy in ('off','on'):paths.append(out/policy/'acceptance.json')
    d=dict(kind='pre_vm_cssselect_runner_runtime_continuation',declared_at=time.time(),original_followup=str(old_f),
        original_protocol=str(original),continued_protocol=str(clone),state_dir=str(out/'queue-runtime-01'),
        retired_controllers=previous,failed_attempt=str(failed),preserved_engine_runtime=state['current_engine_runtime'],
        preserved_engine_identity=engine,python=str(PYTHON),python_binary_sha256=owned.digest(PYTHON),
        runtime_qualification=str(runtime),qualification=str(args.qualification.resolve()),original_profiles_changed=False,
        failed_attempt_overwritten=False,first_failure_retained_as_infrastructure=True,original_unknowns_preserved=True,
        files=[dict(path=str(p),sha256=owned.digest(p)) for p in paths],
        scope='Same frozen eight memory trials in fresh paths; existing qualified OSWorld CPython3.12 venv instead of dependency-incomplete3.13. Original pre-VM failure remains outside model success denominator. No original score or unknown receipt rewritten.')
    path=out/'continuation.json';save(path,d);validate(path,require_engine=True);return path


def run(args):
    path=args.declaration.resolve();d,canonical=validate(path,require_engine=True)
    if sys.executable!=d['python']:raise SafetyHalt('Launch this continuation with the qualified original venv path')
    args.protocol,args.state_dir,args.followup=Path(d['continued_protocol']),Path(d['state_dir']),Path(d['original_followup'])
    namespace=dict(vars(owned));namespace.update(d=canonical,followup_sha=owned.digest(args.followup),
        wrapper_sha=owned.digest(Path(__file__)),runtime_declaration=d,runtime_sha=owned.digest(path),continued_parent=old.continued_parent,
        terminal_parent=lambda p,q:old.terminal_parent(p,q,args.followup))
    def protocol_validate(p):
        validate(path);return owned.validate_protocol(p)
    namespace['validate_protocol']=protocol_validate
    exec(compile(patch_memory_source(inspect.getsource(owned.run)),'pre-vm-runtime-continuation','exec'),namespace)
    return namespace['run'](args)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--followup',type=Path);p.add_argument('--qualification',type=Path);p.add_argument('--declaration',type=Path)
    for stage in ('memory','sdk','image','thinking','feedback'):p.add_argument('--'+stage+'-controller',type=Path)
    p.add_argument('--wait-seconds',type=int,default=172800);p.add_argument('--resource-wait-seconds',type=int,default=21600)
    p.add_argument('--min-free-mib',type=int,default=56000);a=p.parse_args()
    if a.min_free_mib<56000 or not 60<=a.wait_seconds<=172800 or not 60<=a.resource_wait_seconds<=86400:p.error('Keep frozen resource bounds')
    if a.prepare:
        if any(getattr(a,k) is None for k in ('output','followup','qualification','memory_controller','sdk_controller','image_controller','thinking_controller','feedback_controller')):p.error('Provide frozen predecessors and CPU qualification')
        print(json.dumps(dict(declaration=str(prepare(a)),model_calls=0,device_inputs=0)));return
    if a.declaration is None:p.error('Provide the prepared immutable continuation')
    def interrupted(*_):raise SafetyHalt('Runtime continuation interrupted; owned work preserved')
    signal.signal(signal.SIGTERM,interrupted);result=run(a);print(json.dumps(dict(status=result['status'],trials=len(result['trials']))))


if __name__=='__main__':main()
