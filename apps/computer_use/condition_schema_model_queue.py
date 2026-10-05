"""Serial wire-control/repair trials after the actual original native eight."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
import types

import condition_schema_model_study as study
import condition_schema_model_trial as trial
import native_control_model_queue as native

ROOT,RUNTIME,owned,base= native.ROOT,native.RUNTIME,native.owned,native.base
read,save,SafetyHalt=native.read,native.save,native.SafetyHalt
MODULE='condition_schema_model_queue'
NATIVE_SHA='bbb27079bc0d8cb508dbc532e8fe725d807630a03c1efde3382b9c75afcf1c05'
BOUNDS=copy.deepcopy(native.BOUNDS)
HELPERS=tuple(dict.fromkeys((MODULE+'.py','condition_schema_model_trial.py',
    'qualify_condition_schema_model_queue.py','tests/test_condition_schema_model_queue.py',
    'tests/test_condition_schema_model_trial.py','condition_schema_model_study.py',
    'qualify_condition_schema_model_stack.py','audit_condition_schema_model_stack.py',
    'condition_schema_candidate.py','condition_schema_wire.py',*native.HELPERS)))


def qualification(path):
    record=read(path)
    if (record.get('kind')!='condition_schema_model_queue_cpu_qualification' or record.get('passed') is not True
            or record.get('tests_run',0)<12 or any(record.get(k)!=0 for k in ('errors','failures','skipped','model_calls','device_inputs'))
            or set(record.get('source_sha256',{}))!=set(HELPERS)):
        raise SafetyHalt('Require complete trial, parent handoff and serial queue qualification')
    for name,sha in record['source_sha256'].items():
        if owned.digest(ROOT/'apps/computer_use'/name)!=sha:raise SafetyHalt('Qualified source changed: '+name)


def qualified_stacks(path,study_ref):
    audit=read(path)
    if (audit.get('kind')!='independent_separate_condition_schema_actual_vm_stack_audit'
            or audit.get('passed') is not True or audit.get('study')!=study_ref
            or len(audit.get('records',[]))!=2 or audit.get('model_requests')!=0 or audit.get('device_inputs')!=0):
        raise SafetyHalt('Require both actual private study stacks')
    if [r['arm'] for r in audit['records']]!=list(study.ARMS):raise SafetyHalt('Stack qualification arms differ')
    for row in audit['records']:
        for key in ('result','stack','wrapper_receipt','movie','original_video_receipt'):study.adapter.checked_ref(row[key])
        if (not row.get('actual_generations_retired') or not row.get('vm_absent')
                or row.get('sqlite_action_counts')!=[0,0] or row.get('model_request_files')!=0 or row.get('decoded_frames',0)<=0):
            raise SafetyHalt('Actual private stack proof incomplete')
    return audit


def parent_state(p):
    if owned.digest(native.__file__)!=NATIVE_SHA:raise SafetyHalt('Original native queue implementation changed')
    path=study.adapter.checked_ref(p['parent_declaration'])
    reg,state=native.registered_waiter(path)
    if (native.ref(p['parent_registration']['path'])!=p['parent_registration'] or reg['identity']!=p['parent_controller']
            or reg['queue']!=p['parent_queue'] or reg['declaration_sha256']!=p['parent_declaration']['sha256']):
        raise SafetyHalt('Actual original native parent identity changed')
    if str(state['status']).startswith('halted'):raise SafetyHalt('Original native parent halted; preserve evidence')
    return reg,state


def original_receipts(parent,state,directory):
    if (state.get('termination_confirmed') is not True or state.get('parent_services_preserved') is not True
            or state.get('final_engine_policy')!='off' or [r['id'] for r in state['trials']]!=[r['id'] for r in parent['trials']]):
        raise SafetyHalt('All original native eight and final handoff must be complete')
    # Reuse every original collector check in an in-memory sink. Never write
    # the original queue or give its failed trials a second budget.
    collector=object.__new__(native.Controller);collector.p=parent;collector.state={'trials':[]}
    collector.checkpoint=lambda **values:collector.state.update(values)
    for spec,row in zip(parent['trials'],state['trials']):
        native.Controller.collect(collector,spec,Path(directory)/'attempts'/spec['id'],row['exit_code'])
    if collector.state['trials']!=state['trials']:raise SafetyHalt('Original native grades or decoded video evidence changed')
    return collector.state['trials']


def predecessor_ready(p):
    reg,state=parent_state(p)
    if state['status']!='completed':
        base.require_identity(reg['identity']);return None
    if not base.process_retired(reg['identity']):return None
    parent=native.validate(Path(reg['declaration']))
    receipts=original_receipts(parent,state,Path(reg['declaration']).parent)
    query_reg,query_state=native.parent_state(parent)
    if not base.process_retired(query_reg['identity']):raise SafetyHalt('Original query producer still live')
    if not base.process_retired(query_state['registered_engine']):raise SafetyHalt('Original query engine did not retire before native trials')
    query=native.parent_protocol(parent)
    query_receipts=native.query_receipts(query,query_state,Path(query_reg['protocol']).parent)
    services=native.earlier_handoff(query,dict(query_state,engine_runtime=state['engine_runtime'],registered_engine=state['registered_engine']))
    engine=owned.register_engine(Path(state['engine_runtime']),p['engine_off_identity'])
    if not owned.same_process(state['registered_engine'],engine):raise SafetyHalt('Final original native OFF engine changed')
    services.update(registered_engine=engine,original_native_predecessor_receipts=receipts,
        original_query_predecessor_receipts=query_receipts)
    return services


def observe_predecessor(controller,p):
    try:return predecessor_ready(p)
    except FileNotFoundError:raise
    except (OSError,TimeoutError) as error:
        # An observation error is not a new ownership generation or authority
        # to launch again. Confirm the same producer/engine, then re-poll.
        reg,state=parent_state(p)
        if state['status']=='completed' and base.process_retired(reg['identity']):
            base.require_identity(state['registered_engine'])
        else:base.require_identity(reg['identity'])
        controller.checkpoint(transient_parent_observation_error=type(error).__name__,same_parent_handle_preserved=True)
        return None


def validate(path):
    path=Path(path).resolve();p=read(path)
    if (p.get('kind')!='declared_separate_serial_inline_condition_model_trials' or p.get('bounds')!=BOUNDS
            or p.get('model_concurrency')!=1 or p.get('advisor_enabled') is not False or p.get('overwrite_old_attempts') is not False
            or not path.parent.is_relative_to(RUNTIME) or path.name!='declaration.json'
            or p.get('state_dir')!=str(path.parent/'queue-runtime-01') or p.get('registration')!=str(path.parent/'registration.json')):
        raise SafetyHalt('Separate queue scope, resource bounds or runtime changed')
    prepared=study.adapter.checked_ref(p['prepared_study'])
    protocol,_,_,original=study.validate_study(prepared,read(prepared)['trials'][0]['id'])
    if p['trials']!=protocol['trials'] or p['engine_off_identity']!=original['engine_off_identity']:
        raise SafetyHalt('Original material, task budget or engine profile changed')
    if p['parent_declaration']!=protocol['required_original_native_queue_declaration']:
        raise SafetyHalt('Original native predecessor cannot be replaced')
    required=[p[k] for k in ('prepared_study','qualification','stack_audit','parent_registration','parent_declaration')]
    required += [native.ref(ROOT/'apps/computer_use'/name) for name in HELPERS]
    if p['files']!=list({r['path']:r for r in required}.values()):raise SafetyHalt('Required source inventory changed')
    for reference in p['files']:study.adapter.checked_ref(reference)
    qualification(study.adapter.checked_ref(p['qualification']))
    qualified_stacks(study.adapter.checked_ref(p['stack_audit']),p['prepared_study'])
    reg,state=parent_state(p);parent=native.validate(Path(reg['declaration']))
    if p['parent_app']!=parent['parent_app'] or p['engine_off_identity']!=parent['engine_off_identity']:
        raise SafetyHalt('Original parent services or engine differ')
    return p


def prepare(args):
    qualification(args.qualification)
    prepared=args.study.resolve();protocol,_,_,original=study.validate_study(prepared,read(prepared)['trials'][0]['id'])
    qualified_stacks(args.stack_audit,native.ref(prepared))
    parent_path=study.adapter.checked_ref(protocol['required_original_native_queue_declaration'])
    reg,state=native.registered_waiter(parent_path);parent=native.validate(parent_path)
    output=args.output.resolve()
    if not output.is_relative_to(RUNTIME):raise ValueError('Use a new goal-owned queue directory')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    p=dict(kind='declared_separate_serial_inline_condition_model_trials',declared_at=time.time(),
        prepared_study=native.ref(prepared),trials=copy.deepcopy(protocol['trials']),bounds=copy.deepcopy(BOUNDS),
        parent_declaration=native.ref(parent_path),parent_registration=native.ref(parent['registration']),
        parent_controller=reg['identity'],parent_queue=reg['queue'],parent_app=parent['parent_app'],
        engine_off_identity=original['engine_off_identity'],qualification=native.ref(args.qualification),stack_audit=native.ref(args.stack_audit),
        model_concurrency=1,advisor_enabled=False,overwrite_old_attempts=False,state_dir=str(output/'queue-runtime-01'),
        registration=str(output/'registration.json'),files=[])
    refs=[p[k] for k in ('prepared_study','qualification','stack_audit','parent_registration','parent_declaration')]
    refs += [native.ref(ROOT/'apps/computer_use'/name) for name in HELPERS]
    p['files']=list({r['path']:r for r in refs}.values());path=output/'declaration.json';save(path,p);validate(path)
    return path


def dispatch_function():
    if owned.digest(native.__file__)!=NATIVE_SHA:raise SafetyHalt('Original serial dispatch source changed')
    source=textwrap.dedent(inspect.getsource(native.Controller.dispatch))
    for before,after in [("'-m','native_control_model_trial'","'-m','condition_schema_model_trial'"),
                         ("'--port-base','19040'","'--port-base','19060'")]:
        if source.count(before)!=1:raise SafetyHalt('Original native dispatch anchor changed')
        source=source.replace(before,after,1)
    namespace=dict(native.Controller.dispatch.__globals__,validate=validate,
        stack=types.SimpleNamespace(adapter=types.SimpleNamespace(validate_study=study.validate_study)))
    exec(compile(source,'separate-inline-condition-dispatch','exec'),namespace)
    return namespace['dispatch']


class Controller(native.Controller):
    dispatch=dispatch_function()

    def collect(self,row,output,code):
        receipt=read(output/'condition-trial-receipt.json')
        if (receipt.get('status')!='completed' or receipt.get('termination_confirmed') is not True
                or receipt.get('study')!=self.p['prepared_study'] or receipt.get('trial_id')!=row['id']
                or receipt.get('wrapper_source')!=native.ref(trial.__file__)
                or receipt.get('original_trial_result')!=native.ref(output/'result.json')):
            raise SafetyHalt('Separate trial wrapper is nonterminal or changed')
        service=read(output/'stack/stack.json')
        if receipt.get('original_business_outcome')!=read(output/'result.json').get('outcome'):
            raise SafetyHalt('Separate wrapper changed the original business grade')
        actual=trial.wire_conditions(Path(service['services']['client']['runtime'])/'inference',row['arm'])
        if actual!=receipt['actual_condition_wire']:raise SafetyHalt('Actual condition wire evidence changed')
        with study.scoped_stack(self.p['prepared_study']['path'],row['id']):
            super().collect(row,output,code)
        self.state['trials'][-1]['condition_receipt_sha256']=owned.digest(output/'condition-trial-receipt.json')
        self.checkpoint(real_model_trials_completed=len(self.state['trials']))


def run(args):
    p=validate(args.protocol);args.state_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    state=dict(kind='serial_separate_inline_condition_model_trials',pid=os.getpid(),started_at=time.time(),
        status='waiting-for-complete-original-native-eight',trials=[],planned_trials=8,model_concurrency=1,
        protocol_sha256=owned.digest(args.protocol),controller_sha256=owned.digest(__file__),real_model_trials_completed=0,
        business_success_improvement_proven=False)
    controller=Controller(args,p,state);controller.checkpoint()
    reg=dict(kind='actual_separate_condition_schema_serial_waiter',identity=owned.process_identity(os.getpid()),
        controller_sha256=owned.digest(__file__),declaration=str(args.protocol),declaration_sha256=state['protocol_sha256'],
        queue=str(args.state_dir/'queue.json'),planned_trials=8,registered_at=time.time())
    with Path(p['registration']).open('x') as file:json.dump(reg,file,indent=2);file.write('\n')
    try:
        deadline=time.monotonic()+BOUNDS['predecessor_wait_seconds']
        while True:
            services=observe_predecessor(controller,p)
            if services is not None:break
            controller.checkpoint(last_parent_observation_at=time.time())
            if time.monotonic()>=deadline:raise SafetyHalt('Original native observation deadline reached; no replay')
            time.sleep(3)
        controller.checkpoint(parent_services=services)
        for row in p['trials']:controller.dispatch(row)
        controller.checkpoint(status='completed',termination_confirmed=True,finished_at=time.time(),
            parent_services_preserved=True,final_engine_policy='off')
    except BaseException as error:
        controller.checkpoint(status='halted-no-further-dispatch',finished_at=time.time(),
            error=type(error).__name__+': '+str(error),live_children_and_evidence_preserved=True);raise
    return state


def registered_waiter(path):
    path=Path(path).resolve();p=validate(path);reg=read(p['registration'])
    expected=[str(ROOT/'.venv/bin/python'),'-m',MODULE,'run','--declaration',str(path)]
    if (reg.get('kind')!='actual_separate_condition_schema_serial_waiter' or reg.get('controller_sha256')!=owned.digest(__file__)
            or reg.get('declaration')!=str(path) or reg.get('declaration_sha256')!=owned.digest(path)
            or reg.get('queue')!=str(Path(p['state_dir'])/'queue.json') or reg.get('planned_trials')!=8
            or reg['identity']['command']!=expected or reg['identity']['cwd']!=str(ROOT) or reg['identity']['uid']!=os.getuid()):
        raise SafetyHalt('Actual separate waiter registration changed')
    state=native.deferred.q.old.guard_queue(read(reg['queue']),reg['identity'],reg['declaration_sha256'],'protocol_sha256')
    if (state.get('kind')!='serial_separate_inline_condition_model_trials' or state.get('controller_sha256')!=reg['controller_sha256']
            or state.get('planned_trials')!=8 or state.get('model_concurrency')!=1):raise SafetyHalt('Actual separate queue scope changed')
    return reg,state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['prepare','run'])
    for name in ('study','qualification','stack-audit','output','declaration'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':
        for name in ('study','qualification','stack_audit','output'):
            if getattr(args,name) is None:parser.error('Missing '+name)
        print(json.dumps(dict(declaration=str(prepare(args)),model_calls=0,device_inputs=0)))
    else:
        if args.declaration is None:parser.error('Missing declaration')
        if sys.executable!=str(ROOT/'.venv/bin/python'):raise SafetyHalt('Use the qualified lexical controller interpreter')
        args.protocol=args.declaration.resolve();p=validate(args.protocol);args.state_dir=Path(p['state_dir'])
        args.resource_wait_seconds=BOUNDS['resource_wait_seconds'];args.min_free_mib=BOUNDS['min_free_mib']
        def interrupted(*_):raise SafetyHalt('Separate waiter interrupted; same live work preserved')
        signal.signal(signal.SIGTERM,interrupted);print(json.dumps(dict(status=run(args)['status'])))
