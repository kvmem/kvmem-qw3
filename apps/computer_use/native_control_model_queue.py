"""Serial native-control pilots after the complete, unchanged desktop chain.

The prepared study stays immutable. This declaration pins the actual successor
route, qualified private trial entry, and existing owned-engine supervisor.
"""
import argparse
from contextlib import contextmanager
import copy
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import native_control_model_trial as trial
import stable_resource_deferred as deferred

stack = trial.stack
owned, base = stack.adapter.owned, stack.adapter.queue
ROOT, RUNTIME = stack.ROOT, stack.RUNTIME
read, save, SafetyHalt = base.read, base.save, owned.SafetyHalt
MODULE = 'native_control_model_queue'
PARENT_SHA = '26426d1191712e87e40e1d91ac81403b68fa11bd48b8c88b82c5e90e2f0660e6'
TRIAL_SHA = 'cca356568a9964ba0e622b256ca81267acd759051a099dc43a5061b2f11f4953'
HELPERS = (MODULE+'.py', 'tests/test_'+MODULE+'.py', 'qualify_'+MODULE+'.py',
           'native_control_video_probe.py', 'tests/test_query_refresh_replay_queue.py', 'native_control_model_trial.py',
           'native_control_model_stack.py', 'native_control_pilot_adapter.py',
           'stable_resource_deferred.py', 'stable_resource_startup.py', 'stable_query_startup.py')
BOUNDS = dict(predecessor_wait_seconds=172800, resource_wait_seconds=21600,
              min_free_mib=56000, headroom_stability_seconds=30, launcher_wait_seconds=60,
              model_ready_seconds=600, child_observation_interval_seconds=10,
              administrative_observation_notice_seconds=1200)


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=owned.digest(path))


@contextmanager
def historical_scope():
    # Source admission still uses its historical SDK validator. Actual parent
    # registration, full terminal inventories and live engine checks follow.
    historical = deferred.successors.previous.previous.historical_image
    with patch.object(deferred.q.old, 'IMAGE_VALIDATE', historical):
        yield


def qualification(path):
    record = read(path)
    if (record.get('kind') != 'native_control_model_queue_cpu_qualification' or record.get('passed') is not True
            or record.get('tests_run', 0) < 16 or record.get('skipped') != 0
            or record.get('errors') != 0 or record.get('failures') != 0
            or record.get('model_calls') != 0 or record.get('device_inputs') != 0):
        raise SafetyHalt('Require the completed serial queue and full handoff qualification')
    if set(record.get('source_sha256', {})) != set(HELPERS):
        raise SafetyHalt('Require the full qualified native queue dependency inventory')
    for name in HELPERS:
        if owned.digest(ROOT/'apps/computer_use'/name) != record['source_sha256'][name]:
            raise SafetyHalt('Qualified native queue dependency changed')


def parent_state(p):
    reg = read(stack.adapter.checked_ref(p['parent_registration']))
    stack.adapter.checked_ref(p['parent_declaration'])
    stack.adapter.checked_ref(p['parent_protocol'])
    if (reg.get('stage') != 'query' or reg.get('planned_trials') != 4 or reg.get('controller_sha256') != PARENT_SHA
            or reg.get('identity') != p['parent_controller'] or reg.get('queue') != p['parent_queue']
            or reg.get('declaration') != p['parent_declaration']['path']
            or reg.get('declaration_sha256') != p['parent_declaration']['sha256']
            or reg.get('protocol') != p['parent_protocol']['path']
            or reg.get('protocol_sha256') != p['parent_protocol']['sha256']):
        raise SafetyHalt('Actual query predecessor registration changed')
    state = deferred.q.old.guard_queue(read(reg['queue']), reg['identity'], reg['protocol_sha256'], 'protocol_sha256')
    if (state.get('kind') != 'serial_saved_desktop_query_refresh_queue' or state.get('controller_sha256') != PARENT_SHA
            or state.get('planned_trials') != 4 or state.get('model_concurrency') != 1
            or state.get('tool_calls_executed') != 0 or state.get('device_inputs') != 0
            or state.get('business_success_claimed') is not False):
        raise SafetyHalt('Query predecessor scope changed')
    return reg, state


def parent_protocol(p):
    with historical_scope():
        reg, state = deferred.deferred_registration(Path(p['parent_registration']['path']), 'query')
        deferred.QUERY.DECLARATION = Path(reg['declaration'])
        protocol = deferred.QUERY.validate(Path(reg['protocol']))
    if protocol['parent_app'] != p['parent_app'] or protocol['engine_off_identity'] != p['engine_off_identity']:
        raise SafetyHalt('Original user services or OFF engine profile changed')
    return protocol


def query_receipts(protocol, state, directory):
    if (state.get('termination_confirmed') is not True or state.get('parent_services_preserved') is not True
            or state.get('final_engine_policy') != 'off'
            or [row['id'] for row in state.get('trials', [])] != [row['id'] for row in protocol['trials']]):
        raise SafetyHalt('All four query replays and the original final OFF handoff are required')
    receipts = []
    for spec, row in zip(protocol['trials'], state['trials']):
        out = Path(directory)/'attempts'/spec['id']
        path = out/'sequence-result.json'; result = read(path)
        if result != row['result'] or owned.digest(path) != row['result_sha256']:
            raise SafetyHalt('Original query sequence receipt changed')
        if (result.get('status') != 'completed' or result.get('termination_confirmed') is not True
                or result.get('original_generation_profile_unchanged') is not True
                or result.get('tool_calls_executed') != 0 or result.get('device_inputs') != 0
                or result.get('business_success_claimed') is not False or result.get('request_order') != ['seed','pressure']
                or [r['name'] for r in result.get('rows', [])] != ['seed','pressure']):
            raise SafetyHalt('Query sequence is not a complete original read-only replay')
        for item in result['rows']:
            leaf = out/item['name']; actual = read(leaf/'result.json')
            request = trial.archive.read_request(leaf)
            source = protocol['sources'][item['name']]
            if (actual != item['result'] or owned.digest(leaf/'result.json') != item['result_sha256']
                    or actual.get('termination_confirmed') is not True or actual.get('status') != 'completed'
                    or actual.get('original_generation_profile_unchanged') is not True
                    or actual.get('tool_calls_executed') != 0 or actual.get('device_inputs') != 0
                    or actual.get('business_success_claimed') is not False
                    or trial.archive.digest(request) != source['request_sha256']
                    or actual.get('actual_request_sha256') != source['request_sha256']
                    or Path(source['source_request']).read_bytes() != request):
                raise SafetyHalt('Actual query replay wire or terminal request receipt changed')
        evidence = out/'retrieval-evidence.json'
        if owned.digest(evidence) != row['evidence_sha256']:
            raise SafetyHalt('Original retrieval diagnostic changed')
        diagnostic = read(evidence)
        if (diagnostic.get('mode') != spec['mode'] or diagnostic.get('result_sha256') != row['result_sha256']
                or diagnostic.get('tool_calls_executed') != 0 or diagnostic.get('business_or_latency_improvement_proven') is not False):
            raise SafetyHalt('Query mode or evidence scope changed')
        owned.assert_no_services(out)
        receipts.append(dict(id=spec['id'], result_sha256=row['result_sha256'], evidence_sha256=row['evidence_sha256']))
    return receipts


def earlier_handoff(protocol, query_state):
    """Keep every original earlier receipt check; use the actual final engine."""
    source = inspect.getsource(deferred.QUERY.predecessor_ready)
    anchor = '    services = full_prior_handoff(parent, state)\n'
    if source.count(anchor) != 1:
        raise SafetyHalt('Frozen full predecessor handoff source changed')
    source = source.replace(anchor, '    services = full_prior_handoff(parent, query_terminal_state)\n', 1)
    namespace = dict(deferred.QUERY.predecessor_ready.__globals__, query_terminal_state=query_state)
    exec(compile(source, 'qualified-post-query-full-prior-handoff', 'exec'), namespace)
    with historical_scope():
        services = namespace['predecessor_ready'](protocol)
    if services is None:
        raise SafetyHalt('All earlier frozen cohorts must be independently terminal and retired')
    return services


def predecessor_ready(p):
    reg, state = parent_state(p)
    if state['status'] != 'completed':
        base.require_identity(reg['identity'])
        return None
    if not base.process_retired(reg['identity']):
        return None
    protocol = parent_protocol(p)
    receipts = query_receipts(protocol, state, Path(reg['protocol']).parent)
    services = earlier_handoff(protocol, state)
    engine = owned.register_engine(Path(state['engine_runtime']), p['engine_off_identity'])
    if not owned.same_process(state['registered_engine'], engine):
        raise SafetyHalt('Actual final query OFF engine generation changed')
    services.update(registered_engine=engine, native_query_predecessor_receipts=receipts)
    return services


def qualified_stacks(path):
    audit = read(path)
    if (audit.get('kind') != 'independent_real_private_native_stack_qualification_audit' or audit.get('passed') is not True
            or len(audit.get('records', [])) != 2 or audit.get('model_inference_requests') != 0 or audit.get('device_inputs') != 0):
        raise SafetyHalt('Require both actual independently retired private stacks')
    for record in audit['records']:
        for key in ('result','stack','movie'):stack.adapter.checked_ref(record[key])
        result = read(record['result']['path']);service = read(record['stack']['path'])
        if (not result.get('passed') or len(result.get('checks',{})) != 13 or not all(result['checks'].values())
                or service.get('status') != 'retired' or service.get('vm_retired') is not True
                or record.get('original_candidate_inventory_verified') is not True
                or record.get('actual_generations_retired') is not True or record.get('vm_absent') is not True
                or record.get('sqlite_action_counts') != [0,0] or record.get('model_request_files') != 0
                or record.get('decoded_frames',0) <= 0):
            raise SafetyHalt('Actual private stack proof is incomplete')
    return audit


def video_evidence(pilot):
    """Read and independently decode the unchanged pilot's actual recording."""
    receipt_path = pilot/'video/recording.json';receipt = read(receipt_path)
    recorder = read(pilot/'private-recorder-identity.json')
    if not base.process_retired(recorder):raise SafetyHalt('Original private recorder is still live')
    movie = pilot/'continuous.mp4';original = pilot/'video/screen.mp4'
    expected = receipt.get('artifacts',{}).get('screen.mp4',{}).get('sha256')
    if (receipt.get('complete') is not True or receipt.get('status') != 'complete'
            or receipt.get('input_events') != 0 or receipt.get('executor_observe_calls') != 0
            or not expected or owned.digest(movie) != expected or owned.digest(original) != expected):
        raise SafetyHalt('Actual video is incomplete or differs from the original recording receipt')
    command = [str(ROOT/'third_party/OSWorld-V2/.venv/bin/python'),
               str(ROOT/'apps/computer_use/native_control_video_probe.py'),str(movie)]
    env = dict(os.environ, PYTHONPATH=str(RUNTIME/'media-qualification'))
    decoded = subprocess.run(command,env=env,capture_output=True,text=True,timeout=60,check=True)
    evidence = json.loads(decoded.stdout)
    if (evidence.get('sha256') != expected or type(evidence.get('frames')) is not int
            or evidence['frames'] <= 0 or evidence['frames'] != receipt.get('video_frames')
            or evidence.get('width') != receipt.get('width') or evidence.get('height') != receipt.get('height')):
        raise SafetyHalt('Independent video decode disagrees with the original recorder')
    return dict(movie=ref(movie),recording_receipt=ref(receipt_path),recorder=recorder,
                independently_decoded=evidence,decoder_sha256=owned.digest(ROOT/'apps/computer_use/native_control_video_probe.py'))


def validate(path):
    path = Path(path).resolve()
    p = read(path)
    if (p.get('kind') != 'declared_serial_native_control_original_pilots' or p.get('bounds') != BOUNDS
            or p.get('model_concurrency') != 1 or p.get('advisor_enabled') is not False
            or p.get('overwrite_old_attempts') is not False):
        raise SafetyHalt('Native comparison model scope or resource bounds changed')
    if (not path.parent.is_relative_to(RUNTIME) or path.name != 'declaration.json'
            or p.get('state_dir') != str(path.parent/'queue-runtime-01')
            or p.get('registration') != str(path.parent/'registration.json')):
        raise SafetyHalt('Native queue must use its own declared runtime and registration')
    study_path = stack.adapter.checked_ref(p['prepared_study'])
    if owned.digest(study_path) != stack.STUDY_SHA:
        raise SafetyHalt('Preserve the original prepared eight-trial study')
    study, _, _, original = stack.adapter.validate_study(study_path, read(study_path)['trials'][0]['id'])
    if p['trials'] != study['trials'] or p['engine_off_identity'] != original['engine_off_identity']:
        raise SafetyHalt('Native tasks, arm order, deadlines or original engine changed')
    required = [p[key] for key in ('qualification','stack_audit','prepared_study','parent_registration','parent_declaration','parent_protocol')]
    required += [ref(ROOT/'apps/computer_use'/name) for name in HELPERS]
    if p['files'] != list({item['path']:item for item in required}.values()):
        raise SafetyHalt('Required native or parent source pins were removed or changed')
    for item in p['files']:stack.adapter.checked_ref(item)
    qualification(stack.adapter.checked_ref(p['qualification']))
    qualified_stacks(stack.adapter.checked_ref(p['stack_audit']))
    if owned.digest(trial.__file__) != TRIAL_SHA or owned.digest(deferred.__file__) != PARENT_SHA:
        raise SafetyHalt('Qualified real trial or predecessor controller changed')
    parent_state(p)
    return p


def prepare(args):
    qualification(args.qualification)
    study_path = args.study.resolve()
    if owned.digest(study_path) != stack.STUDY_SHA:raise SafetyHalt('Original prepared study changed')
    study = read(study_path)
    _, _, _, original = stack.adapter.validate_study(study_path, study['trials'][0]['id'])
    qualified_stacks(args.stack_audit)
    with historical_scope():
        reg, state = deferred.deferred_registration(args.parent_registration.resolve(), 'query')
        deferred.QUERY.DECLARATION = Path(reg['declaration'])
        parent = deferred.QUERY.validate(Path(reg['protocol']))
    if parent['engine_off_identity'] != original['engine_off_identity']:
        raise SafetyHalt('Actual query and original native study model profiles differ')
    output = args.output.resolve()
    if not output.is_relative_to(RUNTIME):raise ValueError('Use a fresh goal-owned runtime')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    paths = [args.qualification,args.stack_audit,study_path,args.parent_registration,Path(reg['declaration']),Path(reg['protocol'])]
    paths += [ROOT/'apps/computer_use'/name for name in HELPERS]
    p = dict(kind='declared_serial_native_control_original_pilots', declared_at=time.time(),
        prepared_study=ref(study_path),trials=copy.deepcopy(study['trials']),bounds=copy.deepcopy(BOUNDS),
        parent_registration=ref(args.parent_registration),parent_declaration=ref(reg['declaration']),
        parent_protocol=ref(reg['protocol']),parent_queue=reg['queue'],parent_controller=reg['identity'],
        parent_app=parent['parent_app'],engine_off_identity=original['engine_off_identity'],
        qualification=ref(args.qualification),stack_audit=ref(args.stack_audit),model_concurrency=1,
        advisor_enabled=False,overwrite_old_attempts=False,state_dir=str(output/'queue-runtime-01'),
        registration=str(output/'registration.json'),model_trials_dispatched=0,
        files=[ref(path) for path in dict.fromkeys(paths)])
    path = output/'declaration.json';save(path,p);validate(path)
    return path


class Controller(deferred.QUERY.Controller):
    def dispatch(self, row):
        validate(self.args.protocol)
        if owned.digest(self.args.protocol) != self.state['protocol_sha256']:
            raise SafetyHalt('Native declaration changed before dispatch')
        output = self.args.protocol.parent/'attempts'/row['id']
        if output.exists():raise SafetyHalt('Never reuse or replay a native trial attempt')
        self.fresh_engine(row)
        with owned.model_lock():
            self.parent_idle()
            engine = dict(runtime=self.state['engine_runtime'],identity=self.state['registered_engine'])
            binding = self.args.state_dir/(row['id']+'-engine.json')
            if binding.exists():raise SafetyHalt('Never overwrite a registered engine binding')
            save(binding,engine)
            _,_,framework,_ = stack.adapter.validate_study(self.p['prepared_study']['path'],row['id'])
            command = [str(ROOT/'third_party/OSWorld-V2/.venv/bin/python'),'-m','native_control_model_trial',
                       '--study',self.p['prepared_study']['path'],'--trial-id',row['id'],
                       '--engine-binding',str(binding),'--output',str(output),'--port-base','19040']
            self.checkpoint(status='running-original-native-pilot',current_trial=row['id'])
            child, identity = self.spawn(command,row['id'],framework)
            started = time.monotonic()
            while True:
                try:
                    code = child.wait(timeout=BOUNDS['child_observation_interval_seconds']);break
                except subprocess.TimeoutExpired:
                    base.require_identity(identity)
                    self.checkpoint(current_child=identity,same_trial_child_confirmed_live=True,
                        administrative_observation_notice=(time.monotonic()-started > row['seconds']+BOUNDS['administrative_observation_notice_seconds']))
                    # Monitoring the same handle grants no task budget and never
                    # dispatches a replacement or classifies a timeout as exit.
            self.collect(row,output,code)
        self.parent_idle()

    def collect(self,row,output,code):
        result = read(output/'result.json')
        if (code != 0 or result.get('kind') != 'one_frozen_private_native_control_qw3_trial'
                or result.get('status') != 'completed' or result.get('termination_confirmed') is not True
                or result.get('trial_id') != row['id'] or result.get('arm') != row['arm']
                or result.get('case') != row['case'] or result.get('repetition') != row['repetition']
                or result.get('controller_sha256') != TRIAL_SHA or result.get('study_sha256') != self.p['prepared_study']['sha256']):
            raise SafetyHalt('Native child is failed, unconfirmed or belongs to a different trial')
        pilot = output/(row['case']+'-r'+str(row['repetition']))
        outcome = read(pilot/'integration-result.json');trial.terminal_result(outcome)
        if outcome != result['outcome'] or owned.digest(pilot/'integration-result.json') != result['original_result_sha256']:
            raise SafetyHalt('Original native business grade changed')
        task = read(pilot/'task.json');trial.task_profile(task)
        manifest = read(pilot/'pilot-manifest.json')
        if owned.digest(pilot/'task.json') != result['actual_task_sha256'] or trial.hashlib.sha256(manifest['prompt'].encode()).hexdigest() != result['prompt_sha256']:
            raise SafetyHalt('Native actual task or original prompt changed')
        service = read(output/'stack/stack.json')
        if owned.digest(output/'stack/stack.json') != result['stack_sha256'] or service.get('status') != 'retired' or service.get('vm_retired') is not True:
            raise SafetyHalt('Private service stack retirement is unconfirmed')
        study,_,framework,original = stack.adapter.validate_study(self.p['prepared_study']['path'],row['id'])
        if (manifest.get('harness_sha256') != original['helper_sha256']['platform_pilots.py']
                or manifest.get('deadline_seconds') != row['seconds'] or manifest.get('model_output_tokens') != 16384
                or manifest.get('no_model_access_to_initializer_or_grader') is not True
                or manifest.get('container_id') != service['vm']['container_id'] or service.get('framework') != str(framework)):
            raise SafetyHalt('Original task isolation, budget or private framework changed')
        identities = [service['producer'],*service.get('client_descendants',[])]
        identities += [item['identity'] for item in service['services'].values()]
        for identity in identities:
            if not base.process_retired(identity):raise SafetyHalt('A native owned service remains live')
        vm = subprocess.run(['docker','inspect','--type','container',service['vm']['container_id']],capture_output=True,text=True,timeout=10)
        if vm.returncode == 0 or not any(text in vm.stderr for text in ('No such object','No such container')):
            raise SafetyHalt('Actual native VM retirement is not independently confirmed')
        inference = Path(service['services']['client']['runtime'])/'inference'
        if trial.wire_evidence(inference) != result['actual_model_requests']:
            raise SafetyHalt('Actual native model request evidence changed')
        previous = next((r for r in self.state['trials'] if r['case']==row['case'] and r['repetition']==row['repetition']),None)
        if previous and previous['prompt_sha256'] != result['prompt_sha256']:
            raise SafetyHalt('Matched native prompt differs between arms')
        video = video_evidence(pilot)
        owned.assert_no_services(output)
        self.state['trials'].append(dict(row,result=result,result_sha256=owned.digest(output/'result.json'),
            prompt_sha256=result['prompt_sha256'],video=video,exit_code=code))
        self.checkpoint(real_model_trials_completed=len(self.state['trials']))


def run(args):
    p = validate(args.protocol)
    args.state_dir.mkdir(parents=True,exist_ok=False,mode=0o700)
    state = dict(kind='serial_matched_native_control_original_pilots',pid=os.getpid(),started_at=time.time(),
        status='waiting-for-complete-original-query-chain',trials=[],planned_trials=8,
        protocol_sha256=owned.digest(args.protocol),controller_sha256=owned.digest(__file__),
        model_concurrency=1,real_model_trials_completed=0,business_success_improvement_proven=False)
    controller = Controller(args,p,state);controller.checkpoint()
    identity = owned.process_identity(os.getpid())
    registration = dict(kind='actual_native_control_serial_waiter',identity=identity,controller_sha256=owned.digest(__file__),
        declaration=str(args.protocol),declaration_sha256=state['protocol_sha256'],queue=str(args.state_dir/'queue.json'),planned_trials=8,registered_at=time.time())
    with Path(p['registration']).open('x') as file:json.dump(registration,file,indent=2);file.write('\n')
    try:
        deadline = time.monotonic()+BOUNDS['predecessor_wait_seconds']
        while True:
            services = predecessor_ready(p)
            if services is not None:break
            controller.checkpoint(last_parent_observation_at=time.time())
            if time.monotonic() >= deadline:raise SafetyHalt('Original predecessor observation deadline reached; no replay')
            time.sleep(3)
        controller.checkpoint(parent_services=services)
        for row in p['trials']:controller.dispatch(row)
        controller.checkpoint(status='completed',termination_confirmed=True,finished_at=time.time(),
            parent_services_preserved=True,final_engine_policy='off')
    except BaseException as error:
        controller.checkpoint(status='halted-no-further-dispatch',finished_at=time.time(),
            error=type(error).__name__+': '+str(error),live_children_and_evidence_preserved=True)
        raise
    return state


def registered_waiter(path):
    path = Path(path).resolve();p = validate(path)
    reg = read(p['registration'])
    expected = [str(ROOT/'.venv/bin/python'),'-m',MODULE,'run','--declaration',str(path)]
    if (reg.get('kind') != 'actual_native_control_serial_waiter' or reg.get('planned_trials') != 8
            or reg.get('controller_sha256') != owned.digest(__file__) or reg.get('declaration') != str(path)
            or reg.get('declaration_sha256') != owned.digest(path) or reg.get('queue') != str(Path(p['state_dir'])/'queue.json')
            or reg.get('identity',{}).get('command') != expected or reg['identity'].get('cwd') != str(ROOT)
            or reg['identity'].get('uid') != os.getuid()):
        raise SafetyHalt('Actual native serial waiter registration changed')
    state = deferred.q.old.guard_queue(read(reg['queue']),reg['identity'],reg['declaration_sha256'],'protocol_sha256')
    if (state.get('kind') != 'serial_matched_native_control_original_pilots'
            or state.get('controller_sha256') != reg['controller_sha256'] or state.get('planned_trials') != 8
            or state.get('model_concurrency') != 1):raise SafetyHalt('Actual registered queue scope changed')
    return reg,state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','run'])
    for name in ('study','qualification','stack-audit','parent-registration','output','declaration'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.operation=='prepare':
        for name in ('study','qualification','stack_audit','parent_registration','output'):
            if getattr(args,name) is None:parser.error('Missing '+name)
        print(dict(declaration=str(prepare(args)),model_calls=0,device_inputs=0));return
    if args.declaration is None:parser.error('Missing declaration')
    if sys.executable != str(ROOT/'.venv/bin/python'):raise SafetyHalt('Use the qualified lexical controller interpreter')
    args.protocol=args.declaration.resolve();p=validate(args.protocol);args.state_dir=Path(p['state_dir'])
    args.resource_wait_seconds=BOUNDS['resource_wait_seconds'];args.min_free_mib=BOUNDS['min_free_mib']
    def interrupted(*_):raise SafetyHalt('Native waiter interrupted; same live evidence preserved')
    signal.signal(signal.SIGTERM,interrupted)
    print(dict(status=run(args)['status']))


if __name__=='__main__':main()
