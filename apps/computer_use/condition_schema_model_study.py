"""A separate native task study for wire control versus condition restoration.

Reuse the frozen stack and pilot in a new process. Scope substitutions affect
only that process's study lookup; no frozen file or running controller changes.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import condition_schema_candidate as candidate
import native_control_pilot_adapter as adapter

ORIGINAL_VALIDATE = adapter.validate_study
ORIGINAL_STUDY_SHA = 'b36f465ecd64d9b839d5c8f0b949ac70a66dd2181f097f05e31d3948951445e0'
STACK_SHA = '0688bfe05178a82fa26c1642a85f6877bcb942c599c8936d43d33d5aaaa8795e'
ADAPTER_SHA = '8e73897bf0324cf3c3307c593492f72fb3ffe118ce4eda7fe828c23ba2d12b41'
QUALIFIER_SHA = '24ab811950c0b78da7c7c953c06586aac6e8f77e7742cac68daeaa975be2928a'
ARMS = ('wire-control', 'condition-repair')


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=candidate.digest(path))


def rows():
    return [dict(id=f'condition-schema-{case}-{arm}-r{rep}', arm=arm, case=case,
        repetition=rep, seconds=360 if case=='linux_text' else 900, tools=100)
        for rep in (1,2) for arm in (ARMS if rep==1 else ARMS[::-1])
        for case in ('linux_text','linux_calc')]


def inventory(build_reference):
    path=adapter.checked_ref(build_reference);build=json.loads(path.read_text());framework=path.parent/'framework'
    actual={str(p.relative_to(framework)):candidate.digest(p) for p in framework.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts}
    if actual!=build['candidate_files_sha256']:raise adapter.owned.SafetyHalt('Full private framework inventory changed')
    return path,build,framework


def validate_study(path, trial_id):
    study=json.loads(Path(path).read_text())
    if (study.get('kind')!='prepared_separate_inline_condition_matched_qw3_study'
            or study.get('status')!='prepared-not-registered-not-dispatched'
            or study.get('trials')!=rows()):
        raise adapter.owned.SafetyHalt('Require all original eight matched conditions and budgets')
    if candidate.digest(adapter.__file__)!=ADAPTER_SHA:raise adapter.owned.SafetyHalt('Original pilot adapter changed')
    original_path=adapter.checked_ref(study['source_original_native_study'])
    if candidate.digest(original_path)!=ORIGINAL_STUDY_SHA:raise adapter.owned.SafetyHalt('Original native study changed')
    original_study=json.loads(original_path.read_text())
    _,_,_,original=ORIGINAL_VALIDATE(original_path,original_study['trials'][0]['id'])
    if (study['model_profile']!=original_study['model_profile']
            or study['original_sdk_protocol']!=original_study['original_sdk_protocol']
            or study['original_pilot_helpers']!=original_study['helpers']):
        raise adapter.owned.SafetyHalt('Original model, material or grader dependencies changed')
    control_path,control,control_framework=inventory(study['control_build'])
    repair_path,repair,repair_framework=inventory(study['candidate_build'])
    if (study['control_build']!=original_study['candidate_build'] or candidate.digest(control_path)!=candidate.BASE_BUILD_SHA
            or repair.get('kind')!='isolated_native_button_inline_condition_wire_candidate'
            or repair['base_build']!=str(control_path) or repair['base_build_sha256']!=candidate.BASE_BUILD_SHA
            or repair['base_candidate_files_sha256']!=control['candidate_files_sha256']
            or repair['changed_package_files']!=['qw3_computer/protocol.py','qw3_computer/condition_schema_wire.py']):
        raise adapter.owned.SafetyHalt('Separate arms must differ only in inline condition conversion')
    for name,sha in control['candidate_files_sha256'].items():
        if name!='qw3_computer/protocol.py' and repair['candidate_files_sha256'].get(name)!=sha:
            raise adapter.owned.SafetyHalt('A non-wire behavior differs between arms')
    for key in ('wire_cpu_qualification','actual_wire_transport','actual_wire_transport_audit'):
        evidence=json.loads(adapter.checked_ref(study[key]).read_text())
        if evidence.get('passed') is not True:raise adapter.owned.SafetyHalt('Wire qualification incomplete')
    transport=json.loads(adapter.checked_ref(study['actual_wire_transport']).read_text())
    if (transport['build_sha256']!=candidate.digest(repair_path) or transport['fixture_backend_requests']!=6
            or transport['model_inference_requests']!=0 or transport['tool_dispatch_attempts']!=0
            or transport['actual_codex_version']!='0.159.2' or len(transport['cases'])!=6
            or not all(r['passed'] and r['all_registered_processes_retired'] for r in transport['cases'])):
        raise adapter.owned.SafetyHalt('Actual private wire candidate was not qualified')
    audit=json.loads(adapter.checked_ref(study['actual_wire_transport_audit']).read_text())
    if audit.get('source_report')!=study['actual_wire_transport'] or len(audit.get('cases',[]))!=6:
        raise adapter.owned.SafetyHalt('Independent transport audit covers a different candidate')
    cpu=json.loads(adapter.checked_ref(study['wire_cpu_qualification']).read_text())
    if (cpu.get('kind')!='private_inline_condition_wire_candidate_cpu_qualification' or cpu.get('tests_run')!=10
            or any(cpu.get(k)!=0 for k in ('errors','failures','skipped','model_calls','device_inputs'))):
        raise adapter.owned.SafetyHalt('Require the final complete wire qualification')
    for name,sha in cpu['source_sha256'].items():
        if candidate.digest(candidate.ROOT/'apps/computer_use'/name)!=sha:raise adapter.owned.SafetyHalt('Wire CPU-qualified source changed')
    for name,reference in study['new_helpers'].items():
        if adapter.checked_ref(reference)!=candidate.ROOT/'apps/computer_use'/name:raise adapter.owned.SafetyHalt('New helper identity changed')
    parent=adapter.checked_ref(study['required_original_native_queue_declaration'])
    if parent!=candidate.RUNTIME/'native-control-model-queue-20261006-01/declaration.json':
        raise adapter.owned.SafetyHalt('Do not skip the original native queue')
    if json.loads(parent.read_text())['prepared_study']!=study['source_original_native_study']:
        raise adapter.owned.SafetyHalt('Original native predecessor studies differ')
    matches=[r for r in study['trials'] if r['id']==trial_id]
    if len(matches)!=1:raise adapter.owned.SafetyHalt('Trial was not declared')
    row=matches[0]
    return study,row,control_framework if row['arm']=='wire-control' else repair_framework,original


@contextmanager
def scoped_stack(study_path,trial_id):
    import native_control_model_stack as stack
    if candidate.digest(stack.__file__)!=STACK_SHA:raise adapter.owned.SafetyHalt('Original stack implementation changed')
    validate_study(study_path,trial_id)
    with patch.object(stack,'STUDY_SHA',candidate.digest(study_path)),patch.object(adapter,'validate_study',validate_study):
        yield stack


def prepare(args):
    source=args.original_study.resolve();original=json.loads(source.read_text())
    if candidate.digest(source)!=ORIGINAL_STUDY_SHA:raise adapter.owned.SafetyHalt('Original study changed')
    ORIGINAL_VALIDATE(source,original['trials'][0]['id'])
    output=args.output.resolve()
    if not output.is_relative_to(candidate.RUNTIME):raise ValueError('Use a new goal-owned study directory')
    data=dict(kind='prepared_separate_inline_condition_matched_qw3_study',prepared_at=time.time(),
        status='prepared-not-registered-not-dispatched',source_original_native_study=ref(source),
        original_sdk_protocol=copy.deepcopy(original['original_sdk_protocol']),original_pilot_helpers=copy.deepcopy(original['helpers']),
        control_build=copy.deepcopy(original['candidate_build']),candidate_build=ref(args.candidate_build),
        wire_cpu_qualification=ref(args.wire_cpu),actual_wire_transport=ref(args.wire_transport),actual_wire_transport_audit=ref(args.wire_audit),
        model_profile=copy.deepcopy(original['model_profile']),trials=rows(),
        required_original_native_queue_declaration=ref(candidate.RUNTIME/'native-control-model-queue-20261006-01/declaration.json'),
        new_helpers={name:ref(Path(__file__).with_name(name)) for name in
            ['condition_schema_model_study.py','qualify_condition_schema_model_stack.py']},
        requires_all_original_native_eight_terminal_and_actual_handoff=True,
        existing_sources_and_studies_modified=False,controller_registered=False,model_requests=0,device_inputs=0,
        material_initializer_grader_and_task_budget_changed=False,
        change_under_test='Only trusted inline completion-condition restoration before QW3; both arms retain the same native-button execution.',
        remaining_before_dispatch=['Actual private stack qualification and a serial controller bound to the original native-eight terminal handoff.',
            'Bind the unchanged actual OFF engine and qualify 524288 context before invoking an original pilot.'])
    output.mkdir(parents=True,exist_ok=False,mode=0o700);path=output/'protocol.json'
    path.write_text(json.dumps(data,indent=2)+'\n')
    validate_study(path,data['trials'][0]['id'])
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original-study','candidate-build','wire-cpu','wire-transport','wire-audit','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    data=prepare(parser.parse_args());print(json.dumps(dict(planned=len(data['trials']),model_requests=0,device_inputs=0)))
