"""Continue the exact image prefix after a confirmed pre-model resource failure.

The first scored trial is retained in place and copied only into the new queue's
inventory. The other eleven trials and downstream cohorts retain their frozen
packages, prompts, model profiles, budgets, graders and original supervisors.
"""
import argparse
import copy
import inspect
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import static_handoff_successors as previous
import resource_startup as startup

ROOT, owned, SafetyHalt = previous.ROOT, previous.owned, previous.SafetyHalt
MODULE = 'resource_handoff_successors'
STAGES = previous.STAGES
RUNTIME = ROOT / 'apps/computer_use/.runtime'
ORIGINAL = RUNTIME / 'static-handoff-successors-20261005-01/successors.json'
ORIGINAL_SHA = 'cf20b48d3e744c630acb61ae5ddaaa908e5890c9f28c7e3bea93ad601cee822c'
TERMINAL = RUNTIME / 'native-control-pilot-final-state-01/result.json'
TERMINAL_SHA = 'a2b703f0e0b15a53db6a89262c4877f71979e69f41905824cca353e7f22c1d93'
DIAGNOSTIC = RUNTIME / 'native-control-pilot-engine-admission-failure-01.json'
DIAGNOSTIC_SHA = '06967671f46994fbb5dd39f35102fd141c6989bfedeaaee3ad23a750e8e531a8'
HELPERS = (MODULE + '.py', 'resource_startup.py', 'tests/test_' + MODULE + '.py')
REGISTRATIONS = {
    **{s: ORIGINAL.parent / (s + '-registration.json') for s in STAGES},
    **{s: RUNTIME / ('static-handoff-' + s + '-20261005-01/registration.json')
       for s in ('bounded', 'query')},
}
REGISTRATION_SHA = dict(
    image='ca778f3b85bfa8482148dcb17b30eaef279af196b321a9bcfa8d2a8e98245b5b',
    thinking='2ef1ce62b62392ac73f070c3cdb4355e9946fe83b73c008738ea73a1ba1849a0',
    feedback='88eaf36d6d7b1e6402aff7828ef865f4b2aaed5df1df567a0688e75eeadfb340',
    bounded='db4dcb4259fa0276da5fb84013602a07613af305d149c51bccc6894d39cb6725',
    query='4b49754525ad857390bcff5ef13c73746e3a6850fc14b10995276ce8d2d76d07',
)
old = previous.q
ORIGINAL_IMAGE_RUN = old.image.run
ORIGINAL_IMAGE_READY = old.old.image_ready


def pinned(path, sha):
    old.image.pinned(Path(path), sha)


def qualification(path):
    result = old.read(path)
    if (result.get('kind') != 'resource_handoff_successors_cpu_qualification'
            or result.get('passed') is not True or result.get('tests_run', 0) < 20
            or result.get('skipped') != 0 or result.get('model_calls') != 0
            or result.get('device_inputs') != 0):
        raise SafetyHalt('Require complete resource-startup and exact-prefix qualification')
    for name in HELPERS:
        pinned(ROOT / 'apps/computer_use' / name, result['source_sha256'][name])
    startup.source_identity()


def check_prefix(protocol, state, *, actual=False):
    rows = state.get('trials', [])
    if (len(rows) != 1 or rows[0]['id'] != protocol['trials'][0]['id']
            or state.get('real_model_trials_completed') != 1):
        raise SafetyHalt('Retain exactly the completed first image trial')
    row, spec = rows[0], protocol['trials'][0]
    for key in ('kind', 'policy', 'repetition'):
        if row[key] != spec[key]:
            raise SafetyHalt('Retained image prefix declaration changed')
    output = Path(ORIGINAL.parent.parent / 'visual-context-retest-20261004/attempts') / row['id']
    result_path, evidence_path = output / 'result.json', output / 'visual-context-evidence.json'
    pinned(result_path, row['result_sha256']); pinned(evidence_path, row['evidence_sha256'])
    result, evidence = old.read(result_path), old.read(evidence_path)
    old.base.terminal_trial(spec, result)
    if (result != row['result'] or result.get('comparison_protocol_sha256') != state['protocol_sha256']
            or evidence.get('result_sha256') != row['result_sha256']
            or evidence.get('policy') != spec['policy'] or evidence.get('model_profile') != protocol['model']
            or evidence.get('business_grade_unchanged') is not True
            or len(evidence.get('traces', [])) != row['trace_count']):
        raise SafetyHalt('Retained result or actual image evidence changed')
    if actual:
        owned.assert_no_services(output); owned.assert_vm_retired(output)
        for trace in evidence['traces']:
            if old.image.verify_trace(output / 'app/inference' / trace['trace'], spec['policy']) != trace:
                raise SafetyHalt('Retained actual model request archive changed')
    return copy.deepcopy(rows)


def terminal_scope(*, actual=False):
    pinned(ORIGINAL, ORIGINAL_SHA); pinned(TERMINAL, TERMINAL_SHA); pinned(DIAGNOSTIC, DIAGNOSTIC_SHA)
    declaration, evidence, diagnostic = map(old.read, (ORIGINAL, TERMINAL, DIAGNOSTIC))
    if (diagnostic.get('second_task_dispatched') is not False
            or diagnostic.get('failed_engine_runtime_empty') is not True
            or diagnostic.get('retained_completed_prefix') != 1
            or diagnostic.get('min_free_mib') != 56000
            or diagnostic.get('launcher_lock_timeout_seconds') != 5):
        raise SafetyHalt('Only the recorded pre-model resource failure can continue')
    for field in ('engine_log', 'launcher_source', 'actual_terminal_audit'):
        record = diagnostic[field]; pinned(record['path'], record['sha256'])
    if Path(diagnostic['engine_log']['path']).read_text().strip() != 'Insufficient free GPU memory for the shared profile':
        raise SafetyHalt('Different engine failure; no resource continuation')
    if set(evidence['stages']) != set(REGISTRATIONS):
        raise SafetyHalt('All five halted producer records are required')
    states = {}
    for stage, row in evidence['stages'].items():
        pinned(row['queue']['path'], row['queue']['sha256'])
        pinned(REGISTRATIONS[stage], REGISTRATION_SHA[stage])
        registration, state = old.read(REGISTRATIONS[stage]), old.read(row['queue']['path'])
        if (registration.get('queue') != row['queue']['path']
                or registration['identity']['pid'] != row['controller']['pid']
                or state.get('pid') != registration['identity']['pid']
                or state.get('protocol_sha256') != registration['protocol_sha256']
                or state.get('status') != 'halted-no-further-dispatch'
                or registration['identity'].get('uid') != os.getuid()
                or registration['identity'].get('cwd') != str(ROOT)):
            raise SafetyHalt('Original halted producer identity changed')
        if stage != 'image' and state.get('trials') != []:
            raise SafetyHalt('Downstream recovery requires untouched zero-trial cohorts')
        if actual and not old.base.process_retired(registration['identity']):
            raise SafetyHalt('A former producer is still live; no replacement allowed')
        states[stage] = state
    protocol = previous.historical_image(Path(declaration['image_protocol']))
    prefix = check_prefix(protocol, states['image'], actual=actual)
    image_state = states['image']
    failed = image_state['current_child']
    failed_path = Path(failed['command'][failed['command'].index('--run-dir') + 1])
    if (image_state.get('current_trial') != protocol['trials'][1]['id']
            or not failed_path.is_dir() or list(failed_path.iterdir())
            or failed_path != Path(declaration['image_protocol']).parent / 'engines' / protocol['trials'][1]['id']):
        raise SafetyHalt('Failed runtime changed or a second model task may have started')
    if actual:
        for identity in (failed, image_state['registered_engine']):
            if not old.base.process_retired(identity):
                raise SafetyHalt('Former engine or launcher is still live')
    sdk = previous.terminal_scope(actual=actual)['sdk']
    return dict(declaration=declaration, states=states, prefix=prefix, sdk=sdk, evidence=evidence)


def retired_sdk_ready(original_ready, protocol, *args):
    """A retired SDK engine is provenance only, never ready model capacity."""
    scope = terminal_scope(actual=True)
    sdk = scope['sdk']
    if (protocol['sdk_protocol'] != scope['declaration']['sdk_protocol']
            or protocol['sdk_queue'] != old.read(scope['declaration']['sdk_registration'])['queue']):
        raise SafetyHalt('Retired SDK handoff belongs to a different producer')
    real_register = owned.register_engine

    def register(runtime, expected):
        if str(runtime) != sdk['engine_runtime']:
            return real_register(runtime, expected)
        if expected != protocol['engine_off_identity']:
            raise SafetyHalt('Retired SDK receipt cannot replace another engine profile')
        manifest = old.read(Path(runtime) / 'manifest.json')
        from osworld_linux.campaign import engine_identity
        if engine_identity(Path(runtime)) != expected or manifest.get('pid') != sdk['registered_engine']['pid']:
            raise SafetyHalt('Retired SDK manifest or generation changed')
        if not old.base.process_retired(sdk['registered_engine']):
            raise SafetyHalt('The original SDK engine has not retired')
        return copy.deepcopy(sdk['registered_engine'])

    with patch.object(owned, 'register_engine', register):
        result = original_ready(protocol, *args)
    if result is not None:
        result['sdk_engine_retired_before_prefix_continuation'] = True
        result['retained_image_prefix'] = [r['id'] for r in scope['prefix']]
    return result


def derive(declaration, output, cpu):
    result = copy.deepcopy(declaration)
    result.update(recovery_kind='resource-startup-and-one-row-prefix', recovery_parent=str(ORIGINAL),
                  recovery_parent_sha256=ORIGINAL_SHA, recovery_terminal_audit=str(TERMINAL),
                  recovery_terminal_audit_sha256=TERMINAL_SHA, recovery_diagnostic=str(DIAGNOSTIC),
                  recovery_diagnostic_sha256=DIAGNOSTIC_SHA, qualification=str(cpu),
                  retained_image_prefix_count=1, launcher_resource_wait_seconds=startup.STARTUP_WAIT_SECONDS,
                  engine_evidence_root=str(Path(output) / 'engines'),
                  scope='Retain the scored first image trial, execute only remaining frozen trials. Fresh controller/engine records and60-second launcher admission wait; original model, task and readiness budgets remain.')
    for stage in STAGES:
        result[stage + '_state_dir'] = str(Path(output) / (stage + '-queue'))
        result[stage + '_registration'] = str(Path(output) / (stage + '-registration.json'))
    files = [ORIGINAL, TERMINAL, DIAGNOSTIC, Path(cpu), *REGISTRATIONS.values(),
             *[ROOT / 'apps/computer_use' / name for name in HELPERS]]
    result['files'] = copy.deepcopy(declaration['files']) + [dict(path=str(p), sha256=owned.digest(p)) for p in files]
    return result


def validate(path):
    path = Path(path); d = old.read(path); scope = terminal_scope()
    if d != derive(scope['declaration'], path.parent, Path(d['qualification'])):
        raise SafetyHalt('Resource recovery changed a frozen trial or profile')
    qualification(d['qualification'])
    _, canonical, mr = previous.validate(ORIGINAL)
    return d, canonical, mr


def successor_registration(path, stage, declaration, d):
    if stage == 'sdk':
        return previous.successor_registration(path, stage, ORIGINAL, old.read(ORIGINAL))
    return BASE_REGISTRATION(path, stage, declaration, d)


def successor_queue(registration, declaration):
    if registration.get('stage') == 'sdk':
        return previous.successor_queue(registration, ORIGINAL)
    return BASE_QUEUE(registration, declaration)


def resume_source(source):
    for before, after in (
        ('planned_trials=12, trials=[],', 'planned_trials=12, trials=copy.deepcopy(retained_prefix),'),
        ('real_model_trials_completed=0, model_concurrency=1',
         'real_model_trials_completed=len(retained_prefix), model_concurrency=1'),
        ('for row in p["trials"]:', 'for row in p["trials"][len(retained_prefix):]:'),
    ):
        if source.count(before) != 1:
            raise SafetyHalt('Frozen partial-prefix loop anchor changed')
        source = source.replace(before, after, 1)
    return source


def resumed_image_run(args):
    startup.source_identity()
    namespace = dict(ORIGINAL_IMAGE_RUN.__globals__)
    namespace.update(copy=copy, retained_prefix=terminal_scope()['prefix'])
    exec(compile(resume_source(inspect.getsource(ORIGINAL_IMAGE_RUN)), __file__, 'exec'), namespace)
    return namespace['run'](args)


def prepare(output, cpu):
    output, cpu = Path(output).resolve(), Path(cpu).resolve()
    qualification(cpu); scope = terminal_scope(actual=True); previous.validate(ORIGINAL)
    if output.exists():
        raise SafetyHalt('Use a fresh recovery directory; retain every original attempt')
    for stage in STAGES:
        p = old.frozen_protocol(stage, scope['declaration'][stage + '_protocol'])
        for row in p['trials'][1 if stage == 'image' else 0:]:
            if (Path(scope['declaration'][stage + '_protocol']).parent / 'attempts' / row['id']).exists():
                raise SafetyHalt('An unretained attempt already exists; cannot dispatch again')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'successors.json'; old.save(path, derive(scope['declaration'], output, cpu)); validate(path)
    return path


q = previous.build_namespace()
q.__file__, q.MODULE, q.validate, q.qualification = __file__, MODULE, validate, qualification
BASE_REGISTRATION, BASE_QUEUE = q.successor_registration, q.successor_queue
q.successor_registration, q.successor_queue = successor_registration, successor_queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    for name in ('output', 'qualification', 'declaration'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--wait-seconds', type=int, default=172800)
    parser.add_argument('--resource-wait-seconds', type=int, default=21600)
    parser.add_argument('--min-free-mib', type=int, default=56000)
    args = parser.parse_args()
    if not (60 <= args.wait_seconds <= 172800 and 60 <= args.resource_wait_seconds <= 86400 and args.min_free_mib == 56000):
        parser.error('Keep the original bounded shared-resource profile')
    if args.prepare:
        if args.output is None or args.qualification is None:
            parser.error('Provide qualification and fresh output')
        print(dict(declaration=str(prepare(args.output, args.qualification)), model_calls=0, device_inputs=0)); return
    if args.declaration is None or args.stage is None:
        parser.error('Provide immutable declaration and stage')
    args.declaration = args.declaration.resolve(); terminal_scope(actual=True)
    d, canonical, mr = validate(args.declaration)
    if sys.executable != d['controller_python']:
        raise SafetyHalt('Keep the qualified controller interpreter')
    args.protocol, args.state_dir = Path(d[args.stage + '_protocol']), Path(d[args.stage + '_state_dir'])
    def interrupted(*_):
        raise SafetyHalt('Resource continuation interrupted; owned work and evidence preserved')
    signal.signal(signal.SIGTERM, interrupted)
    guarded_image, guarded_feedback = startup.guarded_controller(q.image.Controller), startup.guarded_controller(q.feedback.Controller)
    def ready(protocol, *values):
        return retired_sdk_ready(ORIGINAL_IMAGE_READY, protocol, *values)
    with patch.object(q.old, 'IMAGE_VALIDATE', previous.historical_image), \
            patch.object(q.image, 'Controller', guarded_image), patch.object(q.feedback, 'Controller', guarded_feedback), \
            patch.object(q.image, 'run', resumed_image_run), patch.object(q.old, 'image_ready', ready):
        result = {'image': q.run_image, 'thinking': q.run_thinking, 'feedback': q.run_feedback}[args.stage](args, d, canonical, mr)
    print(dict(status=result['status'], completed=len(result['trials'])))


if __name__ == '__main__':
    main()
