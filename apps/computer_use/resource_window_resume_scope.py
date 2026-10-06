"""Admit fresh routing after the exact seven retired resource-wait producers."""
from pathlib import Path

import stable_resource_handoff as stable

ROOT, RUNTIME, owned = stable.ROOT, stable.RUNTIME, stable.owned
SafetyHalt, read = stable.SafetyHalt, stable.old.read
AUDIT = RUNTIME / 'resource-window-resume-20261006-01/audit.json'
AUDIT_SHA = '31d29eab9ad604369b80fb949243cc91ede24be67bbc816ddbc3c7e74027a08b'
DECLARATION = RUNTIME / 'stable-resource-handoff-successors-20261006-02/successors.json'
DECLARATION_SHA = 'e3d249ebb5c36acbfc843f87438e25dea45735c1122e530b69622b21bff5af74'
QUALIFICATION = RUNTIME / 'resource-window-resume-cpu-20261006-01/result.json'
NATIVE_DECLARATION = RUNTIME / 'resource-window-native-20261006-01/declaration.json'
ORIGINAL_NATIVE_DECLARATION = RUNTIME / 'native-control-model-queue-20261006-01/declaration.json'
OWN = ('resource_window_resume_scope.py', 'resource_window_deferred.py',
       'resource_window_native.py', 'resource_window_condition.py',
       'qualify_resource_window_resume.py', 'tests/test_resource_window_resume.py')
ORIGINAL_QUALIFICATIONS = {
    'deferred': RUNTIME / 'stable-resource-deferred-cpu-01/result.json',
    'native': RUNTIME / 'native-control-model-queue-cpu-03/result.json',
    'condition': RUNTIME / 'condition-schema-model-queue-cpu-04/result.json',
}


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=owned.digest(path))


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise SafetyHalt('Frozen routing source anchor changed')
    return source.replace(before, after, 1)


def check_audit(*, actual=False):
    stable.pinned(AUDIT, AUDIT_SHA)
    stable.pinned(DECLARATION, DECLARATION_SHA)
    audit = read(AUDIT)
    if (audit.get('kind') != 'actual_seven_retired_resource_window_resume_admission'
            or set(audit['stages']) != {'image', 'thinking', 'feedback', 'bounded', 'query', 'native', 'condition_schema'}):
        raise SafetyHalt('Require the exact seven retired producers')
    for stage, row in audit['stages'].items():
        for key in ('registration', 'queue'):
            stable.pinned(row[key]['path'], row[key]['sha256'])
        reg, state = read(row['registration']['path']), read(row['queue']['path'])
        if (reg['identity'] != row['identity'] or state['pid'] != reg['identity']['pid']
                or state['status'] != 'halted-no-further-dispatch'
                or len(state['trials']) != (1 if stage == 'image' else 0)):
            raise SafetyHalt('Old producer or completed inventory changed')
        if actual and not stable.old.base.process_retired(reg['identity']):
            raise SafetyHalt('Original producer still live; never replace it')
    if audit['retained_prefix'] != read(audit['stages']['image']['queue']['path'])['trials']:
        raise SafetyHalt('Completed image prefix changed')
    return audit


def sources():
    import stable_resource_deferred as deferred
    import native_control_model_queue as native
    import condition_schema_model_queue as condition
    return tuple(dict.fromkeys((*OWN, *deferred.HELPERS, *native.HELPERS, *condition.HELPERS)))


def qualification(path, stage):
    path = Path(path).resolve()
    record = read(path)
    if (path != QUALIFICATION or record.get('kind') != 'resource_window_serial_routing_cpu_qualification'
            or record.get('passed') is not True or record.get('tests_run', 0) < 25
            or any(record.get(k) != 0 for k in ('errors', 'failures', 'skipped', 'model_calls', 'device_inputs'))
            or set(record.get('source_sha256', {})) != set(sources())
            or record.get('old_audit') != dict(path=str(AUDIT), sha256=AUDIT_SHA)
            or record.get('image_declaration') != dict(path=str(DECLARATION), sha256=DECLARATION_SHA)):
        raise SafetyHalt('Require complete preserved-prefix and serial routing qualification')
    for name, sha in record['source_sha256'].items():
        stable.pinned(ROOT / 'apps/computer_use' / name, sha)
    for name, original in ORIGINAL_QUALIFICATIONS.items():
        if record['original_qualifications'][name] != ref(original):
            raise SafetyHalt('Original qualification reference changed')
    check_audit()
    return ORIGINAL_QUALIFICATIONS[stage]
