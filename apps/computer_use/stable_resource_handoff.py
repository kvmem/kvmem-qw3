"""Retain both failed admissions and wait for sustained headroom for the same trials."""
from pathlib import Path
import types

import resource_handoff_successors as previous

MODULE = 'stable_resource_handoff'
SOURCE_SHA = 'c1649c4e08ef0601786a3933d6639e990ca23b41397f84b88a7b1fc99773db12'
RECENT = previous.RUNTIME / 'resource-handoff-terminal-audit-01.json'
RECENT_SHA = '3791f84331c1e6694deab43b6c7fff1b514cee43dd4678ddea7e5484fd6e0ae0'


def recent_scope(*, actual=False):
    previous.pinned(RECENT, RECENT_SHA)
    record = previous.old.read(RECENT)
    if (record.get('kind') != 'actual_resource_sixty_second_prefix_continuation_terminal_audit'
            or record.get('new_model_trial_dispatched') is not False or record.get('retained_prefix') != 1
            or set(record['stages']) != set(previous.STAGES)):
        raise previous.SafetyHalt('Use only the proven sixty-second pre-model admission failure')
    for item in (record['declaration'], record['engine_log']):previous.pinned(item['path'], item['sha256'])
    if Path(record['engine_log']['path']).read_text().strip() != 'Insufficient free GPU memory for the shared profile':
        raise previous.SafetyHalt('Another startup failure cannot use stable headroom recovery')
    for stage, row in record['stages'].items():
        for key in ('registration', 'queue'):previous.pinned(row[key]['path'], row[key]['sha256'])
        reg, state = previous.old.read(row['registration']['path']), previous.old.read(row['queue']['path'])
        if (reg['identity'] != row['identity'] or state['pid'] != reg['identity']['pid']
                or state['status'] != 'halted-no-further-dispatch'
                or len(state['trials']) != (1 if stage == 'image' else 0)):
            raise previous.SafetyHalt('Recent halted producer or retained prefix changed')
        if actual and not previous.old.base.process_retired(reg['identity']):
            raise previous.SafetyHalt('A recent producer is still live; do not replace')
    if not Path(record['failed_runtime']).is_dir() or list(Path(record['failed_runtime']).iterdir()):
        raise previous.SafetyHalt('Recent failed runtime changed; retain every partial artifact')
    if actual and not previous.old.base.process_retired(record['failed_child']):
        raise previous.SafetyHalt('Recent launcher still live; do not restart')
    return record


def build_namespace():
    previous.pinned(previous.__file__, SOURCE_SHA)
    source = Path(previous.__file__).read_text()
    replacements = (
        ('import resource_startup as startup', 'import stable_resource_startup as startup'),
        ("MODULE = 'resource_handoff_successors'", "MODULE = 'stable_resource_handoff'"),
        ("HELPERS = (MODULE + '.py', 'resource_startup.py', 'tests/test_' + MODULE + '.py')",
         "HELPERS = (MODULE + '.py', 'stable_resource_startup.py', 'tests/test_' + MODULE + '.py', 'resource_startup.py', 'resource_handoff_successors.py', 'tests/test_resource_handoff_successors.py')"),
        ("'resource_handoff_successors_cpu_qualification'", "'stable_resource_handoff_cpu_qualification'"),
    )
    for before, after in replacements:
        if source.count(before) != 1:raise previous.SafetyHalt('Qualified stable recovery anchor changed')
        source = source.replace(before, after, 1)
    module = types.ModuleType('qualified_stable_resource_handoff');module.__file__ = __file__
    exec(compile(source, previous.__file__, 'exec'), module.__dict__)
    original_scope, original_derive = module.terminal_scope, module.derive

    def terminal_scope(*, actual=False):
        recent_scope(actual=actual)
        return original_scope(actual=actual)

    def derive(declaration, output, cpu):
        result = original_derive(declaration, output, cpu)
        record = recent_scope()
        result.update(resource_headroom_stability_seconds=module.startup.STABLE_SECONDS,
                      recent_resource_failure_audit=str(RECENT), recent_resource_failure_audit_sha256=RECENT_SHA)
        paths = [RECENT, Path(record['declaration']['path']), Path(record['engine_log']['path'])]
        paths += [Path(row[key]['path']) for row in record['stages'].values() for key in ('registration', 'queue')]
        result['files'] += [dict(path=str(p), sha256=previous.owned.digest(p)) for p in paths]
        return result

    module.terminal_scope, module.derive = terminal_scope, derive
    return module


NAMESPACE = build_namespace()
for _name, _value in vars(NAMESPACE).items():
    if not _name.startswith('__') and _name not in {'MODULE', 'SOURCE_SHA', 'RECENT', 'RECENT_SHA', 'build_namespace', 'recent_scope', 'previous'}:
        globals()[_name] = _value


if __name__ == '__main__':NAMESPACE.main()
