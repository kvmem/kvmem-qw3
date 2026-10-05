"""Route the frozen bounded/query cohorts behind exact-prefix continuation."""
from pathlib import Path
import types
from unittest.mock import patch

import resource_handoff_successors as successors
import static_handoff_deferred as previous
import resource_startup as startup
import resource_query_startup as query_startup

MODULE = 'resource_handoff_deferred'
HELPERS = (MODULE + '.py', 'tests/test_' + MODULE + '.py',
           *successors.HELPERS, 'resource_query_startup.py',
           'tests/test_bounded_continuation_retest_queue.py', 'tests/test_query_refresh_replay_queue.py',
           'shared_quota_port_deferred.py', 'shared_quota_handoff_deferred.py')


def qualification(path):
    result = successors.q.read(path)
    if (result.get('kind') != 'resource_handoff_deferred_cpu_qualification'
            or result.get('passed') is not True or result.get('tests_run', 0) < 38
            or result.get('skipped') != 0 or result.get('model_calls') != 0
            or result.get('device_inputs') != 0):
        raise successors.SafetyHalt('Require frozen full-handoff and resource deferred-routing qualification')
    for name in HELPERS:
        successors.pinned(successors.ROOT / 'apps/computer_use' / name, result['source_sha256'][name])
    startup.source_identity()


def build_namespace():
    source_path = successors.ROOT / 'apps/computer_use/shared_quota_port_deferred.py'
    successors.pinned(source_path, previous.PREVIOUS_SHA)
    source = source_path.read_text()
    replacements = (
        ('import shared_quota_port_successors as successors', 'import resource_handoff_successors as successors', 1),
        ('q = successors.RUNTIME', 'q = successors.q', 1),
        ('MODULE = "shared_quota_port_deferred"', 'MODULE = "resource_handoff_deferred"', 1),
        ('"qualified_frozen_port_deferred_routing"', '"qualified_frozen_resource_handoff_deferred_routing"', 2),
        ('"actual_frozen_port_deferred_identity"', '"actual_frozen_resource_handoff_deferred_identity"', 1),
        ('"shared-quota-port-successors-20261005-01/successors.json"',
         '"resource-handoff-successors-20261005-01/successors.json"', 1),
    )
    for before, after, count in replacements:
        if source.count(before) != count:
            raise successors.SafetyHalt('Frozen resource deferred-routing anchor changed')
        source = source.replace(before, after)
    module = types.ModuleType('qualified_resource_handoff_deferred_runtime')
    module.__file__ = __file__
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    module.HELPERS, module.qualification = HELPERS, qualification
    # Keep both frozen engine supervisors, changing only evidence paths and
    # their existing admission waits. Query's mode-specific profile is retained.
    module.BOUNDED.Controller = startup.guarded_controller(previous.admission.controller(module.BOUNDED.Controller))
    module.QUERY.Controller = query_startup.guarded_controller(previous.admission.controller(module.QUERY.Controller))
    return module


NAMESPACE = build_namespace()
for _name, _value in vars(NAMESPACE).items():
    if not _name.startswith('__') and _name not in {'MODULE', 'HELPERS', 'qualification'}:
        globals()[_name] = _value


if __name__ == '__main__':
    with patch.object(successors.q.old, 'IMAGE_VALIDATE', successors.previous.historical_image):
        NAMESPACE.main()
