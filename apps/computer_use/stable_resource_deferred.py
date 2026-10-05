"""Retain frozen bounded/query trials behind the sustained-headroom continuation."""
from pathlib import Path
import types
from unittest.mock import patch

import stable_resource_handoff as successors
import static_handoff_deferred as previous
import stable_resource_startup as startup
import stable_query_startup as query_startup

MODULE = 'stable_resource_deferred'
HELPERS = (MODULE + '.py', 'tests/test_' + MODULE + '.py',
           *successors.HELPERS, 'stable_query_startup.py', 'resource_query_startup.py',
           'tests/test_bounded_continuation_retest_queue.py', 'tests/test_query_refresh_replay_queue.py',
           'shared_quota_port_deferred.py', 'shared_quota_handoff_deferred.py')


def qualification(path):
    result = successors.q.read(path)
    if (result.get('kind') != 'stable_resource_deferred_cpu_qualification'
            or result.get('passed') is not True or result.get('tests_run', 0) < 38
            or result.get('skipped') != 0 or result.get('model_calls') != 0
            or result.get('device_inputs') != 0):
        raise successors.SafetyHalt('Require complete frozen full-handoff and stable deferred-routing qualification')
    for name in HELPERS:
        successors.pinned(successors.ROOT / 'apps/computer_use' / name, result['source_sha256'][name])
    startup.source_identity()


def build_namespace():
    source_path = successors.ROOT / 'apps/computer_use/shared_quota_port_deferred.py'
    successors.pinned(source_path, previous.PREVIOUS_SHA)
    source = source_path.read_text()
    replacements = (
        ('import shared_quota_port_successors as successors', 'import stable_resource_handoff as successors', 1),
        ('q = successors.RUNTIME', 'q = successors.q', 1),
        ('MODULE = "shared_quota_port_deferred"', 'MODULE = "stable_resource_deferred"', 1),
        ('"qualified_frozen_port_deferred_routing"', '"qualified_frozen_stable_resource_deferred_routing"', 2),
        ('"actual_frozen_port_deferred_identity"', '"actual_frozen_stable_resource_deferred_identity"', 1),
        ('"shared-quota-port-successors-20261005-01/successors.json"',
         '"stable-resource-handoff-successors-20261005-01/successors.json"', 1),
    )
    for before, after, count in replacements:
        if source.count(before) != count:raise successors.SafetyHalt('Frozen stable deferred-routing anchor changed')
        source = source.replace(before, after)
    module = types.ModuleType('qualified_stable_resource_deferred_runtime');module.__file__ = __file__
    exec(compile(source, str(source_path), 'exec'), module.__dict__)
    module.HELPERS, module.qualification = HELPERS, qualification
    module.BOUNDED.Controller = startup.guarded_controller(previous.admission.controller(module.BOUNDED.Controller))
    module.QUERY.Controller = query_startup.guarded_controller(previous.admission.controller(module.QUERY.Controller))
    return module


NAMESPACE = build_namespace()
for _name, _value in vars(NAMESPACE).items():
    if not _name.startswith('__') and _name not in {'MODULE', 'HELPERS', 'qualification'}:
        globals()[_name] = _value


if __name__ == '__main__':
    with patch.object(successors.q.old, 'IMAGE_VALIDATE', successors.previous.previous.historical_image):
        NAMESPACE.main()
