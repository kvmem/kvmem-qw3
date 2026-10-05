"""Restore untouched bounded/query cohorts after static admission recovery."""
from pathlib import Path
import types
from unittest.mock import patch

import static_handoff_successors as successors
import shared_quota_staged_deferred as previous

MODULE='static_handoff_deferred'
HELPERS=(MODULE+'.py','tests/test_'+MODULE+'.py',
         'static_handoff_successors.py','tests/test_static_handoff_successors.py',
         'tests/test_bounded_continuation_retest_queue.py','tests/test_query_refresh_replay_queue.py',
         'shared_quota_port_deferred.py','shared_quota_handoff_deferred.py')
PREVIOUS_SHA='3c481a79c339c3c89302f75f45d4ef501c7b999617daf5192ffd820ce382bc38'
admission=previous.admission


def qualification(path):
    result=successors.q.read(path)
    if (result.get('kind')!='static_handoff_deferred_cpu_qualification'
            or result.get('passed') is not True or result.get('tests_run',0)<38
            or result.get('skipped')!=0 or result.get('model_calls')!=0 or result.get('device_inputs')!=0):
        raise successors.SafetyHalt('Require complete deferred inventories and static handoff routing qualification')
    for name in HELPERS:
        successors.q.image.pinned(successors.ROOT/'apps/computer_use'/name,result['source_sha256'][name])


def build_namespace():
    source_path=Path(previous.previous.__file__)
    if successors.owned.digest(source_path)!=PREVIOUS_SHA:
        raise successors.SafetyHalt('Original deferred routing source changed')
    source=source_path.read_text()
    replacements=(
        ('import shared_quota_port_successors as successors','import static_handoff_successors as successors',1),
        ('q = successors.RUNTIME','q = successors.q',1),
        ('MODULE = "shared_quota_port_deferred"','MODULE = "static_handoff_deferred"',1),
        ('"qualified_frozen_port_deferred_routing"','"qualified_frozen_static_handoff_deferred_routing"',2),
        ('"actual_frozen_port_deferred_identity"','"actual_frozen_static_handoff_deferred_identity"',1),
        ('"shared-quota-port-successors-20261005-01/successors.json"','"static-handoff-successors-20261005-01/successors.json"',1),
    )
    for before,after,count in replacements:
        if source.count(before)!=count:raise successors.SafetyHalt('Frozen deferred routing anchor changed')
        source=source.replace(before,after)
    module=types.ModuleType('qualified_static_handoff_deferred_runtime');module.__file__=__file__
    exec(compile(source,str(source_path),'exec'),module.__dict__)
    module.HELPERS,module.qualification=HELPERS,qualification
    module.BOUNDED.Controller=admission.controller(module.BOUNDED.Controller)
    module.QUERY.Controller=admission.controller(module.QUERY.Controller)
    return module


NAMESPACE=build_namespace()
for _name,_value in vars(NAMESPACE).items():
    if not _name.startswith('__') and _name not in {'MODULE','HELPERS','qualification'}:
        globals()[_name]=_value


if __name__=='__main__':
    with patch.object(successors.q.old,'IMAGE_VALIDATE',successors.historical_image):
        NAMESPACE.main()
