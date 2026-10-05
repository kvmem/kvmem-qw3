"""Route the frozen bounded/query cohorts to staged SDK recovery."""
from pathlib import Path
import types

import shared_quota_staged_successors as successors
import shared_quota_port_deferred as previous
import shared_quota_handoff_deferred as admission

PREVIOUS_SHA = "3c481a79c339c3c89302f75f45d4ef501c7b999617daf5192ffd820ce382bc38"
MODULE = "shared_quota_staged_deferred"
HELPERS = (MODULE + ".py", "tests/test_" + MODULE + ".py",
           "tests/test_bounded_continuation_retest_queue.py", "tests/test_query_refresh_replay_queue.py",
           "shared_quota_staged_successors.py", "shared_quota_handoff_deferred.py", "shared_quota_port_deferred.py",
           "tests/test_shared_quota_handoff_deferred.py", "tests/test_shared_quota_port_deferred.py")


def qualification(path):
    result = previous.read(path)
    if (result.get("kind") != "staged_sdk_deferred_cpu_qualification" or result.get("passed") is not True
            or result.get("tests_run", 0) < 45 or result.get("skipped") != 0
            or result.get("model_calls") != 0 or result.get("device_inputs") != 0):
        raise previous.SafetyHalt("Require full frozen deferred inventories and staged routing qualification")
    for name in HELPERS:
        previous.q.image.pinned(previous.ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def build_namespace():
    if previous.owned.digest(previous.__file__) != PREVIOUS_SHA:
        raise previous.SafetyHalt("Frozen deferred source changed")
    source = Path(previous.__file__).read_text()
    for before, after, count in (
        ("import shared_quota_port_successors as successors", "import shared_quota_staged_successors as successors", 1),
        ('MODULE = "shared_quota_port_deferred"', 'MODULE = "shared_quota_staged_deferred"', 1),
        ('"qualified_frozen_port_deferred_routing"', '"qualified_frozen_staged_deferred_routing"', 2),
        ('"actual_frozen_port_deferred_identity"', '"actual_frozen_staged_deferred_identity"', 1),
        ('"shared-quota-port-successors-20261005-01/successors.json"', '"shared-quota-staged-successors-20261005-01/successors.json"', 1),
    ):
        if source.count(before) != count:
            raise previous.SafetyHalt("Frozen staged deferred routing anchor changed")
        source = source.replace(before, after)
    module = types.ModuleType("qualified_staged_deferred_runtime")
    module.__file__ = __file__
    exec(compile(source, previous.__file__, "exec"), module.__dict__)
    module.HELPERS, module.qualification = HELPERS, qualification
    module.BOUNDED.Controller = admission.controller(module.BOUNDED.Controller)
    module.QUERY.Controller = admission.controller(module.QUERY.Controller)
    return module


NAMESPACE = build_namespace()
for _name, _value in vars(NAMESPACE).items():
    if not _name.startswith("__") and _name not in {"HELPERS", "qualification", "MODULE"}:
        globals()[_name] = _value
admitted_spawn, controller = admission.admitted_spawn, admission.controller


if __name__ == "__main__":
    NAMESPACE.main()
