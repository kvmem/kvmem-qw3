"""Route unchanged bounded/query trials to the qualified handoff successors."""
from pathlib import Path
import types

import shared_quota_handoff_successors as successors
import shared_quota_port_deferred as previous

PREVIOUS_SHA = "3c481a79c339c3c89302f75f45d4ef501c7b999617daf5192ffd820ce382bc38"
MODULE = "shared_quota_handoff_deferred"
HELPERS = (MODULE + ".py", "tests/test_" + MODULE + ".py",
           "tests/test_bounded_continuation_retest_queue.py", "tests/test_query_refresh_replay_queue.py",
           "shared_quota_handoff_successors.py", "shared_quota_port_deferred.py")


def qualification(path):
    result = previous.read(path)
    if (result.get("kind") != "shared_quota_handoff_deferred_cpu_qualification"
            or result.get("passed") is not True or result.get("tests_run", 0) < 45
            or result.get("skipped") != 0 or result.get("model_calls") != 0 or result.get("device_inputs") != 0):
        raise previous.SafetyHalt("Require completed handoff deferred routing and port-admission qualification")
    for name in HELPERS:
        previous.q.image.pinned(previous.ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def admitted_spawn(controller, command, name, framework, spawn):
    is_refresh = len(command) > 2 and command[1:3] == ["-m", "query_refresh_engine"]
    is_original = len(command) > 1 and command[1] == str(previous.ROOT / "apps/computer_use/run_engine.py")
    if is_refresh or is_original:
        controller.checkpoint(engine_port_admission=successors.previous.port.wait_for_engine_port())
    return spawn(command, name, framework)


def controller(original):
    class AdmissionController(original):
        def spawn(self, command, name, framework=None):
            return admitted_spawn(self, command, name, framework, super().spawn)
    return AdmissionController


def build_namespace():
    if previous.owned.digest(previous.__file__) != PREVIOUS_SHA:
        raise previous.SafetyHalt("Frozen deferred routing source changed")
    source = Path(previous.__file__).read_text()
    replacements = (
        ("import shared_quota_port_successors as successors", "import shared_quota_handoff_successors as successors"),
        ('MODULE = "shared_quota_port_deferred"', 'MODULE = "shared_quota_handoff_deferred"'),
        ('"qualified_frozen_port_deferred_routing"', '"qualified_frozen_handoff_deferred_routing"'),
        ('"actual_frozen_port_deferred_identity"', '"actual_frozen_handoff_deferred_identity"'),
        ('"shared-quota-port-successors-20261005-01/successors.json"', '"shared-quota-handoff-successors-20261005-01/successors.json"'),
    )
    for before, after in replacements:
        expected = 2 if before == '"qualified_frozen_port_deferred_routing"' else 1
        if source.count(before) != expected:
            raise previous.SafetyHalt("Frozen handoff deferred routing anchor changed")
        source = source.replace(before, after)
    module = types.ModuleType("qualified_handoff_deferred_runtime")
    module.__file__ = __file__
    exec(compile(source, previous.__file__, "exec"), module.__dict__)
    module.HELPERS, module.qualification = HELPERS, qualification
    module.BOUNDED.Controller = controller(module.BOUNDED.Controller)
    module.QUERY.Controller = controller(module.QUERY.Controller)
    return module


NAMESPACE = build_namespace()
for _name, _value in vars(NAMESPACE).items():
    if not _name.startswith("__") and _name not in {"HELPERS", "qualification", "MODULE"}:
        globals()[_name] = _value


if __name__ == "__main__":
    NAMESPACE.main()
