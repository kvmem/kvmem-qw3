"""Resume frozen successors after the recorded pre-task engine handoff failure.

Keep the full predecessor inventories and owned service checks. The one retired
memory engine is admitted only with the exact failed SDK retirement receipt.
Every new engine launch waits for the original port, without relaunching a child.
"""
import copy
from pathlib import Path
import types
from unittest.mock import patch

import shared_quota_port_successors as previous
import shared_quota_port_deferred as stopped_deferred

ROOT, owned, SafetyHalt = previous.ROOT, previous.owned, previous.SafetyHalt
MODULE = "shared_quota_handoff_successors"
SOURCE_SHA = "ead3b577f47e794b6969ffaa853382092918b2df9a02ae43128ae7a07631a534"
FAILURE_AUDIT = ROOT / "apps/computer_use/.runtime/shared-quota-current-progress-20261005-22.json"
FAILURE_SHA = "ff9b0c7d96f93f6b61d88dddd41c9ec28c7d7d5dba9d5554d455ae8111e451e0"
FAILED_DECLARATION = ROOT / "apps/computer_use/.runtime/shared-quota-port-successors-20261005-01/successors.json"
HELPERS = (MODULE + ".py", "tests/test_" + MODULE + ".py",
           "shared_quota_port_successors.py", "tests/test_shared_quota_port_successors.py",
           "tests/test_shared_quota_terminal_handoff.py", "shared_quota_port_continuation.py",
           "shared_quota_port_deferred.py", "tests/test_shared_quota_port_continuation.py")


def qualification(path):
    result = previous.port.read(path)
    if (result.get("kind") != "shared_quota_handoff_successors_cpu_qualification"
            or result.get("passed") is not True or result.get("tests_run", 0) < 60
            or result.get("skipped") != 0 or result.get("model_calls") != 0
            or result.get("device_inputs") != 0):
        raise SafetyHalt("Require the complete retired-engine and port-admission qualification")
    for name in HELPERS:
        previous.port.old.image.pinned(ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def failure_scope(*, require_absent=False):
    """Validate the recorded six zero-trial halts, never a generic engine death."""
    previous.port.old.image.pinned(FAILURE_AUDIT, FAILURE_SHA)
    audit = previous.port.read(FAILURE_AUDIT)
    if (audit.get("kind") != "actual_completed_memory_pre_task_port_halt_readonly_recheck"
            or audit.get("model_requests_submitted") != 0 or audit.get("device_inputs") != 0
            or set(audit.get("successors", {})) != {"sdk", "image", "thinking", "feedback", "bounded", "query"}):
        raise SafetyHalt("Recovery is limited to the recorded pre-task handoff halt")
    declaration, _, memory = previous.RUNTIME.validate(FAILED_DECLARATION)
    state = previous.memory_queue(memory, Path(declaration["memory_continuation"]))
    if state.get("status") != "completed" or len(state.get("trials", [])) != 8:
        raise SafetyHalt("Require the unchanged completed eight-trial memory producer")
    for stage, receipt in audit["successors"].items():
        record = previous.port.read(receipt["registration"])
        halted = previous.port.read(record["queue"])
        if (halted.get("status") != "halted-no-further-dispatch" or halted.get("trials") != []
                or halted.get("pid") != record["identity"]["pid"]
                or record.get("protocol_sha256") != owned.digest(record["protocol"])
                or halted.get("protocol_sha256") != record["protocol_sha256"]
                or not previous.RUNTIME.base.process_retired(record["identity"])):
            raise SafetyHalt("A failed predecessor was changed, remained live or dispatched a task")
        def halted_inventory(candidate, identity, declaration_sha, key):
            # The original guard intentionally rejects every halted producer.
            # This read-only check admits only this exact recorded zero-trial
            # failure; the original wrapper/profile validators still run.
            if (candidate != halted or identity != record["identity"]
                    or candidate.get(key) != declaration_sha
                    or not previous.RUNTIME.base.process_retired(identity)):
                raise SafetyHalt("Recorded halted producer identity or digest changed")
            return candidate
        with patch.object(previous.RUNTIME.old, "guard_queue", halted_inventory):
            if stage in previous.RUNTIME.STAGES:
                previous.RUNTIME.successor_registration(Path(receipt["registration"]), stage, FAILED_DECLARATION, declaration)
                previous.RUNTIME.successor_queue(record, FAILED_DECLARATION)
            else:
                stopped_deferred.deferred_registration(Path(receipt["registration"]), stage)
    sdk = previous.port.read(audit["successors"]["sdk"]["queue"])
    fail = audit["sdk_startup_failure"]
    final = state["final_registered_engine"]
    child = fail["current_child"]
    runtime = Path(declaration["sdk_protocol"]).parent / "engine-off"
    expected_command = [str(previous.RUNTIME.CONTROLLER_PYTHON), str(ROOT / "apps/computer_use/run_engine.py"),
        "--run-dir", str(runtime), "--method", "dflash2", "--kvmem", "off", "--prefix-reuse", "off",
        "--visual-history", "pin-all", "--guided-refresh", "off", "--context-tokens", "524288",
        "--gpu-sharing", "spare-capacity", "--min-free-mib", "56000", "--lock-timeout", "5"]
    if (sdk.get("error") != "SafetyHalt: New owned service exited before readiness"
            or any(key in sdk for key in ("engine_runtime", "registered_engine", "new_gateways", "model_client"))
            or sdk.get("current_child") != child or child.get("command") != expected_command
            or child.get("uid") != final.get("uid") or child.get("cwd") != str(ROOT)
            or sdk.get("engine_retirement") != {"pid": final["pid"], "retired": True, "new_instance_preserved": False}
            or state.get("final_engine_policy") != "on" or audit["memory"]["final_engine"] != final
            or fail.get("engine_runtime_created") is not False or fail.get("new_task_dispatch_count") != 0
            or not previous.RUNTIME.base.process_retired(child)
            or not previous.RUNTIME.base.process_retired(final)):
        raise SafetyHalt("The exact owned memory retirement and failed launcher are not proven")
    previous.port.old.image.pinned(fail["launcher_log"], fail["launcher_log_sha256"])
    if "Requested engine port is already in use; select an unused port" not in Path(fail["launcher_log"]).read_text():
        raise SafetyHalt("The preserved launcher did not report the declared port failure")
    if require_absent and runtime.exists():
        raise SafetyHalt("A fresh OFF runtime already exists; no recovery startup")
    return dict(audit=audit, memory=state, final_engine=final, failed_sdk=sdk)


def failure_files():
    audit = previous.port.read(FAILURE_AUDIT)
    paths = [FAILURE_AUDIT, FAILED_DECLARATION, Path(audit["memory"]["queue"]),
             Path(audit["sdk_startup_failure"]["launcher_log"])]
    for row in audit["successors"].values():
        paths.extend((Path(row["registration"]), Path(row["queue"])))
    return list(dict.fromkeys(paths))


def retired_handoff_ready(original_ready, protocol, memory_protocol, declaration):
    """Run all original receipts; admit only the explicitly retired final engine."""
    scope = failure_scope()
    state = scope["memory"]
    if (declaration["memory_queue"] != scope["audit"]["memory"]["queue"]
            or declaration["memory_protocol"] != scope["audit"]["memory"]["protocol"]):
        raise SafetyHalt("Retired handoff does not belong to this exact memory producer")
    original_registration = owned.register_engine
    def registration(runtime, expected):
        if str(runtime) != state["current_engine_runtime"] or expected != memory_protocol["engine_identities"]["on"]:
            raise SafetyHalt("Retired-engine receipt cannot substitute for another runtime or profile")
        if not previous.RUNTIME.base.process_retired(scope["final_engine"]):
            raise SafetyHalt("Recorded final engine is still live")
        return copy.deepcopy(scope["final_engine"])
    with patch.object(owned, "register_engine", registration):
        result = original_ready(protocol, memory_protocol)
    if result is not None:
        result["registered_engine_retired_before_recovery"] = True
        result["retirement_receipt"] = copy.deepcopy(scope["failed_sdk"]["engine_retirement"])
        result["retired_engine_failure_audit_sha256"] = FAILURE_SHA
    # register_engine returns to its original function before restore/deploy.
    assert owned.register_engine is original_registration
    return result


def admitted_spawn(controller, command, name, framework, spawn):
    if len(command) > 1 and command[1] == str(ROOT / "apps/computer_use/run_engine.py"):
        controller.checkpoint(engine_port_admission=previous.port.wait_for_engine_port())
    return spawn(command, name, framework)


def patched(source):
    source = previous.patched(source)
    replacements = (
        ('"actual_shared_quota_port_successor_identity"', '"actual_shared_quota_handoff_successor_identity"'),
        ('return original_ready(p, kp)', 'return retired_handoff_ready(original_ready, p, kp, d)'),
        ('declared_at=time.time(), counts=COUNTS,',
         'declared_at=time.time(), counts=COUNTS, startup_failure_audit=str(FAILURE_AUDIT), '
         'startup_failure_audit_sha256=FAILURE_SHA, engine_port_admission_seconds=120, '
         'retired_engine_scope="exact-failed-sdk-retirement",'),
        ('files = [continuation, registration,',
         'files = [*failure_files(), continuation, registration,'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise SafetyHalt("Frozen handoff routing anchor changed")
        source = source.replace(before, after, 1)
    before = '"qualified_shared_quota_port_frozen_successors"'
    if source.count(before) != 2:
        raise SafetyHalt("Frozen handoff declaration anchors changed")
    source = source.replace(before, '"qualified_shared_quota_handoff_frozen_successors"')
    for parent in ("sdk.Controller", "image.Controller", "feedback.Controller"):
        before = f"class Controller({parent}):\n        def __init__"
        expected = 2 if parent == "image.Controller" else 1
        if source.count(before) != expected:
            raise SafetyHalt("Frozen engine spawn override anchor changed")
        after = (f"class Controller({parent}):\n"
                 "        def spawn(self, command, name, framework=None):\n"
                 "            return admitted_spawn(self, command, name, framework, super().spawn)\n"
                 "        def __init__")
        source = source.replace(before, after)
    return source


def validate(path):
    declaration, canonical, memory = RUNTIME.FROZEN_VALIDATE(path)
    if (declaration.get("startup_failure_audit") != str(FAILURE_AUDIT)
            or declaration.get("startup_failure_audit_sha256") != FAILURE_SHA
            or declaration.get("engine_port_admission_seconds") != 120
            or declaration.get("retired_engine_scope") != "exact-failed-sdk-retirement"):
        raise SafetyHalt("The engine handoff recovery declaration changed")
    failure_scope()
    return declaration, canonical, memory


def prepare(args):
    scope = failure_scope(require_absent=True)
    if str(args.memory_registration.resolve()) != scope["audit"]["memory"]["registration"]:
        raise SafetyHalt("Keep the completed eight-trial producer unchanged")
    return RUNTIME.FROZEN_PREPARE(args)


def build_namespace():
    if owned.digest(previous.__file__) != SOURCE_SHA:
        raise SafetyHalt("Frozen port-successor source changed")
    module = types.ModuleType("qualified_shared_quota_handoff_successors_runtime")
    module.__file__ = __file__
    original = previous.prior.original
    exec(compile(patched(Path(original.__file__).read_text()), "frozen-successors-engine-handoff", "exec"), module.__dict__)
    quota = types.SimpleNamespace(**vars(previous.port))
    quota.validate_protocol = previous.port.previous.validate_protocol
    module.__dict__.update(MODULE=MODULE, HELPERS=HELPERS, quota=quota,
        qualification=qualification, memory_declaration=previous.port.validate,
        memory_registration=previous.memory_registration, memory_queue=previous.memory_queue,
        admitted_spawn=admitted_spawn, retired_handoff_ready=retired_handoff_ready,
        FAILURE_AUDIT=FAILURE_AUDIT, FAILURE_SHA=FAILURE_SHA, failure_files=failure_files)
    module.FROZEN_VALIDATE, module.FROZEN_PREPARE = module.validate, module.prepare
    module.validate, module.prepare = validate, prepare
    return module


RUNTIME = build_namespace()


if __name__ == "__main__":
    RUNTIME.main()
