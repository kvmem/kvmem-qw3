"""Route unchanged successor trials to the qualified port continuation.

Reuse the frozen successor loops and their full terminal inventory checks.
Only the actual memory producer and new controller state change. The five
finished equal-quota trials stay referenced; no task or model profile changes.
"""
import os
from pathlib import Path
import types

import shared_quota_port_continuation as port
import shared_quota_successors as prior

ROOT, owned, SafetyHalt = port.ROOT, port.owned, port.SafetyHalt
MODULE = "shared_quota_port_successors"
PRIOR_SHA = "d16ef6c30f77bc5ad16c79f53e5bb33041f7f38619442fdc17c5a55c86b2e26b"
PORT_SHA = "223985c661aaf5920c37f39efd311e9e43d0fe53fd43986e2bacf248aab583f5"
HELPERS = (MODULE + ".py", "tests/test_" + MODULE + ".py",
           "tests/test_shared_quota_successors.py", "tests/test_runner_runtime_successors.py",
           "tests/test_shared_quota_terminal_handoff.py")


def qualification(path):
    result = port.read(path)
    if (result.get("kind") != "shared_quota_port_successors_cpu_qualification"
            or result.get("passed") is not True or result.get("tests_run", 0) < 49
            or result.get("skipped") != 0 or result.get("model_calls") != 0
            or result.get("device_inputs") != 0):
        raise SafetyHalt("Require completed port-producer routing qualification")
    for name in HELPERS:
        port.old.image.pinned(ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def memory_registration(path, continuation, runtime):
    record = port.read(path)
    command = [str(port.PYTHON), "-m", port.MODULE, "--declaration", str(continuation)]
    if (record.get("kind") != "actual_shared_quota_port_continuation_identity"
            or runtime.get("kind") != "exact_shared_quota_pre_task_port_continuation"
            or runtime.get("python") != str(port.PYTHON)
            or record.get("declaration") != str(continuation)
            or record.get("declaration_sha256") != owned.digest(continuation)
            or record.get("protocol") != runtime["continued_protocol"]
            or record.get("protocol_sha256") != owned.digest(runtime["continued_protocol"])
            or record.get("queue") != str(Path(runtime["state_dir"]) / "queue.json")
            or record["identity"].get("command") != command
            or record["identity"].get("cwd") != str(ROOT)
            or record["identity"].get("uid") != os.getuid()
            or record.get("planned_trials") != 8
            or record.get("completed_matched_prefix_retained") != 5
            or record.get("remaining_model_trials") != 3
            or record.get("original_failed_launch_and_controller_preserved") is not True):
        raise SafetyHalt("Actual port-continuation registration changed")
    for name in port.HELPERS:
        port.old.image.pinned(ROOT / "apps/computer_use" / name, record["source_sha256"][name])
    return record


def memory_queue(registration, continuation):
    state = port.old.guard_queue(port.read(registration["queue"]), registration["identity"],
                                 owned.digest(continuation), "shared_quota_declaration_sha256")
    declaration = port.read(continuation)
    rows = state.get("trials", [])
    if (state.get("protocol_sha256") != registration["protocol_sha256"]
            or state.get("shared_quota_wrapper_sha256") != owned.digest(port.__file__)
            or state.get("qualified_runner_python") != str(port.PYTHON)
            or state.get("common_gpu_quota_ratio") != 1.0
            or state.get("previous_profile_prefix_separate") != 1
            or state.get("planned_trials") != 8 or not 5 <= len(rows) <= 8
            or rows[:5] != declaration.get("completed_prefix")
            or [row.get("id") for row in rows[:5]] != port.PREFIX_IDS):
        raise SafetyHalt("Memory queue changed the exact five-result matched prefix or producer")
    return state


def patched(source):
    source = prior.patched(source)
    replacements = (
        ('if mr.get("kind") == "actual_shared_quota_memory_identity" and identity == mr["identity"]:',
         'if mr.get("kind") == "actual_shared_quota_port_continuation_identity" and identity == mr["identity"]:'),
        ('"actual_shared_quota_successor_identity"', '"actual_shared_quota_port_successor_identity"'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise SafetyHalt("Frozen port-successor routing anchor changed")
        source = source.replace(before, after, 1)
    before = '"qualified_shared_quota_frozen_successors"'
    if source.count(before) != 2:
        raise SafetyHalt("Frozen port-successor declaration anchors changed")
    return source.replace(before, '"qualified_shared_quota_port_frozen_successors"')


def build_namespace():
    if (owned.digest(prior.__file__) != PRIOR_SHA or owned.digest(port.__file__) != PORT_SHA
            or owned.digest(prior.original.__file__) != prior.SOURCE_SHA):
        raise SafetyHalt("Frozen port continuation or successor source changed")
    module = types.ModuleType("qualified_shared_quota_port_successors_runtime")
    module.__file__ = __file__
    exec(compile(patched(Path(prior.original.__file__).read_text()), "frozen-successors-port-routing", "exec"), module.__dict__)
    # The full eight-case protocol validator is unchanged. Supply its alias in
    # this namespace rather than modifying the active port producer module.
    quota = types.SimpleNamespace(**vars(port))
    quota.validate_protocol = port.previous.validate_protocol
    module.__dict__.update(MODULE=MODULE, HELPERS=HELPERS, quota=quota,
                           qualification=qualification, memory_declaration=port.validate,
                           memory_registration=memory_registration, memory_queue=memory_queue)
    return module


RUNTIME = build_namespace()


if __name__ == "__main__":
    RUNTIME.main()
