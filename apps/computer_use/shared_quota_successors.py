"""Route the frozen10/12/4/8 successors to the fresh equal-quota memory cohort.

Original packages, task bodies and budgets stay unchanged. Both the declared
memory protocol digest and its new eight IDs follow the actual producer.
"""
import inspect
import os
from pathlib import Path
import types

import shared_quota_memory as quota
import runner_runtime_successors as original

SOURCE_SHA = "97976ec85d5f4261d49a84e830e42c44ff314f7f37559c8331a670856648a1d6"
HELPERS = ("shared_quota_successors.py", "tests/test_shared_quota_successors.py", "tests/test_runner_runtime_successors.py")
owned, SafetyHalt, ROOT = quota.owned, quota.SafetyHalt, quota.ROOT


def qualification(path):
    q = quota.read(path)
    if (q.get("kind") != "shared_quota_successors_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 20 or q.get("skipped", 0) != 0
            or q.get("model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require exact fresh-producer routing qualification")
    for name in HELPERS: quota.old.image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def memory_registration(path, continuation, runtime):
    r = quota.read(path)
    command = [runtime["python"], "-m", quota.MODULE, "--declaration", str(continuation)]
    if (r.get("kind") != "actual_shared_quota_memory_identity" or runtime.get("kind") != "fresh_shared_quota_kvmem_cohort"
            or r.get("declaration") != str(continuation) or r.get("declaration_sha256") != owned.digest(continuation)
            or r.get("protocol") != runtime["continued_protocol"] or r.get("protocol_sha256") != owned.digest(runtime["continued_protocol"])
            or r.get("queue") != str(Path(runtime["state_dir"]) / "queue.json")
            or r["identity"].get("command") != command or r["identity"].get("cwd") != str(ROOT)
            or r["identity"].get("uid") != os.getuid() or r.get("old_business_prefix_reused") is not False):
        raise SafetyHalt("Actual fresh eight-case producer registration changed")
    for name in quota.HELPERS: quota.old.image.pinned(ROOT / "apps/computer_use" / name, r["source_sha256"][name])
    return r


def memory_queue(registration, continuation):
    q = quota.old.guard_queue(quota.read(registration["queue"]), registration["identity"], owned.digest(continuation),
                             "shared_quota_declaration_sha256")
    if (q.get("protocol_sha256") != registration["protocol_sha256"]
            or q.get("shared_quota_wrapper_sha256") != owned.digest(quota.__file__)
            or q.get("qualified_runner_python") != str(quota.PYTHON)
            or q.get("common_gpu_quota_ratio") != 1.0 or q.get("previous_profile_prefix_separate") != 1):
        raise SafetyHalt("Memory evidence is not the fresh common-quota producer")
    return q


def patched(source):
    replacements = (
        ('parent_kvmem_protocol=d["memory_protocol"],\n',
         'parent_kvmem_protocol=d["memory_protocol"], parent_kvmem_protocol_sha256=owned.digest(d["memory_protocol"]),\n'),
        ('return sdk_protocol(p, d, canonical, mr), kp',
         'return sdk_protocol(p, d, canonical, mr), quota.validate_protocol(Path(d["memory_protocol"]))'),
        ('sdk_queue=sr["queue"], memory_protocol=d["memory_protocol"])',
         'sdk_queue=sr["queue"], memory_protocol=d["memory_protocol"], memory_protocol_sha256=owned.digest(d["memory_protocol"]))'),
        ('if mr.get("kind") == "actual_kvmem_capacity_continuation_identity" and identity == mr["identity"]:',
         'if mr.get("kind") == "actual_shared_quota_memory_identity" and identity == mr["identity"]:'),
        ('return original_guard(state, identity, owned.digest(d["memory_continuation"]), "capacity_continuation_sha256")',
         'return original_guard(state, identity, owned.digest(d["memory_continuation"]), "shared_quota_declaration_sha256")'),
        ('with patch.object(old, "guard_queue", guard):\n',
         'with patch.object(old, "guard_queue", guard), patch.object(sdk, "validate", '
         'lambda path: (frozen_protocol("sdk", path)[0], quota.validate_protocol(Path(d["memory_protocol"])))):\n'),
        ('scope="Original 10/12/4/8 frozen trials; actual resumed memory receipts; new controller states. Only SDK trial child Python changes to the original qualified OSWorld venv. No model, package, task, budget, grade or input gate changes."',
         'scope="Original 10/12/4/8 frozen successor trials and profiles; route only to actual fresh common-quota memory evidence and unique IDs. Original SDK/image/thinking/feedback packages, model profiles, tasks, budgets, grades and input gates unchanged."'),
    )
    for before, after in replacements:
        if source.count(before) != 1: raise SafetyHalt("Frozen successor routing anchor changed")
        source = source.replace(before, after, 1)
    before = '"qualified_runner_runtime_frozen_successors"'
    if source.count(before) != 2: raise SafetyHalt("Frozen declaration anchors changed")
    source = source.replace(before, '"qualified_shared_quota_frozen_successors"')
    before = '"actual_runner_runtime_successor_identity"'
    if source.count(before) != 1: raise SafetyHalt("Frozen registration anchor changed")
    return source.replace(before, '"actual_shared_quota_successor_identity"')


def build_namespace():
    if owned.digest(original.__file__) != SOURCE_SHA: raise SafetyHalt("Frozen successor source changed")
    module = types.ModuleType("qualified_shared_quota_successors_runtime")
    module.__file__ = __file__
    exec(compile(patched(Path(original.__file__).read_text()), "frozen-successors-new-memory-routing", "exec"), module.__dict__)
    module.__dict__.update(MODULE="shared_quota_successors", HELPERS=HELPERS, quota=quota, qualification=qualification,
                           memory_declaration=quota.validate, memory_registration=memory_registration, memory_queue=memory_queue)
    return module


RUNTIME = build_namespace()


if __name__ == "__main__": RUNTIME.main()
