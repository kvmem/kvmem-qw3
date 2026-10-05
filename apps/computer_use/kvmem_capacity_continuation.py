"""Resume seven frozen memory trials after an exact pre-task capacity failure.

The first completed OFF trial remains unchanged and is referenced in the new
evidence tree. The failed ON startup remains intact. Only fresh paths and a
higher shared-GPU admission threshold change; no model parameter changes.
"""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time

import runner_runtime_continuation as previous

old, owned = previous.old, previous.owned
read, save, SafetyHalt, ROOT, PYTHON = previous.read, previous.save, previous.SafetyHalt, previous.ROOT, previous.PYTHON
HELPERS = ("kvmem_capacity_continuation.py", "tests/test_kvmem_capacity_continuation.py")
ERROR = "SafetyHalt: Fresh engine exited; no model trial dispatched"
CAPACITY = "Need 1536 blocks (1024 for max(--kvmem-budget, --kvmem-prefill-budget) + 512 for --kvmem-gen-budget) but only 1304 fit under the current VRAM ceiling."
MIN_FREE_MIB = 93000


def failure_scope(state, result, log, attempted_paths):
    rows = state.get("trials", [])
    if (state.get("status") != "halted-no-further-dispatch" or state.get("error") != ERROR
            or state.get("current_trial") != "035-kvmem-on-r1" or len(rows) != 1
            or rows[0].get("id") != "035-kvmem-off-r1" or rows[0].get("kvmem") != "off"
            or rows[0].get("result") != result or not result.get("finished_at")
            or type(result.get("uncertain_actions")) is not int or result["uncertain_actions"] != 0
            or result.get("owned_task_stop_unconfirmed") or CAPACITY not in log or attempted_paths):
        raise SafetyHalt("Only the exact failed ON startup with one unchanged terminal OFF prefix is admitted")
    return copy.deepcopy(rows)


def patch_source(source):
    replacements = (
        ('initial = register_engine(initial_runtime, p["engine_identities"]["off"])', 'initial = None'),
        ('"status": "waiting-for-parent-forty", "trials": [],', '"status": "waiting-for-parent-forty", "trials": copy.deepcopy(completed_prefix),'),
        ('"initial_registered_engine": initial, "planned_trials": 8}',
         '"initial_registered_engine": initial, "planned_trials": 8, "capacity_continuation_sha256": declaration_sha, '
         '"capacity_continuation_wrapper_sha256": wrapper_sha, "qualified_runner_python": declaration["python"], '
         '"completed_prefix_preserved": 1, "shared_admission_min_free_mib": declaration["min_free_mib"]}'),
        ('for row in p["trials"]:', 'for row in p["trials"][len(completed_prefix):]:'),
        ('state["last_engine_retirement"] = retire_registered(registered)',
         'state["last_engine_retirement"] = retire_registered(registered) if registered is not None else {"retired": True, "no_prior_engine_owned": True}'),
        ('previous = json.loads((parent_dir / "queue.json").read_text())',
         'previous = old.continued_parent(canonical, owned.digest(declaration["original_followup"]), Path(declaration["original_followup"]))'),
    )
    for before, after in replacements:
        if source.count(before) != 1: raise SafetyHalt("Frozen capacity continuation anchor changed")
        source = source.replace(before, after, 1)
    return source


def qualification(path):
    q = read(path)
    if (q.get("kind") != "kvmem_capacity_continuation_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 12 or q.get("model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require completed capacity continuation admission qualification")
    for name in HELPERS: old.image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def validate(path):
    d = read(path)
    if (d.get("kind") != "exact_kvmem_capacity_failure_continuation" or d.get("python") != str(PYTHON)
            or d.get("min_free_mib") != MIN_FREE_MIB or d.get("original_profiles_changed") is not False
            or d.get("original_failure_overwritten") is not False or d.get("completed_prefix_replayed") is not False):
        raise SafetyHalt("Use the exact frozen-profile capacity continuation")
    for item in d["files"]: old.image.pinned(item["path"], item["sha256"])
    qualification(d["qualification"]); old.image.pinned(PYTHON, d["python_binary_sha256"])
    runtime, canonical = previous.validate(Path(d["previous_continuation"]))
    if d["original_followup"] != runtime["original_followup"]: raise SafetyHalt("Original completed parent changed")
    reg = read(d["previous_registration"]); state = read(reg["queue"])
    if (reg.get("kind") != "actual_runner_runtime_memory_continuation_identity"
            or reg.get("declaration_sha256") != owned.digest(d["previous_continuation"])
            or state.get("pid") != reg["identity"]["pid"]
            or owned.process_identity(reg["identity"]["pid"]) is not None
            or owned.process_identity(state["engine_launcher_pid"]) is not None):
        raise SafetyHalt("Failed memory controller or engine remains live")
    for item in d["retired_successors"]:
        r = read(item["registration"]); q = read(r["queue"])
        if (q.get("pid") != r["identity"]["pid"] or q.get("status") != "halted-no-further-dispatch"
                or q.get("trials") != [] or q.get("current_child")
                or owned.process_identity(r["identity"]["pid"]) is not None):
            raise SafetyHalt("An old successor dispatched work or remains live")
    original = Path(runtime["continued_protocol"])
    first = original.parent / "attempts/035-kvmem-off-r1"
    on = original.parent / "attempts/035-kvmem-on-r1"
    paths = [str(on)] if on.exists() else []
    prefix = failure_scope(state, read(first / "result.json"), Path(d["failed_engine_log"]).read_text(), paths)
    if (owned.digest(first / "result.json") != prefix[0]["result_sha256"]
            or owned.process_identity(state["child"]["pid"]) is not None
            or prefix != d["completed_prefix"]):
        raise SafetyHalt("Completed prefix changed or its runner remains live")
    owned.assert_no_services(original.parent / "attempts"); owned.assert_vm_retired(first); owned.assert_vm_retired(on)
    clone = Path(d["continued_protocol"])
    if owned.digest(original) != owned.digest(clone): raise SafetyHalt("Original eight-case protocol changed")
    owned.validate_protocol(clone)
    link = clone.parent / "attempts/035-kvmem-off-r1"
    if not link.is_symlink() or link.resolve() != first.resolve(): raise SafetyHalt("Prefix reference must retain the original evidence directory")
    return d, canonical


def prepare(args):
    runtime_path, reg_path = args.previous_continuation.resolve(), args.previous_registration.resolve()
    runtime, canonical = previous.validate(runtime_path)
    reg = read(reg_path); state = read(reg["queue"]); original = Path(runtime["continued_protocol"])
    first = original.parent / "attempts/035-kvmem-off-r1"; on = original.parent / "attempts/035-kvmem-on-r1"
    log = original.parent / "engines/035-kvmem-on-r1/engine.log"
    prefix = failure_scope(state, read(first / "result.json"), log.read_text(), [str(on)] if on.exists() else [])
    qualification(args.qualification)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    clone = out / "protocol.json"; clone.write_bytes(original.read_bytes())
    for policy in ("off", "on"):
        target = out / policy; target.mkdir(); (target / "acceptance.json").write_bytes((original.parent / policy / "acceptance.json").read_bytes())
    (out / "attempts").mkdir(); (out / "attempts/035-kvmem-off-r1").symlink_to(first, target_is_directory=True)
    retired = [dict(registration=str(p.resolve())) for p in (args.sdk_registration, args.image_registration)]
    files = [runtime_path, reg_path, Path(reg["queue"]), args.qualification.resolve(), log,
             original.parent / "engines/035-kvmem-on-r1/manifest.json", first / "result.json", first / "vm.json", first / "manifest.json",
             clone, out / "off/acceptance.json", out / "on/acceptance.json"]
    for item in retired:
        p = Path(item["registration"]); files += [p, Path(read(p)["queue"])]
    d = dict(kind="exact_kvmem_capacity_failure_continuation", declared_at=time.time(), previous_continuation=str(runtime_path),
             previous_registration=str(reg_path), original_followup=runtime["original_followup"], retired_successors=retired,
             continued_protocol=str(clone), state_dir=str(out / "queue-runtime-01"), python=str(PYTHON),
             python_binary_sha256=owned.digest(PYTHON), failed_engine_log=str(log), completed_prefix=prefix,
             min_free_mib=MIN_FREE_MIB, qualification=str(args.qualification.resolve()), original_profiles_changed=False,
             original_failure_overwritten=False, completed_prefix_replayed=False,
             files=[dict(path=str(p), sha256=owned.digest(p)) for p in files],
             scope="Keep the exact original eight cases and one OFF result; resume seven in fresh paths. Wait for >=93000MiB free before each unchanged fixed engine. No ratio/budget/context/binary/model change; other GPU processes remain untouched. Resource load differs between trials and must be reported.")
    path = out / "continuation.json"; save(path, d); validate(path)
    return path


def run(args):
    path = args.declaration.resolve(); d, canonical = validate(path)
    if sys.executable != d["python"]: raise SafetyHalt("Keep the original qualified runner interpreter")
    args.protocol, args.state_dir, args.min_free_mib = Path(d["continued_protocol"]), Path(d["state_dir"]), d["min_free_mib"]
    namespace = dict(vars(owned)); namespace.update(copy=copy, declaration=d, declaration_sha=owned.digest(path),
        wrapper_sha=owned.digest(Path(__file__)), completed_prefix=d["completed_prefix"], old=old, owned=owned, canonical=canonical,
        terminal_parent=lambda p, q: old.terminal_parent(p, q, Path(d["original_followup"])))
    def checked(protocol):
        validate(path); return owned.validate_protocol(protocol)
    namespace["validate_protocol"] = checked
    exec(compile(patch_source(inspect.getsource(owned.run)), "frozen-capacity-continuation", "exec"), namespace)
    return namespace["run"](args)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--prepare", action="store_true")
    for name in ("output", "previous-continuation", "previous-registration", "sdk-registration", "image-registration", "qualification", "declaration"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--wait-seconds", type=int, default=172800); p.add_argument("--resource-wait-seconds", type=int, default=86400)
    args = p.parse_args()
    if not 60 <= args.wait_seconds <= 172800 or not 60 <= args.resource_wait_seconds <= 86400: p.error("Keep bounded admission waits")
    if args.prepare:
        if any(getattr(args, name) is None for name in ("output", "previous_continuation", "previous_registration", "sdk_registration", "image_registration", "qualification")):
            p.error("Provide exact retired failure, prefix and qualification")
        print(json.dumps(dict(declaration=str(prepare(args)), model_calls=0, device_inputs=0))); return
    if args.declaration is None: p.error("Provide the immutable continuation")
    def interrupted(*_): raise SafetyHalt("Capacity continuation interrupted; live work and original evidence preserved")
    signal.signal(signal.SIGTERM, interrupted)
    result = run(args); print(json.dumps(dict(status=result["status"], trials=len(result["trials"]))))


if __name__ == "__main__": main()
