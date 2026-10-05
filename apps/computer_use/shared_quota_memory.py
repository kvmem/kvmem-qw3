"""Fresh eight-trial memory cohort with explicit equal sharing quotas.

Old OFF result and ON startup failure stay separate. Original model executable,
windows, task materials, runner, grades and each trial budget stay pinned.
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

import kvmem_capacity_continuation as previous
import shared_quota_engine as engine

old, owned = previous.old, previous.owned
ROOT, PYTHON, SafetyHalt = previous.ROOT, previous.PYTHON, previous.SafetyHalt
HELPERS = ("shared_quota_memory.py", "tests/test_shared_quota_memory.py")
MODULE = "shared_quota_memory"
STAGES = ("memory", "sdk", "image", "thinking", "feedback")


def read(path):
    return json.loads(Path(path).read_text())


def patch_validator(source):
    before = 'r["id"] != f\'{r["task"]}-kvmem-{r["kvmem"]}-r{r["repetition"]}\''
    if source.count(before) != 1:
        raise SafetyHalt("Original memory inventory validator changed")
    return source.replace(before, 'r["id"] != f\'quota1-{r["task"]}-kvmem-{r["kvmem"]}-r{r["repetition"]}\'', 1)


def validate_protocol(path):
    namespace = dict(vars(owned))
    exec(compile(patch_validator(inspect.getsource(owned.validate_protocol)), "quota-cohort-inventory", "exec"), namespace)
    p = namespace["validate_protocol"](Path(path))
    if p.get("shared_quota_profile_revision") != "common-ratio1-fresh-eight":
        raise SafetyHalt("Use the explicit new sharing cohort")
    profile = engine.qualify_profile(read(p["shared_quota_profile"]))
    if (owned.digest(p["shared_quota_profile"]) != p["shared_quota_profile_sha256"]
            or p["engine_identities"] != profile["engine_identities"]):
        raise SafetyHalt("Common sharing profile changed")
    return p


def patch_loop(source):
    replacements = (
        ('initial = register_engine(initial_runtime, p["engine_identities"]["off"])', 'initial = None'),
        ('"initial_registered_engine": initial, "planned_trials": 8}',
         '"initial_registered_engine": initial, "planned_trials": 8, "shared_quota_declaration_sha256": declaration_sha, '
         '"shared_quota_wrapper_sha256": wrapper_sha, "qualified_runner_python": str(PYTHON), '
         '"previous_profile_prefix_separate": 1, "common_gpu_quota_ratio": 1.0}'),
        ('state["last_engine_retirement"] = retire_registered(registered)',
         'state["last_engine_retirement"] = retire_registered(registered) if registered is not None else {"retired": True, "no_initial_engine_owned": True}'),
        ('previous = json.loads((parent_dir / "queue.json").read_text())',
         'previous = old.continued_parent(canonical, owned.digest(declaration["original_followup"]), Path(declaration["original_followup"]))'),
        ('str(ROOT / "apps/computer_use/run_engine.py"),\n',
         'str(ROOT / "apps/computer_use/shared_quota_engine.py"), "--profile", declaration["profile"],\n'),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise SafetyHalt("Original memory loop anchor changed")
        source = source.replace(before, after, 1)
    return source


def qualification(path):
    q = read(path)
    if (q.get("kind") != "shared_quota_memory_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 8 or q.get("skipped", 0) != 0
            or q.get("model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require exact new-cohort CPU qualification")
    for name in HELPERS:
        old.image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def retirement_evidence(d, declaration_path, receipt):
    if (receipt.get("kind") != "exact_idle_quota_transition_retirement" or receipt.get("passed") is not True
            or receipt.get("declaration_sha256") != owned.digest(declaration_path)
            or receipt.get("old_memory_completed_prefix") != 1
            or receipt.get("model_calls") != 0 or receipt.get("device_inputs") != 0):
        raise SafetyHalt("Require actual idle predecessor retirement")
    rows = receipt.get("retired", [])
    if len(rows) != 5 or {r.get("stage") for r in rows} != set(STAGES):
        raise SafetyHalt("Need all five exact predecessors")
    prefix = read(d["previous_continuation"])["completed_prefix"]
    for row in rows:
        path = d["previous_registrations"][row["stage"]]
        reg = read(path)
        if (row.get("registration") != path or row.get("registration_sha256") != owned.digest(path)
                or row.get("identity") != reg["identity"] or row.get("queue") != reg["queue"]
                or row.get("queue_sha256") != owned.digest(reg["queue"])
                or owned.process_identity(reg["identity"]["pid"]) is not None):
            raise SafetyHalt("Retirement must name the exact pinned predecessor")
        state = read(reg["queue"])
        expected = prefix if row["stage"] == "memory" else []
        if (state.get("pid") != reg["identity"]["pid"] or state.get("status") != "halted-no-further-dispatch"
                or state.get("trials") != expected or state.get("child") or state.get("current_child")
                or state.get("registered_engine")):
            raise SafetyHalt("Retired queue must retain only the original prefix and no live work")
        if state.get("engine_launcher_pid"):
            if row["stage"] != "memory" or receipt.get("retirement_method") != "natural-capacity-startup-failure":
                raise SafetyHalt("Historical engine requires exact natural failure evidence")
            capacity_failure_evidence(d, state, receipt["startup_failure"])


def capacity_failure_evidence(d, state, evidence):
    original = Path(read(d["previous_continuation"])["continued_protocol"])
    runtime = original.parent / "engines/035-kvmem-on-r1"
    if (state.get("error") != "SafetyHalt: Fresh engine exited; no model trial dispatched"
            or state.get("current_trial") != "035-kvmem-on-r1"
            or state.get("current_engine_runtime") != str(runtime)
            or state.get("last_gpu_snapshot", {}).get("free_mib", 0) < 93000
            or state["last_gpu_snapshot"].get("compute_pids") != []
            or evidence.get("runtime") != str(runtime)
            or evidence.get("manifest_sha256") != owned.digest(runtime / "manifest.json")
            or evidence.get("log_sha256") != owned.digest(runtime / "engine.log")
            or previous.CAPACITY not in (runtime / "engine.log").read_text()
            or (original.parent / "attempts/035-kvmem-on-r1").exists()):
        raise SafetyHalt("Natural failure must match the exact no-task startup at the original gate")
    manifest = read(runtime / "manifest.json")
    if (manifest.get("pid") != state["engine_launcher_pid"]
            or manifest.get("owns_process_only") is not True or manifest.get("min_free_mib") != 93000
            or manifest.get("existing_compute_pids") != []
            or owned.engine_identity(runtime) != read(original)["engine_identities"]["on"]
            or owned.process_identity(manifest["pid"]) is not None):
        raise SafetyHalt("Failed original engine ownership or actual retirement unproven")
    owned.assert_no_services(original.parent / "attempts")
    owned.assert_vm_retired(original.parent / "attempts/035-kvmem-on-r1")


def startup_evidence(d, receipt):
    profile = engine.qualify_profile(read(d["profile"]))
    if (receipt.get("kind") != "actual_common_quota_startup_qualification" or receipt.get("passed") is not True
            or receipt.get("profile_sha256") != owned.digest(d["profile"])
            or receipt.get("owned_engine_actual_retired") is not True or receipt.get("task_trials_dispatched") != 0
            or receipt.get("device_inputs") != 0 or receipt.get("health_model_calls") != 1
            or receipt.get("health_response") != "READY" or receipt.get("advertised_context_tokens") != 524288
            or receipt.get("wrapper_sha256") != owned.digest(engine.__file__)
            or receipt.get("engine_profile") != profile["engine_identities"]["on"]):
        raise SafetyHalt("Require physical profile startup, health response and exact provenance")
    identity = receipt["engine_identity"]
    if (identity.get("uid") != os.getuid() or identity.get("cwd") != str(ROOT)
            or identity.get("command") != profile["engine_identities"]["on"]["command"]
            or owned.process_identity(identity["pid"]) is not None):
        raise SafetyHalt("Qualification engine ownership or retirement unproven")
    runtime = Path(receipt["engine_runtime"])
    manifest = read(runtime / "manifest.json")
    if (owned.digest(runtime / "manifest.json") != receipt["manifest_sha256"]
            or owned.engine_identity(runtime) != profile["engine_identities"]["on"]
            or manifest.get("pid") != identity["pid"] or manifest.get("owns_process_only") is not True
            or manifest.get("shared_quota_profile_sha256") != receipt["profile_sha256"]
            or manifest.get("shared_quota_wrapper_sha256") != receipt["wrapper_sha256"]
            or receipt.get("startup_gpu", {}).get("free_mib", 0) < 56000):
        raise SafetyHalt("Startup profile, manifest or resource admission changed")


def validate(path, *, require_retirement=True, require_startup=True):
    d = read(path)
    if (d.get("kind") != "fresh_shared_quota_kvmem_cohort" or d.get("python") != str(PYTHON)
            or d.get("old_result_reused_as_new_pair") is not False or d.get("old_protocol_overwritten") is not False
            or d.get("planned_trials") != 8 or set(d.get("previous_registrations", {})) != set(STAGES)):
        raise SafetyHalt("Require the explicit fresh-cohort declaration")
    for item in d["files"]:
        old.image.pinned(item["path"], item["sha256"])
    qualification(d["qualification"])
    previous_decl, canonical = previous.validate(Path(d["previous_continuation"]))
    if previous_decl["original_followup"] != d["original_followup"]:
        raise SafetyHalt("Original completed parent changed")
    validate_protocol(Path(d["continued_protocol"]))
    if require_retirement:
        retirement_evidence(d, path, read(d["retirement_receipt"]))
    if require_startup:
        startup_evidence(d, read(d["startup_qualification"]))
    return d, canonical


def prepare(args):
    earlier, canonical = previous.validate(args.previous_continuation.resolve())
    qualification(args.qualification)
    profile = engine.qualify_profile(read(args.profile))
    original = Path(earlier["continued_protocol"])
    if Path(profile["original_protocol"]).resolve() != original.resolve():
        raise SafetyHalt("Use the exact capacity-wait producer's profile")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    p = copy.deepcopy(read(original))
    p.update(declared_at=time.time(), status="fresh-quota-cohort-not-dispatched", shared_quota_profile_revision="common-ratio1-fresh-eight",
             shared_quota_profile=str(args.profile.resolve()), shared_quota_profile_sha256=owned.digest(args.profile),
             engine_identities=profile["engine_identities"], engine_gpu_memory_cap_gib=None,
             previous_profile_first_off_result_separate=True)
    for row in p["trials"]:
        row["id"] = "quota1-" + row["id"]
    for mode in ("off", "on"):
        a = copy.deepcopy(read(original.parent / mode / "acceptance.json"))
        a.update(engine_identity=p["engine_identities"][mode],
                 trials={r["id"]: {"task": r["task"], "arm": r["arm"]} for r in p["trials"] if r["kvmem"] == mode},
                 scope="Fresh shared-quota development memory comparison; original windows, runner, materials and budgets unchanged.")
        target = out / mode / "acceptance.json"
        target.parent.mkdir()
        owned.save(target, a)
        p["acceptance_sha256"][mode] = owned.digest(target)
    protocol = out / "protocol.json"
    owned.save(protocol, p)
    files = [args.profile.resolve(), args.previous_continuation.resolve(), args.qualification.resolve(),
             protocol, out / "off/acceptance.json", out / "on/acceptance.json"]
    for stage in STAGES:
        registration = Path(args.previous_registrations[stage]).resolve()
        files.append(registration)
    d = dict(kind="fresh_shared_quota_kvmem_cohort", declared_at=time.time(), profile=str(args.profile.resolve()),
             original_followup=earlier["original_followup"], previous_continuation=str(args.previous_continuation.resolve()),
             previous_registrations={k: str(Path(v).resolve()) for k, v in args.previous_registrations.items()},
             continued_protocol=str(protocol), state_dir=str(out / "queue-runtime-01"), python=str(PYTHON),
             qualification=str(args.qualification.resolve()), planned_trials=8, old_result_reused_as_new_pair=False,
             old_protocol_overwritten=False, retirement_receipt=str(out / "predecessor-retirement.json"),
             startup_qualification=str(out / "startup-qualification.json"),
             files=[dict(path=str(p), sha256=owned.digest(p)) for p in files],
             scope="New eight-case paired cohort, no old prefix reuse. Only common GPU quota1 and unique host attempt/VM IDs differ. "
                   "Previous frozen profile and all failures remain. Engine/model/windows, GUI permissions and original per-trial budgets stay fixed.")
    path = out / "declaration.json"
    owned.save(path, d)
    validate(path, require_retirement=False, require_startup=False)
    return path


def run(args):
    d, canonical = validate(args.declaration)
    if sys.executable != d["python"]:
        raise SafetyHalt("Preserve the original qualified task interpreter")
    args.protocol, args.state_dir, args.min_free_mib = Path(d["continued_protocol"]), Path(d["state_dir"]), 56000
    namespace = dict(vars(owned))
    namespace.update(declaration=d, declaration_sha=owned.digest(args.declaration), wrapper_sha=owned.digest(__file__),
                     old=old, owned=owned, canonical=canonical, PYTHON=PYTHON,
                     terminal_parent=lambda p, q: old.terminal_parent(p, q, Path(d["original_followup"])))
    def checked(path):
        validate(args.declaration)
        return validate_protocol(path)
    namespace["validate_protocol"] = checked
    exec(compile(patch_loop(inspect.getsource(owned.run)), "qualified-shared-quota-memory-loop", "exec"), namespace)
    return namespace["run"](args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    for name in ("declaration", "profile", "previous-continuation", "qualification", "output", "previous-registrations"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=86400)
    args = parser.parse_args()
    if not (60 <= args.wait_seconds <= 172800 and 60 <= args.resource_wait_seconds <= 86400):
        parser.error("Keep the original bounded waiting policy")
    if args.prepare:
        if any(getattr(args, name) is None for name in ("profile", "previous_continuation", "qualification", "output", "previous_registrations")):
            parser.error("Provide exact source/profile/registration inventory and qualification")
        args.previous_registrations = read(args.previous_registrations)
        print(json.dumps({"declaration": str(prepare(args)), "model_calls": 0, "device_inputs": 0}))
        return
    if args.declaration is None:
        parser.error("Provide the immutable new-cohort declaration")
    args.declaration = args.declaration.resolve()
    def interrupted(*args):
        raise SafetyHalt("Shared-quota controller interrupted; evidence and live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    result = run(args)
    print(json.dumps({"status": result["status"], "trials": len(result["trials"])}))


if __name__ == "__main__":
    main()
