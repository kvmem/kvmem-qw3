"""Resume the frozen 8/10/12 experiments after two retired unknown VM inputs.

Only predecessor routing and scoped admission change. Frozen trial loops,
private packages, model profiles, tasks, evaluators and failures stay intact.
"""
import argparse
import copy
import inspect
import json
from pathlib import Path
import signal
from unittest.mock import patch

import execution_retest_queue_v2 as sdk
import osworld_retired_trial_successors as old
import retired_051_continuation as parent
import visual_context_retest_queue as image

owned, base, read, save, SafetyHalt = image.owned, image.base, image.read, image.save, image.SafetyHalt
IMAGE_VALIDATE = image.validate
MEMORY_VALIDATE = owned.validate_protocol
MODULE = "retired_051_successors"


def guard_queue(state, identity, declaration_sha, key):
    """A state file alone never proves that an unfinished producer is live."""
    if (state.get("pid") != identity["pid"] or state.get(key) != declaration_sha
            or not isinstance(state.get("status"), str) or state["status"].startswith("halted")):
        raise SafetyHalt("Predecessor halted or its declaration/process identity changed")
    actual = owned.process_identity(identity["pid"])
    if actual is not None and not owned.same_process(identity, actual):
        raise SafetyHalt("Predecessor PID reused")
    if state["status"] != "completed" and (actual is None or actual.get("state") == "Z"):
        raise SafetyHalt("Unfinished predecessor is no longer live")
    return state


def former_waiter(state, identity):
    if (state.get("pid") != identity["pid"] or not str(state.get("status", "")).startswith("halted")
            or state.get("trials") != [] or state.get("child") or state.get("engine_launcher_pid")):
        raise SafetyHalt("Former successor was not halted before all dispatch")
    if owned.process_identity(identity["pid"]) is not None:
        raise SafetyHalt("Former successor PID remains or was reused")


def qualify(path):
    q = read(path)
    if (q.get("kind") != "retired_051_frozen_successors_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 20 or q.get("model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require completed successor CPU admission qualification")
    for name in ("retired_051_successors.py", "tests/test_retired_051_successors.py"):
        image.pinned(Path(__file__).parent / name, q["source_sha256"][name])


def validate_followup(path):
    d = read(path)
    if (d.get("kind") != "retired_051_frozen_successors" or d.get("original_profiles_changed") is not False
            or d.get("original_unknowns_replayed") is not False or d.get("original_unknowns_reclassified") is not False
            or d.get("unknown_receipts") != parent.CASES):
        raise SafetyHalt("Use the exact declared two-VM successor scope")
    for f in d["files"]: image.pinned(f["path"], f["sha256"])
    qualify(d["cpu_qualification"])
    # Nested frozen validators must keep using their original static admission.
    with patch.object(image, "validate", IMAGE_VALIDATE):
        case, protocol, initial = parent.validate(Path(d["continuation_declaration"]), live=False)
        ip = IMAGE_VALIDATE(Path(d["image_protocol"]))
    if (d["continued_osworld_queue"] != case["continued_queue"]
            or d["memory_protocol"] != ip["memory_protocol"] or d["sdk_protocol"] != ip["sdk_protocol"]):
        raise SafetyHalt("Original protocols or continued parent changed")
    registration = read(d["parent_registration"])
    if (registration.get("kind") != "actual_retired_051_continuation_identity"
            or registration["identity"] != d["continuation_controller_identity"]
            or registration["declaration_sha256"] != owned.digest(d["continuation_declaration"])
            or not old.command_path(registration["identity"], "--declaration", d["continuation_declaration"])
            or "retired_051_continuation" not in registration["identity"]["command"]):
        raise SafetyHalt("Actual continued parent registration is unproven")
    continued_parent(d, owned.digest(path), path)
    for f in d["former_waiters"]: former_waiter(read(f["queue"]), f["identity"])
    return d, case, protocol, initial


def continued_parent(d, followup_sha, path):
    image.pinned(path, followup_sha)
    q = guard_queue(read(d["continued_osworld_queue"]), d["continuation_controller_identity"],
                    owned.digest(d["continuation_declaration"]), "continuation_declaration_sha256")
    if q.get("historical_unknowns_preserved") != 2:
        raise SafetyHalt("Both historical unknowns must remain preserved")
    return q


def terminal_rows(protocol, state, initial, results):
    rows = state.get("trials", [])
    expected = parent.original.ordered_trials(protocol)
    if (state.get("status") != "completed" or len(rows) != 40
            or [r["id"] for r in rows] != [r["id"] for r in expected]
            or rows[:19] != initial or len(initial) != 19 or state.get("historical_unknowns_preserved") != 2):
        raise SafetyHalt("Need all forty exact results and the unchanged nineteen-row prefix")
    for row in rows:
        result = results.get(row["id"])
        if (not isinstance(result, dict) or result != row.get("result") or not result.get("finished_at")
                or type(result.get("uncertain_actions")) is not int
                or result["uncertain_actions"] != (1 if row["id"] in parent.CASES else 0)
                or result.get("owned_task_stop_unconfirmed")):
            raise SafetyHalt("Changed terminal evidence or a new unresolved input")
    return copy.deepcopy(rows)


def terminal_parent(memory_protocol, state, followup):
    d, case, protocol, initial = validate_followup(followup)
    campaign = Path(case["campaign"])
    if (Path(memory_protocol["parent_candidate_protocol"]).resolve() != campaign / "protocol.json"
            or state.get("protocol_sha256") != memory_protocol["parent_candidate_protocol_sha256"]
            or state.get("pid") != d["continuation_controller_identity"]["pid"]):
        raise SafetyHalt("Original parent protocol or actual controller changed")
    results = {r["id"]: read(campaign / "attempts" / r["id"] / "result.json") for r in state.get("trials", [])}
    rows = terminal_rows(protocol, state, initial, results)
    owned.assert_no_services(campaign)
    for row in rows: owned.assert_vm_retired(campaign / "attempts" / row["id"])
    return [dict(id=r["id"], result_sha256=owned.digest(campaign / "attempts" / r["id"] / "result.json"),
                 retired_unknown_preserved=r["id"] in parent.CASES) for r in rows]


def registration(path, stage, followup, d):
    r = read(path); identity = r["identity"]; command = identity.get("command", [])
    if (r.get("kind") != "actual_retired_051_successor_identity" or r.get("stage") != stage
            or r.get("followup_sha256") != owned.digest(followup)
            or r.get("wrapper_sha256") != owned.digest(Path(__file__))
            or r.get("protocol_sha256") != owned.digest(d[stage + "_protocol"])
            or r.get("queue") != str(Path(d[stage + "_state_dir"]) / "queue.json")
            or MODULE not in command or command.count("--stage") != 1
            or command[command.index("--stage") + 1:command.index("--stage") + 2] != [stage]
            or not old.command_path(identity, "--followup", followup)):
        raise SafetyHalt("Actual successor registration is not the declared stage")
    return r


def prepare(output, declaration, parent_registration, qualification):
    output, declaration, parent_registration, qualification = map(lambda p: Path(p).resolve(),
        (output, declaration, parent_registration, qualification))
    if output.exists(): raise SafetyHalt("New successor declaration already exists")
    qualify(qualification)
    case, _, _ = parent.validate(declaration, live=False)
    ip = IMAGE_VALIDATE(Path(case["image_protocol"]))
    actual = read(parent_registration); identity = actual["identity"]
    q = guard_queue(read(case["continued_queue"]), identity, owned.digest(declaration), "continuation_declaration_sha256")
    if q["status"] == "completed": raise SafetyHalt("Preparation must register the actual live continued producer")
    states = {"memory": Path(ip["memory_queue"]).parent.with_name("queue-runtime-03"),
              "sdk": Path(ip["sdk_queue"]).parent.with_name("queue-runtime-03")}
    # The image queue is identified by its retired controller, not file ordering.
    retired = case["retired_controllers"][1:]
    if len(retired) != 3: raise SafetyHalt("Need all three retired zero-trial successors")
    states["image"] = Path(retired[2]["queue"]).parent.with_name("queue-runtime-02")
    for r in retired: former_waiter(read(r["queue"]), r["identity"])
    for state in states.values():
        if state.exists(): raise SafetyHalt("Fresh successor state directory already exists")
    paths = [declaration, parent_registration, qualification, Path(__file__),
             Path(__file__).parent / "tests/test_retired_051_successors.py", Path(case["image_protocol"]),
             Path(ip["memory_protocol"]), Path(ip["sdk_protocol"])] + [Path(r["queue"]) for r in retired]
    d = dict(kind="retired_051_frozen_successors", original_profiles_changed=False,
             original_unknowns_replayed=False, original_unknowns_reclassified=False, unknown_receipts=parent.CASES,
             continuation_declaration=str(declaration), parent_registration=str(parent_registration),
             continued_osworld_queue=case["continued_queue"], continuation_controller_identity=identity,
             memory_protocol=ip["memory_protocol"], sdk_protocol=ip["sdk_protocol"], image_protocol=case["image_protocol"],
             cpu_qualification=str(qualification), former_waiters=retired,
             files=[dict(path=str(p), sha256=owned.digest(p)) for p in paths],
             scope="Only continued-parent routing and two exact retired unknowns; frozen 8/10/12 trials unchanged.",
             **{stage + "_state_dir": str(path) for stage, path in states.items()})
    save(output, d); validate_followup(output)
    return d


def patched_memory_source(source):
    before = 'previous = json.loads((parent_dir / "queue.json").read_text())'
    if source.count(before) != 1: raise SafetyHalt("Frozen memory parent-reader shape changed")
    source = source.replace(before, 'previous = continued_parent(d, followup_sha, args.followup)')
    before = '"initial_registered_engine": initial, "planned_trials": 8}'
    if source.count(before) != 1: raise SafetyHalt("Frozen memory state shape changed")
    return source.replace(before, '"initial_registered_engine": initial, "planned_trials": 8, '
        '"continued_successor_wrapper_sha256": wrapper_sha, "continued_successor_declaration_sha256": followup_sha}')


def run_memory(args, d):
    followup_sha = owned.digest(args.followup)
    args.protocol, args.state_dir = Path(d["memory_protocol"]), Path(d["memory_state_dir"])
    def validate(path):
        image.pinned(args.followup, followup_sha); validate_followup(args.followup)
        return MEMORY_VALIDATE(path)
    namespace = dict(vars(owned)); namespace.update(validate_protocol=validate, d=d, followup_sha=followup_sha,
        wrapper_sha=owned.digest(Path(__file__)), continued_parent=continued_parent,
        terminal_parent=lambda p, q: terminal_parent(p, q, args.followup))
    exec(compile(patched_memory_source(inspect.getsource(owned.run)), "two-retired-vm-memory-continuation", "exec"), namespace)
    return namespace["run"](args)


def run_sdk(args, d):
    followup_sha, reg_sha = owned.digest(args.followup), owned.digest(args.memory_controller)
    mr = registration(args.memory_controller, "memory", args.followup, d)
    args.protocol, args.state_dir = Path(d["sdk_protocol"]), Path(d["sdk_state_dir"])
    def validate(path):
        image.pinned(args.followup, followup_sha); image.pinned(args.memory_controller, reg_sha)
        validate_followup(args.followup)
        p, kp = sdk.validate(path); effective = dict(p)
        effective.update(parent_osworld_queue=d["continued_osworld_queue"],
                         parent_kvmem_queue=str(Path(d["memory_state_dir"]) / "queue.json"),
                         predecessor_processes=dict(osworld=d["continuation_controller_identity"], memory=mr["identity"]))
        return effective, kp
    original_ready = base.predecessors_ready
    def ready(p, kp):
        continued_parent(d, followup_sha, args.followup)
        guard_queue(read(Path(d["memory_state_dir"]) / "queue.json"), mr["identity"],
                    followup_sha, "continued_successor_declaration_sha256")
        return original_ready(p, kp)
    class Controller(sdk.Controller):
        def __init__(self, *values):
            super().__init__(*values)
            self.state.update(continued_successor_wrapper_sha256=owned.digest(Path(__file__)),
                              continued_successor_declaration_sha256=followup_sha, memory_registration_sha256=reg_sha)
    sdk.PROTOCOL, _ = validate(args.protocol); sdk.DECLARATION = read(sdk.PROTOCOL["quarantine_declaration"])
    with patch.object(base, "validate", validate), patch.object(base, "predecessors_ready", ready), \
            patch.object(base, "Controller", Controller), patch.object(base, "ledger_terminal", sdk.scoped_ledger), \
            patch.object(owned, "terminal_parent", lambda p, q: terminal_parent(p, q, args.followup)):
        return base.run(args)


def image_ready(p, d, followup, memory_reg, sdk_reg):
    sha = owned.digest(followup)
    osq = continued_parent(d, sha, followup)
    kvq = guard_queue(read(p["memory_queue"]), memory_reg["identity"], sha, "continued_successor_declaration_sha256")
    sq = guard_queue(read(p["sdk_queue"]), sdk_reg["identity"], sha, "continued_successor_declaration_sha256")
    if any(q["status"] != "completed" for q in (osq, kvq, sq)): return None
    if not all(base.process_retired(r["identity"]) for r in (read(d["parent_registration"]), memory_reg, sdk_reg)): return None
    sp, kp = sdk.validate(Path(p["sdk_protocol"]))
    os_receipts = terminal_parent(kp, osq, followup)
    kv_receipts = base.exact_rows(kp["trials"], kvq, Path(p["memory_protocol"]).parent, p["memory_protocol_sha256"])
    if [r["id"] for r in kvq["trials"]] != [r["id"] for r in kp["trials"]]: raise SafetyHalt("Memory trial order changed")
    for row in kvq["trials"]:
        out = Path(p["memory_protocol"]).parent / "attempts" / row["id"]
        owned.assert_no_services(out); owned.assert_vm_retired(out)
    sdk_receipts = image.mixed_terminal_rows(sp["trials"], sq, Path(p["sdk_protocol"]).parent, p["sdk_protocol_sha256"])
    if sq.get("termination_confirmed") is not True or sq.get("new_services_preserved") is not True:
        raise SafetyHalt("SDK final service handoff is not confirmed")
    base.require_identity(sq["model_client"]); owned.assert_apps_idle(p["parent_app"])
    image.pinned(sq["registry"], sq["registry_sha256"])
    for gateway in sq["new_gateways"]:
        base.require_identity(gateway["process_identity"])
        command = gateway["command"]; base.ledger_terminal(Path(command[command.index("--runtime") + 1]))
    engine = owned.register_engine(Path(sq["engine_runtime"]), p["engine_off_identity"])
    if not owned.same_process(sq["registered_engine"], engine): raise SafetyHalt("Final SDK engine identity changed")
    return dict(osworld=os_receipts, kvmem=kv_receipts, sdk=sdk_receipts, registered_engine=engine,
                model_client=sq["model_client"], registry=sq["registry"], registry_sha256=sq["registry_sha256"],
                new_gateways=sq["new_gateways"])


def run_image(args, d):
    sha = owned.digest(args.followup); mr_sha, sr_sha = owned.digest(args.memory_controller), owned.digest(args.sdk_controller)
    mr = registration(args.memory_controller, "memory", args.followup, d)
    sr = registration(args.sdk_controller, "sdk", args.followup, d)
    if not old.command_path(sr["identity"], "--memory-controller", args.memory_controller):
        raise SafetyHalt("SDK producer has a different memory registration")
    args.protocol, args.state_dir = Path(d["image_protocol"]), Path(d["image_state_dir"])
    def validate(path):
        image.pinned(args.followup, sha); image.pinned(args.memory_controller, mr_sha); image.pinned(args.sdk_controller, sr_sha)
        validate_followup(args.followup)
        with patch.object(image, "validate", IMAGE_VALIDATE): p = IMAGE_VALIDATE(path)
        effective = dict(p); effective.update(osworld_queue=d["continued_osworld_queue"],
            memory_queue=str(Path(d["memory_state_dir"]) / "queue.json"), sdk_queue=str(Path(d["sdk_state_dir"]) / "queue.json"))
        return effective
    class Controller(image.Controller):
        def __init__(self, *values):
            super().__init__(*values)
            self.state.update(continued_successor_wrapper_sha256=owned.digest(Path(__file__)),
                continued_successor_declaration_sha256=sha, memory_registration_sha256=mr_sha, sdk_registration_sha256=sr_sha)
    with patch.object(image, "validate", validate), patch.object(image, "Controller", Controller), \
            patch.object(image, "predecessors_ready", lambda p: image_ready(p, d, args.followup, mr, sr)):
        return image.run(args)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--followup", type=Path, required=True); p.add_argument("--stage", choices=("memory", "sdk", "image"))
    p.add_argument("--prepare", action="store_true"); p.add_argument("--declaration", type=Path)
    p.add_argument("--parent-registration", type=Path); p.add_argument("--qualification", type=Path)
    p.add_argument("--memory-controller", type=Path); p.add_argument("--sdk-controller", type=Path)
    p.add_argument("--wait-seconds", type=int, default=172800); p.add_argument("--resource-wait-seconds", type=int, default=21600)
    p.add_argument("--min-free-mib", type=int, default=56000); args = p.parse_args()
    args.followup = args.followup.resolve()
    if args.prepare:
        if None in (args.declaration, args.parent_registration, args.qualification): p.error("Provide qualified continuation evidence")
        d = prepare(args.followup, args.declaration, args.parent_registration, args.qualification)
        print(json.dumps(dict(kind=d["kind"], model_calls=0, device_inputs=0))); return
    if (args.stage is None or args.min_free_mib < 56000 or not 60 <= args.wait_seconds <= 172800
            or not 60 <= args.resource_wait_seconds <= 86400): p.error("Keep the frozen bounded resource profile")
    if args.stage in ("sdk", "image") and args.memory_controller is None: p.error("Register the actual memory producer")
    if args.stage == "image" and args.sdk_controller is None: p.error("Register the actual SDK producer")
    for name in ("memory_controller", "sdk_controller"):
        if getattr(args, name) is not None: setattr(args, name, getattr(args, name).resolve())
    d, _, _, _ = validate_followup(args.followup)
    def interrupted(*_): raise SafetyHalt("Continued successor interrupted; owned children and evidence preserved")
    signal.signal(signal.SIGTERM, interrupted)
    {"memory": run_memory, "sdk": run_sdk, "image": run_image}[args.stage](args, d)


if __name__ == "__main__": main()
