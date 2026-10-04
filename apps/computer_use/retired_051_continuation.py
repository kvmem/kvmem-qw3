"""Continue untouched trials after both exact failed disposable VMs retired.

The original two queues and all nineteen results remain immutable. This adapter
reuses the frozen continuation runner, changing only admission and checkpoint
metadata for the second independently retired VM. Unknowns remain unknown.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sqlite3
import time
from unittest.mock import patch

import osworld_queue_continuation as original
import visual_context_retest_queue as image

owned, read, save, SafetyHalt = image.owned, image.read, image.save, image.SafetyHalt
ORIGINAL_VALIDATE = original.validate
CASES = {
    "022-legacy-r1": "q3_b4b87fef8706f9283e1d8290703162c93da84a6989815808235b7b61147b0913",
    "051-verified-r1": "q3_bf3143ccc6194159f2cc76fa37f38b7ca32471918fe8d40e307a17f9bd7df154"}
LAST_VM = "7341de56a5afea31eff8cddefc93e5ea2d334d5b60de01157472d9e62ca6475a"


def halted_rows(protocol, state, initial, results, controller_pid):
    expected = original.ordered_trials(protocol)
    rows = state.get("trials", [])
    if (state.get("status") != "halted-no-further-model-dispatch" or state.get("pid") != controller_pid
            or len(rows) != 19 or state.get("current_trial") != "051-verified-r1"
            or [r["id"] for r in rows] != [r["id"] for r in expected[:19]]
            or rows[:len(initial)] != initial):
        raise SafetyHalt("Use the exact nineteen-row halted checkpoint; no old trial replay")
    for row in rows:
        result = results.get(row["id"])
        if (result != row.get("result") or not isinstance(result, dict) or not result.get("finished_at")
                or type(result.get("uncertain_actions")) is not int
                or result["uncertain_actions"] != (1 if row["id"] in CASES else 0)
                or result.get("owned_task_stop_unconfirmed") or result.get("accepted") is not False):
            raise SafetyHalt("A result changed or an undeclared unknown remains")
    return copy.deepcopy(rows)


def unknown(attempt, receipt):
    path = attempt / "gateway/receipts.sqlite"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute('SELECT id,status,request,result FROM actions WHERE status NOT IN ("completed","rejected")').fetchall()
    if len(rows) != 1 or rows[0][0] != receipt or rows[0][1] != "uncertain" or rows[0][3] is not None:
        raise SafetyHalt("The original unknown row was changed or another unresolved input exists")
    agent = read(attempt / "final_agent_state.json")
    unresolved = agent.get("uncertain_actions", [])
    if (agent.get("desktop_paused") is not True or agent.get("status") != "paused"
            or len(unresolved) != 1 or unresolved[0].get("receipt_id") != receipt
            or unresolved[0].get("status") != "uncertain"):
        raise SafetyHalt("Original paused client does not retain its exact unknown")
    request = json.loads(rows[0][2])
    return dict(receipt_id=receipt, request_sha256=image.archive.digest(rows[0][2].encode()),
                desktop_id=request["desktop_id"], vm=read(attempt / "vm.json"),
                local_action_id=unresolved[0]["action_id"], request=request)


def _validate(path, *, live=True):
    d = read(path)
    if (d.get("kind") != "exact_two_retired_vm_frozen_trial_continuation"
            or d.get("original_unknowns_changed") is not False or d.get("old_trials_replayed") is not False
            or d.get("remaining_trials") != 21 or d.get("unknown_receipts") != CASES):
        raise SafetyHalt("Use the scoped two-VM continuation declaration")
    for f in d["files"]: image.pinned(f["path"], f["sha256"])
    image.pinned(Path(__file__), d["recipe_sha256"])
    prior, protocol, initial = original.validate(Path(d["prior_declaration"]), live=False)
    campaign = Path(d["campaign"])
    if (campaign != Path(prior["campaign"]) or Path(d["parent_queue"]) != campaign / "continued_queue.json"
            or Path(d["continued_queue"]) != campaign / "continued_queue_02.json"):
        raise SafetyHalt("Continuation output or original campaign changed")
    checkpoint = read(d["parent_queue"])
    if checkpoint["protocol_sha256"] != owned.digest(campaign / "protocol.json"):
        raise SafetyHalt("Original task/model/budget protocol changed")
    results = {r["id"]: read(campaign / "attempts" / r["id"] / "result.json") for r in checkpoint["trials"]}
    rows = halted_rows(protocol, checkpoint, initial, results, d["previous_controller"]["pid"])
    for trial, receipt in CASES.items():
        evidence = unknown(campaign / "attempts" / trial, receipt)
        if evidence != d["retired_case_evidence"][trial]:
            raise SafetyHalt("Original unknown/VM evidence changed")
        if trial == "051-verified-r1" and evidence["vm"]["container_id"] != LAST_VM:
            raise SafetyHalt("Wrong failed disposable GIMP VM")
    image.validate(Path(d["image_protocol"]))
    for registration in d["retired_controllers"]:
        original.retired_controller(registration["identity"])
        state = read(registration["queue"])
        if (state.get("pid") != registration["identity"]["pid"] or not str(state.get("status", "")).startswith("halted")):
            raise SafetyHalt("A previous controller is not the declared halted instance")
    if live:
        owned.assert_no_services(campaign)
        for row in rows: owned.assert_vm_retired(campaign / "attempts" / row["id"])
        owned.assert_apps_idle(protocol["pilot_app"])
        from osworld_linux.campaign import engine_identity
        owned.register_engine(Path(protocol["engine_runtime"]), engine_identity(Path(protocol["engine_runtime"])))
    return d, protocol, rows


def validate(path, *, live=True):
    # Nested source validators still use the original one-VM declaration.
    # Restore its hook temporarily rather than widening its admission rules.
    with patch.object(original, "validate", ORIGINAL_VALIDATE):
        return _validate(path, live=live)


def prepare(image_protocol, image_registration, output, cpu_path):
    ip = image.validate(image_protocol)
    _, protocol, initial = original.validate(Path(read(ip["followup"])["continuation_declaration"]), live=False)
    followup = read(ip["followup"])
    parent = Path(ip["osworld_queue"])
    campaign = parent.parent
    state = read(parent)
    controller = followup["continuation_controller_identity"]
    results = {r["id"]: read(campaign / "attempts" / r["id"] / "result.json") for r in state["trials"]}
    rows = halted_rows(protocol, state, initial, results, controller["pid"])
    if output.exists() or (campaign / "continued_queue_02.json").exists():
        raise SafetyHalt("New declaration or queue already exists; no overwrite")
    registrations = [{"identity": controller, "queue": str(parent)},
                     {"identity": read(ip["memory_registration"])["identity"], "queue": ip["memory_queue"]},
                     {"identity": ip["sdk_controller_identity"], "queue": ip["sdk_queue"]},
                     {"identity": read(image_registration)["identity"], "queue": read(image_registration)["queue"]}]
    for r in registrations: original.retired_controller(r["identity"])
    cpu = read(cpu_path)
    if (cpu.get("kind") != "retired_051_strict_continuation_cpu_qualification" or cpu.get("passed") is not True
            or cpu.get("tests_run", 0) < 10 or cpu.get("model_calls") != 0 or cpu.get("device_inputs") != 0):
        raise SafetyHalt("Require the completed scoped continuation CPU qualification")
    image.pinned(Path(__file__), cpu["source_sha256"]["retired_051_continuation.py"])
    image.pinned(Path(__file__).parent / "tests/test_retired_051_continuation.py", cpu["source_sha256"]["tests/test_retired_051_continuation.py"])
    paths = [parent, campaign / "protocol.json", Path(followup["continuation_declaration"]), image_protocol,
             image_registration, cpu_path, Path(__file__).parent / "tests/test_retired_051_continuation.py"]
    paths += [Path(r["queue"]) for r in registrations]
    paths += [campaign / "attempts" / r["id"] / "result.json" for r in rows]
    for trial in CASES:
        paths += [campaign / "attempts" / trial / n for n in ("vm.json", "final_agent_state.json", "gateway/receipts.sqlite")]
    unique = list(dict.fromkeys(paths))
    d = dict(kind="exact_two_retired_vm_frozen_trial_continuation", at=time.time(), campaign=str(campaign),
             parent_queue=str(parent), prior_declaration=followup["continuation_declaration"],
             previous_controller=controller, retired_controllers=registrations, image_protocol=str(image_protocol),
             unknown_receipts=CASES, retired_case_evidence={c: unknown(campaign / "attempts" / c, r) for c, r in CASES.items()},
             continued_queue=str(campaign / "continued_queue_02.json"), remaining_trials=21,
             files=[dict(path=str(p), sha256=owned.digest(p)) for p in unique], recipe_sha256=owned.digest(Path(__file__)),
             original_unknowns_changed=False, old_trials_replayed=False,
             scope="Keep both unknown receipts and all nineteen grades; only fresh remaining VM trials. No recovery or delivery claim.")
    save(output, d); validate(output)
    return d


def run(path):
    validate(path)
    original_save = original.save
    def checkpoint(target, state):
        if Path(target) == Path(read(path)["continued_queue"]):
            state.update(historical_unknowns_preserved=2, original_nineteen_row_queue_sha256=owned.digest(read(path)["parent_queue"]),
                         retired_051_wrapper_sha256=owned.digest(Path(__file__)), old_unknowns_replayed=False,
                         old_unknowns_reclassified=False)
        return original_save(target, state)
    # This process only; the frozen runner and every model/task helper remain
    # byte-identical. Its loop begins after the retained nineteen exact rows.
    with patch.object(original, "validate", validate), patch.object(original, "save", checkpoint):
        return original.run(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declaration", type=Path, required=True)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--image-protocol", type=Path)
    p.add_argument("--image-registration", type=Path)
    p.add_argument("--qualification", type=Path)
    args = p.parse_args(); path = args.declaration.resolve()
    if args.prepare:
        if args.qualification is None: p.error("Specify the completed CPU boundary qualification")
        d = prepare(args.image_protocol.resolve(), args.image_registration.resolve(), path, args.qualification.resolve())
        print(json.dumps(dict(remaining_trials=d["remaining_trials"], model_calls=0))); return
    def interrupted(*_): raise SafetyHalt("Continuation interrupted; owned live children preserved")
    signal.signal(signal.SIGTERM, interrupted)
    d = run(path)
    print(json.dumps(dict(status=d["status"], trials=len(d["trials"]))))


if __name__ == "__main__": main()
