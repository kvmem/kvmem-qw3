"""Serialize four read-only saved-input replays after all twelve image trials."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import time

import visual_context_retest_queue as image
from thinking_budget_replay import replay, replay_body, Stream

owned, base, ROOT, RUNTIME = image.owned, image.base, image.ROOT, image.RUNTIME
read, save, SafetyHalt = image.read, image.save, image.SafetyHalt
SOURCE = RUNTIME / "linux/candidate49b-matched-16k-20261004/attempts/035-legacy-r1/app/inference/6ecce7c76e284947b8d07aeb1e96526d"
HELPERS = ("thinking_budget_replay.py", "thinking_budget_replay_queue.py", "tests/test_thinking_budget_replay.py")


def trials():
    return [dict(id=f"035-thinking-{budget}-r{rep}", thinking_budget=budget, repetition=rep, seconds=600)
            for rep, budgets in ((1, (0, 8192)), (2, (8192, 0))) for budget in budgets]


def source_input(trace):
    data = image.archive.read_request(trace)
    body = json.loads(data)
    replay_body(body, 0)
    if {t["function"]["name"] for t in body["tools"]} != {"mcp__computer__act", "mcp__computer__observe"}:
        raise SafetyHalt("Use the original saved legacy tool catalogue")
    state = Stream()
    for line in (trace / "backend.jsonl").read_bytes().splitlines(): state.feed(line)
    if (state.finish_reason != "length" or not state.reasoning_chars or state.content_chars or state.tool_deltas):
        raise SafetyHalt("Require the exact original reasoning-only 16K failure")
    metrics = read(trace / "metrics.json")
    if metrics["usage"] != {"input_tokens": 107482, "output_tokens": 16384, "total_tokens": 123866}:
        raise SafetyHalt("Original source usage changed")
    return data


def qualify_cpu(path):
    q = read(path)
    if (q.get("kind") != "readonly_thinking_replay_cpu_http_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 15 or q.get("qw3_model_calls") != 0 or q.get("native_device_inputs") != 0):
        raise SafetyHalt("Require completed CPU and scripted HTTP qualification")
    for name in HELPERS: image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def prepare(parent_protocol, registration, directory, qualification):
    p = image.validate(parent_protocol)
    registered = read(registration)
    if (registered.get("kind") != "actual_visual_context_retest_waiter_identity"
            or registered.get("protocol_sha256") != owned.digest(parent_protocol)
            or not image.successors.command_path(registered["identity"], "--protocol", parent_protocol)
            or "visual_context_retest_queue" not in registered["identity"]["command"]):
        raise SafetyHalt("Actual image-policy predecessor controller is unproven")
    base.require_identity(registered["identity"])
    qualify_cpu(qualification)
    data = source_input(SOURCE)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    source = directory / "source-request.json"
    source.write_bytes(data); source.chmod(0o600)
    p = dict(kind="declared_readonly_saved_input_thinking_budget_replay", status="declared-waiting-for-image-twelve",
             declared_at=time.time(), parent_protocol=str(parent_protocol), parent_protocol_sha256=owned.digest(parent_protocol),
             parent_registration=str(registration), parent_registration_sha256=owned.digest(registration),
             parent_queue=registered["queue"], parent_controller=registered["identity"],
             qualification=str(qualification), qualification_sha256=owned.digest(qualification),
             source_trace=str(SOURCE), source_request=str(source), source_request_sha256=image.archive.digest(data),
             source_metrics_sha256=owned.digest(SOURCE / "metrics.json"), source_backend_sha256=owned.digest(SOURCE / "backend.jsonl"),
             original_result_sha256=owned.digest(SOURCE.parents[2] / "result.json"),
             helper_sha256={n: owned.digest(ROOT / "apps/computer_use" / n) for n in HELPERS},
             model_profile=image.MODEL, engine_off_identity=p["engine_off_identity"], parent_app=p["parent_app"],
             trials=trials(), model_concurrency=1, fresh_engine_per_replay=True,
             tool_calls_executed=0, advisor_enabled=False, business_success_claimed=False)
    path = directory / "protocol.json"; save(path, p); validate(path)
    return p


def validate(path):
    p = read(path)
    if (p.get("kind") != "declared_readonly_saved_input_thinking_budget_replay"
            or p.get("status") != "declared-waiting-for-image-twelve" or p.get("trials") != trials()
            or p.get("model_profile") != image.MODEL or p.get("tool_calls_executed") != 0
            or p.get("model_concurrency") != 1 or p.get("advisor_enabled") is not False
            or p.get("fresh_engine_per_replay") is not True or p.get("business_success_claimed") is not False
            or Path(p["source_trace"]) != SOURCE):
        raise SafetyHalt("Read-only thinking comparison declaration changed")
    for stem in ("parent_protocol", "parent_registration", "qualification", "source_request"):
        image.pinned(p[stem], p[stem + "_sha256"])
    qualify_cpu(p["qualification"])
    for name, sha in p["helper_sha256"].items(): image.pinned(ROOT / "apps/computer_use" / name, sha)
    parent = image.validate(Path(p["parent_protocol"]))
    if p["engine_off_identity"] != parent["engine_off_identity"] or p["parent_app"] != parent["parent_app"]:
        raise SafetyHalt("Original shared model profile or user application changed")
    registered = read(p["parent_registration"])
    if (registered["identity"] != p["parent_controller"] or registered["queue"] != p["parent_queue"]
            or registered["protocol_sha256"] != p["parent_protocol_sha256"]):
        raise SafetyHalt("Image predecessor registration changed")
    image.pinned(SOURCE / "metrics.json", p["source_metrics_sha256"])
    image.pinned(SOURCE / "backend.jsonl", p["source_backend_sha256"])
    image.pinned(SOURCE.parents[2] / "result.json", p["original_result_sha256"])
    if source_input(SOURCE) != Path(p["source_request"]).read_bytes():
        raise SafetyHalt("Replay input differs from the preserved original backend request")
    return p


def predecessor_ready(p):
    q = read(p["parent_queue"])
    if (q.get("pid") != p["parent_controller"]["pid"]
            or q.get("protocol_sha256") != p["parent_protocol_sha256"]
            or str(q.get("status", "")).startswith("halted")):
        raise SafetyHalt("Image predecessor halted or changed identity")
    if q.get("status") != "completed":
        base.require_identity(p["parent_controller"])
        return None
    if not base.process_retired(p["parent_controller"]): return None
    parent = image.validate(Path(p["parent_protocol"]))
    if ([r["id"] for r in q.get("trials", [])] != [r["id"] for r in parent["trials"]]
            or q.get("termination_confirmed") is not True or q.get("parent_services_preserved") is not True
            or q.get("final_engine_policy") != "off"):
        raise SafetyHalt("All twelve image-policy trials must be independently terminal")
    for spec, row in zip(parent["trials"], q["trials"]):
        out = Path(p["parent_protocol"]).parent / "attempts" / row["id"]
        result_path = out / ("result.json" if spec["kind"] == "original-osworld" else "integration-result.json")
        result = read(result_path); base.terminal_trial(spec, result)
        if result != row["result"] or owned.digest(result_path) != row.get("result_sha256"):
            raise SafetyHalt("Image-policy predecessor durable grade changed")
        image.pinned(out / "visual-context-evidence.json", row["evidence_sha256"])
        evidence = read(out / "visual-context-evidence.json")
        if evidence["policy"] != spec["policy"] or evidence["result_sha256"] != row["result_sha256"]:
            raise SafetyHalt("Image-policy predecessor evidence identity changed")
        owned.assert_no_services(out)
        if spec["kind"] == "original-osworld": owned.assert_vm_retired(out)
    owned.assert_no_services(Path(p["parent_protocol"]).parent / "clients")
    services = q["parent_services"]
    base.require_identity(services["model_client"])
    owned.assert_apps_idle(p["parent_app"])
    image.pinned(services["registry"], services["registry_sha256"])
    for gw in services["new_gateways"]:
        base.require_identity(gw["process_identity"])
        command = gw["command"]; base.ledger_terminal(Path(command[command.index("--runtime") + 1]))
    # SDK's original OFF engine was deliberately replaced by each image trial.
    # Admit the last image controller's actual engine, not that retired PID.
    engine = owned.register_engine(Path(q["engine_runtime"]), p["engine_off_identity"])
    if not owned.same_process(q["registered_engine"], engine):
        raise SafetyHalt("Last image-policy engine identity changed")
    return dict(services, registered_engine=engine)


def run(args):
    p = validate(args.protocol)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    state = dict(kind="serial_readonly_thinking_budget_replay", pid=os.getpid(), started_at=time.time(),
                 protocol_sha256=owned.digest(args.protocol), status="waiting-for-image-twelve", trials=[],
                 planned_replays=4, qw3_replays_completed=0, tool_calls_executed=0, device_inputs=0,
                 business_success_claimed=False, model_concurrency=1)
    controller = image.Controller(args, p, state)
    controller.checkpoint()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            previous = predecessor_ready(p)
            if previous is not None: break
            if time.monotonic() >= deadline: raise SafetyHalt("Image predecessor wait deadline reached; services preserved")
            time.sleep(3)
        controller.checkpoint(parent_services=previous)
        for row in p["trials"]:
            validate(args.protocol)
            image.pinned(args.protocol, state["protocol_sha256"])
            controller.fresh_engine(row)
            with owned.model_lock():
                controller.parent_idle()
                owned.register_engine(Path(state["engine_runtime"]), p["engine_off_identity"])
                controller.checkpoint(status="replaying-original-input", current_replay=row["id"])
                out = args.protocol.parent / "attempts" / row["id"]
                result = asyncio.run(replay(Path(p["source_request"]).read_bytes(), row["thinking_budget"], out, seconds=row["seconds"]))
                if (result["termination_confirmed"] is not True or result["tool_calls_executed"] != 0
                        or result["business_success_claimed"] is not False):
                    raise SafetyHalt("Read-only model request retirement or output scope changed")
                state["trials"].append(dict(row, result=result, result_sha256=owned.digest(out / "result.json")))
                controller.parent_idle()
                controller.checkpoint(qw3_replays_completed=len(state["trials"]))
        controller.checkpoint(status="completed", finished_at=time.time(), termination_confirmed=True,
                              parent_services_preserved=True, final_engine_policy="off")
    except BaseException as exc:
        controller.checkpoint(status="halted-no-further-dispatch", finished_at=time.time(),
                              error=type(exc).__name__ + ": " + str(exc), live_children_and_evidence_preserved=True)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--parent-protocol", type=Path)
    parser.add_argument("--parent-registration", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if args.operation == "prepare":
        if any(getattr(args, k) is None for k in ("parent_protocol", "parent_registration", "campaign", "qualification")):
            parser.error("Preparation requires the image predecessor and scripted HTTP qualification")
        p = prepare(args.parent_protocol.resolve(), args.parent_registration.resolve(), args.campaign.resolve(), args.qualification.resolve())
        print(json.dumps(dict(status=p["status"], planned_replays=4, model_calls=0))); return
    if (args.protocol is None or args.state_dir is None or args.min_free_mib < 56000
            or not 60 <= args.wait_seconds <= 172800 or not 60 <= args.resource_wait_seconds <= 86400):
        parser.error("Specify a new state directory and bounded shared-resource waits")
    args.protocol, args.state_dir = args.protocol.resolve(), args.state_dir.resolve()
    def interrupted(*_): raise SafetyHalt("Read-only replay waiter interrupted; live evidence preserved")
    signal.signal(signal.SIGTERM, interrupted)
    result = run(args)
    print(json.dumps(dict(status=result["status"], completed=len(result["trials"]))))


if __name__ == "__main__": main()
