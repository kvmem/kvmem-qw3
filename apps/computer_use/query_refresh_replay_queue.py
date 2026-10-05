"""Run four exact-input refresh qualifications after the frozen desktop chain."""
import argparse
import asyncio
import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import time

import bounded_continuation_retest_queue as bounded
import query_refresh_engine as engine
from query_refresh_replay import pair_identity, sequence
from kvmem_retrieval_diagnostic import analyze
from osworld_linux import archive


q, image, base, owned = bounded.q, bounded.image, bounded.base, bounded.owned
ROOT, RUNTIME, read, save, SafetyHalt = bounded.ROOT, bounded.RUNTIME, bounded.read, bounded.save, bounded.SafetyHalt
BOUNDED_SHA = "d6c83a2abae42355c77ac667f9a7ea7877fe4b0b7a2b69de9afbc1a540c2eb02"
PLAN_SHA = "222749d3fae8a4dd3d89c69b1dfc2164e906eb3af3ce4c6403d53f4d63a05a79"
HELPERS = ("query_refresh_replay_queue.py", "tests/test_query_refresh_replay_queue.py",
           "query_refresh_replay.py", "tests/test_query_refresh_replay.py",
           "query_refresh_engine.py", "tests/test_query_refresh_engine.py",
           "kvmem_retrieval_diagnostic.py", "thinking_budget_replay.py")


def trials():
    return [dict(id=f"refresh-{mode}-r{repeat}", mode=mode, repetition=repeat, request_seconds=600)
            for repeat, modes in ((1, ("off", "both")), (2, ("both", "off"))) for mode in modes]


def registration(path):
    image.pinned(bounded.__file__, BOUNDED_SHA)
    reg = read(path)
    if (reg.get("kind") != "actual_bounded_continuation_retest_identity"
            or reg.get("controller_sha256") != BOUNDED_SHA or reg.get("planned_trials") != 8):
        raise SafetyHalt("Require the same actual eight-case bounded predecessor")
    image.pinned(reg["protocol"], reg["protocol_sha256"])
    state = q.old.guard_queue(read(reg["queue"]), reg["identity"], reg["protocol_sha256"], "protocol_sha256")
    if (state.get("kind") != "serial_matched_bounded_continuation_retest"
            or state.get("controller_sha256") != BOUNDED_SHA):
        raise SafetyHalt("Bounded predecessor source or kind changed")
    return reg, state


def full_prior_handoff(parent, terminal_state):
    # Reuse every original earlier-stage receipt/VM/service check. Only the
    # final engine has moved from feedback to the now-terminal bounded stage.
    image.pinned(bounded.__file__, BOUNDED_SHA)
    source = inspect.getsource(bounded.terminal_predecessor)
    before = '    engine = owned.register_engine(Path(state["engine_runtime"]), p["engine_off_identity"])\n'
    after = '    if not base.process_retired(state["registered_engine"]):\n'
    after += '        raise SafetyHalt("Earlier feedback engine did not retire before bounded trials")\n'
    after += '    engine = owned.register_engine(Path(terminal_state["engine_runtime"]), p["engine_off_identity"])\n'
    compare = '    if not owned.same_process(state["registered_engine"], engine):\n'
    if source.count(before) != 1 or source.count(compare) != 1:
        raise SafetyHalt("Original full-chain terminal handoff anchors changed")
    source = source.replace(before, after, 1).replace(compare,
        '    if not owned.same_process(terminal_state["registered_engine"], engine):\n', 1)
    namespace = dict(vars(bounded), terminal_state=terminal_state)
    exec(compile(source, "qualified-post-bounded-full-chain-handoff", "exec"), namespace)
    services = namespace["terminal_predecessor"](parent, read(parent["parent_queue"]))
    if services is None:
        raise SafetyHalt("Earlier completed chain is not independently terminal")
    return services


def predecessor_ready(p):
    reg, state = registration(p["parent_registration"])
    if reg["identity"] != p["parent_controller"] or reg["queue"] != p["parent_queue"]:
        raise SafetyHalt("Actual bounded producer identity changed")
    if str(state.get("status", "")).startswith("halted"):
        raise SafetyHalt("Bounded predecessor halted; no query refresh dispatch")
    if state.get("status") != "completed":
        base.require_identity(reg["identity"])
        return None
    if not base.process_retired(reg["identity"]):
        return None
    parent = bounded.validate(Path(reg["protocol"]))
    if (state.get("termination_confirmed") is not True or state.get("parent_services_preserved") is not True
            or state.get("final_engine_policy") != "off"
            or [row["id"] for row in state.get("trials", [])] != [row["id"] for row in parent["trials"]]):
        raise SafetyHalt("All original bounded trials and services must be terminal")
    receipts = []
    for spec, row in zip(parent["trials"], state["trials"]):
        out = Path(reg["protocol"]).parent / "attempts" / row["id"]
        result = read(out / "result.json")
        base.terminal_trial(dict(spec, kind="original-osworld"), result)
        if result != row["result"] or owned.digest(out / "result.json") != row["result_sha256"]:
            raise SafetyHalt("A terminal bounded business receipt changed")
        image.pinned(out / "bounded-continuation-evidence.json", row["evidence_sha256"])
        evidence = read(out / "bounded-continuation-evidence.json")
        if (evidence.get("original_result_sha256") != row["result_sha256"] or evidence.get("arm") != spec["arm"]
                or evidence.get("task_sha256") != owned.digest(out / "app/task.json")):
            raise SafetyHalt("Bounded model evidence changed")
        owned.assert_no_services(out)
        owned.assert_vm_retired(out)
        receipts.append(dict(id=spec["id"], result_sha256=row["result_sha256"]))
    owned.assert_no_services(Path(reg["protocol"]).parent / "clients")
    services = full_prior_handoff(parent, state)
    services["query_refresh_predecessor_receipts"] = receipts
    return services


def qualify(path, kind, minimum):
    result = read(path)
    if (result.get("kind") != kind or result.get("passed") is not True
            or result.get("tests_run", 0) < minimum or result.get("skipped") != 0
            or result.get("device_inputs") != 0
            or result.get("model_calls", result.get("qw3_model_requests")) != 0):
        raise SafetyHalt("Require the exact completed independent qualification")
    for name, sha in result["source_sha256"].items():
        image.pinned(ROOT / "apps/computer_use" / name, sha)


def prepare(args):
    image.pinned(args.plan, PLAN_SHA)
    plan = read(args.plan)
    reg, state = registration(args.parent_registration)
    base.require_identity(reg["identity"])
    parent = bounded.validate(Path(reg["protocol"]))
    qualify(args.http_qualification, "query_refresh_http_cpu_qualification", 9)
    qualify(args.engine_qualification, "query_refresh_engine_cpu_qualification", 9)
    qualify(args.qualification, "query_refresh_replay_queue_cpu_qualification", 10)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    sources = {}
    for name, record in plan["traces"].items():
        data = archive.read_request(record["path"])
        if archive.digest(data) != record["request_sha256"]:
            raise SafetyHalt("Original saved desktop input changed")
        image.pinned(Path(record["path"]) / "metrics.json", record["metrics_sha256"])
        path = args.output / (name + "-request.json")
        with path.open("xb") as file: file.write(data)
        path.chmod(0o600)
        sources[name] = dict(record, source_request=str(path))
    pair_identity({name: Path(record["source_request"]).read_bytes() for name, record in sources.items()})
    profile_path = args.output / "engine-profile.json"
    engine.prepare(args.common_profile, profile_path, args.engine_qualification)
    paths = [args.plan, args.parent_registration, Path(reg["protocol"]), args.http_qualification,
             args.engine_qualification, args.qualification, profile_path]
    paths += [ROOT / "apps/computer_use" / name for name in HELPERS]
    paths += [Path(item["source_request"]) for item in sources.values()]
    p = dict(kind="declared_serial_saved_desktop_query_refresh", trials=trials(), source_plan=str(args.plan),
             parent_registration=str(args.parent_registration), parent_protocol=reg["protocol"],
             parent_queue=reg["queue"], parent_controller=reg["identity"], parent_app=parent["parent_app"],
             engine_off_identity=parent["engine_off_identity"], engine_profile=str(profile_path),
             qualification=str(args.qualification), http_qualification=str(args.http_qualification),
             engine_qualification=str(args.engine_qualification), sources=sources,
             model_concurrency=1, generated_tool_calls_executed=0, advisor_enabled=False,
             original_generation_profile_unchanged=True, real_replays_started=0, final_engine_policy="off",
             files=[dict(path=str(path), sha256=owned.digest(path)) for path in dict.fromkeys(paths)])
    path = args.output / "protocol.json"
    with path.open("x") as file: json.dump(p, file, indent=2); file.write("\n")
    validate(path)
    return path


def validate(path):
    p = read(path)
    if (p.get("kind") != "declared_serial_saved_desktop_query_refresh" or p.get("trials") != trials()
            or p.get("model_concurrency") != 1 or p.get("generated_tool_calls_executed") != 0
            or p.get("advisor_enabled") is not False or p.get("final_engine_policy") != "off"
            or p.get("original_generation_profile_unchanged") is not True):
        raise SafetyHalt("Saved-input scope, order or cumulative model profile changed")
    for item in p["files"]: image.pinned(item["path"], item["sha256"])
    image.pinned(p["source_plan"], PLAN_SHA)
    qualify(p["qualification"], "query_refresh_replay_queue_cpu_qualification", 10)
    qualify(p["http_qualification"], "query_refresh_http_cpu_qualification", 9)
    qualify(p["engine_qualification"], "query_refresh_engine_cpu_qualification", 9)
    parent = bounded.validate(Path(p["parent_protocol"]))
    reg, state = registration(p["parent_registration"])
    if (reg["identity"] != p["parent_controller"] or reg["protocol"] != p["parent_protocol"]
            or reg["queue"] != p["parent_queue"] or parent["parent_app"] != p["parent_app"]
            or parent["engine_off_identity"] != p["engine_off_identity"]):
        raise SafetyHalt("Saved-input predecessor or original user services changed")
    engine.validate(read(p["engine_profile"]))
    inputs = {}
    for name, record in p["sources"].items():
        inputs[name] = Path(record["source_request"]).read_bytes()
        if archive.digest(inputs[name]) != record["request_sha256"]:
            raise SafetyHalt("Frozen replay source input changed")
    pair_identity(inputs)
    return p


class Controller(image.Controller):
    def fresh_refresh_engine(self, row):
        with owned.model_lock():
            self.parent_idle()
            previous = self.state.get("registered_engine", self.state["parent_services"]["registered_engine"])
            self.checkpoint(status="retiring-own-predecessor-engine", current_trial=row["id"],
                            engine_retirement=owned.retire_registered(previous))
            deadline = time.monotonic() + self.args.resource_wait_seconds
            while True:
                snapshot = owned.gpu_snapshot(); self.checkpoint(last_gpu_snapshot=snapshot)
                if snapshot["free_mib"] >= self.args.min_free_mib: break
                if time.monotonic() >= deadline: raise SafetyHalt("Shared GPU headroom unavailable; other work preserved")
                time.sleep(5)
            runtime = self.args.protocol.parent / "engines" / row["id"]
            if runtime.exists(): raise SafetyHalt("Never reuse an earlier saved-input engine")
            command = [str(q.CONTROLLER_PYTHON), "-m", "query_refresh_engine", "--profile", self.p["engine_profile"],
                       "--mode", row["mode"], "--run-dir", str(runtime), "--method", "dflash2", "--kvmem", "on",
                       "--prefix-reuse", "off", "--visual-history", "pin-all", "--guided-refresh", row["mode"],
                       "--context-tokens", "524288", "--gpu-sharing", "spare-capacity",
                       "--min-free-mib", str(self.args.min_free_mib), "--lock-timeout", "5"]
            child, _ = self.spawn(command, row["id"] + "-engine")
            self.ready(child, lambda: base.request("http://127.0.0.1:18080", "/v1/models", timeout=3),
                       lambda value: value["data"][0].get("context_window") == 524288, 600)
            expected = read(self.p["engine_profile"])["engine_identities"][row["mode"]]
            actual = owned.register_engine(runtime, expected)
            if actual["pid"] != child.pid: raise SafetyHalt("Fresh refresh engine is not our registered child")
            self.checkpoint(engine_runtime=str(runtime), registered_engine=actual)

    def dispatch(self, row):
        validate(self.args.protocol)
        image.pinned(self.args.protocol, self.state["protocol_sha256"])
        output = self.args.protocol.parent / "attempts" / row["id"]
        if output.exists(): raise SafetyHalt("Never overwrite or resume a saved-input replay")
        self.fresh_refresh_engine(row)
        with owned.model_lock():
            self.parent_idle()
            expected = read(self.p["engine_profile"])["engine_identities"][row["mode"]]
            actual = owned.register_engine(Path(self.state["engine_runtime"]), expected)
            if not owned.same_process(self.state["registered_engine"], actual):
                raise SafetyHalt("Refresh engine identity changed before inference")
            self.checkpoint(status="replaying-exact-inputs", current_trial=row["id"])
            inputs = {name: Path(record["source_request"]).read_bytes() for name, record in self.p["sources"].items()}
            result = asyncio.run(sequence(inputs, output, "http://127.0.0.1:18080", seconds=row["request_seconds"],
                                         expected_prompt_tokens={name: record["input_tokens"] for name, record in self.p["sources"].items()}))
            diagnostic = analyze((Path(self.state["engine_runtime"]) / "engine.log").read_bytes())
            if len(diagnostic["requests"]) != 2: raise SafetyHalt("Another request entered the private replay engine")
            pressure = diagnostic["above_budget_summary"]
            if row["mode"] == "off":
                mechanism = diagnostic["gated_pressure_requests"] == 1 and pressure["captured_query_events"] == 0 and pressure["window_fallback_events"] > 0
            else:
                mechanism = pressure["captured_query_events"] > 0 and pressure["successful_query_events"] > 0
            report = dict(mode=row["mode"], mechanism_qualified=mechanism, diagnostic=diagnostic,
                          result_sha256=owned.digest(output / "sequence-result.json"),
                          tool_calls_executed=0, business_or_latency_improvement_proven=False)
            with (output / "retrieval-evidence.json").open("x") as file: json.dump(report, file, indent=2); file.write("\n")
            self.state["trials"].append(dict(row, result=result, result_sha256=report["result_sha256"],
                                             evidence_sha256=owned.digest(output / "retrieval-evidence.json"), mechanism_qualified=mechanism))
            self.checkpoint(completed_replays=len(self.state["trials"]))
            self.parent_idle()


def run(args):
    p = validate(args.protocol)
    args.state_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    state = dict(kind="serial_saved_desktop_query_refresh_queue", pid=os.getpid(), started_at=time.time(),
                 protocol_sha256=owned.digest(args.protocol), controller_sha256=owned.digest(__file__),
                 status="waiting-for-complete-bounded-chain", trials=[], planned_trials=4,
                 model_concurrency=1, tool_calls_executed=0, device_inputs=0, business_success_claimed=False)
    controller = Controller(args, p, state); controller.checkpoint()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            services = predecessor_ready(p)
            if services is not None: break
            if time.monotonic() >= deadline: raise SafetyHalt("Predecessor observation deadline; live work preserved")
            time.sleep(3)
        controller.checkpoint(parent_services=services)
        for row in p["trials"]: controller.dispatch(row)
        # The user app still owns its original OFF contract after this probe.
        image.Controller.fresh_engine(controller, dict(id="restore-original-off"))
        controller.checkpoint(status="completed", finished_at=time.time(), termination_confirmed=True,
                              parent_services_preserved=True, final_engine_policy="off",
                              mechanism_qualified=all(r["mechanism_qualified"] for r in state["trials"]))
    except BaseException as exc:
        controller.checkpoint(status="halted-no-further-dispatch", finished_at=time.time(),
                              error=type(exc).__name__ + ": " + str(exc), live_evidence_preserved=True)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    for name in ("plan", "parent-registration", "http-qualification", "engine-qualification", "qualification",
                 "common-profile", "output", "protocol", "state-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if not 60 <= args.wait_seconds <= 172800 or not 60 <= args.resource_wait_seconds <= 86400 or args.min_free_mib != 56000:
        parser.error("Keep the qualified bounded wait and exact shared headroom")
    names = ("plan", "parent_registration", "http_qualification", "engine_qualification", "qualification", "common_profile", "output") if args.operation == "prepare" else ("protocol", "state_dir")
    for name in names:
        if getattr(args, name) is None: parser.error("Missing " + name)
        setattr(args, name, getattr(args, name).resolve())
    if args.operation == "prepare":
        print(json.dumps(dict(protocol=str(prepare(args)), model_calls=0, device_inputs=0))); return
    if args.state_dir != args.protocol.parent / "queue-runtime-01" or os.sys.executable != str(q.CONTROLLER_PYTHON):
        raise SafetyHalt("Keep the one declared queue and lexical qualified interpreter")
    def interrupted(*_): raise SafetyHalt("Refresh waiter interrupted; same live work and evidence preserved")
    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(dict(status=run(args)["status"])))


if __name__ == "__main__": main()
