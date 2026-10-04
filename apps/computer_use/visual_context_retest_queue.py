"""Run twelve matched image-policy trials after the frozen 40 + 8 + 10.

Preparation freezes a private package and does not contact a model or device.
Dispatch retains the original runners, initializers, graders and recordings.
Temporary browser clients release their device lease and preserve SDK51 UI20.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

import execution_retest_queue as base
import execution_retest_queue_v2 as sdk
import osworld_retired_trial_successors as successors
from osworld_linux import archive

owned, ROOT, RUNTIME, SafetyHalt = base.owned, base.ROOT, base.RUNTIME, base.SafetyHalt
read, save = base.read, base.save
POLICIES = ("full-retrievable", "recent-observations")
EXTRA_HELPERS = ("visual_context_retest_queue.py", "visual_context_osworld_run.py",
                 "visual_context_browser_run.py", "execution_retest_queue.py",
                 "execution_retest_queue_v2.py", "osworld_retired_trial_successors.py",
                 "osworld_queue_continuation.py", "resumed_execution_retest_queue.py",
                 "resumed_kvmem_queue.py", "osworld_linux/archive.py")
MODEL = dict(weights="Qwen3.8-27B-Q8_0.gguf", vision="Qwen3.8-27B-BF16", thinking=True,
             thinking_budget=0, max_output_tokens=16384, context_tokens=524288, kv_dtype="fp8",
             dflash="dflash2", dflash_max_tokens=7, kvmem="off", prefix_reuse="off",
             visual_history="pin-all", guided_refresh="off", advisor=False)


def declared_trials():
    rows = []
    for rep in (1, 2):
        for index, task in enumerate(("004", "013", "linux-browser")):
            policies = POLICIES if (rep + index) % 2 else POLICIES[::-1]
            for policy in policies:
                osworld = task != "linux-browser"
                rows.append(dict(id=f"{task}-{policy}-r{rep}", kind="original-osworld" if osworld else "normal-pilot",
                                 task=task, policy=policy, repetition=rep,
                                 seconds=1800 if osworld else 900, tool_calls=250 if osworld else 100))
    return rows


def validate_profile(p):
    if p.get("model") != MODEL or p.get("trials") != declared_trials():
        raise SafetyHalt("Only the declared twelve image-policy trials and identical model profile are admitted")


def pinned(path, sha):
    if owned.digest(path) != sha:
        raise SafetyHalt("Frozen source or evidence changed: " + str(path))


def inventory(package):
    return {f.name: owned.digest(f) for f in Path(package).iterdir() if f.is_file()}


def validate_candidate(plan):
    validate_profile(plan)
    candidate = Path(plan["candidate"])
    pinned(candidate / "build.json", plan["candidate_build_sha256"])
    build = read(candidate / "build.json")
    if (build.get("kind") != "isolated_matched_visual_context_comparison_candidate"
            or build["candidate_files_sha256"] != plan["candidate_inventory"]
            or inventory(candidate / "framework/qw3_computer") != plan["candidate_inventory"]):
        raise SafetyHalt("Matched candidate package identity changed")
    pinned(plan["transport_qualification"], plan["transport_qualification_sha256"])
    q = read(plan["transport_qualification"])
    checks = [c for platform in q.get("platforms", []) for c in platform["checks"]]
    if (q.get("passed") is not True or q.get("candidate_build_sha256") != plan["candidate_build_sha256"]
            or len(checks) != 288 or not all(c.get("passed") is True for c in checks)
            or len(q.get("comparison_checks", [])) != 12
            or not all(c.get("passed") is True for c in q["comparison_checks"])
            or q.get("qw3_model_calls") != 0):
        raise SafetyHalt("Require the exact actual-Codex transport qualification; it is not model acceptance")
    for stem in ("native_windows_transport_qualification", "native_windows_export_verification"):
        pinned(plan[stem], plan[stem + "_sha256"])


def validate_cpu(path):
    q = read(path)
    if (q.get("kind") != "visual_context_controller_cpu_boundary_qualification"
            or q.get("passed") is not True or q.get("tests_run", 0) < 15
            or q.get("real_model_calls") != 0 or q.get("native_device_inputs") != 0):
        raise SafetyHalt("Require the completed CPU boundary qualification before declaration")
    for name in (*EXTRA_HELPERS[:3], "tests/test_visual_context_retest_queue.py"):
        pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def prepare(plan_path, directory, followup, memory_registration, cpu_qualification):
    """One-time source preparation, no input injection or GPU requests."""
    plan = read(plan_path)
    validate_candidate(plan)
    validate_cpu(cpu_qualification)
    d, _, _, _ = successors.validate_followup(followup)
    sp, kp = sdk.validate(Path(d["sdk_protocol"]))
    registration = read(memory_registration)
    identity = owned.process_identity(read(Path(d["sdk_state_dir"]) / "queue.json")["pid"])
    if (identity is None or not successors.command_path(identity, "--followup", followup)
            or not successors.command_path(identity, "--memory-controller", memory_registration)
            or "resumed_execution_retest_queue" not in identity["command"]):
        raise SafetyHalt("Current SDK controller identity is unproven")
    base.require_identity(identity)
    base.require_identity(registration["identity"])
    directory.mkdir(parents=True, exist_ok=False)
    private_plan = directory / "plan.json"
    shutil.copyfile(plan_path, private_plan)
    framework = directory / "framework"
    shutil.copytree(Path(plan["candidate"]) / "framework/qw3_computer", framework / "qw3_computer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("device_gateway_entry.py", "bounded_linux_observation.py", "linux_app_semantics.py"):
        shutil.copyfile(Path(sp["framework"]) / name, framework / name)
    acceptance = read(Path(d["memory_protocol"]).parent / "off/acceptance.json")
    acceptance.update(framework_files=base.source_identity(framework), frozen_at=time.time(),
                      scope="Matched image presentation only; original task, initializer, grader and budgets retained.",
                      trials={r["id"]: {"task": r["task"], "arm": "verified"}
                              for r in plan["trials"] if r["kind"] == "original-osworld"})
    save(directory / "acceptance.json", acceptance)
    helpers = tuple(dict.fromkeys((*base.HELPERS, *EXTRA_HELPERS)))
    p = dict(kind="declared_serial_matched_image_policy_retest", status="declared-waiting-for-predecessors",
             declared_at=time.time(), model=plan["model"], trials=plan["trials"],
             plan=str(private_plan), plan_sha256=owned.digest(private_plan), original_plan=str(plan_path),
             original_plan_sha256=owned.digest(plan_path), framework=str(framework),
             framework_inventory=plan["candidate_inventory"], acceptance=str(directory / "acceptance.json"),
             acceptance_sha256=owned.digest(directory / "acceptance.json"), followup=str(followup),
             cpu_qualification=str(cpu_qualification), cpu_qualification_sha256=owned.digest(cpu_qualification),
             followup_sha256=owned.digest(followup), memory_registration=str(memory_registration),
             memory_registration_sha256=owned.digest(memory_registration), sdk_protocol=str(Path(d["sdk_protocol"])),
             sdk_protocol_sha256=owned.digest(d["sdk_protocol"]), sdk_controller_identity=identity,
             sdk_queue=str(Path(d["sdk_state_dir"]) / "queue.json"),
             memory_protocol=str(Path(d["memory_protocol"])), memory_protocol_sha256=owned.digest(d["memory_protocol"]),
             memory_queue=str(Path(d["memory_state_dir"]) / "queue.json"), osworld_queue=d["continued_osworld_queue"],
             engine_off_identity=kp["engine_identities"]["off"], parent_app=sp["app"],
             helper_sha256={n: owned.digest(ROOT / "apps/computer_use" / n) for n in helpers},
             model_concurrency=1, fresh_engine_per_trial=True, overwrite_old_attempts=False,
             advisor_enabled=False, mock_qualification_is_not_task_success=True)
    path = directory / "protocol.json"
    save(path, p)
    validate(path)
    return p


def validate(path):
    path = Path(path).resolve()
    p = read(path)
    validate_profile(p)
    if (p.get("kind") != "declared_serial_matched_image_policy_retest"
            or p.get("status") != "declared-waiting-for-predecessors"
            or p.get("model_concurrency") != 1 or p.get("fresh_engine_per_trial") is not True
            or p.get("overwrite_old_attempts") is not False or p.get("advisor_enabled") is not False):
        raise SafetyHalt("Use a fresh isolated matched image-policy declaration")
    pinned(p["plan"], p["plan_sha256"])
    pinned(p["cpu_qualification"], p["cpu_qualification_sha256"])
    validate_cpu(p["cpu_qualification"])
    plan = read(p["plan"])
    validate_candidate(plan)
    if (Path(p["framework"]) != path.parent / "framework"
            or p["framework_inventory"] != plan["candidate_inventory"]
            or inventory(Path(p["framework"]) / "qw3_computer") != p["framework_inventory"]):
        raise SafetyHalt("Private comparison package changed")
    for name, sha in p["helper_sha256"].items():
        pinned(ROOT / "apps/computer_use" / name, sha)
    for stem in ("followup", "memory_registration", "sdk_protocol", "memory_protocol", "acceptance"):
        pinned(p[stem], p[stem + "_sha256"])
    d, _, _, _ = successors.validate_followup(Path(p["followup"]))
    sp, kp = sdk.validate(Path(p["sdk_protocol"]))
    reg = read(p["memory_registration"])
    if (reg.get("kind") != "actual_resumed_memory_controller_identity"
            or reg.get("followup_sha256") != p["followup_sha256"]
            or reg.get("wrapper_sha256") != p["helper_sha256"]["resumed_kvmem_queue.py"]
            or not successors.command_path(reg["identity"], "--followup", p["followup"])
            or Path(p["sdk_queue"]) != Path(d["sdk_state_dir"]) / "queue.json"
            or Path(p["memory_queue"]) != Path(d["memory_state_dir"]) / "queue.json"
            or p["osworld_queue"] != d["continued_osworld_queue"]
            or not successors.command_path(p["sdk_controller_identity"], "--followup", p["followup"])
            or not successors.command_path(p["sdk_controller_identity"], "--memory-controller", p["memory_registration"])):
        raise SafetyHalt("Frozen predecessor identities or queue routing changed")
    a = read(p["acceptance"])
    if (a["framework_files"] != base.source_identity(Path(p["framework"]))
            or a["engine_identity"] != p["engine_off_identity"]
            or p["engine_off_identity"] != kp["engine_identities"]["off"]
            or (a["seconds"], a["tools"], a["model_output_tokens"]) != (1800, 250, 16384)
            or a["trials"] != {r["id"]: {"task": r["task"], "arm": "verified"}
                               for r in p["trials"] if r["kind"] == "original-osworld"}
            or p["parent_app"] != sp["app"]):
        raise SafetyHalt("Original OSWorld source admission or budget changed")
    pinned(Path(p["framework"]) / "device_gateway_entry.py", a["canonical_entry_sha256"])
    for name, sha in a["framework_auxiliary_files"].items():
        pinned(Path(p["framework"]) / name, sha)
    return p


def mixed_terminal_rows(declared, state, directory, sha):
    if (state.get("status") != "completed" or state.get("protocol_sha256") != sha
            or [r["id"] for r in state.get("trials", [])] != [r["id"] for r in declared]):
        raise SafetyHalt("SDK predecessor inventory is not exactly the declared terminal ten")
    receipts = []
    for spec, row in zip(declared, state["trials"]):
        output = Path(directory) / "attempts" / row["id"]
        result_path = output / ("result.json" if spec["kind"] == "original-osworld" else "integration-result.json")
        result = read(result_path)
        base.terminal_trial(spec, result)
        if result != row["result"] or owned.digest(result_path) != row.get("result_sha256"):
            raise SafetyHalt("SDK predecessor durable result changed")
        owned.assert_no_services(output)
        if spec["kind"] == "original-osworld":
            owned.assert_vm_retired(output)
        receipts.append(dict(id=row["id"], result_sha256=owned.digest(result_path)))
    return receipts


def predecessors_ready(p):
    osq, kvq, sq = (read(p[k]) for k in ("osworld_queue", "memory_queue", "sdk_queue"))
    identities = (read(p["followup"])["continuation_controller_identity"],
                  read(p["memory_registration"])["identity"], p["sdk_controller_identity"])
    for q, identity in zip((osq, kvq, sq), identities):
        if q.get("pid") != identity["pid"] or str(q.get("status", "")).startswith("halted"):
            raise SafetyHalt("A predecessor halted or changed identity; original work preserved")
        if q.get("status") != "completed":
            base.require_identity(identity)
    if any(q.get("status") != "completed" for q in (osq, kvq, sq)):
        return None
    if not all(base.process_retired(identity) for identity in identities):
        return None
    sp, kp = sdk.validate(Path(p["sdk_protocol"]))
    os_receipts = successors.terminal_parent(kp, osq, Path(p["followup"]))
    kv_receipts = base.exact_rows(kp["trials"], kvq, Path(p["memory_protocol"]).parent,
                                  p["memory_protocol_sha256"])
    if [r["id"] for r in kvq["trials"]] != [r["id"] for r in kp["trials"]]:
        raise SafetyHalt("Memory trial order changed")
    for row in kvq["trials"]:
        out = Path(p["memory_protocol"]).parent / "attempts" / row["id"]
        owned.assert_no_services(out)
        owned.assert_vm_retired(out)
    sdk_receipts = mixed_terminal_rows(sp["trials"], sq, Path(p["sdk_protocol"]).parent,
                                       p["sdk_protocol_sha256"])
    if sq.get("termination_confirmed") is not True or sq.get("new_services_preserved") is not True:
        raise SafetyHalt("SDK final service handoff is not confirmed")
    base.require_identity(sq["model_client"])
    owned.assert_apps_idle(p["parent_app"])
    pinned(sq["registry"], sq["registry_sha256"])
    for gateway in sq["new_gateways"]:
        base.require_identity(gateway["process_identity"])
        command = gateway["command"]
        base.ledger_terminal(Path(command[command.index("--runtime") + 1]))
    engine = owned.register_engine(Path(sq["engine_runtime"]), p["engine_off_identity"])
    if not owned.same_process(sq["registered_engine"], engine):
        raise SafetyHalt("Final SDK engine identity changed")
    return dict(osworld=os_receipts, kvmem=kv_receipts, sdk=sdk_receipts,
                registered_engine=engine, model_client=sq["model_client"],
                registry=sq["registry"], registry_sha256=sq["registry_sha256"], new_gateways=sq["new_gateways"])


def verify_trace(trace, policy):
    """Compare saved source and actual wire bytes, without exporting task text."""
    report = read(trace / "image-context.json")
    meta = read(trace / "request.full.archive.json")
    source = json.dumps(archive.decode(meta["tree"], trace.parent / "full-history-blobs"), ensure_ascii=False).encode()
    presented = archive.read_request(trace)
    if (report.get("policy") != policy or archive.digest(source) != meta["sha256"]
            or len(source) != meta["bytes"] or meta["sha256"] != report["source_sha256"]
            or archive.digest(presented) != report["presented_sha256"]
            or len(presented) != report["presented_bytes"]):
        raise SafetyHalt("Saved source or actual model wire differs from the declared image view")
    metrics = read(trace / "metrics.json") if (trace / "metrics.json").exists() else None
    return dict(trace=trace.name, source_sha256=meta["sha256"], presented_sha256=report["presented_sha256"],
                image_context=report, metrics=metrics)


class Controller(base.Controller):
    def parent_idle(self):
        base.require_identity(self.state["parent_services"]["model_client"])
        owned.assert_apps_idle(self.p["parent_app"])
        pinned(self.state["parent_services"]["registry"], self.state["parent_services"]["registry_sha256"])
        for gateway in self.state["parent_services"]["new_gateways"]:
            base.require_identity(gateway["process_identity"])
            command = gateway["command"]
            base.ledger_terminal(Path(command[command.index("--runtime") + 1]))

    def fresh_engine(self, row):
        with owned.model_lock():
            self.parent_idle()
            owned.assert_no_services(self.args.protocol.parent / "attempts")
            owned.assert_no_services(self.args.protocol.parent / "clients")
            previous = self.state.get("registered_engine", self.state["parent_services"]["registered_engine"])
            self.checkpoint(status="restoring-fresh-off-engine", current_trial=row["id"],
                            engine_retirement=owned.retire_registered(previous))
            deadline = time.monotonic() + self.args.resource_wait_seconds
            while True:
                snapshot = owned.gpu_snapshot()
                self.checkpoint(last_gpu_snapshot=snapshot)
                if snapshot["free_mib"] >= self.args.min_free_mib:
                    break
                if time.monotonic() >= deadline:
                    raise SafetyHalt("Shared GPU headroom unavailable; other workloads preserved")
                time.sleep(5)
            runtime = self.args.protocol.parent / "engines" / row["id"]
            if runtime.exists():
                raise SafetyHalt("Existing engine runtime cannot be reused")
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "apps/computer_use/run_engine.py"),
                       "--run-dir", str(runtime), "--method", "dflash2", "--kvmem", "off", "--prefix-reuse", "off",
                       "--visual-history", "pin-all", "--guided-refresh", "off", "--context-tokens", "524288",
                       "--gpu-sharing", "spare-capacity", "--min-free-mib", str(self.args.min_free_mib), "--lock-timeout", "5"]
            child, _ = self.spawn(command, row["id"] + "-engine")
            self.ready(child, lambda: base.request("http://127.0.0.1:18080", "/v1/models", timeout=3),
                       lambda v: v["data"][0].get("context_window") == 524288, 600)
            engine = owned.register_engine(runtime, self.p["engine_off_identity"])
            if engine["pid"] != child.pid:
                raise SafetyHalt("Fresh engine is not the registered owned child")
            self.checkpoint(engine_runtime=str(runtime), registered_engine=engine)

    @contextlib.contextmanager
    def browser_client(self, row):
        services = self.state["parent_services"]
        original = base.request(self.p["parent_app"], "/api/state")["devices"]["selected"]
        self.checkpoint(parent_selected_before_browser=original)
        if original == "linux-vm":
            base.request(self.p["parent_app"], "/api/device", {"device_id": "windows-vm"})
            owned.assert_idle(self.p["parent_app"])
        runtime = self.args.protocol.parent / "clients" / row["id"]
        if runtime.exists():
            raise SafetyHalt("Existing browser client cannot be resumed")
        command = [str(ROOT / ".venv/bin/python"), "-m", "qw3_computer.app", "--port", "18217",
                   "--runtime", str(runtime), "--device-registry", services["registry"], "--selected-device", "linux-vm",
                   "--execution-mode", "verified", "--thinking", "--image-history", row["policy"],
                   "--model-output-tokens", "16384", "--tool-call-limit", "100", "--output-policy", "task"]
        child, identity = self.spawn(command, row["id"] + "-client", self.p["framework"])
        app = "http://127.0.0.1:18217"
        self.ready(child, lambda: base.request(app, "/api/state", timeout=3),
                   lambda v: v.get("ready") is True and v.get("status") == "idle"
                   and v.get("desktop_paused") is True and not v.get("uncertain_actions")
                   and v.get("thinking") is True and v.get("model_output_tokens") == 16384
                   and v.get("tool_call_limit") == 100)
        self.checkpoint(temporary_client=identity, temporary_client_runtime=str(runtime))
        # On an unknown input or timeout keep the live client and lease for audit.
        yield app, runtime
        owned.assert_idle(app)
        self.checkpoint(temporary_client_retirement=owned.retire_registered(identity))
        owned.assert_no_services(runtime)
        reset = base.request(self.p["parent_app"], "/api/device", {"device_id": original})
        if reset["devices"]["selected"] != original:
            raise SafetyHalt("Parent device lease restoration is unconfirmed")
        self.parent_idle()

    def trial_command(self, row, output, app=None, runtime=None):
        python = str(ROOT / "third_party/OSWorld-V2/.venv/bin/python")
        common = [python, "-m"]
        declaration = ["--comparison-protocol", str(self.args.protocol), "--image-history", row["policy"]]
        if row["kind"] == "normal-pilot":
            return common + ["visual_context_browser_run", *declaration, "--case", "linux_browser", "--seconds", "900",
                             "--app", app, "--runtime", str(runtime), "--output", str(output),
                             "--retire-task", "--linux-video-mode", "vnc"]
        return common + ["visual_context_osworld_run", *declaration, "--task", row["task"], "--arm", "verified",
                         "--output", str(output), "--framework", self.p["framework"],
                         "--engine-runtime", self.state["engine_runtime"], "--formal-campaign",
                         str(RUNTIME / "linux/campaign-v2-20261004"), "--acceptance-manifest", self.p["acceptance"],
                         "--pilot-app", self.p["parent_app"], "--model-output-tokens", "16384"]

    def collect(self, row, output, code, inference):
        result_path = output / ("result.json" if row["kind"] == "original-osworld" else "integration-result.json")
        result = read(result_path)
        base.terminal_trial(row, result)
        owned.assert_no_services(output)
        if row["kind"] == "original-osworld":
            owned.assert_vm_retired(output)
        task = read(output / ("app/task.json" if row["kind"] == "original-osworld" else "task.json"))
        if task.get("image_history") != row["policy"]:
            raise SafetyHalt("Actual task did not use its declared image policy")
        traces = sorted(Path(inference).glob("*/image-context.json"))
        wire_traces = {f.parent for pattern in ("*/request.json", "*/request.archive.json")
                       for f in Path(inference).glob(pattern)}
        if not wire_traces.issubset({f.parent for f in traces}):
            raise SafetyHalt("Actual model requests lack declared image-policy evidence")
        verified = [verify_trace(f.parent, row["policy"]) for f in traces]
        if not verified:
            raise SafetyHalt("No actual image-policy request evidence; do not count a mock or empty trial")
        # Collection occurs only after the task is durably terminal. Preserve the
        # original grade unchanged; trace archives roundtrip the actual wire.
        archive_completed = archive.archive_completed(inference)
        for entry, f in zip(verified, traces):
            if archive.digest(archive.read_request(f.parent)) != entry["presented_sha256"]:
                raise SafetyHalt("Archived actual wire changed during lossless compaction")
        summary = dict(kind="actual_qw3_matched_image_policy_trial_evidence", policy=row["policy"],
                       model_profile=self.p["model"], traces=verified, losslessly_compacted_requests=archive_completed,
                       result_sha256=owned.digest(result_path), business_grade_unchanged=True,
                       administrative_recovery_is_not_business_success=True)
        if row["kind"] == "normal-pilot":
            stopped = read(output / "browser-service-stop.json")
            if stopped.get("owned_process_identity_checked") is not True:
                raise SafetyHalt("Browser fixture service retirement is unconfirmed")
            prompt = read(output / "pilot-manifest.json")["prompt"]
            summary["user_prompt_sha256"] = archive.digest(prompt.encode())
            pair = next((r for r in self.state["trials"] if r["kind"] == "normal-pilot"
                         and r["repetition"] == row["repetition"]), None)
            if pair and pair["user_prompt_sha256"] != summary["user_prompt_sha256"]:
                raise SafetyHalt("Matched browser user fields or prompt changed between arms")
        evidence = output / "visual-context-evidence.json"
        if evidence.exists():
            raise SafetyHalt("Existing trial evidence cannot be overwritten")
        save(evidence, summary)
        self.state["trials"].append(dict(id=row["id"], kind=row["kind"], policy=row["policy"], repetition=row["repetition"],
                                         exit_code=code, result=result, result_sha256=owned.digest(result_path),
                                         evidence_sha256=owned.digest(evidence), trace_count=len(verified),
                                         user_prompt_sha256=summary.get("user_prompt_sha256")))
        self.checkpoint(real_model_trials_completed=len(self.state["trials"]))

    def dispatch(self, row):
        validate(self.args.protocol)
        pinned(self.args.protocol, self.state["protocol_sha256"])
        self.parent_idle()
        output = self.args.protocol.parent / "attempts" / row["id"]
        if output.exists():
            raise SafetyHalt("Existing attempt cannot be replayed")
        self.fresh_engine(row)
        self.checkpoint(status="running", current_trial=row["id"])
        if row["kind"] == "normal-pilot":
            with owned.model_lock(), self.browser_client(row) as (app, runtime):
                child, _ = self.spawn(self.trial_command(row, output, app, runtime), row["id"])
                code = child.wait(timeout=1500)
                owned.assert_idle(app)
                self.collect(row, output, code, runtime / "inference")
        else:
            child, _ = self.spawn(self.trial_command(row, output), row["id"])
            code = child.wait(timeout=5100)
            self.collect(row, output, code, output / "app/inference")
        self.parent_idle()


def run(args):
    p = validate(args.protocol)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    state = dict(kind="serial_matched_visual_context_development_retest", pid=os.getpid(), started_at=time.time(),
                 status="waiting-for-forty-eight-and-ten", protocol_sha256=owned.digest(args.protocol),
                 controller_sha256=owned.digest(Path(__file__)), planned_trials=12, trials=[],
                 real_model_trials_completed=0, model_concurrency=1, business_success_improvement_proven=False)
    controller = Controller(args, p, state)
    controller.checkpoint()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            previous = predecessors_ready(p)
            if previous is not None:
                break
            if time.monotonic() >= deadline:
                raise SafetyHalt("Predecessor wait deadline reached; all services preserved")
            time.sleep(3)
        validate(args.protocol)
        controller.checkpoint(parent_services=previous)
        for row in p["trials"]:
            controller.dispatch(row)
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
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--followup", type=Path)
    parser.add_argument("--memory-registration", type=Path)
    parser.add_argument("--cpu-qualification", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if args.operation == "prepare":
        if any(getattr(args, key) is None for key in ("plan", "campaign", "followup", "memory_registration", "cpu_qualification")):
            parser.error("Preparation requires the existing plan and predecessor registrations")
        p = prepare(args.plan.resolve(), args.campaign.resolve(), args.followup.resolve(),
                    args.memory_registration.resolve(), args.cpu_qualification.resolve())
        print(json.dumps(dict(status=p["status"], declared_trials=len(p["trials"]), model_calls=0)))
        return
    if (args.protocol is None or args.state_dir is None or args.min_free_mib < 56000
            or not 60 <= args.wait_seconds <= 172800 or not 60 <= args.resource_wait_seconds <= 86400):
        parser.error("Specify a fresh state directory and bounded shared-resource waits")
    args.protocol, args.state_dir = args.protocol.resolve(), args.state_dir.resolve()
    def interrupted(*_):
        raise SafetyHalt("Image-policy waiter interrupted; live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    state = run(args)
    print(json.dumps(dict(status=state["status"], completed=len(state["trials"]))))


if __name__ == "__main__":
    main()
