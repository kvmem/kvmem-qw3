"""Matched literal-feedback ablation after the frozen 40/8/10/12/4 queue.

Reuse the existing OSWorld runner, initializer, evaluator, recorder and owned
engine supervisor. Only the two qualified feedback package files differ.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import time

import thinking_budget_replay_queue as predecessor
from windows_literal_feedback_bundle import wire_conditions

image, base, owned = predecessor.image, predecessor.base, predecessor.owned
ROOT, RUNTIME, read, save, SafetyHalt = predecessor.ROOT, predecessor.RUNTIME, predecessor.read, predecessor.save, predecessor.SafetyHalt
HELPERS = ("feedback_retest_queue.py", "tests/test_feedback_retest_queue.py", "windows_literal_feedback_bundle.py")


def trials():
    rows = []
    for repetition in (1, 2):
        for index, task in enumerate(("013", "004")):
            modes = ("off", "on") if (repetition + index) % 2 else ("on", "off")
            for mode in modes:
                rows.append(dict(id=f"{task}-literal-feedback-{mode}-r{repetition}",
                                 kind="original-osworld", task=task, feedback=mode,
                                 arm="verified", repetition=repetition, seconds=1800, tools=250))
    return rows


def validate_profile(p):
    if (p.get("kind") != "declared_matched_literal_feedback_osworld_retest"
            or p.get("trials") != trials() or p.get("model") != image.MODEL
            or p.get("model_concurrency") != 1 or p.get("overwrite_old_attempts") is not False
            or p.get("advisor_enabled") is not False):
        raise SafetyHalt("Use the exact eight matched feedback trials and unchanged local profile")


def qualify_cpu(path):
    q = read(path)
    if (q.get("kind") != "feedback_retest_controller_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 14 or q.get("qw3_model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require completed queue boundary qualification")
    for name in HELPERS:
        image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def candidate_proof(candidate, native_path, export_path):
    candidate = Path(candidate)
    build = read(candidate / "build.json")
    native, exported = read(native_path), read(export_path)
    if (build.get("kind") != "isolated_sdk51_literal_completion_feedback_candidate"
            or build.get("changed_package_files") != ["execution.py", "completion_feedback.py"]
            or build.get("input_schemas_changed") is not False or build.get("original_predicates_changed") is not False
            or native.get("passed") is not True or native.get("transport_checks_passed") != 36
            or native.get("candidate_build_sha256") != owned.digest(candidate / "build.json")
            or exported.get("passed") is not True or exported.get("native_request_schemas_verified") != 9
            or exported.get("native_failed_condition_feedback_requests_verified") != 3
            or exported.get("native_pending_databases_verified") != 3
            or exported.get("native_completion_records") != 0
            or exported.get("source_zip_sha256") != native.get("guest_evidence_sha256")):
        raise SafetyHalt("Require the qualified literal-only candidate and independent native evidence")
    for field in ("qw3_model_calls", "fixture_device_inputs"):
        if native.get(field) != 0 or exported.get(field) != 0:
            raise SafetyHalt("Synthetic transport qualification must not be treated as native business success")
    image.pinned(build["source_manifest"], build["source_manifest_sha256"])
    image.pinned(build["source_acceptance"], build["source_acceptance_sha256"])
    for name, sha in build["candidate_files_sha256"].items():
        image.pinned(candidate / "framework/qw3_computer" / name, sha)
    for name, sha in build["source_files_sha256"].items():
        image.pinned(Path(build["source_framework"]) / "qw3_computer" / name, sha)
    for name, sha in build["unchanged_auxiliary_files_sha256"].items():
        image.pinned(candidate / "framework" / name, sha)
        image.pinned(Path(build["source_framework"]) / name, sha)
    return build


def prepare(candidate, native_path, export_path, parent_protocol, registration, qualification, directory):
    candidate, native_path, export_path, parent_protocol, registration, qualification, directory = map(
        lambda p: Path(p).resolve(), (candidate, native_path, export_path, parent_protocol, registration, qualification, directory))
    qualify_cpu(qualification)
    build = candidate_proof(candidate, native_path, export_path)
    parent = predecessor.validate(parent_protocol)
    registered = read(registration)
    if (registered.get("kind") != "actual_readonly_thinking_budget_replay_waiter_identity"
            or registered.get("protocol_sha256") != owned.digest(parent_protocol)
            or registered.get("protocol") != str(parent_protocol)
            or "thinking_budget_replay_queue" not in registered["identity"]["command"]):
        raise SafetyHalt("Require the actual registered four-replay predecessor")
    base.require_identity(registered["identity"])
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    frameworks, acceptances = {}, {}
    for mode in ("off", "on"):
        source = Path(build["source_framework"]) if mode == "off" else candidate / "framework"
        destination = directory / mode / "framework"
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__"))
        frameworks[mode] = str(destination)
        a = read(build["source_acceptance"])
        a.update(framework_files=base.source_identity(destination), frozen_at=time.time(),
                 trials={r["id"]: {"task": r["task"], "arm": "verified"} for r in trials() if r["feedback"] == mode},
                 scope="Matched additive literal feedback only; original task, initializer, grader, recording and budgets unchanged.")
        if a["engine_identity"] != parent["engine_off_identity"]:
            raise SafetyHalt("SDK51 and predecessor engine profiles differ")
        acceptance = directory / mode / "acceptance.json"
        save(acceptance, a)
        acceptances[mode] = dict(path=str(acceptance), sha256=owned.digest(acceptance))
    helpers = tuple(dict.fromkeys((*base.HELPERS, *image.EXTRA_HELPERS, *predecessor.HELPERS, *HELPERS)))
    p = dict(kind="declared_matched_literal_feedback_osworld_retest", declared_at=time.time(),
             model=image.MODEL, trials=trials(), model_concurrency=1, overwrite_old_attempts=False, advisor_enabled=False,
             candidate=str(candidate), candidate_build_sha256=owned.digest(candidate / "build.json"),
             native_qualification=str(native_path), native_qualification_sha256=owned.digest(native_path),
             native_export=str(export_path), native_export_sha256=owned.digest(export_path),
             parent_protocol=str(parent_protocol), parent_protocol_sha256=owned.digest(parent_protocol),
             parent_registration=str(registration), parent_registration_sha256=owned.digest(registration),
             parent_queue=registered["queue"], parent_controller=registered["identity"],
             qualification=str(qualification), qualification_sha256=owned.digest(qualification),
             frameworks=frameworks, acceptances=acceptances, engine_off_identity=parent["engine_off_identity"],
             parent_app=parent["parent_app"], helper_sha256={n: owned.digest(ROOT / "apps/computer_use" / n) for n in helpers},
             primary_metric="Original independent task score; no grader information enters the agent",
             unchanged_recording_policy=True, native_transport_is_not_model_success=True,
             real_model_trials_started=0, business_success_improvement_proven=False)
    save(directory / "protocol.json", p)
    validate(directory / "protocol.json")
    return p


def validate(path):
    path = Path(path).resolve()
    p = read(path)
    validate_profile(p)
    for stem in ("native_qualification", "native_export", "parent_protocol", "parent_registration", "qualification"):
        image.pinned(p[stem], p[stem + "_sha256"])
    qualify_cpu(p["qualification"])
    image.pinned(Path(p["candidate"]) / "build.json", p["candidate_build_sha256"])
    build = candidate_proof(p["candidate"], p["native_qualification"], p["native_export"])
    parent = predecessor.validate(Path(p["parent_protocol"]))
    registered = read(p["parent_registration"])
    if (registered["identity"] != p["parent_controller"] or registered["queue"] != p["parent_queue"]
            or parent["engine_off_identity"] != p["engine_off_identity"] or parent["parent_app"] != p["parent_app"]):
        raise SafetyHalt("Actual predecessor or model ownership profile changed")
    for name, sha in p["helper_sha256"].items():
        image.pinned(ROOT / "apps/computer_use" / name, sha)
    for mode in ("off", "on"):
        framework = Path(p["frameworks"][mode])
        if framework != path.parent / mode / "framework":
            raise SafetyHalt("Require a private per-arm source copy")
        inventory = build["source_files_sha256"] if mode == "off" else build["candidate_files_sha256"]
        if image.inventory(framework / "qw3_computer") != inventory:
            raise SafetyHalt("Only the two declared feedback files may differ")
        for name, sha in build["unchanged_auxiliary_files_sha256"].items():
            image.pinned(framework / name, sha)
        acceptance = p["acceptances"][mode]
        image.pinned(acceptance["path"], acceptance["sha256"])
        a = read(acceptance["path"])
        if (a["framework_files"] != base.source_identity(framework) or a["engine_identity"] != p["engine_off_identity"]
                or (a["seconds"], a["tools"], a["model_output_tokens"]) != (1800, 250, 16384)
                or a["trials"] != {r["id"]: {"task": r["task"], "arm": "verified"}
                                   for r in p["trials"] if r["feedback"] == mode}):
            raise SafetyHalt("Original task source admission, initializer or budget changed")
    return p


def terminal_predecessor(p, state):
    """Four actual replays must be terminal, unchanged and retired before input."""
    if (state.get("pid") != p["parent_controller"]["pid"]
            or state.get("protocol_sha256") != p["parent_protocol_sha256"]
            or str(state.get("status", "")).startswith("halted")):
        raise SafetyHalt("Predecessor stopped or changed; live work preserved")
    if state.get("status") != "completed":
        base.require_identity(p["parent_controller"])
        return None
    if not base.process_retired(p["parent_controller"]):
        return None
    if (state.get("termination_confirmed") is not True or state.get("parent_services_preserved") is not True
            or state.get("final_engine_policy") != "off"
            or [r.get("id") for r in state.get("trials", [])] != [r["id"] for r in predecessor.trials()]):
        raise SafetyHalt("Require all four original saved-input replay results")
    for spec, row in zip(predecessor.trials(), state["trials"]):
        path = Path(p["parent_protocol"]).parent / "attempts" / spec["id"] / "result.json"
        result = read(path)
        if (owned.digest(path) != row.get("result_sha256") or result != row.get("result")
                or result.get("termination_confirmed") is not True or result.get("tool_calls_executed") != 0
                or result.get("business_success_claimed") is not False):
            raise SafetyHalt("Read-only predecessor evidence changed or work remains")
    owned.assert_no_services(Path(p["parent_protocol"]).parent / "attempts")
    services = copy.deepcopy(state["parent_services"])
    base.require_identity(services["model_client"])
    owned.assert_apps_idle(p["parent_app"])
    image.pinned(services["registry"], services["registry_sha256"])
    for gateway in services["new_gateways"]:
        base.require_identity(gateway["process_identity"])
        command = gateway["command"]
        base.ledger_terminal(Path(command[command.index("--runtime") + 1]))
    engine = owned.register_engine(Path(state["engine_runtime"]), p["engine_off_identity"])
    if not owned.same_process(state["registered_engine"], engine):
        raise SafetyHalt("Actual last thinking-replay engine changed")
    services["registered_engine"] = engine
    return services


def trace_evidence(inference, mode):
    directories = sorted({p.parent for pattern in ("*/request.json", "*/request.archive.json")
                          for p in Path(inference).glob(pattern)})
    rows = []
    for directory in directories:
        raw = image.archive.read_request(directory)
        body = json.loads(raw)
        if body.get("enable_thinking") is not True or body.get("max_tokens") != 16384:
            raise SafetyHalt("Actual model request differs from its frozen reasoning budget")
        feedback = [e for message in body.get("messages", []) if message.get("role") == "tool"
                    for e in wire_conditions(message.get("content"))]
        if mode == "off" and feedback:
            raise SafetyHalt("OFF request unexpectedly contains additive condition feedback")
        if any(e["comparison_evidence"].get("input_delivery_inferred") is not False
               or e["comparison_evidence"].get("independent_business_result_verified") is not False
               for e in feedback):
            raise SafetyHalt("UI condition feedback promoted to delivery or business proof")
        rows.append(dict(trace=directory.name,request_sha256=image.archive.digest(raw),
                         conditions_on_wire=len(feedback),
                         failed_conditions_on_wire=sum(e.get("passed") is False for e in feedback)))
    if not rows:
        raise SafetyHalt("No actual model request evidence; cannot count an empty or synthetic trial")
    return rows


class Controller(image.Controller):
    def trial_command(self, row, output):
        return [str(ROOT / "third_party/OSWorld-V2/.venv/bin/python"), "-m", "osworld_office_candidate_run",
                "--task", row["task"], "--arm", "verified", "--output", str(output),
                "--framework", self.p["frameworks"][row["feedback"]], "--engine-runtime", self.state["engine_runtime"],
                "--formal-campaign", str(RUNTIME / "linux/campaign-v2-20261004"),
                "--acceptance-manifest", self.p["acceptances"][row["feedback"]]["path"],
                "--pilot-app", self.p["parent_app"], "--model-output-tokens", "16384"]

    def collect(self, row, output, code):
        path = output / "result.json"
        result = read(path)
        base.terminal_trial(row, result)
        owned.assert_no_services(output)
        owned.assert_vm_retired(output)
        evidence = dict(kind="actual_qw3_matched_literal_feedback_trial",feedback=row["feedback"],
                        original_result_sha256=owned.digest(path),original_business_grade_unchanged=True,
                        traces=trace_evidence(output / "app/inference", row["feedback"]),
                        feedback_exposure_is_not_task_success=True)
        save(output / "literal-feedback-evidence.json", evidence)
        self.state["trials"].append(dict(row, exit_code=code, result=result, result_sha256=owned.digest(path),
                                         evidence_sha256=owned.digest(output / "literal-feedback-evidence.json")))
        self.checkpoint(real_model_trials_completed=len(self.state["trials"]))

    def dispatch(self, row):
        validate(self.args.protocol)
        image.pinned(self.args.protocol, self.state["protocol_sha256"])
        self.parent_idle()
        output = self.args.protocol.parent / "attempts" / row["id"]
        if output.exists():
            raise SafetyHalt("Original attempt cannot be replayed or overwritten")
        self.fresh_engine(row)
        self.checkpoint(status="running", current_trial=row["id"])
        child, _ = self.spawn(self.trial_command(row, output), row["id"], self.p["frameworks"][row["feedback"]])
        code = child.wait(timeout=5100)
        self.collect(row, output, code)
        self.parent_idle()


def run(args):
    p = validate(args.protocol)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    state = dict(kind="serial_matched_literal_feedback_retest",pid=os.getpid(),started_at=time.time(),
                 status="waiting-for-forty-eight-ten-twelve-four",protocol_sha256=owned.digest(args.protocol),
                 planned_trials=8,trials=[],real_model_trials_completed=0,model_concurrency=1)
    controller = Controller(args, p, state)
    controller.checkpoint()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            services = terminal_predecessor(p, read(p["parent_queue"]))
            if services is not None:
                break
            if time.monotonic() >= deadline:
                raise SafetyHalt("Predecessor observation deadline; all live services preserved")
            time.sleep(3)
        validate(args.protocol)
        controller.checkpoint(parent_services=services)
        for row in p["trials"]:
            controller.dispatch(row)
        controller.checkpoint(status="completed",finished_at=time.time(),termination_confirmed=True,
                              parent_services_preserved=True,final_engine_policy="off")
    except BaseException as error:
        controller.checkpoint(status="halted-no-further-dispatch",finished_at=time.time(),
                              error=type(error).__name__+": "+str(error),live_children_and_evidence_preserved=True)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    for name in ("candidate", "native-qualification", "native-export", "parent-protocol", "registration", "qualification", "directory"):
        prep.add_argument("--" + name, type=Path, required=True)
    work = sub.add_parser("run")
    work.add_argument("--protocol", type=Path, required=True)
    work.add_argument("--state-dir", type=Path, required=True)
    work.add_argument("--wait-seconds", type=int, default=172800)
    work.add_argument("--resource-wait-seconds", type=int, default=21600)
    work.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if args.operation == "prepare":
        p = prepare(args.candidate,args.native_qualification,args.native_export,args.parent_protocol,
                    args.registration,args.qualification,args.directory)
        print(json.dumps(dict(trials=len(p["trials"]),real_model_trials_started=0)))
        return
    if not (60 <= args.wait_seconds <= 172800 and 60 <= args.resource_wait_seconds <= 86400 and args.min_free_mib >= 56000):
        parser.error("Use the existing bounded shared-resource profile")
    args.protocol, args.state_dir = args.protocol.resolve(), args.state_dir.resolve()
    def interrupted(*_):
        raise SafetyHalt("Feedback supervisor interrupted; owned live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(dict(status=run(args)["status"])))


if __name__ == "__main__":
    main()
