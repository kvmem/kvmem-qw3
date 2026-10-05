"""Run the declared eight bounded-turn comparisons after the frozen successors.

Reuse the existing owned-engine supervisor, unchanged OSWorld runner, media and
grading. Only the isolated SDK51 app lifecycle differs between the two arms.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from unittest.mock import patch

import feedback_retest_queue as prior
import shared_quota_successors as successors

q = successors.RUNTIME
image, base, owned = q.image, q.base, q.owned
ROOT, RUNTIME, read, save, SafetyHalt = q.ROOT, q.image.RUNTIME, q.read, q.save, q.SafetyHalt
PLAN_SHA = "be29cc072e64c0a67aa9d9ce8eb77cee08c5522a159ea6ebfb2f58bfd8cde4ee"
PRIOR_SHA = "f43e296ae1e41ab8dbe9d13b7377246830b6b0b2137136c7ab0c7881c62406e4"
HELPERS = ("bounded_continuation_retest_queue.py", "tests/test_bounded_continuation_retest_queue.py")
AUXILIARY = ("device_gateway_entry.py", "bounded_linux_observation.py", "linux_app_semantics.py")


def trials():
    return [dict(id=f"{task}-bounded-{arm}-r{repeat}", task=task, arm=arm, platform="linux",
                 execution_mode="verified", seconds=1800, tools=250, repetition=repeat, seed=0)
            for repeat in (1, 2) for task in ("051", "004")
            for arm in (("control", "candidate") if repeat == 1 else ("candidate", "control"))]


def plan_proof(path):
    image.pinned(path, PLAN_SHA)
    plan = read(path)
    expected = dict(thinking=True, dflash2=True, kvmem="off", model_output_tokens=16384,
                    model_context_tokens=524288, gui_only=True, advisor=False, software=False,
                    prefix_reuse=False, visual_history="pin-all", model_concurrency=1)
    if (plan.get("kind") != "bounded_zero_tool_turn_development_retest_plan" or plan.get("trials") != trials()
            or plan.get("inherited_configuration") != expected or plan.get("task_budget_reset") is not False
            or plan.get("additional_turn_limit") != 2):
        raise SafetyHalt("Use the original eight bounded-turn development comparisons")
    image.pinned(plan["inherited_protocol"], plan["inherited_protocol_sha256"])
    for item in plan["task_sources"].values():
        image.pinned(ROOT / item["source"], item["source_sha256"])
        image.pinned(item["original_manifest"], item["original_manifest_sha256"])
    candidate = ROOT / plan["candidate"]
    image.pinned(candidate / "build.json", plan["candidate_manifest_sha256"])
    build = read(candidate / "build.json")
    if (build.get("framework_files") != plan["framework_files"] or build.get("changed_files") != ["app.py", "bounded_turn_continuation.py"]
            or build.get("source_pins_unchanged") is not True):
        raise SafetyHalt("Declared isolated continuation candidate changed")
    for arm, inventory in plan["framework_files"].items():
        package = candidate / arm / "qw3_computer"
        if {p.name for p in package.iterdir() if p.suffix in (".py", ".html")} != set(inventory):
            raise SafetyHalt("Candidate package inventory differs")
        for name, sha in inventory.items():
            image.pinned(package / name, sha)
    return plan, build


def native_proof(path, plan):
    native = read(path)
    if (native.get("kind") != "actual_codex_bounded_turn_native_batch" or native.get("passed") is not True
            or native.get("runs") != 24 or native.get("checks_passed") != 294
            or native.get("qw3_model_calls") != 0 or native.get("native_device_inputs") != 0
            or native.get("actual_codex_transport") is not True):
        raise SafetyHalt("Require the completed actual Codex lifecycle qualification")
    expected = {f"{device}-{scenario}-{arm}" for device in ("windows", "linux", "android")
                for scenario in ("bounded", "takeover", "finish", "lost-reply") for arm in ("control", "candidate")}
    rows = native.get("rows", [])
    if len(rows) != 24 or {r["id"] for r in rows} != expected:
        raise SafetyHalt("Native qualification inventory changed")
    for row in rows:
        if row.get("passed") is not True or row.get("exit_code") != 0:
            raise SafetyHalt("Native lifecycle qualification incomplete")
        result = read(row["result"])
        if (result.get("passed") is not True or result.get("candidate_build_sha256") != plan["candidate_manifest_sha256"]
                or result.get("qw3_model_calls") != 0 or result.get("native_device_inputs") != 0
                or result.get("actual_codex_transport") is not True):
            raise SafetyHalt("Native proof does not qualify this exact candidate")
    return native


def qualify(path):
    result = read(path)
    if (result.get("kind") != "bounded_continuation_retest_cpu_qualification" or result.get("passed") is not True
            or result.get("tests_run", 0) < 16 or result.get("skipped") != 0
            or result.get("model_calls") != 0 or result.get("device_inputs") != 0):
        raise SafetyHalt("Require the independent bounded comparison queue qualification")
    for name in HELPERS:
        image.pinned(ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def registration(path, declaration, d):
    return q.successor_registration(path, "feedback", declaration, d)


def validate(path):
    p = read(path)
    if (p.get("kind") != "declared_bounded_continuation_original_osworld_retest" or p.get("trials") != trials()
            or p.get("model") != image.MODEL or p.get("model_concurrency") != 1
            or p.get("advisor_enabled") is not False or p.get("overwrite_old_attempts") is not False):
        raise SafetyHalt("Bounded comparison tasks, local profile or permissions changed")
    for item in p["files"]:
        image.pinned(item["path"], item["sha256"])
    qualify(p["qualification"])
    plan, build = plan_proof(Path(p["plan"]))
    native_proof(p["native_qualification"], plan)
    d, _, _ = q.validate(Path(p["parent_declaration"]))
    parent = q.frozen_protocol("feedback", Path(d["feedback_protocol"]))
    reg = registration(Path(p["parent_registration"]), Path(p["parent_declaration"]), d)
    q.successor_queue(reg, Path(p["parent_declaration"]))
    if (reg["protocol"] != p["parent_protocol"] or reg["queue"] != p["parent_queue"]
            or reg["identity"] != p["parent_controller"] or parent["engine_off_identity"] != p["engine_off_identity"]
            or parent["parent_app"] != p["parent_app"]):
        raise SafetyHalt("Actual frozen feedback predecessor or engine profile changed")
    original = read(plan["inherited_protocol"])
    for arm in ("control", "candidate"):
        framework = Path(p["frameworks"][arm])
        actual = base.source_identity(framework)
        expected = dict(read(original["acceptance"])["framework_files"])
        for name, sha in plan["framework_files"][arm].items():
            expected["qw3_computer/" + name] = sha
        if actual != expected:
            raise SafetyHalt("Full candidate/auxiliary source identity changed")
        for name in AUXILIARY:
            image.pinned(framework / name, owned.digest(Path(original["framework"]) / name))
        receipt = p["acceptances"][arm]
        image.pinned(receipt["path"], receipt["sha256"])
        acceptance = read(receipt["path"])
        if (acceptance["framework_files"] != actual or acceptance["engine_identity"] != p["engine_off_identity"]
                or acceptance["trials"] != {r["id"]: dict(task=r["task"], arm="verified") for r in trials() if r["arm"] == arm}
                or (acceptance["seconds"], acceptance["tools"], acceptance["model_output_tokens"], acceptance["thinking"], acceptance["advisor"]) != (1800, 250, 16384, True, False)):
            raise SafetyHalt("Runtime acceptance no longer matches the original task budgets")
    return p


def prepare(args):
    plan, build = plan_proof(args.plan)
    native = native_proof(args.native_qualification, plan)
    qualify(args.qualification)
    image.pinned(prior.__file__, PRIOR_SHA)
    d, _, _ = q.validate(args.parent_declaration)
    regpath = Path(d["feedback_registration"])
    reg = registration(regpath, args.parent_declaration, d)
    q.successor_queue(reg, args.parent_declaration)
    parent = q.frozen_protocol("feedback", Path(reg["protocol"]))
    original = read(plan["inherited_protocol"])
    acceptance = read(original["acceptance"])
    if acceptance["engine_identity"] != parent["engine_off_identity"]:
        raise SafetyHalt("Original SDK51 and actual parent engine profiles differ")
    output = args.output
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copyfile(args.plan, output / "plan.json")
    frameworks, acceptances = {}, {}
    for arm in ("control", "candidate"):
        destination = output / arm / "framework"
        shutil.copytree(ROOT / plan["candidate"] / arm, destination,
                        ignore=shutil.ignore_patterns("__pycache__"))
        for name in AUXILIARY:
            shutil.copyfile(Path(original["framework"]) / name, destination / name)
        a = dict(acceptance, framework_files=base.source_identity(destination), frozen_at=time.time(),
                 trials={r["id"]: dict(task=r["task"], arm="verified") for r in trials() if r["arm"] == arm},
                 scope="Matched bounded zero-tool continuation only; unchanged original task, GUI permissions, cumulative budget, recorder and evaluator.")
        acceptance_path = destination.parent / "acceptance.json"
        save(acceptance_path, a)
        frameworks[arm] = str(destination)
        acceptances[arm] = dict(path=str(acceptance_path), sha256=owned.digest(acceptance_path))
    checks = []
    for arm in ("control", "candidate"):
        environment = dict(os.environ, PYTHONPATH=frameworks[arm] + os.pathsep + str(ROOT / "apps/computer_use")
                           + os.pathsep + str(RUNTIME / "media-qualification"))
        imports = "cssselect,lxml.cssselect,numpy,av,aiohttp,osworld_office_candidate_run,qw3_computer.app,device_gateway_entry"
        if arm == "candidate": imports += ",qw3_computer.bounded_turn_continuation"
        command = [str(q.quota.PYTHON), "-c", "import json,sys;import " + imports
                   + ";print(json.dumps(dict(executable=sys.executable,app_source=qw3_computer.app.__file__,model_calls=0,device_inputs=0)))"]
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20, check=True)
        check = json.loads(result.stdout)
        if check["executable"] != str(q.quota.PYTHON) or Path(check["app_source"]) != Path(frameworks[arm]) / "qw3_computer/app.py":
            raise SafetyHalt("Runtime import did not load the declared private framework")
        checks.append(dict(arm=arm, **check))
    save(output / "runtime-qualification.json", dict(passed=True, checks=checks, model_calls=0, device_inputs=0))
    paths = [args.plan, output / "plan.json", args.parent_declaration, regpath, Path(reg["protocol"]),
             args.native_qualification, args.qualification, Path(prior.__file__),
             ROOT / plan["candidate"] / "build.json", Path(original["acceptance"]), output / "runtime-qualification.json"]
    paths += [ROOT / "apps/computer_use" / name for name in (*base.HELPERS, *image.EXTRA_HELPERS, *HELPERS)]
    paths += [Path(row["result"]) for row in native["rows"]]
    p = dict(kind="declared_bounded_continuation_original_osworld_retest", declared_at=time.time(),
             trials=trials(), model=copy.deepcopy(image.MODEL), model_concurrency=1, advisor_enabled=False,
             overwrite_old_attempts=False, plan=str(output / "plan.json"), qualification=str(args.qualification),
             native_qualification=str(args.native_qualification), parent_declaration=str(args.parent_declaration),
             parent_registration=str(regpath), parent_protocol=reg["protocol"], parent_queue=reg["queue"],
             parent_controller=reg["identity"], parent_app=parent["parent_app"], engine_off_identity=parent["engine_off_identity"],
             frameworks=frameworks, acceptances=acceptances, runner_python=str(q.quota.PYTHON),
             controller_python=str(q.CONTROLLER_PYTHON), real_model_trials_started=0,
             business_success_improvement_proven=False,
             files=[dict(path=str(path), sha256=owned.digest(path)) for path in dict.fromkeys(paths)])
    path = output / "protocol.json"
    save(path, p)
    validate(path)
    return path


def terminal_predecessor(p, state):
    d, canonical, mr = q.validate(Path(p["parent_declaration"]))
    sr = registration(Path(p["parent_registration"]), Path(p["parent_declaration"]), d)
    actual = q.successor_queue(sr, Path(p["parent_declaration"]))
    if actual != state:
        # Producer checkpoint may change during a read; retry the same handle.
        if actual["status"] != "completed":
            return None
        raise SafetyHalt("Completed predecessor state changed during admission")
    if state["status"] != "completed" or not base.process_retired(sr["identity"]):
        return None
    if (state.get("termination_confirmed") is not True or state.get("parent_services_preserved") is not True
            or state.get("final_engine_policy") != "off"):
        raise SafetyHalt("Original feedback final service handoff unconfirmed")
    receipts = []
    for stage in q.STAGES:
        reg = q.successor_registration(Path(d[stage + "_registration"]), stage, Path(p["parent_declaration"]), d)
        queue = q.successor_queue(reg, Path(p["parent_declaration"]))
        if queue["status"] != "completed" or not base.process_retired(reg["identity"]):
            raise SafetyHalt("The complete frozen predecessor chain has not retired")
        protocol = q.frozen_protocol(stage, Path(reg["protocol"]))
        if stage == "sdk": protocol = protocol[0]
        if [r["id"] for r in queue["trials"]] != [r["id"] for r in protocol["trials"]]:
            raise SafetyHalt("Predecessor terminal inventory changed")
        for spec, row in zip(protocol["trials"], queue["trials"]):
            out = Path(reg["protocol"]).parent / "attempts" / row["id"]
            result_path = out / ("integration-result.json" if spec.get("kind") in ("receipt-loss", "normal-pilot") else "result.json")
            result = read(result_path)
            if result != row["result"] or owned.digest(result_path) != row["result_sha256"]:
                raise SafetyHalt("A predecessor durable result changed")
            if stage == "thinking":
                if (result.get("termination_confirmed") is not True or result.get("tool_calls_executed") != 0
                        or result.get("business_success_claimed") is not False):
                    raise SafetyHalt("Thinking replay was not read-only and retired")
            else:
                base.terminal_trial(spec, result)
                if spec.get("kind") == "original-osworld": owned.assert_vm_retired(out)
            owned.assert_no_services(out)
            if row.get("evidence_sha256"):
                name = {"image": "visual-context-evidence.json", "feedback": "literal-feedback-evidence.json"}.get(stage)
                if name: image.pinned(out / name, row["evidence_sha256"])
            receipts.append(dict(stage=stage, id=row["id"], result_sha256=row["result_sha256"]))
    memory = q.memory_queue(mr, Path(d["memory_continuation"]))
    kp = q.quota.validate_protocol(Path(d["memory_protocol"]))
    if memory["status"] != "completed" or not base.process_retired(mr["identity"]):
        raise SafetyHalt("Fresh memory comparison not retired")
    receipts += base.exact_rows(kp["trials"], memory, Path(d["memory_protocol"]).parent, owned.digest(d["memory_protocol"]))
    for row in memory["trials"]:
        out = Path(d["memory_protocol"]).parent / "attempts" / row["id"]
        owned.assert_no_services(out)
        owned.assert_vm_retired(out)
    parent_state = q.old.continued_parent(canonical, owned.digest(d["original_followup"]), Path(d["original_followup"]))
    if parent_state["status"] != "completed" or not base.process_retired(canonical["continuation_controller_identity"]):
        raise SafetyHalt("Original forty-case controller has not retired")
    receipts += q.old.terminal_parent(kp, parent_state, Path(d["original_followup"]))
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
        raise SafetyHalt("Last actual feedback engine changed")
    services.update(registered_engine=engine, bounded_predecessor_receipts=receipts)
    return services


def task_evidence(output, arm):
    task = read(output / "app/task.json")
    if (task.get("thinking") is not True or task.get("model_output_tokens") != 16384
            or task.get("tool_call_limit") != 250 or task.get("execution_mode") != "verified"
            or task.get("model_context_window") != 524288):
        raise SafetyHalt("Actual task differs from its original cumulative local profile")
    saved = task.get("bounded_turn_continuation")
    if arm == "control":
        if saved is not None: raise SafetyHalt("Control unexpectedly enabled continuation")
        attempts = []
    else:
        from bounded_turn_continuation import POLICY
        if not isinstance(saved, dict) or saved.get("policy") != POLICY or len(saved.get("attempts", [])) > 2:
            raise SafetyHalt("Candidate continuation policy or bound differs")
        attempts = saved["attempts"]
        ids = [a.get("completed_turn_id") for a in attempts]
        if len(set(ids)) != len(ids) or any(a.get("thread_id") != task.get("thread_id") for a in attempts):
            raise SafetyHalt("Continuation task/thread identity changed")
    traces = {f.parent for pattern in ("*/request.json", "*/request.archive.json") for f in (output / "app/inference").glob(pattern)}
    if not traces: raise SafetyHalt("No actual model evidence for the bounded task")
    for trace in traces:
        body = json.loads(image.archive.read_request(trace))
        if body.get("enable_thinking") is not True or body.get("max_tokens") != 16384:
            raise SafetyHalt("Actual request reasoning/output profile differs")
    return dict(arm=arm, task_sha256=owned.digest(output / "app/task.json"), request_count=len(traces),
                additional_turns=len(attempts), deliveries=[a.get("delivery") for a in attempts],
                task_budget_reset=False, continuation_is_not_business_success=True)


class Controller(image.Controller):
    def __init__(self, args, protocol, state):
        super().__init__(args, protocol, state)
        self.state.update(kind="serial_matched_bounded_continuation_retest", status="waiting-for-forty-eight-ten-twelve-four-eight",
                          controller_sha256=owned.digest(__file__), bounded_plan_sha256=owned.digest(protocol["plan"]),
                          parent_declaration_sha256=owned.digest(protocol["parent_declaration"]),
                          business_success_improvement_proven=False)

    def trial_command(self, row, output):
        return [self.p["runner_python"], "-m", "osworld_office_candidate_run", "--task", row["task"],
                "--arm", "verified", "--output", str(output), "--framework", self.p["frameworks"][row["arm"]],
                "--engine-runtime", self.state["engine_runtime"], "--formal-campaign", str(RUNTIME / "linux/campaign-v2-20261004"),
                "--acceptance-manifest", self.p["acceptances"][row["arm"]]["path"], "--pilot-app", self.p["parent_app"],
                "--model-output-tokens", "16384"]

    def collect(self, row, output, code):
        result_path = output / "result.json"
        result = read(result_path)
        base.terminal_trial(dict(row, kind="original-osworld"), result)
        owned.assert_no_services(output)
        owned.assert_vm_retired(output)
        evidence = task_evidence(output, row["arm"])
        evidence.update(original_result_sha256=owned.digest(result_path), original_business_grade_unchanged=True)
        evidence_path = output / "bounded-continuation-evidence.json"
        if evidence_path.exists(): raise SafetyHalt("Existing evidence cannot be overwritten")
        save(evidence_path, evidence)
        self.state["trials"].append(dict(row, exit_code=code, result=result, result_sha256=owned.digest(result_path),
                                         evidence_sha256=owned.digest(evidence_path)))
        self.checkpoint(real_model_trials_completed=len(self.state["trials"]))

    def dispatch(self, row):
        validate(self.args.protocol)
        image.pinned(self.args.protocol, self.state["protocol_sha256"])
        self.parent_idle()
        output = self.args.protocol.parent / "attempts" / row["id"]
        if output.exists(): raise SafetyHalt("A bounded attempt cannot be overwritten or replayed")
        self.fresh_engine(row)
        self.checkpoint(status="running", current_trial=row["id"])
        child, _ = self.spawn(self.trial_command(row, output), row["id"], self.p["frameworks"][row["arm"]])
        code = child.wait(timeout=5100)
        self.collect(row, output, code)
        self.parent_idle()


def run(args):
    image.pinned(prior.__file__, PRIOR_SHA)
    with patch.object(prior, "validate", validate), patch.object(prior, "Controller", Controller), \
            patch.object(prior, "terminal_predecessor", terminal_predecessor):
        return prior.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    for name in ("plan", "native-qualification", "qualification", "parent-declaration", "output", "protocol", "state-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if not (60 <= args.wait_seconds <= 172800 and 60 <= args.resource_wait_seconds <= 86400 and args.min_free_mib >= 56000):
        parser.error("Keep the original bounded resource waits")
    keys = ("plan", "native_qualification", "qualification", "parent_declaration", "output") if args.operation == "prepare" else ("protocol", "state_dir")
    for key in keys:
        if getattr(args, key) is None: parser.error("Missing " + key)
        setattr(args, key, getattr(args, key).resolve())
    if args.operation == "prepare":
        print(json.dumps(dict(protocol=str(prepare(args)), model_calls=0, device_inputs=0)))
        return
    if args.state_dir != args.protocol.parent / "queue-runtime-01":
        raise SafetyHalt("Use the single declared fresh bounded queue directory")
    if os.fsdecode(os.fsencode(q.CONTROLLER_PYTHON)) != os.sys.executable:
        raise SafetyHalt("Keep the qualified controller interpreter")
    def interrupted(*_): raise SafetyHalt("Bounded comparison supervisor interrupted; live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(dict(status=run(args)["status"])))


if __name__ == "__main__": main()
