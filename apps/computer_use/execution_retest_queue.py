"""Admit an isolated execution-fix retest only after both frozen comparisons.

The successor preserves original attempts, packages and receipts. It replaces
only registered idle services, creates new runtime directories, and dispatches
the declared ten trials serially. A failed safety gate preserves live work.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener

import kvmem_osworld_queue as owned
from osworld_linux.runner import request, save, source_identity
from osworld_windows.bundle import verify_bundle
from osworld_windows.source import ROOT


SafetyHalt = owned.SafetyHalt
RUNTIME = ROOT / "apps/computer_use/.runtime"
HELPERS = ("receipt_model_pilot.py", "receipt_fault_proxy.py", "platform_pilots.py",
           "browser_pilot.py", "browser_pilot_server.js", "calc_pilot.py",
           "android_contacts_pilot.py", "windows_fault_fixture.py", "windows_fault_pilot.py",
           "fault_browser_pilot.py", "fault_browser_server.js", "device_receipt_qualification.py",
           "linux_recording.py", "live_android_recording.py", "vm/record_console.py",
           "vm/vnc_admin.py", "run_engine.py", "device_gateway_entry.py",
           "office_device_gateway_entry.py", "bounded_linux_observation.py", "linux_app_semantics.py",
           "osworld_candidate_run.py", "osworld_office_candidate_run.py", "kvmem_osworld_queue.py")


def read(path):
    return json.loads(Path(path).read_text())


def require_identity(expected):
    actual = owned.process_identity(expected["pid"])
    if not owned.same_process(expected, actual) or expected["uid"] != os.getuid() or actual["state"] == "Z":
        raise SafetyHalt("Registered service changed; its replacement is preserved")
    return actual


def process_retired(expected):
    actual = owned.process_identity(expected["pid"])
    if actual is None or actual["state"] == "Z":
        return True
    if not owned.same_process(expected, actual):
        raise SafetyHalt("Predecessor PID was reused; no successor dispatch")
    return False


def ledger_terminal(runtime):
    path = Path(runtime) / "receipts.sqlite"
    with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT status,COUNT(*) FROM actions GROUP BY status").fetchall()
    counts = dict(rows)
    if set(counts) - {"completed", "rejected"}:
        raise SafetyHalt("Gateway has pending or unknown input; receipt history preserved")
    return counts


def gateway_api(item, route, body=None):
    # Authentication stays inside the request; no credential is saved in logs.
    token = Path(item["token_file"]).read_text().strip()
    req = Request(item["endpoint"] + route,
                  data=None if body is None else json.dumps(body).encode(),
                  headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(req, timeout=30) as response:
        return json.load(response)


def find_client(runtime, port, entries=None):
    matches = []
    for entry in Path("/proc").iterdir() if entries is None else entries:
        if not entry.name.isdigit() or int(entry.name) <= 1:
            continue
        try:
            identity = owned.process_identity(int(entry.name))
            command = identity["command"] if identity else []
            if (identity and identity["uid"] == os.getuid() and identity["cwd"] == str(ROOT)
                    and identity["state"] != "Z" and "qw3_computer.app" in command
                    and "--runtime" in command and "--port" in command
                    and command[command.index("--port") + 1] == str(port)
                    and Path(command[command.index("--runtime") + 1]).resolve() == runtime):
                matches.append(identity)
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    if len(matches) != 1:
        raise SafetyHalt("Previous model client identity is not unique")
    return matches[0]


def exact_rows(declared, state, directory, protocol_sha):
    rows = state.get("trials", [])
    ids = {row["id"] for row in declared}
    if (state.get("status") != "completed" or state.get("protocol_sha256") != protocol_sha
            or len(rows) != len(declared) or len(ids) != len(declared)
            or {row["id"] for row in rows} != ids):
        raise SafetyHalt("Predecessor trial inventory is incomplete or changed")
    receipts = []
    for row in rows:
        path = Path(directory) / "attempts" / row["id"] / "result.json"
        result = read(path)
        if (result != row["result"] or owned.digest(path) != row.get("result_sha256")
                or not result.get("finished_at") or type(result.get("uncertain_actions")) is not int
                or result["uncertain_actions"] != 0 or result.get("owned_task_stop_unconfirmed")):
            raise SafetyHalt("Predecessor receipt is altered, nonterminal or uncertain")
        receipts.append({"id": row["id"], "result_sha256": owned.digest(path)})
    return receipts


def prepare(path):
    """Freeze sources without stopping services or submitting model requests."""
    p = read(path)
    if p.get("status") != "prepared-not-dispatched" or p.get("framework"):
        raise SafetyHalt("Preparation is allowed exactly once before declaration")
    bundle = Path(p["bundle"])
    qualified = verify_bundle(bundle)
    if (qualified["archive"]["sha256"] != p["bundle_sha256"]
            or owned.digest(bundle / "qualification.json") != p["qualification_sha256"]):
        raise SafetyHalt("Native qualification identity changed")
    framework = path.parent / "framework"
    shutil.copytree(bundle / "app/qw3_computer", framework / "qw3_computer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    kp = owned.validate_protocol((ROOT / p["parent_kvmem_protocol"]).resolve())
    original = Path(kp["framework"])
    for name in ("device_gateway_entry.py", "bounded_linux_observation.py", "linux_app_semantics.py"):
        shutil.copyfile(original / name, framework / name)
    old_acceptance = read((ROOT / p["parent_kvmem_protocol"]).parent / "off/acceptance.json")
    a = dict(old_acceptance, framework_files=source_identity(framework), frozen_at=time.time(),
             scope="Isolated SDK51 development retest; original task, initializer, grader and budget unchanged.",
             trials={r["id"]: {"task": r["task"], "arm": r["arm"]}
                     for r in p["trials"] if r["kind"] == "original-osworld"})
    acceptance = path.parent / "acceptance.json"
    save(acceptance, a)
    app_runtime = RUNTIME / "cross-device-ui-20"
    if app_runtime.exists():
        raise SafetyHalt("New client runtime already exists; never overwrite")
    app_runtime.mkdir()
    inventory = read(bundle / "manifest.json")["source_files"]
    files = {n.removeprefix("qw3_computer/"): value for n, value in inventory.items()
             if n.startswith("qw3_computer/")}
    manifest = {"kind": "isolated-qualified-execution-fix-candidate", "files": files,
                "native_bundle": str(bundle), "native_qualification": qualified,
                "candidate_changes": p["candidate_changes"], "framework": str(framework),
                "model_output_tokens": 16384, "real_model_qualification": False,
                "original_frozen_packages_unchanged": True, "frozen_at": time.time()}
    # Receipt pilot expects framework beneath the UI runtime. A read-only source
    # copy keeps all downstream identities explicit, without modifying SDK51.
    shutil.copytree(framework, app_runtime / "framework")
    save(app_runtime / "candidate-manifest.json", manifest)
    from qw3_computer.device_registry import load_registry
    from device_receipt_qualification import owned_gateway
    catalogue = RUNTIME / "cross-device-catalogue.json"
    devices = load_registry(catalogue)
    registrations = {}
    for key in ("linux-vm", "android-emulator"):
        item = devices[key]
        found = owned_gateway(item)
        identity = owned.process_identity(found["pid"])
        require_identity(identity)
        registrations[key] = {"identity": identity, "item": dict(item, token_file=str(item["token_file"]))}
    app_state = request("http://127.0.0.1:18213", "/api/state")
    owned.assert_idle("http://127.0.0.1:18213")
    previous_client = find_client(RUNTIME / "cross-device-ui-19", 18213)
    osq = read(ROOT / p["parent_osworld_queue"])
    kvq = read(ROOT / p["parent_kvmem_queue"])
    p.update(status="declared-waiting-for-predecessors", framework=str(framework),
             acceptance=str(acceptance), acceptance_sha256=owned.digest(acceptance),
             app_runtime=str(app_runtime), app="http://127.0.0.1:18215",
             previous_app="http://127.0.0.1:18213", previous_app_identity=previous_client,
             previous_gateways=registrations, previous_selected_device=app_state["devices"]["selected"],
             previous_catalogue_sha256=owned.digest(catalogue),
             parent_kvmem_protocol_sha256=owned.digest(ROOT / p["parent_kvmem_protocol"]),
             candidate_manifest_sha256=owned.digest(app_runtime / "candidate-manifest.json"),
             predecessor_processes={"osworld": owned.process_identity(osq["pid"]),
                                    "kvmem": owned.process_identity(kvq["pid"])},
             engine_off_identity=kp["engine_identities"]["off"],
             helper_sha256={n: owned.digest(ROOT / "apps/computer_use" / n) for n in HELPERS},
             controller_sha256=owned.digest(Path(__file__)))
    for identity in p["predecessor_processes"].values():
        require_identity(identity)
    save(path, p)
    return p


def validate(path):
    p = read(path)
    if (p.get("kind") != "declared_isolated_execution_fix_development_retest"
            or p.get("status") != "declared-waiting-for-predecessors"
            or (p.get("model_output_tokens"), p.get("model_context_tokens"),
                p.get("ordinary_tasks_tools")) != (16384, 524288, 100)
            or p.get("thinking") is not True or p.get("dflash2") is not True
            or p.get("kvmem") != "off" or p.get("gui_only") is not True
            or p.get("advisor") is not False or p.get("software") is not False
            or p.get("prefix_reuse") is not False or p.get("visual_history") != "pin-all"
            or p.get("model_concurrency") != 1 or p.get("input_owner_limit_per_device") != 1
            or len(p.get("trials", [])) != 10 or len({r["id"] for r in p["trials"]}) != 10):
        raise SafetyHalt("Execution retest declaration or local model profile changed")
    expected = [("receipt-loss", "linux", None), ("receipt-loss", "linux", None),
                ("normal-pilot", "linux", "linux_text"), ("normal-pilot", "linux", "linux_text"),
                ("normal-pilot", "linux", "linux_calc"), ("normal-pilot", "linux", "linux_calc"),
                ("receipt-loss", "windows", None), ("receipt-loss", "android", None),
                ("original-osworld", "linux", None), ("original-osworld", "linux", None)]
    if [(r["kind"], r["platform"], r.get("case")) for r in p["trials"]] != expected:
        raise SafetyHalt("The declared ten retest identities or order changed")
    for row in p["trials"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{8,45}", row["id"]):
            raise SafetyHalt("Unsafe trial identity")
        if row["kind"] == "receipt-loss" and row["device"] != {
                "linux": "linux-vm", "windows": "windows-vm", "android": "android-emulator"}[row["platform"]]:
            raise SafetyHalt("Receipt pilot device identity changed")
        if row["kind"] == "normal-pilot" and row["seconds"] != (360 if row["case"] == "linux_text" else 900):
            raise SafetyHalt("Pilot budget changed")
        if row["kind"] == "original-osworld" and (row["task"], row["arm"], row["seconds"], row["tools"]) != ("004", "verified", 1800, 250):
            raise SafetyHalt("Original task or budget changed")
    for name, value in p["helper_sha256"].items():
        if owned.digest(ROOT / "apps/computer_use" / name) != value:
            raise SafetyHalt("Frozen retest helper changed: " + name)
    if owned.digest(Path(__file__)) != p["controller_sha256"]:
        raise SafetyHalt("Controller changed after declaration")
    kp_path = (ROOT / p["parent_kvmem_protocol"]).resolve()
    if owned.digest(kp_path) != p["parent_kvmem_protocol_sha256"]:
        raise SafetyHalt("Parent memory protocol changed")
    kp = owned.validate_protocol(kp_path)
    q = verify_bundle(Path(p["bundle"]))
    if (q["archive"]["sha256"] != p["bundle_sha256"]
            or q["qualification_sha256"] != p["qualification_sha256"]):
        raise SafetyHalt("Qualified bundle changed")
    a = read(p["acceptance"])
    if (owned.digest(p["acceptance"]) != p["acceptance_sha256"]
            or a["framework_files"] != source_identity(Path(p["framework"]))
            or a["engine_identity"] != kp["engine_identities"]["off"]
            or a["trials"] != {r["id"]: {"task": r["task"], "arm": r["arm"]}
                               for r in p["trials"] if r["kind"] == "original-osworld"}):
        raise SafetyHalt("Isolated OSWorld source admission changed")
    client = Path(p["app_runtime"])
    for framework in (Path(p["framework"]), client / "framework"):
        if owned.digest(framework / "device_gateway_entry.py") != a["canonical_entry_sha256"]:
            raise SafetyHalt("Private canonical gateway entry changed")
        for name, value in a["framework_auxiliary_files"].items():
            if owned.digest(framework / name) != value:
                raise SafetyHalt("Private observation helper changed")
    candidate_path = client / "candidate-manifest.json"
    expected_files = {n.removeprefix("qw3_computer/"): value
                      for n, value in read(Path(p["bundle"]) / "manifest.json")["source_files"].items()
                      if n.startswith("qw3_computer/")}
    candidate = read(candidate_path)
    if (owned.digest(candidate_path) != p["candidate_manifest_sha256"]
            or candidate["files"] != expected_files):
        raise SafetyHalt("Qualified client source inventory changed")
    for name, value in candidate["files"].items():
        if (owned.digest(client / "framework/qw3_computer" / name) != value
                or owned.digest(Path(p["framework"]) / "qw3_computer" / name) != value):
            raise SafetyHalt("Private qualified client source changed")
    return p, kp


def predecessors_ready(p, kp):
    osq = read(ROOT / p["parent_osworld_queue"])
    kvq = read(ROOT / p["parent_kvmem_queue"])
    for state in (osq, kvq):
        if state["status"].startswith("halted"):
            raise SafetyHalt("A predecessor halted; no successor service change")
    if osq["status"] != "completed" or kvq["status"] != "completed":
        return None
    os_receipts = owned.terminal_parent(kp, osq)
    kv_dir = (ROOT / p["parent_kvmem_protocol"]).parent
    kv_receipts = exact_rows(kp["trials"], kvq, kv_dir, p["parent_kvmem_protocol_sha256"])
    if not all(process_retired(identity) for identity in p["predecessor_processes"].values()):
        return None
    for directory, rows in (((ROOT / p["parent_osworld_queue"]).parent, osq["trials"]),
                            (kv_dir, kvq["trials"])):
        owned.assert_no_services(directory / "attempts")
        for row in rows:
            owned.assert_vm_retired(directory / "attempts" / row["id"])
    if kvq.get("final_engine_policy") != "on":
        raise SafetyHalt("Predecessor final engine declaration changed")
    engine = owned.register_engine(Path(kvq["current_engine_runtime"]), kp["engine_identities"]["on"])
    if not owned.same_process(kvq["final_registered_engine"], engine):
        raise SafetyHalt("Last registered predecessor engine changed")
    return {"osworld": os_receipts, "kvmem": kv_receipts, "registered_engine": engine}


class Controller:
    def __init__(self, args, protocol, state):
        self.args, self.p, self.state = args, protocol, state
        self.children = []

    def checkpoint(self, **values):
        self.state.update(values)
        save(self.args.state_dir / "queue.json", self.state)

    def spawn(self, command, name, framework=None):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(framework) if framework else None,
                   str(ROOT / "apps/computer_use"), str(RUNTIME / "media-qualification")])))
        with (self.args.state_dir / (name + ".log")).open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        identity = owned.register_child(child.pid, command)
        self.children.append(child)
        self.checkpoint(current_child=identity)
        return child, identity

    def ready(self, child, getter, predicate, seconds=120):
        deadline = time.monotonic() + seconds
        while True:
            if child.poll() is not None:
                raise SafetyHalt("New owned service exited before readiness")
            try:
                value = getter()
            except (OSError, ValueError, KeyError):
                value = None
            if value is not None and predicate(value):
                return value
            if time.monotonic() >= deadline:
                raise SafetyHalt("New owned service readiness remains unconfirmed")
            time.sleep(.5)

    def restore_engine(self, previous):
        with owned.model_lock():
            owned.assert_apps_idle(self.p["previous_app"])
            self.checkpoint(status="restoring-fresh-off-engine",
                            engine_retirement=owned.retire_registered(previous))
            deadline = time.monotonic() + self.args.resource_wait_seconds
            while True:
                gpu = owned.gpu_snapshot()
                self.checkpoint(last_gpu_snapshot=gpu)
                if gpu["free_mib"] >= self.args.min_free_mib:
                    break
                if time.monotonic() >= deadline:
                    raise SafetyHalt("Shared GPU headroom unavailable; other workloads preserved")
                time.sleep(5)
            runtime = self.args.protocol.parent / "engine-off"
            command = [sys.executable, str(ROOT / "apps/computer_use/run_engine.py"), "--run-dir", str(runtime),
                       "--method", "dflash2", "--kvmem", "off", "--prefix-reuse", "off",
                       "--visual-history", "pin-all", "--guided-refresh", "off", "--context-tokens", "524288",
                       "--gpu-sharing", "spare-capacity", "--min-free-mib", str(self.args.min_free_mib),
                       "--lock-timeout", "5"]
            child, _ = self.spawn(command, "engine-launch")
            self.ready(child, lambda: request("http://127.0.0.1:18080", "/v1/models", timeout=3),
                       lambda v: v["data"][0].get("context_window") == 524288, 600)
            identity = owned.register_engine(runtime, self.p["engine_off_identity"])
            if identity["pid"] != child.pid:
                raise SafetyHalt("Fresh engine launcher identity differs")
            self.checkpoint(engine_runtime=str(runtime), registered_engine=identity)

    def deploy(self):
        p = self.p
        with owned.model_lock():
            owned.assert_apps_idle(p["previous_app"])
            if owned.digest(RUNTIME / "cross-device-catalogue.json") != p["previous_catalogue_sha256"]:
                raise SafetyHalt("Previous catalogue changed; service handoff refused")
            require_identity(p["previous_app_identity"])
            for registration in p["previous_gateways"].values():
                require_identity(registration["identity"])
                ledger_terminal(Path(registration["item"]["token_file"]).parent)
            from platform_pilots import owned_container
            containers = {}
            for platform in ("windows", "linux", "android"):
                label, owner = (("com.qw3.computer-use.owner", "codex-goal-20260930") if platform == "windows"
                                else ("qw3.goal", "cross-device-20261003"))
                container = owned_container("qw3-computer-" + platform, label, owner)
                if not container["State"]["Running"]:
                    raise SafetyHalt("Required owned device VM is stopped")
                containers[platform] = container["Id"]
            self.checkpoint(status="handing-off-paused-services", containers=containers)
            save(self.args.state_dir / "previous-catalogue.json", read(RUNTIME / "cross-device-catalogue.json"))
            retired = {"app": owned.retire_registered(p["previous_app_identity"])}
            devices = []
            for key, registration in p["previous_gateways"].items():
                item = registration["item"]
                paused = gateway_api(item, "/control", {"command": "pause"})
                if paused.get("paused") is not True:
                    raise SafetyHalt("Old gateway pause is unconfirmed")
                counts = ledger_terminal(Path(item["token_file"]).parent)
                retired[key] = dict(owned.retire_registered(registration["identity"]), receipt_counts=counts)
                platform = item["platform"]
                runtime = RUNTIME / platform / "gateway-04"
                if runtime.exists():
                    raise SafetyHalt("New gateway runtime already exists; history cannot be overwritten")
                command = [str(ROOT / ".venv/bin/python"), "-m", "device_gateway_entry", "--platform", platform,
                           "--driver-endpoint", "http://127.0.0.1:" + ("18126" if platform == "linux" else "18203"),
                           "--device-id", registration["identity"]["command"][registration["identity"]["command"].index("--device-id") + 1],
                           "--runtime", str(runtime), "--port", "18130" if platform == "linux" else "18204"]
                if platform == "android":
                    command += ["--android-serial", "emulator-5556"]
                child, identity = self.spawn(command, platform + "-gateway", p["framework"])
                new_item = dict(item, token_file=str(runtime / "token.txt"))
                observed = self.ready(child, lambda: gateway_api(new_item, "/observe"),
                                      lambda v: v.get("paused") is True)
                devices.append({"catalogue_id": key, "pid": identity["pid"], "command": command,
                                "process_identity": identity, "device_id": observed["device_id"],
                                "session_id": observed["session_id"], "paused": True})
                self.checkpoint(new_gateways=devices, service_retirements=retired)
            catalogue = read(self.args.state_dir / "previous-catalogue.json")
            for row in catalogue["devices"]:
                if row["id"] in p["previous_gateways"]:
                    row["token_file"] = str(RUNTIME / row["platform"] / "gateway-04/token.txt")
            registry = self.args.protocol.parent / "registry.json"
            save(registry, catalogue)
            registry.chmod(0o600)
            # The original normal-pilot helper explicitly reads this catalogue.
            # Handoff occurs only after all predecessors have terminal receipts.
            save(RUNTIME / "cross-device-catalogue.json", catalogue)
            self.checkpoint(registry=str(registry), registry_sha256=owned.digest(registry))
            runtime = Path(p["app_runtime"])
            declaration = {"kind": "separate_candidate_device_dependency_upgrade", "devices": devices,
                           "entry_sha256": p["helper_sha256"]["device_gateway_entry.py"],
                           "source_sha256": {"apps/computer_use/" + name: value
                                              for name, value in p["helper_sha256"].items()},
                           "candidate_framework": p["framework"], "frozen_formal_unchanged": True}
            save(runtime / "device-dependencies-canonical-20261004.json", declaration)
            command = [str(ROOT / ".venv/bin/python"), "-m", "qw3_computer.app", "--port", "18215",
                       "--runtime", str(runtime), "--device-registry", str(registry), "--selected-device", "windows-vm",
                       "--execution-mode", "verified", "--thinking", "--model-output-tokens", "16384",
                       "--tool-call-limit", "100", "--output-policy", "task"]
            child, identity = self.spawn(command, "model-client", runtime / "framework")
            self.ready(child, lambda: request(p["app"], "/api/state", timeout=3),
                       lambda v: v.get("ready") is True and v.get("status") == "idle"
                       and v.get("desktop_paused") is True and not v.get("uncertain_actions")
                       and v.get("thinking") is True and v.get("model_output_tokens") == 16384
                       and v.get("tool_call_limit") == 100)
            self.checkpoint(status="admitted", model_client=identity)

    def trial_command(self, row, output):
        common = [sys.executable, "-m"]
        p = self.p
        if row["kind"] == "receipt-loss":
            return common + ["receipt_model_pilot", "--app", p["app"], "--runtime", p["app_runtime"],
                             "--registry", self.state["registry"], "--device", row["device"],
                             "--bundle", p["bundle"], "--output", str(output), "--seconds", "900"]
        if row["kind"] == "normal-pilot":
            return common + ["platform_pilots", "--case", row["case"], "--seconds", str(row["seconds"]),
                             "--app", p["app"], "--runtime", p["app_runtime"], "--output", str(output),
                             "--retire-task", "--linux-video-mode", "vnc"]
        return common + ["osworld_office_candidate_run", "--task", row["task"], "--arm", row["arm"],
                         "--output", str(output), "--framework", p["framework"], "--engine-runtime", self.state["engine_runtime"],
                         "--formal-campaign", str(RUNTIME / "linux/campaign-v2-20261004"),
                         "--acceptance-manifest", p["acceptance"], "--pilot-app", p["app"], "--model-output-tokens", "16384"]

    def dispatch(self, row):
        validate(self.args.protocol)
        if owned.digest(self.args.protocol) != self.state["protocol_sha256"]:
            raise SafetyHalt("Declared successor protocol changed")
        owned.assert_apps_idle(self.p["app"])
        require_identity(self.state["model_client"])
        owned.register_engine(Path(self.state["engine_runtime"]), self.p["engine_off_identity"])
        owned.assert_no_services(self.args.protocol.parent / "attempts")
        if owned.digest(self.state["registry"]) != self.state["registry_sha256"]:
            raise SafetyHalt("Declared successor device registry changed")
        output = self.args.protocol.parent / "attempts" / row["id"]
        if output.exists():
            raise SafetyHalt("Existing attempt cannot be replayed")
        command = self.trial_command(row, output)
        def execute():
            self.checkpoint(status="running", current_trial=row["id"])
            child, _ = self.spawn(command, row["id"], self.p["framework"])
            code = child.wait(timeout=5100 if row["kind"] == "original-osworld" else 1500)
            path = output / ("result.json" if row["kind"] == "original-osworld" else "integration-result.json")
            result = read(path)
            terminal_trial(row, result)
            owned.assert_no_services(output)
            if row["kind"] == "original-osworld":
                owned.assert_vm_retired(output)
            owned.assert_idle(self.p["app"])
            self.state["trials"].append({"id": row["id"], "kind": row["kind"], "exit_code": code,
                                         "result_sha256": owned.digest(path), "result": result})
            self.checkpoint()
        # Receipt and OSWorld helpers independently claim the queue. The normal
        # helper has no queue lock, so the supervisor holds it for that child.
        if row["kind"] == "normal-pilot":
            with owned.model_lock():
                execute()
        else:
            execute()


def terminal_trial(row, result):
    if (type(result.get("uncertain_actions")) is not int or result["uncertain_actions"] != 0
            or result.get("owned_task_stop_unconfirmed")):
        raise SafetyHalt("Trial has unconfirmed input; no next trial")
    if row["kind"] == "original-osworld":
        if not result.get("finished_at"):
            raise SafetyHalt("Original trial is not durably terminal")
    elif result.get("owned_task_retired") is not True:
        raise SafetyHalt("Model task retirement is unconfirmed")
    if row["kind"] == "receipt-loss" and (result.get("administrative_recovery") is not True
            or result.get("unassisted_recovery_claimed") is not False or result.get("accepted") is not False):
        raise SafetyHalt("Administrative receipt recovery cannot become business success")


def run(args):
    p, kp = validate(args.protocol)
    args.state_dir.mkdir(parents=True, exist_ok=False)
    state = {"kind": "isolated_execution_fix_serial_retest", "pid": os.getpid(), "started_at": time.time(),
             "status": "waiting-for-forty-and-eight", "trials": [], "planned_trials": 10,
             "protocol_sha256": owned.digest(args.protocol), "controller_sha256": owned.digest(Path(__file__)),
             "real_model_trials_completed": 0, "model_concurrency": 1}
    controller = Controller(args, p, state)
    controller.checkpoint()
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            previous = predecessors_ready(p, kp)
            if previous is not None:
                break
            if time.monotonic() >= deadline:
                raise SafetyHalt("Predecessor wait deadline reached; all services preserved")
            time.sleep(3)
        validate(args.protocol)
        controller.checkpoint(predecessor_receipts=previous)
        controller.restore_engine(previous["registered_engine"])
        controller.deploy()
        for row in p["trials"]:
            controller.dispatch(row)
            controller.checkpoint(real_model_trials_completed=len(state["trials"]))
        controller.checkpoint(status="completed", finished_at=time.time(), termination_confirmed=True,
                              user_application=p["app"], new_services_preserved=True)
    except BaseException as exc:
        controller.checkpoint(status="halted-no-further-dispatch", finished_at=time.time(),
                              error=type(exc).__name__ + ": " + str(exc), live_children_and_evidence_preserved=True)
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "run"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    if args.operation == "prepare":
        p = prepare(args.protocol)
        print(json.dumps({"status": p["status"], "declared_trials": len(p["trials"])}))
        return
    if args.state_dir is None or args.min_free_mib < 56000 or not (60 <= args.wait_seconds <= 172800
                                                               and 60 <= args.resource_wait_seconds <= 86400):
        parser.error("Specify a new state directory and bounded shared-resource wait")
    args.state_dir = args.state_dir.resolve()
    def interrupted(*_):
        raise SafetyHalt("Successor interrupted; live work preserved")
    signal.signal(signal.SIGTERM, interrupted)
    state = run(args)
    print(json.dumps({"status": state["status"], "trials": len(state["trials"])}))


if __name__ == "__main__":
    main()
