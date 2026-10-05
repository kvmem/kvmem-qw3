"""Continue the exact SDK deployment interrupted by the Android OOM.

Reuse the proven owned OFF engine and paused Linux gateway. Create a fresh
Android gateway, then use the original client and frozen trial loops. Failed
controllers, the failed Android runtime and all original task records remain.
"""
import copy
import errno
from pathlib import Path
import socket
import time
from unittest.mock import patch

import shared_quota_handoff_successors as previous
from platform_pilots import owned_container

ROOT, owned, SafetyHalt = previous.ROOT, previous.owned, previous.SafetyHalt
MODULE = "shared_quota_staged_successors"
PREVIOUS_SHA = "dd2f4b21ab42e99316873da62e8c33e6e0642c4771f5df598a3cf4e6c1afe9c8"
AUDIT = ROOT / "apps/computer_use/.runtime/shared-quota-current-progress-20261005-23.json"
AUDIT_SHA = "b325e3ee44254cf9a631baacc584c5727d0c7ffd9b3500654c29d6275de277c2"
FAILED_DECLARATION = ROOT / "apps/computer_use/.runtime/shared-quota-handoff-successors-20261005-01/successors.json"
HELPERS = (MODULE + ".py", "tests/test_" + MODULE + ".py",
           "shared_quota_handoff_successors.py", "tests/test_shared_quota_handoff_successors.py",
           "tests/test_shared_quota_port_successors.py", "tests/test_shared_quota_terminal_handoff.py")


def qualification(path):
    result = previous.previous.port.read(path)
    if (result.get("kind") != "staged_sdk_successors_cpu_qualification" or result.get("passed") is not True
            or result.get("tests_run", 0) < 57 or result.get("skipped") != 0
            or result.get("model_calls") != 0 or result.get("device_inputs") != 0):
        raise SafetyHalt("Require the complete staged SDK recovery qualification")
    for name in HELPERS:
        previous.RUNTIME.image.pinned(ROOT / "apps/computer_use" / name, result["source_sha256"][name])


def stage_scope(*, live=False):
    previous.RUNTIME.image.pinned(AUDIT, AUDIT_SHA)
    audit = previous.RUNTIME.read(AUDIT)
    if (audit.get("kind") != "actual_handoff_port_ready_android_oom_and_service_recovery_audit"
            or set(audit.get("successors", {})) != set(previous.RUNTIME.STAGES)
            or audit.get("catalogue_not_updated_before_failure") is not True):
        raise SafetyHalt("Use only the recorded zero-trial partial SDK deployment")
    declaration, _, _ = previous.RUNTIME.validate(FAILED_DECLARATION)
    states = {}
    for stage, row in audit["successors"].items():
        previous.RUNTIME.image.pinned(row["registration"], row["registration_sha256"])
        previous.RUNTIME.image.pinned(row["queue"], row["queue_sha256"])
        record = previous.RUNTIME.successor_registration(Path(row["registration"]), stage, FAILED_DECLARATION, declaration)
        state = previous.RUNTIME.read(row["queue"])
        if (state.get("trials") != [] or state.get("status") != "halted-no-further-dispatch"
                or state.get("pid") != record["identity"]["pid"]
                or state.get("protocol_sha256") != record["protocol_sha256"]
                or state.get("runtime_successor_wrapper_sha256") != PREVIOUS_SHA
                or not previous.RUNTIME.base.process_retired(record["identity"])):
            raise SafetyHalt("Partial deployment producer changed or dispatched a task")
        states[stage] = state
    sdk = states["sdk"]
    partial = audit["sdk_partial_service_checkpoint"]
    if (any(sdk.get(key) != value for key, value in partial.items())
            or any(key in sdk for key in ("model_client", "registry", "registry_sha256"))
            or len(partial["new_gateways"]) != 1
            or partial["new_gateways"][0].get("catalogue_id") != "linux-vm"
            or partial["new_gateways"][0].get("paused") is not True
            or not previous.RUNTIME.base.process_retired(partial["current_child"])):
        raise SafetyHalt("The SDK did not stop at the declared Android startup boundary")
    protocol, _ = previous.RUNTIME.frozen_protocol("sdk", declaration["sdk_protocol"])
    for key, identity in [("app", protocol["previous_app_identity"]),
                          *[(key, value["identity"]) for key, value in protocol["previous_gateways"].items()]]:
        receipt = partial["service_retirements"][key]
        if receipt.get("pid") != identity["pid"] or receipt.get("retired") is not True or not previous.RUNTIME.base.process_retired(identity):
            raise SafetyHalt("Original service retirement is not proven; no recovery replay")
    recovery = audit["android_recovery"]
    for key in ("oom_and_first_recovery", "stale_lock_recovery"):
        record = recovery[key]; previous.RUNTIME.image.pinned(record["path"], record["sha256"])
    if live:
        engine = owned.register_engine(Path(partial["engine_runtime"]), protocol["engine_off_identity"])
        if not owned.same_process(partial["registered_engine"], engine):
            raise SafetyHalt("Partial deployment engine was replaced")
        previous.RUNTIME.base.require_identity(partial["new_gateways"][0]["process_identity"])
        container = owned_container("qw3-computer-android", "qw3.goal", "cross-device-20261003")
        if (container["Id"] != partial["containers"]["android"] or not container["State"]["Running"]
                or container["State"]["StartedAt"] != recovery["actual_state"]["StartedAt"]
                or container["HostConfig"]["Memory"] != 12 * 1024**3):
            raise SafetyHalt("Qualified Android recovery identity or capacity changed")
        status = previous.RUNTIME.base.request("http://127.0.0.1:18203", "/status", timeout=5)
        if status.get("value", {}).get("ready") is not True:
            raise SafetyHalt("Recovered Android service is not ready")
    return dict(audit=audit, protocol=protocol, partial=partial)


def staged_restore(controller, old_engine):
    with owned.model_lock():
        scope = stage_scope(live=True)
        predecessor = previous.failure_scope()["final_engine"]
        if old_engine != predecessor:
            raise SafetyHalt("Staged recovery received a different memory predecessor")
        partial = scope["partial"]
        controller.checkpoint(status="reusing-qualified-off-engine", engine_runtime=partial["engine_runtime"],
                              registered_engine=copy.deepcopy(partial["registered_engine"]),
                              engine_reuse_proven=True, staged_recovery_audit_sha256=AUDIT_SHA)


def admit_service_port(port):
    if port not in (18204, 18215):
        raise SafetyHalt("Keep the original Android gateway and model client ports")
    deadline = time.monotonic() + 120
    while True:
        try:
            with socket.socket() as probe: probe.bind(("127.0.0.1", port))
            return dict(port=port, bind_check_passed=True, bounded_wait_seconds=120, other_processes_signalled=0)
        except OSError as error:
            if error.errno != errno.EADDRINUSE or time.monotonic() >= deadline:
                raise SafetyHalt("Original service port unavailable; no new child launched") from error
            time.sleep(.2)


def staged_deploy(controller):
    p, base = controller.p, RUNTIME.base
    with owned.model_lock():
        scope = stage_scope(live=True)
        partial = scope["partial"]
        if owned.digest(base.RUNTIME / "cross-device-catalogue.json") != p["previous_catalogue_sha256"]:
            raise SafetyHalt("Original catalogue changed before partial deployment recovery")
        registry = controller.args.protocol.parent / "registry.json"
        android_runtime = base.RUNTIME / "android/gateway-05"
        runtime = Path(p["app_runtime"])
        dependency_path = runtime / "device-dependencies-canonical-20261004.json"
        if registry.exists() or android_runtime.exists() or dependency_path.exists():
            raise SafetyHalt("A staged deployment runtime already exists; preserve it")
        controller.checkpoint(service_port_admission=[admit_service_port(port) for port in (18204, 18215)])
        containers = {}
        for platform in ("windows", "linux", "android"):
            label, owner = (("com.qw3.computer-use.owner", "codex-goal-20260930") if platform == "windows"
                            else ("qw3.goal", "cross-device-20261003"))
            item = owned_container("qw3-computer-" + platform, label, owner)
            if not item["State"]["Running"] or item["Id"] != partial["containers"][platform]:
                raise SafetyHalt("A qualified owned device VM changed or stopped")
            containers[platform] = item["Id"]
        linux = copy.deepcopy(partial["new_gateways"][0])
        linux_item = dict(p["previous_gateways"]["linux-vm"]["item"],
                          token_file=str(base.RUNTIME / "linux/gateway-04/token.txt"))
        base.require_identity(linux["process_identity"])
        observed = base.gateway_api(linux_item, "/observe")
        if (observed.get("paused") is not True or observed.get("session_id") != linux["session_id"]
                or observed.get("device_id") != linux["device_id"]):
            raise SafetyHalt("Preserved Linux gateway session or pause state changed")
        base.ledger_terminal(base.RUNTIME / "linux/gateway-04")
        controller.checkpoint(status="recovering-android-gateway", containers=containers,
                              new_gateways=[linux], service_retirements=copy.deepcopy(partial["service_retirements"]),
                              linux_gateway_reused=True, original_service_retirements_not_replayed=True)
        command = [str(RUNTIME.CONTROLLER_PYTHON), "-m", "device_gateway_entry", "--platform", "android",
                   "--driver-endpoint", "http://127.0.0.1:18203", "--device-id", "android-api35",
                   "--runtime", str(android_runtime), "--port", "18204", "--android-serial", "emulator-5556"]
        child, identity = controller.spawn(command, "android-gateway", p["framework"])
        android_item = dict(p["previous_gateways"]["android-emulator"]["item"],
                            token_file=str(android_runtime / "token.txt"))
        observed = controller.ready(child, lambda: base.gateway_api(android_item, "/observe"),
                                    lambda value: value.get("paused") is True and value.get("device_id") == "android-api35"
                                    and bool(value.get("session_id")), 180)
        android = dict(catalogue_id="android-emulator", pid=identity["pid"], command=command,
                       process_identity=identity, device_id=observed["device_id"], session_id=observed["session_id"], paused=True)
        devices = [linux, android]
        controller.checkpoint(new_gateways=devices, android_gateway_created=True)
        catalogue = base.read(base.RUNTIME / "cross-device-catalogue.json")
        for row in catalogue["devices"]:
            if row["id"] == "linux-vm": row["token_file"] = linux_item["token_file"]
            if row["id"] == "android-emulator": row["token_file"] = android_item["token_file"]
        base.save(registry, catalogue); registry.chmod(0o600)
        base.save(base.RUNTIME / "cross-device-catalogue.json", catalogue)
        controller.checkpoint(registry=str(registry), registry_sha256=owned.digest(registry))
        dependency = dict(kind="separate_candidate_device_dependency_upgrade", devices=devices,
            entry_sha256=p["helper_sha256"]["device_gateway_entry.py"],
            source_sha256={"apps/computer_use/" + name:value for name,value in p["helper_sha256"].items()},
            candidate_framework=p["framework"], frozen_formal_unchanged=True)
        base.save(dependency_path, dependency)
        command = [str(RUNTIME.CONTROLLER_PYTHON), "-m", "qw3_computer.app", "--port", "18215",
                   "--runtime", str(runtime), "--device-registry", str(registry), "--selected-device", "windows-vm",
                   "--execution-mode", "verified", "--thinking", "--model-output-tokens", "16384",
                   "--tool-call-limit", "100", "--output-policy", "task"]
        child, identity = controller.spawn(command, "model-client", runtime / "framework")
        controller.ready(child, lambda: base.request(p["app"], "/api/state", timeout=3),
                         lambda value: value.get("ready") is True and value.get("status") == "idle"
                         and value.get("desktop_paused") is True and not value.get("uncertain_actions")
                         and value.get("thinking") is True and value.get("model_output_tokens") == 16384
                         and value.get("tool_call_limit") == 100)
        controller.checkpoint(status="admitted", model_client=identity, staged_deployment_completed=True,
                              android_container_memory_bytes=12 * 1024**3)


def run_sdk(*args):
    with patch.object(BASE_CONTROLLER, "restore_engine", staged_restore), patch.object(BASE_CONTROLLER, "deploy", staged_deploy):
        return RUNTIME.FROZEN_RUN_SDK(*args)


def validate(path):
    declaration, canonical, memory = RUNTIME.FROZEN_VALIDATE(path)
    stage_scope()
    previous.failure_scope()
    return declaration, canonical, memory


def prepare(args):
    scope = stage_scope(live=True)
    if (RUNTIME.base.RUNTIME / "android/gateway-05").exists():
        raise SafetyHalt("Never repeat an existing staged recovery")
    path = RUNTIME.FROZEN_PREPARE(args)
    declaration = RUNTIME.read(path)
    declaration.update(staged_sdk_recovery_audit=str(AUDIT), staged_sdk_recovery_audit_sha256=AUDIT_SHA,
        partial_deployment_reuse=True, android_container_memory_bytes=12 * 1024**3,
        scope="Original frozen 10/12/4/8 tasks, packages, model and budgets; resume only the recorded zero-trial Android startup failure. Reuse qualified owned OFF engine and paused Linux gateway; fresh Android gateway runtime, original model client. Android container capacity 8→12GiB is recorded separately.")
    files = [AUDIT, FAILED_DECLARATION, *[Path(row[key]) for row in scope["audit"]["successors"].values() for key in ("registration", "queue")]]
    declaration["files"] += [dict(path=str(item), sha256=owned.digest(item)) for item in dict.fromkeys(files)]
    RUNTIME.save(path, declaration); validate(path)
    return path


def build_namespace():
    if owned.digest(previous.__file__) != PREVIOUS_SHA:
        raise SafetyHalt("Frozen handoff recovery source changed")
    module = previous.build_namespace()
    module.__file__, module.MODULE, module.HELPERS = __file__, MODULE, HELPERS
    module.qualification = qualification
    module.FROZEN_RUN_SDK = module.run_sdk
    module.run_sdk, module.validate, module.prepare = run_sdk, validate, prepare
    return module


RUNTIME = build_namespace()
BASE_CONTROLLER = RUNTIME.base.Controller


if __name__ == "__main__":
    RUNTIME.main()
