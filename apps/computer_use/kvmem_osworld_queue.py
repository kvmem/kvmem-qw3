"""Run the declared eight KVMem trials after the qualified49 comparison.

This controller never changes a frozen runner, initializer, grader or package.
It admits only terminal predecessors, registers exact owned process identities,
and leaves live or uncertain work in place when a gate fails.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import URLError

from osworld_candidate_run import engine_identity
from osworld_linux.runner import request, save, source_identity
from osworld_windows.source import ROOT


class SafetyHalt(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_identity(pid, proc_root=Path("/proc")):
    if type(pid) is not int or pid <= 1:
        raise SafetyHalt("Invalid process identity")
    folder = proc_root / str(pid)
    try:
        stat = (folder / "stat").read_text().rpartition(")")[2].split()
        return {"pid": pid, "uid": folder.stat().st_uid, "start_ticks": int(stat[19]),
                "command": [part.decode() for part in (folder / "cmdline").read_bytes().split(b"\0") if part],
                "cwd": str((folder / "cwd").resolve(strict=True)), "state": stat[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected, actual):
    return actual is not None and all(expected.get(k) == actual.get(k)
        for k in ("pid", "uid", "start_ticks", "command", "cwd"))


def register_child(pid, command, timeout=2):
    # exec can briefly expose an empty cmdline. Wait for the expected command,
    # rather than recording it as a durable identity or guessing from the PID.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = process_identity(pid)
        if current and current["command"] == command:
            if current["uid"] != os.getuid() or current["cwd"] != str(ROOT):
                raise SafetyHalt("Child owner or directory changed")
            return current
        time.sleep(.01)
    raise SafetyHalt("Child command identity remains unconfirmed")


def retire_registered(expected, timeout=60):
    """Only the registered instance may receive SIGTERM; never force-kill."""
    current = process_identity(expected["pid"])
    if current is None or current["state"] == "Z":
        return {"pid": expected["pid"], "retired": True, "already_exited": True}
    if expected.get("uid") != os.getuid() or not same_process(expected, current):
        raise SafetyHalt("Registered process owner, start time, command or directory changed")
    os.kill(expected["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = process_identity(expected["pid"])
        if current is None or current["state"] == "Z" or not same_process(expected, current):
            return {"pid": expected["pid"], "retired": True,
                    "new_instance_preserved": current is not None and current["state"] != "Z"}
        time.sleep(.2)
    raise SafetyHalt("Registered process did not retire; no replacement started")


def validate_protocol(path):
    p = json.loads(path.read_text())
    if (p.get("kind") != "declared_local_qw3_kvmem_osworld_comparison"
            or p.get("source_candidate") != "qualified49"
            or p.get("experimental_visual_context_candidate_included") is not False
            or (p.get("seconds"), p.get("tools"), p.get("model_output_tokens")) != (1800, 250, 16384)
            or p.get("thinking") is not True or p.get("advisor") is not False
            or p.get("gui_only") is not True or p.get("model_concurrency") != 1
            or p.get("require_fresh_owned_engine_per_trial") is not True):
        raise SafetyHalt("Use the declared qualified49 local-only comparison")
    expected = {(task, policy, repeat) for task in ("035", "080")
                for policy in ("off", "on") for repeat in (1, 2)}
    rows = p.get("trials", [])
    if len(rows) != 8 or {(r["task"], r["kvmem"], r["repetition"]) for r in rows} != expected:
        raise SafetyHalt("Declare exactly two repetitions of both tasks and policies")
    for r in rows:
        if r["arm"] != "verified" or r["id"] != f'{r["task"]}-kvmem-{r["kvmem"]}-r{r["repetition"]}':
            raise SafetyHalt("Unexpected task/arm identity")
    parent = Path(p["parent_candidate_protocol"])
    if digest(parent) != p["parent_candidate_protocol_sha256"]:
        raise SafetyHalt("Parent protocol changed")
    if digest(ROOT / "apps/computer_use/run_engine.py") != p["engine_launcher_sha256"]:
        raise SafetyHalt("Engine launcher changed")
    framework = Path(p["framework"])
    for policy in ("off", "on"):
        acceptance = path.parent / policy / "acceptance.json"
        if digest(acceptance) != p["acceptance_sha256"][policy]:
            raise SafetyHalt("Acceptance declaration changed")
        a = json.loads(acceptance.read_text())
        if (a["framework_files"] != source_identity(framework)
                or a["engine_identity"] != p["engine_identities"][policy]
                or a["trials"] != {r["id"]: {"task": r["task"], "arm": "verified"}
                                      for r in rows if r["kvmem"] == policy}):
            raise SafetyHalt("Qualified source, engine or task declaration changed")
        for name, value in a["candidate_source_sha256"].items():
            if digest(ROOT / name) != value:
                raise SafetyHalt("Frozen runner helper changed: " + name)
        for name, value in a["framework_auxiliary_files"].items():
            if digest(framework / name) != value:
                raise SafetyHalt("Frozen observation helper changed: " + name)
        if digest(ROOT / "apps/computer_use/osworld_candidate_run.py") != a["candidate_runner_sha256"]:
            raise SafetyHalt("Frozen candidate runner changed")
    return p


def terminal_parent(p, parent_state):
    parent = Path(p["parent_candidate_protocol"])
    declaration = json.loads(parent.read_text())
    expected = {r["id"] for r in declaration["trials"]}
    rows = parent_state.get("trials", [])
    if (parent_state.get("status") != "completed" or len(expected) != 40
            or len(rows) != 40 or {r["id"] for r in rows} != expected
            or parent_state.get("protocol_sha256") != p["parent_candidate_protocol_sha256"]):
        raise SafetyHalt("Parent does not prove all forty terminal trials")
    receipts = []
    for row in rows:
        path = parent.parent / "attempts" / row["id"] / "result.json"
        result = json.loads(path.read_text())
        if (not result.get("finished_at") or type(result.get("uncertain_actions")) is not int
                or result["uncertain_actions"] != 0
                or result.get("owned_task_stop_unconfirmed") or result != row["result"]):
            raise SafetyHalt("Missing, altered or uncertain parent receipt: " + row["id"])
        receipts.append({"id": row["id"], "result_sha256": digest(path)})
    return receipts


def assert_idle(app):
    state = request(app, "/api/state", timeout=10)
    if (state.get("status") not in {"idle", "completed", "failed", "stopped"}
            or state.get("uncertain_actions") or state.get("desktop_paused") is not True):
        raise SafetyHalt("Independent model app is active, unreconciled or not paused")


def assert_apps_idle(pilot_app):
    assert_idle(pilot_app)
    try:
        assert_idle("http://127.0.0.1:18083")
    except URLError as exc:
        if not isinstance(exc.reason, ConnectionRefusedError):
            raise SafetyHalt("Native Windows app state is unconfirmed") from exc


def assert_no_services(directory):
    directory = directory.resolve()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            identity = process_identity(int(entry.name))
        except (OSError, ValueError, UnicodeError, IndexError, SafetyHalt):
            continue
        if not identity or identity["uid"] != os.getuid() or identity["state"] == "Z":
            continue
        argv = identity["command"]
        if not any(name in argv for name in ("qw3_computer.app", "device_gateway_entry",
                "osworld_office_candidate_run", "osworld_candidate_run", "osworld_linux.runner")):
            continue
        for field in ("--runtime", "--output"):
            if field in argv:
                path = Path(argv[argv.index(field) + 1]).resolve()
                if path == directory or directory in path.parents:
                    raise SafetyHalt("Owned model or device service remains alive")


def assert_vm_retired(attempt):
    name = "qw3-linux-" + attempt.name
    if not re.fullmatch(r"qw3-linux-[a-zA-Z0-9_-]{1,80}", name):
        raise SafetyHalt("Invalid owned VM identity")
    result = subprocess.run(["docker", "inspect", "--type", "container", name],
                            capture_output=True, text=True, timeout=15)
    if result.returncode:
        if "No such object" in result.stderr or "No such container" in result.stderr:
            return
        raise SafetyHalt("Cannot confirm owned VM retirement")
    container = json.loads(result.stdout)[0]
    if (container["Config"].get("Labels", {}).get("qw3.goal") != "cross-device-20261003"
            or container["Name"] != "/" + name or container["State"]["Running"]):
        raise SafetyHalt("Owned VM is live or has a different owner")


def register_engine(runtime, expected):
    manifest = json.loads((runtime / "manifest.json").read_text())
    current = process_identity(manifest["pid"])
    if (engine_identity(runtime) != expected or current is None or current["state"] == "Z"
            or current["uid"] != os.getuid() or current["cwd"] != str(ROOT)
            or current["command"] != manifest["command"] or not manifest.get("owns_process_only")
            or digest(Path(manifest["command"][0])) != expected["binary_sha256"]):
        raise SafetyHalt("Owned engine identity is unproven")
    return current


@contextlib.contextmanager
def model_lock():
    with Path("/tmp/qw3-cross-device-model-queue.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def gpu_snapshot():
    args = ["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    lines = subprocess.check_output(args, text=True, timeout=10).strip().splitlines()
    if len(lines) != 1:
        raise SafetyHalt("Explicit GPU selection is required")
    free, used, load = (int(x.strip()) for x in lines[0].split(","))
    pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                   text=True, timeout=10).strip().splitlines()
    return {"at": time.time(), "free_mib": free, "used_mib": used, "gpu_percent": load,
            "compute_pids": pids}


def run(args):
    p = validate_protocol(args.protocol)
    parent_dir = Path(p["parent_candidate_protocol"]).parent
    parent_protocol = json.loads(Path(p["parent_candidate_protocol"]).read_text())
    initial_runtime = Path(parent_protocol["engine_runtime"])
    initial = register_engine(initial_runtime, p["engine_identities"]["off"])
    args.state_dir.mkdir(parents=True, exist_ok=False)
    state = {"kind": "qualified49_kvmem_serial_queue", "pid": os.getpid(), "started_at": time.time(),
             "status": "waiting-for-parent-forty", "trials": [], "model_concurrency": 1,
             "protocol_sha256": digest(args.protocol), "controller_sha256": digest(Path(__file__)),
             "initial_registered_engine": initial, "planned_trials": 8}
    def checkpoint():
        save(args.state_dir / "queue.json", state)
    checkpoint()
    engine = None
    registered = initial
    try:
        deadline = time.monotonic() + args.wait_seconds
        while True:
            previous = json.loads((parent_dir / "queue.json").read_text())
            if previous["status"] == "completed":
                parent_receipts = terminal_parent(p, previous)
                parent_process = process_identity(previous["pid"])
                if parent_process is None or parent_process["state"] == "Z":
                    break
            elif previous["status"].startswith("halted"):
                raise SafetyHalt("Parent halted; no KVMem dispatch or engine change")
            if time.monotonic() >= deadline:
                raise SafetyHalt("Parent wait deadline reached; engine preserved")
            time.sleep(3)
        validate_protocol(args.protocol)
        assert_apps_idle(p["pilot_app"])
        assert_no_services(parent_dir)
        for row in previous["trials"]:
            assert_vm_retired(parent_dir / "attempts" / row["id"])
        state.update(parent_terminal_receipts=parent_receipts, status="admitted")
        checkpoint()
        for row in p["trials"]:
            validate_protocol(args.protocol)
            assert digest(args.protocol) == state["protocol_sha256"]
            output = args.protocol.parent / "attempts" / row["id"]
            if output.exists():
                raise SafetyHalt("Existing attempt cannot be replayed or overwritten")
            with model_lock():
                assert_apps_idle(p["pilot_app"])
                assert_no_services(args.protocol.parent / "attempts")
                state.update(status="retiring-registered-engine", current_trial=row["id"])
                checkpoint()
                state["last_engine_retirement"] = retire_registered(registered)
                if engine is not None and engine.pid == registered["pid"]:
                    engine.wait(timeout=5)
                checkpoint()
                resource_deadline = time.monotonic() + args.resource_wait_seconds
                while True:
                    resource = gpu_snapshot()
                    state.update(status="resource-wait", last_gpu_snapshot=resource)
                    checkpoint()
                    if resource["free_mib"] >= args.min_free_mib:
                        break
                    if time.monotonic() >= resource_deadline:
                        raise SafetyHalt("Shared GPU headroom unavailable; other processes preserved")
                    time.sleep(5)
                runtime = args.protocol.parent / "engines" / row["id"]
                command = [sys.executable, str(ROOT / "apps/computer_use/run_engine.py"),
                    "--run-dir", str(runtime), "--method", "dflash2", "--kvmem", row["kvmem"],
                    "--prefix-reuse", "off", "--visual-history", "pin-all", "--guided-refresh", "off",
                    "--context-tokens", "524288", "--kvmem-cpu-gib", "48",
                    "--gpu-sharing", "spare-capacity", "--min-free-mib", str(args.min_free_mib),
                    "--lock-timeout", "5"]
                log_path = args.state_dir / (row["id"] + "-engine-launch.log")
                with log_path.open("xb") as log:
                    engine = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state.update(status="starting-fresh-engine", engine_launcher_pid=engine.pid,
                             current_engine_runtime=str(runtime))
                checkpoint()
                readiness_deadline = time.monotonic() + 600
                while True:
                    if engine.poll() is not None:
                        raise SafetyHalt("Fresh engine exited; no model trial dispatched")
                    try:
                        models = request(p["engine_endpoint"], "/v1/models", timeout=3)
                        if models["data"][0].get("context_window") == 524288:
                            registered = register_engine(runtime, p["engine_identities"][row["kvmem"]])
                            if registered["pid"] != engine.pid:
                                raise SafetyHalt("Fresh launcher and backend process differ")
                            break
                    except (OSError, KeyError, ValueError):
                        pass
                    if time.monotonic() >= readiness_deadline:
                        raise SafetyHalt("Fresh engine readiness unknown; no replacement started")
                    time.sleep(2)
                state["registered_engine"] = registered
                checkpoint()
            # The unchanged runner independently acquires the same queue lock.
            # Any intervening owner makes it fail before task input, not overlap.
            command = [sys.executable, "-m", "osworld_office_candidate_run", "--task", row["task"],
                "--arm", "verified", "--output", str(output), "--framework", p["framework"],
                "--engine-runtime", str(runtime), "--formal-campaign", p["formal_campaign"],
                "--acceptance-manifest", str(args.protocol.parent / row["kvmem"] / "acceptance.json"),
                "--pilot-app", p["pilot_app"], "--model-output-tokens", "16384"]
            environment = dict(os.environ, PYTHONPATH=str(ROOT / "apps/computer_use") + ":" +
                               str(ROOT / "apps/computer_use/.runtime/media-qualification"))
            with (args.state_dir / (row["id"] + "-runner.log")).open("xb") as log:
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                state.update(status="running", child=register_child(child.pid, command), child_command=command)
                checkpoint()
                code = child.wait(timeout=5100)
            result_path = output / "result.json"
            result = json.loads(result_path.read_text())
            if (not result.get("finished_at") or type(result.get("uncertain_actions")) is not int
                    or result["uncertain_actions"] != 0 or result.get("owned_task_stop_unconfirmed")):
                raise SafetyHalt("Trial is not durably terminal; engine and evidence preserved")
            assert_no_services(output)
            assert_vm_retired(output)
            state["trials"].append({"id": row["id"], "kvmem": row["kvmem"], "exit_code": code,
                                    "result_sha256": digest(result_path), "result": result})
            checkpoint()
        state.update(status="completed", finished_at=time.time(), final_registered_engine=registered,
                     final_engine_policy=p["trials"][-1]["kvmem"])
        checkpoint()
    except BaseException as exc:
        state.update(status="halted-no-further-dispatch", error=type(exc).__name__ + ": " + str(exc),
                     finished_at=time.time(), engine_preserved=True,
                     last_registered_engine=registered)
        checkpoint()
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=172800)
    parser.add_argument("--resource-wait-seconds", type=int, default=21600)
    parser.add_argument("--min-free-mib", type=int, default=56000)
    args = parser.parse_args()
    if args.min_free_mib < 56000 or not (60 <= args.wait_seconds <= 172800
                                        and 60 <= args.resource_wait_seconds <= 86400):
        parser.error("Use explicit shared-capacity and finite waiting bounds")
    args.protocol = args.protocol.resolve()
    args.state_dir = args.state_dir.resolve()
    def interrupted(*_):
        raise SafetyHalt("Queue interrupted; live children and engine preserved")
    signal.signal(signal.SIGTERM, interrupted)
    outcome = run(args)
    print(json.dumps({key: outcome[key] for key in ("status", "planned_trials")}))


if __name__ == "__main__":
    main()
