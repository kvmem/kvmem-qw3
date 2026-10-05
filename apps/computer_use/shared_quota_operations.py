"""Retire exact idle waiters and qualify the common quota on the real GPU.

No device input or business task runs here. Linux pidfds pin each signalled
process instance; a failed idle check resumes the frozen processes unchanged.
"""
import argparse
import contextlib
import ctypes
import json
import os
import platform
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

import shared_quota_memory as memory
import runner_runtime_successors as successors

owned, ROOT, SafetyHalt = memory.owned, memory.ROOT, memory.SafetyHalt
HELPERS = ("shared_quota_operations.py", "tests/test_shared_quota_operations.py")
WAITING = dict(memory="resource-wait", sdk="waiting-for-forty-and-eight",
               image="waiting-for-forty-eight-and-ten", thinking="waiting-for-image-twelve",
               feedback="waiting-for-forty-eight-ten-twelve-four")


def qualification(path):
    q = memory.read(path)
    if (q.get("kind") != "shared_quota_operations_cpu_qualification" or q.get("passed") is not True
            or q.get("tests_run", 0) < 8 or q.get("skipped", 0) != 0
            or q.get("model_calls") != 0 or q.get("device_inputs") != 0):
        raise SafetyHalt("Require exact process transition qualification")
    for name in HELPERS:
        memory.old.image.pinned(ROOT / "apps/computer_use" / name, q["source_sha256"][name])


def children(pid):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit(): continue
        try:
            stat = (entry / "stat").read_text().rpartition(")")[2].split()
            if int(stat[1]) == pid: found.append(int(entry.name))
        except (OSError, ValueError, IndexError): continue
    return found


def checked_instance(identity):
    actual = owned.process_identity(identity["pid"])
    if (identity.get("uid") != os.getuid() or identity.get("cwd") != str(ROOT)
            or not owned.same_process(identity, actual) or actual["state"] == "Z"):
        raise SafetyHalt("Only the exact live owned process can be signalled")
    return actual


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"): return os.pidfd_open(pid)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SafetyHalt("Native pidfd API required on this host")
    # Host asm/unistd_64.h declares pidfd_open434 and send_signal424.
    libc = ctypes.CDLL(None, use_errno=True); libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    return int(result)


def pidfd_signal(fd, number):
    if hasattr(signal, "pidfd_send_signal"): return signal.pidfd_send_signal(fd, number)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SafetyHalt("Native pidfd API required on this host")
    libc = ctypes.CDLL(None, use_errno=True); libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(number), ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))


@contextlib.contextmanager
def frozen_controls(rows):
    handles = []
    try:
        for row in rows:
            identity = row["identity"]
            checked_instance(identity)
            fd = pidfd_open(identity["pid"])
            try: checked_instance(identity)
            except BaseException:
                os.close(fd); raise
            handles.append((row, fd))
            pidfd_signal(fd, signal.SIGSTOP)
            deadline = time.monotonic() + 2
            while checked_instance(identity)["state"] not in {"T", "t"}:
                if time.monotonic() >= deadline: raise SafetyHalt("Own waiter stop unconfirmed")
                time.sleep(.01)
        yield handles
    finally:
        for row, fd in reversed(handles):
            # The descriptor refers to the original instance even if its PID
            # was reused. ESRCH means that instance has already exited.
            try: pidfd_signal(fd, signal.SIGCONT)
            except ProcessLookupError: pass
            finally: os.close(fd)


def port_free():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 18080))


def predecessor_rows(d):
    producer, _ = memory.previous.validate(Path(d["previous_continuation"]))
    rows = []
    for stage in memory.STAGES:
        path = Path(d["previous_registrations"][stage])
        reg = memory.read(path)
        if stage == "memory":
            successors.memory_registration(path, Path(d["previous_continuation"]), producer)
            successors.memory_queue(reg, Path(d["previous_continuation"]))
        else:
            declaration = Path(reg["declaration"]); earlier = memory.read(declaration)
            successors.successor_registration(path, stage, declaration, earlier)
            successors.successor_queue(reg, declaration)
            if earlier["memory_continuation"] != d["previous_continuation"]:
                raise SafetyHalt("Successor belongs to another producer")
        rows.append(dict(stage=stage, identity=reg["identity"], registration=str(path),
                         registration_sha256=owned.digest(path), queue=reg["queue"]))
    return rows


def idle_states(d, rows, *, stopped):
    prefix = memory.read(d["previous_continuation"])["completed_prefix"]
    snapshots = []
    for row in rows:
        actual = checked_instance(row["identity"])
        state = memory.read(row["queue"])
        if (state.get("pid") != actual["pid"] or state.get("status") != WAITING[row["stage"]]
                or state.get("trials") != (prefix if row["stage"] == "memory" else [])
                or any(state.get(k) for k in ("child", "current_child", "registered_engine", "engine_launcher_pid", "current_engine_runtime"))
                or (stopped and actual["state"] not in {"T", "t"}) or children(actual["pid"])):
            raise SafetyHalt("Predecessor is not an exact idle waiter; resume without retirement")
        if row["stage"] == "memory" and (state.get("current_trial") != "035-kvmem-on-r1"
                or state.get("shared_admission_min_free_mib") != 93000):
            raise SafetyHalt("The original capacity wait changed")
        snapshots.append(dict(row, actual_at_check=actual, queue_sha256=owned.digest(row["queue"]), state=state))
    owned.assert_apps_idle(memory.read(d["continued_protocol"])["pilot_app"])
    original = Path(memory.read(d["previous_continuation"])["continued_protocol"])
    owned.assert_no_services(original.parent / "attempts")
    if (original.parent / "attempts/035-kvmem-on-r1").exists() or (original.parent / "engines/035-kvmem-on-r1").exists():
        raise SafetyHalt("An ON launch or task appeared while waiting")
    port_free()
    return snapshots


def retire(declaration, cpu_qualification):
    qualification(cpu_qualification)
    d, _ = memory.validate(declaration, require_retirement=False, require_startup=False)
    target = Path(d["retirement_receipt"])
    if target.exists(): raise SafetyHalt("Do not overwrite a prior transition")
    if all(owned.process_identity(memory.read(path)["identity"]["pid"]) is None
           for path in d["previous_registrations"].values()):
        rows = []
        for stage in memory.STAGES:
            path = Path(d["previous_registrations"][stage]); reg = memory.read(path)
            if stage == "memory":
                producer, _ = memory.previous.validate(Path(d["previous_continuation"]))
                successors.memory_registration(path, Path(d["previous_continuation"]), producer)
            else:
                earlier = memory.read(reg["declaration"])
                successors.successor_registration(path, stage, Path(reg["declaration"]), earlier)
                if earlier["memory_continuation"] != d["previous_continuation"]:
                    raise SafetyHalt("Retired successor belongs to another producer")
            rows.append(dict(stage=stage, identity=reg["identity"], registration=str(path),
                             registration_sha256=owned.digest(path), queue=reg["queue"], queue_sha256=owned.digest(reg["queue"])))
        runtime = Path(memory.read(rows[0]["queue"])["current_engine_runtime"])
        receipt = dict(kind="exact_idle_quota_transition_retirement", passed=True, checked_at=time.time(),
            declaration_sha256=owned.digest(declaration), old_memory_completed_prefix=1, retired=rows,
            retirement_method="natural-capacity-startup-failure", signals_sent=0, model_calls=0, device_inputs=0,
            startup_failure=dict(runtime=str(runtime), manifest_sha256=owned.digest(runtime/"manifest.json"), log_sha256=owned.digest(runtime/"engine.log")))
        memory.retirement_evidence(d, declaration, receipt)
        owned.assert_apps_idle(memory.read(d["continued_protocol"])["pilot_app"]); port_free()
        owned.save(target, receipt)
        return receipt
    rows = predecessor_rows(d)
    idle_states(d, rows, stopped=False)
    with frozen_controls(rows) as handles:
        frozen = idle_states(d, rows, stopped=True)
        owned.save(target.parent / "predecessor-frozen-idle.json", dict(declaration_sha256=owned.digest(declaration), rows=frozen))
        # No waiter can move from the validated idle state before receiving
        # TERM. Resume delivers its normal evidence-preserving handler.
        for row, fd in handles: pidfd_signal(fd, signal.SIGTERM)
    deadline = time.monotonic() + 55
    while any(owned.process_identity(r["identity"]["pid"]) is not None for r in rows):
        if time.monotonic() >= deadline:
            raise SafetyHalt("Original handles still exist; no GPU startup allowed")
        time.sleep(.1)
    for row in rows: row["queue_sha256"] = owned.digest(row["queue"])
    receipt = dict(kind="exact_idle_quota_transition_retirement", passed=True, checked_at=time.time(),
                   declaration_sha256=owned.digest(declaration), old_memory_completed_prefix=1,
                   retired=rows, retirement_method="pidfd-idle", model_calls=0, device_inputs=0)
    memory.retirement_evidence(d, declaration, receipt)
    owned.save(target, receipt)
    return receipt


def startup(declaration, cpu_qualification):
    qualification(cpu_qualification)
    d, _ = memory.validate(declaration, require_startup=False)
    target = Path(d["startup_qualification"])
    if target.exists(): raise SafetyHalt("Do not overwrite a physical startup result")
    profile = memory.engine.qualify_profile(memory.read(d["profile"]))
    runtime = target.parent / "startup-engine-on-01"
    command = [str(memory.PYTHON), str(ROOT / "apps/computer_use/shared_quota_engine.py"), "--profile", d["profile"],
        "--run-dir", str(runtime), "--method", "dflash2", "--kvmem", "on", "--prefix-reuse", "off",
        "--visual-history", "pin-all", "--guided-refresh", "off", "--context-tokens", "524288",
        "--kvmem-cpu-gib", "48", "--gpu-sharing", "spare-capacity", "--min-free-mib", "56000", "--lock-timeout", "5"]
    state = dict(kind="actual_common_quota_startup_qualification", passed=False, status="preflight",
        profile_sha256=owned.digest(d["profile"]), wrapper_sha256=owned.digest(memory.engine.__file__),
        operations_sha256=owned.digest(__file__), engine_profile=profile["engine_identities"]["on"],
        engine_runtime=str(runtime), launch_command=command, task_trials_dispatched=0, device_inputs=0, health_model_calls=0)
    journal = target.parent / "startup-progress.json"
    process = None; first = None
    with owned.model_lock():
        try:
            port_free()
            owned.assert_apps_idle(memory.read(d["continued_protocol"])["pilot_app"])
            state["startup_gpu"] = owned.gpu_snapshot()
            if state["startup_gpu"]["free_mib"] < 56000:
                raise SafetyHalt("Shared headroom below unchanged admission gate; no engine launched")
            with (target.parent / "startup-launch.log").open("xb") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state.update(status="starting", launcher_pid=process.pid); owned.save(journal, state)
            deadline = time.monotonic() + 5
            while first is None:
                actual = owned.process_identity(process.pid)
                if actual and actual["uid"] == os.getuid() and actual["cwd"] == str(ROOT) and actual["command"] in (command, profile["engine_identities"]["on"]["command"]):
                    first = actual; state["first_owned_identity"] = first; break
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise SafetyHalt("Fresh launcher identity could not be registered")
                time.sleep(.01)
            deadline = time.monotonic() + 600
            samples = []
            while True:
                if process.poll() is not None: raise SafetyHalt("Common-quota engine exited during startup")
                sample = owned.gpu_snapshot(); samples.append(sample)
                state.update(last_gpu=sample, gpu_samples=samples); owned.save(journal, state)
                try:
                    models = owned.request("http://127.0.0.1:18080", "/v1/models", timeout=3)
                    if models["data"][0].get("context_window") == 524288:
                        state["engine_identity"] = owned.register_engine(runtime, profile["engine_identities"]["on"])
                        if state["engine_identity"]["pid"] != process.pid: raise SafetyHalt("Startup backend does not belong to this handle")
                        state["advertised_context_tokens"] = 524288; break
                except (OSError, KeyError, ValueError): pass
                if time.monotonic() >= deadline: raise SafetyHalt("Backend readiness remains unconfirmed")
                time.sleep(2)
            payload = dict(model=models["data"][0]["id"], messages=[
                dict(role="system", content="Follow the user's output instruction exactly."),
                dict(role="user", content="This is an isolated engine health check. No desktop task is active."),
                dict(role="assistant", content="Acknowledged."),
                dict(role="user", content="Reply exactly READY, with no punctuation.")],
                           stream=False, max_tokens=32, temperature=0, top_p=1, enable_thinking=False)
            state.update(status="health-inference", health_model_calls=1, health_probe_has_synthetic_history=True); owned.save(journal, state)
            owned.save(target.parent / "health-request.json", payload)
            request = Request("http://127.0.0.1:18080/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            started = time.monotonic()
            with build_opener(ProxyHandler({})).open(request, timeout=60) as response:
                answer = json.load(response)
            owned.save(target.parent / "health-request-and-response.json", dict(request=payload, response=answer))
            state.update(health_response=answer["choices"][0]["message"]["content"].strip(), health_seconds=time.monotonic()-started,
                         manifest_sha256=owned.digest(runtime / "manifest.json"), after_health_gpu=owned.gpu_snapshot())
            if state["health_response"] != "READY": raise SafetyHalt("Health response differs from declared exact output")
        except BaseException as exc:
            state["error"] = type(exc).__name__ + ": " + str(exc)
            if isinstance(exc, HTTPError):
                state["health_http_error_body"] = exc.read(65536).decode(errors="replace")
        finally:
            if process is not None:
                actual = owned.process_identity(process.pid)
                if actual is not None and process.poll() is None:
                    if first is None or any(actual.get(k) != first.get(k) for k in ("pid", "uid", "start_ticks", "cwd")) or actual["command"] not in (command, profile["engine_identities"]["on"]["command"]):
                        state["retirement_error"] = "Fresh process identity changed; preserved without signalling"
                    else:
                        try: state["retirement"] = owned.retire_registered(actual, timeout=55)
                        except BaseException as exc: state["retirement_error"] = type(exc).__name__ + ": " + str(exc)
                try: state["exit_code"] = process.wait(timeout=5)
                except subprocess.TimeoutExpired: state["retirement_error"] = "Same startup handle is still live"
                state["owned_engine_actual_retired"] = owned.process_identity(process.pid) is None
            else: state["owned_engine_actual_retired"] = True
            state.update(status="finished", finished_at=time.time(), gpu_after_retirement=owned.gpu_snapshot())
            if "error" not in state and "retirement_error" not in state:
                state["passed"] = True
                try:
                    memory.startup_evidence(d, state); port_free()
                except BaseException as exc:
                    state.update(passed=False, error=type(exc).__name__ + ": " + str(exc))
            owned.save(journal, state); owned.save(target, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--stage", choices=("retire", "startup"), required=True)
    args = parser.parse_args()
    result = {"retire": retire, "startup": startup}[args.stage](args.declaration.resolve(), args.qualification.resolve())
    print(json.dumps(dict(passed=result["passed"], stage=args.stage, task_trials_dispatched=0)))


if __name__ == "__main__": main()
