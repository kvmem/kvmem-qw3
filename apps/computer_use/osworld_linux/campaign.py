"""Frozen, interleaved workflow comparison; retain every interruption."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from osworld_windows.source import ROOT
from .preflight import TASKS, preflight
from .runner import request, save, source_identity


def engine_identity(runtime):
    manifest = json.loads((runtime / "manifest.json").read_text())
    # Process identity is checked separately; expected configuration does not
    # depend on PID, startup timestamp or the user's concurrent GPU processes.
    return {k: manifest[k] for k in ["command", "qw3_env", "binary_sha256", "kvmem",
                                    "prefix_reuse", "visual_history", "guided_refresh", "context_tokens"]}


def jobs():
    result = []
    for repetition in (1, 2):
        for index, task in enumerate(TASKS):
            arms = ["legacy", "verified"] if (index + repetition) % 2 else ["verified", "legacy"]
            for arm in arms:
                result.append({"id": f"{task}-{arm}-r{repetition}", "task": task, "arm": arm,
                               "repetition": repetition, "status": "pending"})
    return result


def freeze(directory, engine):
    directory.mkdir(parents=True, exist_ok=False)
    package = directory / "framework"
    package.mkdir()
    shutil.copytree(ROOT / "apps/computer_use/qw3_computer", package / "qw3_computer",
                    ignore=shutil.ignore_patterns("__pycache__"))
    materials = preflight()
    if materials["ready_count"] != len(TASKS):
        save(directory / "material-preflight.json", materials)
        raise ValueError("Frozen task materials are not all available")
    configuration = engine_identity(engine)
    if configuration["kvmem"] != "off" or configuration["context_tokens"] != 524288:
        raise ValueError("The revised common campaign requires KVMem off and524288 context")
    if configuration["command"][configuration["command"].index("--speculative-method") + 1] != "dflash2":
        raise ValueError("The common campaign requires DFlash2")
    save(directory / "material-preflight.json", materials)
    plan = {"kind": "linux-original-osworld-workflow-ablation", "created_at": time.time(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "framework": str(package), "framework_files": source_identity(package),
            "engine_runtime": str(engine), "engine_identity": configuration,
            "seconds": 1800, "tool_calls": 250, "thinking": True, "advisor": False,
            "gui_only": True, "jobs": jobs(), "status": "frozen",
            "limitations": ["Two repetitions are descriptive, not statistical significance",
                            "Common compact observation and repaired adapter in both arms",
                            "Task035 cloud judge is disabled; any local score is reported separately",
                            "Task043 source rubric has known business-constraint conflicts"]}
    save(directory / "plan.json", plan)
    return plan


def verify_frozen(plan):
    if source_identity(Path(plan["framework"])) != plan["framework_files"]:
        raise ValueError("Frozen application or host harness source changed")
    engine = Path(plan["engine_runtime"])
    if engine_identity(engine) != plan["engine_identity"]:
        raise ValueError("Owned engine configuration changed")
    manifest = json.loads((engine / "manifest.json").read_text())
    pid = manifest["pid"]
    if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
        raise ValueError("Engine process owner changed")
    actual = [s.decode() for s in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if s]
    if actual != manifest["command"]:
        raise ValueError("Engine PID no longer identifies the owned command")
    backend = request("http://127.0.0.1:18080", "/v1/models", timeout=5)
    if backend["data"][0].get("context_window") != plan["engine_identity"]["context_tokens"]:
        raise ValueError("Backend metadata no longer matches frozen context")


def run(directory, limit=None):
    plan = json.loads((directory / "plan.json").read_text())
    plan["status"] = "running"
    save(directory / "plan.json", plan)
    completed = 0
    for job in plan["jobs"]:
        if job["status"] == "finished":
            continue
        if limit is not None and completed >= limit:
            break
        verify_frozen(plan)
        free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
        if len(free) != 1 or int(free[0]) < 2048:
            plan.update(status="resource-wait", notice="Insufficient shared GPU headroom; other processes are preserved")
            save(directory / "plan.json", plan)
            return plan
        if shutil.disk_usage(directory).free < 50 * 1024 ** 3:
            plan["status"] = "resource-wait"
            plan["notice"] = "Less than50GiB free; no new attempts launched"
            save(directory / "plan.json", plan)
            return plan
        output = directory / job["id"]
        if output.exists():
            # Resume reads existing evidence; never overwrite/restart an input
            # attempt after an uncertain host interruption.
            if not (output / "result.json").exists():
                job.update(status="needs-reconciliation", notice="Existing attempt has no terminal receipt")
                plan["status"] = "needs-reconciliation"
                save(directory / "plan.json", plan)
                return plan
        else:
            job.update(status="running", started_at=time.time())
            save(directory / "plan.json", plan)
            env = dict(os.environ, PYTHONPATH=str(ROOT / "apps/computer_use"))
            command = [sys.executable, "-m", "osworld_linux.runner", "--task", job["task"], "--arm", job["arm"],
                       "--output", str(output), "--framework", plan["framework"],
                       "--engine-runtime", plan["engine_runtime"], "--seconds", str(plan["seconds"]),
                       "--tools", str(plan["tool_calls"])]
            with (directory / (job["id"] + ".launch.log")).open("xb") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                job["runner_pid"] = process.pid
                save(directory / "plan.json", plan)
                try:
                    code = process.wait(timeout=plan['seconds']+2100)
                except subprocess.TimeoutExpired:
                    job['watchdog_interruption'] = time.time()
                    save(directory / 'plan.json', plan)
                    process.terminate()
                    try:
                        code = process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        # Stop at reconciliation rather than starting another
                        # VM after an unknown cleanup. Only our child is killed.
                        process.kill()
                        code = process.wait(timeout=10)
                job["runner_exit_code"] = code
            if not (output / "result.json").exists():
                job.update(status="needs-reconciliation", notice="Runner ended without terminal result")
                plan["status"] = "needs-reconciliation"
                save(directory / "plan.json", plan)
                return plan
        result = json.loads((output / "result.json").read_text())
        job.update(status="finished", result=result, finished_at=result["finished_at"])
        completed += 1
        save(directory / "plan.json", plan)
        print(json.dumps({"attempt": job["id"], "result": result}, ensure_ascii=False), flush=True)
    plan["status"] = "finished" if all(j["status"] == "finished" for j in plan["jobs"]) else "ready"
    save(directory / "plan.json", plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["freeze", "run"])
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--engine-runtime", type=Path)
    p.add_argument("--limit", type=int, help="Bound new attempts this invocation; never resets job budgets")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("limit must be positive")
    if args.operation == "freeze":
        if args.engine_runtime is None:
            p.error("freeze requires the owned engine runtime")
        plan = freeze(args.directory.resolve(), args.engine_runtime.resolve())
    else:
        import fcntl
        with (args.directory / "campaign.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plan = run(args.directory.resolve(), args.limit)
    print(json.dumps({"status": plan["status"], "finished": sum(j["status"] == "finished" for j in plan["jobs"]),
                      "total": len(plan["jobs"])}), flush=True)


if __name__ == "__main__":
    main()
