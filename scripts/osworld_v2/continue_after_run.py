#!/usr/bin/env python3
"""Run a frozen OSWorld task list after an existing evaluation releases the GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return fields[19] if fields[0] != "Z" else None
    except FileNotFoundError:
        return None


def predecessor_alive(plan):
    return any(process_identity(item["pid"]) == item["start_ticks"]
               for item in plan["wait_for_processes"])


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # the producer may still be writing the last record
    return result


def score(path):
    try:
        value = float(path.read_text().strip())
        return value if math.isfinite(value) else None
    except (FileNotFoundError, ValueError):
        return None


def progress(run, task_ids):
    task_root = run / "osworld/pyautogui/screenshot/gpt-qw3/tasks"
    scores = {task: score(task_root / task / "result.txt") for task in task_ids}
    completed = {task: value for task, value in scores.items() if value is not None}
    pending = [task for task in task_ids if scores[task] is None]
    prediction_failures = [task for task in pending
                           if any(row.get("status") == "predict_error"
                                  for row in rows(task_root / task / "step_timing.jsonl"))]
    value = {"completed_tasks": len(completed), "task_scores": completed,
             "tasks_without_result": pending,
             "prediction_failure_tasks": prediction_failures,
             "mean_score_completed": statistics.mean(completed.values()) if completed else None,
             "current_task": None}
    traces = [task_root / task / "agent_step_trace.jsonl" for task in pending]
    traces = [p for p in traces if p.exists()]
    if traces:
        trace = max(traces, key=lambda p: p.stat().st_mtime_ns)
        records = rows(trace)
        timings = rows(trace.with_name("step_timing.jsonl"))
        events = rows(trace.with_name("history_compaction.jsonl"))
        ttfts = [r.get("api", {}).get("server_timing", {}).get("server_ttft_sec")
                 for r in records[-20:]]
        ttfts = [v for v in ttfts if isinstance(v, (int, float))]
        step_times = [r["turn_wall_time_ms"] / 1000 for r in timings[-20:]
                      if isinstance(r.get("turn_wall_time_ms"), (int, float))]
        value["current_task"] = {
            "task_id": trace.parent.name, "completed_steps": len(timings),
            "summary_compactions": sum(e.get("status") == "success" for e in events),
            "summary_failures": sum(e.get("status") == "error" for e in events),
            "recent20_mean_ttft_sec": statistics.mean(ttfts) if ttfts else None,
            "recent20_mean_step_sec": statistics.mean(step_times) if step_times else None,
            "last_prompt_tokens": records[-1].get("api", {}).get("usage", {}).get("prompt_tokens") if records else None,
            "image_folding_events": sum(bool(r.get("compaction", {}).get("triggered")) for r in records),
            "history_truncation_steps": sum(r.get("compaction", {}).get("history_truncated_steps", 0) > 0 for r in records),
        }
    return value


def run_plan(plan, state_path):
    poll_seconds = max(0.05, float(plan.get("poll_seconds", 15)))
    state = {"status": "waiting_for_predecessor", "supervisor_pid": os.getpid(),
             "created_at_utc": now(), "predecessor_run": plan["predecessor_run"],
             "task_count": len(plan["task_ids"]), "task_order": plan["task_ids"],
             "run_dir": None, "completed_tasks": 0}

    def save():
        state["updated_at_utc"] = now()
        write_json(state_path, state)

    child = None

    def stop(signum, frame):
        state["status"] = "cancelled"
        save()
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        while True:
            previous = json.loads(Path(plan["predecessor_metadata"]).read_text())
            if previous.get("status", "").startswith("stopped"):
                state["status"] = "cancelled_predecessor_stopped"
                save()
                return
            if not predecessor_alive(plan):
                break
            save()
            time.sleep(poll_seconds)

        # Waiting for both the runner and its exact server process keeps final
        # evaluation and shutdown ahead of the next GPU allocation. PID reuse
        # cannot make us wait on or terminate an unrelated process.
        previous_task = plan.get("predecessor_task_id", "003")
        previous_result = (Path(plan["predecessor_run"]) /
                           "osworld/pyautogui/screenshot/gpt-qw3/tasks" / previous_task / "result.txt")
        state["predecessor_task_id"] = previous_task
        state["predecessor_score"] = score(previous_result)
        state["predecessor_outcome"] = (
            "completed" if state["predecessor_score"] is not None else
            "interrupted_by_user" if plan.get("predecessor_interrupted") else "infrastructure_failure")
        port = int(plan["settings"].get("QW3_OSWORLD_PORT", "18080"))
        while True:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    pass
            except OSError:
                break
            state["status"] = "waiting_for_server_port"
            save()
            time.sleep(poll_seconds)

        settings = dict(plan["settings"])
        settings["QW3_OSWORLD_RUN_ID"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(plan["root"])
        run_root = Path(settings.get("QW3_OSWORLD_RESULT_ROOT", str(root / "benchmark/osworld_v2/results")))
        run = run_root / (settings["QW3_OSWORLD_RUN_LABEL"] + "-" + settings["QW3_OSWORLD_RUN_ID"])
        if run.exists():
            raise FileExistsError(f"Refusing to reuse existing result directory: {run}")
        selected = json.loads(Path(settings["OSWORLD_TEST_META"]).read_text())
        if selected != {"tasks": plan["task_ids"]}:
            raise ValueError("Queued task list changed after scheduling")
        env = {k: v for k, v in os.environ.items()
               if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
        env.update(settings)
        write_json(Path(state_path).with_name("continuation_launch_settings.json"), settings)
        with Path(state_path).with_name("continuation_runner.log").open("w") as log:
            child = subprocess.Popen(["bash", str(root / "scripts/osworld_v2/run_qw3_full.sh")],
                                     cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        state.update(status="running", runner_pid=child.pid, run_dir=str(run), started_at_utc=now())
        save()
        while child.poll() is None:
            state.update(progress(run, plan["task_ids"]))
            if plan.get("stop_on_prediction_failure") and state["prediction_failure_tasks"]:
                state["status"] = "stopped_prediction_failure"
                save()
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                state.update(exit_code=child.returncode, finished_at_utc=now())
                save()
                return
            save()
            time.sleep(poll_seconds)
        state.update(progress(run, plan["task_ids"]))
        state.update(exit_code=child.returncode, finished_at_utc=now())
        state["infrastructure_failure_tasks"] = state["tasks_without_result"]
        state["status"] = ("completed" if child.returncode == 0 and not state["tasks_without_result"]
                           else "completed_with_infrastructure_failures")
        save()
    except Exception as exc:
        state.update(status="supervisor_error", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    with args.plan.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = Path(plan["state_path"])
        if state_path.exists():
            raise SystemExit("Continuation already scheduled; refusing to launch a duplicate")
        run_plan(plan, state_path)


if __name__ == "__main__":
    main()
