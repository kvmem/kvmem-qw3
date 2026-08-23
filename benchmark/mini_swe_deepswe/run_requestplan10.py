#!/usr/bin/env python3
"""Run the historical DeepSWE requestplan10 set with official Pier + mini-swe-agent.

Pier remains the authority for task container construction, the MiniSweAgent
prompt/tool loop, artifact collection, and verification.  This wrapper only
pins versions/configuration, starts one fresh single-trajectory QW3 service per
task, and makes the ten-task run safely resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_TASKS = HERE / "requestplan10.json"
VERSION_LOCK = HERE / "versions.lock.json"
PIER = REPO / ".venv-deepswe-pier" / "bin" / "pier"
PYTHON = REPO / ".venv-deepswe-pier" / "bin" / "python"
QW3 = REPO / "build" / "qw3"
MODEL = REPO / "models" / "Qwen3.8-27B-Q8_0.gguf"
DEEPSWE_TASKS = REPO / "benchmark" / "deep-swe" / "tasks"
DEEPSWE_MANIFEST = DEEPSWE_TASKS / "manifest.json"
SOURCE_TASK_LIST = (
    REPO / "benchmark" / "claude_deepswe_ab" / "tasks_requestplan_10_no_happy.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def run_text(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=REPO,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def validate_environment(tasks_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = read_json(VERSION_LOCK)
    task_spec = read_json(tasks_path)

    required = [PIER, PYTHON, QW3, MODEL, DEEPSWE_MANIFEST, SOURCE_TASK_LIST]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required artifact(s): " + ", ".join(missing))

    actual = {
        "pier_version": run_text(
            [
                str(PYTHON),
                "-c",
                "import importlib.metadata as m; print(m.version('datacurve-pier'))",
            ]
        ),
        "mini_swe_agent_version": run_text(
            [
                str(PYTHON),
                "-c",
                "import importlib.metadata as m; print(m.version('mini-swe-agent'))",
            ]
        ),
        "python_version": run_text(
            [str(PYTHON), "-c", "import platform; print(platform.python_version())"]
        ),
        "deep_swe_task_manifest_sha256": sha256(DEEPSWE_MANIFEST),
        "requestplan10_source_sha256": sha256(SOURCE_TASK_LIST),
    }
    expected = {
        "pier_version": lock["pier"]["version"],
        "mini_swe_agent_version": lock["mini_swe_agent"]["version"],
        "python_version": lock["python"],
        "deep_swe_task_manifest_sha256": lock["deep_swe_task_manifest_sha256"],
        "requestplan10_source_sha256": lock["requestplan10_source_sha256"],
    }
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise RuntimeError("Version/data lock mismatch:\n" + json.dumps(mismatches, indent=2))

    if task_spec.get("source_sha256") != actual["requestplan10_source_sha256"]:
        raise RuntimeError("Task selection file no longer matches its authoritative source")
    for item in task_spec["tasks"]:
        task_dir = DEEPSWE_TASKS / item["task_id"]
        if not (task_dir / "task.toml").is_file():
            raise RuntimeError(f"DeepSWE task is missing: {item['task_id']}")
    return lock, task_spec


def qwen_command(host: str, port: int) -> list[str]:
    return [
        str(QW3),
        "serve",
        "--model",
        str(MODEL),
        "--kv-dtype",
        "fp8",
        "--enable-thinking",
        "--thinking-budget",
        "28672",
        "--prefill-chunk",
        "2048",
        "--temp",
        "1.0",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--min-p",
        "0.0",
        "--presence-penalty",
        "0.0",
        "--repetition-penalty",
        "1.0",
        "--seed",
        "73",
        "--native-mtp-speculate",
        "--mtp-chain",
        "4",
        "--host",
        host,
        "--port",
        str(port),
        "--ctx",
        "1048576",
        "--kvmem",
        "--kvmem-prefix-cache",
        "--kvmem-block-tokens",
        "32",
        "--kvmem-budget",
        "131072",
        "--kvmem-prefill-budget",
        "131072",
        "--kvmem-gen-budget",
        "65536",
        "--kvmem-method",
        "retrieval",
        "--kvmem-retrieval-method",
        "mean-k",
        "--kvmem-index-placement",
        "gpu",
        "--kvmem-index-staging-mb",
        "64",
        "--kvmem-update-mode",
        "step",
        "--kvmem-query-conditioned",
        "--kvmem-guided-reselect",
        "both",
        "--kvmem-guided-thinking-tokens",
        "0",
        "--kvmem-guided-query-tokens",
        "512",
        "--kvmem-middecode-trigger-tokens",
        "61440",
        "--kvmem-middecode-max-refreshes",
        "2",
        "--kvmem-immutable-k",
        "--kvmem-gpu-memory-ratio",
        "1.0",
        "--kvmem-cpu-gb",
        "40",
        "--kvmem-opt-stage-out",
        "on",
        "--kvmem-opt-stage-in",
        "on",
        "--kvmem-opt-pack",
        "on",
        "--mtp-batched-draft",
        "--mtp-paged-prefix",
    ]


def qwen_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "QW3_PREFIX_CACHE_TRACE": "1",
            "QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES": "1",
            "QW3_PREFIX_CACHE_MAX_ENTRIES": "32",
            "QW3_KVMEM_PREFIX_CACHE_TRACE": "1",
            "QW3_KVMEM_PERF_TRACE": "1",
            "QW3_KVMEM_TIER_TRACE": "1",
        }
    )
    return env


def wait_for_health(process: subprocess.Popen[Any], url: str, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    # The host often exports an HTTP(S) proxy for outbound traffic.  The QW3
    # health endpoint is a host-local Docker bridge address and must never be
    # sent through that proxy.  Pier's agent-side proxy remains untouched.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"QW3 exited during startup with code {process.returncode}")
        try:
            with opener.open(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"QW3 health check timed out: {last_error}")


def stop_process(process: subprocess.Popen[Any], timeout: float = 30.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def pier_command(task_id: str, task_dir: Path, task_artifact: Path, api_base: str) -> list[str]:
    jobs_dir = task_artifact / "pier_jobs"
    return [
        str(PIER),
        "run",
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        task_id,
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--yes",
        "--path",
        str(task_dir),
        "--agent",
        "mini-swe-agent",
        "--model",
        "openai/Qwen3.8-27B",
        "--agent-kwarg",
        "version=2.4.6",
        "--agent-kwarg",
        "model_class=litellm",
        "--agent-kwarg",
        "cost_limit=0",
        "--agent-kwarg",
        'model_kwargs={"drop_params":true,"extra_headers":{"x-qw3-harness":"mini-swe-agent"}}',
        "--agent-env",
        "OPENAI_API_KEY=local-qw3",
        "--agent-env",
        f"OPENAI_BASE_URL={api_base}",
        "--agent-env",
        "MSWEA_COST_TRACKING=ignore_errors",
    ]


def trial_result(task_artifact: Path) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((task_artifact / "pier_jobs").glob("*/*/result.json")):
        try:
            data = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        verifier = data.get("verifier_result")
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        if isinstance(rewards, dict) and isinstance(rewards.get("reward"), (int, float)):
            candidates.append((path, data))
    if len(candidates) != 1:
        return None
    return candidates[0]


def completed_summary(task_artifact: Path) -> dict[str, Any] | None:
    path = task_artifact / "summary.json"
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("status") == "completed" and "reward" in data else None


def git_snapshot() -> dict[str, Any]:
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "status_porcelain": run_text(["git", "status", "--porcelain"], check=False),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=HERE / "recordings")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--host", default="172.17.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lock, task_spec = validate_environment(args.tasks_file.resolve())
    selected = [item for item in task_spec["tasks"] if not args.only or item["task_id"] in args.only]
    unknown = sorted(set(args.only) - {item["task_id"] for item in task_spec["tasks"]})
    if unknown:
        raise RuntimeError(f"Unknown --only task(s): {', '.join(unknown)}")

    run_dir = args.output_root.resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    api_base = f"http://{args.host}:{args.port}/v1"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("task_ids") != [item["task_id"] for item in selected]:
            raise RuntimeError("Existing run manifest has a different task selection")
    else:
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "run_name": args.run_name,
            "method": "official Pier + mini-swe-agent; one attempt and fresh QW3 per task",
            "task_ids": [item["task_id"] for item in selected],
            "historical_score": task_spec["historical_score"],
            "locks": lock,
            "git": git_snapshot(),
            "artifacts": {
                "qw3_binary": str(QW3),
                "qw3_binary_sha256": sha256(QW3),
                "model": str(MODEL),
                "model_size_bytes": MODEL.stat().st_size,
            },
            "qwen_command": qwen_command(args.host, args.port),
            "api_base": api_base,
        }
        write_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        for item in selected:
            print("TASK", item["task_id"])
            print("PIER", json.dumps(pier_command(item["task_id"], DEEPSWE_TASKS / item["task_id"], run_dir / item["task_id"], api_base)))
        return 0

    for ordinal, item in enumerate(selected, start=1):
        task_id = item["task_id"]
        task_artifact = run_dir / task_id
        task_artifact.mkdir(parents=True, exist_ok=True)
        existing = completed_summary(task_artifact)
        if existing is not None:
            print(f"[{ordinal}/{len(selected)}] SKIP completed {task_id}: reward={existing['reward']}", flush=True)
            continue

        print(f"[{ordinal}/{len(selected)}] START {task_id}", flush=True)
        qwen_log_path = task_artifact / "qw3_server.log"
        pier_log_path = task_artifact / "pier_console.log"
        qwen_log = qwen_log_path.open("ab", buffering=0)
        server = subprocess.Popen(
            qwen_command(args.host, args.port),
            cwd=REPO,
            env=qwen_environment(),
            stdout=qwen_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started_at = utc_now()
        pier_exit: int | None = None
        try:
            wait_for_health(server, f"http://{args.host}:{args.port}/health")
            command = pier_command(task_id, DEEPSWE_TASKS / task_id, task_artifact, api_base)
            write_json(
                task_artifact / "attempt.json",
                {
                    "task_id": task_id,
                    "started_at": started_at,
                    "historical_reward": item["historical_reward"],
                    "pier_command": command,
                    "qwen_pid": server.pid,
                },
            )
            with pier_log_path.open("ab", buffering=0) as pier_log:
                pier_exit = subprocess.run(
                    command,
                    cwd=REPO,
                    stdout=pier_log,
                    stderr=subprocess.STDOUT,
                ).returncode
        finally:
            stop_process(server)
            qwen_log.close()

        parsed = trial_result(task_artifact)
        if parsed is None:
            write_json(
                task_artifact / "summary.json",
                {
                    "status": "infrastructure_error",
                    "task_id": task_id,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "pier_exit_code": pier_exit,
                    "reason": "Pier did not produce exactly one trial result with a numeric verifier reward",
                },
            )
            raise RuntimeError(f"{task_id}: no valid official verifier result; inspect {task_artifact}")

        result_path, result = parsed
        if result.get("exception_info") is not None:
            write_json(
                task_artifact / "summary.json",
                {
                    "status": "infrastructure_error",
                    "task_id": task_id,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "pier_exit_code": pier_exit,
                    "trial_result": str(result_path),
                    "exception_info": result["exception_info"],
                },
            )
            raise RuntimeError(f"{task_id}: Pier trial ended with an exception; inspect {result_path}")

        reward = result["verifier_result"]["rewards"]["reward"]
        summary = {
            "status": "completed",
            "task_id": task_id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "pier_exit_code": pier_exit,
            "reward": reward,
            "historical_reward": item["historical_reward"],
            "trial_result": str(result_path),
        }
        write_json(task_artifact / "summary.json", summary)
        print(f"[{ordinal}/{len(selected)}] DONE {task_id}: reward={reward}", flush=True)

    summaries = [completed_summary(run_dir / item["task_id"]) for item in selected]
    aggregate = {
        "status": "completed" if all(summary is not None for summary in summaries) else "partial",
        "updated_at": utc_now(),
        "completed": sum(summary is not None for summary in summaries),
        "passes": sum(float(summary["reward"]) == 1.0 for summary in summaries if summary),
        "total": len(selected),
        "tasks": summaries,
    }
    write_json(run_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
