#!/usr/bin/env python3
"""Supervise the remaining independent KVMem DeepSWE rollouts.

This is intentionally a run-level supervisor, not a second task harness.  The
authoritative work remains in ``run_requestplan10.py``.  This wrapper waits for
an already-running round, resumes an incomplete round after an unexpected
runner exit, and starts the next seed only after validating all ten accepted
results and their per-task QW3 binary hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PYTHON = REPO / ".venv-deepswe-pier" / "bin" / "python"
RUNNER = Path(__file__).with_name("run_requestplan10.py")
RECORDINGS = Path(__file__).with_name("recordings")
QW3 = REPO / "build" / "qw3"
EXPECTED_QW3_SHA256 = (
    "5561be1da39096b76a641cfb9e0252d32b7d679b6956f7470fd289ab479f451e"
)
ROUNDS = (
    ("R2", 73, "requestplan10_kvmem_r2_s73_source_rollback_20260824"),
    ("R3", 1000, "requestplan10_kvmem_r3_s1000_source_rollback_20260824"),
    ("R4", 2000, "requestplan10_kvmem_r4_s2000_source_rollback_20260824"),
)
STATUS_PATH = RECORDINGS / "requestplan10_kvmem_pass4_sequence_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_status(**values: Any) -> None:
    payload = read_json(STATUS_PATH)
    payload.update(values)
    payload["updated_at"] = utc_now()
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(STATUS_PATH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_alive(run_name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "run_requestplan10.py"],
        check=False,
        text=True,
        capture_output=True,
    )
    return any(run_name in line for line in result.stdout.splitlines())


def accepted_attempt(task_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    result_value = summary.get("trial_result")
    if not isinstance(result_value, str):
        return {}
    result_path = Path(result_value)
    for parent in (result_path.parent, *result_path.parents):
        if parent == task_dir.parent:
            break
        attempt_path = parent / "attempt.json"
        if attempt_path.is_file():
            return read_json(attempt_path)
    return {}


def validate_round(seed: int, run_name: str) -> tuple[bool, str, dict[str, Any]]:
    run = RECORDINGS / run_name
    manifest = read_json(run / "manifest.json")
    summary = read_json(run / "summary.json")
    if not manifest:
        return False, "manifest_missing", {}
    if manifest.get("execution_mode") != "kvmem":
        return False, "execution_mode_not_kvmem", {}
    manifest_sha = (manifest.get("artifacts") or {}).get("qw3_binary_sha256")
    if manifest_sha != EXPECTED_QW3_SHA256:
        return False, f"manifest_binary_sha_mismatch:{manifest_sha}", {}
    if manifest.get("rollout_seeds") != [seed]:
        return False, f"manifest_seed_mismatch:{manifest.get('rollout_seeds')}", {}
    if summary.get("status") != "completed" or summary.get("completed") != 10:
        return False, "round_incomplete", summary

    task_summaries = []
    for task_id in manifest.get("task_ids", []):
        task_dir = run / str(task_id)
        task_summary = read_json(task_dir / "summary.json")
        if task_summary.get("status") != "completed" or not isinstance(
            task_summary.get("reward"), (int, float)
        ):
            return False, f"task_incomplete:{task_id}", summary
        if task_summary.get("rollout_seed") != seed:
            return False, f"task_seed_mismatch:{task_id}", summary
        attempt = accepted_attempt(task_dir, task_summary)
        if attempt.get("qwen_binary_sha256") != EXPECTED_QW3_SHA256:
            return False, f"task_binary_sha_mismatch:{task_id}", summary
        task_summaries.append(task_summary)

    passes = sum(float(item["reward"]) == 1.0 for item in task_summaries)
    if summary.get("passes") != passes:
        return False, "aggregate_pass_count_mismatch", summary
    return True, "complete", {"completed": 10, "passes": passes}


def run_round(label: str, seed: int, run_name: str) -> None:
    while True:
        complete, reason, aggregate = validate_round(seed, run_name)
        write_status(
            current_round=label,
            current_seed=seed,
            current_run=run_name,
            state="complete" if complete else "waiting",
            reason=reason,
            aggregate=aggregate,
        )
        if complete:
            print(f"[{label}] validated: {aggregate}", flush=True)
            return
        if runner_alive(run_name):
            time.sleep(30)
            continue

        actual_sha = sha256(QW3)
        if actual_sha != EXPECTED_QW3_SHA256:
            raise RuntimeError(
                f"refusing to start {label}: build/qw3 SHA is {actual_sha}, "
                f"expected {EXPECTED_QW3_SHA256}"
            )
        print(f"[{label}] starting/resuming seed={seed}: {reason}", flush=True)
        write_status(state="launching", reason=reason)
        result = subprocess.run(
            [
                str(PYTHON),
                str(RUNNER),
                "--run-name",
                run_name,
                "--mode",
                "kvmem",
                "--seed",
                str(seed),
            ],
            cwd=REPO,
            check=False,
        )
        write_status(state="runner_exited", runner_exit_code=result.returncode)
        time.sleep(5)


def main() -> int:
    write_status(
        supervisor_pid=os.getpid(),
        expected_qw3_sha256=EXPECTED_QW3_SHA256,
        rounds=[
            {"label": label, "seed": seed, "run_name": run_name}
            for label, seed, run_name in ROUNDS
        ],
        state="starting",
    )
    results: dict[str, Any] = {}
    for label, seed, run_name in ROUNDS:
        run_round(label, seed, run_name)
        _complete, _reason, aggregate = validate_round(seed, run_name)
        results[label] = {"seed": seed, **aggregate}
        write_status(completed_rounds=results)
    write_status(state="complete", current_round=None, results=results)
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
