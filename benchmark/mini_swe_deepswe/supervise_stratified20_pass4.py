#!/usr/bin/env python3
"""Resume the frozen stratified-20 KVMem evaluation one seed at a time."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = REPO / ".venv-deepswe-pier" / "bin" / "python"
RUNNER = HERE / "run_requestplan10.py"
TASKS = HERE / "stratified20_seed20260831.json"
RECORDINGS = HERE / "recordings"
PINNED_QW3 = REPO / "build" / "qw3-deepswe-strat20-d1b0971e76a861a1"
EXPECTED_QW3_SHA256 = (
    "d1b0971e76a861a1ed6eb712d0802a5d7823f0789c5ae56ac26b7cb8478f84d1"
)
SEEDS = (1000, 2000, 3000, 4000)
RUN_NAMES = {
    seed: f"stratified20_kvmem_official_r{ordinal}_s{seed}_20260831"
    for ordinal, seed in enumerate(SEEDS, start=1)
}
STATUS = RECORDINGS / "stratified20_kvmem_pass4_status.json"
CURRENT_CHILD: subprocess.Popen[Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_status(**values: Any) -> None:
    status = read_json(STATUS)
    status.update(values)
    status["updated_at"] = utc_now()
    write_json(STATUS, status)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_complete(seed: int) -> bool:
    run = RECORDINGS / RUN_NAMES[seed]
    summary = read_json(run / "summary.json")
    if summary.get("status") != "completed" or summary.get("completed") != 20:
        return False
    task_ids = [item["task_id"] for item in read_json(TASKS).get("tasks", [])]
    for task_id in task_ids:
        task = read_json(run / task_id / "summary.json")
        if task.get("status") != "completed":
            return False
        if task.get("rollout_seed") != seed:
            return False
        if not isinstance(task.get("reward"), (int, float)):
            return False
    return True


def runner_alive(run_name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "run_requestplan10.py"],
        check=False,
        text=True,
        capture_output=True,
    )
    return any(run_name in line for line in result.stdout.splitlines())


def run_seed(seed: int) -> int:
    global CURRENT_CHILD
    environment = os.environ.copy()
    environment["QW3_DEEPSWE_BINARY"] = str(PINNED_QW3)
    command = [
        str(PYTHON),
        str(RUNNER),
        "--run-name",
        RUN_NAMES[seed],
        "--tasks-file",
        str(TASKS),
        "--mode",
        "kvmem",
        "--seed",
        str(seed),
        "--guided-query-tokens",
        "4096",
        "--kvmem-budget",
        "131072",
        "--kvmem-prefill-budget",
        "131072",
        "--kvmem-gen-budget",
        "65536",
    ]
    CURRENT_CHILD = subprocess.Popen(command, cwd=REPO, env=environment)
    try:
        return CURRENT_CHILD.wait()
    finally:
        CURRENT_CHILD = None


def stop(signum: int, _frame: object) -> None:
    if CURRENT_CHILD is not None and CURRENT_CHILD.poll() is None:
        CURRENT_CHILD.send_signal(signal.SIGTERM)
        CURRENT_CHILD.wait(timeout=120)
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if sha256(PINNED_QW3) != EXPECTED_QW3_SHA256:
        raise RuntimeError("Pinned QW3 binary hash mismatch")
    update_status(
        state="running",
        supervisor_pid=os.getpid(),
        schedule="seed-major",
        seeds=list(SEEDS),
        run_names={str(seed): name for seed, name in RUN_NAMES.items()},
        tasks_file=str(TASKS),
        tasks_file_sha256=sha256(TASKS),
        qw3_binary=str(PINNED_QW3),
        qw3_binary_sha256=EXPECTED_QW3_SHA256,
    )
    for seed in SEEDS:
        while not run_complete(seed):
            update_status(state="waiting" if runner_alive(RUN_NAMES[seed]) else "launching", current_seed=seed)
            if runner_alive(RUN_NAMES[seed]):
                time.sleep(30)
                continue
            exit_code = run_seed(seed)
            update_status(last_runner_exit_code=exit_code)
            if exit_code in {130, 143}:
                return exit_code
            time.sleep(5)
        update_status(state="seed_complete", current_seed=seed)
    update_status(state="complete", current_seed=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
