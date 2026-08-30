#!/usr/bin/env python3
"""Restart a resumable DeepSWE batch after an unexpected runner failure.

The inner batch runner is already idempotent at completed-arm granularity.
This outer process therefore only restarts failures that are not explicit
protocol/inference pauses.  Run this supervisor under tmux (or another host
process manager) so an SSH disconnect cannot terminate the queue.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


NON_RESTARTABLE_PHASES = {
    "finished",
    "finished_with_infrastructure_errors",
    "paused_for_investigation",
    "stopped_by_user",
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def atomic_json(path: Path, value: Any, retry_seconds: float = 30.0) -> None:
    """Keep the durable supervisor alive while a full disk is remediated."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    while True:
        try:
            temporary.write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
            return
        except OSError as error:
            if error.errno != errno.ENOSPC:
                raise
            print(
                f"disk full while publishing {path}; retrying in "
                f"{retry_seconds:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(max(0.1, retry_seconds))


def restart_decision(experiment: Path, returncode: int) -> tuple[bool, str]:
    state = read_json(experiment / "state.json", {}) or {}
    phase = str(state.get("phase") or "unknown")
    if returncode == 0 or phase in {
        "finished",
        "finished_with_infrastructure_errors",
    }:
        return False, "completed"
    if phase in NON_RESTARTABLE_PHASES:
        return False, phase
    return True, f"unexpected_exit:{returncode}:phase={phase}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--backoff", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a batch command is required after --")
    if args.max_restarts < 0:
        parser.error("--max-restarts must be non-negative")

    experiment = args.experiment.resolve()
    status_path = experiment / "durable_supervisor.json"
    child: subprocess.Popen[bytes] | None = None
    stopping = False
    stopping_signal = signal.SIGTERM

    def stop_child(signum: int, _frame: object) -> None:
        nonlocal stopping, stopping_signal
        stopping = True
        stopping_signal = signum
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)

    attempts = 0
    while True:
        atomic_json(
            status_path,
            {
                "state": "starting",
                "supervisor_pid": os.getpid(),
                "attempt": attempts + 1,
                "command": command,
                "updated_at": time.time(),
            },
        )
        child = subprocess.Popen(command, start_new_session=True)
        atomic_json(
            status_path,
            {
                "state": "running",
                "supervisor_pid": os.getpid(),
                "runner_pid": child.pid,
                "attempt": attempts + 1,
                "command": command,
                "updated_at": time.time(),
            },
        )
        returncode = child.wait()
        restart, reason = restart_decision(experiment, returncode)
        if stopping:
            atomic_json(
                status_path,
                {
                    "state": "stopped",
                    "returncode": returncode,
                    "reason": "supervisor_signal",
                    "attempts": attempts + 1,
                    "updated_at": time.time(),
                },
            )
            return 128 + stopping_signal
        if not restart:
            atomic_json(
                status_path,
                {
                    "state": "finished" if reason == "completed" else "paused",
                    "returncode": returncode,
                    "reason": reason,
                    "attempts": attempts + 1,
                    "updated_at": time.time(),
                },
            )
            return returncode
        if attempts >= args.max_restarts:
            atomic_json(
                status_path,
                {
                    "state": "restart_exhausted",
                    "returncode": returncode,
                    "reason": reason,
                    "attempts": attempts + 1,
                    "updated_at": time.time(),
                },
            )
            return returncode or 1
        attempts += 1
        atomic_json(
            status_path,
            {
                "state": "restart_backoff",
                "returncode": returncode,
                "reason": reason,
                "attempts": attempts,
                "next_attempt": attempts + 1,
                "updated_at": time.time(),
            },
        )
        time.sleep(max(0.0, args.backoff))


if __name__ == "__main__":
    raise SystemExit(main())
