#!/usr/bin/env python3
"""Wait for a complete KVMem requestplan10 run, then start dense compaction.

This is intentionally a completion gate rather than a generic retry daemon.
It never promotes partial/failed KVMem results and never reruns a scored task.
The dense baseline starts only after all ten authoritative summaries exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = REPO / ".venv-deepswe-pier" / "bin" / "python"
RUNNER = HERE / "run_requestplan10.py"
DASHBOARD = HERE / "dashboard.py"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--target-run", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dashboard-port", type=int, default=64366)
    args = parser.parse_args()

    source = HERE / "recordings" / args.source_run
    target = HERE / "recordings" / args.target_run
    print(
        f"[{stamp()}] waiting for complete KVMem run: {source}",
        flush=True,
    )
    last_state: tuple[object, ...] | None = None
    while True:
        summary = read_json(source / "summary.json")
        state = (
            summary.get("status"),
            summary.get("completed"),
            summary.get("passes"),
            summary.get("total"),
        )
        if state != last_state:
            print(f"[{stamp()}] source state={state}", flush=True)
            last_state = state
        if (
            summary.get("status") == "completed"
            and summary.get("completed") == 10
            and summary.get("total") == 10
        ):
            break
        time.sleep(max(10, args.poll_seconds))

    print(
        f"[{stamp()}] source complete; launching dense 256K compaction run: "
        f"{target}",
        flush=True,
    )
    target.mkdir(parents=True, exist_ok=True)
    dashboard_log = (target / "dashboard.log").open("ab", buffering=0)
    dashboard = subprocess.Popen(
        [
            str(PYTHON),
            str(DASHBOARD),
            "--run",
            str(target),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.dashboard_port),
        ],
        cwd=REPO,
        stdout=dashboard_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(
        f"[{stamp()}] dense dashboard pid={dashboard.pid} "
        f"url=http://127.0.0.1:{args.dashboard_port}/",
        flush=True,
    )
    command = [
        str(PYTHON),
        str(RUNNER),
        "--run-name",
        args.target_run,
        "--mode",
        "dense-compaction",
    ]
    os.chdir(REPO)
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
