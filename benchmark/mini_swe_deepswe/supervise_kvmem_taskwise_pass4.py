#!/usr/bin/env python3
"""Run the requestplan10 KVMem pass@4 queue one task at a time.

The OPA task is already being evaluated independently.  This supervisor first
waits for (and, if necessary, resumes) its four accepted rollouts.  It then
runs each of the remaining nine tasks at seeds 73, 1000, 2000, and 3000 before
moving to the next task.  Every new rollout uses the pinned QW3 executable and
a 4096-token guided-query cap.

The underlying Pier/mini-swe-agent execution and verifier remain owned by
``run_requestplan10.py``.  This file only supplies a persistent, resumable
task-major schedule and validates artifacts before advancing.
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
HERE = Path(__file__).resolve().parent
PYTHON = REPO / ".venv-deepswe-pier" / "bin" / "python"
RUNNER = HERE / "run_requestplan10.py"
TASK_SPEC = HERE / "requestplan10.json"
RECORDINGS = HERE / "recordings"
GUIDED_QUERY_TOKENS = 4096
TASK_CPUS = 8
SEEDS = (73, 1000, 2000, 3000)
OPA_TASK = "opa-template-string-reconstruction"
OPA_RUNS = {
    73: "requestplan10_kvmem_opa_r1strategy_s73_20260825",
    1000: "requestplan10_kvmem_opa_guided4k_v2_s1000_20260825",
    2000: "requestplan10_kvmem_opa_guided4k_s2000_20260825",
    3000: "requestplan10_kvmem_opa_guided4k_s3000_20260825",
}
PINNED_QW3 = REPO / "build" / (
    "qw3-deepswe-guided4k-"
    "0dc6ebc9d24cba719260431d489636e2f7b193d1bd15ed53898181f56cab4afb"
)
EXPECTED_QW3_SHA256 = (
    "0dc6ebc9d24cba719260431d489636e2f7b193d1bd15ed53898181f56cab4afb"
)
STATUS_PATH = RECORDINGS / "requestplan10_kvmem_taskwise_pass4_status.json"
RUN_PREFIX = "requestplan10_kvmem_guided4k_taskwise"
ROUND_VIEWS = {
    seed: RECORDINGS / f"requestplan10_kvmem_guided4k_taskwise_round_s{seed}_20260825"
    for seed in SEEDS
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def update_status(**values: Any) -> None:
    payload = read_json(STATUS_PATH)
    payload.update(values)
    payload["updated_at"] = utc_now()
    atomic_write_json(STATUS_PATH, payload)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(task_id: str, seed: int) -> str:
    return f"{RUN_PREFIX}_{task_id}_s{seed}_20260825"


def prepare_round_views(task_ids: list[str]) -> None:
    """Expose task-major artifacts as four dashboard-friendly round views."""
    source_manifest = read_json(
        RECORDINGS / OPA_RUNS[73] / "manifest.json"
    )
    for seed, view in ROUND_VIEWS.items():
        view.mkdir(parents=True, exist_ok=True)
        manifest = dict(source_manifest)
        manifest.update(
            {
                "run_name": view.name,
                "execution_mode": "kvmem",
                "method": (
                    "official Pier + mini-swe-agent; independent task-major "
                    "KVMem rollouts with a 4096-token guided-query cap"
                ),
                "task_ids": task_ids,
                "rollout_seeds": [seed],
                "taskwise_view": True,
                "guided_query_tokens": GUIDED_QUERY_TOKENS,
                "expected_qw3_sha256": EXPECTED_QW3_SHA256,
            }
        )
        artifacts = dict(manifest.get("artifacts") or {})
        artifacts.update(
            {
                "qw3_binary": str(PINNED_QW3),
                "qw3_binary_sha256": EXPECTED_QW3_SHA256,
            }
        )
        manifest["artifacts"] = artifacts
        atomic_write_json(view / "manifest.json", manifest)
        for task_id in task_ids:
            source_run = (
                OPA_RUNS[seed]
                if task_id == OPA_TASK
                else run_name(task_id, seed)
            )
            target = RECORDINGS / source_run / task_id
            link = view / task_id
            relative_target = Path(os.path.relpath(target, start=view))
            if link.is_symlink():
                if Path(os.readlink(link)) == relative_target:
                    continue
                link.unlink()
            elif link.exists():
                raise RuntimeError(f"round-view path is not a symlink: {link}")
            link.symlink_to(relative_target, target_is_directory=True)


def runner_alive(name: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "run_requestplan10.py"],
        check=False,
        text=True,
        capture_output=True,
    )
    return any(name in line for line in result.stdout.splitlines())


def qwen_command_value(manifest: dict[str, Any], option: str) -> str | None:
    command = manifest.get("qwen_command")
    if not isinstance(command, list) or option not in command:
        return None
    index = command.index(option) + 1
    return str(command[index]) if index < len(command) else None


def validate_single(
    task_id: str,
    seed: int,
    name: str,
    *,
    legacy_opa_allowed: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    run = RECORDINGS / name
    manifest = read_json(run / "manifest.json")
    aggregate = read_json(run / "summary.json")
    task_summary = read_json(run / task_id / "summary.json")
    if not manifest:
        return False, "manifest_missing", {}
    if manifest.get("execution_mode") != "kvmem":
        return False, "execution_mode_not_kvmem", {}
    if manifest.get("task_ids") != [task_id]:
        return False, f"task_selection_mismatch:{manifest.get('task_ids')}", {}
    if seed not in manifest.get("rollout_seeds", []):
        return False, f"seed_mismatch:{manifest.get('rollout_seeds')}", {}
    if not legacy_opa_allowed:
        guided = qwen_command_value(
            manifest, "--kvmem-guided-query-tokens"
        )
        if guided != str(GUIDED_QUERY_TOKENS):
            return False, f"guided_query_cap_mismatch:{guided}", {}
        manifest_sha = (manifest.get("artifacts") or {}).get(
            "qw3_binary_sha256"
        )
        if manifest_sha != EXPECTED_QW3_SHA256:
            return False, f"binary_sha_mismatch:{manifest_sha}", {}
    if aggregate.get("status") != "completed" or aggregate.get("completed") != 1:
        return False, "aggregate_incomplete", aggregate
    if task_summary.get("status") != "completed":
        return False, "task_incomplete", task_summary
    reward = task_summary.get("reward")
    if not isinstance(reward, (int, float)):
        return False, "reward_missing", task_summary
    if task_summary.get("rollout_seed") != seed:
        return False, f"task_seed_mismatch:{task_summary.get('rollout_seed')}", {}
    return True, "complete", {
        "reward": float(reward),
        "finished_at": task_summary.get("finished_at"),
        "run_name": name,
    }


def launch_or_resume(task_id: str, seed: int, name: str) -> None:
    environment = os.environ.copy()
    environment["QW3_DEEPSWE_BINARY"] = str(PINNED_QW3)
    result = subprocess.run(
        [
            str(PYTHON),
            str(RUNNER),
            "--run-name",
            name,
            "--mode",
            "kvmem",
            "--seed",
            str(seed),
            "--guided-query-tokens",
            str(GUIDED_QUERY_TOKENS),
            "--task-cpus",
            str(TASK_CPUS),
            "--only",
            task_id,
        ],
        cwd=REPO,
        env=environment,
        check=False,
    )
    update_status(last_runner_exit_code=result.returncode)


def ensure_rollout(
    task_id: str,
    seed: int,
    name: str,
    *,
    legacy_opa_allowed: bool = False,
) -> dict[str, Any]:
    while True:
        complete, reason, evidence = validate_single(
            task_id,
            seed,
            name,
            legacy_opa_allowed=legacy_opa_allowed,
        )
        update_status(
            state="complete" if complete else "waiting",
            current_task=task_id,
            current_seed=seed,
            current_run=name,
            current_reason=reason,
            current_evidence=evidence,
        )
        if complete:
            print(
                f"[complete] task={task_id} seed={seed} evidence={evidence}",
                flush=True,
            )
            return evidence
        if runner_alive(name):
            time.sleep(30)
            continue
        if sha256(PINNED_QW3) != EXPECTED_QW3_SHA256:
            raise RuntimeError(
                f"pinned QW3 hash mismatch: {sha256(PINNED_QW3)}"
            )
        print(
            f"[launch] task={task_id} seed={seed} run={name} reason={reason}",
            flush=True,
        )
        update_status(state="launching")
        launch_or_resume(task_id, seed, name)
        time.sleep(5)


def main() -> int:
    task_spec = read_json(TASK_SPEC)
    task_ids = [str(item["task_id"]) for item in task_spec.get("tasks", [])]
    if len(task_ids) != 10 or OPA_TASK not in task_ids:
        raise RuntimeError(f"unexpected requestplan10 task list: {task_ids}")
    remaining_tasks = [task for task in task_ids if task != OPA_TASK]
    prepare_round_views(task_ids)
    update_status(
        supervisor_pid=os.getpid(),
        state="starting",
        schedule="task-major",
        guided_query_tokens=GUIDED_QUERY_TOKENS,
        seeds=list(SEEDS),
        tasks=task_ids,
        pinned_qw3=str(PINNED_QW3),
        expected_qw3_sha256=EXPECTED_QW3_SHA256,
        round_views={str(seed): str(path) for seed, path in ROUND_VIEWS.items()},
    )

    results: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        evidence = ensure_rollout(
            OPA_TASK,
            seed,
            OPA_RUNS[seed],
            legacy_opa_allowed=(seed == 73),
        )
        results.setdefault(OPA_TASK, {})[str(seed)] = evidence
        update_status(results=results)

    for task_id in remaining_tasks:
        for seed in SEEDS:
            evidence = ensure_rollout(
                task_id, seed, run_name(task_id, seed)
            )
            results.setdefault(task_id, {})[str(seed)] = evidence
            update_status(results=results)

    update_status(
        state="complete",
        current_task=None,
        current_seed=None,
        current_run=None,
        results=results,
    )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
