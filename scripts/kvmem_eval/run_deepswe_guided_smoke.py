#!/usr/bin/env python3
"""Run selected DeepSWE checkpoint continuations against guided 64K KVMem.

The model service is intentionally external so one locked Qwen3.8 process is
reused for all samples.  Each sample receives its own proxy, terminal capture,
session export, patch, targeted tests, hidden verifier, and metrics document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any

import yaml


REPO = pathlib.Path("/home/chaidi/qw3")
REGISTRY = REPO / "benchmark/kvmem_deepswe_batch/registry.yaml"
EXTERNAL_ARM = REPO / "benchmark/kvmem_deepswe_batch/run_external_arm.sh"
SUMMARIZE = REPO / "benchmark/opencode_deepswe_long/summarize_condition.py"
DEFAULT_BINARY = REPO / "build-query-attention-probe-cuda/qw3"
DEFAULT_MODEL = REPO / "models/Qwen3.8-27B-Q8_0.gguf"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else REPO / path


def no_proxy_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if extra:
        env.update(extra)
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "ALL_PROXY", "all_proxy",
    ):
        env.pop(key, None)
    return env


def registry_rows() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return {str(row["task_id"]): row for row in document["jobs"]}


def run_logged(
    command: list[str], log: pathlib.Path, env: dict[str, str] | None = None
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        stream.write(("\n$ " + " ".join(command) + "\n").encode())
        stream.flush()
        return subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def verify(row: dict[str, Any], arm: pathlib.Path) -> int:
    patch = arm / "model.patch"
    if not patch.exists():
        return 125
    verifier_dir = arm / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker", "run", "--rm", "--network", "none",
        "--cpus", "2", "--memory", "8g",
        "-v", f"{patch}:/logs/artifacts/model.patch:ro",
        "-v", f"{verifier_dir}:/logs/verifier",
        str(row["verifier_image"]), "/tests/test.sh",
    ]
    status = run_logged(
        command, arm / "verifier.console.log", no_proxy_environment()
    )
    (arm / "verifier.exit").write_text(f"{status}\n", encoding="utf-8")
    run_logged(
        ["python3", str(SUMMARIZE), str(arm)],
        arm / "metrics.stdout",
    )
    return status


def server_evidence(arm: pathlib.Path) -> dict[str, Any]:
    path = arm / "qw3_server.timed.log"
    text = path.read_text(encoding="utf-8", errors="replace") \
        if path.exists() else ""
    mandatory = [
        int(value) for value in re.findall(r"mandatory_tokens=(\d+)", text)
    ]
    root_spans = [
        int(value) for value in re.findall(r"spans_root=(\d+)", text)
    ]
    root_blocks = [
        int(value) for value in re.findall(r"blocks_root=(\d+)", text)
    ]
    return {
        "timed_log_bytes": path.stat().st_size if path.exists() else 0,
        "http_500_count": len(re.findall(r"POST .* -> 500", text)),
        "http_413_count": len(re.findall(r"POST .* -> 413", text)),
        "boundary_guided_query_count": text.count(
            "native kvmem guided query:"
        ),
        "middecode_guided_refresh_count": text.count(
            "native kvmem middecode guided reselect: refresh="
        ),
        "cross_tool_guided_refresh_count": text.count(
            "trigger=cross-tool-turn"
        ),
        "semantic_selection_count": text.count(
            "[kvmem-reselect-perf] kind=semantic"
        ),
        "mandatory_tokens_min": min(mandatory) if mandatory else None,
        "mandatory_tokens_max": max(mandatory) if mandatory else None,
        "harness_pin_request_count": len(root_spans),
        "root_task_pin_request_count": sum(value > 0 for value in root_spans),
        "all_harness_requests_pin_root_task": (
            bool(root_spans) and all(value > 0 for value in root_spans)
        ),
        "root_task_blocks_min": min(root_blocks) if root_blocks else None,
        "root_task_blocks_max": max(root_blocks) if root_blocks else None,
    }


def verifier_grade(metrics: Any) -> dict[str, Any]:
    """Return a strict DeepSWE hidden-verifier grade.

    DeepSWE verifier containers deliberately exit zero after writing metrics,
    even when tests fail.  Process status therefore proves only that grading
    completed; task success requires reward=1 and complete F2P/P2P counts.
    """
    verifier = metrics.get("verifier") if isinstance(metrics, dict) else None
    if not isinstance(verifier, dict):
        return {
            "passed": False,
            "reason": "missing verifier metrics",
            "reward": None,
            "f2p": None,
            "p2p": None,
        }

    reward = verifier.get("reward")
    f2p_passed = verifier.get("f2p_passed")
    f2p_total = verifier.get("f2p_total")
    p2p_passed = verifier.get("p2p_passed")
    p2p_total = verifier.get("p2p_total")
    counts_are_complete = all(
        isinstance(value, int)
        for value in (f2p_passed, f2p_total, p2p_passed, p2p_total)
    ) and f2p_total >= 0 and p2p_total >= 0
    passed = (
        reward == 1
        and counts_are_complete
        and f2p_passed == f2p_total
        and p2p_passed == p2p_total
    )
    return {
        "passed": passed,
        "reason": "full hidden verifier pass" if passed else
            "reward/F2P/P2P not complete",
        "reward": reward,
        "f2p": {"passed": f2p_passed, "total": f2p_total},
        "p2p": {"passed": p2p_passed, "total": p2p_total},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--proxy-port-base", type=int, default=18120)
    parser.add_argument(
        "--tasks",
        default="expr-try-catch-errors,termenv-preserve-ansi-resets",
    )
    args = parser.parse_args()

    root = pathlib.Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    server_log = pathlib.Path(args.server_log).resolve()
    if not server_log.exists():
        raise SystemExit(f"server log does not exist: {server_log}")
    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve()
    for label, path in (("binary", binary), ("model", model)):
        if not path.is_file():
            raise SystemExit(f"{label} does not exist: {path}")
    build_evidence = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=REPO,
                text=True,
            ).strip()
        ),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "model": str(model),
        "model_bytes": model.stat().st_size,
        "model_sha256": sha256_file(model),
    }
    (root / "build_evidence.json").write_text(
        json.dumps(build_evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rows = registry_rows()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    missing = [task for task in tasks if task not in rows]
    if missing:
        raise SystemExit(f"tasks are absent from registry: {missing}")

    results: list[dict[str, Any]] = []
    for index, task_id in enumerate(tasks):
        row = rows[task_id]
        arm = root / task_id / "kvmem_1m_guided"
        arm.mkdir(parents=True, exist_ok=True)
        env = no_proxy_environment(
            {
                "QW3_AGENT_IMAGE": str(row["agent_image"]),
                "QW3_BASE_COMMIT": str(row["base_commit"]),
                "QW3_TASK_HOOKS": str(resolve(row["task_hooks"])),
                "QW3_MAX_WALL_TIME": "unlimited",
                "LOOP_GUARD_REPEATS": "10",
            }
        )
        (root / "active_task.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "index": index + 1,
                    "total": len(tasks),
                    "status": "running",
                    "started_unix": time.time(),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        started = time.monotonic()
        arm_status = run_logged(
            [
                "bash", str(EXTERNAL_ARM), "kvmem_1m_guided", str(arm),
                str(resolve(row["checkpoint"])),
                str(resolve(row["continuation"])),
                str(args.server_port), str(args.proxy_port_base + index),
                str(server_log),
            ],
            arm / "guided_smoke_runner.log",
            env,
        )
        verifier_status = verify(row, arm)
        elapsed = time.monotonic() - started
        metrics_path = arm / "metrics.json"
        metrics = None
        metrics_error = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                metrics_error = str(error)
        grade = verifier_grade(metrics)
        results.append(
            {
                "task_id": task_id,
                "arm_status": arm_status,
                "verifier_status": verifier_status,
                "elapsed_sec": elapsed,
                "passed": (
                    arm_status == 0
                    and verifier_status == 0
                    and grade["passed"]
                ),
                "artifact_dir": str(arm),
                "metrics": metrics,
                "metrics_error": metrics_error,
                "verifier_grade": grade,
                "server_evidence": server_evidence(arm),
            }
        )
        (root / "progress.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    (root / "active_task.json").write_text(
        json.dumps({"status": "finished", "total": len(tasks)}, indent=2) +
        "\n",
        encoding="utf-8",
    )

    report = {
        "schema": "qw3.deepswe_guided_kvmem_smoke.v2",
        "model": "Qwen3.8-27B-Q8_0.gguf",
        "logical_context_tokens": 1048576,
        "selection_budget_tokens": 65536,
        "generation_reserve_tokens": 32768,
        "block_tokens": 128,
        "guided_reselect": "both",
        "update_mode": "step",
        "build_evidence": build_evidence,
        "tasks": results,
        "passed": all(row["passed"] for row in results),
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
