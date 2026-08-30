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
QW3 = Path(
    os.environ.get("QW3_DEEPSWE_BINARY", str(REPO / "build" / "qw3"))
).resolve()
MODEL = REPO / "models" / "Qwen3.8-27B-Q8_0.gguf"
DEEPSWE_TASKS = REPO / "benchmark" / "deep-swe" / "tasks"
DEEPSWE_MANIFEST = DEEPSWE_TASKS / "manifest.json"
RELAY_SCRIPT = HERE / "tcp_relay.py"
AGENT_ADAPTER = HERE / "reliable_mini_swe_agent.py"
AGENT_IMPORT_PATH = (
    "benchmark.mini_swe_deepswe.reliable_mini_swe_agent:ReliableMiniSweAgent"
)
COMPACT_AGENT_RUNTIME = HERE / "compacting_mini_swe_agent_runtime.py"
COMPACT_AGENT_IMPORT_PATH = (
    "benchmark.mini_swe_deepswe.reliable_mini_swe_agent:CompactingMiniSweAgent"
)
# This is the official base image of the first locked DeepSWE task.  Pier must
# use it for that task anyway, so the relay adds no independent image source.
RELAY_IMAGE = (
    "public.ecr.aws/d3j8x8q7/swe-bench-202605:"
    "kh71gkadwafw4ry4r6g37era0182qpms-v1.1"
)
SOURCE_TASK_LIST = (
    REPO / "benchmark" / "claude_deepswe_ab" / "tasks_requestplan_10_no_happy.json"
)
MAX_INFRASTRUCTURE_ATTEMPTS = 10
LOCAL_INFRASTRUCTURE_TIMEOUT_MULTIPLIER = 4.0
QW3_STARTUP_TIMEOUT_SEC = 1800.0
SCORED_AGENT_TERMINAL_EXCEPTIONS = {"AgentTimeoutError"}
# Pier 0.3.1 has no explicit "disable agent timeout" switch.  A very large,
# finite multiplier is operationally unbounded while remaining faithfully
# serializable in Pier's JSON artifacts (unlike IEEE infinity, which Pydantic
# serializes as null).  5400 seconds * 1,000,000 is over 171 years.
UNBOUNDED_AGENT_TIMEOUT_MULTIPLIER = 1_000_000.0


def local_reliability_policy(mode: str = "kvmem") -> dict[str, Any]:
    policy: dict[str, Any] = {
        "qwen_logical_context_tokens": 3_145_728,
        "qwen_cpu_tier_gib": 110,
        "qwen_startup_timeout_sec": QW3_STARTUP_TIMEOUT_SEC,
        "infrastructure_timeout_multiplier": (
            LOCAL_INFRASTRUCTURE_TIMEOUT_MULTIPLIER
        ),
        "max_infrastructure_attempts": MAX_INFRASTRUCTURE_ATTEMPTS,
        "mini_swe_agent": {
            "step_limit": 0,
            "cost_limit": 0,
            "wall_time_limit_seconds": 0,
            "max_consecutive_format_errors": 10,
            "shell_command_timeout_sec": 1800,
            "model_request_timeout_sec": 21600,
        },
    }
    if mode == "dense-compaction":
        policy["qwen_logical_context_tokens"] = 262_144
        policy["mini_swe_agent"]["compaction"] = {
            "enabled": True,
            "trigger_tokens": 220_000,
            "summary_max_tokens": 16_384,
            "keep_original_system_and_task": True,
            "keep_raw_tail": False,
        }
    return policy


def agent_import_path(mode: str) -> str:
    return (
        COMPACT_AGENT_IMPORT_PATH
        if mode == "dense-compaction"
        else AGENT_IMPORT_PATH
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

    required = [
        PIER,
        PYTHON,
        QW3,
        MODEL,
        DEEPSWE_MANIFEST,
        SOURCE_TASK_LIST,
        RELAY_SCRIPT,
        AGENT_ADAPTER,
        COMPACT_AGENT_RUNTIME,
    ]
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
        "pier_commit": run_text(
            [
                str(PYTHON),
                "-c",
                (
                    "import importlib.metadata as m,json; "
                    "d=json.loads(m.distribution('datacurve-pier').read_text('direct_url.json')); "
                    "print(d['vcs_info']['commit_id'])"
                ),
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
        "pier_commit": lock["pier"]["commit"],
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
    authoritative_task_ids = [item["task_id"] for item in read_json(SOURCE_TASK_LIST)["tasks"]]
    selected_task_ids = [item["task_id"] for item in task_spec["tasks"]]
    if selected_task_ids != authoritative_task_ids:
        raise RuntimeError(
            "Task IDs/order differ from the authoritative historical requestplan10 list"
        )
    for item in task_spec["tasks"]:
        task_dir = DEEPSWE_TASKS / item["task_id"]
        if not (task_dir / "task.toml").is_file():
            raise RuntimeError(f"DeepSWE task is missing: {item['task_id']}")
    return lock, task_spec


def qwen_command(
    host: str,
    port: int,
    mode: str = "kvmem",
    seed: int = 73,
    guided_query_tokens: int = 512,
    kvmem_budget: int = 131072,
    kvmem_prefill_budget: int = 131072,
    kvmem_gen_budget: int = 65536,
) -> list[str]:
    common = [
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
        str(seed),
        "--native-mtp-speculate",
        "--mtp-chain",
        "4",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if mode == "dense-compaction":
        return common + [
            "--ctx",
            "262144",
            "--continuous-batching",
            "--max-active",
            "1",
            "--max-pending",
            "8",
            # One MiniSWE trajectory is executed at a time, but LiteLLM may
            # briefly submit a retry before the prior HTTP worker has returned.
            # The continuous scheduler already limits physical execution to
            # max-active=1 and the paged KV pool to ctx.  Do not also charge
            # queued requests a full-ctx logical reservation, which would turn
            # that harmless overlap into a spurious HTTP 429.
            "--max-total-tokens",
            "0",
            "--prefix-cache",
            "--mtp-batched-draft",
            "--mtp-paged-prefix",
        ]
    return common + [
        "--ctx",
        "3145728",
        "--kvmem",
        "--kvmem-prefix-cache",
        "--kvmem-block-tokens",
        "32",
        "--kvmem-budget",
        str(kvmem_budget),
        "--kvmem-prefill-budget",
        str(kvmem_prefill_budget),
        "--kvmem-gen-budget",
        str(kvmem_gen_budget),
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
        str(guided_query_tokens),
        "--kvmem-middecode-trigger-tokens",
        "61440",
        "--kvmem-middecode-max-refreshes",
        "2",
        "--kvmem-immutable-k",
        "--kvmem-gpu-memory-ratio",
        "1.0",
        "--kvmem-cpu-gb",
        "110",
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
            # Accuracy control: use the pre-optimization MTP verifier paths.
            # Keep these explicit so a parent shell cannot silently turn an
            # agent rollout back into the experimental implementation.
        }
    )
    return env


def wait_for_health(
    process: subprocess.Popen[Any],
    url: str,
    timeout: float = QW3_STARTUP_TIMEOUT_SEC,
) -> None:
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


def stop_process_group(
    process: subprocess.Popen[Any] | None, timeout: float = 30.0
) -> None:
    """Stop a Pier process and every host-side child it launched.

    Pier invokes ``docker compose exec`` below its own process.  Killing only
    the runner leaves that exec process and the in-container MiniSWE agent
    alive, so a resumed attempt can silently share the same QW3 endpoint with
    the stale trajectory.  Every Pier child is started in a fresh session;
    terminate that whole process group before stopping the model service.
    """

    if process is None or process.poll() is not None:
        return
    for sig, wait_seconds in (
        (signal.SIGINT, timeout),
        (signal.SIGTERM, 10.0),
        (signal.SIGKILL, 10.0),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def cleanup_attempt_containers(attempt_root: Path) -> None:
    """Remove only containers owned by trials in one interrupted attempt."""

    projects = {
        trial.name.lower()
        for trial in attempt_root.glob("pier_jobs/*/*")
        if trial.is_dir()
    }
    for project in sorted(projects):
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-aq",
                "--filter",
                f"label=com.docker.compose.project={project}",
            ],
            cwd=REPO,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        container_ids = listed.stdout.split()
        if container_ids:
            subprocess.run(
                ["docker", "rm", "-f", *container_ids],
                cwd=REPO,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def start_relay(
    name: str,
    *,
    listen_host: str,
    listen_port: int,
    target_host: str,
    target_port: int,
) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "--network",
        "host",
        "--entrypoint",
        "python3",
        "-v",
        f"{RELAY_SCRIPT}:/opt/qw3/tcp_relay.py:ro",
        RELAY_IMAGE,
        "/opt/qw3/tcp_relay.py",
        "--listen-host",
        listen_host,
        "--listen-port",
        str(listen_port),
        "--target-host",
        target_host,
        "--target-port",
        str(target_port),
    ]
    run_text(command)


def stop_relay(name: str) -> None:
    subprocess.run(
        ["docker", "stop", "--time", "5", name],
        cwd=REPO,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def pier_command(
    task_id: str,
    task_dir: Path,
    task_artifact: Path,
    api_base: str,
    *,
    mode: str = "kvmem",
    unbounded_agent_timeout: bool = False,
    task_cpus: int | None = None,
) -> list[str]:
    jobs_dir = task_artifact / "pier_jobs"
    command = [
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
        "--timeout-multiplier",
        str(LOCAL_INFRASTRUCTURE_TIMEOUT_MULTIPLIER),
        "--yes",
    ]
    if task_cpus is not None:
        command.extend(["--override-cpus", str(task_cpus)])
    # Keep the solver timeout independent from the local-infrastructure
    # multiplier.  Local runs default to operationally unbounded; the explicit
    # official mode restores the task's original 90-minute value.
    command.extend(
        [
            "--agent-timeout-multiplier",
            str(
                UNBOUNDED_AGENT_TIMEOUT_MULTIPLIER
                if unbounded_agent_timeout
                else 1.0
            ),
        ]
    )
    command.extend(
        [
            "--path",
            str(task_dir),
            "--agent-import-path",
            agent_import_path(mode),
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
    )
    return command


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


def is_terminal_scored_outcome(result: dict[str, Any]) -> bool:
    """Return whether an official numeric score is a terminal model outcome.

    Pier verifies the workspace after an agent timeout and can therefore emit a
    complete numeric reward together with ``AgentTimeoutError``.  That timeout
    is a valid one-attempt model failure, not an infrastructure failure: the
    runner must record it and continue to the next task.  Other post-agent
    exceptions remain diagnostic stops until they are classified explicitly.
    """
    exception = result.get("exception_info")
    if exception is None:
        return True
    if not isinstance(exception, dict):
        return False
    return exception.get("exception_type") in SCORED_AGENT_TERMINAL_EXCEPTIONS


def attempt_artifact(task_artifact: Path, attempt: int) -> Path:
    """Return a stable artifact root for an infrastructure attempt.

    Attempt one retains the original layout.  Retry artifacts live under a
    separate directory so an interrupted or failed official Pier run is never
    overwritten and remains auditable.
    """
    if attempt == 1:
        return task_artifact
    return task_artifact / "infrastructure_retries" / f"attempt_{attempt:02d}"


def attempted_count(task_artifact: Path) -> int:
    count = int((task_artifact / "pier_jobs").exists())
    retry_root = task_artifact / "infrastructure_retries"
    if retry_root.is_dir():
        count += sum(
            1
            for path in retry_root.glob("attempt_*")
            if (path / "pier_jobs").exists()
        )
    return count


def retryable_pre_agent_infrastructure_failure(attempt_root: Path) -> bool:
    """Retry only an official environment failure before the agent ran.

    A model/serving failure after MiniSweAgent has started needs diagnosis and
    must not be hidden by an automatic rerun.  A numeric verifier result is
    handled before this function and is never retried, including reward zero.
    """
    if any(attempt_root.glob("pier_jobs/*/*/agent/mini-swe-agent.trajectory.json")):
        return False
    exceptions = list(attempt_root.glob("pier_jobs/*/*/exception.txt"))
    if not exceptions:
        return False
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in exceptions)
    markers = (
        "Docker compose command failed",
        "failed to download",
        "HTTP/2 stream",
        "Temporary failure resolving",
        "Could not resolve host",
        "connection reset",
        "network error",
        "timed out",
        "timeout",
    )
    return any(marker.lower() in text.lower() for marker in markers)


def completed_summary(task_artifact: Path) -> dict[str, Any] | None:
    path = task_artifact / "summary.json"
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return data if data.get("status") == "completed" and "reward" in data else None


def recover_terminal_scored_summary(
    task_artifact: Path, task_id: str, historical_reward: Any
) -> dict[str, Any] | None:
    """Promote an already verified timeout result to a resumable completion.

    This is needed when resuming an older runner invocation that incorrectly
    wrote ``infrastructure_error`` after Pier had already produced a numeric
    verifier result for an agent timeout.
    """
    roots = [task_artifact]
    retry_root = task_artifact / "infrastructure_retries"
    if retry_root.is_dir():
        roots.extend(sorted(retry_root.glob("attempt_*")))

    for current_artifact in reversed(roots):
        parsed = trial_result(current_artifact)
        if parsed is None or not is_terminal_scored_outcome(parsed[1]):
            continue
        result_path, result = parsed
        rewards = result["verifier_result"]["rewards"]
        attempt_meta_path = current_artifact / "attempt.json"
        attempt_meta = (
            read_json(attempt_meta_path) if attempt_meta_path.is_file() else {}
        )
        infrastructure_attempt = int(attempt_meta.get("infrastructure_attempt", 1))
        summary: dict[str, Any] = {
            "status": "completed",
            "task_id": task_id,
            "started_at": attempt_meta.get("started_at", result.get("started_at")),
            "finished_at": result.get("finished_at", utc_now()),
            "pier_exit_code": 0,
            "infrastructure_attempt": infrastructure_attempt,
            "reward": rewards["reward"],
            "historical_reward": historical_reward,
            "trial_result": str(result_path),
            "recovered_from_scored_result": True,
        }
        exception = result.get("exception_info")
        if isinstance(exception, dict):
            summary["agent_terminal_exception"] = {
                "exception_type": exception.get("exception_type"),
                "exception_message": exception.get("exception_message"),
            }
        write_json(task_artifact / "summary.json", summary)
        return summary
    return None


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
    parser.add_argument("--api-host", default="172.17.0.1.nip.io")
    parser.add_argument("--api-port", type=int, default=80)
    parser.add_argument(
        "--seed",
        type=int,
        default=73,
        help="QW3 sampling seed recorded for this rollout (default: 73)",
    )
    parser.add_argument(
        "--mode",
        choices=("kvmem", "dense-compaction"),
        default="kvmem",
        help=(
            "KVMem long-context run, or ordinary dense QW3 with automatic "
            "MiniSWE compaction at a 256K context limit"
        ),
    )
    parser.add_argument(
        "--guided-query-tokens",
        type=int,
        default=512,
        help=(
            "maximum generated KVMem guided-retrieval query length "
            "(default: 512; only used in kvmem mode)"
        ),
    )
    parser.add_argument(
        "--kvmem-budget",
        type=int,
        default=131072,
        help="KVMem semantic-selection budget in tokens (default: 131072)",
    )
    parser.add_argument(
        "--kvmem-prefill-budget",
        type=int,
        default=None,
        help=(
            "KVMem pressure-prefill budget in tokens; defaults to "
            "--kvmem-budget"
        ),
    )
    parser.add_argument(
        "--kvmem-gen-budget",
        type=int,
        default=65536,
        help="KVMem generation reserve in tokens (default: 65536)",
    )
    parser.add_argument(
        "--task-cpus",
        type=int,
        default=None,
        help="override Pier's task-container CPU limit",
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument(
        "--rerun",
        action="append",
        default=[],
        help="rerun a selected task even when a completed summary exists",
    )
    parser.set_defaults(unbounded_agent_timeout=True)
    parser.add_argument(
        "--unbounded-agent-timeout",
        dest="unbounded_agent_timeout",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--official-agent-timeout",
        dest="unbounded_agent_timeout",
        action="store_false",
        help="restore the locked task's official 90-minute agent deadline",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.kvmem_prefill_budget is None:
        args.kvmem_prefill_budget = args.kvmem_budget
    if args.kvmem_budget <= 0:
        raise RuntimeError("--kvmem-budget must be positive")
    if args.kvmem_prefill_budget <= 0:
        raise RuntimeError("--kvmem-prefill-budget must be positive")
    if args.kvmem_gen_budget <= 0:
        raise RuntimeError("--kvmem-gen-budget must be positive")

    lock, task_spec = validate_environment(args.tasks_file.resolve())
    selected = [item for item in task_spec["tasks"] if not args.only or item["task_id"] in args.only]
    unknown = sorted(set(args.only) - {item["task_id"] for item in task_spec["tasks"]})
    if unknown:
        raise RuntimeError(f"Unknown --only task(s): {', '.join(unknown)}")
    unknown_reruns = sorted(
        set(args.rerun) - {item["task_id"] for item in selected}
    )
    if unknown_reruns:
        raise RuntimeError(
            "Unknown or unselected --rerun task(s): " + ", ".join(unknown_reruns)
        )
    rerun_ids = set(args.rerun)

    run_dir = args.output_root.resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    api_base = f"http://{args.api_host}:{args.api_port}/v1"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("task_ids") != [item["task_id"] for item in selected]:
            raise RuntimeError("Existing run manifest has a different task selection")
        existing_mode = manifest.get("execution_mode", "kvmem")
        if existing_mode != args.mode:
            raise RuntimeError(
                f"Existing run uses mode={existing_mode}, requested mode={args.mode}"
            )
        manifest["agent_install_adapter"] = {
            "import_path": agent_import_path(args.mode),
            "path": str(AGENT_ADAPTER),
            "sha256": sha256(AGENT_ADAPTER),
            "runtime_path": (
                str(COMPACT_AGENT_RUNTIME)
                if args.mode == "dense-compaction"
                else None
            ),
            "runtime_sha256": (
                sha256(COMPACT_AGENT_RUNTIME)
                if args.mode == "dense-compaction"
                else None
            ),
            "scope": (
                "installer transport, slow-local-run reliability, and "
                "automatic dense-context compaction"
                if args.mode == "dense-compaction"
                else "installer transport and slow-local-run reliability hardening"
            ),
        }
        manifest["last_resumed_at"] = utc_now()
        manifest["last_resume_git"] = git_snapshot()
        manifest["last_resume_agent_timeout_policy"] = (
            "unbounded" if args.unbounded_agent_timeout else "task-default"
        )
        manifest["last_resume_rerun_tasks"] = sorted(rerun_ids)
        manifest["last_resume_seed"] = args.seed
        rollout_seeds = manifest.setdefault("rollout_seeds", [])
        if args.seed not in rollout_seeds:
            rollout_seeds.append(args.seed)
        manifest["qwen_command"] = qwen_command(
            args.host,
            args.port,
            args.mode,
            args.seed,
            args.guided_query_tokens,
            args.kvmem_budget,
            args.kvmem_prefill_budget,
            args.kvmem_gen_budget,
        )
        manifest["local_reliability_policy"] = local_reliability_policy(args.mode)
        write_json(manifest_path, manifest)
    else:
        manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "run_name": args.run_name,
            "execution_mode": args.mode,
            "method": (
                "official Pier + mini-swe-agent with automatic compaction; "
                "one attempt and fresh dense QW3 per task"
                if args.mode == "dense-compaction"
                else "official Pier + mini-swe-agent; one attempt and fresh QW3 per task"
            ),
            "task_ids": [item["task_id"] for item in selected],
            "historical_score": task_spec["historical_score"],
            "locks": lock,
            "git": git_snapshot(),
            "artifacts": {
                "qw3_binary": str(QW3),
                "qw3_binary_sha256": sha256(QW3),
                "model": str(MODEL),
                "model_size_bytes": MODEL.stat().st_size,
                "agent_install_adapter": str(AGENT_ADAPTER),
                "agent_install_adapter_sha256": sha256(AGENT_ADAPTER),
            },
            "qwen_command": qwen_command(
                args.host,
                args.port,
                args.mode,
                args.seed,
                args.guided_query_tokens,
                args.kvmem_budget,
                args.kvmem_prefill_budget,
                args.kvmem_gen_budget,
            ),
            "rollout_seeds": [args.seed],
            "api_base": api_base,
            "agent_timeout_policy": (
                "unbounded" if args.unbounded_agent_timeout else "task-default"
            ),
            "agent_timeout_multiplier": (
                UNBOUNDED_AGENT_TIMEOUT_MULTIPLIER
                if args.unbounded_agent_timeout
                else 1.0
            ),
            "local_reliability_policy": local_reliability_policy(args.mode),
            "agent_install_adapter": {
                "import_path": agent_import_path(args.mode),
                "path": str(AGENT_ADAPTER),
                "sha256": sha256(AGENT_ADAPTER),
                "runtime_path": (
                    str(COMPACT_AGENT_RUNTIME)
                    if args.mode == "dense-compaction"
                    else None
                ),
                "runtime_sha256": (
                    sha256(COMPACT_AGENT_RUNTIME)
                    if args.mode == "dense-compaction"
                    else None
                ),
                "scope": (
                    "installer transport, slow-local-run reliability, and "
                    "automatic dense-context compaction"
                    if args.mode == "dense-compaction"
                    else "installer transport and slow-local-run reliability hardening"
                ),
            },
            "pier_safe_port_relay": {
                "image": RELAY_IMAGE,
                "listen": f"{args.host}:{args.api_port}",
                "target": f"{args.host}:{args.port}",
            },
        }
        write_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        for item in selected:
            print("TASK", item["task_id"])
            print(
                "PIER",
                json.dumps(
                    pier_command(
                        item["task_id"],
                        DEEPSWE_TASKS / item["task_id"],
                        run_dir / item["task_id"],
                        api_base,
                        mode=args.mode,
                        unbounded_agent_timeout=args.unbounded_agent_timeout,
                        task_cpus=args.task_cpus,
                    )
                ),
            )
        return 0

    for ordinal, item in enumerate(selected, start=1):
        task_id = item["task_id"]
        task_artifact = run_dir / task_id
        task_artifact.mkdir(parents=True, exist_ok=True)
        force_rerun = task_id in rerun_ids
        existing = completed_summary(task_artifact)
        if existing is None and not force_rerun:
            existing = recover_terminal_scored_summary(
                task_artifact, task_id, item["historical_reward"]
            )
        if existing is not None and not force_rerun:
            print(f"[{ordinal}/{len(selected)}] SKIP completed {task_id}: reward={existing['reward']}", flush=True)
            continue

        if force_rerun and existing is not None:
            next_attempt = attempted_count(task_artifact) + 1
            write_json(
                task_artifact
                / f"summary_before_rerun_attempt_{next_attempt:02d}.json",
                existing,
            )
            write_json(
                task_artifact / "summary.json",
                {
                    "status": "rerunning",
                    "task_id": task_id,
                    "previous_reward": existing.get("reward"),
                    "rerun_requested_at": utc_now(),
                    "agent_timeout_policy": (
                        "unbounded"
                        if args.unbounded_agent_timeout
                        else "task-default"
                    ),
                },
            )

        first_attempt = attempted_count(task_artifact) + 1
        if first_attempt > MAX_INFRASTRUCTURE_ATTEMPTS:
            print(
                f"[{ordinal}/{len(selected)}] SKIP failed {task_id}: exhausted "
                f"{MAX_INFRASTRUCTURE_ATTEMPTS} infrastructure attempts; "
                f"inspect {task_artifact}",
                flush=True,
            )
            continue

        successful: tuple[Path, dict[str, Any], str, int | None, int] | None = None
        for infrastructure_attempt in range(first_attempt, MAX_INFRASTRUCTURE_ATTEMPTS + 1):
            current_artifact = attempt_artifact(task_artifact, infrastructure_attempt)
            current_artifact.mkdir(parents=True, exist_ok=True)
            print(
                f"[{ordinal}/{len(selected)}] START {task_id} "
                f"infrastructure_attempt={infrastructure_attempt}/{MAX_INFRASTRUCTURE_ATTEMPTS}",
                flush=True,
            )
            qwen_log_path = current_artifact / "qw3_server.log"
            pier_log_path = current_artifact / "pier_console.log"
            qwen_log = qwen_log_path.open("ab", buffering=0)
            relay_name = f"qw3-pier-relay-{os.getpid()}"
            relay_started = False
            pier_process: subprocess.Popen[Any] | None = None
            server = subprocess.Popen(
                qwen_command(
                    args.host,
                    args.port,
                    args.mode,
                    args.seed,
                    args.guided_query_tokens,
                    args.kvmem_budget,
                    args.kvmem_prefill_budget,
                    args.kvmem_gen_budget,
                ),
                cwd=REPO,
                env=qwen_environment(),
                stdout=qwen_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            started_at = utc_now()
            pier_exit: int | None = None
            runner_exception: dict[str, str] | None = None
            try:
                wait_for_health(server, f"http://{args.host}:{args.port}/health")
                start_relay(
                    relay_name,
                    listen_host=args.host,
                    listen_port=args.api_port,
                    target_host=args.host,
                    target_port=args.port,
                )
                relay_started = True
                wait_for_health(server, f"http://{args.host}:{args.api_port}/health")
                command = pier_command(
                    task_id,
                    DEEPSWE_TASKS / task_id,
                    current_artifact,
                    api_base,
                    mode=args.mode,
                    unbounded_agent_timeout=args.unbounded_agent_timeout,
                    task_cpus=args.task_cpus,
                )
                write_json(
                    current_artifact / "attempt.json",
                    {
                        "task_id": task_id,
                        "infrastructure_attempt": infrastructure_attempt,
                        "rollout_seed": args.seed,
                        "started_at": started_at,
                        "historical_reward": item["historical_reward"],
                        "agent_timeout_policy": (
                            "unbounded"
                            if args.unbounded_agent_timeout
                            else "task-default"
                        ),
                        "agent_timeout_multiplier": (
                            UNBOUNDED_AGENT_TIMEOUT_MULTIPLIER
                            if args.unbounded_agent_timeout
                            else 1.0
                        ),
                        "pier_command": command,
                        "qwen_command": qwen_command(
                            args.host,
                            args.port,
                            args.mode,
                            args.seed,
                            args.guided_query_tokens,
                            args.kvmem_budget,
                            args.kvmem_prefill_budget,
                            args.kvmem_gen_budget,
                        ),
                        "qwen_binary_sha256": sha256(QW3),
                        "qwen_pid": server.pid,
                    },
                )
                with pier_log_path.open("ab", buffering=0) as pier_log:
                    pier_process = subprocess.Popen(
                        command,
                        cwd=REPO,
                        stdout=pier_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    pier_exit = pier_process.wait()
            except Exception as exc:
                # A QW3 startup/relay/Pier-launch failure used to escape this
                # loop and terminate the remaining benchmark queue.  Persist it
                # as an infrastructure attempt so a fresh service can retry.
                runner_exception = {
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            finally:
                stop_process_group(pier_process)
                cleanup_attempt_containers(current_artifact)
                if relay_started:
                    stop_relay(relay_name)
                stop_process(server)
                qwen_log.close()

            parsed = trial_result(current_artifact)
            if parsed is not None and is_terminal_scored_outcome(parsed[1]):
                successful = (*parsed, started_at, pier_exit, infrastructure_attempt)
                break

            attempt_summary: dict[str, Any] = {
                "status": "infrastructure_error",
                "task_id": task_id,
                "infrastructure_attempt": infrastructure_attempt,
                "rollout_seed": args.seed,
                "started_at": started_at,
                "finished_at": utc_now(),
                "pier_exit_code": pier_exit,
            }
            if runner_exception is not None:
                attempt_summary["runner_exception"] = runner_exception
            if parsed is None:
                attempt_summary["reason"] = (
                    "Pier did not produce exactly one trial result with a numeric verifier reward"
                )
            else:
                attempt_summary["trial_result"] = str(parsed[0])
                attempt_summary["exception_info"] = parsed[1].get("exception_info")
            agent_started = any(
                current_artifact.glob(
                    "pier_jobs/*/*/agent/mini-swe-agent.trajectory.json"
                )
            )
            retryable = (
                runner_exception is not None and not agent_started
            ) or retryable_pre_agent_infrastructure_failure(current_artifact)
            attempt_summary["retryable_pre_agent_failure"] = retryable
            write_json(current_artifact / "infrastructure_attempt.json", attempt_summary)

            if retryable and infrastructure_attempt < MAX_INFRASTRUCTURE_ATTEMPTS:
                print(
                    f"[{ordinal}/{len(selected)}] RETRY {task_id}: "
                    "transient official container build failure before agent startup",
                    flush=True,
                )
                continue

            write_json(task_artifact / "summary.json", attempt_summary)
            print(
                f"[{ordinal}/{len(selected)}] FAILED {task_id}: no valid official "
                f"verifier result; continuing queue; inspect {current_artifact}",
                flush=True,
            )
            break

        if successful is None:
            continue

        result_path, result, started_at, pier_exit, infrastructure_attempt = successful

        reward = result["verifier_result"]["rewards"]["reward"]
        summary = {
            "status": "completed",
            "task_id": task_id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "pier_exit_code": pier_exit,
            "infrastructure_attempt": infrastructure_attempt,
            "rollout_seed": args.seed,
            "reward": reward,
            "historical_reward": item["historical_reward"],
            "trial_result": str(result_path),
        }
        exception = result.get("exception_info")
        if isinstance(exception, dict):
            summary["agent_terminal_exception"] = {
                "exception_type": exception.get("exception_type"),
                "exception_message": exception.get("exception_message"),
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
