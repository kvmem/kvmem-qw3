#!/usr/bin/env python3
"""Compare regular qw3 serving against qw3 serving with KVMem on SWE tasks.

This is a lifecycle wrapper around scripts/run_openhands_swebench.py. It starts
one qw3 server per variant, runs the exact same OpenHands SWE-bench selection,
stops the server, then summarizes correctness artifacts and serving efficiency
from the server log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "Qwen3.6-27B-Q8_0.gguf"
DEFAULT_OPENHANDS_DIR = REPO_ROOT / "third_party" / "openhands-benchmarks"

CONTINUOUS_SUMMARY_RE = re.compile(
    r"native continuous_batch: request=(?P<request>\d+) "
    r"prompt_tokens=(?P<prompt>\d+) "
    r"prefill=(?P<prefill>[0-9.]+)s .*? "
    r"decoded=(?P<decoded>\d+) "
    r"decode=(?P<decode>[0-9.]+)s .*? "
    r".*?max_batch=(?P<max_batch>\d+)"
)
GENERATE_RE = re.compile(
    r"native generate:\s+"
    r"prompt_tokens=(?P<prompt>\d+)\s+"
    r"prefill=(?P<prefill>[0-9.]+)s"
    r"(?:\s+\([0-9.]+\s+tok/s\))?\s+"
    r"decoded=(?P<decoded>\d+)\s+"
    r"decode=(?P<decode>[0-9.]+)s"
)
TIMING_RE = re.compile(
    r"native continuous_batch_timing:\s+"
    r"mode=(?P<mode>\S+)\s+"
    r"batch=(?P<batch>\d+)\s+"
    r"kernel_batch=(?P<kernel_batch>\d+)\s+"
    r"total=(?P<total>[0-9.]+)s"
)
EVAL_PROGRESS_RE = re.compile(
    r"Evaluation:.*?✓=(?P<resolved>\d+),\s*✖=(?P<failed>\d+),\s*error=(?P<error>\d+)"
)


@dataclass
class EfficiencySummary:
    request_count: int = 0
    prompt_tokens: int = 0
    prefill_s: float = 0.0
    decoded_tokens: int = 0
    decode_s: float = 0.0
    prefill_tokens_per_s: float = 0.0
    decode_tokens_per_s: float = 0.0
    timing_step_tokens: int = 0
    timing_step_s: float = 0.0
    timing_step_tokens_per_s: float = 0.0
    max_summary_batch: int = 0
    max_timing_batch: int = 0
    kvmem_stage_out: int = 0
    kvmem_stage_in: int = 0
    kvmem_stage_in_async_read: int = 0
    kvmem_cpu_evict: int = 0
    kvmem_bounded_gpu_pool: int = 0


@dataclass
class AccuracySummary:
    output_count: int = 0
    prediction_count: int = 0
    nonempty_patch_count: int = 0
    empty_patch_count: int = 0
    resolved_count: int | None = None
    failed_count: int | None = None
    error_count: int | None = None
    total_evaluated: int | None = None
    resolved_rate: float | None = None
    report_json: str | None = None
    report_source: str | None = None


@dataclass
class VariantResult:
    name: str
    command: list[str]
    base_url: str
    result_dir: str | None = None
    server_log: str | None = None
    runner_log: str | None = None
    started_at: str = ""
    finished_at: str = ""
    elapsed_s: float = 0.0
    returncode: int | None = None
    error: str | None = None
    accuracy: AccuracySummary = field(default_factory=AccuracySummary)
    efficiency: EfficiencySummary = field(default_factory=EfficiencySummary)


def git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "nogit"


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def jsonl_count(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def parse_predictions(path: Path | None) -> tuple[int, int, int]:
    if path is None or not path.is_file():
        return 0, 0, 0
    total = 0
    nonempty = 0
    empty = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                empty += 1
                continue
            patch = str(row.get("model_patch") or "").strip()
            if patch and patch.lower() not in {"none", "null"}:
                nonempty += 1
            else:
                empty += 1
    return total, nonempty, empty


def newest_matching(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(
        root.rglob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def discover_result_dir(variant_root: Path, runner_stdout: str) -> Path | None:
    for line in reversed(runner_stdout.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        result_dir = data.get("result_dir")
        if result_dir:
            path = Path(str(result_dir))
            if path.is_dir():
                return path
    summary = newest_matching(variant_root, "summary_openhands_swebench.md")
    if summary is not None:
        return summary.parent
    for pattern in ("eval.log", "predictions.swebench.jsonl", "output.jsonl"):
        artifact = newest_matching(variant_root, pattern)
        if artifact is not None:
            return artifact.parent
    return None


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_count_value(obj: Any, names: set[str]) -> int | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            lk = key.lower()
            if lk in names:
                if isinstance(value, int):
                    return value
                if isinstance(value, list):
                    return len(value)
                if isinstance(value, dict):
                    return len(value)
        for value in obj.values():
            found = find_count_value(value, names)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_count_value(value, names)
            if found is not None:
                return found
    return None


def parse_report_json(path: Path | None) -> tuple[int | None, int | None, int | None, int | None]:
    if path is None or not path.is_file():
        return None, None, None, None
    data = read_json(path)
    if data is None:
        return None, None, None, None
    resolved = find_count_value(data, {"resolved", "resolved_ids", "resolved_instances"})
    failed = find_count_value(data, {"failed", "failed_ids", "unresolved", "unresolved_ids"})
    errored = find_count_value(data, {"error", "errors", "error_ids"})
    total = find_count_value(data, {"total", "total_instances", "submitted", "instances"})
    return resolved, failed, errored, total


def parse_eval_log(path: Path | None) -> tuple[int | None, int | None, int | None]:
    if path is None or not path.is_file():
        return None, None, None
    text = path.read_text(encoding="utf-8", errors="replace")
    last: tuple[int, int, int] | None = None
    for m in EVAL_PROGRESS_RE.finditer(text):
        last = (
            int(m.group("resolved")),
            int(m.group("failed")),
            int(m.group("error")),
        )
    return last if last is not None else (None, None, None)


def summarize_accuracy(result_dir: Path | None) -> AccuracySummary:
    if result_dir is None:
        return AccuracySummary()
    output_json = result_dir / "output.jsonl"
    predictions = result_dir / "predictions.swebench.jsonl"
    report = newest_matching(result_dir, "*.report.json")
    summary = AccuracySummary(
        output_count=jsonl_count(output_json),
        report_json=str(report) if report else None,
    )
    (
        summary.prediction_count,
        summary.nonempty_patch_count,
        summary.empty_patch_count,
    ) = parse_predictions(predictions)

    resolved, failed, errored, total = parse_report_json(report)
    if resolved is not None:
        summary.resolved_count = resolved
        summary.failed_count = failed
        summary.error_count = errored
        summary.total_evaluated = total or sum(x or 0 for x in (resolved, failed, errored))
        summary.report_source = "report_json"
    else:
        resolved, failed, errored = parse_eval_log(result_dir / "eval.log")
        if resolved is not None:
            summary.resolved_count = resolved
            summary.failed_count = failed
            summary.error_count = errored
            summary.total_evaluated = sum(x or 0 for x in (resolved, failed, errored))
            summary.report_source = "eval_log_progress"
        elif summary.prediction_count and summary.nonempty_patch_count == 0:
            summary.resolved_count = 0
            summary.failed_count = summary.prediction_count
            summary.error_count = 0
            summary.total_evaluated = summary.prediction_count
            summary.report_source = "empty_patch"

    if summary.resolved_count is not None and summary.total_evaluated:
        summary.resolved_rate = summary.resolved_count / summary.total_evaluated
    return summary


def summarize_efficiency(server_log: Path) -> EfficiencySummary:
    text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.is_file() else ""
    summary = EfficiencySummary()
    for m in CONTINUOUS_SUMMARY_RE.finditer(text):
        summary.request_count += 1
        summary.prompt_tokens += int(m.group("prompt"))
        summary.prefill_s += float(m.group("prefill"))
        summary.decoded_tokens += int(m.group("decoded"))
        summary.decode_s += float(m.group("decode"))
        summary.max_summary_batch = max(summary.max_summary_batch, int(m.group("max_batch")))
    for m in GENERATE_RE.finditer(text):
        summary.request_count += 1
        summary.prompt_tokens += int(m.group("prompt"))
        summary.prefill_s += float(m.group("prefill"))
        summary.decoded_tokens += int(m.group("decoded"))
        summary.decode_s += float(m.group("decode"))
        summary.max_summary_batch = max(summary.max_summary_batch, 1)
    for m in TIMING_RE.finditer(text):
        summary.timing_step_tokens += int(m.group("batch"))
        summary.timing_step_s += float(m.group("total"))
        summary.max_timing_batch = max(summary.max_timing_batch, int(m.group("kernel_batch")))
    summary.prefill_tokens_per_s = (
        summary.prompt_tokens / summary.prefill_s if summary.prefill_s > 0.0 else 0.0
    )
    summary.decode_tokens_per_s = (
        summary.decoded_tokens / summary.decode_s if summary.decode_s > 0.0 else 0.0
    )
    summary.timing_step_tokens_per_s = (
        summary.timing_step_tokens / summary.timing_step_s if summary.timing_step_s > 0.0 else 0.0
    )
    summary.kvmem_stage_out = len(re.findall(r"\bkvmem\b.*\bstage_out\b", text))
    summary.kvmem_stage_in = len(re.findall(r"\bkvmem\b.*\bstage_in\b", text))
    summary.kvmem_stage_in_async_read = len(re.findall(r"\bstage_in_async_read\b", text))
    summary.kvmem_cpu_evict = len(re.findall(r"\bcpu_evict\b", text))
    summary.kvmem_bounded_gpu_pool = len(re.findall(r"\bbounded_gpu_pool\b", text))
    return summary


def wait_for_health(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.URLError as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy at {url}: {last_error}")


def terminate_process(proc: subprocess.Popen[Any], timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.send_signal(signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=timeout_s)


def append_if(args: list[str], condition: bool, values: Sequence[str]) -> None:
    if condition:
        args.extend(values)


def build_server_cmd(args: argparse.Namespace, variant: str, port: int) -> list[str]:
    cmd = [
        str(args.qw3),
        "serve",
        "--model",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(port),
        "--ctx",
        str(args.ctx),
        "--temp",
        "0",
    ]
    if args.server_max_tokens:
        cmd.extend(["-n", str(args.server_max_tokens)])
    if args.prefill_chunk is not None:
        cmd.extend(["--prefill-chunk", str(args.prefill_chunk)])
    if args.continuous_batching:
        cmd.extend(["--continuous-batching", "--paged-kv", "--max-active", str(args.max_active)])
    else:
        cmd.append("--no-continuous-batching")
    cmd.extend(args.server_extra_arg)

    if variant == "kvmem":
        cmd.append("--kvmem")
        cmd.extend(["--kvmem-block-tokens", str(args.kvmem_block_tokens)])
        cmd.extend(["--kvmem-budget", str(args.kvmem_budget)])
        cmd.extend(["--kvmem-sink-blocks", str(args.kvmem_sink_blocks)])
        cmd.extend(["--kvmem-recent-blocks", str(args.kvmem_recent_blocks)])
        cmd.extend(["--kvmem-method", args.kvmem_method])
        cmd.extend(["--kvmem-select-policy", args.kvmem_select_policy])
        cmd.extend(["--kvmem-retrieval-method", args.kvmem_retrieval_method])
        cmd.extend(["--kvmem-update-mode", args.kvmem_update_mode])
        cmd.extend(["--kvmem-interval", str(args.kvmem_interval)])
        cmd.extend(["--kvmem-gpu-memory-ratio", str(args.kvmem_gpu_memory_ratio)])
        if args.kvmem_retrieval_blocks:
            cmd.extend(["--kvmem-retrieval-blocks", str(args.kvmem_retrieval_blocks)])
        if args.kvmem_profile_blocks:
            cmd.extend(["--kvmem-profile-blocks", str(args.kvmem_profile_blocks)])
        if args.kvmem_cpu_bytes:
            cmd.extend(["--kvmem-cpu-bytes", str(args.kvmem_cpu_bytes)])
        if args.kvmem_nvme_dir:
            cmd.extend(["--kvmem-nvme-dir", str(args.kvmem_nvme_dir)])
        if args.kvmem_nvme_bytes:
            cmd.extend(["--kvmem-nvme-bytes", str(args.kvmem_nvme_bytes)])
        cmd.extend(args.kvmem_extra_arg)
    return cmd


def build_openhands_cmd(args: argparse.Namespace, variant: str, port: int, result_root: Path) -> list[str]:
    base_url = args.base_url_template.format(port=port)
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "run_openhands_swebench.py"),
        "--openhands-dir",
        str(args.openhands_dir),
        "--uv-bin",
        args.uv_bin,
        "--base-url",
        base_url,
        "--model",
        args.openhands_model,
        "--api-key",
        args.api_key,
        "--temperature",
        str(args.temperature),
        "--dataset",
        args.dataset,
        "--split",
        args.split,
        "--workspace",
        args.workspace,
        "--num-workers",
        str(args.num_workers),
        "--eval-workers",
        str(args.eval_workers),
        "--max-iterations",
        str(args.max_iterations),
        "--max-retries",
        str(args.max_retries),
        "--memory-policy",
        args.memory_policy,
        "--eval-timeout",
        str(args.eval_timeout),
        "--result-root",
        str(result_root),
        "--run-id",
        f"{args.run_id}_{variant}",
        "--note",
        f"qw3-swe-kvmem-compare-{variant}",
    ]
    if args.n_limit:
        cmd.extend(["--n-limit", str(args.n_limit)])
    if args.select:
        cmd.extend(["--select", str(args.select)])
    if args.max_output_tokens:
        cmd.extend(["--max-output-tokens", str(args.max_output_tokens)])
    if args.custom_tokenizer:
        cmd.extend(["--custom-tokenizer", args.custom_tokenizer])
    if args.disable_native_tool_calling:
        cmd.append("--disable-native-tool-calling")
    if args.clone_openhands:
        cmd.append("--clone-openhands")
    if args.skip_evaluation:
        cmd.append("--skip-evaluation")
    if args.build_images:
        cmd.append("--build-images")
    if args.image_tag_prefix:
        cmd.extend(["--image-tag-prefix", args.image_tag_prefix])
    if args.swebench_image_template:
        cmd.extend(["--swebench-image-template", args.swebench_image_template])
    if args.strip_dockerfile_syntax:
        cmd.append("--strip-dockerfile-syntax")
    if args.modal:
        cmd.append("--modal")
    if args.apptainer_eval:
        cmd.append("--apptainer-eval")
    cmd.extend(args.openhands_extra_arg)
    return cmd


def run_logged_command(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        ret = proc.wait()
    return ret, "".join(chunks)


def detect_openhands_phased_prefix(args: argparse.Namespace) -> str | None:
    """Return the local OpenHands phased image prefix when available.

    SWE-bench in OpenHands uses get_phased_image_tag_prefix() for its expected
    agent-server image, while some local build paths still emit IMAGE_TAG_PREFIX
    directly. Passing the phased value explicitly makes both sides agree.
    """
    if args.image_tag_prefix or not args.auto_image_tag_prefix:
        return args.image_tag_prefix
    if not args.openhands_dir.is_dir():
        return None
    cmd = [
        args.uv_bin,
        "run",
        "python",
        "-c",
        (
            "from benchmarks.utils.version import get_phased_image_tag_prefix; "
            "print(get_phased_image_tag_prefix())"
        ),
    ]
    env = os.environ.copy()
    env.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(args.openhands_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.splitlines()):
        value = line.strip()
        if re.fullmatch(r"[A-Za-z0-9_.-]+", value) and "-" in value:
            return value
    return None


def selected_instance_ids(select_file: Path | None, n_limit: int, *, apply_limit: bool = True) -> list[str]:
    if select_file is None or not select_file.is_file():
        return []
    ids: list[str] = []
    with select_file.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
            if apply_limit and n_limit and len(ids) >= n_limit:
                break
    return ids


def swebench_custom_tag(instance_id: str) -> str:
    repo, name = instance_id.split("__", 1)
    return f"sweb.eval.x86_64.{repo}_1776_{name}".lower()


def docker_image_exists(image: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def prepare_phased_local_images(args: argparse.Namespace) -> list[dict[str, str]]:
    """Retag local OpenHands SWE images so phased run_infer checks can pass."""
    if not args.image_tag_prefix or "-" not in args.image_tag_prefix:
        return []
    # OpenHands applies --n-limit after dataset filtering, not necessarily in
    # the same order as the select file, so retag all selected instances.
    ids = selected_instance_ids(args.select, args.n_limit, apply_limit=False)
    if not ids:
        return []
    sdk_prefix = args.image_tag_prefix.split("-", 1)[0]
    actions: list[dict[str, str]] = []
    for instance_id in ids:
        custom_tag = swebench_custom_tag(instance_id)
        source = (
            "ghcr.io/openhands/eval-agent-server:"
            f"{sdk_prefix}-{custom_tag}-source-minimal"
        )
        target = (
            "ghcr.io/openhands/eval-agent-server:"
            f"{args.image_tag_prefix}-{custom_tag}-source-minimal"
        )
        if docker_image_exists(target):
            actions.append({"instance_id": instance_id, "status": "already_exists", "target": target})
            continue
        if not docker_image_exists(source):
            actions.append({"instance_id": instance_id, "status": "missing_source", "source": source, "target": target})
            continue
        proc = subprocess.run(
            ["docker", "tag", source, target],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        status = "retagged" if proc.returncode == 0 else "retag_failed"
        actions.append(
            {
                "instance_id": instance_id,
                "status": status,
                "source": source,
                "target": target,
                "output": proc.stdout.strip(),
            }
        )
    return actions


def run_variant(args: argparse.Namespace, run_root: Path, variant: str, port: int) -> VariantResult:
    variant_root = run_root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    server_log = variant_root / "qw3_server.log"
    runner_log = variant_root / "openhands_runner.log"
    result_root = variant_root / "openhands_results"
    cmd = build_server_cmd(args, variant, port)
    base_url = args.base_url_template.format(port=port)
    result = VariantResult(
        name=variant,
        command=cmd,
        base_url=base_url,
        server_log=str(server_log),
        runner_log=str(runner_log),
        started_at=dt.datetime.now().isoformat(timespec="seconds"),
    )

    env = os.environ.copy()
    if variant == "kvmem" and (args.kvmem_cpu_bytes or args.kvmem_nvme_bytes):
        env["QW3_KVMEM_TIER_TRACE"] = "1"
    if args.continuous_timing:
        env["QW3_CONTINUOUS_BATCHING_TIMING"] = "1"
    if args.continuous_trace:
        env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"

    start = time.monotonic()
    with server_log.open("w", encoding="utf-8") as slog:
        slog.write("+ " + " ".join(cmd) + "\n")
        slog.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=slog,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_health(f"http://{args.health_host}:{port}/health", args.server_start_timeout)
            runner_cmd = build_openhands_cmd(args, variant, port, result_root)
            ret, stdout = run_logged_command(
                runner_cmd,
                cwd=REPO_ROOT,
                log_path=runner_log,
                env=os.environ.copy(),
            )
            result.returncode = ret
            result_dir = discover_result_dir(result_root, stdout)
            if result_dir is not None:
                result.result_dir = str(result_dir)
            if ret != 0:
                result.error = f"OpenHands runner exited with code {ret}"
        except Exception as exc:  # noqa: BLE001 - result object records failure.
            result.returncode = -1
            result.error = str(exc)
        finally:
            terminate_process(proc)

    result.elapsed_s = time.monotonic() - start
    result.finished_at = dt.datetime.now().isoformat(timespec="seconds")
    result_dir_path = Path(result.result_dir) if result.result_dir else None
    result.accuracy = summarize_accuracy(result_dir_path)
    result.efficiency = summarize_efficiency(server_log)
    return result


def pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.1f}%"


def speedup(new: float, base: float) -> str:
    if base <= 0.0 or new <= 0.0:
        return ""
    return f"{new / base:.3f}x"


def write_summary(path: Path, args: argparse.Namespace, results: Sequence[VariantResult]) -> None:
    by_name = {r.name: r for r in results}
    baseline = by_name.get("baseline")
    lines = [
        "# SWE KVMem Compare",
        "",
        f"- run_id: `{args.run_id}`",
        f"- created_at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- model: `{args.model}`",
        f"- select: `{args.select or ''}`",
        f"- n_limit: `{args.n_limit}`",
        f"- dataset/split: `{args.dataset}` / `{args.split}`",
        f"- OpenHands memory_policy: `{args.memory_policy}`",
        f"- KVMem budget/block/update: `{args.kvmem_budget}` / `{args.kvmem_block_tokens}` / `{args.kvmem_update_mode}`",
        f"- KVMem method/retrieval: `{args.kvmem_method}` / `{args.kvmem_retrieval_method}`",
        f"- KVMem GPU ratio / CPU bytes / NVMe bytes: `{args.kvmem_gpu_memory_ratio}` / `{args.kvmem_cpu_bytes}` / `{args.kvmem_nvme_bytes}`",
        "",
        "## Accuracy",
        "",
        "| variant | runner | outputs | predictions | nonempty patches | resolved | resolved rate | source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in results:
        a = r.accuracy
        resolved = "" if a.resolved_count is None else str(a.resolved_count)
        lines.append(
            "| {variant} | {runner} | {outputs} | {preds} | {patches} | {resolved} | {rate} | {source} |".format(
                variant=r.name,
                runner="ok" if r.returncode == 0 else f"fail({r.returncode})",
                outputs=a.output_count,
                preds=a.prediction_count,
                patches=a.nonempty_patch_count,
                resolved=resolved,
                rate=pct(a.resolved_rate),
                source=a.report_source or "",
            )
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            "| variant | elapsed_s | requests | prompt tokens | prefill tok/s | decoded tokens | decode tok/s | decode speedup | max batch | stage out | stage in | async nvme reads |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    base_decode = baseline.efficiency.decode_tokens_per_s if baseline else 0.0
    for r in results:
        e = r.efficiency
        lines.append(
            "| {variant} | {elapsed:.1f} | {req} | {prompt} | {prefill:.2f} | {decoded} | {decode:.2f} | {spd} | {batch} | {out} | {inn} | {async_reads} |".format(
                variant=r.name,
                elapsed=r.elapsed_s,
                req=e.request_count,
                prompt=e.prompt_tokens,
                prefill=e.prefill_tokens_per_s,
                decoded=e.decoded_tokens,
                decode=e.decode_tokens_per_s,
                spd=speedup(e.decode_tokens_per_s, base_decode),
                batch=max(e.max_summary_batch, e.max_timing_batch),
                out=e.kvmem_stage_out,
                inn=e.kvmem_stage_in,
                async_reads=e.kvmem_stage_in_async_read,
            )
        )
    lines.extend(["", "## Artifacts", ""])
    for r in results:
        lines.append(f"- `{r.name}` result_dir: `{r.result_dir or ''}`")
        lines.append(f"- `{r.name}` server_log: `{r.server_log or ''}`")
        lines.append(f"- `{r.name}` runner_log: `{r.runner_log or ''}`")
        if r.error:
            lines.append(f"- `{r.name}` error: `{r.error}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    default_python = REPO_ROOT / ".venv" / "bin" / "python"
    parser = argparse.ArgumentParser(
        description="Run baseline qw3 and KVMem qw3 on the same OpenHands SWE-bench tasks."
    )
    parser.add_argument("--qw3", type=Path, default=REPO_ROOT / "build" / "qw3")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=default_python if default_python.exists() else Path(sys.executable))
    parser.add_argument("--openhands-dir", type=Path, default=DEFAULT_OPENHANDS_DIR)
    parser.add_argument("--clone-openhands", action="store_true")
    parser.add_argument("--uv-bin", default="uv")
    parser.add_argument("--result-root", type=Path, default=REPO_ROOT / "benchmark" / "results")
    parser.add_argument("--run-id", default=f"swe_kvmem_compare_{git_sha_short()}_{now_stamp()}")
    parser.add_argument("--variants", default="baseline,kvmem", help="Comma-separated subset: baseline,kvmem")

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--health-host", default="127.0.0.1")
    parser.add_argument("--base-url-template", default="http://172.17.0.1:{port}/v1")
    parser.add_argument("--port-base", type=int, default=8080)
    parser.add_argument("--server-start-timeout", type=float, default=300.0)
    parser.add_argument("--ctx", type=int, default=65536)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--server-max-tokens", type=int, default=0)
    parser.add_argument("--continuous-batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-active", type=int, default=2)
    parser.add_argument("--continuous-trace", action="store_true")
    parser.add_argument("--continuous-timing", action="store_true")
    parser.add_argument("--server-extra-arg", action="append", default=[])

    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workspace", default="docker", choices=["docker", "remote", "apptainer"])
    parser.add_argument("--select", type=Path)
    parser.add_argument("--n-limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--eval-workers", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--memory-policy", default="no_condenser", choices=["condenser", "no_condenser"])
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--custom-tokenizer")
    parser.add_argument("--disable-native-tool-calling", action="store_true")
    parser.add_argument("--openhands-model", default="openai/qw3")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--build-images", action="store_true")
    parser.add_argument("--image-tag-prefix")
    parser.add_argument(
        "--auto-image-tag-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect OpenHands phased IMAGE_TAG_PREFIX when --image-tag-prefix is unset.",
    )
    parser.add_argument("--swebench-image-template")
    parser.add_argument("--strip-dockerfile-syntax", action="store_true")
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--apptainer-eval", action="store_true")
    parser.add_argument("--openhands-extra-arg", action="append", default=[])

    parser.add_argument("--kvmem-block-tokens", type=int, default=128)
    parser.add_argument("--kvmem-budget", type=int, default=32768)
    parser.add_argument("--kvmem-interval", type=int, default=64)
    parser.add_argument("--kvmem-sink-blocks", type=int, default=1)
    parser.add_argument("--kvmem-recent-blocks", type=int, default=0)
    parser.add_argument("--kvmem-method", default="retrieval", choices=["retrieval", "h2o", "recency"])
    parser.add_argument("--kvmem-select-policy", default="topk", choices=["topk", "quota"])
    parser.add_argument("--kvmem-retrieval-method", default="mean_attention", choices=["mean_attention", "content_mean"])
    parser.add_argument("--kvmem-update-mode", default="step", choices=["interval", "step"])
    parser.add_argument("--kvmem-retrieval-blocks", type=int, default=0)
    parser.add_argument("--kvmem-profile-blocks", type=int, default=0)
    parser.add_argument("--kvmem-gpu-memory-ratio", type=float, default=0.50)
    parser.add_argument("--kvmem-cpu-bytes", type=int, default=0)
    parser.add_argument("--kvmem-nvme-dir", type=Path)
    parser.add_argument("--kvmem-nvme-bytes", type=int, default=0)
    parser.add_argument("--kvmem-extra-arg", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variants if v not in {"baseline", "kvmem"}]
    if invalid:
        raise SystemExit(f"unknown variant(s): {', '.join(invalid)}")
    if not args.qw3.is_file():
        raise SystemExit(f"qw3 binary not found: {args.qw3}")
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.kvmem_nvme_bytes and not args.kvmem_nvme_dir:
        raise SystemExit("--kvmem-nvme-bytes requires --kvmem-nvme-dir")
    detected_prefix = detect_openhands_phased_prefix(args)
    if detected_prefix and not args.image_tag_prefix:
        args.image_tag_prefix = detected_prefix
        print(f"[compare] using OpenHands image tag prefix {detected_prefix}", flush=True)
    image_actions = prepare_phased_local_images(args)
    for item in image_actions:
        print(
            f"[compare] image {item.get('status')} instance={item.get('instance_id')} "
            f"target={item.get('target')}",
            flush=True,
        )

    run_root = args.result_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "variants": variants,
        "image_actions": image_actions,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    results: list[VariantResult] = []
    for idx, variant in enumerate(variants):
        port = args.port_base + idx
        print(f"[compare] starting variant={variant} port={port}", flush=True)
        result = run_variant(args, run_root, variant, port)
        results.append(result)
        (run_root / "compare.partial.json").write_text(
            json.dumps([asdict(r) for r in results], indent=2) + "\n",
            encoding="utf-8",
        )
        if result.error:
            print(f"[compare] variant={variant} completed with error: {result.error}", flush=True)
        else:
            print(f"[compare] variant={variant} completed", flush=True)

    compare_json = run_root / "compare.json"
    compare_json.write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
    summary = run_root / "summary_swe_kvmem_compare.md"
    write_summary(summary, args, results)
    print(json.dumps({"run_root": str(run_root), "compare_json": str(compare_json), "summary": str(summary)}))
    return 0 if all(r.returncode == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
