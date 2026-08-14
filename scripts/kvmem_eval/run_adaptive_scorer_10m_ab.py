#!/usr/bin/env python3
"""Benchmark exact Adaptive CPU scorers against one frozen 10M archive.

The three production arms trade bounded GPU scratch for index passes.  A
fourth scalar two-pass arm is retained only as an internal before/after
diagnostic.  Every arm rebuilds the same index from the same raw-K authority,
runs the same repeated 128-token query, and records process peak GPU/RSS.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path(
    "/home/chaidi/kca/beam10m_c1_9998336_b64_scorer_ab_20260812"
)

ARMS: dict[str, dict[str, Any]] = {
    "layer_one_pass": {
        "mode": "layer-one-pass",
        # Per-slot cap. The allocator uses the observed largest layer rather
        # than the full cap; two device slots stay within the 3--6 GiB target.
        "staging_mb": 3072,
        "block_stats_mb": 512,
        "two_pass_gemm": True,
    },
    "tiled_one_pass": {
        "mode": "tiled-one-pass",
        "staging_mb": 64,
        # Exact per-block max/sum state is ~3.6 GiB at 10M.
        "block_stats_mb": 4096,
        "two_pass_gemm": True,
    },
    "tiled_two_pass_gemm": {
        "mode": "tiled-two-pass",
        "staging_mb": 64,
        "block_stats_mb": 512,
        "two_pass_gemm": True,
    },
    "tiled_two_pass_scalar": {
        "mode": "tiled-two-pass",
        "staging_mb": 64,
        "block_stats_mb": 512,
        "two_pass_gemm": False,
    },
}

STREAM_RE = re.compile(r"\[bs-adaptive-stream\]\s+(.*)")
SEMANTIC_RE = re.compile(r"\[kvmem-reselect-perf\]\s+kind=semantic\s+(.*)")
ATTACH_RE = re.compile(
    r"archive attach: requested=(?P<requested>\d+).*?attach=(?P<attach>[0-9.]+)s"
    r".*?replay=(?P<replay>[0-9.]+)s.*?index=(?P<index>[0-9.]+)s"
)
PINNED_RE = re.compile(
    r"\[bs-adaptive-index\] pinned_authority=1 layers=(?P<layers>\d+/\d+)"
    r" bytes=(?P<bytes>\d+) gib=(?P<gib>[0-9.]+)"
    r".*?migrate_ms=(?P<migrate>[0-9.]+)"
)


def fields(text: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", text))


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered) + 0.999999) - 1))
    return ordered[index]


def gpu_mib_for_pid(pid: int) -> int:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def rss_mib_for_pid(pid: int) -> float:
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError):
        return 0.0
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    return int(match.group(1)) / 1024.0 if match else 0.0


def run_and_monitor(
    cmd: list[str], env: dict[str, str], log_path: Path
) -> tuple[int, float, int, float, int]:
    started = time.monotonic()
    peak_gpu_mib = 0
    peak_rss_mib = 0.0
    samples = 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True
        )
        while process.poll() is None:
            peak_gpu_mib = max(peak_gpu_mib, gpu_mib_for_pid(process.pid))
            peak_rss_mib = max(peak_rss_mib, rss_mib_for_pid(process.pid))
            samples += 1
            time.sleep(0.10)
        return (
            process.returncode,
            time.monotonic() - started,
            peak_gpu_mib,
            peak_rss_mib,
            samples,
        )


def parse_arm(
    name: str,
    config: dict[str, Any],
    log_path: Path,
    results_path: Path,
    wall_s: float,
    peak_gpu_mib: int,
    peak_rss_mib: float,
    samples: int,
) -> dict[str, Any]:
    log = log_path.read_text(encoding="utf-8", errors="replace")
    stream = [fields(match.group(1)) for match in STREAM_RE.finditer(log)]
    semantic = [fields(match.group(1)) for match in SEMANTIC_RE.finditer(log)]
    results = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not stream or not semantic or not results:
        raise RuntimeError(f"{name}: missing scorer, semantic, or result records")
    expected_mode = config["mode"]
    if any(row.get("mode") != expected_mode for row in stream):
        raise RuntimeError(f"{name}: scorer did not stay in {expected_mode}")
    if any(int(row.get("query_tokens", "0")) != 128 for row in stream):
        raise RuntimeError(f"{name}: scorer query length was not 128 tokens")
    expected_scorer = (
        "tensor-gemm" if config["two_pass_gemm"] else "scalar"
    )
    if expected_mode == "tiled-two-pass" and any(
        row.get("tile_scorer") != expected_scorer for row in stream
    ):
        raise RuntimeError(
            f"{name}: expected tile_scorer={expected_scorer}, got {stream}"
        )

    warm_stream = stream[1:] if len(stream) > 1 else stream
    warm_semantic = semantic[1:] if len(semantic) > 1 else semantic
    warm_results = results[1:] if len(results) > 1 else results
    attach = ATTACH_RE.search(log)
    pinned = PINNED_RE.search(log)
    hashes = [row.get("selected_hash", "") for row in semantic]
    staging_mib = float(stream[-1]["staging_mib"])
    block_stats_mib = (
        float(stream[-1]["block_stats_mib"])
        if "block_stats_mib" in stream[-1]
        else 0.0
    )
    logits_mib = (
        float(stream[-1]["logits_mib"])
        if "logits_mib" in stream[-1]
        else 0.0
    )
    return {
        "name": name,
        "mode": expected_mode,
        "tile_scorer": stream[-1].get("tile_scorer"),
        "passes": int(stream[-1]["passes"]),
        "queries": len(results),
        "selected_hashes": hashes,
        "warm_selected_hashes": [
            row.get("selected_hash", "") for row in warm_semantic
        ],
        "scorer_core_ms": [float(row["total_ms"]) for row in stream],
        "warm_scorer_core_p50_ms": med(
            [float(row["total_ms"]) for row in warm_stream]
        ),
        "warm_scorer_core_p95_ms": percentile(
            [float(row["total_ms"]) for row in warm_stream], 0.95
        ),
        "warm_selection_p50_ms": med(
            [float(row["selection_ms"]) for row in warm_semantic]
        ),
        "warm_reselect_total_p50_ms": med(
            [float(row["total_ms"]) for row in warm_semantic]
        ),
        "ttft_s": [row["ttft_s"] for row in results],
        "warm_ttft_p50_s": med(
            [float(row["ttft_s"]) for row in warm_results]
        ),
        "warm_ttft_p95_s": percentile(
            [float(row["ttft_s"]) for row in warm_results], 0.95
        ),
        "transferred_gib": float(stream[-1]["transferred_gib"]),
        "staging_mib": staging_mib,
        "block_stats_mib": block_stats_mib,
        "logits_mib": logits_mib,
        # Both streaming implementations double-buffer index staging.
        "scorer_gpu_workspace_estimate_mib": (
            2.0 * staging_mib + block_stats_mib + logits_mib
        ),
        "warm_pack_p50_ms": med(
            [float(row["pack_ms"]) for row in warm_stream]
        ),
        "warm_h2d_wait_p50_ms": med(
            [float(row["h2d_wait_ms"]) for row in warm_stream]
        ),
        "peak_process_gpu_mib": peak_gpu_mib,
        "peak_process_rss_mib": round(peak_rss_mib, 1),
        "monitor_samples": samples,
        "process_wall_s": wall_s,
        "attach_s": float(attach.group("attach")) if attach else None,
        "index_build_s": float(attach.group("index")) if attach else None,
        "pinned_index_layers": pinned.group("layers") if pinned else None,
        "pinned_index_gib": float(pinned.group("gib")) if pinned else None,
        "pinned_migrate_ms": float(pinned.group("migrate")) if pinned else None,
        "log": str(log_path),
        "results": str(results_path),
    }


def fmt(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Adaptive scorer 10M exact A/B",
        "",
        "| Arm | Passes | GPU peak (MiB) | GPU delta (MiB) | "
        "Core p50 (s) | Selection p50 (s) | TTFT p50 (s) | H2D GiB | Hash |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm in summary["arms"]:
        hashes = arm["warm_selected_hashes"]
        hash_text = hashes[0] if hashes and len(set(hashes)) == 1 else "MISMATCH"
        lines.append(
            f"| {arm['name']} | {arm['passes']} | "
            f"{arm['peak_process_gpu_mib']} | {arm['peak_gpu_delta_mib']} | "
            f"{fmt(arm['warm_scorer_core_p50_ms'] / 1000.0)} | "
            f"{fmt(arm['warm_selection_p50_ms'] / 1000.0)} | "
            f"{fmt(arm['warm_ttft_p50_s'])} | "
            f"{fmt(arm['transferred_gib'], 3)} | {hash_text} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build/qw3")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/Qwen3.6-27B-Q8_0.gguf"
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--query",
        type=Path,
        default=ROOT / "scripts/kvmem_eval/adaptive_scorer_query_128.txt",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results/adaptive_scorer_ab_10m"
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--archive-tokens", type=int, default=9_998_336)
    parser.add_argument("--ctx", type=int, default=10_065_536)
    parser.add_argument(
        "--arms", default=",".join(ARMS), help="comma-separated arm names"
    )
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be >= 2 (one warmup plus measurements)")
    selected = [name for name in args.arms.split(",") if name]
    unknown = sorted(set(selected) - ARMS.keys())
    if unknown:
        parser.error(f"unknown arms: {', '.join(unknown)}")
    for path in (args.binary, args.model, args.query):
        if not path.is_file():
            parser.error(f"missing file: {path}")
    if not args.archive.is_dir():
        parser.error(f"missing archive: {args.archive}")

    active = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    ).stdout.strip()
    if active:
        parser.error(f"GPU is busy with compute pid(s): {active}")

    query = args.query.read_text(encoding="utf-8").strip()
    if not query or "\n" in query:
        parser.error("--query must contain exactly one non-empty line")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.out_dir / "questions.repeated.txt"
    questions_path.write_text((query + "\n") * args.repeats, encoding="utf-8")
    shutil.copy2(args.binary, args.out_dir / "qw3.tested")

    summary: dict[str, Any] = {
        "schema": "qw3.adaptive-scorer-ab-10m.v1",
        "archive": str(args.archive),
        "archive_tokens": args.archive_tokens,
        "ctx": args.ctx,
        "block_tokens": 64,
        "execution_view_tokens": 65_536,
        "query_score_tokens": 128,
        "repeats": args.repeats,
        "gpu": subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            stdout=subprocess.PIPE,
            check=False,
        ).stdout.strip(),
        "arms": [],
    }
    for name in selected:
        config = ARMS[name]
        arm_dir = args.out_dir / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        log_path = arm_dir / "binary.log"
        results_path = arm_dir / "questions.jsonl"
        cmd = [
            str(args.binary),
            "archive",
            "query",
            "--model",
            str(args.model),
            "--kvmem-archive",
            str(args.archive),
            "--ctx",
            str(args.ctx),
            "--archive-tokens",
            str(args.archive_tokens),
            "--kvmem-block-tokens",
            "64",
            "--kvmem-budget",
            "65536",
            "--kvmem-gen-budget",
            "32768",
            "--kvmem-sink-tokens",
            "512",
            "--kvmem-recent-tokens",
            "0",
            "--kvmem-cpu-gb",
            "32",
            "--kvmem-gpu-memory-ratio",
            "0.50",
            "--kvmem-method",
            "retrieval",
            "--kvmem-query-conditioned",
            "--kvmem-retrieval-method",
            "key-direction-adaptive",
            "--kvmem-adaptive-gain-1to2",
            "0.10",
            "--kvmem-adaptive-gain-2to4",
            "0.06",
            "--kvmem-adaptive-score-mode",
            str(config["mode"]),
            "--kvmem-index-placement",
            "cpu",
            "--kvmem-index-staging-mb",
            str(config["staging_mb"]),
            "--kvmem-numa-policy",
            "auto",
            "--kvmem-opt-stage-out",
            "on",
            "--kvmem-opt-stage-in",
            "on",
            "--kvmem-opt-pack",
            "on",
            "--prefill-chunk",
            "2048",
            "--archive-question-format",
            "qwen-chat-no-thinking",
            "--archive-questions-file",
            str(questions_path),
            "--archive-results-file",
            str(results_path),
            "--temp",
            "0",
            "-n",
            "8",
        ]
        env = os.environ.copy()
        env.update(
            {
                "QW3_Q8_BF16_MAIN": "0",
                "QW3_KVMEM_QUERY_REPLAY": "1",
                "QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS": "1",
                "QW3_KVMEM_DROP_PAGE_CACHE": "1",
                "QW3_KVMEM_PERF_TRACE": "1",
                "QW3_KVMEM_TRACE": "1",
                "QW3_KVMEM_ADAPTIVE_PINNED_INDEX": "1",
                "QW3_KVMEM_ADAPTIVE_BLOCK_STATS_MIB": str(
                    config["block_stats_mb"]
                ),
                "QW3_KVMEM_ADAPTIVE_TWO_PASS_GEMM": (
                    "1" if config["two_pass_gemm"] else "0"
                ),
            }
        )
        print(f"[{time.strftime('%F %T')}] START {name}", flush=True)
        rc, wall_s, peak_gpu_mib, peak_rss_mib, samples = run_and_monitor(
            cmd, env, log_path
        )
        print(
            f"[{time.strftime('%F %T')}] END {name} rc={rc} "
            f"wall={wall_s:.1f}s peak_gpu={peak_gpu_mib}MiB "
            f"peak_rss={peak_rss_mib:.1f}MiB",
            flush=True,
        )
        if rc != 0:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                    -100:
                ]
            )
            print(tail, file=sys.stderr)
            return rc
        arm = parse_arm(
            name,
            config,
            log_path,
            results_path,
            wall_s,
            peak_gpu_mib,
            peak_rss_mib,
            samples,
        )
        arm["requested_staging_mb"] = config["staging_mb"]
        arm["block_stats_cap_mb"] = config["block_stats_mb"]
        arm["two_pass_gemm_enabled"] = config["two_pass_gemm"]
        arm["command"] = cmd
        summary["arms"].append(arm)
        baseline_gpu = min(row["peak_process_gpu_mib"] for row in summary["arms"])
        for row in summary["arms"]:
            row["peak_gpu_delta_mib"] = row["peak_process_gpu_mib"] - baseline_gpu
        reference = summary["arms"][0]["warm_selected_hashes"]
        for row in summary["arms"]:
            row["hash_match_reference"] = row["warm_selected_hashes"] == reference
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "summary.md").write_text(render(summary), encoding="utf-8")
        print(render(summary), flush=True)

    production = [
        row for row in summary["arms"] if row["name"] != "tiled_two_pass_scalar"
    ]
    if production and not all(row["hash_match_reference"] for row in production):
        raise RuntimeError("production scorer arms changed exact selected-block hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
