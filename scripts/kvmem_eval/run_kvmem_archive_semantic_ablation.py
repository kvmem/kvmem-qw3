#!/usr/bin/env python3
"""Run semantic-reselection ablations against one frozen KVMem archive.

The expensive historical prefill is not part of this harness.  Every arm
attaches the same sealed archive, rebuilds the derived retrieval index once,
and then asks the same list of frozen-branch questions.  Only trace records
tagged ``archive-qN`` are included; initial pressure/base materialization and
other explicit selections are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
ARMS = (
    ("all-off", "off", "off", "off"),
    ("pack-only", "off", "off", "on"),
    ("pack+stage-out", "on", "off", "on"),
    ("all-on", "on", "on", "on"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/qw3"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-tokens", type=int, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--question-limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, default=128)
    parser.add_argument("--budget", type=int, default=204800)
    parser.add_argument("--gen-budget", type=int, default=32768)
    parser.add_argument("--sink-tokens", type=int, default=2048)
    parser.add_argument("--recent-tokens", type=int, default=16384)
    parser.add_argument("--cpu-gb", type=float, default=96.0)
    parser.add_argument("--gpu-memory-ratio", type=float, default=0.50)
    parser.add_argument("--index-placement", choices=("gpu", "cpu"),
                        default="cpu")
    parser.add_argument("--index-staging-mb", type=int, default=64)
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--question-format",
                        choices=("raw", "qwen-chat", "qwen-chat-no-thinking"),
                        default="qwen-chat-no-thinking")
    parser.add_argument("--arms", default=",".join(arm[0] for arm in ARMS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_kv(line: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in KV_RE.finditer(line)}


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def summarize_log(path: Path, expected_questions: int) -> dict[str, Any]:
    semantic: list[dict[str, str]] = []
    accounting: list[dict[str, str]] = []
    scorer: list[dict[str, str]] = []
    opt: dict[str, str] = {}
    archive_attach: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "[kvmem-reselect-perf] kind=semantic" in line:
            row = parse_kv(line)
            if row.get("tag", "").startswith("archive-q"):
                semantic.append(row)
        elif "[qw3-native-accounting]" in line:
            accounting.append(parse_kv(line))
        elif "[kvmem-scorer]" in line:
            row = parse_kv(line)
            if row.get("tag", "").startswith("archive-q"):
                scorer.append(row)
        elif "[kvmem-opt]" in line:
            opt = parse_kv(line)
        elif "archive attach: requested=" in line:
            archive_attach = line.strip()

    if len(semantic) != expected_questions:
        raise RuntimeError(
            f"{path}: semantic events={len(semantic)}, expected={expected_questions}"
        )
    if len(accounting) != expected_questions:
        raise RuntimeError(
            f"{path}: accounting rows={len(accounting)}, expected={expected_questions}"
        )
    if len(scorer) != expected_questions:
        raise RuntimeError(
            f"{path}: scorer rows={len(scorer)}, expected={expected_questions}"
        )
    fallback = [row for row in scorer if row.get("fallback") != "0"]
    if fallback:
        raise RuntimeError(f"{path}: scorer fallback detected: {fallback}")

    fields = (
        "selection_ms",
        "selection_wall_ms",
        "stage_out_submit_wall_ms",
        "materialize_wall_ms",
        "stage_out_ms",
        "stage_in_wall_ms",
        "cpu_cache_admit_ms",
        "cpu_cache_admit_gib",
        "cpu_in_blocks",
        "nvme_in_blocks",
        "assemble_ms",
        "total_ms",
    )
    metrics = {
        # Admission fields were added with archive heat-cache support. Keep
        # older ablation logs readable as zero-admission baselines.
        field: stats([float(row.get(field, "0")) for row in semantic])
        for field in fields
    }
    cpu_hit_rates = []
    for row in semantic:
        cpu_blocks = float(row.get("cpu_in_blocks", "0"))
        nvme_blocks = float(row.get("nvme_in_blocks", "0"))
        incoming = cpu_blocks + nvme_blocks
        cpu_hit_rates.append(cpu_blocks / incoming if incoming else 1.0)
    metrics["incoming_cpu_hit_rate"] = stats(cpu_hit_rates)
    replay = [float(row["post_query_replay_ms"]) for row in accounting]
    kvmem_total = [float(row["total_ms"]) + replay[i]
                   for i, row in enumerate(semantic)]
    metrics["query_replay_ms"] = stats(replay)
    metrics["semantic_kvmem_total_ms"] = stats(kvmem_total)
    steady = semantic[1:] if len(semantic) > 1 else []
    replay_steady = replay[1:] if len(replay) > 1 else []
    kvmem_total_steady = kvmem_total[1:] if len(kvmem_total) > 1 else []
    steady_metrics = {
        "reselection_total_ms": stats(
            [float(row["total_ms"]) for row in steady]
        ),
        "query_replay_ms": stats(replay_steady),
        "semantic_kvmem_total_ms": stats(kvmem_total_steady),
    }
    closure_errors = [
        abs(
            float(row["total_ms"])
            - float(row["selection_wall_ms"])
            - float(row["stage_out_submit_wall_ms"])
            - float(row["materialize_wall_ms"])
        )
        for row in semantic
    ]
    return {
        "questions": expected_questions,
        "optimization": opt,
        "archive_attach": archive_attach,
        "selected_hashes": [row["selected_hash"] for row in semantic],
        "fallbacks": 0,
        "metrics": metrics,
        "cold_first_query": {
            "reselection_total_ms": float(semantic[0]["total_ms"]),
            "query_replay_ms": replay[0],
            "semantic_kvmem_total_ms": kvmem_total[0],
        },
        "steady_state_questions": len(steady),
        "steady_state_metrics": steady_metrics,
        "max_nonoverlap_closure_error_ms": max(closure_errors),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Frozen-archive semantic reselection ablation",
        "",
        "Only `kind=semantic`, `tag=archive-qN` events are included. Initial "
        "archive attach/index rebuild and pressure/base materialization are excluded.",
        "",
        "The non-overlapping operation identity is: `semantic KVMem total = "
        "reselection total + query replay`. Inside reselection, `selection wall + "
        "stage-out submit wall + materialization critical path = reselection total`.",
        "Stage-in and assembly can overlap and are reported only as diagnostics.",
        "",
        "| Arm | N | Score/select (ms) | Exposed stage-out submit (ms) | "
        "Materialization critical path (ms) | Reselection total (ms) | "
        "Query replay (ms) | Semantic KVMem total (ms) | Stage-in diagnostic (ms) | "
        "Assembly diagnostic (ms) | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arms"]:
        m = arm["metrics"]
        mean = lambda key: m[key]["mean"]
        lines.append(
            f"| {arm['name']} | {arm['questions']} | {mean('selection_wall_ms'):.3f} | "
            f"{mean('stage_out_submit_wall_ms'):.3f} | "
            f"{mean('materialize_wall_ms'):.3f} | {mean('total_ms'):.3f} | "
            f"{mean('query_replay_ms'):.3f} | "
            f"{mean('semantic_kvmem_total_ms'):.3f} | "
            f"{mean('stage_in_wall_ms'):.3f} | {mean('assemble_ms'):.3f} | "
            f"{arm['fallbacks']} |"
        )
    lines.extend([
        "",
        "Cold/steady-state split (q0 is cold; q1..qN reuse the process and host cache):",
        "",
        "| Arm | Cold q0 total (ms) | Steady N | Steady mean (ms) | "
        "Steady median (ms) | Steady p95 (ms) | Speedup vs all-off | "
        "Max closure error (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    baseline = None
    for arm in summary["arms"]:
        steady = arm["steady_state_metrics"]["semantic_kvmem_total_ms"]
        if arm["name"] == "all-off" and steady:
            baseline = steady["mean"]
        speedup = baseline / steady["mean"] if baseline and steady else None
        lines.append(
            f"| {arm['name']} | "
            f"{arm['cold_first_query']['semantic_kvmem_total_ms']:.3f} | "
            f"{arm['steady_state_questions']} | "
            f"{steady.get('mean', float('nan')):.3f} | "
            f"{steady.get('median', float('nan')):.3f} | "
            f"{steady.get('p95', float('nan')):.3f} | "
            f"{f'{speedup:.3f}x' if speedup else 'NA'} | "
            f"{arm['max_nonoverlap_closure_error_ms']:.6f} |"
        )
    lines.extend([
        "",
        "Storage diagnostics (means over semantic questions; CPU hit rate is "
        "CPU incoming blocks / (CPU + NVMe incoming blocks)):",
        "",
        "| Arm | CPU hit rate | CPU hits (blocks) | SSD misses (blocks) | "
        "CPU admission (GiB) | Admission work (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in summary["arms"]:
        m = arm["metrics"]
        lines.append(
            f"| {arm['name']} | "
            f"{m['incoming_cpu_hit_rate']['mean']:.3%} | "
            f"{m['cpu_in_blocks']['mean']:.1f} | "
            f"{m['nvme_in_blocks']['mean']:.1f} | "
            f"{m['cpu_cache_admit_gib']['mean']:.3f} | "
            f"{m['cpu_cache_admit_ms']['mean']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    selected_names = [item.strip() for item in args.arms.split(",") if item.strip()]
    arm_map = {arm[0]: arm for arm in ARMS}
    unknown = set(selected_names) - set(arm_map)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    questions = [line.rstrip("\n") for line in
                 args.questions.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if args.question_limit is not None:
        questions = questions[: args.question_limit]
    if not questions:
        raise ValueError("question file is empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    question_file = args.out_dir / "questions.used.txt"
    question_file.write_text("\n".join(questions) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "archive": str(args.archive),
        "archive_tokens": args.archive_tokens,
        "questions_file": str(args.questions),
        "questions": len(questions),
        "arms": [],
    }
    reference_hashes: list[str] | None = None
    for name in selected_names:
        _, stage_out, stage_in, pack = arm_map[name]
        log_path = args.out_dir / f"{name}.log"
        if not (args.resume and log_path.exists()):
            cmd = [
                str(args.binary), "archive", "query",
                "--model", str(args.model),
                "--kvmem-archive", str(args.archive),
                "--ctx", str(args.ctx),
                "--archive-tokens", str(args.archive_tokens),
                "--kvmem-block-tokens", str(args.block_tokens),
                "--kvmem-budget", str(args.budget),
                "--kvmem-gen-budget", str(args.gen_budget),
                "--kvmem-sink-tokens", str(args.sink_tokens),
                "--kvmem-recent-tokens", str(args.recent_tokens),
                "--kvmem-cpu-gb", str(args.cpu_gb),
                "--kvmem-gpu-memory-ratio", str(args.gpu_memory_ratio),
                "--kvmem-index-placement", args.index_placement,
                "--kvmem-index-staging-mb", str(args.index_staging_mb),
                "--prefill-chunk", str(args.prefill_chunk),
                "--kvmem-retrieval-method", "mean-k",
                "--kvmem-opt-stage-out", stage_out,
                "--kvmem-opt-stage-in", stage_in,
                "--kvmem-opt-pack", pack,
                "--archive-question-format", args.question_format,
                "--archive-questions-file", str(question_file),
                "-n", str(args.max_tokens),
            ]
            env = os.environ.copy()
            env.update({
                "QW3_Q8_BF16_MAIN": "0",
                "QW3_KVMEM_QUERY_REPLAY": "1",
                "QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS": "1",
                "QW3_KVMEM_DROP_PAGE_CACHE": "1",
                "QW3_KVMEM_PERF_TRACE": "1",
            })
            print(f"[{time.strftime('%F %T')}] START {name}", flush=True)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                        env=env, text=True)
            print(f"[{time.strftime('%F %T')}] END {name} "
                  f"rc={result.returncode} wall={time.monotonic()-started:.1f}s",
                  flush=True)
            if result.returncode != 0:
                tail = "\n".join(log_path.read_text(
                    encoding="utf-8", errors="replace").splitlines()[-80:])
                print(tail, file=sys.stderr)
                return result.returncode

        arm_summary = summarize_log(log_path, len(questions))
        arm_summary["name"] = name
        arm_summary["log"] = str(log_path)
        expected_opt = {
            "stage_out": "1" if stage_out == "on" else "0",
            "stage_in": "1" if stage_in == "on" else "0",
            "pack": "1" if pack == "on" else "0",
        }
        actual_opt = arm_summary["optimization"]
        for key, expected in expected_opt.items():
            if actual_opt.get(key) != expected:
                raise RuntimeError(
                    f"{name}: requested {key}={expected}, trace says "
                    f"{actual_opt.get(key)!r}"
                )
        hashes = arm_summary["selected_hashes"]
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise RuntimeError(
                f"{name}: selected block hashes differ from the first arm; "
                "the optimization ablation changed semantics"
            )
        summary["arms"].append(arm_summary)
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.out_dir / "summary.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print(render_markdown(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
