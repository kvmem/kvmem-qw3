#!/usr/bin/env python3
"""Summarize archive construction, reuse, and per-question KVMem latency.

The generator's row summaries intentionally stay small.  This companion reads
the durable JSONL and compact PERF_TRACE rows so the final benchmark can report
both utility and the cost saved by building each context exactly once.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


BUILD_RE = re.compile(r"archive build: sealed .*? total=([0-9.]+)s")
ATTACH_RE = re.compile(
    r"archive attach:.*?\battach=([0-9.]+)s"
    r"(?:\s+replay=([0-9.]+)s)?\s+index=([0-9.]+)s"
)
KEY_VALUE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output-stem", default="archive_performance_summary")
    return p.parse_args()


def values(line: str) -> dict[str, str]:
    return dict(KEY_VALUE_RE.findall(line))


def number(fields: dict[str, str], key: str) -> float:
    raw = fields.get(key, "nan").rstrip(",")
    try:
        return float(raw)
    except ValueError:
        return math.nan


def mean(xs: list[float]) -> float | None:
    clean = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(clean) if clean else None


def percentile(xs: list[float], fraction: float) -> float | None:
    clean = sorted(x for x in xs if math.isfinite(x))
    if not clean:
        return None
    position = fraction * (len(clean) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - position) + clean[hi] * (position - lo)


def metric_summary(xs: list[float]) -> dict[str, Any]:
    clean = [x for x in xs if math.isfinite(x)]
    return {
        "n": len(clean),
        "mean": mean(clean),
        "median": percentile(clean, 0.5),
        "p95": percentile(clean, 0.95),
    }


def parse_row(row_dir: Path) -> dict[str, Any]:
    summary = json.loads((row_dir / "row_summary.json").read_text(
        encoding="utf-8"))
    build_text = (row_dir / "build.log").read_text(
        encoding="utf-8", errors="replace")
    query_text = (row_dir / "query.log").read_text(
        encoding="utf-8", errors="replace")
    build_matches = BUILD_RE.findall(build_text)
    attach_matches = ATTACH_RE.findall(query_text)
    answers = [
        json.loads(line)
        for line in (row_dir / "archive_answers.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    semantic: list[dict[str, str]] = []
    initial_explicit: list[dict[str, str]] = []
    accounting: list[dict[str, str]] = []
    scorer: list[dict[str, str]] = []
    for line in query_text.splitlines():
        if "[kvmem-reselect-perf]" in line and "kind=semantic" in line:
            fields = values(line)
            if fields.get("tag", "").startswith("archive-q"):
                semantic.append(fields)
        elif "[kvmem-reselect-perf]" in line and "kind=explicit" in line:
            fields = values(line)
            if fields.get("tag") == "-":
                initial_explicit.append(fields)
        elif "[qw3-native-accounting]" in line:
            accounting.append(values(line))
        elif "[kvmem-scorer]" in line:
            fields = values(line)
            if fields.get("tag", "").startswith("archive-q"):
                scorer.append(fields)
    expected = int(summary["questions_completed"])
    if len(answers) != expected:
        raise RuntimeError(
            f"{row_dir.name}: answers={len(answers)} expected={expected}"
        )
    if len(semantic) != expected or len(accounting) != expected:
        raise RuntimeError(
            f"{row_dir.name}: semantic={len(semantic)} accounting="
            f"{len(accounting)} expected={expected}"
        )
    if len(scorer) != expected:
        raise RuntimeError(
            f"{row_dir.name}: scorer={len(scorer)} expected={expected}"
        )
    if len(build_matches) != 1 or len(attach_matches) != 1:
        raise RuntimeError(
            f"{row_dir.name}: build markers={len(build_matches)} attach "
            f"markers={len(attach_matches)}"
        )
    attach_raw, residual_replay_raw, index_raw = attach_matches[0]
    attach_s = float(attach_raw)
    residual_replay_s = float(residual_replay_raw or 0.0)
    index_s = float(index_raw)
    generation_limit = int(summary["max_generation_tokens"])
    return {
        "row": row_dir.name,
        "split": summary["split"],
        "source": summary["source"],
        "questions": expected,
        "archive_tokens": summary["archive_tokens"],
        "archive_storage": summary["archive_storage"],
        "build_s": float(build_matches[0]),
        "attach_s": attach_s,
        "residual_replay_s": residual_replay_s,
        "index_rebuild_s": index_s,
        "initial_materialization_ms": [
            number(x, "total_ms") for x in initial_explicit
        ],
        "question_wall_s": [float(a["wall_s"]) for a in answers],
        "question_prefill_s": [float(a["prefill_s"]) for a in answers],
        "question_decode_s": [float(a["decode_s"]) for a in answers],
        "decoded_tokens": [int(a["decoded_tokens"]) for a in answers],
        # The archive CLI does not currently persist finish_reason.  Equality
        # with the configured source-specific limit is therefore reported as
        # a conservative "limit hit", not asserted to be truncation.
        "generation_limit": generation_limit,
        "generation_limit_hits": sum(
            int(a["decoded_tokens"] >= generation_limit) for a in answers
        ),
        "empty_answers": sum(
            int(not str(a.get("answer", "")).strip()) for a in answers
        ),
        "selection_ms": [number(x, "selection_wall_ms") for x in semantic],
        "stage_out_ms": [
            number(x, "stage_out_submit_wall_ms") for x in semantic
        ],
        "materialize_ms": [
            number(x, "materialize_wall_ms") for x in semantic
        ],
        "reselect_ms": [number(x, "total_ms") for x in semantic],
        "post_semantic_ms": [
            number(x, "post_semantic_ms") for x in accounting
        ],
        "query_replay_ms": [
            number(x, "post_query_replay_ms") for x in accounting
        ],
        "kvmem_question_ms": [
            number(x, "post_semantic_ms") + number(x, "post_query_replay_ms")
            for x in accounting
        ],
        "accounting_error_ms": [
            number(x, "accounting_error_ms") for x in semantic
        ],
        "cpu_in_blocks": sum(number(x, "cpu_in_blocks") for x in semantic),
        "nvme_in_blocks": sum(number(x, "nvme_in_blocks") for x in semantic),
        "fallbacks": sum(int(x.get("fallback", "0")) for x in scorer),
    }


def main() -> int:
    args = parse_args()
    rows_root = args.results_dir / "rows"
    rows = [parse_row(p.parent) for p in sorted(
        rows_root.glob("*/row_summary.json"))]
    if not rows:
        raise RuntimeError("no completed rows")
    questions = sum(row["questions"] for row in rows)
    build_once = sum(row["build_s"] for row in rows)
    build_without_reuse = sum(
        row["build_s"] * row["questions"] for row in rows
    )
    all_values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name in (
            "question_wall_s", "question_prefill_s", "question_decode_s",
            "selection_ms", "stage_out_ms", "materialize_ms", "reselect_ms",
            "post_semantic_ms", "query_replay_ms", "kvmem_question_ms",
            "initial_materialization_ms", "accounting_error_ms",
        ):
            all_values[name].extend(row[name])
    cpu_in = sum(row["cpu_in_blocks"] for row in rows)
    nvme_in = sum(row["nvme_in_blocks"] for row in rows)
    generation_by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        limits = {row["generation_limit"] for row in source_rows}
        if len(limits) != 1:
            raise RuntimeError(
                f"inconsistent generation limits for {source}: {limits}"
            )
        decoded = [
            value for row in source_rows for value in row["decoded_tokens"]
        ]
        generation_by_source[source] = {
            "questions": len(decoded),
            "generation_limit": next(iter(limits)),
            "limit_hits": sum(
                row["generation_limit_hits"] for row in source_rows
            ),
            "empty_answers": sum(row["empty_answers"] for row in source_rows),
            "decoded_tokens": metric_summary(decoded),
        }
    output = {
        "contexts": len(rows),
        "questions": questions,
        "archive_tokens_total": sum(row["archive_tokens"] for row in rows),
        "storage_contexts": dict(sorted(defaultdict(int, {
            name: sum(row["archive_storage"] == name for row in rows)
            for name in {row["archive_storage"] for row in rows}
        }).items())),
        "context_build_once_s": build_once,
        "context_build_amortized_s_per_question": build_once / questions,
        "context_build_if_repeated_s": build_without_reuse,
        "context_build_time_avoided_s": build_without_reuse - build_once,
        "attach_s_total": sum(row["attach_s"] for row in rows),
        "index_rebuild_s_total": sum(row["index_rebuild_s"] for row in rows),
        "initial_materialization_s_total": sum(
            sum(row["initial_materialization_ms"]) / 1000.0 for row in rows
        ),
        "fixed_archive_setup_amortized_s_per_question": sum(
            row["attach_s"] + row["index_rebuild_s"] +
            sum(row["initial_materialization_ms"]) / 1000.0
            for row in rows
        ) / questions,
        "metrics": {
            name: metric_summary(xs) for name, xs in all_values.items()
        },
        "semantic_stage_in": {
            "cpu_blocks": cpu_in,
            "nvme_blocks": nvme_in,
            "cpu_hit_rate": cpu_in / (cpu_in + nvme_in)
            if cpu_in + nvme_in else None,
        },
        "generation_health": {
            "limit_hits": sum(row["generation_limit_hits"] for row in rows),
            "empty_answers": sum(row["empty_answers"] for row in rows),
            "by_source": generation_by_source,
        },
        "fallbacks": sum(row["fallbacks"] for row in rows),
        "max_accounting_error_ms": max(
            abs(x) for x in all_values["accounting_error_ms"]
        ),
        "rows": rows,
    }
    json_path = args.results_dir / f"{args.output_stem}.json"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    m = output["metrics"]
    md = [
        "# MemoryAgentBench archive performance summary",
        "",
        f"- Contexts/questions: {len(rows)} / {questions}",
        f"- Context build once: {build_once:.3f} s",
        "- Amortized build per question: "
        f"{output['context_build_amortized_s_per_question']:.3f} s",
        "- Amortized attach/index/initial-materialization per question: "
        f"{output['fixed_archive_setup_amortized_s_per_question']:.3f} s",
        "- Avoided repeated context-build time: "
        f"{output['context_build_time_avoided_s']:.3f} s",
        f"- Scorer fallbacks: {output['fallbacks']}",
        "- Generation limit hits / empty answers: "
        f"{output['generation_health']['limit_hits']} / "
        f"{output['generation_health']['empty_answers']}",
        "",
        "| Per-question metric | Mean | Median | P95 | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "question_wall_s", "kvmem_question_ms", "post_semantic_ms", "selection_ms",
        "stage_out_ms", "materialize_ms", "query_replay_ms",
    ):
        x = m[name]
        md.append(
            f"| {name} | {x['mean']:.3f} | {x['median']:.3f} | "
            f"{x['p95']:.3f} | {x['n']} |"
        )
    md.extend([
        "",
        "| Source | Questions | Generation cap | Limit hits | Empty | "
        "Decoded mean | Decoded p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for source, health in generation_by_source.items():
        decoded = health["decoded_tokens"]
        md.append(
            f"| {source} | {health['questions']} | "
            f"{health['generation_limit']} | {health['limit_hits']} | "
            f"{health['empty_answers']} | {decoded['mean']:.1f} | "
            f"{decoded['p95']:.1f} |"
        )
    md_path = args.results_dir / f"{args.output_stem}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "rows"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
