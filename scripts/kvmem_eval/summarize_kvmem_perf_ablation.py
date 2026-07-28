#!/usr/bin/env python3
"""Summarize KVMem core-ablation logs into JSON and a paper-ready table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
from typing import Any


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")


def parse_kv_line(line: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in KV_RE.finditer(line)}


def as_float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return None


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def aggregate(rows: list[dict[str, str]], key: str) -> dict[str, float | None]:
    values = [value for row in rows if (value := as_float(row, key)) is not None]
    return {
        "sum": sum(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def read_gpu_peak(path: pathlib.Path) -> float | None:
    if not path.exists():
        return None
    peak: float | None = None
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        next(f, None)
        for line in f:
            # Older logs used a locale comma in the fractional timestamp.
            # The four nvidia-smi fields are always the final four columns.
            fields = next(csv.reader([line]))
            if len(fields) < 5:
                continue
            try:
                used = float(fields[-3].strip())
            except ValueError:
                continue
            peak = used if peak is None else max(peak, used)
    return peak


def read_answer_timing(path: pathlib.Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    timings = [row.get("timing", {}) for row in rows]
    usage = [row.get("usage", {}) for row in rows]

    def mean_field(source: list[dict[str, Any]], key: str) -> float | None:
        values = [
            float(row[key]) for row in source
            if row.get(key) is not None
        ]
        return statistics.fmean(values) if values else None

    return {
        "samples": len(rows),
        "mean_ttft_sec": mean_field(timings, "ttft_sec"),
        "mean_content_ttft_sec": mean_field(timings, "content_ttft_sec"),
        "mean_total_sec": mean_field(timings, "total_sec"),
        "mean_decode_after_ttft_sec": mean_field(
            timings, "decode_elapsed_after_ttft_sec"
        ),
        "mean_output_tokens_per_sec": mean_field(
            timings, "output_tokens_per_sec_after_ttft"
        ),
        "mean_prompt_tokens": mean_field(usage, "prompt_tokens"),
        "mean_completion_tokens": mean_field(usage, "completion_tokens"),
    }


def summarize_cell(
    name: str,
    tag: str,
    log_root: pathlib.Path,
    results_root: pathlib.Path,
) -> dict[str, Any]:
    server_log = log_root / f"{tag}_server.log"
    gpu_log = log_root / f"{tag}_gpu.csv"
    answers = results_root / tag / "answers.jsonl"
    reselect: list[dict[str, str]] = []
    assembly: list[dict[str, str]] = []
    reuse: list[dict[str, str]] = []
    statuses: list[dict[str, str]] = []
    config: dict[str, str] = {}
    if server_log.exists():
        with server_log.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if "[kvmem-reselect-perf]" in line:
                    reselect.append(parse_kv_line(line))
                elif "[kvmem-assembly-perf]" in line:
                    assembly.append(parse_kv_line(line))
                elif "[kvmem-reuse]" in line:
                    reuse.append(parse_kv_line(line))
                elif "[kvmem-opt-status]" in line:
                    statuses.append(parse_kv_line(line))
                elif "[kvmem-opt]" in line:
                    config = parse_kv_line(line)

    metric_names = (
        "selection_ms",
        "stage_out_ms",
        "stage_out_d2h_ms",
        "stage_out_cpu_copy_ms",
        "stage_out_disk_write_ms",
        "stage_in_wall_ms",
        "cpu_gather_ms",
        "cpu_h2d_enqueue_ms",
        "cpu_h2d_wait_ms",
        "nvme_read_ms",
        "h2d_wait_ms",
        "assemble_ms",
        "total_ms",
    )
    metrics = {key: aggregate(reselect, key) for key in metric_names}
    overlap = sum(
        int(row.get("selection_overlap_blocks", "0")) for row in reuse
    )
    reused = sum(int(row.get("gpu_reused_blocks", "0")) for row in reuse)
    stage_in = sum(int(row.get("stage_in_blocks", "0")) for row in reuse)
    return {
        "name": name,
        "tag": tag,
        "server_log": str(server_log),
        "gpu_log": str(gpu_log),
        "answers": str(answers),
        "optimization_config": config,
        "optimization_status": statuses,
        "reselection_events": len(reselect),
        "assembly_events": len(assembly),
        "metrics_ms": metrics,
        "reuse": {
            "events": len(reuse),
            "selection_overlap_blocks": overlap,
            "gpu_reused_blocks": reused,
            "gpu_reuse_fraction_of_overlap": (
                reused / overlap if overlap else None
            ),
            "stage_in_blocks": stage_in,
        },
        "peak_gpu_memory_mib": read_gpu_peak(gpu_log),
        "request": read_answer_timing(answers),
    }


def fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(cells: list[dict[str, Any]]) -> str:
    lines = [
        "# KVMem core performance ablation",
        "",
        "## Request-level latency",
        "",
        "TTFT is the request time to the first reasoning token. Reselection is "
        "already included in TTFT; it must not be added to TTFT again. The "
        "non-reselection column is a residual, not a separately instrumented "
        "timer.",
        "",
        "| Cell | Prompt tokens | TTFT total (s) | Reselection within TTFT (s) | "
        "TTFT minus reselection (s) | Post-TTFT generation, reasoning + answer (s) | "
        "Completion tokens | Generation throughput (token/s) | "
        "Full request total (s) | Peak GPU (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        metrics = cell["metrics_ms"]
        request = cell["request"]
        reselect_total = (
            metrics["total_ms"]["sum"] / 1000.0
            if metrics["total_ms"]["sum"] is not None else None
        )
        ttft = request.get("mean_ttft_sec")
        non_reselection = (
            ttft - reselect_total
            if ttft is not None and reselect_total is not None else None
        )
        lines.append(
            "| {name} | {prompt} | {ttft} | {reselect} | {residual} | "
            "{decode} | {completion} | {throughput} | {total} | {gpu} |".format(
                name=cell["name"],
                prompt=fmt(request.get("mean_prompt_tokens"), 0),
                ttft=fmt(ttft),
                reselect=fmt(reselect_total),
                residual=fmt(non_reselection),
                decode=fmt(request.get("mean_decode_after_ttft_sec")),
                completion=fmt(request.get("mean_completion_tokens"), 0),
                throughput=fmt(request.get("mean_output_tokens_per_sec")),
                total=fmt(request.get("mean_total_sec")),
                gpu=fmt(cell["peak_gpu_memory_mib"], 0),
            )
        )
    lines.extend(
        [
            "",
            "## Reselection breakdown",
            "",
            "The indented columns are internal components of "
            "`Reselection total`, which is itself included in TTFT. "
            "Stage-in and assembly can overlap, so component times are "
            "diagnostic timers and are not required to add up exactly to the "
            "parent total.",
            "",
            "| Cell | Reselections | **Reselection total (s)** | "
            "↳ Selection (s) | ↳ Stage-out (s) | ↳ Stage-in wall (s) | "
            "↳ Assembly (s) | GPU reuse fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in cells:
        metrics = cell["metrics_ms"]
        reuse = cell["reuse"]
        lines.append(
            "| {name} | {events} | **{total}** | {selection} | "
            "{stage_out} | {stage_in} | {assembly} | {reuse} |".format(
                name=cell["name"],
                events=cell["reselection_events"],
                total=fmt(
                    metrics["total_ms"]["sum"] / 1000.0
                    if metrics["total_ms"]["sum"] is not None else None,
                    3,
                ),
                selection=fmt(
                    metrics["selection_ms"]["sum"] / 1000.0
                    if metrics["selection_ms"]["sum"] is not None else None,
                    3,
                ),
                stage_out=fmt(
                    metrics["stage_out_ms"]["sum"] / 1000.0
                    if metrics["stage_out_ms"]["sum"] is not None else None,
                    3,
                ),
                stage_in=fmt(
                    metrics["stage_in_wall_ms"]["sum"] / 1000.0
                    if metrics["stage_in_wall_ms"]["sum"] is not None else None,
                    3,
                ),
                assembly=fmt(
                    metrics["assemble_ms"]["sum"] / 1000.0
                    if metrics["assemble_ms"]["sum"] is not None else None,
                    3,
                ),
                reuse=fmt(reuse["gpu_reuse_fraction_of_overlap"], 3),
            )
        )
    lines.extend(
        [
            "",
            "All reselection values are sums over the events in that row. "
            "The JSON artifact also contains per-component mean, median, and "
            "p95 values.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="NAME=TAG",
        help="logical cell name and run tag",
    )
    parser.add_argument("--log-root", type=pathlib.Path, required=True)
    parser.add_argument("--results-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-markdown", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cells = []
    for spec in args.cell:
        if "=" not in spec:
            raise SystemExit(f"invalid --cell {spec!r}; expected NAME=TAG")
        name, tag = spec.split("=", 1)
        cells.append(
            summarize_cell(name, tag, args.log_root, args.results_root)
        )
    output = {"schema_version": 2, "cells": cells}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown(cells), encoding="utf-8")
    print(markdown(cells))


if __name__ == "__main__":
    main()
