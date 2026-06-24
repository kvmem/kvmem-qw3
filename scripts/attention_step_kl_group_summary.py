#!/usr/bin/env python3
"""Summarize sliding-window KL results by source-segment groups."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_group(raw: str) -> dict[str, object]:
    try:
        name, span = raw.split(":", 1)
        start_s, end_s = span.split("-", 1)
        start = int(start_s)
        end = int(end_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "group must be NAME:START-END, for example astropy__astropy-6938:0-13"
        ) from exc
    if not name:
        raise argparse.ArgumentTypeError("group name cannot be empty")
    if start < 0 or end < start:
        raise argparse.ArgumentTypeError("group segment range is invalid")
    return {"name": name, "start": start, "end": end}


def quantile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "p95": quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def summarize_group(
    summary: dict[str, object],
    group: dict[str, object],
) -> dict[str, object]:
    start = int(group["start"])
    end = int(group["end"])
    name = str(group["name"])

    kept_steps = [
        step for step in summary["kept_steps"]
        if start <= int(step["source_segment"]) <= end
    ]
    source_segments = sorted({int(step["source_segment"]) for step in kept_steps})
    windows = sum(int(step["window_count"]) for step in kept_steps)

    within: list[float] = []
    boundary: list[float] = []
    excluded_cross_group = 0
    for transition in summary["transition_rows"]:
        from_seg = int(transition["from_source_segment"])
        to_seg = int(transition["to_source_segment"])
        if start <= from_seg <= end and start <= to_seg <= end:
            if transition["kind"] == "within":
                within.append(float(transition["kl"]))
            elif transition["kind"] == "boundary":
                boundary.append(float(transition["kl"]))
        elif start <= from_seg <= end or start <= to_seg <= end:
            excluded_cross_group += 1

    within_stats = stats(within)
    boundary_stats = stats(boundary)
    within_mean = float(within_stats["mean"])
    boundary_mean = float(boundary_stats["mean"])
    within_max = float(within_stats["max"])
    ratio = boundary_mean / within_mean if within_mean > 0.0 and boundary else 0.0
    boundary_gt_within_max = sum(1 for value in boundary if value > within_max)
    return {
        "name": name,
        "source_segment_range": [start, end],
        "source_segments_with_kept_steps": source_segments,
        "kept_steps": len(kept_steps),
        "windows": windows,
        "within": within_stats,
        "boundary": boundary_stats,
        "boundary_over_within_mean": ratio,
        "boundary_gt_within_max": boundary_gt_within_max,
        "excluded_cross_group_transitions": excluded_cross_group,
    }


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Per-Instance Sliding KL Summary",
        "",
        "| instance | kept steps | windows | within n | within mean | within max | boundary n | boundary mean | boundary min | boundary / within | boundary > within max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        within = row["within"]
        boundary = row["boundary"]
        assert isinstance(within, dict)
        assert isinstance(boundary, dict)
        lines.append(
            "| {name} | {steps} | {windows} | {within_n} | {within_mean:.4f} | "
            "{within_max:.4f} | {boundary_n} | {boundary_mean:.4f} | "
            "{boundary_min:.4f} | {ratio:.2f}x | {gt}/{boundary_n} |".format(
                name=row["name"],
                steps=row["kept_steps"],
                windows=row["windows"],
                within_n=within["n"],
                within_mean=within["mean"],
                within_max=within["max"],
                boundary_n=boundary["n"],
                boundary_mean=boundary["mean"],
                boundary_min=boundary["min"],
                ratio=row["boundary_over_within_mean"],
                gt=row["boundary_gt_within_max"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--group", action="append", type=parse_group, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [summarize_group(summary, group) for group in args.group]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, rows)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
