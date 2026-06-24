#!/usr/bin/env python3
"""Plot top-block cumulative attention mass curves."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from pathlib import Path

import attention_step_kl_curve as kl


DEFAULT_FRACTIONS = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.75,
    1.00,
]
DEFAULT_TOP_K = [1, 2, 4, 8, 16, 32, 64, 128]


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


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "group"


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


def split_always_kept(
    dist: dict[int, float],
    always_keep_blocks: int,
) -> tuple[float, list[float]]:
    kept_mass = sum(
        value for block, value in dist.items()
        if 0 <= block < always_keep_blocks and value > 0.0
    )
    candidate_values = sorted(
        (
            value for block, value in dist.items()
            if block >= always_keep_blocks and value > 0.0
        ),
        reverse=True,
    )
    return kept_mass, candidate_values


def coverage_at_fraction(
    dist: dict[int, float],
    fraction: float,
    always_keep_blocks: int,
) -> float:
    kept_mass, values = split_always_kept(dist, always_keep_blocks)
    if not values:
        return min(1.0, kept_mass)
    k = max(1, math.ceil(len(values) * fraction))
    return min(1.0, kept_mass + sum(values[:k]))


def coverage_at_k(dist: dict[int, float], k: int, always_keep_blocks: int) -> float:
    kept_mass, values = split_always_kept(dist, always_keep_blocks)
    if not values:
        return min(1.0, kept_mass)
    dynamic_k = max(0, k - always_keep_blocks)
    return min(1.0, kept_mass + sum(values[: min(dynamic_k, len(values))]))


def summarize_group(
    windows: list[dict[str, object]],
    group: dict[str, object],
    *,
    fractions: list[float],
    top_k: list[int],
    always_keep_blocks: int,
) -> dict[str, object]:
    start = int(group["start"])
    end = int(group["end"])
    group_windows = [
        window for window in windows
        if start <= int(window["source_segment"]) <= end
    ]
    dists = [window["dist"] for window in group_windows]
    dists = [dist for dist in dists if isinstance(dist, dict) and dist]
    n_blocks = [len(dist) for dist in dists]

    fraction_rows = []
    for fraction in fractions:
        values = [
            coverage_at_fraction(dist, fraction, always_keep_blocks)
            for dist in dists
        ]
        fraction_rows.append(
            {
                "fraction": fraction,
                "mean": sum(values) / len(values) if values else 0.0,
                "p10": quantile(values, 0.10),
                "p50": quantile(values, 0.50),
                "p90": quantile(values, 0.90),
            }
        )

    top_k_rows = []
    for k in top_k:
        values = [coverage_at_k(dist, k, always_keep_blocks) for dist in dists]
        top_k_rows.append(
            {
                "k": k,
                "mean": sum(values) / len(values) if values else 0.0,
                "p10": quantile(values, 0.10),
                "p50": quantile(values, 0.50),
                "p90": quantile(values, 0.90),
            }
        )

    return {
        "name": str(group["name"]),
        "source_segment_range": [start, end],
        "windows": len(dists),
        "block_count": {
            "mean": sum(n_blocks) / len(n_blocks) if n_blocks else 0.0,
            "min": min(n_blocks) if n_blocks else 0,
            "max": max(n_blocks) if n_blocks else 0,
        },
        "always_keep_blocks": always_keep_blocks,
        "coverage_by_fraction": fraction_rows,
        "coverage_by_top_k": top_k_rows,
    }


def write_svg(path: Path, row: dict[str, object]) -> None:
    width, height = 980, 520
    pad_l, pad_r, pad_t, pad_b = 78, 34, 56, 78
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    curves = row["coverage_by_fraction"]
    assert isinstance(curves, list)

    def x_at(frac: float) -> float:
        return pad_l + inner_w * frac

    def y_at(value: float) -> float:
        return pad_t + inner_h * (1.0 - max(0.0, min(1.0, value)))

    title = f"Attention coverage curve: {row['name']}"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_l}" y="28" font-size="21" font-family="sans-serif">{html.escape(title)}</text>',
        f'<text x="{pad_l}" y="48" font-size="13" font-family="sans-serif" fill="#4b5563">Always-kept sink blocks included; line = mean, band = p10-p90 across windows</text>',
        f'<line x1="{pad_l}" y1="{pad_t + inner_h}" x2="{pad_l + inner_w}" y2="{pad_t + inner_h}" stroke="#555"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + inner_h}" stroke="#555"/>',
        f'<text x="12" y="{pad_t + 16}" font-size="14" font-family="sans-serif">attention mass</text>',
        f'<text x="{width // 2 - 128}" y="{height - 22}" font-size="15" font-family="sans-serif">fraction of selected KV blocks</text>',
    ]

    for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
        x = x_at(frac)
        y = y_at(frac)
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + inner_h}" stroke="#eef2f7"/>')
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x - 12:.1f}" y="{height - 50}" font-size="12" font-family="monospace">{int(frac * 100)}%</text>')
        parts.append(f'<text x="32" y="{y + 4:.1f}" font-size="12" font-family="monospace">{frac:.2f}</text>')

    p90_points = [
        (x_at(float(item["fraction"])), y_at(float(item["p90"])))
        for item in curves
    ]
    p10_points = [
        (x_at(float(item["fraction"])), y_at(float(item["p10"])))
        for item in reversed(curves)
    ]
    band_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in p90_points + p10_points)
    parts.append(f'<polygon points="{band_points}" fill="#bfdbfe" opacity="0.45"/>')

    mean_points = " ".join(
        f'{x_at(float(item["fraction"])):.1f},{y_at(float(item["mean"])):.1f}'
        for item in curves
    )
    median_points = " ".join(
        f'{x_at(float(item["fraction"])):.1f},{y_at(float(item["p50"])):.1f}'
        for item in curves
    )
    parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2.6" points="{mean_points}"/>')
    parts.append(f'<polyline fill="none" stroke="#0f172a" stroke-width="1.4" stroke-dasharray="4,4" points="{median_points}"/>')
    for item in curves:
        x = x_at(float(item["fraction"]))
        y = y_at(float(item["mean"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#2563eb"/>')

    parts.extend(
        [
            f'<line x1="{width - 230}" y1="34" x2="{width - 190}" y2="34" stroke="#2563eb" stroke-width="2.6"/>',
            f'<text x="{width - 182}" y="38" font-size="13" font-family="sans-serif">mean coverage</text>',
            f'<line x1="{width - 230}" y1="54" x2="{width - 190}" y2="54" stroke="#0f172a" stroke-width="1.4" stroke-dasharray="4,4"/>',
            f'<text x="{width - 182}" y="58" font-size="13" font-family="sans-serif">median coverage</text>',
            f'<rect x="{width - 230}" y="68" width="40" height="12" fill="#bfdbfe" opacity="0.45"/>',
            f'<text x="{width - 182}" y="79" font-size="13" font-family="sans-serif">p10-p90</text>',
        ]
    )
    block_count = row["block_count"]
    assert isinstance(block_count, dict)
    parts.append(
        f'<text x="{pad_l}" y="{height - 6}" font-size="12" font-family="sans-serif" fill="#4b5563">'
        f'windows={row["windows"]}, always-kept sink blocks={row["always_keep_blocks"]}, '
        f'blocks/window mean={float(block_count["mean"]):.1f}, min={block_count["min"]}, '
        f'max={block_count["max"]}</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Attention Coverage Summary",
        "",
        "| instance | figure | windows | always kept | blocks/window mean | top 1 | top 2 | top 4 | top 8 | top 16 | top 10% | top 20% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        filename = f"{safe_filename(str(row['name']))}.svg"
        by_k = {int(item["k"]): float(item["mean"]) for item in row["coverage_by_top_k"]}
        by_f = {float(item["fraction"]): float(item["mean"]) for item in row["coverage_by_fraction"]}
        block_count = row["block_count"]
        assert isinstance(block_count, dict)
        lines.append(
            "| {name} | [{file}]({file}) | {windows} | {always_keep} | {blocks:.1f} | "
            "{top1:.3f} | {top2:.3f} | {top4:.3f} | {top8:.3f} | {top16:.3f} | "
            "{top10p:.3f} | {top20p:.3f} |".format(
                name=row["name"],
                file=filename,
                windows=row["windows"],
                always_keep=row["always_keep_blocks"],
                blocks=float(block_count["mean"]),
                top1=by_k.get(1, 0.0),
                top2=by_k.get(2, 0.0),
                top4=by_k.get(4, 0.0),
                top8=by_k.get(8, 0.0),
                top16=by_k.get(16, 0.0),
                top10p=by_f.get(0.10, 0.0),
                top20p=by_f.get(0.20, 0.0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path, nargs="+")
    ap.add_argument("--group", action="append", type=parse_group, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--min-samples", type=int, default=32)
    ap.add_argument("--mode", choices=["full_decode", "context_only"], default="context_only")
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument(
        "--always-keep-blocks",
        type=int,
        default=0,
        help=(
            "Number of prefix blocks that are always retained and counted in "
            "coverage before selecting additional high-weight blocks. Use 1 "
            "for an always-kept attention sink block."
        ),
    )
    ap.add_argument("--recent-prefill-blocks", type=int, default=8)
    args = ap.parse_args()

    segments = kl.read_trace(args.trace)
    _, windows = kl.build_windows(
        segments,
        window=args.window,
        stride=args.stride,
        min_samples=args.min_samples,
        mode=args.mode,
        sink_blocks=args.sink_blocks,
        recent_prefill_blocks=args.recent_prefill_blocks,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_group(
            windows,
            group,
            fractions=DEFAULT_FRACTIONS,
            top_k=DEFAULT_TOP_K,
            always_keep_blocks=args.always_keep_blocks,
        )
        for group in args.group
    ]
    (args.out_dir / "coverage_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        write_svg(args.out_dir / f"{safe_filename(str(row['name']))}.svg", row)
    write_markdown(args.out_dir / "coverage_summary.md", rows)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
