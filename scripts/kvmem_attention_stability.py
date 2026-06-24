#!/usr/bin/env python3
"""Analyze QW3_ATTN_TRACE / QW3_KVMEM_ATTN_TRACE attention-mass traces.

The trace records one line per sampled token per standard-attention layer.
This script aggregates layers into a per-sample block distribution, then
computes within-step stability curves:

  * Jensen-Shannon similarity to the first observed distribution.
  * Jensen-Shannon similarity to the previous sample.
  * Top-K Jaccard overlap to the first observed distribution.
  * Mass retained on the top-K blocks from the first observe window.

It intentionally uses only the Python standard library so the report can run in
minimal benchmark environments.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def normalize(dist: dict[int, float]) -> dict[int, float]:
    total = sum(v for v in dist.values() if v > 0.0)
    if total <= 0.0:
        return {}
    return {k: max(0.0, v) / total for k, v in dist.items() if v > 0.0}


def filtered_dist(
    dist: dict[int, float],
    *,
    sink_blocks: int,
    recent_blocks: int,
    drop_tail: bool,
) -> dict[int, float]:
    keys = [k for k in dist if k >= 0]
    recent_cut: set[int] = set()
    if recent_blocks > 0 and keys:
        recent_cut = set(sorted(keys)[-recent_blocks:])
    out: dict[int, float] = {}
    for k, v in dist.items():
        if k == -1 and drop_tail:
            continue
        if k >= 0 and k < sink_blocks:
            continue
        if k in recent_cut:
            continue
        out[k] = v
    return normalize(out)


def js_similarity(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)

    def kl(p: dict[int, float], m: dict[int, float]) -> float:
        s = 0.0
        for k in keys:
            pv = p.get(k, 0.0)
            if pv <= 0.0:
                continue
            mv = m.get(k, 0.0)
            if mv > 0.0:
                s += pv * math.log2(pv / mv)
        return s

    m = {k: 0.5 * (a.get(k, 0.0) + b.get(k, 0.0)) for k in keys}
    js = 0.5 * kl(a, m) + 0.5 * kl(b, m)
    return max(0.0, 1.0 - js)


def topk(dist: dict[int, float], k: int) -> set[int]:
    return {x for x, _ in sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:k]}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mean_dist(dists: Iterable[dict[int, float]]) -> dict[int, float]:
    acc: dict[int, float] = defaultdict(float)
    n = 0
    for d in dists:
        if not d:
            continue
        n += 1
        for k, v in d.items():
            acc[k] += v
    if n == 0:
        return {}
    return normalize({k: v / n for k, v in acc.items()})


def read_trace(path: Path) -> list[list[dict[int, float]]]:
    segments: list[list[dict[int, float]]] = []
    current_layers: dict[int, list[dict[int, float]]] = defaultdict(list)
    last_sample: int | None = None

    def flush_segment() -> None:
        if not current_layers:
            return
        samples: list[dict[int, float]] = []
        for sample in sorted(current_layers):
            samples.append(mean_dist(current_layers[sample]))
        if samples:
            segments.append(samples)
        current_layers.clear()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") not in {"attention_mass", "kvmem_attention_mass"}:
                continue
            sample = int(row["sample"])
            if last_sample is not None and sample < last_sample:
                flush_segment()
            last_sample = sample
            block_ids = [int(x) for x in row["block_ids"]]
            mass = [float(x) for x in row["mass"]]
            current_layers[sample].append(normalize(dict(zip(block_ids, mass))))
    flush_segment()
    return segments


def compute_curves(
    samples: list[dict[int, float]],
    *,
    top_k: int,
    observe_samples: int,
    sink_blocks: int,
    recent_blocks: int,
    drop_tail: bool,
) -> dict[str, object]:
    filt = [
        filtered_dist(
            d,
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
            drop_tail=drop_tail,
        )
        for d in samples
    ]
    if not filt:
        return {}
    ref_start = mean_dist(filt[: min(4, len(filt))])
    ref_obs = mean_dist(filt[: min(observe_samples, len(filt))])
    top_start = topk(ref_start, top_k)
    top_obs = topk(ref_obs, top_k)

    js_start: list[float] = []
    js_prev: list[float] = []
    jac_start: list[float] = []
    obs_mass: list[float] = []
    prev: dict[int, float] | None = None
    for d in filt:
        js_start.append(js_similarity(d, ref_start))
        js_prev.append(js_similarity(d, prev) if prev is not None else 1.0)
        jac_start.append(jaccard(topk(d, top_k), top_start))
        obs_mass.append(sum(d.get(k, 0.0) for k in top_obs))
        prev = d

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    late = obs_mass[min(observe_samples, len(obs_mass)) :]
    return {
        "samples": len(samples),
        "js_start": js_start,
        "js_prev": js_prev,
        "topk_jaccard_start": jac_start,
        "observe_topk_mass": obs_mass,
        "summary": {
            "js_start_mean": avg(js_start),
            "js_prev_mean": avg(js_prev),
            "topk_jaccard_start_mean": avg(jac_start),
            "observe_topk_mass_late_mean": avg(late) if late else avg(obs_mass),
        },
        "filtered_samples": filt,
    }


def svg_polyline(values: list[float], width: int, height: int, color: str) -> str:
    if not values:
        return ""
    pad = 32
    inner_w = max(1, width - 2 * pad)
    inner_h = max(1, height - 2 * pad)
    pts = []
    for i, v in enumerate(values):
        x = pad + (inner_w * i / max(1, len(values) - 1))
        y = pad + inner_h * (1.0 - max(0.0, min(1.0, v)))
        pts.append(f"{x:.1f},{y:.1f}")
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}" />'


def write_curve_svg(path: Path, curves: dict[str, list[float]]) -> None:
    width, height = 900, 360
    colors = {
        "js_start": "#2563eb",
        "js_prev": "#16a34a",
        "topk_jaccard_start": "#dc2626",
        "observe_topk_mass": "#9333ea",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<line x1="32" y1="328" x2="868" y2="328" stroke="#888"/>',
        '<line x1="32" y1="32" x2="32" y2="328" stroke="#888"/>',
    ]
    y = 18
    for name, values in curves.items():
        parts.append(svg_polyline(values, width, height, colors.get(name, "black")))
        parts.append(
            f'<text x="52" y="{y}" font-size="12" fill="{colors.get(name, "black")}">{name}</text>'
        )
        y += 16
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    r = int(255 * value)
    g = int(255 * (1.0 - 0.65 * value))
    b = int(255 * (1.0 - value))
    return f"#{r:02x}{g:02x}{b:02x}"


def write_heatmap_svg(path: Path, samples: list[dict[int, float]], top_n: int) -> None:
    totals: dict[int, float] = defaultdict(float)
    for d in samples:
        for k, v in d.items():
            totals[k] += v
    blocks = [k for k, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    cell_w, cell_h = 6, 10
    left, top = 80, 24
    width = left + max(1, len(samples)) * cell_w + 20
    height = top + max(1, len(blocks)) * cell_h + 30
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for y, block in enumerate(blocks):
        parts.append(f'<text x="4" y="{top + y * cell_h + 8}" font-size="9">blk {block}</text>')
        for x, d in enumerate(samples):
            v = d.get(block, 0.0)
            parts.append(
                f'<rect x="{left + x * cell_w}" y="{top + y * cell_h}" '
                f'width="{cell_w}" height="{cell_h}" fill="{color(v)}"/>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark/results/kvmem_attention_stability"))
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--observe-samples", type=int, default=128)
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-blocks", type=int, default=0)
    ap.add_argument("--heatmap-top-n", type=int, default=64)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    segments = read_trace(args.trace)
    report: dict[str, object] = {"trace": str(args.trace), "segments": []}
    md = ["# Attention Stability", "", f"- trace: `{args.trace}`", ""]

    for i, samples in enumerate(segments):
        variants = {
            "all": dict(sink_blocks=0, recent_blocks=0, drop_tail=False),
            "no_sink": dict(sink_blocks=args.sink_blocks, recent_blocks=0, drop_tail=False),
            "no_sink_no_recent": dict(
                sink_blocks=args.sink_blocks,
                recent_blocks=args.recent_blocks,
                drop_tail=True,
            ),
        }
        seg_out: dict[str, object] = {"segment": i, "samples": len(samples), "variants": {}}
        md.extend([f"## Segment {i}", "", f"- samples: `{len(samples)}`", ""])
        for name, opts in variants.items():
            curves = compute_curves(
                samples,
                top_k=args.top_k,
                observe_samples=args.observe_samples,
                **opts,
            )
            if not curves:
                continue
            filtered = curves.pop("filtered_samples")
            seg_out["variants"][name] = curves
            summary = curves["summary"]
            md.append(f"### {name}")
            for k, v in summary.items():
                md.append(f"- {k}: `{v:.4f}`")
            md.append("")
            write_curve_svg(
                args.out_dir / f"segment{i}_{name}_curves.svg",
                {
                    "js_start": curves["js_start"],
                    "js_prev": curves["js_prev"],
                    "topk_jaccard_start": curves["topk_jaccard_start"],
                    "observe_topk_mass": curves["observe_topk_mass"],
                },
            )
            write_heatmap_svg(
                args.out_dir / f"segment{i}_{name}_heatmap.svg",
                filtered,
                args.heatmap_top_n,
            )
        report["segments"].append(seg_out)

    (args.out_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
