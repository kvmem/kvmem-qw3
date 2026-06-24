#!/usr/bin/env python3
"""Plot adjacent sliding-window KL divergence over attention traces.

The KL statistic is computed over the execution-time sequence of decode
windows. Step boundaries are only annotations on the plot, not inputs to the
statistic.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def normalize(dist: dict[int, float]) -> dict[int, float]:
    total = sum(v for v in dist.values() if v > 0.0)
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in dist.items() if v > 0.0}


def mean_dist(dists: list[dict[int, float]]) -> dict[int, float]:
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


def kl_divergence(p: dict[int, float], q: dict[int, float], eps: float) -> float:
    keys = set(p) | set(q)
    z_p = 1.0 + eps * len(keys)
    z_q = 1.0 + eps * len(keys)
    out = 0.0
    for k in keys:
        pv = (p.get(k, 0.0) + eps) / z_p
        qv = (q.get(k, 0.0) + eps) / z_q
        out += pv * math.log2(pv / qv)
    return out


def read_trace(paths: list[Path]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    current_layers: dict[int, list[dict[int, float]]] = defaultdict(list)
    current_seq_lens: dict[int, list[int]] = defaultdict(list)
    current_block_tokens: int | None = None
    last_sample: int | None = None

    def flush() -> None:
        nonlocal current_block_tokens
        if not current_layers:
            return
        samples: list[dict[int, float]] = []
        seq_lens: list[int] = []
        for sample in sorted(current_layers):
            samples.append(mean_dist(current_layers[sample]))
            seq_lens.append(min(current_seq_lens[sample]))
        if samples:
            segments.append(
                {
                    "samples": samples,
                    "seq_lens": seq_lens,
                    "block_tokens": current_block_tokens or 128,
                }
            )
        current_layers.clear()
        current_seq_lens.clear()
        current_block_tokens = None

    for path in paths:
        flush()
        last_sample = None
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
                    flush()
                last_sample = sample
                current_block_tokens = int(row.get("block_tokens", current_block_tokens or 128))
                block_ids = [int(x) for x in row["block_ids"]]
                mass = [float(x) for x in row["mass"]]
                current_layers[sample].append(normalize(dict(zip(block_ids, mass))))
                current_seq_lens[sample].append(int(row["seq_len"]))
        flush()
    return segments


def context_filter(
    dist: dict[int, float],
    *,
    block_tokens: int,
    prefill_len: int,
    sink_blocks: int,
    recent_prefill_blocks: int,
) -> dict[int, float]:
    prefill_blocks = max(0, (prefill_len + block_tokens - 1) // block_tokens)
    recent_cut = max(sink_blocks, prefill_blocks - recent_prefill_blocks)
    out: dict[int, float] = {}
    for block, value in dist.items():
        if block < sink_blocks:
            continue
        if block < 0:
            continue
        # Keep only blocks that begin in the prefill context; generated suffix
        # blocks are not part of the context distribution being profiled.
        if block * block_tokens >= prefill_len:
            continue
        if recent_prefill_blocks > 0 and block >= recent_cut:
            continue
        out[block] = value
    return normalize(out)


def visible_filter(
    dist: dict[int, float],
    *,
    sink_blocks: int,
) -> dict[int, float]:
    out = {
        block: value
        for block, value in dist.items()
        if block >= sink_blocks
    }
    return normalize(out)


def window_starts(n: int, window: int, stride: int) -> list[int]:
    if n <= 0:
        return []
    if n <= window:
        return [0]
    starts = list(range(0, n - window + 1, stride))
    last = n - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_windows(
    segments: list[dict[str, object]],
    *,
    window: int,
    stride: int,
    min_samples: int,
    mode: str,
    sink_blocks: int,
    recent_prefill_blocks: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    kept_steps: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    for source_idx, segment in enumerate(segments):
        samples = segment["samples"]
        seq_lens = segment["seq_lens"]
        block_tokens = int(segment["block_tokens"])
        assert isinstance(samples, list)
        assert isinstance(seq_lens, list)
        if len(samples) < min_samples:
            continue
        prefill_len = int(min(seq_lens)) - 1
        if mode == "context_only":
            filtered = [
                context_filter(
                    d,
                    block_tokens=block_tokens,
                    prefill_len=prefill_len,
                    sink_blocks=sink_blocks,
                    recent_prefill_blocks=recent_prefill_blocks,
                )
                for d in samples
            ]
        elif mode == "full_decode":
            filtered = [
                visible_filter(d, sink_blocks=sink_blocks)
                for d in samples
            ]
        else:
            raise ValueError(f"unknown mode: {mode}")
        step_idx = len(kept_steps)
        starts = window_starts(len(filtered), window, stride)
        kept_steps.append(
            {
                "step": step_idx,
                "source_segment": source_idx,
                "samples": len(filtered),
                "prefill_len": prefill_len,
                "window_count": len(starts),
            }
        )
        for start in starts:
            end = min(len(filtered), start + window)
            windows.append(
                {
                    "step": step_idx,
                    "source_segment": source_idx,
                    "start": start,
                    "end": end,
                    "dist": mean_dist(filtered[start:end]),
                }
            )
    return kept_steps, windows


def write_svg(
    path: Path,
    transitions: list[dict[str, object]],
    steps: list[dict[str, object]],
    *,
    width: int = 1100,
    height: int = 430,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 72, 26, 36, 70
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    values = [float(t["kl"]) for t in transitions]
    max_v = max(values) if values else 1.0
    # Keep moderate within-step changes visible while still showing boundary
    # spikes. SVG labels report true values.
    y_max = max(max_v * 1.08, 1e-6)

    def x_at(i: int) -> float:
        return pad_l + inner_w * i / max(1, len(values) - 1)

    def y_at(v: float) -> float:
        return pad_t + inner_h * (1.0 - min(v, y_max) / y_max)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_l}" y="24" font-size="20" font-family="sans-serif">Sliding-window attention KL divergence</text>',
        f'<line x1="{pad_l + 390}" y1="18" x2="{pad_l + 430}" y2="18" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="5,4"/>',
        f'<text x="{pad_l + 438}" y="23" font-size="15" font-family="sans-serif" fill="#991b1b">red dashed line = step switch</text>',
        f'<line x1="{pad_l}" y1="{pad_t + inner_h}" x2="{pad_l + inner_w}" y2="{pad_t + inner_h}" stroke="#555"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + inner_h}" stroke="#555"/>',
        f'<text x="10" y="{pad_t + 14}" font-size="14" font-family="sans-serif">KL bits</text>',
        f'<text x="{width // 2 - 108}" y="{height - 16}" font-size="15" font-family="sans-serif">sliding-window transition index</text>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = pad_t + inner_h * (1.0 - frac)
        val = y_max * frac
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + inner_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="24" y="{y + 5:.1f}" font-size="12" font-family="monospace">{val:.3f}</text>')

    boundary_indices = [i for i, t in enumerate(transitions) if t["kind"] == "boundary"]
    for i in boundary_indices:
        x = x_at(i)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
            f'y2="{pad_t + inner_h}" stroke="#dc2626" stroke-width="1.5" '
            f'stroke-dasharray="5,4"/>'
        )
    if values:
        pts = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts}"/>')

    for i, trans in enumerate(transitions):
        x = x_at(i)
        y = y_at(float(trans["kl"]))
        if trans["kind"] == "boundary":
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#dc2626"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#2563eb"/>')

    step_ranges: list[tuple[int, int, int]] = []
    start = 0
    for step_idx, boundary_idx in enumerate(boundary_indices):
        step_ranges.append((step_idx, start, boundary_idx - 1))
        start = boundary_idx + 1
    if transitions:
        step_ranges.append((len(step_ranges), start, len(transitions) - 1))
    step_labels: list[tuple[float, str]] = []
    for step_idx, start_i, end_i in step_ranges:
        if start_i > end_i or start_i >= len(transitions):
            continue
        x1 = x_at(start_i)
        x2 = x_at(min(end_i, len(transitions) - 1))
        cx = 0.5 * (x1 + x2)
        if end_i - start_i >= 2:
            step_labels.append((cx, f"Step{step_idx}"))

    last_right = -math.inf
    for cx, label in step_labels:
        # Approximate text width for sans-serif 13px. Keep a small gap so dense
        # step switches do not make the bottom annotation unreadable.
        label_width = 7.2 * len(label)
        left = cx - 0.5 * label_width
        right = cx + 0.5 * label_width
        if left < last_right + 8:
            continue
        last_right = right
        parts.append(
            f'<text x="{left:.1f}" y="{height - 42}" '
            f'font-size="13" font-family="sans-serif" fill="#6b7280">{label}</text>'
        )

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark/results/attention_step_kl"))
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--min-samples", type=int, default=64)
    ap.add_argument(
        "--mode",
        choices=["full_decode", "context_only"],
        default="full_decode",
        help=(
            "full_decode compares attention over all visible KV blocks; "
            "context_only keeps only prefill context blocks."
        ),
    )
    ap.add_argument("--sink-blocks", type=int, default=1)
    ap.add_argument("--recent-prefill-blocks", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-9)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    segments = read_trace(args.trace)
    steps, windows = build_windows(
        segments,
        window=args.window,
        stride=args.stride,
        min_samples=args.min_samples,
        mode=args.mode,
        sink_blocks=args.sink_blocks,
        recent_prefill_blocks=args.recent_prefill_blocks,
    )
    transitions: list[dict[str, object]] = []
    for i in range(1, len(windows)):
        prev = windows[i - 1]
        cur = windows[i]
        kind = "within" if prev["step"] == cur["step"] else "boundary"
        transitions.append(
            {
                "index": i - 1,
                "kind": kind,
                "from_step": prev["step"],
                "to_step": cur["step"],
                "from_source_segment": prev["source_segment"],
                "to_source_segment": cur["source_segment"],
                "from_window": [prev["start"], prev["end"]],
                "to_window": [cur["start"], cur["end"]],
                "kl": kl_divergence(prev["dist"], cur["dist"], args.eps),
            }
        )

    within = [float(t["kl"]) for t in transitions if t["kind"] == "within"]
    boundary = [float(t["kl"]) for t in transitions if t["kind"] == "boundary"]
    summary = {
        "trace": [str(path) for path in args.trace],
        "window": args.window,
        "stride": args.stride,
        "mode": args.mode,
        "min_samples": args.min_samples,
        "sink_blocks": args.sink_blocks,
        "recent_prefill_blocks": args.recent_prefill_blocks,
        "source_segments": len(segments),
        "kept_steps": steps,
        "windows": len(windows),
        "transitions": len(transitions),
        "within_mean_kl": sum(within) / len(within) if within else 0.0,
        "boundary_mean_kl": sum(boundary) / len(boundary) if boundary else 0.0,
        "boundary_over_within": (
            (sum(boundary) / len(boundary)) / (sum(within) / len(within))
            if within and boundary and sum(within) > 0.0
            else 0.0
        ),
        "transition_rows": transitions,
    }
    (args.out_dir / "sliding_kl_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_svg(args.out_dir / "sliding_kl_curve.svg", transitions, steps)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
