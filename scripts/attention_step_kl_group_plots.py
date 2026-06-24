#!/usr/bin/env python3
"""Draw one sliding-window KL curve per source-segment group."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
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


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "group"


def transitions_for_group(
    summary: dict[str, object],
    group: dict[str, object],
) -> list[dict[str, object]]:
    start = int(group["start"])
    end = int(group["end"])
    rows: list[dict[str, object]] = []
    for transition in summary["transition_rows"]:
        from_seg = int(transition["from_source_segment"])
        to_seg = int(transition["to_source_segment"])
        if start <= from_seg <= end and start <= to_seg <= end:
            rows.append(transition)
    return rows


def write_svg(
    path: Path,
    title: str,
    transitions: list[dict[str, object]],
    *,
    width: int = 1100,
    height: int = 430,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 72, 26, 42, 70
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    values = [float(t["kl"]) for t in transitions]
    max_v = max(values) if values else 1.0
    y_max = max(max_v * 1.08, 1e-6)

    def x_at(i: int) -> float:
        return pad_l + inner_w * i / max(1, len(values) - 1)

    def y_at(v: float) -> float:
        return pad_t + inner_h * (1.0 - min(v, y_max) / y_max)

    title_text = html.escape(title)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_l}" y="24" font-size="20" font-family="sans-serif">{title_text}</text>',
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

    boundary_indices = [
        i for i, transition in enumerate(transitions)
        if transition["kind"] == "boundary"
    ]
    for i in boundary_indices:
        x = x_at(i)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" '
            f'y2="{pad_t + inner_h}" stroke="#dc2626" stroke-width="1.5" '
            f'stroke-dasharray="5,4"/>'
        )

    if values:
        pts = " ".join(f"{x_at(i):.1f},{y_at(value):.1f}" for i, value in enumerate(values))
        parts.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts}"/>')

    for i, transition in enumerate(transitions):
        x = x_at(i)
        y = y_at(float(transition["kl"]))
        if transition["kind"] == "boundary":
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#dc2626"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.2" fill="#2563eb"/>')

    step_ranges: list[tuple[int, int, int]] = []
    start_i = 0
    for local_step, boundary_idx in enumerate(boundary_indices):
        step_ranges.append((local_step, start_i, boundary_idx - 1))
        start_i = boundary_idx + 1
    if transitions:
        step_ranges.append((len(step_ranges), start_i, len(transitions) - 1))

    step_labels: list[tuple[float, str]] = []
    for local_step, left_i, right_i in step_ranges:
        if left_i > right_i or left_i >= len(transitions):
            continue
        x1 = x_at(left_i)
        x2 = x_at(min(right_i, len(transitions) - 1))
        if right_i - left_i >= 2:
            step_labels.append((0.5 * (x1 + x2), f"Step{local_step}"))

    last_right = -math.inf
    for cx, label in step_labels:
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


def write_index(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Per-Instance Sliding KL Figures",
        "",
        "| instance | figure | transitions | boundary transitions |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | [{file}]({file}) | {transitions} | {boundary} |".format(
                name=row["name"],
                file=row["file"],
                transitions=row["transitions"],
                boundary=row["boundary_transitions"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--group", action="append", type=parse_group, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--title-prefix", default="Sliding-window attention KL")
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    for group in args.group:
        name = str(group["name"])
        transitions = transitions_for_group(summary, group)
        filename = f"{safe_filename(name)}.svg"
        write_svg(
            args.out_dir / filename,
            f"{args.title_prefix}: {name}",
            transitions,
        )
        index_rows.append(
            {
                "name": name,
                "file": filename,
                "transitions": len(transitions),
                "boundary_transitions": sum(
                    1 for transition in transitions
                    if transition["kind"] == "boundary"
                ),
            }
        )
    (args.out_dir / "index.json").write_text(
        json.dumps(index_rows, indent=2) + "\n", encoding="utf-8"
    )
    write_index(args.out_dir / "index.md", index_rows)
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
