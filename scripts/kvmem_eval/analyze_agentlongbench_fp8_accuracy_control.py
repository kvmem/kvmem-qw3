#!/usr/bin/env python3
"""Summarize the four-cell FP16/FP8 × prefill-chunk control experiment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CELLS = ("fp16_c2048", "fp16_c8192", "fp8_c2048", "fp8_c8192")
PAIRS = (
    ("fp16_c2048", "fp16_c8192", "chunk_effect_fp16"),
    ("fp8_c2048", "fp8_c8192", "chunk_effect_fp8"),
    ("fp16_c2048", "fp8_c2048", "dtype_effect_c2048"),
    ("fp16_c8192", "fp8_c8192", "dtype_effect_c8192"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root", type=Path, default=Path("/data/chaidi/kvmem_eval/results")
    )
    parser.add_argument(
        "--tag-prefix",
        default="agentlongbench_512k_fp8_accuracy_control_20260725",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_selected(path: Path) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {}
    trace_tag: str | None = None
    current: set[int] = set()

    def finish() -> None:
        nonlocal trace_tag, current
        if trace_tag is not None:
            selected[trace_tag] = current
        trace_tag = None
        current = set()

    if not path.exists():
        return selected
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("type") == "meta":
                finish()
                trace_tag = str(row["trace_tag"])
            elif trace_tag is not None and row.get("sel"):
                current.add(int(row["b"]))
    finish()
    return selected


def main() -> None:
    args = parse_args()
    cells: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        root = args.result_root / f"{args.tag_prefix}_{cell}"
        eval_rows = read_jsonl(root / "eval.jsonl")
        answer_rows = read_jsonl(root / "answers.jsonl")
        cells[cell] = {
            "root": str(root),
            "eval": {str(row["stable_sample_id"]): row for row in eval_rows},
            "answers": {
                str(row["stable_sample_id"]): row for row in answer_rows
            },
            "selected": load_selected(root / "kvmem_scores.jsonl"),
        }

    summary: dict[str, Any] = {"tag_prefix": args.tag_prefix, "cells": {}, "pairs": {}}
    for cell, data in cells.items():
        eval_map = data["eval"]
        answer_map = data["answers"]
        summary["cells"][cell] = {
            "evaluated": len(eval_map),
            "score": sum(float(row.get("score", 0.0)) for row in eval_map.values()),
            "finish_reasons": dict(
                Counter(row.get("finish_reason") for row in answer_map.values())
            ),
            "selection_dumps": len(data["selected"]),
        }

    for left, right, label in PAIRS:
        a = cells[left]
        b = cells[right]
        shared = sorted(set(a["eval"]) & set(b["eval"]))
        rows: list[dict[str, Any]] = []
        for sid in shared:
            sa = a["selected"].get(sid, set())
            sb = b["selected"].get(sid, set())
            inter = len(sa & sb)
            union = len(sa | sb)
            rows.append(
                {
                    "stable_sample_id": sid,
                    "left_score": float(a["eval"][sid].get("score", 0.0)),
                    "right_score": float(b["eval"][sid].get("score", 0.0)),
                    "left_finish": a["answers"][sid].get("finish_reason"),
                    "right_finish": b["answers"][sid].get("finish_reason"),
                    "selected_left": len(sa),
                    "selected_right": len(sb),
                    "selected_intersection": inter,
                    "selected_jaccard": inter / union if union else None,
                    "selected_left_recall": inter / len(sa) if sa else None,
                }
            )
        jaccards = [
            row["selected_jaccard"]
            for row in rows
            if row["selected_jaccard"] is not None
        ]
        summary["pairs"][label] = {
            "left": left,
            "right": right,
            "shared_samples": len(shared),
            "same_score": sum(row["left_score"] == row["right_score"] for row in rows),
            "left_wins": sum(row["left_score"] > row["right_score"] for row in rows),
            "right_wins": sum(row["right_score"] > row["left_score"] for row in rows),
            "mean_selected_jaccard": (
                sum(jaccards) / len(jaccards) if jaccards else None
            ),
            "samples": rows,
        }

    output = args.output or (
        args.result_root / f"{args.tag_prefix}_summary.json"
    )
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
