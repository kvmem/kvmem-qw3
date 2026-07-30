#!/usr/bin/env python3
"""Summarize the cached / fresh-KV / fresh-KV-and-state attribution arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import run_agentlongbench_kvmem as runner


ARMS = ("cached", "kv_only", "kv_and_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def latest_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["stable_sample_id"]): row
        for row in runner.read_jsonl(path)
        if row.get("stable_sample_id")
    }


def load_latest_dump(path: Path) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for row in runner.read_jsonl(path):
        if row.get("type") == "meta":
            if current is not None:
                snapshots[str(current["meta"]["trace_tag"])] = current
            current = {"meta": row, "blocks": []}
        elif current is None:
            raise RuntimeError(f"{path} begins with a non-meta row")
        else:
            current["blocks"].append(row)
    if current is not None:
        snapshots[str(current["meta"]["trace_tag"])] = current
    for sid, snapshot in snapshots.items():
        expected = int(snapshot["meta"].get("block_count") or -1)
        if len(snapshot["blocks"]) != expected:
            raise RuntimeError(
                f"incomplete retrieval snapshot {path}:{sid}: "
                f"{len(snapshot['blocks'])}/{expected}"
            )
    return snapshots


def selection_fingerprint(snapshot: dict[str, Any]) -> dict[str, Any]:
    selected = sorted(
        (
            row
            for row in snapshot["blocks"]
            if int(row.get("sel") or 0) == 1
        ),
        key=lambda row: int(row["b"]),
    )
    digest = hashlib.sha256()
    ids: list[int] = []
    source_tokens = 0
    for row in selected:
        block_id = int(row["b"])
        p0 = int(row["p0"])
        nt = int(row["nt"])
        tokens = [int(value) for value in row.get("tok") or []]
        if len(tokens) != nt:
            raise RuntimeError(
                f"selected block {block_id} has {len(tokens)}/{nt} token ids"
            )
        ids.append(block_id)
        source_tokens += nt
        digest.update(struct.pack("<III", block_id, p0, nt))
        for token in tokens:
            digest.update(struct.pack("<I", token))
    return {
        "selected_blocks": len(selected),
        "selected_source_tokens": source_tokens,
        "selected_block_ids": ids,
        "selected_source_sha256": digest.hexdigest(),
    }


def classify(scores: dict[str, bool]) -> str:
    cached = scores["cached"]
    kv_only = scores["kv_only"]
    kv_and_state = scores["kv_and_state"]
    if cached:
        if kv_only and kv_and_state:
            return "cached_recovers_at_temperature_0"
        if not kv_only and not kv_and_state:
            return "refresh_regression_after_cached_recovery"
        if kv_only and not kv_and_state:
            return "fresh_recurrent_state_regression"
        return "historical_recurrent_state_regression"
    if kv_only and kv_and_state:
        return "normal_attention_kv_representation_dominant"
    if not kv_only and kv_and_state:
        return "recurrent_state_or_kv_state_interaction_dominant"
    if kv_only and not kv_and_state:
        return "historical_recurrent_state_helpful"
    return "not_resolved_by_compact_refresh"


def compact(value: Any) -> str:
    if value is None:
        return "-"
    rendered = str(value).replace("\n", " ")
    return rendered if len(rendered) <= 36 else rendered[:33] + "..."


def main() -> None:
    args = parse_args()
    ids = read_ids(args.ids_file)
    if len(ids) != 4:
        raise RuntimeError(f"expected four IDs, got {len(ids)}")

    evals: dict[str, dict[str, dict[str, Any]]] = {}
    answers: dict[str, dict[str, dict[str, Any]]] = {}
    dumps: dict[str, dict[str, dict[str, Any]]] = {}
    fingerprints: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ARMS:
        arm_root = args.root / arm
        evals[arm] = latest_by_id(arm_root / "eval.jsonl")
        answers[arm] = latest_by_id(arm_root / "answers.jsonl")
        dumps[arm] = load_latest_dump(arm_root / "retrieval_scores.jsonl")
        fingerprints[arm] = {
            sid: selection_fingerprint(dumps[arm][sid]) for sid in ids
        }

    rows: list[dict[str, Any]] = []
    all_selection_equal = True
    for sid in ids:
        missing = [
            f"{arm}:{kind}"
            for arm in ARMS
            for kind, table in (
                ("eval", evals[arm]),
                ("answer", answers[arm]),
                ("dump", dumps[arm]),
            )
            if sid not in table
        ]
        if missing:
            raise RuntimeError(f"{sid} is missing {', '.join(missing)}")
        baseline = fingerprints["cached"][sid]
        selection_equal = {
            arm: fingerprints[arm][sid]["selected_source_sha256"]
            == baseline["selected_source_sha256"]
            for arm in ARMS
        }
        all_selection_equal &= all(selection_equal.values())
        correct = {
            arm: evals[arm][sid].get("correct") is True for arm in ARMS
        }
        row = {
            "stable_sample_id": sid,
            "task_type": evals["cached"][sid].get("task_type"),
            "reference": evals["cached"][sid].get("reference"),
            "arms": {
                arm: {
                    "correct": correct[arm],
                    "score": evals[arm][sid].get("score"),
                    "prediction": evals[arm][sid].get("prediction"),
                    "finish_reason": answers[arm][sid].get("finish_reason"),
                    "elapsed_sec": (
                        answers[arm][sid].get("timing") or {}
                    ).get("total_sec"),
                    "selected_blocks": fingerprints[arm][sid][
                        "selected_blocks"
                    ],
                    "selected_source_tokens": fingerprints[arm][sid][
                        "selected_source_tokens"
                    ],
                    "selected_source_sha256": fingerprints[arm][sid][
                        "selected_source_sha256"
                    ],
                    "selection_matches_cached": selection_equal[arm],
                }
                for arm in ARMS
            },
            "selection_equal_across_arms": all(selection_equal.values()),
            "attribution": classify(correct),
        }
        rows.append(row)

    summary = {
        "schema_version": "agentlongbench_inline_refresh_attribution.v1",
        "samples": len(rows),
        "temperature": 0.0,
        "arms": {
            "cached": "historical normal-attention KV + historical recurrent state",
            "kv_only": "fresh selected normal-attention KV + restored historical recurrent state",
            "kv_and_state": "fresh selected normal-attention KV + fresh recurrent state",
        },
        "selection_equal_across_all_arms": all_selection_equal,
        "accuracy": {
            arm: {
                "correct": sum(
                    row["arms"][arm]["correct"] for row in rows
                ),
                "total": len(rows),
            }
            for arm in ARMS
        },
        "per_sample": rows,
    }
    runner.write_json(args.root / "attribution_summary.json", summary)

    lines = [
        "# AgentLongBench 1M inline-refresh three-arm attribution",
        "",
        f"- Selection identical across all arms: `{all_selection_equal}`",
        *[
            f"- `{arm}`: {summary['accuracy'][arm]['correct']}/{len(rows)}"
            for arm in ARMS
        ],
        "",
        "| Sample | Task | Ref | cached | kv_only | kv+state | Selection | Attribution |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        cells = []
        for arm in ARMS:
            result = row["arms"][arm]
            cells.append(
                f"{compact(result['prediction'])} "
                f"({'✓' if result['correct'] else '✗'})"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["stable_sample_id"][:12],
                    compact(row["task_type"]),
                    compact(row["reference"]),
                    *cells,
                    "same" if row["selection_equal_across_arms"] else "MISMATCH",
                    row["attribution"],
                ]
            )
            + " |"
        )
    (args.root / "attribution_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["accuracy"], ensure_ascii=False))
    print(f"[selection-equal] {all_selection_equal}")
    print(f"[summary] {args.root / 'attribution_summary.json'}")
    if not all_selection_equal:
        raise SystemExit(
            "selected source-token sets differ across arms; attribution is invalid"
        )


if __name__ == "__main__":
    main()
