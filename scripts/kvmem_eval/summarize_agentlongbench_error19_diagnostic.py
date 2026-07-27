#!/usr/bin/env python3
"""Join gold coverage, original KVMem, baseline, and selected-text outcomes."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import run_agentlongbench_kvmem as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--selected-text-eval", type=Path, required=True)
    parser.add_argument("--selected-text-answers", type=Path, required=True)
    parser.add_argument("--kvmem-eval", type=Path, required=True)
    parser.add_argument("--kvmem-answers", type=Path, required=True)
    parser.add_argument("--compact-only-eval", type=Path, required=True)
    parser.add_argument("--rag1024-eval", type=Path, required=True)
    parser.add_argument("--rag32-eval", type=Path, required=True)
    parser.add_argument("--sliding-eval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return runner.latest_by_id(runner.read_jsonl(path))


def correctness(row: dict[str, Any]) -> bool:
    return row.get("correct") is True


def main() -> None:
    args = parse_args()
    coverage = by_id(args.coverage)
    selected_eval = by_id(args.selected_text_eval)
    selected_answers = by_id(args.selected_text_answers)
    kvmem_eval = by_id(args.kvmem_eval)
    kvmem_answers = by_id(args.kvmem_answers)
    baselines = {
        "compact_only": by_id(args.compact_only_eval),
        "rag1024": by_id(args.rag1024_eval),
        "rag32": by_id(args.rag32_eval),
        "sliding32k": by_id(args.sliding_eval),
    }
    ids = list(coverage)
    required_maps = {
        "selected_text_eval": selected_eval,
        "selected_text_answers": selected_answers,
        "kvmem_eval": kvmem_eval,
        "kvmem_answers": kvmem_answers,
        **baselines,
    }
    for name, rows in required_maps.items():
        missing = [sid for sid in ids if sid not in rows]
        if missing:
            raise RuntimeError(f"{name} is missing {len(missing)} IDs; first={missing[0]}")

    output_rows: list[dict[str, Any]] = []
    for sid in ids:
        cov = coverage[sid]
        answer_cov = float(cov["answer_evidence"]["metrics"]["block_coverage"])
        verify_cov = float(cov["verification_scope"]["metrics"]["block_coverage"])
        replay_correct = correctness(selected_eval[sid])
        if answer_cov < 1.0:
            diagnosis = "direct_answer_evidence_not_fully_selected"
        elif replay_correct:
            diagnosis = "same_selected_text_succeeds_kv_path_differentiator"
        elif verify_cov < 1.0:
            diagnosis = "direct_evidence_present_global_verification_incomplete"
        else:
            diagnosis = "same_selected_text_also_fails_reasoning_or_noise"
        output_rows.append(
            {
                "delivery_index": cov["delivery_index"],
                "stable_sample_id": sid,
                "task_type": cov["task_type"],
                "question": cov["question"],
                "reference_answer": cov["reference_answer"],
                "evidence_policy": cov["evidence_policy"],
                "answer_evidence_block_coverage": answer_cov,
                "answer_evidence_token_coverage": cov["answer_evidence"]["metrics"][
                    "token_coverage"
                ],
                "verification_block_coverage": verify_cov,
                "verification_token_coverage": cov["verification_scope"]["metrics"][
                    "token_coverage"
                ],
                "kvmem_prediction": kvmem_eval[sid].get("prediction"),
                "kvmem_hypothesis": kvmem_answers[sid].get("hypothesis"),
                "kvmem_correct": correctness(kvmem_eval[sid]),
                "selected_text_prediction": selected_eval[sid].get("prediction"),
                "selected_text_hypothesis": selected_answers[sid].get("hypothesis"),
                "selected_text_correct": replay_correct,
                "baseline_correct": {
                    name: correctness(rows[sid]) for name, rows in baselines.items()
                },
                "primary_diagnosis": diagnosis,
            }
        )

    diagnosis_counts = collections.Counter(
        row["primary_diagnosis"] for row in output_rows
    )
    task_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in output_rows:
        task_rows[row["task_type"]].append(row)
    summary = {
        "schema_version": "agentlongbench_error19_diagnostic.v1",
        "samples": len(output_rows),
        "original_kvmem_correct": sum(row["kvmem_correct"] for row in output_rows),
        "selected_text_correct": sum(
            row["selected_text_correct"] for row in output_rows
        ),
        "selected_text_accuracy": (
            sum(row["selected_text_correct"] for row in output_rows)
            / len(output_rows)
        ),
        "answer_evidence_fully_covered": sum(
            row["answer_evidence_block_coverage"] == 1.0 for row in output_rows
        ),
        "verification_scope_fully_covered": sum(
            row["verification_block_coverage"] == 1.0 for row in output_rows
        ),
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "by_task": {
            task: {
                "samples": len(rows),
                "selected_text_correct": sum(
                    row["selected_text_correct"] for row in rows
                ),
                "mean_answer_evidence_block_coverage": sum(
                    row["answer_evidence_block_coverage"] for row in rows
                )
                / len(rows),
                "mean_verification_block_coverage": sum(
                    row["verification_block_coverage"] for row in rows
                )
                / len(rows),
            }
            for task, rows in sorted(task_rows.items())
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    runner.write_json(args.output_root / "diagnostic_summary.json", summary)
    with (args.output_root / "per_sample_diagnostic.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in output_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    lines = [
        "# AgentLongBench 1M KVMem error-19 diagnostic",
        "",
        f"- Samples: {summary['samples']}",
        f"- Selected-text correct: {summary['selected_text_correct']}/{summary['samples']} "
        f"({summary['selected_text_accuracy']:.2%})",
        f"- Full answer-evidence coverage: "
        f"{summary['answer_evidence_fully_covered']}/{summary['samples']}",
        f"- Full exact-verification coverage: "
        f"{summary['verification_scope_fully_covered']}/{summary['samples']}",
        "",
        "| # | Task | Answer block coverage | Verify block coverage | "
        "Text replay | Primary diagnosis |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in output_rows:
        lines.append(
            f"| {row['delivery_index']} | {row['task_type']} | "
            f"{row['answer_evidence_block_coverage']:.1%} | "
            f"{row['verification_block_coverage']:.1%} | "
            f"{'correct' if row['selected_text_correct'] else 'wrong'} | "
            f"{row['primary_diagnosis']} |"
        )
    (args.output_root / "diagnostic_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        f"[complete] selected_text={summary['selected_text_correct']}/"
        f"{summary['samples']} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
