#!/usr/bin/env python3
"""Apply MemoryAgentBench's official non-LLM post-processing to KVMem output.

LongMemEval's answer-equivalence judge and InfBench's three-call summary judge
remain explicitly pending here; they are API-based second-stage metrics. All
other benchmark metrics, including ReDial entity-name Recall@K, are computed by
the exact evaluator at the pinned official repository commit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any


PINNED_COMMIT = "455306dcabc3842526eb83cd4e225e5d486c5c5d"
FULL_QUESTION_COUNT = 3671


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument(
        "--official-repo", type=Path,
        default=Path("/home/chaidi/MemoryAgentBench-official"),
    )
    p.add_argument(
        "--entity-map", type=Path,
        default=Path(
            "/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/"
            "MemoryAgentBench/entity2id.json"
        ),
    )
    p.add_argument(
        "--allow-partial", action="store_true",
        help="permit diagnostic scoring before all 3,671 questions exist",
    )
    return p.parse_args()


def builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def headline_metric(source: str) -> str:
    if source.startswith("eventqa_"):
        return "substring_exact_match"
    if source.startswith("ruler_"):
        return "substring_exact_match"
    if source.startswith("longmemeval_"):
        return "llm_judge_correct"
    if source.startswith("factconsolidation_"):
        return "substring_exact_match"
    if source.startswith("detective_"):
        return "exact_match"
    if source.startswith("infbench_"):
        return "gpt4_summary_f1"
    if source.startswith("icl_"):
        return "exact_match"
    if source.startswith("recsys_"):
        return "recsys_recall@5"
    raise ValueError(source)


def main() -> int:
    args = parse_args()
    head = subprocess_head(args.official_repo)
    if head != PINNED_COMMIT:
        raise RuntimeError(
            f"official repository HEAD {head} != pinned {PINNED_COMMIT}"
        )
    recsys_dir = args.official_repo / "processed_data/Recsys_Redial"
    recsys_dir.mkdir(parents=True, exist_ok=True)
    target_map = recsys_dir / "entity2id.json"
    if not target_map.exists():
        try:
            target_map.symlink_to(args.entity_map.resolve())
        except OSError:
            shutil.copyfile(args.entity_map, target_map)

    sys.path.insert(0, str(args.official_repo))
    from utils.eval_other_utils import post_process  # type: ignore

    result_paths = sorted((args.results_dir / "rows").glob("*/results.jsonl"))
    record_keys: set[tuple[str, int, int]] = set()
    record_count = 0
    for path in result_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            record_count += 1
            identity = (
                str(record["split"]),
                int(record["dataset_row"]),
                int(record["question_index"]),
            )
            if identity in record_keys:
                raise RuntimeError(f"duplicate benchmark question: {identity}")
            record_keys.add(identity)
    if not args.allow_partial and record_count != FULL_QUESTION_COUNT:
        raise RuntimeError(
            f"refusing partial official score: found {record_count} unique "
            f"questions, expected {FULL_QUESTION_COUNT}; use --allow-partial "
            "only for diagnostics"
        )

    old_cwd = Path.cwd()
    os.chdir(args.official_repo)
    try:
        source_values: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        total = 0
        for results_path in result_paths:
            records = [
                json.loads(line) for line in results_path.read_text(
                    encoding="utf-8"
                ).splitlines() if line.strip()
            ]
            for record in records:
                output = {"output": record["answer"]}
                metrics, additional = post_process(
                    output,
                    record["gold_answer"],
                    {"sub_dataset": record["source"]},
                )
                metrics = builtin(metrics)
                record["official_local_metrics"] = metrics
                record["official_parsed_output"] = builtin(
                    additional.get("parsed_output")
                )
                for key, value in metrics.items():
                    if isinstance(value, (bool, int, float)):
                        source_values[record["source"]][key].append(float(value))
                total += 1
            temp = results_path.with_suffix(".jsonl.tmp")
            temp.write_text(
                "".join(
                    json.dumps(builtin(record), ensure_ascii=False) + "\n"
                    for record in records
                ),
                encoding="utf-8",
            )
            temp.replace(results_path)
    finally:
        os.chdir(old_cwd)

    by_source: dict[str, Any] = {}
    pending = []
    for source, metrics in sorted(source_values.items()):
        averaged = {
            key: sum(values) / len(values) for key, values in sorted(metrics.items())
        }
        headline = headline_metric(source)
        if headline not in averaged:
            pending.append({"source": source, "metric": headline})
        by_source[source] = {
            "questions": len(next(iter(metrics.values()))) if metrics else 0,
            "headline_metric": headline,
            "headline_score": averaged.get(headline),
            "metrics": averaged,
        }
    summary = {
        "official_repository": str(args.official_repo.resolve()),
        "official_repository_commit": head,
        "questions_scored": total,
        "by_source": by_source,
        "pending_api_judges": pending,
        "note": (
            "All available deterministic official post-processing is complete. "
            "LongMemEval LLM correctness and InfBench GPT-4 summary F1 require "
            "their separate API-based judge stage."
        ),
    }
    (args.results_dir / "official_local_summary.json").write_text(
        json.dumps(builtin(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(builtin(summary), ensure_ascii=False, indent=2))
    return 0


def subprocess_head(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


if __name__ == "__main__":
    raise SystemExit(main())
