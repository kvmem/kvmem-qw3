#!/usr/bin/env python3
"""Build a like-for-like MemoryAgentBench method table for a token-length phase."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from typing import Any


CATEGORIES = {
    "eventqa_": "Accurate_Retrieval",
    "ruler_": "Accurate_Retrieval",
    "longmemeval_": "Accurate_Retrieval",
    "factconsolidation_": "Conflict_Resolution",
    "detective_": "Long_Range_Understanding",
    "infbench_": "Long_Range_Understanding",
    "icl_": "Test_Time_Learning",
    "recsys_": "Test_Time_Learning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvmem-results", type=Path, required=True)
    parser.add_argument(
        "--method", action="append", nargs=2, metavar=("NAME", "RESULTS"),
        required=True,
    )
    parser.add_argument("--min-context-tokens-exclusive", type=int)
    parser.add_argument("--max-context-tokens-inclusive", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def category(source: str) -> str:
    for prefix, name in CATEGORIES.items():
        if source.startswith(prefix):
            return name
    raise ValueError(source)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_identities(args: argparse.Namespace) -> set[tuple[str, int]]:
    selected = set()
    for path in sorted((args.kvmem_results / "rows").glob("*/row_summary.json")):
        row = read_json(path)
        tokens = int(row["archive_tokens_unpadded"])
        if (
            args.min_context_tokens_exclusive is not None
            and tokens <= args.min_context_tokens_exclusive
        ):
            continue
        if (
            args.max_context_tokens_inclusive is not None
            and tokens > args.max_context_tokens_inclusive
        ):
            continue
        selected.add((str(row["split"]), int(row["dataset_row"])))
    return selected


def headline_metric(source: str) -> str:
    if source.startswith(("eventqa_", "ruler_", "factconsolidation_")):
        return "substring_exact_match"
    if source.startswith(("detective_", "icl_")):
        return "exact_match"
    if source.startswith("recsys_"):
        return "recsys_recall@5"
    if source.startswith("longmemeval_"):
        return "llm_judge_correct"
    if source.startswith("infbench_"):
        return "gpt4_summary_f1"
    raise ValueError(source)


def question_key(record: dict[str, Any]) -> str:
    return (
        f"{record['split']}|{int(record['dataset_row'])}|"
        f"{int(record['question_index'])}"
    )


def selected_question_keys(
    kvmem_root: Path, identities: set[tuple[str, int]],
) -> set[str]:
    result = set()
    for path in sorted((kvmem_root / "rows").glob("*/results.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            identity = (str(record["split"]), int(record["dataset_row"]))
            if identity in identities:
                result.add(question_key(record))
    return result


def special_scores(root: Path) -> dict[str, float]:
    path = root / "special_judgments.jsonl"
    if not path.is_file():
        return {}
    return {
        str(row["key"]): float(row["score"])
        for row in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def summarize_method(
    root: Path,
    identities: set[tuple[str, int]],
    expected_keys: set[str],
) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    context_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    all_values: list[float] = []
    metrics: dict[str, str] = {}
    found_keys: set[str] = set()
    judged = special_scores(root)
    for path in sorted((root / "rows").glob("*/results.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            identity = (str(row["split"]), int(row["dataset_row"]))
            if identity not in identities:
                continue
            key = question_key(row)
            if key in found_keys:
                raise RuntimeError(f"{root}: duplicate selected question {key}")
            found_keys.add(key)
            source = str(row["source"])
            metric = headline_metric(source)
            metrics[source] = metric
            if metric in ("llm_judge_correct", "gpt4_summary_f1"):
                if key not in judged:
                    raise RuntimeError(f"{root}: missing API judgment for {key}")
                score = judged[key]
            else:
                official = row.get("official_local_metrics") or {}
                if metric not in official:
                    raise RuntimeError(
                        f"{root}: missing official metric {metric} for {key}"
                    )
                score = float(official[metric])
            values[source].append(score)
            context_values[identity].append(score)
            all_values.append(score)
    missing = expected_keys - found_keys
    extra = found_keys - expected_keys
    if missing or extra:
        raise RuntimeError(
            f"{root}: selected question mismatch missing={len(missing)} "
            f"extra={len(extra)}"
        )
    sources = {}
    for source, scores in sorted(values.items()):
        sources[source] = {
            "questions": len(scores),
            "headline_metric": metrics[source],
            "headline_score": statistics.fmean(scores),
        }
    category_rows: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for source, item in sources.items():
        if item["headline_score"] is not None:
            category_rows[category(source)].append(
                (float(item["headline_score"]), int(item["questions"]))
            )
    categories = {}
    for name, rows in sorted(category_rows.items()):
        questions = sum(count for _, count in rows)
        categories[name] = {
            "sources": len(rows),
            "questions": questions,
            "macro_source_mean": statistics.fmean(score for score, _ in rows),
            "question_weighted_mean": sum(score * count for score, count in rows) / questions,
        }
    return {
        "results": str(root.resolve()),
        "contexts": len(identities),
        "questions": len(found_keys),
        "by_source": sources,
        "by_category": categories,
        # MemoryAgentBench contexts contain very different numbers of
        # questions.  Publish both common aggregations explicitly: the first
        # gives every immutable context equal weight, while the second gives
        # every question equal weight.  Scores remain the source-specific
        # official headline metrics normalized to [0, 1].
        "context_macro_mean": statistics.fmean(
            statistics.fmean(scores) for scores in context_values.values()
        ),
        "question_weighted_mean": statistics.fmean(all_values),
        "overall_macro_category_mean": (
            statistics.fmean(row["macro_source_mean"] for row in categories.values())
            if categories else None
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    methods = list(summary["methods"])
    categories = sorted({
        category
        for result in summary["methods"].values()
        for category in result["by_category"]
    })
    lines = [
        "# MemoryAgentBench method comparison",
        "",
        f"Selected contexts: {summary['selected_contexts']}",
        f"Selected questions: {summary['selected_questions']}",
        "",
        "## Aggregation summary",
        "",
        "| Aggregation | " + " | ".join(methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for label, key in (
        ("Context-macro mean", "context_macro_mean"),
        ("Question-weighted mean", "question_weighted_mean"),
        ("Macro-category mean", "overall_macro_category_mean"),
    ):
        cells = []
        for method in methods:
            value = summary["methods"][method].get(key)
            cells.append("NA" if value is None else f"{100 * value:.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "## Category summary",
        "",
        "| Category | " + " | ".join(methods) + " |",
        "|---|" + "---:|" * len(methods),
    ])
    for name in categories:
        cells = []
        for method in methods:
            value = summary["methods"][method]["by_category"].get(name, {}).get(
                "macro_source_mean"
            )
            cells.append("NA" if value is None else f"{100 * value:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    overall = []
    for method in methods:
        value = summary["methods"][method].get("overall_macro_category_mean")
        overall.append("NA" if value is None else f"{100 * value:.2f}")
    lines.append("| Overall macro-category mean | " + " | ".join(overall) + " |")

    sources = sorted({
        source
        for result in summary["methods"].values()
        for source in result["by_source"]
    })
    lines.extend([
        "",
        "## Dataset details",
        "",
        "| Source | Metric | Questions | " + " | ".join(methods) + " |",
        "|---|---|---:|" + "---:|" * len(methods),
    ])
    for source in sources:
        reference = next(
            result["by_source"][source]
            for result in summary["methods"].values()
            if source in result["by_source"]
        )
        cells = []
        for method in methods:
            item = summary["methods"][method]["by_source"].get(source)
            value = item.get("headline_score") if item is not None else None
            cells.append("NA" if value is None else f"{100 * value:.2f}")
        lines.append(
            f"| {source} | {reference['headline_metric']} | "
            f"{reference['questions']} | " + " | ".join(cells) + " |"
        )
    lines.extend([
        "",
        "Scores are percentages using each source's official headline metric. "
        "Category cells are macro means over sources, not pooled-question accuracy.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    identities = selected_identities(args)
    expected_keys = selected_question_keys(args.kvmem_results, identities)
    methods = {
        "KVMem": summarize_method(
            args.kvmem_results, identities, expected_keys
        )
    }
    for name, path in args.method:
        methods[name] = summarize_method(Path(path), identities, expected_keys)
    output = {
        "selection": {
            "min_context_tokens_exclusive": args.min_context_tokens_exclusive,
            "max_context_tokens_inclusive": args.max_context_tokens_inclusive,
        },
        "selected_contexts": len(identities),
        "selected_questions": len(expected_keys),
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(output), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
