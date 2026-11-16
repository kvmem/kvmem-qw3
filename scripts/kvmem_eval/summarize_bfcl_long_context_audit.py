#!/usr/bin/env python3
"""Join analyst notes and summarize an existing BFCL reference-path audit.

Does not execute tools, run a model, or repeat tokenization. The TSV annotations
are a qualitative reading by the assistant, not official BFCL ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from pathlib import Path

from audit_bfcl_long_context import distribution, read_jsonl, write_csv


RELATIONS = {
    "linked_chain": "存在同一实体、前轮结果或参数的延续；不表示每一对相邻问题都依赖历史文本",
    "shared_scene": "主要共享车辆、账户等场景，后续问题通常自足；环境状态连续不等于文本记忆依赖",
    "mixed_workflow": "同一会话混有较独立任务和局部连续任务链",
    "single_turn": "只有一条用户消息，即使其中包含多个操作也只计一个 Query",
}
DOMAINS = ["file", "vehicle", "trading", "travel"]
BUDGETS = [4096, 8192, 16384, 32768, 65536, 131072]


def save_records(directory, name, rows):
    write_csv(directory / f"{name}.csv", rows)
    (directory / f"{name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root
    analysis = root / "analysis"
    conversations = read_jsonl(analysis / "conversations.jsonl")
    turns = read_jsonl(analysis / "turns.jsonl")
    pairs = read_jsonl(analysis / "adjacent_query_relations.jsonl")
    references = read_jsonl(analysis / "reference_trajectories.jsonl")
    with (root / "reviewed_relations.tsv").open() as f:
        review_rows = list(csv.DictReader(f, delimiter="\t"))
    reviews = {int(r["index"]): r for r in review_rows}
    assert len(reviews) == len(review_rows) == len(conversations) == 200
    assert set(reviews) == set(range(200))
    assert len({c["id"] for c in conversations}) == 200
    by_id = {c["id"]: [] for c in conversations}
    for turn in turns:
        by_id[turn["id"]].append(turn)
    assert len(turns) == sum(c["user_query_count"] for c in conversations) == 734
    assert len(pairs) == sum(c["user_query_count"] - 1 for c in conversations)
    assert all(not r["tool_errors"] for r in references)
    assert len(references) == 200
    for row in conversations:
        index = int(row["id"].rsplit("_", 1)[1])
        review = reviews[index]
        assert review["category"] in RELATIONS
        row.update(primary_domain=DOMAINS[index // 50],
                   reviewed_relation_category=review["category"],
                   reviewed_relation_note=review["review_note"])
        actual_turns = by_id[row["id"]]
        assert len(actual_turns) == row["user_query_count"]
        assert row["user_query_count"] == row["distinct_user_queries_nfc_whitespace"]
        for budget in BUDGETS:
            later = [t["turn_index"] for t in actual_turns
                     if t["turn_index"] > 0 and
                     t["reference_prompt_tokens_before_query_answer"] > budget]
            row[f"reference_later_user_queries_over_{budget}"] = len(later)
        row["later_query_indices_over_32768"] = [t["turn_index"] for t in actual_turns
            if t["turn_index"] > 0 and t["reference_prompt_tokens_before_query_answer"] > 32768]
    save_records(analysis, "conversations", conversations)
    pressure_candidates = [c for c in conversations if c["reference_later_user_queries_over_32768"] > 0]
    candidates = [c for c in pressure_candidates if c["distinct_user_queries_nfc_whitespace"] >= 4
                  and c["reference_later_user_queries_over_32768"] >= 2]
    save_records(analysis, "pressure_candidates_32k", pressure_candidates)
    save_records(analysis, "recommended_candidates_32k", candidates)
    summary = json.loads((analysis / "summary.json").read_text())
    for budget in BUDGETS:
        entry = summary["pressure"][str(budget)]
        entry["sessions_with_any_user_query_over_budget"] = sum(
            c[f"reference_query_starts_over_{budget}"] > 0 for c in conversations)
        entry["sessions_with_later_user_query_over_budget"] = sum(
            c[f"reference_later_user_queries_over_{budget}"] > 0 for c in conversations)
        entry["total_later_user_queries_over_budget"] = sum(
            c[f"reference_later_user_queries_over_{budget}"] for c in conversations)
    summary["relation_definition"] = (
        "Automatic surface/reference-argument evidence is not semantic ground truth. "
        "Conversation categories and Chinese notes are assistant reading judgments, "
        "not official BFCL labels or independent human validation.")
    summary["review_category_definitions"] = RELATIONS
    summary["review_category_counts"] = dict(Counter(c["reviewed_relation_category"] for c in conversations))
    summary["domain_definition"] = "Audit grouping by inspected primary API and fixed ID blocks of 50, not an upstream field."
    summary["domains"] = {}
    for domain in DOMAINS:
        rows = [c for c in conversations if c["primary_domain"] == domain]
        summary["domains"][domain] = {"conversations": len(rows),
            "queries": sum(c["user_query_count"] for c in rows),
            "reference_max_prompt_tokens": distribution([c["reference_max_prompt_tokens"] for c in rows]),
            "over_32k": sum(c["reference_max_over_32768"] for c in rows),
            "later_user_query_over_32k": sum(c["reference_later_user_queries_over_32768"] > 0 for c in rows)}
    summary["candidate_definition"] = (
        "At least 4 distinct user messages and at least 2 later user-query starts "
        "with an uncompressed reference prompt over 32768 tokens; metadata selection, "
        "not evidence of a mandatory eviction or a history benefit under a particular policy.")
    summary["recommended_candidates_32k"] = [c["id"] for c in candidates]
    summary["pressure_candidates_32k"] = [c["id"] for c in pressure_candidates]
    summary["runtime"] = {"python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ["transformers", "tokenizers", "mpmath"]}}
    summary["validation"] = {"conversation_ids_unique": True, "all_200_semantically_reviewed": True,
        "query_counts_consistent": True, "reference_tool_errors": 0,
        "reference_calls": sum(len(r["tool_calls"]) for r in references)}
    parity_path = analysis / "native_tokenizer_parity.json"
    if parity_path.exists():
        summary["validation"]["native_tokenizer_parity"] = json.loads(parity_path.read_text())
    (analysis / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    commit = summary["source_commit"]
    manifest = []
    for path in sorted((root / "source").rglob("*")):
        if not path.is_file() or path.name == "source_manifest.json":
            continue
        relative = path.relative_to(root / "source").as_posix()
        if relative.startswith("upstream/"):
            upstream_path = "berkeley-function-call-leaderboard/" + relative.removeprefix("upstream/")
        elif relative == "BFCL_v4_multi_turn_long_context.json":
            upstream_path = "berkeley-function-call-leaderboard/bfcl_eval/data/" + relative
        elif relative == "LICENSE":
            upstream_path = relative
        else:
            upstream_path = None
        if "__pycache__" in path.parts:
            continue
        manifest.append({"path": relative, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_url": f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{commit}/{upstream_path}"
                          if upstream_path else None})
    (root / "source/source_manifest.json").write_text(json.dumps(
        {"commit": commit, "files": manifest}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"review_counts": summary["review_category_counts"],
        "pressure_candidates_32k": len(pressure_candidates), "recommended_candidates_32k": len(candidates),
        "validation": summary["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
