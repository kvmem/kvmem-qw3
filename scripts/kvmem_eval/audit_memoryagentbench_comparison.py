#!/usr/bin/env python3
"""Strict completion audit for the frozen-branch MemoryAgentBench comparison.

This verifier is intentionally separate from generation and scoring.  It proves
that all four methods cover the same questions, that every official/special
metric exists, and that the three dense baselines retained their advertised
context semantics and frozen parameters.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FULL_CONTEXTS = 146
FULL_QUESTIONS = 3671
SPECIAL_QUESTIONS = 400
SPECIAL_QUESTIONS_BY_PHASE = {
    "over256k": 316,
    "under256k": 84,
    "full": SPECIAL_QUESTIONS,
}
EXPECTED_METHODS = {"KVMem", "Compact", "Compact+RAG", "Window32K"}
PHASES = {
    "over256k": (30, 1316),
    "under256k": (116, 2355),
    "full": (146, 3671),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvmem-results", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=tuple(PHASES), default="full",
        help="audit one length cohort or the complete benchmark",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing required artifact: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict[str, Any]) -> str:
    return (
        f"{row['split']}|{int(row['dataset_row'])}|"
        f"{int(row['question_index'])}"
    )


def row_key(row: dict[str, Any]) -> str:
    return f"{row['split']}|{int(row['dataset_row'])}"


def metric(source: str) -> str:
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
    raise RuntimeError(f"unsupported source: {source}")


def collect_records(root: Path) -> tuple[dict[str, dict[str, Any]], int]:
    records: dict[str, dict[str, Any]] = {}
    rows = 0
    for path in sorted((root / "rows").glob("*/results.jsonl")):
        rows += 1
        for record in read_jsonl(path):
            identity = key(record)
            if identity in records:
                raise RuntimeError(f"{root}: duplicate question {identity}")
            records[identity] = record
    return records, rows


def select_kvmem_records(
    records: dict[str, dict[str, Any]], phase: str,
) -> dict[str, dict[str, Any]]:
    if phase == "full":
        selected = records
    elif phase == "over256k":
        selected = {
            identity: record for identity, record in records.items()
            if int(record.get("archive_tokens", -1)) > 262144
        }
    else:
        selected = {
            identity: record for identity, record in records.items()
            if 0 <= int(record.get("archive_tokens", -1)) <= 262144
        }
    expected_contexts, expected_questions = PHASES[phase]
    contexts = {row_key(record) for record in selected.values()}
    if len(contexts) != expected_contexts or len(selected) != expected_questions:
        raise RuntimeError(
            f"KVMem {phase}: selected contexts={len(contexts)}/"
            f"{expected_contexts} questions={len(selected)}/{expected_questions}"
        )
    return selected


def validate_scores(
    root: Path, records: dict[str, dict[str, Any]], phase: str,
) -> dict[str, Any]:
    special_expected = {
        identity
        for identity, record in records.items()
        if metric(str(record["source"])) in (
            "llm_judge_correct", "gpt4_summary_f1"
        )
    }
    expected_special = SPECIAL_QUESTIONS_BY_PHASE[phase]
    if len(special_expected) != expected_special:
        raise RuntimeError(
            f"{root}: special target count {len(special_expected)} "
            f"!= {expected_special}"
        )
    judgments = read_jsonl(root / "special_judgments.jsonl")
    judged: dict[str, dict[str, Any]] = {}
    for row in judgments:
        identity = str(row["key"])
        if identity in judged:
            raise RuntimeError(f"{root}: duplicate special judgment {identity}")
        score = float(row["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError(f"{root}: invalid judge score {identity}={score}")
        judged[identity] = row
    missing_judgments = special_expected - set(judged)
    extra_judgments = set(judged) - special_expected
    if missing_judgments or (phase == "full" and extra_judgments):
        raise RuntimeError(
            f"{root}: special key mismatch missing="
            f"{len(missing_judgments)} extra={len(extra_judgments)}"
        )

    deterministic = 0
    for identity, record in records.items():
        name = metric(str(record["source"]))
        if name in ("llm_judge_correct", "gpt4_summary_f1"):
            continue
        metrics = record.get("official_local_metrics") or {}
        if name not in metrics:
            raise RuntimeError(f"{root}: missing {name} for {identity}")
        value = float(metrics[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"{root}: invalid {name} for {identity}: {value}")
        deterministic += 1

    final = read_json(root / "final_summary.json")
    final_questions = int(final.get("questions", -1))
    if final_questions not in (len(records), FULL_QUESTIONS):
        raise RuntimeError(
            f"{root}: invalid final_summary question count {final_questions}; "
            f"expected {len(records)} or {FULL_QUESTIONS}"
        )
    return {
        "deterministic_questions": deterministic,
        "special_questions": len(special_expected),
        "special_judgments_available": len(judged),
        "judge_model": final.get("judge_model_for_special_metrics"),
    }


def validate_run_config(path: Path, method: str) -> None:
    config = read_json(path)
    if config.get("method") != method:
        raise RuntimeError(f"{path}: method mismatch")
    server = config.get("server_required") or {}
    expected_server = {
        "context_tokens": 262144,
        "kv_dtype": "fp8",
        "prefill_chunk_tokens": 2048,
        "mtp_chain": 0,
        "continuous_batching": True,
        "prefix_cache": True,
        "prefix_cache_commit_guard_pages": 1,
        "parallel": 1,
        "kvmem": False,
    }
    for name, expected in expected_server.items():
        if server.get(name) != expected:
            raise RuntimeError(
                f"{path}: server_required.{name}={server.get(name)!r}, "
                f"expected {expected!r}"
            )
    sampling = config.get("final_sampling") or {}
    expected_sampling = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "thinking": False,
    }
    for name, expected in expected_sampling.items():
        if sampling.get(name) != expected:
            raise RuntimeError(
                f"{path}: final_sampling.{name} mismatch"
            )
    rag = config.get("rag") or {}
    if (rag.get("top_k"), rag.get("block_size"), rag.get("overlap")) != (
        30, 1024, 128
    ):
        raise RuntimeError(f"{path}: RAG configuration mismatch")
    if (config.get("sliding") or {}).get("complete_prompt_tokens") != 32768:
        raise RuntimeError(f"{path}: sliding-window budget mismatch")
    if method == "sliding-window":
        sliding = config.get("sliding") or {}
        reused = sliding.get("reused_from_256k_reference") or {}
        if reused != {
            "context_window": 262144,
            "complete_prompt_tokens": 32768,
            "selection": "largest recent raw-history suffix",
            "prompt_budget_scope": "complete final prompt",
        }:
            raise RuntimeError(f"{path}: 256K sliding-reference semantics mismatch")
        for name in ("reference", "reference_256k_config"):
            reference = Path(str(sliding.get(name) or ""))
            expected_sha = sliding.get(f"{name}_sha256")
            if not reference.is_file() or sha256_file(reference) != expected_sha:
                raise RuntimeError(f"{path}: stale sliding reference {name}")


def validate_kvmem_config(root: Path) -> dict[str, Any]:
    path = root / "run_config.json"
    config = read_json(path)
    expected = {
        "contexts_selected": FULL_CONTEXTS,
        "retrieval": "key-direction-adaptive",
        "adaptive_gain_1to2": 0.10,
        "adaptive_gain_2to4": 0.06,
        "block_tokens": 512,
        "budget": 204800,
        "gen_budget": 32768,
        "sink_tokens": 2048,
        "recent_tokens": 16384,
        "index_placement": "cpu",
        "prefill_chunk": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "thinking": False,
        "kv_dtype": "fp8 archive; FP16 index/query",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise RuntimeError(
                f"{path}: {name}={config.get(name)!r}, expected {value!r}"
            )
    return expected


def validate_baseline_semantics(
    method: str, root: Path, records: dict[str, dict[str, Any]], phase: str,
) -> dict[str, Any]:
    config_phases = ("over256k", "under256k") if phase == "full" else (phase,)
    for config_phase in config_phases:
        validate_run_config(root / f"run_config_{config_phase}.json", method)

    shared: dict[str, set[str]] = defaultdict(set)
    retrieval_variants: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    rag_tokens: list[int] = []
    for identity, record in records.items():
        if record.get("previous_question_or_answer_appended") is not False:
            raise RuntimeError(f"{root}: non-frozen branch at {identity}")
        shared[row_key(record)].add(str(record["shared_context_sha256"]))
        retrieval = record.get("retrieval")
        if method == "compact-rag":
            if not isinstance(retrieval, dict):
                raise RuntimeError(f"{root}: missing RAG payload at {identity}")
            tokens = int(retrieval["retrieved_tokens_estimate"])
            # The exact top-30 payload is at most 30 * 1024 Qwen source tokens.
            if not 0 < tokens <= 30720:
                raise RuntimeError(f"{root}: invalid RAG budget at {identity}: {tokens}")
            block_ids = tuple(str(value) for value in retrieval["retrieved_block_ids"])
            if len(block_ids) != len(set(block_ids)):
                raise RuntimeError(f"{root}: duplicate RAG block at {identity}")
            if len(block_ids) != int(retrieval["effective_top_k"]):
                raise RuntimeError(f"{root}: RAG top-k mismatch at {identity}")
            rag_tokens.append(tokens)
            retrieval_variants[row_key(record)].add(block_ids)
        elif retrieval is not None:
            raise RuntimeError(f"{root}: unexpected retrieval payload at {identity}")
    bad_shared = [name for name, values in shared.items() if len(values) != 1]
    if bad_shared:
        raise RuntimeError(f"{root}: question-dependent shared contexts: {bad_shared[:3]}")
    if method == "compact-rag" and not any(
        len(values) > 1 for values in retrieval_variants.values()
    ):
        raise RuntimeError(f"{root}: RAG selections never vary by question")
    return {
        "frozen_context_rows": len(shared),
        "question_dependent_rag_rows": sum(
            len(values) > 1 for values in retrieval_variants.values()
        ),
        "rag_token_min": min(rag_tokens) if rag_tokens else None,
        "rag_token_max": max(rag_tokens) if rag_tokens else None,
    }


def validate_comparison(path: Path, phase: str) -> dict[str, Any]:
    expected_contexts, expected_questions = PHASES[phase]
    report = read_json(path)
    if (
        int(report.get("selected_contexts", -1)) != expected_contexts
        or int(report.get("selected_questions", -1)) != expected_questions
    ):
        raise RuntimeError(f"{path}: selection count mismatch")
    methods = report.get("methods") or {}
    if set(methods) != EXPECTED_METHODS:
        raise RuntimeError(f"{path}: method set mismatch {set(methods)}")
    for name, values in methods.items():
        if (
            int(values.get("contexts", -1)) != expected_contexts
            or int(values.get("questions", -1)) != expected_questions
        ):
            raise RuntimeError(f"{path}: incomplete method {name}")
        for aggregate in (
            "context_macro_mean",
            "question_weighted_mean",
            "overall_macro_category_mean",
        ):
            score = float(values[aggregate])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise RuntimeError(
                    f"{path}: invalid {aggregate} for {name}: {score}"
                )
    markdown = path.with_suffix(".md")
    text = markdown.read_text(encoding="utf-8")
    if not all(
        section in text
        for section in (
            "## Aggregation summary",
            "## Category summary",
            "## Dataset details",
        )
    ):
        raise RuntimeError(f"{markdown}: incomplete human-readable tables")
    return {
        "contexts": expected_contexts,
        "questions": expected_questions,
        "methods": sorted(methods),
    }


def main() -> int:
    args = parse_args()
    roots = {
        "KVMem": args.kvmem_results,
        "Compact": args.workspace / "methods/compact-only",
        "Compact+RAG": args.workspace / "methods/compact-rag",
        "Window32K": args.workspace / "methods/sliding-window",
    }
    all_records: dict[str, dict[str, dict[str, Any]]] = {}
    scoring = {}
    kvmem_all, kvmem_rows = collect_records(roots["KVMem"])
    if kvmem_rows != FULL_CONTEXTS or len(kvmem_all) != FULL_QUESTIONS:
        raise RuntimeError(
            f"{roots['KVMem']}: incomplete canonical KVMem results "
            f"contexts={kvmem_rows}/{FULL_CONTEXTS} "
            f"questions={len(kvmem_all)}/{FULL_QUESTIONS}"
        )
    all_records["KVMem"] = select_kvmem_records(kvmem_all, args.phase)
    expected_keys = set(all_records["KVMem"])
    for name, root in roots.items():
        if name == "KVMem":
            continue
        available, _ = collect_records(root)
        missing = expected_keys - set(available)
        if missing:
            raise RuntimeError(
                f"{name}: question identity mismatch missing="
                f"{len(missing)} available={len(available)}"
            )
        all_records[name] = {
            identity: available[identity] for identity in expected_keys
        }
    for name, root in roots.items():
        scoring[name] = validate_scores(root, all_records[name], args.phase)

    semantics = {}
    for method, label in (
        ("compact-only", "Compact"),
        ("compact-rag", "Compact+RAG"),
        ("sliding-window", "Window32K"),
    ):
        semantics[label] = validate_baseline_semantics(
            method, roots[label], all_records[label], args.phase
        )
    compact_hashes = {
        row_key(record): record["shared_context_sha256"]
        for record in all_records["Compact"].values()
    }
    rag_hashes = {
        row_key(record): record["shared_context_sha256"]
        for record in all_records["Compact+RAG"].values()
    }
    if compact_hashes != rag_hashes:
        raise RuntimeError("Compact and Compact+RAG did not reuse identical compaction")

    row_summaries = [
        read_json(path)
        for path in sorted((args.kvmem_results / "rows").glob("*/row_summary.json"))
    ]
    if len(row_summaries) != FULL_CONTEXTS:
        raise RuntimeError("KVMem row-summary count mismatch")
    scorer_events = sum(int(row.get("scorer_events", 0)) for row in row_summaries)
    if scorer_events != FULL_QUESTIONS:
        raise RuntimeError(
            f"KVMem scorer event count {scorer_events} != {FULL_QUESTIONS}"
        )
    fallbacks = sum(int(row.get("scorer_fallbacks", 0)) for row in row_summaries)
    if fallbacks:
        raise RuntimeError(f"KVMem scorer fallbacks detected: {fallbacks}")

    comparison_phases = tuple(PHASES) if args.phase == "full" else (args.phase,)
    comparisons = {
        phase: validate_comparison(
            args.workspace / f"comparison_{phase}.json", phase
        )
        for phase in comparison_phases
    }
    expected_contexts, expected_questions = PHASES[args.phase]
    audit = {
        "schema_version": 1,
        "status": "pass",
        "phase": args.phase,
        "contexts": expected_contexts,
        "questions": expected_questions,
        "identical_question_keys": True,
        "kvmem_config": validate_kvmem_config(args.kvmem_results),
        "kvmem_scorer_events": scorer_events,
        "kvmem_scorer_fallbacks": fallbacks,
        "scoring": scoring,
        "baseline_semantics": semantics,
        "compact_shared_context_identical_to_compact_rag": True,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
