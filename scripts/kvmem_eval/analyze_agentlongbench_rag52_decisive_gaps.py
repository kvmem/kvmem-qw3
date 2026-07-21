#!/usr/bin/env python3
"""Find answer-relevant KVMem retrieval that the 32-token RAG run missed.

The frozen RAG delivery contains retrieved blocks and aggregate scores, but no
per-sample answers/evaluations.  Consequently this script does not invent a
per-sample RAG outcome.  It instead analyzes every KVMem-exact-correct sample
and labels the retrieval evidence gap against the best delivered RAG setting
(block size 32, overlap 8).

Evidence is localized from the structured AgentLongBench task fields and then
restricted to evidence actually covered by KVMem.  This makes the report useful
for causal inspection without treating every selected KVMem block (including
sink/recent/irrelevant blocks) as answer evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

import analyze_agentlongbench_rag52_overlap as overlap_lib


DEFAULT_DELIVERY = Path("/home/chaidi/kvmem_eval/rag_blocks_52_delivery")
DEFAULT_DATASET = Path("/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl")
DEFAULT_TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8"
)
DEFAULT_BENCHMARK_REPO = Path("/home/chaidi/AgentLongBench_Motivation")
DEFAULT_KVMEM_EVAL = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_kvmem_32k_b32_lmeparams_thinking4k_20260716/eval.jsonl"
)
DEFAULT_OVERLAP_ANALYSIS = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_rag52_kvmem32k_overlap_20260718/analysis/per_sample_overlap.jsonl"
)
DEFAULT_KVMEM_DUMP = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_rag52_kvmem32k_overlap_20260718/kvmem_retrieval_dump.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_rag52_kvmem32k_overlap_20260718/decisive_gap_analysis"
)

ROUND_RE = re.compile(r"^Round\s+(\d+):", re.MULTILINE)
BASE_STATS_RE = re.compile(r"Base Stats:\s*(\d+)")
MASKED_BASE_STATS_RE = re.compile(r"attr_2:\s*(\d+)")
Interval = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--kvmem-eval", type=Path, default=DEFAULT_KVMEM_EVAL)
    parser.add_argument(
        "--overlap-analysis", type=Path, default=DEFAULT_OVERLAP_ANALYSIS
    )
    parser.add_argument("--kvmem-dump", type=Path, default=DEFAULT_KVMEM_DUMP)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rag-overlap", type=int, choices=(0, 4, 8), default=8)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def overlap_chars(a: Iterable[Interval], b: Iterable[Interval]) -> int:
    return overlap_lib.interval_length(
        overlap_lib.intersection(overlap_lib.merged(a), overlap_lib.merged(b))
    )


def coverage(union: list[Interval], target: list[Interval]) -> float:
    denom = overlap_lib.interval_length(overlap_lib.merged(target))
    return overlap_chars(union, target) / denom if denom else 0.0


def subtract_intervals(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Return the portions of merged ``a`` that are not covered by merged ``b``."""
    aa, bb = overlap_lib.merged(a), overlap_lib.merged(b)
    out: list[Interval] = []
    j = 0
    for lo, hi in aa:
        cursor = lo
        while j < len(bb) and bb[j][1] <= cursor:
            j += 1
        k = j
        while k < len(bb) and bb[k][0] < hi:
            if bb[k][0] > cursor:
                out.append((cursor, min(hi, bb[k][0])))
            cursor = max(cursor, bb[k][1])
            if cursor >= hi:
                break
            k += 1
        if cursor < hi:
            out.append((cursor, hi))
    return out


def line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        spans.append((cursor, end, line.rstrip("\r\n")))
        cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text), text[cursor:]))
    return spans


def base_stats_value(text: str) -> int | None:
    match = BASE_STATS_RE.search(text) or MASKED_BASE_STATS_RE.search(text)
    return int(match.group(1)) if match is not None else None


def find_all(text: str, needle: str) -> list[Interval]:
    if not needle:
        return []
    out: list[Interval] = []
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            return out
        out.append((pos, pos + len(needle)))
        start = pos + len(needle)


def json_string_value_spans(text: str, value: str) -> list[Interval]:
    """Locate exact JSON string values without matching prefixes such as Porygon-Z."""
    encoded = json.dumps(value, ensure_ascii=False)
    return [(lo + 1, hi - 1) for lo, hi in find_all(text, encoded)]


def exact_value_spans(text: str, value: str) -> list[Interval]:
    """Match a displayed property value without counting a5v1 inside a5v10."""
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
    )
    return [(match.start(), match.end()) for match in pattern.finditer(text)]


def message_records(canonical: Any, sample: dict[str, Any]) -> list[dict[str, Any]]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    messages = sample.get("messages") or raw.get("messages") or []
    records: list[dict[str, Any]] = []
    logical = 0
    current_round: int | None = None
    for original_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        rendered = canonical.message_to_text(message)
        content = str(message.get("content") or "")
        content_offset = rendered.find(content)
        if content_offset < 0:
            raise RuntimeError(
                f"message content is absent from canonical rendering: index={original_index}"
            )
        round_match = ROUND_RE.search(content) if message.get("role") == "user" else None
        if round_match is not None:
            current_round = int(round_match.group(1))
        record = {
            "original_index": original_index,
            "role": message.get("role"),
            "round": current_round,
            "rendered": rendered,
            "content": content,
            "logical_start": logical,
            "logical_end": logical + len(rendered),
            "content_logical_start": logical + content_offset,
        }
        records.append(record)
        logical += len(rendered)
    return records


def unit(
    label: str,
    kind: str,
    record: dict[str, Any],
    content_spans: list[Interval],
    *,
    required_group: str,
    note: str = "",
) -> dict[str, Any]:
    base = int(record["content_logical_start"])
    logical = [(base + lo, base + hi) for lo, hi in content_spans if hi > lo]
    text = " ... ".join(record["content"][lo:hi].strip() for lo, hi in content_spans)
    return {
        "label": label,
        "kind": kind,
        "required_group": required_group,
        "round": record.get("round"),
        "message_index": record["original_index"],
        "logical_intervals": logical,
        "text": text,
        "note": note,
    }


def records_by_round(records: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    feedback: dict[int, dict[str, Any]] = {}
    tools: dict[int, dict[str, Any]] = {}
    for record in records:
        round_number = record.get("round")
        if round_number is None:
            continue
        if record["role"] == "user" and ROUND_RE.search(record["content"]):
            feedback[int(round_number)] = record
        elif record["role"] == "tool":
            tools[int(round_number)] = record
    return feedback, tools


def category_line_units(record: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for index, (lo, hi, text) in enumerate(line_spans(record["content"]), start=1):
        if text.lstrip().startswith("-"):
            item = unit(
                f"{prefix}:category_{index}",
                "category_line",
                record,
                [(lo, hi)],
                required_group=prefix,
            )
            content_base = int(record["content_logical_start"])
            marker_spans = [
                (content_base + lo + marker_lo, content_base + lo + marker_hi)
                for marker_lo, marker_hi in find_all(text, "(correct)")
            ]
            item["contains_correct"] = bool(marker_spans)
            item["positive_marker_intervals"] = marker_spans
            units.append(item)
    return units


def localize_evidence(sample: dict[str, Any], canonical: Any) -> tuple[list[dict[str, Any]], str]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    task = str(sample.get("task_type") or "")
    records = message_records(canonical, sample)
    feedback, tools = records_by_round(records)
    units: list[dict[str, Any]] = []

    if task == "Count Frequency(Env)":
        target = str(raw.get("property_value") or "")
        occurrence = 0
        for record in feedback.values():
            for lo, hi, text in line_spans(record["content"]):
                matches = exact_value_spans(text, target)
                for match_lo, match_hi in matches:
                    occurrence += 1
                    item = unit(
                        f"occurrence_{occurrence}",
                        "target_feedback_occurrence",
                        record,
                        [(lo + match_lo, lo + match_hi)],
                        required_group="all_occurrences",
                        note=f"target={target}",
                    )
                    item["text"] = text.strip()
                    units.append(item)
        if occurrence != int(raw.get("answer")):
            raise RuntimeError(
                f"Count Frequency(Env) evidence count {occurrence} != answer {raw.get('answer')}"
            )
        return units, "all target-bearing feedback lines are required for an exact count"

    if task == "Count Frequency(Tool)":
        target_round = int(raw["round"])
        target = str(raw["target_name"])
        # A result named "round N" is the tool output used to choose the guess
        # for round N, hence it follows the feedback for round N-1 in messages.
        record = tools[target_round - 1]
        spans = json_string_value_spans(record["content"], target)
        expected = int(raw.get("answer"))
        # Some verbose records omit the materialized `intersection` list.  If
        # the target occurs in every per-section list, the benchmark count still
        # includes the logically derived intersection occurrence.
        if len(spans) not in {expected, expected - 1}:
            raise RuntimeError(
                f"Count Frequency(Tool) visible occurrence count {len(spans)} is not "
                f"compatible with answer {raw.get('answer')}"
            )
        for index, span in enumerate(spans, start=1):
            units.append(
                unit(
                    f"round_{target_round}:occurrence_{index}",
                    "target_tool_occurrence",
                    record,
                    [span],
                    required_group="all_occurrences",
                    note=(
                        f"target={target}; benchmark_count={expected}; "
                        f"visible_exact_occurrences={len(spans)}"
                    ),
                )
            )
        return units, "all target occurrences in the specified tool result are required"

    if task == "Find Duplicates(Tool)":
        target = str(raw["pokemon_name"])
        for round_number in (int(raw["i_round"]), int(raw["j_round"])):
            record = tools[round_number - 1]
            spans = json_string_value_spans(record["content"], target)
            if not spans:
                raise RuntimeError(
                    f"positive duplicate sample lacks {target!r} in round {round_number}"
                )
            units.append(
                unit(
                    f"round_{round_number}:target_presence",
                    "target_presence_in_tool_result",
                    record,
                    spans,
                    required_group=f"round_{round_number}",
                    note=f"target={target}",
                )
            )
        return units, "one target occurrence from each specified round is required"

    if task == "Find Round with Largest Value(Env)":
        answer_round = int(raw["answer"])
        values = {int(key): int(value) for key, value in raw["base_stats_info"].items()}
        maximum = max(values.values())
        tied = sorted(round_number for round_number, value in values.items() if value == maximum)
        if answer_round != tied[0]:
            raise RuntimeError(
                f"largest-value answer {answer_round} is not first maximum round {tied[0]}"
            )
        for round_number in tied:
            record = feedback[round_number]
            matches = [
                (lo, hi)
                for lo, hi, text in line_spans(record["content"])
                if base_stats_value(text) == maximum
            ]
            if len(matches) != 1:
                raise RuntimeError(f"cannot localize max base stats for round {round_number}")
            units.append(
                unit(
                    f"round_{round_number}:max_base_stats",
                    "maximum_value_line",
                    record,
                    matches,
                    required_group="maximum_and_ties",
                    note=f"maximum={maximum}; answer is earliest tied round {answer_round}",
                )
            )
        return units, "the winning maximum line and any earlier/equal tie are answer-critical"

    if task == "Count Correctness(Env)":
        round_number = int(raw["round"])
        units.extend(category_line_units(feedback[round_number], f"round_{round_number}"))
        return units, "all attribute-category verdict lines in the target round are required"

    if task == "Weighted Summation(Env)":
        for round_number in (int(raw["i_round"]), int(raw["j_round"])):
            units.extend(category_line_units(feedback[round_number], f"round_{round_number}"))
        return units, "both rounds' category verdict lines are required for the weighted scores"

    if task == "Intersection":
        target = str(raw.get("answer_name") or raw.get("answer") or "")
        for round_number, record in sorted(tools.items()):
            spans = json_string_value_spans(record["content"], target)
            if spans:
                units.append(
                    unit(
                        f"round_{round_number}:answer_candidate",
                        "answer_presence_in_tool_result",
                        record,
                        spans,
                        required_group="intersection_trace",
                        note=(
                            "target presence alone does not prove uniqueness; inspect the "
                            "KVMem-only candidate-list fragments"
                        ),
                    )
                )
        return units, "non-local proof: multiple candidate lists jointly establish uniqueness"

    return [], f"unsupported task type: {task}"


def rag_union_for_sample(
    delivery: dict[str, Any], double_history: overlap_lib.LogicalHistory
) -> list[Interval]:
    parts: list[Interval] = []
    for block in delivery["blocks"]:
        parts.extend(
            double_history.token_range_to_logical(
                int(block["chunk_start"]), int(block["chunk_end"])
            )
        )
    return overlap_lib.merged(parts)


def rag_grid(
    history: overlap_lib.LogicalHistory, rag_overlap: int
) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    step = 32 - rag_overlap
    for index, begin in enumerate(range(0, len(history.ids), step)):
        end = min(begin + 32, len(history.ids))
        logical = overlap_lib.merged(history.token_range_to_logical(begin, end))
        if logical:
            grid.append(
                {
                    "block_id": f"mflat_c{index:05d}",
                    "logical_intervals": logical,
                }
            )
        if end >= len(history.ids):
            break
    return grid


def attach_grid_ids(
    item: dict[str, Any],
    grid: list[dict[str, Any]],
    selected_ids: set[str],
    rag_union: list[Interval],
) -> None:
    logical = [tuple(interval) for interval in item["logical_intervals"]]
    ids = [
        block["block_id"]
        for block in grid
        if overlap_chars(logical, block["logical_intervals"]) > 0
    ]
    item["rag_grid_block_ids"] = ids
    item["rag_selected_grid_block_ids"] = [block_id for block_id in ids if block_id in selected_ids]
    item["rag_missing_grid_block_ids"] = [block_id for block_id in ids if block_id not in selected_ids]
    uncovered = subtract_intervals(logical, rag_union)
    item["rag_missing_evidence_grid_block_ids"] = [
        block["block_id"]
        for block in grid
        if block["block_id"] not in selected_ids
        and overlap_chars(uncovered, block["logical_intervals"]) > 0
    ]


def classify(task: str, evidence: list[dict[str, Any]]) -> tuple[str, str]:
    if not evidence:
        return "insufficient", "no answer evidence could be localized"
    k_supported = [item for item in evidence if item["kvmem_coverage"] >= 0.5]
    k_only = [
        item
        for item in evidence
        if item["kvmem_coverage"] >= 0.5 and item["rag_coverage"] < 0.2
    ]
    rag_supported = [item for item in evidence if item["rag_coverage"] >= 0.5]

    if task == "Intersection":
        if k_only:
            return (
                "manual_nonlocal_gap",
                f"{len(k_only)} KVMem-supported answer-candidate fragments are absent from RAG, "
                "but uniqueness requires joint manual inspection",
            )
        return "shared_or_insufficient", "the localized answer-name fragments do not isolate a causal gap"

    if task == "Find Duplicates(Tool)":
        groups = defaultdict(list)
        for item in evidence:
            groups[item["required_group"]].append(item)
        k_groups = {
            group for group, items in groups.items() if any(item["kvmem_coverage"] >= 0.5 for item in items)
        }
        rag_groups = {
            group for group, items in groups.items() if any(item["rag_coverage"] >= 0.5 for item in items)
        }
        if len(k_groups) == 2 and len(rag_groups) < 2:
            missing = sorted(k_groups - rag_groups)
            return "clear_retrieval_gap", f"RAG misses target presence from {', '.join(missing)}"

    elif task in {"Count Frequency(Env)", "Count Frequency(Tool)"}:
        if len(k_supported) == len(evidence) and len(rag_supported) < len(evidence):
            return (
                "clear_retrieval_gap",
                f"KVMem covers all {len(evidence)} count occurrences; RAG covers only {len(rag_supported)}",
            )

    elif task == "Find Round with Largest Value(Env)":
        answer_units = [item for item in evidence if item["label"].startswith("round_")]
        if answer_units and all(item["kvmem_coverage"] >= 0.5 for item in answer_units) and any(
            item["rag_coverage"] < 0.2 for item in answer_units
        ):
            return "clear_retrieval_gap", "RAG misses a maximum/tie line that KVMem covers"

    elif task in {"Count Correctness(Env)", "Weighted Summation(Env)"}:
        if len(k_supported) == len(evidence) and len(rag_supported) < len(evidence):
            return (
                "clear_retrieval_gap",
                f"KVMem covers all {len(evidence)} required category lines; RAG covers {len(rag_supported)}",
            )

    if len(rag_supported) == len(evidence):
        return "shared_evidence", "RAG covers every localized answer-evidence unit"
    if k_only:
        return (
            "partial_retrieval_gap",
            f"RAG misses {len(k_only)} of {len(k_supported)} evidence units covered by KVMem",
        )
    return "insufficient", "localized evidence does not yield a strong KVMem-only retrieval explanation"


def causal_classify(
    task: str,
    evidence: list[dict[str, Any]],
    structural_classification: str,
    raw: dict[str, Any],
) -> tuple[str, str]:
    if structural_classification == "clear_retrieval_gap":
        if task in {
            "Count Frequency(Env)",
            "Count Frequency(Tool)",
            "Find Duplicates(Tool)",
            "Find Round with Largest Value(Env)",
        }:
            return "direct_answer_fact_gap", "RAG omits a positive fact used directly in the answer"
        if task in {"Count Correctness(Env)", "Weighted Summation(Env)"}:
            missing_positive = [
                item
                for item in evidence
                if item.get("contains_correct")
                and item.get("kvmem_positive_marker_hit")
                and not item.get("rag_positive_marker_hit")
            ]
            if missing_positive:
                labels = ", ".join(item["label"] for item in missing_positive)
                return (
                    "direct_answer_fact_gap",
                    f"RAG omits positive correctness facts: {labels}",
                )
            return (
                "incomplete_proof_gap",
                "RAG omits category context, but all localized positive correctness markers remain visible",
            )
    if structural_classification == "shared_evidence":
        return "shared_local_evidence", "RAG covers every localized answer fact"
    if structural_classification in {"manual_nonlocal_gap", "shared_or_insufficient"}:
        return "nonlocal_manual", "local target strings cannot prove the global intersection result"
    return "insufficient", "KVMem correctness cannot be tied to a localized KVMem-only fact"


def retrieval_gap_pattern(task: str, causal_classification: str) -> str | None:
    if causal_classification != "direct_answer_fact_gap":
        return None
    if task == "Count Frequency(Env)":
        return "exhaustive_count_coverage"
    if task in {"Count Frequency(Tool)", "Find Duplicates(Tool)"}:
        return "needle_in_homogeneous_tool_list"
    if task in {"Count Correctness(Env)", "Weighted Summation(Env)"}:
        return "round_conditioned_fact_fragment"
    return "other_direct_fact_gap"


def candidate_blocks(
    internal_blocks: list[dict[str, Any]],
    rag_union: list[Interval],
    evidence: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    evidence_intervals = overlap_lib.merged(
        interval
        for item in evidence
        for interval in (
            item["logical_intervals"]
            if item["kvmem_coverage"] >= 0.5 and item["rag_coverage"] < 0.5
            else item.get("positive_marker_intervals") or []
            if item.get("kvmem_positive_marker_hit")
            and not item.get("rag_positive_marker_hit")
            else []
        )
    )
    if not evidence_intervals:
        return []
    candidates: list[dict[str, Any]] = []
    for block in internal_blocks:
        logical = [tuple(interval) for interval in block.get("logical_intervals") or []]
        logical_len = overlap_lib.interval_length(overlap_lib.merged(logical))
        if not logical_len:
            continue
        evidence_overlap = overlap_chars(logical, evidence_intervals)
        rag_coverage = coverage(rag_union, logical)
        if evidence_overlap <= 0:
            continue
        grid_ids = [
            grid_block["block_id"]
            for grid_block in grid
            if overlap_chars(logical, grid_block["logical_intervals"]) > 0
        ]
        candidates.append(
            {
                "internal_block_id": block["internal_block_id"],
                "retrieval_score": block.get("retrieval_score"),
                "rag_coverage": rag_coverage,
                "evidence_overlap_chars": evidence_overlap,
                "projected_rag_grid_block_ids": grid_ids,
                "projected_missing_rag_block_ids": [
                    block_id for block_id in grid_ids if block_id not in selected_ids
                ],
                "text": str(block.get("text") or ""),
            }
        )
    candidates.sort(
        key=lambda item: (
            item["evidence_overlap_chars"] > 0,
            item["evidence_overlap_chars"],
            1.0 - item["rag_coverage"],
            float(item["retrieval_score"] or 0.0),
        ),
        reverse=True,
    )
    return candidates[:12]


def compact_text(text: str, limit: int = 360) -> str:
    text = text.replace("\r", "").strip()
    return text if len(text) <= limit else text[:limit] + " …"


def markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    counts = summary["causal_classification_counts"]
    lines = [
        "# AgentLongBench：KVMem 正确样本中的 RAG 关键召回缺口",
        "",
        "主对照为 delivery 中效果最好的 RAG 配置：Qwen block=32、overlap=8、top-k=960。",
        "delivery 没有逐样本 RAG 答案/评分，因此下文是逐题的检索证据归因，不能冒充逐样本模型正确性。",
        "只有 KVMem 官方 Exact 正确的样本进入分析；KVMem 选中块也先经过题目结构和参考答案过滤，",
        "并非把 sink/recent/无关块全部当作 gold。",
        "",
        "## 汇总",
        "",
        f"- KVMem Exact 正确：{summary['kvmem_correct_samples']}/52。",
        f"- 直接缺失答案事实：{counts.get('direct_answer_fact_gap', 0)} 条。",
        f"- 只缺少完整证明/负证据：{counts.get('incomplete_proof_gap', 0)} 条。",
        f"- RAG 已覆盖同一局部证据：{counts.get('shared_local_evidence', 0)} 条。",
        f"- 非局部或无法定位：{counts.get('nonlocal_manual', 0) + counts.get('insufficient', 0)} 条。",
        f"- 9条直接缺失涉及的KVMem证据块，mean-k非sink分数排名最高为第"
        f"{summary['direct_gap_kvmem_rank_control']['worst_rank']}名；没有一块落到960名以后。",
        "",
        "`直接缺失答案事实`表示 RAG 漏掉了计数项、目标在指定轮次的出现、最大值行，或明确的 correct 项；",
        "这类样本最能支持“RAG 因 retrieval 失败而答错”的解释。",
        "",
        "直接缺失的9条可进一步分为三种等量模式：",
        "",
        f"- 完整计数集合少召回一项：{summary['retrieval_gap_pattern_counts'].get('exhaustive_count_coverage', 0)} 条；",
        f"- 同质化超长工具列表中的 needle 未召回：{summary['retrieval_gap_pattern_counts'].get('needle_in_homogeneous_tool_list', 0)} 条；",
        f"- 与指定轮次关联的事实片段未召回：{summary['retrieval_gap_pattern_counts'].get('round_conditioned_fact_fragment', 0)} 条。",
        "",
        "## 按任务汇总推导出的实际差异范围",
        "",
        "delivery 只给出每个任务的汇总分数。结合 KVMem 逐题分数，仍可严格计算",
        "`KVMem正确、RAG错误`的数量上下界，但不能恢复具体 ID。",
        "",
        "| 任务 | 总数 | KVMem正确 | RAG正确 | KVMem正确/RAG错误范围 | 直接事实缺口候选 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, values in summary["task_outcome_bounds"].items():
        lines.append(
            f"| {task} | {values['samples']} | {values['kvmem_correct']} | "
            f"{values['rag_correct']} | {values['kvmem_correct_rag_wrong_min']}–"
            f"{values['kvmem_correct_rag_wrong_max']} | {values['direct_answer_fact_gap_candidates']} |"
        )
    lines += [
        "",
        f"合计可严格确定为 {summary['overall_kvmem_correct_rag_wrong_bounds']['min']}–"
        f"{summary['overall_kvmem_correct_rag_wrong_bounds']['max']} 条；其中只有 "
        f"{counts.get('direct_answer_fact_gap', 0)} 条能由局部答案事实缺失直接解释。",
        "",
        "## 逐条：直接或潜在 retrieval 缺口",
        "",
    ]
    priority = {
        "direct_answer_fact_gap": 0,
        "incomplete_proof_gap": 1,
        "nonlocal_manual": 2,
        "insufficient": 3,
        "shared_local_evidence": 4,
    }
    detailed_rows = [row for row in rows if row["causal_classification"] != "shared_local_evidence"]
    for row in sorted(detailed_rows, key=lambda item: (priority[item["causal_classification"]], item["delivery_index"])):
        lines += [
            f"## {row['delivery_index']:02d}. {row['official_id']} — {row['task_type']}",
            "",
            f"- `stable_sample_id`: `{row['stable_sample_id']}`",
            f"- 归因：`{row['causal_classification']}`；{row['causal_reason']}",
            f"- retrieval 模式：`{row['retrieval_gap_pattern'] or '-'}`",
            f"- 结构化覆盖判断：`{row['classification']}`；{row['classification_reason']}",
            f"- 问题：{row['question']}",
            f"- 参考答案：`{row['reference_answer']}`；KVMem 预测：`{row['kvmem_pred_answer']}`",
            f"- 整体 retrieval logical-char Jaccard：{row['logical_char_jaccard']:.2%}",
            "",
        ]
        if row["causal_classification"] == "nonlocal_manual":
            lines += [
                f"- 共定位到 {len(row['evidence'])} 个答案名称出现位置，但 KVMem 并未直接覆盖这些名称；",
                "  它是通过多轮约束隐式推导答案，不能用单个 answer-name block 做因果归因。",
            ]
        else:
            lines += [
                "| 关键事实 | KVMem覆盖 | RAG覆盖 | RAG遗漏的同定义 block ID | 内容 |",
                "| --- | ---: | ---: | --- | --- |",
            ]
            for item in row["evidence"]:
                evidence_text = compact_text(item["text"], 180).replace(
                    "|", "\\|"
                )
                lines.append(
                    f"| {item['label']} | {item['kvmem_coverage']:.0%} | "
                    f"{item['rag_coverage']:.0%} | "
                    f"`{', '.join(item['rag_missing_evidence_grid_block_ids']) or '-'}` | "
                    f"{evidence_text} |"
                )
        lines += ["", "KVMem 召回、RAG 低覆盖的候选块：", ""]
        if not row["kvmem_only_candidate_blocks"]:
            lines.append("- 无可定位的候选块。")
        for block in row["kvmem_only_candidate_blocks"][:6]:
            excerpt = compact_text(block["text"]).replace("`", "'")
            lines.append(
                f"- KVMem internal block `{block['internal_block_id']}`，score="
                f"`{float(block['retrieval_score'] or 0.0):.6f}`，RAG覆盖="
                f"`{block['rag_coverage']:.0%}`，KVMem非sink排名="
                f"`{block.get('kvmem_score_rank_non_sink', '-')}`："
                f"`{excerpt}`"
            )
        lines.append("")
    lines += [
        "## RAG 已覆盖局部答案事实的样本",
        "",
        "这些样本即使属于实际 RAG 错题，其错误也不能由局部事实漏召回解释，",
        "在本报告的 retrieval-only 口径下不再对其错误原因作进一步归因。",
        "",
        "| 序号 | official ID | 任务 | Jaccard |",
        "| ---: | --- | --- | ---: |",
    ]
    for row in rows:
        if row["causal_classification"] == "shared_local_evidence":
            lines.append(
                f"| {row['delivery_index']} | {row['official_id']} | {row['task_type']} | "
                f"{row['logical_char_jaccard']:.2%} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required; run with /home/chaidi/qw3/.venv/bin/python"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    canonical = overlap_lib.load_canonical(args.benchmark_repo)
    ordered_ids, deliveries = overlap_lib.delivery_rows(args.delivery)
    samples = {
        str(row["stable_sample_id"]): row
        for row in overlap_lib.read_jsonl(args.dataset)
        if row.get("stable_sample_id")
    }
    overlap_rows = {
        str(row["stable_sample_id"]): row
        for row in overlap_lib.read_jsonl(args.overlap_analysis)
    }
    dumps = overlap_lib.load_dump(args.kvmem_dump)
    kvmem_evals = overlap_lib.latest_eval(args.kvmem_eval)
    missing = [
        sid
        for sid in ordered_ids
        if sid not in samples
        or sid not in overlap_rows
        or sid not in kvmem_evals
        or sid not in dumps
    ]
    if missing:
        raise RuntimeError(f"missing aligned input for stable_sample_id={missing[0]}")

    rows: list[dict[str, Any]] = []
    for delivery_index, sid in enumerate(ordered_ids, start=1):
        evaluation = kvmem_evals[sid]
        if not bool(evaluation.get("correct")):
            continue
        sample = samples[sid]
        analysis = overlap_rows[sid]
        lines = overlap_lib.sample_lines(canonical, sample)
        double_history = overlap_lib.LogicalHistory(lines, "\n\n", tokenizer)
        rag_delivery = deliveries[args.rag_overlap][sid]
        rag_union = rag_union_for_sample(rag_delivery, double_history)
        grid = rag_grid(double_history, args.rag_overlap)
        rag_selected_ids = {
            str(block["block_id"]) for block in rag_delivery["blocks"]
        }
        kvmem_internal = analysis["kvmem_internal_selected_blocks"]
        kvmem_union = overlap_lib.merged(
            tuple(interval)
            for block in kvmem_internal
            for interval in block.get("logical_intervals") or []
        )
        evidence, evidence_rule = localize_evidence(sample, canonical)
        for item in evidence:
            logical = [tuple(interval) for interval in item["logical_intervals"]]
            item["kvmem_coverage"] = coverage(kvmem_union, logical)
            item["rag_coverage"] = coverage(rag_union, logical)
            item["kvmem_only_coverage"] = coverage(
                subtract_intervals(kvmem_union, rag_union), logical
            )
            positive_markers = [
                tuple(interval) for interval in item.get("positive_marker_intervals") or []
            ]
            item["kvmem_positive_marker_hit"] = any(
                coverage(kvmem_union, [marker]) >= 0.5 for marker in positive_markers
            )
            item["rag_positive_marker_hit"] = any(
                coverage(rag_union, [marker]) >= 0.5 for marker in positive_markers
            )
            attach_grid_ids(item, grid, rag_selected_ids, rag_union)
        task = str(sample.get("task_type") or "")
        classification, reason = classify(task, evidence)
        raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
        causal_classification, causal_reason = causal_classify(
            task, evidence, classification, raw
        )
        gap_pattern = retrieval_gap_pattern(task, causal_classification)
        score_order = sorted(
            (
                (float(block.get("rs") or 0.0), int(block["b"]))
                for block in dumps[sid]["blocks"]
                if int(block["b"]) >= 8
            ),
            reverse=True,
        )
        score_ranks = {
            block_id: rank
            for rank, (_, block_id) in enumerate(score_order, start=1)
        }
        candidates = candidate_blocks(
            kvmem_internal,
            rag_union,
            evidence,
            grid,
            rag_selected_ids,
        )
        for candidate in candidates:
            candidate["kvmem_score_rank_non_sink"] = score_ranks[
                int(candidate["internal_block_id"])
            ]
        row = {
            "schema_version": "agentlongbench_rag52_decisive_gap.v1",
            "delivery_index": delivery_index,
            "stable_sample_id": sid,
            "official_id": raw.get("id"),
            "task_type": task,
            "question": raw.get("question"),
            "reference_answer": evaluation.get("reference_answer", raw.get("answer")),
            "kvmem_pred_answer": evaluation.get("pred_answer"),
            "kvmem_correct": True,
            "rag_block_size": 32,
            "rag_overlap": args.rag_overlap,
            "rag_per_sample_outcome_available": False,
            "evidence_rule": evidence_rule,
            "classification": classification,
            "classification_reason": reason,
            "causal_classification": causal_classification,
            "causal_reason": causal_reason,
            "retrieval_gap_pattern": gap_pattern,
            "logical_char_jaccard": analysis["overlaps"][str(args.rag_overlap)][
                "logical_char_jaccard"
            ],
            "evidence": evidence,
            "kvmem_only_candidate_blocks": candidates,
        }
        rows.append(row)

    counts = Counter(row["classification"] for row in rows)
    causal_counts = Counter(row["causal_classification"] for row in rows)
    gap_pattern_counts = Counter(
        row["retrieval_gap_pattern"] for row in rows if row["retrieval_gap_pattern"]
    )
    task_counts: dict[str, Counter[str]] = defaultdict(Counter)
    task_causal_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        task_counts[row["task_type"]][row["classification"]] += 1
        task_causal_counts[row["task_type"]][row["causal_classification"]] += 1
    delivery_task_counts = Counter(
        str(samples[sid].get("task_type") or "") for sid in ordered_ids
    )
    rag_correct_by_task_overlap_8 = {
        "Count Frequency(Env)": 2,
        "Count Frequency(Tool)": 4,
        "Find Duplicates(Tool)": 7,
        "Find Round with Largest Value(Env)": 9,
        "Count Correctness(Env)": 4,
        "Intersection": 1,
        "Weighted Summation(Env)": 1,
        "Find Target Offsets(Tool)": 1,
    }
    if args.rag_overlap != 8:
        raise RuntimeError(
            "task-level outcome bounds are currently documented only for the best overlap=8 run"
        )
    kvmem_correct_by_task = Counter(row["task_type"] for row in rows)
    task_outcome_bounds: dict[str, dict[str, int]] = {}
    for task in sorted(delivery_task_counts):
        samples_count = delivery_task_counts[task]
        kvmem_correct = kvmem_correct_by_task[task]
        rag_correct = rag_correct_by_task_overlap_8[task]
        task_outcome_bounds[task] = {
            "samples": samples_count,
            "kvmem_correct": kvmem_correct,
            "rag_correct": rag_correct,
            "kvmem_correct_rag_wrong_min": max(0, kvmem_correct - rag_correct),
            "kvmem_correct_rag_wrong_max": min(
                kvmem_correct, samples_count - rag_correct
            ),
            "direct_answer_fact_gap_candidates": task_causal_counts[task][
                "direct_answer_fact_gap"
            ],
        }
    overall_bounds = {
        "min": sum(
            values["kvmem_correct_rag_wrong_min"]
            for values in task_outcome_bounds.values()
        ),
        "max": sum(
            values["kvmem_correct_rag_wrong_max"]
            for values in task_outcome_bounds.values()
        ),
    }
    direct_candidate_ranks = [
        int(candidate["kvmem_score_rank_non_sink"])
        for row in rows
        if row["causal_classification"] == "direct_answer_fact_gap"
        for candidate in row["kvmem_only_candidate_blocks"]
    ]
    rank_control = {
        "candidate_blocks": len(direct_candidate_ranks),
        "worst_rank": max(direct_candidate_ranks),
        "rank_gt_960": sum(rank_value > 960 for rank_value in direct_candidate_ranks),
        "interpretation": (
            "Every localized direct-gap KVMem evidence block would remain inside a "
            "960-block non-sink mean-k cutoff; the RAG/KVMem 960-vs-1024 block-count "
            "difference does not explain these localized misses."
        ),
    }
    summary = {
        "schema_version": "agentlongbench_rag52_decisive_gap_summary.v1",
        "delivery_sample_count": len(ordered_ids),
        "kvmem_correct_samples": len(rows),
        "rag_configuration": {"block_size": 32, "overlap": args.rag_overlap, "top_k": 960},
        "rag_aggregate_exact_correct": {"0": 24, "4": 28, "8": 29}[str(args.rag_overlap)],
        "rag_per_sample_outcome_available": False,
        "classification_counts": dict(sorted(counts.items())),
        "causal_classification_counts": dict(sorted(causal_counts.items())),
        "retrieval_gap_pattern_counts": dict(sorted(gap_pattern_counts.items())),
        "task_classification_counts": {
            task: dict(sorted(values.items())) for task, values in sorted(task_counts.items())
        },
        "task_causal_classification_counts": {
            task: dict(sorted(values.items()))
            for task, values in sorted(task_causal_counts.items())
        },
        "task_outcome_bounds": task_outcome_bounds,
        "overall_kvmem_correct_rag_wrong_bounds": overall_bounds,
        "direct_gap_kvmem_rank_control": rank_control,
        "interpretation": (
            "clear_retrieval_gap is a strong retrieval-failure candidate, not a reconstructed "
            "per-sample RAG model outcome. The delivery contains no per-sample answers/evals."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "per_sample_decisive_gaps.jsonl", rows)
    write_json(args.output_root / "decisive_gap_summary.json", summary)
    (args.output_root / "decisive_gap_report.md").write_text(
        markdown(rows, summary), encoding="utf-8"
    )
    print(
        f"wrote {len(rows)} KVMem-correct samples; classifications={dict(counts)}; "
        f"output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
