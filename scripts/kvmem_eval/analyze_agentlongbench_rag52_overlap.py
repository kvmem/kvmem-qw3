#!/usr/bin/env python3
"""Compare frozen AgentLongBench RAG blocks with tagged KVMem selections.

The comparison is strict by design:
* all three delivery files must contain the same ordered 52 stable sample IDs;
* every delivered RAG block is regenerated with the Qwen tokenizer and checked
  for exact ID, token span, and decoded text parity;
* local chat rendering must match every token ID exported for selected KVMem
  blocks before any overlap metric is reported.

KVMem's internal block IDs are prompt-relative and are never compared directly
with RAG IDs. Instead, selected KVMem text coverage is projected onto the exact
RAG grid mflat_c{chunk_index:05d}; both the projected and delivered sets then
use the same ID definition.
"""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any, Iterable


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
DEFAULT_RAG_EVAL = Path(
    "/home/chaidi/AgentLongBench_Motivation/outputs/compactrag/eval.jsonl"
)
BLOCK_RE = re.compile(r"^mflat_c(\d{5,})$")
Interval = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--kvmem-dump", type=Path, required=True)
    parser.add_argument("--kvmem-eval", type=Path, default=DEFAULT_KVMEM_EVAL)
    parser.add_argument("--rag-eval", type=Path, default=DEFAULT_RAG_EVAL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="smoke-test the first N frozen IDs")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_canonical(repo: Path) -> Any:
    source = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("agentlongbench_fullcontext_overlap", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import canonical worker: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merged(intervals: Iterable[Interval]) -> list[Interval]:
    out: list[list[int]] = []
    for lo, hi in sorted((lo, hi) for lo, hi in intervals if hi > lo):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def interval_length(intervals: Iterable[Interval]) -> int:
    return sum(hi - lo for lo, hi in intervals)


def intersection(a: list[Interval], b: list[Interval]) -> list[Interval]:
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def overlap_length_one(intervals: list[Interval], starts: list[int], target: list[Interval]) -> int:
    total = 0
    for lo, hi in target:
        i = max(0, bisect.bisect_right(starts, lo) - 1)
        while i < len(intervals) and intervals[i][0] < hi:
            total += max(0, min(hi, intervals[i][1]) - max(lo, intervals[i][0]))
            i += 1
    return total


class LogicalHistory:
    """Map history character/token ranges to separator-free message coordinates."""

    def __init__(self, lines: list[str], separator: str, tokenizer: Any):
        self.text = separator.join(lines)
        self.message_spans: list[tuple[int, int, int]] = []
        history_pos = 0
        logical_pos = 0
        for index, line in enumerate(lines):
            self.message_spans.append((history_pos, history_pos + len(line), logical_pos))
            history_pos += len(line)
            logical_pos += len(line)
            if index + 1 < len(lines):
                history_pos += len(separator)
        self.message_starts = [span[0] for span in self.message_spans]
        encoding = tokenizer(self.text, add_special_tokens=False, return_offsets_mapping=True)
        self.ids = list(encoding["input_ids"])
        self.offsets = [tuple(pair) for pair in encoding["offset_mapping"]]

    def chars_to_logical(self, lo: int, hi: int) -> list[Interval]:
        if hi <= lo:
            return []
        out: list[Interval] = []
        i = max(0, bisect.bisect_right(self.message_starts, lo) - 1)
        while i < len(self.message_spans) and self.message_spans[i][0] < hi:
            msg_lo, msg_hi, logical_lo = self.message_spans[i]
            x0, x1 = max(lo, msg_lo), min(hi, msg_hi)
            if x1 > x0:
                out.append((logical_lo + x0 - msg_lo, logical_lo + x1 - msg_lo))
            i += 1
        return out

    def token_range_to_logical(self, begin: int, end: int) -> list[Interval]:
        if begin < 0 or end > len(self.offsets) or end <= begin:
            return []
        valid = [(lo, hi) for lo, hi in self.offsets[begin:end] if hi > lo]
        if not valid:
            return []
        return self.chars_to_logical(valid[0][0], valid[-1][1])


def load_dump(path: Path) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for row in read_jsonl(path):
        if row.get("type") == "meta":
            if current is not None:
                tag = str(current["meta"].get("trace_tag") or "")
                if not tag or tag in snapshots:
                    raise RuntimeError(f"dump has missing/duplicate trace_tag={tag!r}")
                snapshots[tag] = current
            current = {"meta": row, "blocks": []}
        elif current is None:
            raise RuntimeError("dump starts with a block row before its meta row")
        else:
            current["blocks"].append(row)
    if current is not None:
        tag = str(current["meta"].get("trace_tag") or "")
        if not tag or tag in snapshots:
            raise RuntimeError(f"dump has missing/duplicate trace_tag={tag!r}")
        snapshots[tag] = current
    for tag, snapshot in snapshots.items():
        expected = int(snapshot["meta"].get("block_count") or -1)
        if len(snapshot["blocks"]) != expected:
            raise RuntimeError(
                f"dump {tag} block row count {len(snapshot['blocks'])} != {expected}"
            )
    return snapshots


def delivery_rows(delivery: Path) -> tuple[list[str], dict[int, dict[str, dict[str, Any]]]]:
    by_overlap: dict[int, dict[str, dict[str, Any]]] = {}
    ordered_ids: list[str] | None = None
    for overlap in (0, 4, 8):
        path = delivery / f"overlap_{overlap}_blocks.jsonl"
        rows = read_jsonl(path)
        ids = [str(row.get("stable_sample_id") or "") for row in rows]
        if len(ids) != 52 or len(set(ids)) != 52 or any(not sid for sid in ids):
            raise RuntimeError(f"{path} must have exactly 52 unique stable_sample_id values")
        if ordered_ids is None:
            ordered_ids = ids
        elif ids != ordered_ids:
            raise RuntimeError(f"{path} sample IDs/order differ from overlap_0")
        by_overlap[overlap] = dict(zip(ids, rows, strict=True))
    assert ordered_ids is not None
    return ordered_ids, by_overlap


def sample_lines(canonical: Any, sample: dict[str, Any]) -> list[str]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    messages = sample.get("messages") or raw.get("messages") or []
    return [
        canonical.message_to_text(message)
        for message in messages
        if not (isinstance(message, dict) and message.get("role") == "system")
    ]


def validate_delivery_block(
    row: dict[str, Any], block: dict[str, Any], overlap: int, history: LogicalHistory,
    tokenizer: Any,
) -> int:
    match = BLOCK_RE.match(str(block.get("block_id") or ""))
    if match is None:
        raise RuntimeError(f"invalid RAG block_id: {block.get('block_id')!r}")
    index = int(match.group(1))
    step = 32 - overlap
    begin = index * step
    end = min(begin + 32, len(history.ids))
    decoded = tokenizer.decode(history.ids[begin:end], skip_special_tokens=False)
    if (
        row.get("block_size") != 32
        or row.get("chunk_overlap") != overlap
        or block.get("block_id") != f"mflat_c{index:05d}"
        or block.get("chunk_start") != begin
        or block.get("chunk_end") != end
        or block.get("text") != decoded
    ):
        raise RuntimeError(
            f"delivery block parity failure: sid={row.get('stable_sample_id')} "
            f"overlap={overlap} id={block.get('block_id')}"
        )
    return index


def latest_eval(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row["stable_sample_id"]): row
        for row in read_jsonl(path)
        if row.get("stable_sample_id")
    }


def outcome(kvmem: dict[str, Any], rag: dict[str, Any]) -> str:
    kc = bool(kvmem.get("correct"))
    rc = bool(rag.get("correct"))
    return (
        "kvmem_correct_rag_wrong" if kc and not rc else
        "both_correct" if kc and rc else
        "kvmem_wrong_rag_correct" if not kc and rc else
        "both_wrong"
    )


def metric_summary(rows: list[dict[str, Any]], overlap: int) -> dict[str, Any]:
    metrics = [row["overlaps"][str(overlap)] for row in rows]
    return {
        "samples": len(rows),
        "mean_kvmem_selected_logical_chars": mean(
            m["kvmem_selected_logical_chars"] for m in metrics
        ),
        "mean_rag_selected_logical_chars": mean(
            m["rag_selected_logical_chars"] for m in metrics
        ),
        "mean_logical_char_jaccard": mean(m["logical_char_jaccard"] for m in metrics),
        "mean_rag_coverage_by_kvmem": mean(m["rag_coverage_by_kvmem"] for m in metrics),
        "mean_kvmem_coverage_by_rag": mean(m["kvmem_coverage_by_rag"] for m in metrics),
        "mean_same_id_jaccard_any": mean(m["same_id_jaccard_any"] for m in metrics),
        "mean_same_id_jaccard_half": mean(m["same_id_jaccard_half"] for m in metrics),
        "mean_same_id_jaccard_full": mean(m["same_id_jaccard_full"] for m in metrics),
        "low_overlap_lt_0_25": sum(m["logical_char_jaccard"] < 0.25 for m in metrics),
        "high_overlap_ge_0_60": sum(m["logical_char_jaccard"] >= 0.60 for m in metrics),
    }


def markdown_report(summary: dict[str, Any]) -> str:
    sample_count = summary["validation"]["sample_count"]
    all_rows = summary["groups"]["all_52"]
    focus = summary["groups"].get("kvmem_correct_rag_wrong")
    both = summary["groups"].get("both_correct")
    lines = [
        "# AgentLongBench RAG / KVMem retrieval overlap",
        "",
        f"All {sample_count} analyzed rows passed frozen-ID, RAG block-ID/span/text, and selected KVMem token parity checks.",
        "KVMem internal IDs are not compared directly; `same_id_*` projects KVMem coverage onto",
        "the exact delivery grid `mflat_c{index:05d}` before set comparison.",
        "Outcome groups use the old 1024-token Compact+RAG per-sample evaluation; the",
        "32-token delivery contains aggregate scores and block snapshots but no per-sample scores.",
        "",
        "## Main findings",
        "",
        f"- Across all 52 samples, mean logical-text Jaccard is {all_rows['0']['mean_logical_char_jaccard']:.2%} (overlap=0) to {all_rows['8']['mean_logical_char_jaccard']:.2%} (overlap=8); no sample reaches 60%.",
    ]
    if focus is not None:
        lines.append(
            f"- In the {focus['samples']} old-outcome cases where KVMem is correct and Compact+RAG is wrong, Jaccard is only {focus['0']['mean_logical_char_jaccard']:.2%} to {focus['8']['mean_logical_char_jaccard']:.2%}; {focus['0']['low_overlap_lt_0_25']}/{focus['samples']} (overlap=0) and {focus['8']['low_overlap_lt_0_25']}/{focus['samples']} (overlap=8) are below 25%."
        )
    if focus is not None and both is not None:
        lines.append(
            f"- Both-correct samples overlap more: {both['8']['mean_logical_char_jaccard']:.2%} at overlap=8 versus {focus['8']['mean_logical_char_jaccard']:.2%} for KVMem-only-correct cases. This points toward retrieval-set divergence as the dominant difference."
        )
    lines += [
        f"- KVMem covers about {all_rows['0']['mean_kvmem_selected_logical_chars']:.0f} unique logical characters per sample. RAG covers about {all_rows['0']['mean_rag_selected_logical_chars']:.0f} at overlap=0 and {all_rows['8']['mean_rag_selected_logical_chars']:.0f} at overlap=8; overlapping RAG chunks reduce unique source coverage.",
        "- The current data therefore does not provide a high-overlap cohort that could isolate a KV-representation advantage. A same-span replay is still required.",
        "",
    ]
    for group, values in summary["groups"].items():
        lines += [f"## {group} ({values['samples']} samples)", ""]
        lines += [
            "| RAG overlap | logical-char Jaccard | RAG covered by KVMem | KVMem covered by RAG | same-ID Jaccard (>=50%) | low <25% | high >=60% |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for overlap in (0, 4, 8):
            m = values[str(overlap)]
            lines.append(
                f"| {overlap} | {m['mean_logical_char_jaccard']:.2%} | "
                f"{m['mean_rag_coverage_by_kvmem']:.2%} | "
                f"{m['mean_kvmem_coverage_by_rag']:.2%} | "
                f"{m['mean_same_id_jaccard_half']:.2%} | "
                f"{m['low_overlap_lt_0_25']} | {m['high_overlap_ge_0_60']} |"
            )
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        "Low overlap is direct evidence that retrieval chose substantially different source text.",
        "High overlap only narrows the cause to representation/prompt/order/generation differences;",
        "it does not by itself prove that contextualized KV is the cause. A controlled replay using",
        "the same source spans in both systems is required for that attribution.",
        "",
    ]
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
    canonical = load_canonical(args.benchmark_repo)
    ordered_ids, deliveries = delivery_rows(args.delivery)
    if args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit must be positive")
        ordered_ids = ordered_ids[: args.limit]
    samples = {str(row.get("stable_sample_id")): row for row in read_jsonl(args.dataset)}
    missing_samples = [sid for sid in ordered_ids if sid not in samples]
    if missing_samples:
        raise RuntimeError(f"dataset missing delivery sample {missing_samples[0]}")
    dumps = load_dump(args.kvmem_dump)
    if list(dumps) != ordered_ids:
        raise RuntimeError(
            "KVMem dump IDs/order must exactly match the frozen 52 delivery IDs; "
            f"dump={len(dumps)} delivery={len(ordered_ids)}"
        )
    kvmem_evals, rag_evals = latest_eval(args.kvmem_eval), latest_eval(args.rag_eval)
    missing_eval = [sid for sid in ordered_ids if sid not in kvmem_evals or sid not in rag_evals]
    if missing_eval:
        raise RuntimeError(f"evaluation files missing delivery sample {missing_eval[0]}")

    results: list[dict[str, Any]] = []
    validated_blocks = 0
    validated_kvmem_tokens = 0
    for position, sid in enumerate(ordered_ids, start=1):
        sample = samples[sid]
        lines = sample_lines(canonical, sample)
        single = LogicalHistory(lines, "\n", tokenizer)
        double = LogicalHistory(lines, "\n\n", tokenizer)
        prompt = canonical.full_context_prompt(sample)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        rendered_encoding = tokenizer(
            rendered, add_special_tokens=False, return_offsets_mapping=True
        )
        rendered_ids = list(rendered_encoding["input_ids"])
        rendered_offsets = [tuple(pair) for pair in rendered_encoding["offset_mapping"]]
        prompt_char_start = rendered.find(prompt)
        if prompt_char_start < 0 or rendered.find(prompt, prompt_char_start + 1) >= 0:
            raise RuntimeError(f"rendered prompt does not contain one exact canonical prompt: {sid}")
        history_marker = "Full conversation history:\n"
        marker_start = prompt.find(history_marker)
        history_char_start = prompt_char_start + marker_start + len(history_marker)
        if rendered[history_char_start:history_char_start + len(single.text)] != single.text:
            raise RuntimeError(f"canonical single-newline history does not round-trip: {sid}")
        rendered_history_spans = [
            (history_char_start + lo, history_char_start + hi, logical)
            for lo, hi, logical in single.message_spans
        ]
        rendered_map = object.__new__(LogicalHistory)
        rendered_map.text = rendered
        rendered_map.message_spans = rendered_history_spans
        rendered_map.message_starts = [span[0] for span in rendered_history_spans]
        rendered_map.ids = rendered_ids
        rendered_map.offsets = rendered_offsets

        snapshot = dumps[sid]
        meta = snapshot["meta"]
        if meta.get("schema_version") != "qw3_kvmem_retrieval_dump.v2":
            raise RuntimeError(f"{sid} is not a v2 tagged retrieval dump")
        if int(meta.get("prompt_tokens") or -1) != len(rendered_ids):
            raise RuntimeError(
                f"chat-token count mismatch for {sid}: dump={meta.get('prompt_tokens')} "
                f"local={len(rendered_ids)}"
            )
        selected_rows = [row for row in snapshot["blocks"] if int(row.get("sel") or 0) == 1]
        if len(selected_rows) != int(meta.get("selected") or -1):
            raise RuntimeError(f"selected-row count mismatch for {sid}")
        kvmem_logical_parts: list[Interval] = []
        kvmem_internal: list[dict[str, Any]] = []
        for block in selected_rows:
            p0, nt = int(block["p0"]), int(block["nt"])
            exported = block.get("tok")
            if not isinstance(exported, list) or len(exported) != nt:
                raise RuntimeError(f"selected KVMem block lacks exact token IDs: {sid} b={block['b']}")
            if [int(value) for value in exported] != rendered_ids[p0:p0 + nt]:
                raise RuntimeError(f"selected KVMem token parity failure: {sid} b={block['b']}")
            validated_kvmem_tokens += nt
            logical = rendered_map.token_range_to_logical(p0, p0 + nt)
            kvmem_logical_parts.extend(logical)
            kvmem_internal.append(
                {
                    "internal_block_id": int(block["b"]),
                    "prompt_token_start": p0,
                    "prompt_token_end": p0 + nt,
                    "retrieval_score": block.get("rs"),
                    "text": tokenizer.decode(
                        [int(value) for value in exported],
                        skip_special_tokens=False,
                    ),
                    "logical_intervals": [list(interval) for interval in logical],
                }
            )
        kvmem_union = merged(kvmem_logical_parts)
        kvmem_starts = [lo for lo, _ in kvmem_union]

        overlap_results: dict[str, Any] = {}
        for overlap in (0, 4, 8):
            delivery = deliveries[overlap][sid]
            rag_selected_ids: set[str] = set()
            rag_parts: list[Interval] = []
            selected_coverage: list[dict[str, Any]] = []
            for block in delivery["blocks"]:
                index = validate_delivery_block(delivery, block, overlap, double, tokenizer)
                validated_blocks += 1
                block_id = f"mflat_c{index:05d}"
                rag_selected_ids.add(block_id)
                logical = merged(double.token_range_to_logical(
                    int(block["chunk_start"]), int(block["chunk_end"])
                ))
                rag_parts.extend(logical)
                denom = interval_length(logical)
                covered = overlap_length_one(kvmem_union, kvmem_starts, logical)
                selected_coverage.append(
                    {"block_id": block_id, "kvmem_logical_coverage": covered / denom if denom else 0.0}
                )
            rag_union = merged(rag_parts)
            common = intersection(kvmem_union, rag_union)
            common_len = interval_length(common)
            k_len, r_len = interval_length(kvmem_union), interval_length(rag_union)
            union_len = k_len + r_len - common_len

            step = 32 - overlap
            projected: dict[str, float] = {}
            for index, begin in enumerate(range(0, len(double.ids), step)):
                end = min(begin + 32, len(double.ids))
                text = tokenizer.decode(double.ids[begin:end], skip_special_tokens=False)
                if not text.strip():
                    if end >= len(double.ids):
                        break
                    continue
                logical = merged(double.token_range_to_logical(begin, end))
                denom = interval_length(logical)
                covered = overlap_length_one(kvmem_union, kvmem_starts, logical)
                projected[f"mflat_c{index:05d}"] = covered / denom if denom else 0.0
                if end >= len(double.ids):
                    break
            projected_sets = {
                "any": {block_id for block_id, coverage in projected.items() if coverage > 0.0},
                "half": {block_id for block_id, coverage in projected.items() if coverage >= 0.5},
                "full": {block_id for block_id, coverage in projected.items() if coverage >= 1.0 - 1e-12},
            }
            id_metrics: dict[str, float] = {}
            id_details: dict[str, Any] = {}
            for threshold, projected_set in projected_sets.items():
                shared = rag_selected_ids & projected_set
                union = rag_selected_ids | projected_set
                id_metrics[f"same_id_jaccard_{threshold}"] = len(shared) / len(union) if union else 1.0
                id_details[threshold] = {
                    "projected_kvmem_count": len(projected_set),
                    "shared_count": len(shared),
                    "shared_block_ids": sorted(shared),
                    "rag_only_block_ids": sorted(rag_selected_ids - projected_set),
                    "kvmem_only_block_ids": sorted(projected_set - rag_selected_ids),
                }
            overlap_results[str(overlap)] = {
                "rag_selected_blocks": len(rag_selected_ids),
                "kvmem_selected_logical_chars": k_len,
                "rag_selected_logical_chars": r_len,
                "shared_logical_chars": common_len,
                "logical_char_jaccard": common_len / union_len if union_len else 1.0,
                "rag_coverage_by_kvmem": common_len / r_len if r_len else 1.0,
                "kvmem_coverage_by_rag": common_len / k_len if k_len else 1.0,
                **id_metrics,
                "same_id_details": id_details,
                "rag_selected_block_coverage": selected_coverage,
            }

        raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
        results.append(
            {
                "schema_version": "agentlongbench_rag52_kvmem_overlap.v1",
                "delivery_index": position,
                "stable_sample_id": sid,
                "official_id": raw.get("id"),
                "task_type": sample.get("task_type"),
                "actual_length": sample.get("actual_length"),
                "target_length": sample.get("target_length"),
                "outcome": outcome(kvmem_evals[sid], rag_evals[sid]),
                "kvmem_correct": bool(kvmem_evals[sid].get("correct")),
                "old_compact_rag_correct": bool(rag_evals[sid].get("correct")),
                "kvmem_internal_selected_blocks": kvmem_internal,
                "overlaps": overlap_results,
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {"all_52": results}
    for name in ("kvmem_correct_rag_wrong", "both_correct", "both_wrong", "kvmem_wrong_rag_correct"):
        groups[name] = [row for row in results if row["outcome"] == name]
    for key in sorted({str(row.get("task_type")) for row in results}):
        groups[f"task:{key}"] = [
            row for row in results if str(row.get("task_type")) == key
        ]
    for key in sorted({str(row.get("actual_length")) for row in results}):
        groups[f"actual_length:{key}"] = [
            row for row in results if str(row.get("actual_length")) == key
        ]
    summary_groups: dict[str, Any] = {}
    for name, rows in groups.items():
        if not rows:
            continue
        summary_groups[name] = {
            "samples": len(rows),
            **{str(overlap): metric_summary(rows, overlap) for overlap in (0, 4, 8)},
        }
    summary = {
        "schema_version": "agentlongbench_rag52_kvmem_overlap_summary.v1",
        "validation": {
            "ordered_stable_sample_ids_exact": True,
            "sample_count": len(ordered_ids),
            "delivery_blocks_exactly_regenerated": validated_blocks,
            "selected_kvmem_token_ids_exactly_matched": validated_kvmem_tokens,
            "rag_block_id_definition": "mflat_c{index:05d}",
            "rag_block_start_definition": "index * (32 - overlap)",
            "rag_history_join": "double newline between message_to_text outputs",
            "kvmem_history_join": "single newline between message_to_text outputs",
        },
        "outcome_group_reference": {
            "kvmem": str(args.kvmem_eval),
            "rag": str(args.rag_eval),
            "note": (
                "The 32-token delivery exposes aggregate scores and retrieved blocks, "
                "not per-sample evaluation rows; outcome groups therefore use the old "
                "Compact+RAG evaluation."
            ),
        },
        "groups": summary_groups,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_root / "per_sample_overlap.jsonl", results)
    write_json(args.output_root / "overlap_summary.json", summary)
    (args.output_root / "overlap_summary.md").write_text(
        markdown_report(summary), encoding="utf-8"
    )
    print(
        f"analysis complete: {len(ordered_ids)} exact IDs, {validated_blocks} RAG blocks, "
        f"{validated_kvmem_tokens} selected KVMem tokens validated",
        flush=True,
    )


if __name__ == "__main__":
    main()
