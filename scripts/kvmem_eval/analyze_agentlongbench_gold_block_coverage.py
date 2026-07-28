#!/usr/bin/env python3
"""Measure task-grounded evidence coverage in a KVMem selected-block dump.

Two evidence scopes are reported:

* ``answer_evidence`` is the smallest benchmark-derived evidence that directly
  supports the answer.  For an absent item this necessarily expands to the
  complete searched tool result.
* ``verification_scope`` is the evidence needed to verify an exact/global
  claim.  For global counts and argmax this includes all relevant feedback
  records, not just the positive occurrence or winning row.

AgentLongBench's Intersection task has no source-provenance annotation and no
unique minimal proof.  Its conservative answer/verification scope is therefore
all constraint-bearing ``Sections`` feedback.  The output labels this policy
explicitly; it must not be mistaken for a benchmark-provided minimal rationale.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import re
from typing import Any, Iterable

from transformers import AutoTokenizer

import run_agentlongbench_kvmem as runner


DEFAULT_TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/"
    "Qwen3.6-27B-FP8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kvmem-dump", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    return parser.parse_args()


def load_dump(path: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in runner.read_jsonl(path):
        if row.get("type") == "meta":
            if current is not None:
                snapshots.append(current)
            current = {"meta": row, "blocks": []}
        elif current is None:
            raise RuntimeError("dump starts with a block row")
        else:
            current["blocks"].append(row)
    if current is not None:
        snapshots.append(current)
    seen: set[str] = set()
    for snapshot in snapshots:
        meta = snapshot["meta"]
        sid = str(meta.get("trace_tag") or "")
        if not sid or sid in seen:
            raise RuntimeError(f"missing/duplicate dump trace tag: {sid!r}")
        seen.add(sid)
        if meta.get("schema_version") != "qw3_kvmem_retrieval_dump.v2":
            raise RuntimeError(f"unsupported dump schema for {sid}")
        if len(snapshot["blocks"]) != int(meta.get("block_count") or -1):
            raise RuntimeError(f"block-row count mismatch for {sid}")
    return snapshots


def sample_field(sample: dict[str, Any], key: str) -> Any:
    if key in sample:
        return sample[key]
    raw = sample.get("raw")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    raise KeyError(key)


def message_records(canonical: Any, sample: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    marker = "Full conversation history:\n"
    history_start = prompt.index(marker) + len(marker)
    records: list[dict[str, Any]] = []
    cursor = history_start
    messages = sample.get("messages") or (sample.get("raw") or {}).get("messages") or []
    for message_index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "system":
            continue
        line = canonical.message_to_text(message)
        if prompt[cursor : cursor + len(line)] != line:
            raise RuntimeError(
                f"history rendering mismatch at message {message_index}"
            )
        content = str(message.get("content") or "") if isinstance(message, dict) else str(message)
        local_content = line.find(content)
        if local_content < 0:
            raise RuntimeError(f"message content missing from render at {message_index}")
        records.append(
            {
                "message_index": message_index,
                "role": str(message.get("role") or "") if isinstance(message, dict) else "",
                "content": content,
                "line_start": cursor,
                "line_end": cursor + len(line),
                "content_start": cursor + local_content,
                "content_end": cursor + local_content + len(content),
            }
        )
        cursor += len(line) + 1
    return records


def feedback_record(records: list[dict[str, Any]], round_number: int) -> dict[str, Any]:
    prefix = f"Round {round_number}:"
    found = [
        record
        for record in records
        if record["role"] == "user" and record["content"].startswith(prefix)
    ]
    if len(found) != 1:
        raise RuntimeError(
            f"expected one feedback record for round {round_number}, got {len(found)}"
        )
    return found[0]


def preceding_tool(
    records: list[dict[str, Any]], feedback: dict[str, Any]
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if record["message_index"] < feedback["message_index"]
        and record["role"] == "tool"
    ]
    if not candidates:
        raise RuntimeError(
            f"no tool result precedes feedback message {feedback['message_index']}"
        )
    return candidates[-1]


def subspan(
    record: dict[str, Any], local_start: int, local_end: int, label: str
) -> dict[str, Any]:
    return {
        "label": label,
        "prompt_char_start": record["content_start"] + local_start,
        "prompt_char_end": record["content_start"] + local_end,
        "message_index": record["message_index"],
    }


def whole_content(record: dict[str, Any], label: str) -> dict[str, Any]:
    return subspan(record, 0, len(record["content"]), label)


def sections_span(record: dict[str, Any], label: str) -> dict[str, Any]:
    content = record["content"]
    start = content.find("Sections:")
    if start < 0:
        raise RuntimeError(f"feedback has no Sections marker: {label}")
    end = content.find("\nResult:", start)
    if end < 0:
        end = len(content)
    return subspan(record, start, end, label)


def base_stats_span(record: dict[str, Any], label: str) -> dict[str, Any]:
    match = re.search(r"(?m)^  - Base Stats:.*$", record["content"])
    if not match:
        raise RuntimeError(f"feedback has no Base Stats line: {label}")
    return subspan(record, match.start(), match.end(), label)


def exact_occurrences(record: dict[str, Any], needle: str, label: str) -> list[dict[str, Any]]:
    return [
        subspan(record, match.start(), match.end(), f"{label}[{index}]")
        for index, match in enumerate(re.finditer(re.escape(needle), record["content"]))
    ]


def feedback_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["role"] == "user"
        and re.match(r"^Round \d+:", record["content"])
        and "Sections:" in record["content"]
    ]


def evidence_for_sample(
    sample: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    task = str(sample.get("task_type") or sample.get("question_type") or "")
    if task == "Count Frequency(Tool)":
        round_number = int(sample_field(sample, "round"))
        tool = preceding_tool(records, feedback_record(records, round_number))
        target = str(sample_field(sample, "target_name"))
        occurrences = exact_occurrences(tool, target, f"round_{round_number}_target")
        expected = int(sample_field(sample, "answer"))
        if len(occurrences) != expected:
            raise RuntimeError(
                f"target occurrence mismatch: found={len(occurrences)} expected={expected}"
            )
        answer = occurrences or [
            whole_content(tool, f"round_{round_number}_complete_absence_scope")
        ]
        verification = [
            whole_content(tool, f"round_{round_number}_complete_tool_result")
        ]
        return answer, verification, "target occurrences; full result for absence"

    if task == "Find Duplicates(Tool)":
        target = str(sample_field(sample, "pokemon_name"))
        answer: list[dict[str, Any]] = []
        verification: list[dict[str, Any]] = []
        for round_number in (
            int(sample_field(sample, "i_round")),
            int(sample_field(sample, "j_round")),
        ):
            tool = preceding_tool(records, feedback_record(records, round_number))
            occurrences = exact_occurrences(
                tool, target, f"round_{round_number}_target"
            )
            answer.extend(
                occurrences
                or [whole_content(tool, f"round_{round_number}_absence_scope")]
            )
            verification.append(
                whole_content(tool, f"round_{round_number}_complete_tool_result")
            )
        return answer, verification, "membership evidence for both target rounds"

    if task == "Find Target Offsets(Tool)":
        round_number = int(sample_field(sample, "i_round"))
        tool = preceding_tool(records, feedback_record(records, round_number))
        guess = str(sample_field(sample, "guess_name"))
        answers = [str(value) for value in sample_field(sample, "answer")]
        start = tool["content"].find(guess)
        if start < 0:
            raise RuntimeError(
                f"round {round_number} tool result is missing guess {guess!r}"
            )
        cursor = start + len(guess)
        for answer in answers:
            found = tool["content"].find(answer, cursor)
            if found < 0:
                raise RuntimeError(
                    f"round {round_number} result is missing successor "
                    f"{answer!r} after {guess!r}"
                )
            cursor = found + len(answer)
        answer_span = subspan(
            tool,
            start,
            cursor,
            f"round_{round_number}_guess_and_successors",
        )
        return (
            [answer_span],
            [whole_content(tool, f"round_{round_number}_complete_tool_result")],
            "guessed item through the two requested successors",
        )

    if task == "Count Correctness(Env)":
        round_number = int(sample_field(sample, "round"))
        span = sections_span(
            feedback_record(records, round_number),
            f"round_{round_number}_sections",
        )
        return [span], [span], "complete target-round Sections feedback"

    if task == "Count Frequency(Env)":
        target = str(sample_field(sample, "property_value"))
        all_feedback = feedback_records(records)
        answer = [
            span
            for record in all_feedback
            for span in exact_occurrences(record, target, "property_occurrence")
        ]
        expected = int(sample_field(sample, "answer"))
        if len(answer) != expected:
            raise RuntimeError(
                f"feedback occurrence mismatch for {target!r}: "
                f"found={len(answer)} expected={expected}"
            )
        verification = [
            sections_span(record, f"feedback_message_{record['message_index']}")
            for record in all_feedback
        ]
        return answer, verification, "positive occurrences; all feedback for exact count"

    if task == "Find Round with Largest Value(Env)":
        answer_round = int(sample_field(sample, "answer"))
        answer = [
            base_stats_span(
                feedback_record(records, answer_round),
                f"winning_round_{answer_round}_base_stats",
            )
        ]
        verification = [
            base_stats_span(record, f"base_stats_message_{record['message_index']}")
            for record in feedback_records(records)
        ]
        return answer, verification, "winning row; all rows for global argmax"

    if task == "Weighted Summation(Env)":
        rounds = (
            int(sample_field(sample, "i_round")),
            int(sample_field(sample, "j_round")),
        )
        spans = [
            sections_span(
                feedback_record(records, round_number),
                f"round_{round_number}_sections",
            )
            for round_number in rounds
        ]
        return spans, spans, "complete Sections feedback for both scored rounds"

    if task == "Intersection":
        spans = [
            sections_span(record, f"constraint_message_{record['message_index']}")
            for record in feedback_records(records)
        ]
        return (
            spans,
            spans,
            "conservative full constraint corpus; no canonical minimal provenance",
        )

    raise RuntimeError(f"unsupported task type: {task}")


def overlapping_token_positions(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> list[int]:
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end and end > start
    ]


def overlap_length(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def decorate_scope(
    units: list[dict[str, Any]],
    rendered_prompt_start: int,
    offsets: list[tuple[int, int]],
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_blocks = sorted(blocks, key=lambda block: int(block["p0"]))
    decorated: list[dict[str, Any]] = []
    union_gold: set[int] = set()
    union_selected: set[int] = set()
    union_tokens: set[int] = set()
    union_selected_tokens: set[int] = set()
    for unit in units:
        start = rendered_prompt_start + int(unit["prompt_char_start"])
        end = rendered_prompt_start + int(unit["prompt_char_end"])
        tokens = overlapping_token_positions(offsets, start, end)
        if not tokens:
            raise RuntimeError(f"evidence unit mapped to zero tokens: {unit['label']}")
        token_set = set(tokens)
        token_start, token_end = min(tokens), max(tokens) + 1
        gold_blocks = [
            block
            for block in ordered_blocks
            if int(block["p0"]) < token_end
            and int(block["p0"]) + int(block["nt"]) > token_start
        ]
        selected_gold = [
            block for block in gold_blocks if int(block.get("sel") or 0) == 1
        ]
        selected_token_set = {
            token
            for token in tokens
            if any(
                int(block["p0"]) <= token < int(block["p0"]) + int(block["nt"])
                for block in selected_gold
            )
        }
        gold_ids = {int(block["b"]) for block in gold_blocks}
        selected_ids = {int(block["b"]) for block in selected_gold}
        union_gold.update(gold_ids)
        union_selected.update(selected_ids)
        union_tokens.update(token_set)
        union_selected_tokens.update(selected_token_set)
        decorated.append(
            {
                **unit,
                "rendered_token_start": min(tokens),
                "rendered_token_end": max(tokens) + 1,
                "evidence_tokens": len(token_set),
                "selected_evidence_tokens": len(selected_token_set),
                "token_coverage": len(selected_token_set) / len(token_set),
                "gold_block_ids": sorted(gold_ids),
                "selected_gold_block_ids": sorted(selected_ids),
                "gold_blocks": len(gold_ids),
                "selected_gold_blocks": len(selected_ids),
                "block_coverage": len(selected_ids) / len(gold_ids),
                "fully_covered": selected_ids == gold_ids,
            }
        )
    return decorated, {
        "evidence_units": len(decorated),
        "fully_covered_units": sum(unit["fully_covered"] for unit in decorated),
        "gold_blocks": len(union_gold),
        "selected_gold_blocks": len(union_selected),
        "block_coverage": len(union_selected) / len(union_gold) if union_gold else None,
        "gold_tokens": len(union_tokens),
        "selected_gold_tokens": len(union_selected_tokens),
        "token_coverage": (
            len(union_selected_tokens) / len(union_tokens) if union_tokens else None
        ),
    }


def aggregate(rows: Iterable[dict[str, Any]], scope: str) -> dict[str, Any]:
    rows = list(rows)
    metrics = [row[scope]["metrics"] for row in rows]
    gold_blocks = sum(metric["gold_blocks"] for metric in metrics)
    selected_blocks = sum(metric["selected_gold_blocks"] for metric in metrics)
    gold_tokens = sum(metric["gold_tokens"] for metric in metrics)
    selected_tokens = sum(metric["selected_gold_tokens"] for metric in metrics)
    return {
        "samples": len(rows),
        "macro_block_coverage": (
            sum(metric["block_coverage"] for metric in metrics) / len(metrics)
            if metrics else None
        ),
        "micro_block_coverage": selected_blocks / gold_blocks if gold_blocks else None,
        "macro_token_coverage": (
            sum(metric["token_coverage"] for metric in metrics) / len(metrics)
            if metrics else None
        ),
        "micro_token_coverage": selected_tokens / gold_tokens if gold_tokens else None,
        "fully_covered_samples": sum(
            metric["selected_gold_blocks"] == metric["gold_blocks"]
            for metric in metrics
        ),
        "gold_blocks": gold_blocks,
        "selected_gold_blocks": selected_blocks,
        "gold_tokens": gold_tokens,
        "selected_gold_tokens": selected_tokens,
    }


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise RuntimeError("gold-span analysis requires a fast tokenizer")
    canonical = runner.load_canonical_module(args.benchmark_repo)
    samples = {
        str(row.get("stable_sample_id")): row
        for row in runner.read_jsonl(args.dataset)
    }
    snapshots = load_dump(args.kvmem_dump)
    output_rows: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshots, start=1):
        meta = snapshot["meta"]
        sid = str(meta["trace_tag"])
        sample = samples.get(sid)
        if sample is None:
            raise RuntimeError(f"dataset is missing dump sample {sid}")
        prompt = canonical.full_context_prompt(sample)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_start = rendered.find(prompt)
        if prompt_start < 0 or rendered.find(prompt, prompt_start + 1) >= 0:
            raise RuntimeError(f"rendered prompt embedding is ambiguous for {sid}")
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        if len(token_ids) != int(meta["prompt_tokens"]):
            raise RuntimeError(
                f"prompt token parity failure for {sid}: "
                f"local={len(token_ids)} dump={meta['prompt_tokens']}"
            )
        blocks = snapshot["blocks"]
        for block in blocks:
            exported = block.get("tok")
            if int(block.get("sel") or 0) == 1:
                p0, nt = int(block["p0"]), int(block["nt"])
                if exported != token_ids[p0 : p0 + nt]:
                    raise RuntimeError(
                        f"selected-token parity failure for {sid} block={block['b']}"
                    )
        records = message_records(canonical, sample, prompt)
        answer_units, verification_units, policy = evidence_for_sample(
            sample, records
        )
        answer_rows, answer_metrics = decorate_scope(
            answer_units, prompt_start, offsets, blocks
        )
        verification_rows, verification_metrics = decorate_scope(
            verification_units, prompt_start, offsets, blocks
        )
        task = str(sample.get("task_type") or sample.get("question_type") or "")
        output_rows.append(
            {
                "delivery_index": index,
                "stable_sample_id": sid,
                "task_type": task,
                "question": sample_field(sample, "question"),
                "reference_answer": sample_field(sample, "answer"),
                "evidence_policy": policy,
                "prompt_tokens": len(token_ids),
                "selected_blocks": int(meta["selected"]),
                "source_blocks": int(meta["block_count"]),
                "scorer_requested": meta.get("scorer_requested"),
                "scorer_used": meta.get("scorer_used"),
                "scorer_fallback_reason": meta.get("scorer_fallback_reason"),
                "answer_evidence": {
                    "metrics": answer_metrics,
                    "units": answer_rows,
                },
                "verification_scope": {
                    "metrics": verification_metrics,
                    "units": verification_rows,
                },
            }
        )
        print(
            f"[coverage] {index}/{len(snapshots)} {sid} {task} "
            f"answer={answer_metrics['block_coverage']:.3f} "
            f"verify={verification_metrics['block_coverage']:.3f}",
            flush=True,
        )

    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in output_rows:
        by_task[row["task_type"]].append(row)
    summary = {
        "schema_version": "agentlongbench_kvmem_gold_coverage.v1",
        "sample_count": len(output_rows),
        "kvmem_dump": str(args.kvmem_dump),
        "answer_evidence": aggregate(output_rows, "answer_evidence"),
        "verification_scope": aggregate(output_rows, "verification_scope"),
        "by_task": {
            task: {
                "answer_evidence": aggregate(rows, "answer_evidence"),
                "verification_scope": aggregate(rows, "verification_scope"),
            }
            for task, rows in sorted(by_task.items())
        },
        "intersection_provenance_caveat": (
            "The benchmark has no canonical minimal evidence provenance for "
            "Intersection; coverage uses the conservative full constraint corpus."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.output_root / "per_sample_gold_coverage.jsonl"
    with per_sample_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    runner.write_json(args.output_root / "gold_coverage_summary.json", summary)
    print(f"[complete] coverage={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
