#!/usr/bin/env python3
"""Measure LongMemEval-S KVMem query TTFT from a frozen history session."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "kvmem_eval"))

from dataset import load_all  # noqa: E402
from measure_frozen_kvmem_latency import (  # noqa: E402
    append_jsonl,
    post_json,
    timed_completion,
    tokenize,
    wait_for_accounting,
)
from prompt import render_messages  # noqa: E402


def read_selection(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_span(full_ids: list[int], empty_ids: list[int]) -> tuple[int, int]:
    begin = 0
    while begin < min(len(full_ids), len(empty_ids)) and full_ids[begin] == empty_ids[begin]:
        begin += 1
    suffix = 0
    while (
        suffix < len(full_ids) - begin
        and suffix < len(empty_ids) - begin
        and full_ids[-1 - suffix] == empty_ids[-1 - suffix]
    ):
        suffix += 1
    end = len(full_ids) - suffix
    if end <= begin:
        raise RuntimeError("query maps to an empty token span")
    return begin, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-context-budget", type=int, default=32768)
    parser.add_argument("--generation-reserve", type=int, default=32768)
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    selected = read_selection(args.selection)
    wanted = [str(row["question_id"]) for row in selected]
    samples = {sample.question_id: sample for sample in load_all(args.data)}
    missing = [qid for qid in wanted if qid not in samples]
    if missing:
        raise RuntimeError(f"selection IDs missing from dataset: {missing}")
    completed: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "completed":
                    completed.add(str(row.get("question_id")))

    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    endpoint = args.api_base.rstrip("/") + "/completions"
    common = {
        "model": args.model,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "thinking_budget": 4096,
        "seed": 20260722,
    }

    for qid in wanted:
        if qid in completed:
            print(f"[skip] {qid}", flush=True)
            continue
        sample = samples[qid]
        messages = render_messages(sample)
        prefix_messages = messages[:-1]
        empty_messages = [*prefix_messages, {"role": "user", "content": str(messages[-1]["content"]).replace(sample.question, "", 1)}]
        full_rendered = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        message_prefix_rendered = hf_tokenizer.apply_chat_template(
            prefix_messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
        )
        empty_rendered = hf_tokenizer.apply_chat_template(
            empty_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        if not full_rendered.startswith(message_prefix_rendered):
            raise RuntimeError(f"{qid}: chat prefix is not byte-identical")

        full_ids = tokenize(args.api_base, full_rendered, args.timeout)
        empty_ids = tokenize(args.api_base, empty_rendered, args.timeout)
        qb, qe = token_span(full_ids, empty_ids)

        # Use the exact integer-token continuation API and freeze at the nearest
        # physical block boundary before the question.  The measured request
        # consequently replays fewer than B historical tokens and never depends
        # on finding a tokenizer-independent textual newline.
        split_token = (qb // args.block_tokens) * args.block_tokens
        if split_token <= 0:
            raise RuntimeError(f"{qid}: query has no preceding block boundary")
        prefix_ids = full_ids[:split_token]
        finish_ids = full_ids[split_token:]
        if qb < len(prefix_ids):
            raise RuntimeError(f"{qid}: query unexpectedly overlaps frozen history")

        session_id = f"paper-lme-s-{qid}"
        prep_tag = f"paper-lme-s-{qid}-prep"
        final_tag = f"paper-lme-s-{qid}-final"
        prep_started = time.perf_counter()
        prep = post_json(endpoint, {
            **common,
            "prompt": prefix_ids,
            "max_tokens": 0,
            "kvmem_session_id": session_id,
            "kvmem_session_op": "start",
            "kvmem_reselect": "off",
            "kvmem_trace_tag": prep_tag,
        }, args.timeout)
        prep_ms = (time.perf_counter() - prep_started) * 1000.0
        if (prep.get("choices") or [{}])[0].get("finish_reason") != "prefill_only":
            raise RuntimeError(f"{qid}: history ingest was not prefill_only")

        with args.server_log.open("rb") as handle:
            log_start = handle.seek(0, 2)
        result = timed_completion(endpoint, {
            **common,
            "prompt": finish_ids,
            "max_tokens": 1,
            "kvmem_session_id": session_id,
            "kvmem_session_op": "finish",
            "kvmem_reselect": "force",
            "kvmem_query_token_span": {
                "begin": qb - len(prefix_ids),
                "end": qe - len(prefix_ids),
            },
            "kvmem_trace_tag": final_tag,
        }, args.timeout)
        log_text, native = wait_for_accounting(args.server_log, log_start, final_tag)
        query_lines = [line for line in log_text.splitlines() if "native kvmem session query-conditioned" in line]
        if len(query_lines) != 1 or f"span=[{qb},{qe})" not in query_lines[0]:
            raise RuntimeError(f"{qid}: exact query span was not honored: {query_lines}")
        native["kvmem_operation_ms"] = native["post_semantic_ms"] + native["post_query_replay_ms"]
        row = {
            "question_id": qid,
            "question_type": sample.question_type,
            "method": "kvmem",
            "latency_protocol": "final_query_boundary_v2",
            "history_maintenance_excluded": True,
            "history_tokens": len(prefix_ids),
            "query_tokens": qe - qb,
            "full_prompt_tokens": len(full_ids),
            "active_context_budget": args.active_context_budget,
            "generation_reserve": args.generation_reserve,
            "block_tokens": args.block_tokens,
            "prefill_window": "pressure",
            "final_reselect": "force",
            "cache_state": "clean_frozen_raw_token_session",
            "preparation_ingest_ms_excluded": prep_ms,
            **result,
            **native,
            "pre_answer_latency_ms": result["server_ttft_ms"],
            "query_span_tokens": [qb, qe],
            "finish_fragment_tokens": len(finish_ids),
            "alignment_replay_prefix_tokens": qb - len(prefix_ids),
            "trace_tag": final_tag,
            "status": "completed",
        }
        append_jsonl(args.output, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
