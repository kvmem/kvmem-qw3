#!/usr/bin/env python3
"""Run direct-KV AgentLongBench oracle-selection controls.

The experiment keeps the ordinary KVMem mean-K selection and direct historical
KV reuse, but forces the complete task-grounded message chain into the same
fixed selection budget.  Nothing is decoded to text or densely re-prefilled.

This implementation intentionally supports local-evidence tasks only.  For
Count Frequency(Tool) it forces one contiguous chain, and for Find
Duplicates(Tool) it forces both target-round chains:

    assistant tool call -> tool result -> assistant answer -> Round N feedback

The server must be started with ``QW3_KVMEM_ENABLE_ORACLE=1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from transformers import AutoTokenizer

import run_agentlongbench_kvmem as runner
from analyze_agentlongbench_gold_block_coverage import (
    feedback_record,
    load_dump,
    message_records,
    preceding_tool,
)


DEFAULT_TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/"
    "Qwen3.6-27B-FP8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dump-file",
        type=Path,
        help=(
            "Optional QW3_KVMEM_DUMP_SCORES file used to verify forced "
            "selection. Omit for the one-shot inline-refresh path."
        ),
    )
    parser.add_argument(
        "--inline-refresh",
        choices=("kv_only", "kv_and_state"),
        help=(
            "Rebuild the frozen selected context inside the same request; "
            "requires QW3_KVMEM_ENABLE_INLINE_REFRESH=1 on the server."
        ),
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Select only sink + oracle evidence + pinned query-tail blocks.",
    )
    parser.add_argument("--timeout-sec", type=int, default=7200)
    return parser.parse_args()


def read_ids(path: Path, limit: int) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return values[:limit]


def token_span_for_chars(
    offsets: list[tuple[int, int]], begin: int, end: int
) -> tuple[int, int]:
    positions = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > begin and start < end and stop > start
    ]
    if not positions:
        raise RuntimeError(f"character span [{begin},{end}) maps to no tokens")
    return min(positions), max(positions) + 1


def round_message_chain(
    records: list[dict[str, Any]],
    round_number: int,
) -> tuple[int, int, dict[str, Any]]:
    feedback = feedback_record(records, round_number)
    tool = preceding_tool(records, feedback)
    record_pos = {
        int(record["message_index"]): index
        for index, record in enumerate(records)
    }
    tool_pos = record_pos[int(tool["message_index"])]
    feedback_pos = record_pos[int(feedback["message_index"])]
    if tool_pos == 0 or feedback_pos <= tool_pos:
        raise RuntimeError("target tool result has no contiguous message chain")
    chain_start = tool_pos - 1
    if records[chain_start]["role"] != "assistant":
        raise RuntimeError("target tool result is not preceded by an assistant call")
    return (
        int(records[chain_start]["line_start"]),
        int(records[feedback_pos]["line_end"]),
        {
            "round": round_number,
            "message_indices": [
                int(record["message_index"])
                for record in records[chain_start : feedback_pos + 1]
            ],
            "tool_message_index": int(tool["message_index"]),
            "feedback_message_index": int(feedback["message_index"]),
        },
    )


def sample_field(sample: dict[str, Any], key: str) -> Any:
    if key in sample:
        return sample[key]
    raw = sample.get("raw")
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    raise KeyError(key)


def oracle_message_chains(
    sample: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[tuple[int, int, dict[str, Any]]]:
    task = str(sample.get("task_type") or sample.get("question_type") or "")
    if task == "Count Frequency(Tool)":
        rounds = [int(sample_field(sample, "round"))]
    elif task == "Find Duplicates(Tool)":
        rounds = [
            int(sample_field(sample, "i_round")),
            int(sample_field(sample, "j_round")),
        ]
    else:
        raise RuntimeError(
            "direct-KV local oracle supports Count Frequency(Tool) and "
            f"Find Duplicates(Tool), not {task!r}"
        )
    chains = [round_message_chain(records, value) for value in rounds]
    chains.sort(key=lambda item: (item[0], item[1]))
    for previous, current in zip(chains, chains[1:]):
        if current[0] < previous[1]:
            raise RuntimeError("oracle target-round message chains overlap")
    return chains


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def main() -> None:
    args = parse_args()
    if not runner.health_ok(args.api_base):
        raise RuntimeError(f"KVMem server is not healthy at {args.api_base}")
    ids = read_ids(args.ids_file, args.limit)
    samples = {
        str(row.get("stable_sample_id")): row
        for row in runner.read_jsonl(args.dataset)
    }
    missing = [sid for sid in ids if sid not in samples]
    if missing:
        raise RuntimeError(f"dataset is missing {missing[0]}")
    canonical = runner.load_canonical_module(args.benchmark_repo)
    method = (
        "kvmem_k224k_direct_kv_oracle_only_contiguous_chain"
        if args.oracle_only
        else "kvmem_k224k_direct_kv_oracle_contiguous_chain"
    )
    if args.inline_refresh:
        method += f"_inline_{args.inline_refresh}"
    benchmark_name = "AgentLongBench-1M-direct-KV-oracle"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    answers_path = args.output_root / "answers.jsonl"
    eval_path = args.output_root / "eval.jsonl"
    manifest_path = args.output_root / "oracle_manifest.jsonl"
    for path in (answers_path, eval_path, manifest_path):
        if path.exists():
            path.unlink()
    runner.write_json(
        args.output_root / "run_config.json",
        {
            "schema_version": "agentlongbench_direct_kv_oracle.v2",
            "dataset": str(args.dataset),
            "stable_sample_ids": ids,
            "model": args.model,
            "context_window": 1310720,
            "kvmem_context_budget": 229376,
            "kvmem_generation_budget": 32768,
            "kvmem_block_tokens": 32,
            "kvmem_retrieval_method": "mean-k",
            "kvmem_oracle_only": args.oracle_only,
            "kvmem_inline_refresh": args.inline_refresh,
            "kv_dtype": "fp8",
            "query_replay": True,
            "immutable_k": True,
            "immutable_refresh_tokens": 1,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "thinking_budget": 8192,
            "prefill_chunk": 2048,
            "mtp_chain": 4,
            "seed": 20260722,
            "created_at": runner.now_iso(),
        },
    )

    for index, sid in enumerate(ids, start=1):
        sample = samples[sid]
        prompt = canonical.full_context_prompt(sample)
        records = message_records(canonical, sample, prompt)
        char_chains = oracle_message_chains(sample, records)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt_start = rendered.find(prompt)
        if prompt_start < 0 or rendered.find(prompt, prompt_start + 1) >= 0:
            raise RuntimeError("canonical prompt embedding is ambiguous")
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prompt_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        token_spans = [
            token_span_for_chars(
                offsets,
                prompt_start + char_begin,
                prompt_start + char_end,
            )
            for char_begin, char_end, _ in char_chains
        ]
        # /tokenize accepts an already-rendered string.  Sending the canonical
        # user content here would omit chat-template control tokens and produce
        # a small but fatal coordinate offset.
        server_prompt_tokens = runner.tokenize_count(
            args.api_base, rendered, args.timeout_sec
        )
        if server_prompt_tokens != len(prompt_ids):
            raise RuntimeError(
                f"tokenizer parity failed for {sid}: "
                f"local={len(prompt_ids)} server={server_prompt_tokens}"
            )
        query_span = runner.query_byte_span(sample, prompt)
        context_span = runner.history_byte_span(canonical, sample, prompt)
        trace_tag = f"{sid}.oracle"
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "enable_thinking": True,
            "thinking_budget": 8192,
            "seed": 20260722,
            "stream": False,
            "kvmem_query_span": {
                "message_index": 0,
                "content_start": query_span[0],
                "content_end": query_span[1],
            },
            "kvmem_context_span": {
                "message_index": 0,
                "content_start": context_span[0],
                "content_end": context_span[1],
            },
            "kvmem_trace_tag": trace_tag,
            "kvmem_oracle_token_spans": [
                {"begin": begin, "end": end}
                for begin, end in token_spans
            ],
        }
        if args.oracle_only:
            payload["kvmem_oracle_only"] = True
        if args.inline_refresh:
            payload["kvmem_inline_refresh"] = args.inline_refresh
        print(
            f"[oracle] {index}/{len(ids)} {sid} "
            f"prompt={len(prompt_ids)} spans={token_spans} "
            f"tokens={sum(end-begin for begin, end in token_spans)}",
            flush=True,
        )
        started = time.perf_counter()
        response = runner.post_json(
            args.api_base.rstrip("/") + "/chat/completions",
            payload,
            args.timeout_sec,
        )
        elapsed = time.perf_counter() - started
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"oracle request returned no choice for {sid}")
        choice = choices[0]
        message = choice.get("message") or {}
        raw_response = str(message.get("content") or "")
        oracle_blocks: list[dict[str, Any]] = []
        selected_blocks: list[dict[str, Any]] = []
        oracle_selected_verified = False
        if args.dump_file is not None:
            snapshots = {
                str(snapshot["meta"].get("trace_tag") or ""): snapshot
                for snapshot in load_dump(args.dump_file)
            }
            if trace_tag not in snapshots:
                raise RuntimeError(
                    f"retrieval dump has no completed oracle snapshot for {trace_tag}"
                )
            oracle_blocks = [
                block
                for block in snapshots[trace_tag]["blocks"]
                if any(
                    int(block["p0"]) < end
                    and int(block["p0"]) + int(block["nt"]) > begin
                    for begin, end in token_spans
                )
            ]
            if not oracle_blocks:
                raise RuntimeError(
                    f"oracle span overlaps no materialized block for {sid}"
                )
            missing_oracle_blocks = [
                int(block["b"])
                for block in oracle_blocks
                if int(block.get("sel") or 0) != 1
            ]
            if missing_oracle_blocks:
                raise RuntimeError(
                    f"oracle blocks were not selected for {sid}: "
                    f"{missing_oracle_blocks[:16]}"
                )
            selected_blocks = [
                block
                for block in snapshots[trace_tag]["blocks"]
                if int(block.get("sel") or 0) == 1
            ]
            if args.oracle_only:
                meta = snapshots[trace_tag]["meta"]
                block_tokens = int(meta["block_tokens"])
                sink_blocks = int(meta["sink"])
                pin_from_block = int(meta["query_begin"]) // block_tokens
                unexpected = [
                    int(block["b"])
                    for block in selected_blocks
                    if int(block["b"]) >= sink_blocks
                    and not (
                        any(
                            int(block["p0"]) < end
                            and int(block["p0"]) + int(block["nt"]) > begin
                            for begin, end in token_spans
                        )
                    )
                    and int(block["b"]) < pin_from_block
                ]
                if unexpected:
                    raise RuntimeError(
                        "oracle-only selection contains ordinary retrieval "
                        f"blocks for {sid}: {unexpected[:16]}"
                    )
            oracle_selected_verified = True
        evaluated = canonical.evaluate_response(sample, raw_response)
        answer = {
            **runner.base_row(
                index,
                len(ids),
                sample,
                method,
                benchmark_name,
            ),
            "hypothesis": raw_response,
            "raw_response": raw_response,
            "finish_reason": choice.get("finish_reason"),
            "elapsed_sec": elapsed,
            "answered_at": runner.now_iso(),
        }
        eval_row = {
            **{
                key: answer.get(key)
                for key in (
                    "benchmark",
                    "method",
                    "index",
                    "stable_sample_id",
                    "question_type",
                    "setting",
                    "target_length",
                    "actual_length",
                    "task_type",
                    "source_path",
                    "finish_reason",
                )
            },
            **evaluated,
            "evaluated_at": runner.now_iso(),
        }
        append_jsonl(answers_path, answer)
        append_jsonl(eval_path, eval_row)
        append_jsonl(
            manifest_path,
            {
                "schema_version": "agentlongbench_direct_kv_oracle.v1",
                "stable_sample_id": sid,
                "trace_tag": trace_tag,
                "prompt_tokens": len(prompt_ids),
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "oracle_token_spans": [list(span) for span in token_spans],
                "oracle_span_tokens": sum(
                    end - begin for begin, end in token_spans
                ),
                "oracle_block_count_32": sum(
                    (end - 1) // 32 - begin // 32 + 1
                    for begin, end in token_spans
                ),
                "oracle_selected_verified": oracle_selected_verified,
                "oracle_selected_block_count": (
                    len(oracle_blocks) if oracle_selected_verified else None
                ),
                "oracle_only": args.oracle_only,
                "total_selected_block_count": (
                    len(selected_blocks) if oracle_selected_verified else None
                ),
                "inline_refresh": args.inline_refresh,
                "chains": [chain for _, _, chain in char_chains],
                "elapsed_sec": elapsed,
                "correct": eval_row.get("correct"),
                "score": eval_row.get("score"),
            },
        )
        print(
            f"[oracle-result] {index}/{len(ids)} correct="
            f"{eval_row.get('correct')} score={eval_row.get('score')} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    evals = runner.read_jsonl(eval_path)
    runner.write_json(
        args.output_root / "summary.json",
        {
            "schema_version": "agentlongbench_direct_kv_oracle_summary.v1",
            "samples": len(evals),
            "correct": sum(row.get("correct") is True for row in evals),
            "accuracy": (
                sum(row.get("correct") is True for row in evals) / len(evals)
                if evals
                else None
            ),
            "selection": (
                "sink plus contiguous gold message chain plus pinned query tail"
                if args.oracle_only
                else "mean-k plus mandatory contiguous gold message chain"
            ),
            "representation": (
                "direct historical K/V; no text re-prefill"
                if not args.inline_refresh
                else (
                    "inline compact selected-context refresh: "
                    + args.inline_refresh
                )
            ),
            "completed_at": runner.now_iso(),
        },
    )
    runner.write_final(
        args.output_root,
        [samples[sid] for sid in ids],
        method,
        benchmark_name,
    )
    print(f"[complete] output={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
