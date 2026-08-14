#!/usr/bin/env python3
"""Measure LongMemEval-S TTFT from an already-resident active window.

The benchmark history is rendered with the same LongMemEval prompt vocabulary
as KVMem_Motivation.  Only the newest history suffix that fits beside the task
instruction and final question is retained. Window fitting and history prefill
are pre-query state; ``pre_answer_latency_sec`` measures only the final-query
continuation through the first output token. It excludes answer decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MOTIVATION = Path("/home/chaidi/kvmem_eval/KVMem_Motivation")
sys.path.insert(0, str(ROOT / "scripts" / "kvmem_eval"))
sys.path.insert(0, str(MOTIVATION))

from client import Qw3Client  # noqa: E402
from scripts.common.dataset import full_history_text  # noqa: E402
from scripts.common.prompts import SYSTEM_INSTRUCTION  # noqa: E402


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise RuntimeError(f"expected a JSON array/JSONL file: {path}")
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prompt_parts(sample: dict[str, Any]) -> tuple[str, str, str]:
    prefix = (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question date: {sample.get('question_date', '')}\n\n"
        "Recent history window:\n"
    )
    suffix = (
        f"\n\nQuestion: {sample.get('question', '')}\n"
        "Answer:"
    )
    return prefix, full_history_text(sample), suffix


def render_chat_tokens(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # Transformers 5 returns a BatchEncoding by default for some fast
    # tokenizers, whereas older releases returned the input-id list directly.
    # Taking len(BatchEncoding) counts fields (normally two), not tokens.
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif hasattr(encoded, "get"):
        encoded = encoded.get("input_ids")
    if encoded is None:
        raise RuntimeError("chat template did not return input_ids")
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError("unexpected batched chat-template result")
        encoded = encoded[0]
    return list(encoded)


def fit_prompt(tokenizer: Any, sample: dict[str, Any], cap: int) -> tuple[str, dict[str, Any]]:
    prefix, history, suffix = prompt_parts(sample)
    history_ids = tokenizer.encode(history, add_special_tokens=False)
    fixed_tokens = len(render_chat_tokens(tokenizer, prefix + suffix))
    keep = max(0, min(len(history_ids), cap - fixed_tokens))
    # Decoding a suffix can alter the first token at the byte boundary.  Fit
    # against the fully rendered chat prompt and trim until the invariant is
    # exact rather than relying on additive token counts.
    while True:
        history_tail = tokenizer.decode(history_ids[-keep:]) if keep else ""
        prompt = prefix + history_tail + suffix
        actual = len(render_chat_tokens(tokenizer, prompt))
        if actual <= cap:
            break
        keep = max(0, keep - max(1, actual - cap))
    # Fill any small boundary slack without ever crossing the cap.
    lo, hi = keep, len(history_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = prefix + tokenizer.decode(history_ids[-mid:]) + suffix
        if len(render_chat_tokens(tokenizer, candidate)) <= cap:
            lo = mid
        else:
            hi = mid - 1
    keep = lo
    history_tail = tokenizer.decode(history_ids[-keep:]) if keep else ""
    prompt = prefix + history_tail + suffix
    actual = len(render_chat_tokens(tokenizer, prompt))
    if actual > cap:
        raise RuntimeError(f"window fit failed: actual={actual} cap={cap}")
    return prompt, {
        "active_prompt_tokens_local": actual,
        "full_history_tokens_local": len(history_ids),
        "selected_history_tokens_local": keep,
        "truncated": keep < len(history_ids),
    }


def warm_prefix(prompt: str) -> str:
    marker = "\n\nQuestion:"
    position = prompt.rfind(marker)
    if position < 0:
        raise RuntimeError("LongMemEval sliding prompt lacks final Question section")
    return prompt[:position] + marker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18126/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-FP8")
    parser.add_argument("--active-cap", type=int, default=32768)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    samples = {str(row["question_id"]): row for row in read_rows(args.data)}
    selected = read_rows(args.selection)
    selected_ids = [str(row["question_id"]) for row in selected]
    wanted = list(args.question_id) if args.question_id else selected_ids
    unknown = [qid for qid in wanted if qid not in selected_ids]
    if unknown:
        raise RuntimeError(f"question IDs not present in selection: {unknown}")
    missing = [qid for qid in wanted if qid not in samples]
    if missing:
        raise RuntimeError(f"selection IDs missing from dataset: {missing}")
    completed = {
        str(row.get("question_id"))
        for row in (read_rows(args.output) if args.output.exists() else [])
        if row.get("status") == "completed"
    }
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    client = Qw3Client(
        base_url=args.api_base,
        model=args.model,
        temperature=0.0,
        top_p=0.95,
        max_tokens=1,
        enable_thinking=True,
        read_timeout=args.timeout,
    )
    if not client.health():
        raise RuntimeError(f"server is not healthy: {args.api_base}")

    for rank, qid in enumerate(wanted, start=1):
        if qid in completed:
            print(f"[skip] {qid}", flush=True)
            continue
        selection_started = time.perf_counter()
        prompt, meta = fit_prompt(tokenizer, samples[qid], args.active_cap)
        selection_sec = time.perf_counter() - selection_started
        resident = client.chat(
            [{"role": "user", "content": warm_prefix(prompt)}],
            max_tokens=0,
            enable_thinking=False,
        )
        if resident.error or resident.finish_reason != "prefill_only":
            raise RuntimeError(
                f"{qid}: resident-window prefill failed: "
                f"{resident.error or resident.finish_reason}"
            )
        result = client.chat([{"role": "user", "content": prompt}], max_tokens=1)
        if result.error or result.ttft_s is None:
            raise RuntimeError(f"{qid}: {result.error or 'missing TTFT'}")
        measured_ttft = result.server_ttft_s or result.ttft_s
        row = {
            "question_id": qid,
            "question_type": samples[qid].get("question_type"),
            "sample_rank": rank,
            "method": "sliding_window",
            "active_cap_tokens": args.active_cap,
            **meta,
            "server_reported_prompt_tokens": result.prompt_tokens,
            "window_selection_sec": selection_sec,
            "latency_protocol": "final_query_from_resident_window_v1",
            "history_window_preparation_excluded": True,
            "resident_window_prefill_sec_excluded": resident.latency_s,
            "resident_window_prompt_tokens": resident.prompt_tokens,
            "query_ttft_sec": measured_ttft,
            "server_ttft_sec": result.server_ttft_s,
            "client_ttft_sec": result.ttft_s,
            "engine_ttft_sec": result.engine_ttft_s,
            "response_ttft_sec": result.response_ttft_s,
            "pre_answer_latency_sec": measured_ttft,
            "request_total_sec": result.latency_s,
            "finish_reason": result.finish_reason,
            "first_part": result.first_part,
            "status": "completed",
        }
        append_jsonl(args.output, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
