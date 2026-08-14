#!/usr/bin/env python3
"""Measure AgentLongBench Full-Context final-query TTFT from a warm prefix."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "kvmem_eval"))
from client import Qw3Client  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_worker(repo: Path) -> Any:
    path = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("alb_full_latency_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_prompt(prompt: str) -> tuple[str, str]:
    marker = "\n\nQuestion:\n"
    pos = prompt.rfind(marker)
    if pos < 0:
        raise RuntimeError("canonical prompt lacks final Question section")
    return prompt[:pos], prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--question-id", action="append", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    worker = load_worker(args.benchmark_repo)
    wanted = list(args.question_id)
    samples = {
        str(row.get("stable_sample_id")): row
        for row in read_jsonl(args.dataset)
        if str(row.get("stable_sample_id")) in set(wanted)
    }
    missing = [qid for qid in wanted if qid not in samples]
    if missing:
        raise RuntimeError(f"missing sample IDs: {missing}")
    completed = {
        str(row.get("sample_id"))
        for row in (read_jsonl(args.output) if args.output.exists() else [])
        if row.get("status") == "completed"
    }
    client = Qw3Client(
        base_url=args.api_base,
        model=args.model,
        temperature=0.7,
        top_p=0.9,
        max_tokens=1,
        enable_thinking=True,
        read_timeout=args.timeout,
    )
    if not client.health():
        raise RuntimeError(f"server is not healthy: {args.api_base}")

    for qid in wanted:
        if qid in completed:
            print(f"[skip] {qid}", flush=True)
            continue
        prompt = worker.full_context_prompt(samples[qid])
        prefix, full = split_prompt(prompt)
        # The prefix request is deliberately excluded. It materializes the
        # lossless history in qw3's dense prefix cache without generating a
        # synthetic assistant token; the final question remains cold.
        warm = client.chat(
            [{"role": "user", "content": prefix}],
            max_tokens=0,
            enable_thinking=False,
        )
        if warm.error or warm.finish_reason != "prefill_only":
            raise RuntimeError(
                f"{qid}: prefix warmup failed: "
                f"{warm.error or warm.finish_reason}"
            )
        result = client.chat([{"role": "user", "content": full}], max_tokens=1)
        if result.error or result.ttft_s is None:
            raise RuntimeError(f"{qid}: final request failed: {result.error or 'missing TTFT'}")
        measured_ttft = result.server_ttft_s or result.ttft_s
        row = {
            "sample_id": qid,
            "method": "full_context_cached_history",
            "latency_protocol": "final_query_from_resident_full_context_v1",
            "history_warmup_sec_excluded": warm.latency_s,
            "history_warmup_prompt_tokens": warm.prompt_tokens,
            "active_prompt_tokens": result.prompt_tokens,
            "pre_answer_latency_sec": measured_ttft,
            "server_ttft_sec": result.server_ttft_s,
            "client_ttft_sec": result.ttft_s,
            "engine_ttft_sec": result.engine_ttft_s,
            "response_ttft_sec": result.response_ttft_s,
            "request_total_sec": result.latency_s,
            "finish_reason": result.finish_reason,
            "first_part": result.first_part,
            "cache_state": "same-process_qw3_dense_prefix_cache",
            "status": "completed",
        }
        append_jsonl(args.output, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
