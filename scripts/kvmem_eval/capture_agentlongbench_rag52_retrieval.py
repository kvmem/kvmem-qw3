#!/usr/bin/env python3
"""Capture KVMem retrieval blocks for the frozen 52-sample RAG comparison.

This is a retrieval-only probe: it sends the exact canonical FullContext prompt
with max_tokens=1, because the post-prefill KVMem selection is complete before
the first generated token. The server must be launched with
QW3_KVMEM_DUMP_SCORES pointing at --dump-file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import run_agentlongbench_kvmem as runner


DEFAULT_DELIVERY = Path("/home/chaidi/kvmem_eval/rag_blocks_52_delivery")
DEFAULT_DATASET = Path("/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl")
DEFAULT_OUTPUT = Path(
    "/data/chaidi/kvmem_eval/results/agentlongbench_rag52_kvmem_retrieval"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--kvmem-gpu-memory-ratio", type=float, default=0.9)
    return parser.parse_args()


def frozen_delivery_ids(delivery: Path) -> list[str]:
    files = [delivery / f"overlap_{overlap}_blocks.jsonl" for overlap in (0, 4, 8)]
    id_lists: list[list[str]] = []
    for path in files:
        rows = runner.read_jsonl(path)
        ids = [str(row.get("stable_sample_id") or "") for row in rows]
        if len(ids) != 52 or any(not sid for sid in ids) or len(set(ids)) != 52:
            raise RuntimeError(
                f"{path} must contain exactly 52 unique non-empty stable_sample_id values"
            )
        id_lists.append(ids)
    if id_lists[1] != id_lists[0] or id_lists[2] != id_lists[0]:
        raise RuntimeError("the three delivery files do not have identical sample ID order")
    return id_lists[0]


def dumped_tags(path: Path) -> list[str]:
    if not path.exists():
        return []
    tags: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid dump JSON {path}:{line_number}: {exc}") from exc
            if row.get("type") == "meta" and row.get("trace_tag"):
                tags.append(str(row["trace_tag"]))
    return tags


def payload(
    args: argparse.Namespace,
    prompt: str,
    query_span: tuple[int, int],
    context_span: tuple[int, int],
    stable_sample_id: str,
) -> dict[str, Any]:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 1,
        "enable_thinking": True,
        "thinking_budget": 4096,
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
        "kvmem_trace_tag": stable_sample_id,
    }


def main() -> None:
    args = parse_args()
    frozen_ids = frozen_delivery_ids(args.delivery)
    if args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit must be positive")
        frozen_ids = frozen_ids[: args.limit]
    frozen_set = set(frozen_ids)

    samples = runner.read_jsonl(args.dataset)
    sample_map = {str(row.get("stable_sample_id")): row for row in samples}
    missing = [sid for sid in frozen_ids if sid not in sample_map]
    if missing:
        raise RuntimeError(f"dataset is missing {len(missing)} frozen IDs; first={missing[0]}")
    canonical = runner.load_canonical_module(args.benchmark_repo)
    if not runner.health_ok(args.api_base):
        raise RuntimeError(f"qw3 server is not healthy at {args.api_base}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "capture_manifest.jsonl"
    config_path = args.output_root / "capture_config.json"
    runner.write_json(
        config_path,
        {
            "schema_version": "agentlongbench_rag52_kvmem_capture.v1",
            "delivery": str(args.delivery),
            "dataset": str(args.dataset),
            "dump_file": str(args.dump_file),
            "sample_count": len(frozen_ids),
            "stable_sample_ids": frozen_ids,
            "generation": {
                "max_tokens": 1,
                "enable_thinking": True,
                "thinking_budget": 4096,
                "temperature": 0.6,
                "top_p": 0.95,
            },
            "kvmem_retrieval": {
                "budget_tokens": 32768,
                "generation_reserve_tokens": 32768,
                "block_tokens": 32,
                "sink_blocks": 8,
                "recent_blocks": 0,
                "method": "retrieval",
                "retrieval_method": "mean-k",
                "update_mode": "step",
                "query_conditioned": True,
                "gpu_memory_ratio_ceiling": args.kvmem_gpu_memory_ratio,
            },
            "created_at": runner.now_iso(),
        },
    )

    existing_tags = dumped_tags(args.dump_file)
    extra = sorted(set(existing_tags) - frozen_set)
    duplicates = sorted(tag for tag in set(existing_tags) if existing_tags.count(tag) > 1)
    if extra or duplicates:
        raise RuntimeError(
            "dump file must be dedicated to this capture; "
            f"extra_tags={extra[:3]} duplicate_tags={duplicates[:3]}"
        )
    completed = set(existing_tags)
    manifest_ids = {
        str(row.get("stable_sample_id")) for row in runner.read_jsonl(manifest_path)
    }
    endpoint = args.api_base.rstrip("/") + "/chat/completions"

    for index, sid in enumerate(frozen_ids, start=1):
        if sid in completed:
            print(f"[skip] {index}/{len(frozen_ids)} {sid}", flush=True)
            continue
        sample = sample_map[sid]
        prompt = canonical.full_context_prompt(sample)
        expected_hash = str(sample.get("prompt_sha256") or "")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if expected_hash and prompt_hash != expected_hash:
            raise RuntimeError(f"canonical prompt hash mismatch for {sid}")
        query_span = runner.query_byte_span(sample, prompt)
        context_span = runner.history_byte_span(canonical, sample, prompt)
        prompt_tokens = runner.tokenize_count(args.api_base, prompt, args.timeout_sec)
        print(
            f"[capture] {index}/{len(frozen_ids)} {sid} prompt={prompt_tokens}",
            flush=True,
        )
        started = time.perf_counter()
        response = runner.post_json(
            endpoint,
            payload(args, prompt, query_span, context_span, sid),
            args.timeout_sec,
        )
        elapsed = time.perf_counter() - started
        tags_now = dumped_tags(args.dump_file)
        if tags_now.count(sid) != 1:
            raise RuntimeError(
                f"server response completed but dump has {tags_now.count(sid)} records "
                f"for {sid}; verify QW3_KVMEM_DUMP_SCORES={args.dump_file}"
            )
        completed.add(sid)
        if sid not in manifest_ids:
            choices = response.get("choices") or [] if isinstance(response, dict) else []
            runner.append_jsonl(
                manifest_path,
                {
                    "schema_version": "agentlongbench_rag52_kvmem_capture.v1",
                    "stable_sample_id": sid,
                    "delivery_index": index,
                    "prompt_sha256": prompt_hash,
                    "prompt_tokens_unframed": prompt_tokens,
                    "kvmem_query_span_content_bytes": list(query_span),
                    "kvmem_context_span_content_bytes": list(context_span),
                    "elapsed_sec": elapsed,
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                    "captured_at": runner.now_iso(),
                },
            )
            manifest_ids.add(sid)

    final_tags = dumped_tags(args.dump_file)
    if final_tags != frozen_ids:
        raise RuntimeError(
            "dump meta order/IDs do not exactly match the frozen delivery IDs: "
            f"captured={len(final_tags)} expected={len(frozen_ids)}"
        )
    print(
        f"capture complete: {len(final_tags)} exact delivery IDs; dump={args.dump_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
