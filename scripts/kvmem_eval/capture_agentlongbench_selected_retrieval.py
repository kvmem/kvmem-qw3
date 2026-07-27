#!/usr/bin/env python3
"""Capture exact KVMem-selected blocks for an explicit AgentLongBench ID list.

This is a retrieval-only probe.  It sends the canonical FullContext prompt with
``max_tokens=1`` because the final query-conditioned selection is complete
before the first generated token.  The server must be launched with
``QW3_KVMEM_DUMP_SCORES`` pointing at ``--dump-file``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import run_agentlongbench_kvmem as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not ids or any(len(sid) != 64 for sid in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(
            f"{path} must contain unique non-empty 64-character stable IDs"
        )
    return ids


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
                raise RuntimeError(
                    f"invalid dump JSON {path}:{line_number}: {exc}"
                ) from exc
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
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": 1,
        "enable_thinking": True,
        "thinking_budget": args.thinking_budget,
        "seed": args.seed,
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
    selected_ids = read_ids(args.ids_file)
    if args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit must be positive")
        selected_ids = selected_ids[: args.limit]
    selected_set = set(selected_ids)

    samples = runner.read_jsonl(args.dataset)
    sample_map = {str(row.get("stable_sample_id")): row for row in samples}
    missing = [sid for sid in selected_ids if sid not in sample_map]
    if missing:
        raise RuntimeError(
            f"dataset is missing {len(missing)} selected IDs; first={missing[0]}"
        )
    canonical = runner.load_canonical_module(args.benchmark_repo)
    if not runner.health_ok(args.api_base):
        raise RuntimeError(f"qw3 server is not healthy at {args.api_base}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "capture_manifest.jsonl"
    config_path = args.output_root / "capture_config.json"
    runner.write_json(
        config_path,
        {
            "schema_version": "agentlongbench_selected_kvmem_capture.v1",
            "ids_file": str(args.ids_file),
            "dataset": str(args.dataset),
            "dump_file": str(args.dump_file),
            "sample_count": len(selected_ids),
            "stable_sample_ids": selected_ids,
            "generation": {
                "max_tokens": 1,
                "enable_thinking": True,
                "thinking_budget": args.thinking_budget,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
            },
            "created_at": runner.now_iso(),
        },
    )

    existing_tags = dumped_tags(args.dump_file)
    extra = sorted(set(existing_tags) - selected_set)
    duplicates = sorted(
        tag for tag in set(existing_tags) if existing_tags.count(tag) > 1
    )
    if extra or duplicates:
        raise RuntimeError(
            "dump file must be dedicated to this capture; "
            f"extra_tags={extra[:3]} duplicate_tags={duplicates[:3]}"
        )
    completed = set(existing_tags)
    manifest_ids = {
        str(row.get("stable_sample_id"))
        for row in runner.read_jsonl(manifest_path)
    }
    endpoint = args.api_base.rstrip("/") + "/chat/completions"

    for index, sid in enumerate(selected_ids, start=1):
        if sid in completed:
            print(f"[skip] {index}/{len(selected_ids)} {sid}", flush=True)
            continue
        sample = sample_map[sid]
        prompt = canonical.full_context_prompt(sample)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        query_span = runner.query_byte_span(sample, prompt)
        context_span = runner.history_byte_span(canonical, sample, prompt)
        prompt_tokens = runner.tokenize_count(
            args.api_base, prompt, args.timeout_sec
        )
        print(
            f"[capture] {index}/{len(selected_ids)} {sid} "
            f"prompt={prompt_tokens}",
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
                f"server completed but dump has {tags_now.count(sid)} records "
                f"for {sid}; verify QW3_KVMEM_DUMP_SCORES={args.dump_file}"
            )
        completed.add(sid)
        if sid not in manifest_ids:
            choices = response.get("choices") or []
            runner.append_jsonl(
                manifest_path,
                {
                    "schema_version":
                        "agentlongbench_selected_kvmem_capture.v1",
                    "stable_sample_id": sid,
                    "delivery_index": index,
                    "prompt_sha256": prompt_hash,
                    "prompt_tokens_unframed": prompt_tokens,
                    "kvmem_query_span_content_bytes": list(query_span),
                    "kvmem_context_span_content_bytes": list(context_span),
                    "elapsed_sec": elapsed,
                    "finish_reason":
                        choices[0].get("finish_reason") if choices else None,
                    "captured_at": runner.now_iso(),
                },
            )
            manifest_ids.add(sid)

    final_tags = dumped_tags(args.dump_file)
    if final_tags != selected_ids:
        raise RuntimeError(
            "dump meta order/IDs do not exactly match selected IDs: "
            f"captured={len(final_tags)} expected={len(selected_ids)}"
        )
    print(
        f"capture complete: {len(final_tags)} exact IDs; dump={args.dump_file}",
        flush=True,
    )


if __name__ == "__main__":
    main()
