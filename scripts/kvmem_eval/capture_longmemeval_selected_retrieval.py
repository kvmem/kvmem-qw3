#!/usr/bin/env python3
"""Capture the exact KVMem-selected blocks for chosen LongMemEval samples.

The server must be launched with ``QW3_KVMEM_DUMP_SCORES``.  Requests use the
same three-message prompt as the canonical LongMemEval run, while attaching a
diagnostic context span and trace tag.  Generation is capped at one token: the
post-prefill selection is already final before that token is generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import requests

try:
    from .dataset import load_all
    from .prompt import render_messages
except ImportError:
    from dataset import load_all  # type: ignore
    from prompt import render_messages  # type: ignore


DEFAULT_DATA = Path("/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl")
DEFAULT_BASELINE = Path(
    "/data/chaidi/kvmem_eval/results/"
    "m102_2m_k224k_g32k_bt32_meank_r051_20260719_eval_20260719_063739.jsonl"
)


def parse_indices(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(index < 0 for index in result):
        raise argparse.ArgumentTypeError("indices must be unique non-negative integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def dumped_tags(path: Path) -> list[str]:
    return [
        str(row["trace_tag"])
        for row in read_jsonl(path)
        if row.get("type") == "meta" and row.get("trace_tag")
    ]


def health_ok(api_base: str) -> bool:
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        response = requests.get(
            root + "/health", timeout=(10, 10), proxies={"http": None, "https": None}
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def main() -> None:
    args = parse_args()
    samples = load_all(args.data)
    baseline_rows = read_jsonl(args.baseline)
    baseline_by_index = {int(row["subset_index"]): row for row in baseline_rows}
    selected_indices = args.indices[: args.limit] if args.limit is not None else args.indices
    if not selected_indices:
        raise RuntimeError("no samples selected")
    if max(selected_indices) >= len(samples):
        raise RuntimeError(f"sample index outside dataset: max={max(selected_indices)} n={len(samples)}")

    selected: list[dict[str, Any]] = []
    for index in selected_indices:
        sample = samples[index]
        baseline = baseline_by_index.get(index)
        if baseline is None or baseline.get("question_id") != sample.question_id:
            raise RuntimeError(f"baseline/dataset mismatch at index {index}")
        if baseline.get("correct") is not False:
            raise RuntimeError(f"selected sample {index}/{sample.question_id} is not a baseline error")
        selected.append(
            {
                "subset_index": index,
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "question": sample.question,
                "gold": sample.answer,
                "baseline_answer": baseline.get("answer"),
                "baseline_correct": baseline.get("correct"),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_root / "selection.json",
        {
            "schema_version": "longmemeval_kvmem_text_control.selection.v1",
            "data": str(args.data),
            "baseline": str(args.baseline),
            "indices": selected_indices,
            "samples": selected,
            "capture_generation": {
                "max_tokens": 1,
                "enable_thinking": True,
                "thinking_budget": 4096,
                "temperature": 0.6,
                "top_p": 0.95,
            },
            "kvmem": {
                "ctx": 2_000_000,
                "budget_tokens": 229_376,
                "generation_reserve_tokens": 32_768,
                "block_tokens": 32,
                "sink_blocks": 8,
                "recent_blocks": 0,
                "retrieval_method": "mean-k",
                "update_mode": "step",
                "query_conditioned": True,
            },
        },
    )

    if not health_ok(args.api_base):
        raise RuntimeError(f"qw3 server is not healthy at {args.api_base}")
    tags = dumped_tags(args.dump_file)
    expected_tags = [str(samples[index].question_id) for index in selected_indices]
    extras = sorted(set(tags) - set(expected_tags))
    duplicates = sorted(tag for tag in set(tags) if tags.count(tag) > 1)
    if extras or duplicates:
        raise RuntimeError(
            f"dump must be dedicated to this experiment: extras={extras} duplicates={duplicates}"
        )
    completed = set(tags)
    manifest_path = args.output_root / "capture_manifest.jsonl"
    manifest_ids = {str(row.get("question_id")) for row in read_jsonl(manifest_path)}
    endpoint = args.api_base.rstrip("/") + "/chat/completions"
    session = requests.Session()
    session.trust_env = False

    for position, index in enumerate(selected_indices, start=1):
        sample = samples[index]
        tag = sample.question_id
        if tag in completed:
            print(f"[skip] {position}/{len(selected_indices)} index={index} id={tag}", flush=True)
            continue
        messages = render_messages(sample)
        history = str(messages[1]["content"])
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 1,
            "enable_thinking": True,
            "thinking_budget": 4096,
            "stream": False,
            "kvmem_context_span": {
                "message_index": 1,
                "content_start": 0,
                "content_end": len(history.encode("utf-8")),
            },
            "kvmem_trace_tag": tag,
        }
        print(
            f"[capture] {position}/{len(selected_indices)} index={index} "
            f"id={tag} type={sample.question_type}",
            flush=True,
        )
        started = time.perf_counter()
        response = session.post(
            endpoint,
            json=payload,
            timeout=(30, args.timeout_sec),
            headers={"Authorization": "Bearer dummy"},
        )
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(f"capture HTTP {response.status_code}: {response.text[:1000]}")
        body = response.json()
        tags_now = dumped_tags(args.dump_file)
        if tags_now.count(tag) != 1:
            raise RuntimeError(
                f"response completed but dump contains {tags_now.count(tag)} snapshots for {tag}"
            )
        completed.add(tag)
        if tag not in manifest_ids:
            choices = body.get("choices") or []
            append_jsonl(
                manifest_path,
                {
                    "schema_version": "longmemeval_kvmem_text_control.capture.v1",
                    "position": position,
                    "subset_index": index,
                    "question_id": tag,
                    "question_type": sample.question_type,
                    "elapsed_sec": elapsed,
                    "usage": body.get("usage"),
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                },
            )
            manifest_ids.add(tag)

    final_tags = dumped_tags(args.dump_file)
    if final_tags != expected_tags:
        raise RuntimeError(f"dump tag order mismatch: actual={final_tags} expected={expected_tags}")
    print(f"[complete] captured={len(final_tags)} dump={args.dump_file}", flush=True)


if __name__ == "__main__":
    main()
