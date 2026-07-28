#!/usr/bin/env python3
"""Replay the exact text decoded from a frozen KVMem selected window.

This implements the representation-control ("Mode B") experiment:

1. validate every selected block in a tagged KVMem dump against the canonical
   chat-rendered AgentLongBench prompt;
2. concatenate the selected token IDs in original prompt order and decode that
   token stream to one raw prompt (including task instructions, final question,
   and the assistant generation prefix);
3. restart qw3 without KVMem and submit the raw prompt to /v1/completions so it
   is densely prefetched with contiguous positions;
4. grade the result with AgentLongBench's official evaluator.

The final question is already part of the selected KVMem window and is not
appended a second time.  Tokenization after decoding is audited explicitly:
non-contiguous source runs can acquire different BPE merges at their new text
boundaries, which is an intrinsic property of text replay rather than a silent
input mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Any

import run_agentlongbench_kvmem as runner


DEFAULT_DATASET = Path("/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl")
DEFAULT_TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8"
)
DEFAULT_DUMP = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_rag52_kvmem32k_overlap_20260718/kvmem_retrieval_dump.jsonl"
)
DEFAULT_KVMEM_EVAL = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_kvmem_32k_b32_lmeparams_thinking4k_20260716/eval.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_kvmem32k_selected_text_replay_20260718"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--kvmem-dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--kvmem-eval", type=Path, default=DEFAULT_KVMEM_EVAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--method", default="kvmem_selected_text_dense_replay")
    parser.add_argument("--benchmark-name", default="AgentLongBench-selected-text")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--context-window", type=int, default=65536)
    parser.add_argument("--context-safety-margin", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--max-sample-sec", type=int, default=3600)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int)
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
            raise RuntimeError("retrieval dump starts with a block before its meta row")
        else:
            current["blocks"].append(row)
    if current is not None:
        snapshots.append(current)
    seen: set[str] = set()
    for snapshot in snapshots:
        meta = snapshot["meta"]
        sid = str(meta.get("trace_tag") or "")
        if not sid or sid in seen:
            raise RuntimeError(f"retrieval dump has missing/duplicate trace_tag={sid!r}")
        seen.add(sid)
        if meta.get("schema_version") != "qw3_kvmem_retrieval_dump.v2":
            raise RuntimeError(f"unsupported retrieval dump schema for {sid}")
        expected = int(meta.get("block_count") or -1)
        if len(snapshot["blocks"]) != expected:
            raise RuntimeError(
                f"retrieval dump block count mismatch for {sid}: "
                f"rows={len(snapshot['blocks'])} meta={expected}"
            )
    return snapshots


def ids_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(struct.pack("<i", token_id))
    return digest.hexdigest()


def contiguous_runs(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for block in selected:
        p0 = int(block["p0"])
        nt = int(block["nt"])
        ids = [int(value) for value in block["tok"]]
        if runs and p0 == runs[-1]["prompt_token_end"]:
            runs[-1]["prompt_token_end"] += nt
            runs[-1]["block_ids"].append(int(block["b"]))
            runs[-1]["token_ids"].extend(ids)
        else:
            runs.append(
                {
                    "prompt_token_start": p0,
                    "prompt_token_end": p0 + nt,
                    "block_ids": [int(block["b"])],
                    "token_ids": ids,
                }
            )
    return runs


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def prepare_replays(
    args: argparse.Namespace,
    tokenizer: Any,
    canonical: Any,
    samples: dict[str, dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for delivery_index, snapshot in enumerate(snapshots, start=1):
        meta = snapshot["meta"]
        trace_tag = str(meta["trace_tag"])
        sid = trace_tag
        # Controlled oracle captures suffix the request tag so their dump can
        # coexist with an ordinary capture.  Dataset/evaluation IDs remain the
        # underlying stable sample ID.
        if sid not in samples and sid.endswith(".oracle"):
            sid = sid[: -len(".oracle")]
        if sid not in samples:
            raise RuntimeError(
                f"dataset is missing dump sample {trace_tag}"
            )
        sample = samples[sid]
        prompt = canonical.full_context_prompt(sample)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected_hash = str(sample.get("prompt_sha256") or "")
        if expected_hash and prompt_hash != expected_hash:
            raise RuntimeError(f"canonical prompt hash mismatch for {sid}")
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        rendered_ids = [
            int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
        ]
        if len(rendered_ids) != int(meta.get("prompt_tokens") or -1):
            raise RuntimeError(
                f"rendered prompt token count mismatch for {sid}: "
                f"local={len(rendered_ids)} dump={meta.get('prompt_tokens')}"
            )
        selected = sorted(
            (row for row in snapshot["blocks"] if int(row.get("sel") or 0) == 1),
            key=lambda row: int(row["p0"]),
        )
        if len(selected) != int(meta.get("selected") or -1):
            raise RuntimeError(f"selected block count mismatch for {sid}")
        previous_end = -1
        selected_ids: list[int] = []
        for block in selected:
            p0, nt = int(block["p0"]), int(block["nt"])
            exported = block.get("tok")
            if not isinstance(exported, list) or len(exported) != nt:
                raise RuntimeError(f"selected block lacks exact IDs for {sid} b={block.get('b')}")
            block_ids = [int(value) for value in exported]
            if p0 < previous_end:
                raise RuntimeError(f"overlapping selected blocks for {sid}")
            if block_ids != rendered_ids[p0 : p0 + nt]:
                raise RuntimeError(f"selected token parity failure for {sid} b={block.get('b')}")
            previous_end = p0 + nt
            selected_ids.extend(block_ids)
        runs = contiguous_runs(selected)
        for run in runs:
            decoded_run = tokenizer.decode(
                run["token_ids"],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            run_reencoded = [
                int(value)
                for value in tokenizer.encode(decoded_run, add_special_tokens=False)
            ]
            run["retokenization_exact"] = run_reencoded == run["token_ids"]
            run["retokenization_token_delta"] = len(run_reencoded) - len(run["token_ids"])
            run["retokenization_first_mismatch"] = first_mismatch(
                run["token_ids"], run_reencoded
            )

        # This is the actual Mode-B input: detokenize the selected sparse token
        # stream, then let the dense run tokenize the resulting raw text again.
        replay_prompt = tokenizer.decode(
            selected_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        replay_ids = [
            int(value) for value in tokenizer.encode(replay_prompt, add_special_tokens=False)
        ]
        q0, q1 = int(meta["query_begin"]), int(meta["query_end"])
        selected_query_tokens = sum(
            max(0, min(int(block["p0"]) + int(block["nt"]), q1) - max(int(block["p0"]), q0))
            for block in selected
        )
        original_span = (
            int(selected[-1]["p0"]) + int(selected[-1]["nt"]) - int(selected[0]["p0"])
        )
        audit_runs = [
            {
                "prompt_token_start": run["prompt_token_start"],
                "prompt_token_end": run["prompt_token_end"],
                "block_id_start": run["block_ids"][0],
                "block_id_end": run["block_ids"][-1],
                "block_count": len(run["block_ids"]),
                "token_count": len(run["token_ids"]),
                "token_sha256": ids_sha256(run["token_ids"]),
                "retokenization_exact": run["retokenization_exact"],
                "retokenization_token_delta": run["retokenization_token_delta"],
                "retokenization_first_mismatch": run["retokenization_first_mismatch"],
            }
            for run in runs
        ]
        prepared.append(
            {
                "delivery_index": delivery_index,
                "stable_sample_id": sid,
                "dump_trace_tag": trace_tag,
                "sample": sample,
                "prompt_sha256": prompt_hash,
                "original_prompt_tokens": len(rendered_ids),
                "selected_block_count": len(selected),
                "selected_token_count": len(selected_ids),
                "selected_token_sha256": ids_sha256(selected_ids),
                "selected_query_token_count": selected_query_tokens,
                "query_token_span": [q0, q1],
                "selected_run_count": len(runs),
                "selected_runs": audit_runs,
                "selected_original_span_tokens": original_span,
                "omitted_position_tokens_inside_span": original_span - len(selected_ids),
                "replay_prompt": replay_prompt,
                "replay_prompt_sha256": hashlib.sha256(replay_prompt.encode("utf-8")).hexdigest(),
                "replay_token_count_local": len(replay_ids),
                "replay_token_sha256_local": ids_sha256(replay_ids),
                "retokenization_exact": replay_ids == selected_ids,
                "retokenization_token_delta": len(replay_ids) - len(selected_ids),
                "retokenization_first_mismatch": first_mismatch(selected_ids, replay_ids),
            }
        )
    return prepared


def select_prepared(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if args.question_id:
        wanted = set(args.question_id)
        rows = [
            row
            for row in rows
            if row["stable_sample_id"] in wanted
            or str((row["sample"].get("raw") or {}).get("id")) in wanted
        ]
    elif args.limit is not None:
        if args.limit <= 0:
            raise RuntimeError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no replay samples selected")
    return rows


def tokenize_server(args: argparse.Namespace, text: str) -> list[int]:
    result = runner.post_json(
        args.api_base.rstrip("/") + "/tokenize",
        {"content": text, "return_tokens": True},
        args.timeout_sec,
    )
    tokens = result.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize result: {result}")
    return [int(value) for value in tokens]


def complete(args: argparse.Namespace, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": max_tokens,
        "thinking_budget": args.thinking_budget,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    started = time.perf_counter()
    response = runner.post_json(
        args.api_base.rstrip("/") + "/completions", payload, args.timeout_sec
    )
    elapsed = time.perf_counter() - started
    if elapsed > args.max_sample_sec:
        raise TimeoutError(f"sample exceeded {args.max_sample_sec}s")
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"completion response has no choices: {response}")
    text = str(choices[0].get("text") or "")
    return {
        "hypothesis": text.strip(),
        "raw_response": text,
        "reasoning": "",
        "usage": response.get("usage") or {},
        "finish_reason": choices[0].get("finish_reason"),
        "timing": {"total_sec": elapsed, "ttft_sec": None},
    }


def append_once(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("run this script with /home/chaidi/qw3/.venv/bin/python") from exc
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    canonical = runner.load_canonical_module(args.benchmark_repo)
    sample_rows = runner.read_jsonl(args.dataset)
    samples = {str(row.get("stable_sample_id")): row for row in sample_rows}
    snapshots = load_dump(args.kvmem_dump)
    prepared = select_prepared(
        args, prepare_replays(args, tokenizer, canonical, samples, snapshots)
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    replay_path = args.output_root / "replay_prompts.jsonl"
    audit_path = args.output_root / "replay_manifest.jsonl"
    append_once(
        replay_path,
        [
            {
                "stable_sample_id": row["stable_sample_id"],
                "replay_prompt_sha256": row["replay_prompt_sha256"],
                "replay_prompt": row["replay_prompt"],
            }
            for row in prepared
        ],
    )
    audit_keys = [
        "delivery_index",
        "stable_sample_id",
        "dump_trace_tag",
        "prompt_sha256",
        "original_prompt_tokens",
        "selected_block_count",
        "selected_token_count",
        "selected_token_sha256",
        "selected_query_token_count",
        "query_token_span",
        "selected_run_count",
        "selected_runs",
        "selected_original_span_tokens",
        "omitted_position_tokens_inside_span",
        "replay_prompt_sha256",
        "replay_token_count_local",
        "replay_token_sha256_local",
        "retokenization_exact",
        "retokenization_token_delta",
        "retokenization_first_mismatch",
    ]
    append_once(audit_path, [{key: row[key] for key in audit_keys} for row in prepared])
    runner.write_json(
        args.output_root / "run_config.json",
        {
            "schema_version": "agentlongbench_kvmem_selected_text_replay.v1",
            "mode": "decode_full_selected_window_then_dense_raw_prefill",
            "dataset": str(args.dataset),
            "kvmem_dump": str(args.kvmem_dump),
            "kvmem_eval": str(args.kvmem_eval),
            "tokenizer": str(args.tokenizer),
            "selected_samples": len(prepared),
            "stable_sample_ids": [row["stable_sample_id"] for row in prepared],
            "query_policy": "already_present_in_selected_window_do_not_append",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "thinking_budget": args.thinking_budget,
            "context_window": args.context_window,
            "seed": args.seed,
            "created_at": runner.now_iso(),
        },
    )
    print(
        f"[prepared] samples={len(prepared)} prompts={replay_path} audit={audit_path}",
        flush=True,
    )
    if args.prepare_only:
        return
    if not runner.health_ok(args.api_base):
        raise RuntimeError(f"dense qw3 server is not healthy at {args.api_base}")

    output = runner.paths(args.output_root)
    answers = runner.latest_by_id(runner.read_jsonl(output["answers"]))
    evals = runner.latest_by_id(runner.read_jsonl(output["eval"]))
    baseline = runner.latest_by_id(runner.read_jsonl(args.kvmem_eval))
    selected_samples = [row["sample"] for row in prepared]
    for index, item in enumerate(prepared, start=1):
        sid = item["stable_sample_id"]
        sample = item["sample"]
        if sid in evals:
            print(f"[skip] {index}/{len(prepared)} {sid}", flush=True)
            continue
        answer = answers.get(sid)
        if answer is None:
            server_ids = tokenize_server(args, item["replay_prompt"])
            if ids_sha256(server_ids) != item["replay_token_sha256_local"]:
                raise RuntimeError(f"local/server replay tokenizer mismatch for {sid}")
            request_max = min(
                args.max_tokens,
                max(1, args.context_window - len(server_ids) - args.context_safety_margin),
            )
            row = {
                **runner.base_row(
                    index,
                    len(prepared),
                    sample,
                    args.method,
                    args.benchmark_name,
                ),
                "prompt_mode": "raw_dense_replay_of_decoded_kvmem_selected_window",
                "prompt_sha256": item["replay_prompt_sha256"],
                "prompt_tokens": len(server_ids),
                "selected_token_count_before_text_replay": item["selected_token_count"],
                "selected_query_token_count": item["selected_query_token_count"],
                "selected_run_count": item["selected_run_count"],
                "retokenization_token_delta": item["retokenization_token_delta"],
                "generation_max_tokens_configured": args.max_tokens,
                "generation_max_tokens_request": request_max,
                "thinking_budget": args.thinking_budget,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            runner.append_jsonl(
                output["status"],
                {"time": runner.now_iso(), "stable_sample_id": sid, "index": index, "status": "started"},
            )
            print(
                f"[answer] {index}/{len(prepared)} {sid} replay={len(server_ids)} "
                f"selected={item['selected_token_count']} max={request_max}",
                flush=True,
            )
            last_error: Exception | None = None
            for attempt in range(1, args.attempts + 1):
                try:
                    result = complete(args, item["replay_prompt"], request_max)
                    answer = {**row, **result, "answered_at": runner.now_iso()}
                    runner.append_jsonl(output["answers"], answer)
                    answers[sid] = answer
                    break
                except Exception as exc:
                    last_error = exc
                    runner.append_jsonl(
                        output["status"],
                        {
                            "time": runner.now_iso(),
                            "stable_sample_id": sid,
                            "index": index,
                            "status": "answer_failed",
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:4000],
                        },
                    )
                    print(f"[error] attempt={attempt} {type(exc).__name__}: {exc}", flush=True)
                    if not runner.health_ok(args.api_base):
                        raise RuntimeError("dense qw3 server stopped during replay") from exc
                    time.sleep(min(30, attempt * 2))
            if answer is None:
                raise RuntimeError(f"all attempts failed for {sid}: {last_error}")
        evaluated = canonical.evaluate_response(
            sample, answer.get("raw_response") or answer.get("hypothesis") or ""
        )
        baseline_row = baseline.get(sid, {})
        eval_row = {
            **{
                key: answer.get(key)
                for key in (
                    "benchmark",
                    "method",
                    "index",
                    "stable_sample_id",
                    "official_id",
                    "sample_id",
                    "question_type",
                    "setting",
                    "target_length",
                    "actual_length",
                    "length_bucket",
                    "task_type",
                    "source_path",
                    "prompt_sha256",
                    "prompt_tokens",
                    "finish_reason",
                )
            },
            **evaluated,
            "kvmem_baseline_correct": baseline_row.get("correct"),
            "kvmem_baseline_score": baseline_row.get("score"),
            "evaluated_at": runner.now_iso(),
        }
        runner.append_jsonl(output["eval"], eval_row)
        evals[sid] = eval_row
        runner.append_jsonl(
            output["status"],
            {
                "time": runner.now_iso(),
                "stable_sample_id": sid,
                "index": index,
                "status": "evaluated",
                "score": eval_row.get("score"),
                "correct": eval_row.get("correct"),
            },
        )
        print(
            f"[eval] {index}/{len(prepared)} correct={eval_row.get('correct')} "
            f"kvmem={baseline_row.get('correct')}",
            flush=True,
        )
        runner.write_progress(args.output_root, selected_samples, index)
    runner.write_final(
        args.output_root,
        selected_samples,
        args.method,
        args.benchmark_name,
    )
    print(f"[complete] results={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
