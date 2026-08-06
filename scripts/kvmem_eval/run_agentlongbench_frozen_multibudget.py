#!/usr/bin/env python3
"""Evaluate several semantic budgets from one frozen AgentLongBench prefill.

For each sample this runner renders the canonical one-user-message prompt,
tokenizes that exact rendered prompt once, and freezes a checkpoint immediately
before the block-aligned final-query replay boundary. Every semantic-budget arm
then appends the identical token suffix through /v1/completions. Thus the arms
share history KV/index/state without introducing an extra Chat role boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-repo",
        type=Path,
        default=Path("/home/chaidi/AgentLongBench_Motivation"),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18088/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument(
        "--semantic-budgets",
        default="32768,65536,131072,163840",
    )
    parser.add_argument("--block-tokens", type=int, default=32)
    parser.add_argument("--context-window", type=int, default=655360)
    parser.add_argument("--context-safety-margin", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
                raise RuntimeError(f"invalid {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(
    api_base: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
    timeout: int,
) -> tuple[dict[str, Any], float]:
    data = None if payload is None else json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    started = time.perf_counter()
    try:
        with no_proxy_opener().open(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} {method} {path}: {raw[:4000]}"
        ) from exc
    return body, time.perf_counter() - started


def health_ok(api_base: str) -> bool:
    url = api_base.removesuffix("/v1").rstrip("/") + "/health"
    try:
        with no_proxy_opener().open(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def tokenize_exact(api_base: str, text: str, timeout: int) -> list[int]:
    body, _ = request_json(
        api_base.removesuffix("/v1"),
        "POST",
        "/tokenize",
        {"content": text, "return_tokens": True},
        timeout,
    )
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(v, int) for v in tokens):
        raise RuntimeError("/tokenize did not return integer token IDs")
    if body.get("count") != len(tokens):
        raise RuntimeError("/tokenize count/token mismatch")
    return tokens


def trim_ascii_bytes(data: bytes) -> tuple[bytes, int]:
    whitespace = b" \n\r\t"
    begin = 0
    while begin < len(data) and data[begin] in whitespace:
        begin += 1
    end = len(data)
    while end > begin and data[end - 1] in whitespace:
        end -= 1
    return data[begin:end], begin


def rendered_prompt_and_empty_query(
    prompt: str, query_span: tuple[int, int]
) -> tuple[str, str]:
    raw = prompt.encode("utf-8")
    trimmed, trim_begin = trim_ascii_bytes(raw)
    query_begin = query_span[0] - trim_begin
    query_end = query_span[1] - trim_begin
    if query_begin < 0 or query_end > len(trimmed) or query_end <= query_begin:
        raise RuntimeError("query span is outside trimmed canonical prompt")
    empty = trimmed[:query_begin] + trimmed[query_end:]
    prefix = b"<|im_start|>user\n"
    suffix = b"<|im_end|>\n<|im_start|>assistant\n<think>\n"
    return (
        (prefix + trimmed + suffix).decode("utf-8"),
        (prefix + empty + suffix).decode("utf-8"),
    )


def removed_token_span(full: list[int], empty: list[int]) -> tuple[int, int]:
    begin = 0
    while begin < min(len(full), len(empty)) and full[begin] == empty[begin]:
        begin += 1
    suffix = 0
    while (
        suffix < len(full) - begin
        and suffix < len(empty) - begin
        and full[len(full) - 1 - suffix] == empty[len(empty) - 1 - suffix]
    ):
        suffix += 1
    end = len(full) - suffix
    if end <= begin:
        raise RuntimeError("final question maps to an empty token span")
    return begin, end


def split_completion(text: str) -> tuple[str, str]:
    marker = "</think>"
    if marker not in text:
        return text, ""
    reasoning, answer = text.split(marker, 1)
    return reasoning.strip(), answer.lstrip("\r\n ").strip()


def latest_by_key(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row["stable_sample_id"]), int(row["semantic_budget"])): row
        for row in rows
        if row.get("stable_sample_id") and row.get("semantic_budget") is not None
    }


def main() -> None:
    args = parse_args()
    budgets = [int(value.strip()) for value in args.semantic_budgets.split(",")]
    if not budgets or any(value <= 0 for value in budgets):
        raise RuntimeError("semantic budgets must be positive")
    if len(set(budgets)) != len(budgets):
        raise RuntimeError("semantic budgets must be unique")
    if any(value % args.block_tokens != 0 for value in budgets):
        raise RuntimeError("semantic budgets must be block aligned")
    budgets.sort()

    canonical = load_module(
        args.benchmark_repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py",
        "agentlongbench_fullcontext_frozen",
    )
    span_helpers = load_module(
        Path(__file__).with_name("run_agentlongbench_kvmem.py"),
        "agentlongbench_kvmem_span_helpers",
    )
    samples = read_jsonl(args.dataset)
    manifest = read_jsonl(args.manifest)
    if len(samples) != len(manifest):
        raise RuntimeError("dataset/manifest length mismatch")
    sample_ids = [str(row.get("stable_sample_id") or "") for row in samples]
    manifest_ids = [str(row.get("stable_sample_id") or "") for row in manifest]
    if not all(sample_ids) or sample_ids != manifest_ids:
        raise RuntimeError("dataset/manifest stable IDs do not match")
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no samples selected")
    if not health_ok(args.api_base):
        raise RuntimeError(f"server is not healthy: {args.api_base}")

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    answers_path = root / "answers.jsonl"
    eval_path = root / "eval.jsonl"
    audit_path = root / "cache_audit.jsonl"
    answers = latest_by_key(read_jsonl(answers_path))
    evals = latest_by_key(read_jsonl(eval_path))
    write_json(
        root / "run_config.json",
        {
            "benchmark": "AgentLongBench-512K-Frozen-MultiBudget",
            "dataset": str(args.dataset),
            "manifest": str(args.manifest),
            "semantic_budgets": budgets,
            "prefill_budget": 229376,
            "generation_budget": args.max_tokens,
            "block_tokens": args.block_tokens,
            "retrieval": "key-direction-adaptive",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "thinking_budget": args.thinking_budget,
            "seed": args.seed,
            "selected_samples": len(samples),
            "prompt_mode": "canonical_exact_tokens_frozen_before_query",
            "started_at": now_iso(),
        },
    )

    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        sid = str(sample["stable_sample_id"])
        needed = [budget for budget in budgets if (sid, budget) not in answers]
        if needed:
            prompt = canonical.full_context_prompt(sample)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            expected_hash = str(sample.get("prompt_sha256") or "")
            if expected_hash and prompt_hash != expected_hash:
                raise RuntimeError(f"prompt hash mismatch for {sid}")
            query_bytes = span_helpers.query_byte_span(sample, prompt)
            rendered, empty_query = rendered_prompt_and_empty_query(prompt, query_bytes)
            full_tokens = tokenize_exact(args.api_base, rendered, args.timeout_sec)
            empty_tokens = tokenize_exact(args.api_base, empty_query, args.timeout_sec)
            query_begin, query_end = removed_token_span(full_tokens, empty_tokens)
            boundary = (query_begin // args.block_tokens) * args.block_tokens
            if boundary == 0 or query_end > len(full_tokens):
                raise RuntimeError(f"invalid frozen split for {sid}")
            prefix_tokens = full_tokens[:boundary]
            suffix_tokens = full_tokens[boundary:]
            local_query = (query_begin - boundary, query_end - boundary)
            request_max_tokens = min(
                args.max_tokens,
                max(
                    1,
                    args.context_window
                    - len(full_tokens)
                    - args.context_safety_margin,
                ),
            )
            cache_id = "alb512-" + sid[:48]
            print(
                f"[prefill] {index}/{total} {sid} full={len(full_tokens)} "
                f"base={boundary} suffix={len(suffix_tokens)} "
                f"query=[{query_begin},{query_end}) arms={needed}",
                flush=True,
            )
            save_payload = {
                "model": args.model,
                "prompt": prefix_tokens,
                "max_tokens": 0,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "enable_thinking": True,
                "thinking_budget": args.thinking_budget,
                "seed": args.seed,
                "kvmem_reselect": "off",
                "kvmem_cache": {
                    "save": {
                        "id": cache_id,
                        "scope": "local",
                        "when": "after_request",
                        "ttl_seconds": 7200,
                    }
                },
            }
            save_body, save_sec = request_json(
                args.api_base, "POST", "/completions", save_payload, args.timeout_sec
            )
            save_info = save_body.get("kvmem_cache") or {}
            if save_body.get("choices", [{}])[0].get("finish_reason") != "prefill_only":
                raise RuntimeError(f"cache save was not prefill-only: {save_body}")
            if save_info.get("version") != 1:
                raise RuntimeError(f"unexpected cache metadata: {save_info}")
            append_jsonl(
                audit_path,
                {
                    "time": now_iso(),
                    "stable_sample_id": sid,
                    "index": index,
                    "phase": "save",
                    "cache_id": cache_id,
                    "cache_version": 1,
                    "full_prompt_tokens": len(full_tokens),
                    "frozen_prefix_tokens": boundary,
                    "query_token_span": [query_begin, query_end],
                    "elapsed_sec": save_sec,
                },
            )
            try:
                for budget in needed:
                    branch_payload = {
                        "model": args.model,
                        "prompt": suffix_tokens,
                        "max_tokens": request_max_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "enable_thinking": True,
                        "thinking_budget": args.thinking_budget,
                        "seed": args.seed,
                        "kvmem_reselect": "force",
                        "kvmem_query_token_span": {
                            "begin": local_query[0],
                            "end": local_query[1],
                        },
                        "kvmem_semantic_budget": budget,
                        "kvmem_trace_tag": f"{sid}:k{budget}",
                        "kvmem_cache": {
                            "load": {
                                "id": cache_id,
                                "mode": "frozen",
                                "required": True,
                                "expected_version": 1,
                            }
                        },
                    }
                    last_error: Exception | None = None
                    for attempt in range(1, args.attempts + 1):
                        try:
                            body, elapsed = request_json(
                                args.api_base,
                                "POST",
                                "/completions",
                                branch_payload,
                                args.timeout_sec,
                            )
                            choice = body.get("choices", [{}])[0]
                            raw_text = str(choice.get("text") or "")
                            reasoning, hypothesis = split_completion(raw_text)
                            row = {
                                "benchmark": "AgentLongBench-512K-Frozen-MultiBudget",
                                "method": f"kvmem_adaptive_p224k_s{budget}",
                                "index": index,
                                "total": total,
                                "stable_sample_id": sid,
                                "official_id": (sample.get("raw") or {}).get("id"),
                                "task_type": sample.get("task_type"),
                                "setting": sample.get("setting"),
                                "target_length": sample.get("target_length"),
                                "actual_length": sample.get("actual_length"),
                                "semantic_budget": budget,
                                "prefill_budget": 229376,
                                "prompt_sha256": prompt_hash,
                                "full_prompt_tokens": len(full_tokens),
                                "frozen_prefix_tokens": boundary,
                                "branch_prompt_tokens": len(suffix_tokens),
                                "query_token_span": [query_begin, query_end],
                                "query_token_span_local": list(local_query),
                                "generation_max_tokens": request_max_tokens,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "seed": args.seed,
                                "hypothesis": hypothesis,
                                "raw_response": hypothesis,
                                "raw_completion": raw_text,
                                "reasoning": reasoning,
                                "finish_reason": choice.get("finish_reason"),
                                "usage": body.get("usage") or {},
                                "timing": body.get("timing") or {},
                                "request_elapsed_sec": elapsed,
                                "answered_at": now_iso(),
                            }
                            append_jsonl(answers_path, row)
                            answers[(sid, budget)] = row
                            append_jsonl(
                                audit_path,
                                {
                                    "time": now_iso(),
                                    "stable_sample_id": sid,
                                    "index": index,
                                    "phase": "frozen_query",
                                    "semantic_budget": budget,
                                    "cache_version": (body.get("kvmem_cache") or {}).get("version"),
                                    "elapsed_sec": elapsed,
                                    "finish_reason": choice.get("finish_reason"),
                                },
                            )
                            print(
                                f"[answer] {index}/{total} K={budget} "
                                f"elapsed={elapsed:.1f}s finish={choice.get('finish_reason')}",
                                flush=True,
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            print(
                                f"[error] {sid} K={budget} attempt={attempt}: {exc}",
                                flush=True,
                            )
                            if not health_ok(args.api_base):
                                raise RuntimeError("qw3 server stopped") from exc
                            time.sleep(min(30, 2 * attempt))
                    if (sid, budget) not in answers:
                        raise RuntimeError(
                            f"all attempts failed for {sid} K={budget}: {last_error}"
                        )
            finally:
                try:
                    request_json(
                        args.api_base.removesuffix("/v1"),
                        "DELETE",
                        f"/v1/kvmem/caches/{cache_id}",
                        None,
                        60,
                    )
                except Exception as exc:
                    print(f"[warn] cache delete failed for {cache_id}: {exc}", flush=True)

        for budget in budgets:
            key = (sid, budget)
            if key in evals:
                continue
            answer = answers.get(key)
            if answer is None:
                raise RuntimeError(f"missing answer for {sid} K={budget}")
            evaluated = canonical.evaluate_response(
                sample, answer.get("raw_response") or ""
            )
            eval_row = {
                **{
                    name: answer.get(name)
                    for name in (
                        "benchmark",
                        "method",
                        "index",
                        "stable_sample_id",
                        "official_id",
                        "task_type",
                        "setting",
                        "target_length",
                        "actual_length",
                        "semantic_budget",
                        "finish_reason",
                    )
                },
                **evaluated,
                "evaluated_at": now_iso(),
            }
            append_jsonl(eval_path, eval_row)
            evals[key] = eval_row
            print(
                f"[eval] {index}/{total} K={budget} "
                f"score={eval_row.get('score')} correct={eval_row.get('correct')}",
                flush=True,
            )

        per_budget: dict[str, Any] = {}
        for budget in budgets:
            rows = [
                row for (row_sid, row_budget), row in evals.items()
                if row_budget == budget and row_sid in set(sample_ids[:total])
            ]
            score_sum = sum(float(row.get("score") or 0.0) for row in rows)
            exact = sum(row.get("correct") is True for row in rows)
            per_budget[str(budget)] = {
                "evaluated": len(rows),
                "official_score_evaluated": score_sum / len(rows) if rows else None,
                "exact_correct": exact,
                "exact_accuracy_evaluated": exact / len(rows) if rows else None,
            }
        write_json(
            root / "progress.json",
            {
                "updated_at": now_iso(),
                "current_index": index,
                "total_samples": total,
                "per_budget": per_budget,
            },
        )

    write_json(
        root / "validation_report.json",
        {
            "updated_at": now_iso(),
            "samples": total,
            "budgets": budgets,
            "expected_answers": total * len(budgets),
            "answers": len(answers),
            "evaluated": len(evals),
            "passed": len(answers) == total * len(budgets)
            and len(evals) == total * len(budgets),
        },
    )
    print(f"[complete] results={root}", flush=True)


if __name__ == "__main__":
    main()
