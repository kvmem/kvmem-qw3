#!/usr/bin/env python3
"""Measure query-time KVMem latency from an independently frozen prefix.

The canonical Chat prompt is rendered once and tokenized once.  A raw-token
session ingests the exact prefix before the final query with reselection off;
the measured finish request appends the remaining token IDs, forces semantic
reselection with an absolute token span, and generates one token.  This avoids
both chat-template drift at the split and legacy sparse-checkpoint cold misses.
Cold history ingestion is excluded from the reported latency.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ACCOUNTING_RE = re.compile(
    r"\[qw3-native-accounting\]\s+trace_tag=(?P<tag>\S+)\s+"
    r"total_ms=(?P<total>[0-9.]+)\s+setup_ms=(?P<setup>[0-9.]+)\s+"
    r"prefill_ms=(?P<prefill>[0-9.]+)\s+"
    r"postprefill_ms=(?P<postprefill>[0-9.]+)\s+"
    r"decode_ms=(?P<decode>[0-9.]+)\s+"
    r"finalize_ms=(?P<finalize>[0-9.]+)\s+sum_ms=(?P<sum>[0-9.]+)\s+"
    r"error_ms=(?P<error>-?[0-9.]+)\s+"
    r"post_semantic_ms=(?P<semantic>[0-9.]+)\s+"
    r"post_query_replay_ms=(?P<replay>[0-9.]+)\s+"
    r"post_other_ms=(?P<other>[0-9.]+)"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def load_canonical(repo: Path) -> Any:
    path = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("paper_latency_canonical", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener().open(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:4000]}") from exc


def tokenize_count(api_base: str, text: str, timeout: int) -> int:
    body = post_json(
        api_base.rstrip("/") + "/tokenize", {"content": text}, timeout
    )
    if isinstance(body.get("count"), int):
        return int(body["count"])
    if isinstance(body.get("tokens"), list):
        return len(body["tokens"])
    raise RuntimeError(f"unexpected /tokenize response: {body}")


def tokenize(api_base: str, text: str, timeout: int) -> list[int]:
    body = post_json(
        api_base.rstrip("/") + "/tokenize",
        {"content": text, "return_tokens": True},
        timeout,
    )
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(v, int) for v in tokens):
        raise RuntimeError(f"unexpected /tokenize response: {body}")
    return tokens


def query_span(sample: dict[str, Any], prompt: str) -> tuple[int, int, int, int]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    question = str(sample.get("question") or raw.get("question") or "")
    if not question:
        raise RuntimeError("sample has an empty final question")
    suffix = f"Question:\n{question}\n\nAnswer:"
    suffix_start = prompt.rfind(suffix)
    if suffix_start < 0:
        raise RuntimeError("canonical prompt lacks the final question suffix")
    char_begin = suffix_start + len("Question:\n")
    char_end = char_begin + len(question)
    if prompt[char_begin:char_end] != question:
        raise RuntimeError("question character span does not round-trip")
    byte_begin = len(prompt[:char_begin].encode("utf-8"))
    byte_end = byte_begin + len(question.encode("utf-8"))
    return char_begin, char_end, byte_begin, byte_end


def timed_completion(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    started = time.perf_counter()
    body = post_json(url, payload, timeout)
    total = time.perf_counter() - started
    timing = body.get("timing") or {}
    # ``server_ttft_sec`` starts at HTTP request arrival and therefore covers
    # parsing/tokenization, queueing, prefix restore, KVMem reselection, query
    # replay, query prefill, and the first decode step.  The legacy
    # ``first_token_sec`` starts only when the engine is entered and is kept as
    # a diagnostic, not as the paper's pre-answer-latency boundary.
    server_ttft_sec = timing.get("server_ttft_sec")
    engine_ttft_sec = timing.get("engine_ttft_sec")
    response_ttft_sec = timing.get("response_ttft_sec")
    legacy_first_token_sec = timing.get("first_token_sec")
    if not isinstance(server_ttft_sec, (int, float)):
        raise RuntimeError(f"completion lacks server-side TTFT: {body}")
    choice = (body.get("choices") or [{}])[0]
    return {
        "server_ttft_ms": float(server_ttft_sec) * 1000.0,
        "engine_ttft_ms": (
            float(engine_ttft_sec) * 1000.0
            if isinstance(engine_ttft_sec, (int, float)) else None
        ),
        "response_ttft_ms": (
            float(response_ttft_sec) * 1000.0
            if isinstance(response_ttft_sec, (int, float)) else None
        ),
        "legacy_engine_first_token_ms": (
            float(legacy_first_token_sec) * 1000.0
            if isinstance(legacy_first_token_sec, (int, float)) else None
        ),
        # Retain the old field name for readers of existing artifacts, but
        # make it explicitly equal to the request-arrival metric.
        "client_ttft_ms": float(server_ttft_sec) * 1000.0,
        "client_first_reasoning_ms": None,
        "client_first_content_ms": None,
        "client_total_ms": total * 1000.0,
        "server_request_total_ms": float(timing.get("request_total_sec", total)) * 1000.0,
        "stream_chunks": None,
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
        "first_token_text": str(choice.get("text") or ""),
    }


def render_single_user_prompt(content: str, enable_thinking: bool) -> str:
    # Must stay byte-identical to render_messages() for one ordinary user
    # message. full_context_prompt() has no leading/trailing ASCII whitespace,
    # but strip the same four characters defensively.
    content = content.strip(" \n\r\t")
    prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    return prompt + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")


def accounting_for_tag(log_text: str, tag: str) -> dict[str, float]:
    matches = [m for m in ACCOUNTING_RE.finditer(log_text) if m.group("tag") == tag]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one native-accounting row for {tag}, found {len(matches)}"
        )
    match = matches[0]
    return {
        "native_total_ms": float(match.group("total")),
        "native_setup_ms": float(match.group("setup")),
        "query_prefill_ms": float(match.group("prefill")),
        "native_postprefill_ms": float(match.group("postprefill")),
        "native_decode_ms": float(match.group("decode")),
        "native_finalize_ms": float(match.group("finalize")),
        "native_sum_ms": float(match.group("sum")),
        "native_error_ms": float(match.group("error")),
        "post_semantic_ms": float(match.group("semantic")),
        "post_query_replay_ms": float(match.group("replay")),
        "post_other_ms": float(match.group("other")),
    }


def wait_for_accounting(log_path: Path, start: int, tag: str) -> tuple[str, dict[str, float]]:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(start)
            text = handle.read()
        if f"[qw3-native-accounting] trace_tag={tag} " in text:
            return text, accounting_for_tag(text, tag)
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for native accounting tag={tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--question-id", action="append", required=True)
    parser.add_argument("--slice", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:18082/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--active-context-budget", type=int, required=True)
    parser.add_argument("--generation-reserve", type=int, required=True)
    parser.add_argument("--block-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--thinking-budget", type=int, default=8192)
    parser.add_argument(
        "--prefill-window",
        choices=("pressure", "semantic_chunk"),
        default="pressure",
        help="history-KV construction policy from the matching utility run",
    )
    parser.add_argument("--prefill-semantic-start-tokens", type=int, default=0)
    parser.add_argument("--prefill-semantic-query-tokens", type=int, default=0)
    parser.add_argument(
        "--final-reselect",
        choices=("force", "off"),
        default="force",
        help=(
            "Use force for KVMem semantic retrieval.  Use off only for the "
            "all-fit Full Context reference, where the frozen dense history "
            "must remain unchanged."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    canonical = load_canonical(args.benchmark_repo)
    wanted = set(args.question_id)
    samples = [
        row
        for row in read_jsonl(args.dataset)
        if str(row.get("stable_sample_id")) in wanted
        or str((row.get("raw") or {}).get("id")) in wanted
    ]
    found = {str(row.get("stable_sample_id")) for row in samples}
    if len(samples) != len(wanted):
        raise RuntimeError(f"question ID mismatch: wanted={wanted}, found={found}")

    completed = {
        str(row.get("sample_id"))
        for row in (read_jsonl(args.output) if args.output.exists() else [])
        if row.get("status") == "completed"
    }
    endpoint = args.api_base.rstrip("/") + "/completions"
    common = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "enable_thinking": args.enable_thinking,
        "thinking_budget": args.thinking_budget,
        "seed": args.seed,
    }
    for sample in samples:
        sid = str(sample["stable_sample_id"])
        if sid in completed:
            print(f"[skip] {sid}", flush=True)
            continue
        prompt = canonical.full_context_prompt(sample)
        char_begin, char_end, byte_begin, byte_end = query_span(sample, prompt)
        empty_prompt = prompt[:char_begin] + prompt[char_end:]
        rendered = render_single_user_prompt(prompt, args.enable_thinking)
        rendered_empty = render_single_user_prompt(
            empty_prompt, args.enable_thinking
        )
        full_token_ids = tokenize(args.api_base, rendered, args.timeout)
        empty_token_ids = tokenize(args.api_base, rendered_empty, args.timeout)
        qb = 0
        while (
            qb < len(full_token_ids)
            and qb < len(empty_token_ids)
            and full_token_ids[qb] == empty_token_ids[qb]
        ):
            qb += 1
        suffix = 0
        while (
            suffix < len(full_token_ids) - qb
            and suffix < len(empty_token_ids) - qb
            and full_token_ids[-1 - suffix] == empty_token_ids[-1 - suffix]
        ):
            suffix += 1
        qe = len(full_token_ids) - suffix
        if qe <= qb:
            raise RuntimeError("final query did not map to a non-empty token span")
        # Persistent raw-completion sessions accept integer token arrays.  Split
        # the canonical one-shot stream at the immediately preceding physical
        # block boundary.  This is both tokenizer-exact and the smallest replay
        # suffix accepted by Adaptive prototype capture: at most B-1 historical
        # tokens are charged to the measured final request.
        split_token = (qb // args.block_tokens) * args.block_tokens
        if split_token <= 0:
            raise RuntimeError("query has no preceding physical block boundary")
        split_prefix_ids = full_token_ids[:split_token]
        split_finish_ids = full_token_ids[split_token:]
        if split_prefix_ids + split_finish_ids != full_token_ids:
            raise RuntimeError("integer-token session split changed the prompt")
        history_tokens = len(split_prefix_ids)
        query_tokens = qe - qb
        full_tokens = len(full_token_ids)
        prep_tag = f"paper-{args.slice}-{sid[:16]}-prep"
        final_tag = f"paper-{args.slice}-{sid[:16]}-final"
        session_id = f"paper-{args.slice}-{sid[:20]}"

        prep_payload = {
            **common,
            "prompt": split_prefix_ids,
            "max_tokens": 0,
            "kvmem_session_id": session_id,
            "kvmem_session_op": "start",
            "kvmem_reselect": "off",
            "kvmem_prefill_window": args.prefill_window,
            "kvmem_prefill_semantic_start_tokens": (
                args.prefill_semantic_start_tokens
            ),
            "kvmem_prefill_semantic_query_tokens": (
                args.prefill_semantic_query_tokens
            ),
            "kvmem_trace_tag": prep_tag,
        }
        prep_started = time.perf_counter()
        prep_body = post_json(endpoint, prep_payload, args.timeout)
        prep_ms = (time.perf_counter() - prep_started) * 1000.0
        prep_finish = (prep_body.get("choices") or [{}])[0].get("finish_reason")
        if prep_finish != "prefill_only":
            raise RuntimeError(f"unexpected prefill-only finish reason: {prep_finish}")

        with args.server_log.open("rb") as handle:
            log_start = handle.seek(0, 2)
        final_payload = {
            **common,
            "prompt": split_finish_ids,
            "max_tokens": 1,
            "kvmem_session_id": session_id,
            "kvmem_session_op": "finish",
            "kvmem_reselect": args.final_reselect,
            "kvmem_query_token_span": {
                "begin": qb - len(split_prefix_ids),
                "end": qe - len(split_prefix_ids),
            },
            "kvmem_trace_tag": final_tag,
        }
        client = timed_completion(endpoint, final_payload, args.timeout)
        log_text, native = wait_for_accounting(args.server_log, log_start, final_tag)
        query_lines = [
            line
            for line in log_text.splitlines()
            if "native kvmem session query-conditioned" in line
        ]
        if args.final_reselect == "force":
            if len(query_lines) != 1 or f"span=[{qb},{qe})" not in query_lines[0]:
                raise RuntimeError(
                    "raw session did not use the requested absolute query span: "
                    f"{query_lines}"
                )
            query_log = query_lines[0]
        else:
            if query_lines:
                raise RuntimeError(
                    "all-fit Full Context control unexpectedly entered the "
                    f"query-conditioned path: {query_lines}"
                )
            query_log = None
        native["kvmem_operation_ms"] = (
            native["post_semantic_ms"] + native["post_query_replay_ms"]
        )
        row = {
            "sample_id": sid,
            "slice": args.slice,
            "method": (
                "kvmem" if args.final_reselect == "force" else "full_context"
            ),
            "latency_protocol": "final_query_boundary_v2",
            "history_maintenance_excluded": True,
            "history_tokens": history_tokens,
            "query_tokens": query_tokens,
            "active_prompt_tokens": full_tokens,
            "active_context_budget": args.active_context_budget,
            "generation_reserve": args.generation_reserve,
            "kvmem_block_tokens": args.block_tokens,
            "prefill_window": args.prefill_window,
            "prefill_semantic_start_tokens": (
                args.prefill_semantic_start_tokens
            ),
            "prefill_semantic_query_tokens": (
                args.prefill_semantic_query_tokens
            ),
            "top_k": args.top_k,
            "thinking_budget": args.thinking_budget,
            "preparation_ingest_ms_excluded": prep_ms,
            **client,
            **native,
            "pre_answer_latency_ms": client["server_ttft_ms"],
            "pre_answer_latency_boundary": (
                "final-query arrival through first non-empty model token; "
                "includes semantic reselect, materialization, query replay/prefill"
            ),
            "cache_state": "clean_frozen_raw_token_session",
            "session_query_log": query_log,
            "final_reselect": args.final_reselect,
            "query_span_tokens": [qb, qe],
            "query_span_content_bytes": [byte_begin, byte_end],
            "finish_fragment_tokens": len(split_finish_ids),
            "alignment_replay_prefix_tokens": qb - len(split_prefix_ids),
            "split_token": split_token,
            "trace_tag": final_tag,
            "attempt": 1,
            "status": "completed",
        }
        append_jsonl(args.output, row)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
