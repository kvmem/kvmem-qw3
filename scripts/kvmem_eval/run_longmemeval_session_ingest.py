#!/usr/bin/env python3
"""Causal, incremental LongMemEval session-ingest evaluation.

The dataset controls session boundaries and query choice.  qw3 receives only
generic API operations: prefill-only append, semantic reselection, fixed-window
append, and final decode.  Historical assistant turns are teacher-forced and
never generated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

try:
    from .dataset import load_all
    from .judge import DeepSeekJudge
    from .prompt import SYSTEM_INSTRUCTION
except ImportError:
    from dataset import load_all  # type: ignore
    from judge import DeepSeekJudge  # type: ignore
    from prompt import SYSTEM_INSTRUCTION  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run causal per-session LongMemEval ingest over the qw3 API")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--use-all", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("/data/chaidi/kvmem_eval/results"))
    parser.add_argument("--tag", default="longmemeval_session_ingest")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--active-capacity", type=int, default=262144,
                        help="begin per-session reselection once prior history "
                             "has filled this many rendered tokens")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--read-timeout", type=float, default=7200.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    return parser.parse_args()


def byte_span(content: str, needle: str, *, last: bool = False) -> tuple[int, int]:
    char_start = content.rfind(needle) if last else content.find(needle)
    if char_start < 0:
        raise RuntimeError("query text is not present in its rendered user message")
    byte_start = len(content[:char_start].encode("utf-8"))
    return byte_start, byte_start + len(needle.encode("utf-8"))


def render_session(sample: Any, index: int) -> list[dict[str, str]]:
    date = sample.haystack_dates[index] if index < len(sample.haystack_dates) else ""
    header = (f"=== Conversation on {date} ===" if date
              else f"=== Conversation {index + 1} ===")
    rendered: list[dict[str, str]] = []
    first = True
    for turn in sample.haystack_sessions[index]:
        role = str(turn.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        original = str(turn.get("content", "")).strip()
        content = f"{header}\n{original}" if first else original
        rendered.append({"role": role, "content": content,
                         "_query_text": original})
        first = False
    return rendered


class SessionClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.http = requests.Session()
        self.http.trust_env = False

    def health(self) -> bool:
        root = self.args.base_url[:-3] if self.args.base_url.endswith("/v1") \
            else self.args.base_url
        try:
            response = self.http.get(f"{root}/health", timeout=(10, 10))
            return response.ok
        except requests.RequestException:
            return False

    def chat(self, messages: list[dict[str, str]], *, session_id: str,
             operation: str, max_tokens: int, reselect: str,
             prefill_window: str, query_span: dict[str, int] | None = None,
             trace_tag: str = "") -> tuple[dict[str, Any], float]:
        clean_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        payload: dict[str, Any] = {
            "model": self.args.model,
            "messages": clean_messages,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": max_tokens,
            "enable_thinking": not self.args.no_thinking,
            "stream": False,
            "kvmem_session_id": session_id,
            "kvmem_session_op": operation,
            "kvmem_reselect": reselect,
            "kvmem_prefill_window": prefill_window,
        }
        if query_span is not None:
            payload["kvmem_query_span"] = query_span
        if trace_tag:
            payload["kvmem_trace_tag"] = trace_tag
        started = time.monotonic()
        response = self.http.post(
            f"{self.args.base_url}/chat/completions",
            headers={"Authorization": "Bearer local-kvmem-eval"},
            json=payload,
            timeout=(30, self.args.read_timeout),
        )
        latency = time.monotonic() - started
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
        body = response.json()
        return body, latency


def response_fields(body: dict[str, Any]) -> tuple[str, str, int, int]:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("chat response contains no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    answer = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    usage = body.get("usage") or {}
    return (answer, reasoning, int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0))


def prefill(client: SessionClient, audit: Any, messages: list[dict[str, str]],
            *, session_id: str, operation: str, reselect: str,
            prefill_window: str, query_span: dict[str, int] | None,
            sample_id: str, session_index: int | str,
            phase: str) -> int:
    body, latency = client.chat(
        messages, session_id=session_id, operation=operation, max_tokens=0,
        reselect=reselect, prefill_window=prefill_window,
        query_span=query_span,
        trace_tag=f"lme-session-{sample_id}-{session_index}-{phase}",
    )
    answer, reasoning, prompt_tokens, completion_tokens = response_fields(body)
    finish_reason = str(body["choices"][0].get("finish_reason") or "")
    if answer or reasoning or completion_tokens != 0 or finish_reason != "prefill_only":
        raise RuntimeError(
            "prefill-only contract violated: "
            f"answer={len(answer)} reasoning={len(reasoning)} "
            f"completion_tokens={completion_tokens} finish={finish_reason!r}")
    audit.write(json.dumps({
        "question_id": sample_id,
        "session_index": session_index,
        "phase": phase,
        "operation": operation,
        "reselect": reselect,
        "prefill_window": prefill_window,
        "message_count": len(messages),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "latency_s": latency,
    }, ensure_ascii=False) + "\n")
    audit.flush()
    return prompt_tokens


def selected_samples(samples: list[Any], args: argparse.Namespace) -> list[tuple[int, Any]]:
    if args.indices:
        indices = [int(value) for value in args.indices.split(",") if value.strip()]
        return [(index, samples[index]) for index in indices]
    chosen = list(enumerate(samples))
    return chosen if args.limit is None else chosen[:args.limit]


def main() -> int:
    args = parse_args()
    samples = load_all(args.data)
    chosen = selected_samples(samples, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = args.out_dir / f"{args.tag}_eval_{timestamp}.jsonl"
    audit_path = args.out_dir / f"{args.tag}_requests_{timestamp}.jsonl"
    summary_path = args.out_dir / f"{args.tag}_eval_{timestamp}_summary.json"

    client = SessionClient(args)
    if not client.health():
        raise RuntimeError(f"qw3 endpoint is not healthy: {args.base_url}")
    judge = None if args.no_judge else DeepSeekJudge()
    print(f"[session-ingest] samples={len(chosen)} active_capacity="
          f"{args.active_capacity} results={result_path}")

    done = correct = errors = 0
    started_all = time.monotonic()
    with result_path.open("w", encoding="utf-8") as results, \
            audit_path.open("w", encoding="utf-8") as audit:
        for ordinal, (sample_index, sample) in enumerate(chosen, start=1):
            sample_started = time.monotonic()
            session_id = f"lme-{sample.question_id}-{timestamp}"
            logical_tokens = 0
            reselect_count = 0
            causal_prefix_sessions = 0
            no_user_sessions = 0
            error: str | None = None
            answer = reasoning = finish_reason = ""
            completion_tokens = 0
            verdict: bool | None = None
            judge_raw = judge_error = None
            try:
                system = SYSTEM_INSTRUCTION.format(
                    question_date=sample.question_date or "unknown")
                logical_tokens += prefill(
                    client, audit, [{"role": "system", "content": system}],
                    session_id=session_id, operation="start", reselect="off",
                    prefill_window="pressure", query_span=None,
                    sample_id=sample.question_id, session_index="system",
                    phase="system")

                for session_index in range(len(sample.haystack_sessions)):
                    turns = render_session(sample, session_index)
                    if not turns:
                        continue
                    first_user = next(
                        (index for index, turn in enumerate(turns)
                         if turn["role"] == "user"), None)
                    pressure_reached = logical_tokens >= args.active_capacity
                    if not pressure_reached or first_user is None:
                        if pressure_reached and first_user is None:
                            no_user_sessions += 1
                        logical_tokens += prefill(
                            client, audit, turns, session_id=session_id,
                            operation="append", reselect="off",
                            prefill_window="pressure", query_span=None,
                            sample_id=sample.question_id,
                            session_index=session_index, phase="whole")
                        continue

                    if first_user > 0:
                        causal_prefix_sessions += 1
                        logical_tokens += prefill(
                            client, audit, turns[:first_user],
                            session_id=session_id, operation="append",
                            reselect="off", prefill_window="pressure",
                            query_span=None, sample_id=sample.question_id,
                            session_index=session_index, phase="pre_user")

                    user = turns[first_user]
                    query_start, query_end = byte_span(
                        user["content"], user["_query_text"])
                    logical_tokens += prefill(
                        client, audit, [user], session_id=session_id,
                        operation="append", reselect="force",
                        prefill_window="pressure",
                        query_span={"message_index": 0,
                                    "content_start": query_start,
                                    "content_end": query_end},
                        sample_id=sample.question_id,
                        session_index=session_index, phase="query")
                    reselect_count += 1

                    remainder = turns[first_user + 1:]
                    if remainder:
                        logical_tokens += prefill(
                            client, audit, remainder,
                            session_id=session_id, operation="append",
                            reselect="off", prefill_window="keep_selected",
                            query_span=None, sample_id=sample.question_id,
                            session_index=session_index, phase="remainder")

                final_content = (
                    "=== End of conversation history ===\n\n"
                    "Based on the conversations above, answer the following "
                    "question.\n"
                    f"Question (asked on {sample.question_date}): "
                    f"{sample.question}")
                question_start, question_end = byte_span(
                    final_content, sample.question, last=True)
                body, final_latency = client.chat(
                    [{"role": "user", "content": final_content}],
                    session_id=session_id, operation="finish",
                    max_tokens=args.max_tokens, reselect="force",
                    prefill_window="pressure",
                    query_span={"message_index": 0,
                                "content_start": question_start,
                                "content_end": question_end},
                    trace_tag=f"lme-session-{sample.question_id}-final")
                answer, reasoning, prompt_tokens, completion_tokens = \
                    response_fields(body)
                finish_reason = str(
                    body["choices"][0].get("finish_reason") or "")
                logical_tokens += prompt_tokens
                reselect_count += 1
                audit.write(json.dumps({
                    "question_id": sample.question_id,
                    "session_index": "final",
                    "phase": "final_query",
                    "operation": "finish",
                    "reselect": "force",
                    "prefill_window": "pressure",
                    "message_count": 1,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "latency_s": final_latency,
                }, ensure_ascii=False) + "\n")
                audit.flush()
                if judge is not None:
                    judgment = judge.judge(sample, answer)
                    verdict = judgment.correct
                    judge_raw = judgment.raw
                    judge_error = judgment.error
                    if verdict:
                        correct += 1
            except Exception as exc:  # keep the remaining frozen samples runnable
                error = f"{type(exc).__name__}: {exc}"
                errors += 1

            done += 1
            row = {
                "subset_index": sample_index,
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "question": sample.question,
                "gold": sample.answer,
                "answer": answer,
                "reasoning": reasoning
                    if os.environ.get("QW3_EVAL_STORE_REASONING") == "1" else None,
                "reasoning_chars": len(reasoning),
                "correct": verdict,
                "judge_raw": judge_raw,
                "judge_error": judge_error,
                "finish_reason": finish_reason,
                "completion_tokens": completion_tokens,
                "logical_prompt_tokens": logical_tokens,
                "history_sessions": len(sample.haystack_sessions),
                "semantic_reselections": reselect_count,
                "assistant_prefix_sessions": causal_prefix_sessions,
                "no_user_sessions": no_user_sessions,
                "latency_s": time.monotonic() - sample_started,
                "client_error": error,
            }
            results.write(json.dumps(row, ensure_ascii=False) + "\n")
            results.flush()
            accuracy_label = (
                f"{100.0 * correct / done:.1f}%" if judge is not None else "n/a")
            print(f"[{ordinal:>2}/{len(chosen)}] correct={verdict} "
                  f"acc={accuracy_label} reselections={reselect_count} "
                  f"tokens={logical_tokens} id={sample.question_id}" +
                  (f" error={error}" if error else ""), flush=True)

    summary = {
        "schema_version": "longmemeval_session_ingest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "judged": judge is not None,
        "judge_model": judge.model if judge is not None else None,
        "samples": done,
        "correct": correct,
        "accuracy": correct / done if done and judge is not None else None,
        "errors": errors,
        "active_capacity": args.active_capacity,
        "max_tokens": args.max_tokens,
        "wall_seconds": time.monotonic() - started_all,
        "results": str(result_path),
        "request_audit": str(audit_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
