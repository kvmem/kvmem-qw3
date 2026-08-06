#!/usr/bin/env python3
"""Run BEAM-10M against qw3 with one frozen KVMem base per conversation.

The server must be launched with ``--kvmem --kvmem-query-conditioned`` and a
query-replay-compatible Mean-K retrieval method.  The history is submitted once
as a prefill-only local-cache save.  Every probing question then sends only its
short final user message and loads the same version-1 checkpoint in ``frozen``
mode.  qw3 restores the immutable history before the branch and again after it,
so no answer or query from one probe can affect another probe.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .beam_dataset import (
        QUESTION_TYPES,
        BeamConversation,
        BeamQuestion,
        discover_conversation_ids,
        load_conversation,
        parse_id_list,
    )
    from .client import Qw3Client
except ImportError:
    from beam_dataset import (  # type: ignore
        QUESTION_TYPES,
        BeamConversation,
        BeamQuestion,
        discover_conversation_ids,
        load_conversation,
        parse_id_list,
    )
    from client import Qw3Client  # type: ignore


QUESTION_PREFIX = (
    "NOTE: Only provide the answer without any explanations.\n"
    "Question: "
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BEAM-10M from a frozen KVMem history checkpoint"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--use-all", action="store_true")
    parser.add_argument(
        "--conversation-ids",
        default=None,
        help="comma-separated ids/ranges; default: every local conversation",
    )
    parser.add_argument(
        "--question-types",
        default=None,
        help="comma-separated BEAM question types; default: all ten",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="comma-separated zero-based question ordinals after filtering",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum total probing questions after filtering",
    )
    parser.add_argument(
        "--max-history-messages",
        type=int,
        default=None,
        help="diagnostic truncation only; omit for a valid BEAM run",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--read-timeout", type=float, default=7200.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--cache-ttl-seconds", type=int, default=86400)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/chaidi/kvmem_eval/results"),
    )
    parser.add_argument("--tag", default="beam_10m")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append missing questions to this tag's existing JSONL",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.cache_ttl_seconds <= 31536000:
        parser.error("--cache-ttl-seconds must be in [0,31536000]")
    return args


def parse_question_types(value: str | None) -> tuple[str, ...]:
    if value is None:
        return QUESTION_TYPES
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(result) - set(QUESTION_TYPES)
    if unknown:
        raise ValueError(f"unknown BEAM question types: {sorted(unknown)}")
    if not result:
        raise ValueError("--question-types selected no question types")
    return result


def parse_indices(value: str | None) -> set[int] | None:
    if value is None:
        return None
    result = {int(part) for part in value.split(",") if part.strip()}
    if any(index < 0 for index in result):
        raise ValueError("--indices cannot contain negative values")
    return result


def select_conversations(args: argparse.Namespace) -> list[BeamConversation]:
    ids = (
        parse_id_list(args.conversation_ids)
        if args.conversation_ids
        else discover_conversation_ids(args.data)
    )
    if not ids:
        raise ValueError(f"no BEAM conversations found under {args.data}")
    result = [load_conversation(args.data, conversation_id) for conversation_id in ids]
    if args.max_history_messages is not None:
        if args.max_history_messages <= 0:
            raise ValueError("--max-history-messages must be positive")
        result = [
            BeamConversation(
                conversation_id=item.conversation_id,
                messages=item.messages[: args.max_history_messages],
                questions=item.questions,
                raw_questions=item.raw_questions,
                source_dir=item.source_dir,
            )
            for item in result
        ]
    return result


def select_questions(
    conversations: list[BeamConversation], args: argparse.Namespace
) -> list[BeamQuestion]:
    allowed_types = set(parse_question_types(args.question_types))
    questions = [
        question
        for conversation in conversations
        for question in conversation.questions
        if question.question_type in allowed_types
    ]
    indices = parse_indices(args.indices)
    if indices is not None:
        missing = indices - set(range(len(questions)))
        if missing:
            raise IndexError(
                f"--indices outside selected question range 0.."
                f"{max(len(questions) - 1, 0)}: {sorted(missing)}"
            )
        questions = [
            question
            for index, question in enumerate(questions)
            if index in indices
        ]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit cannot be negative")
        questions = questions[: args.limit]
    return questions


def load_existing(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(
                    f"missing question_id in {path}:{line_number}"
                )
            if question_id in completed:
                raise ValueError(
                    f"duplicate question_id in resume file: {question_id}"
                )
            rows.append(row)
            completed.add(question_id)
    return rows, completed


def question_message(question: str) -> tuple[dict[str, str], tuple[int, int]]:
    content = QUESTION_PREFIX + question
    start = len(QUESTION_PREFIX.encode("utf-8"))
    end = start + len(question.encode("utf-8"))
    return {"role": "user", "content": content}, (start, end)


def cache_id(tag: str, conversation_id: str) -> str:
    """Return a stable ID accepted by qw3's process-local cache API."""
    def sanitize(value: str) -> str:
        return "".join(
            character
            if character.isascii() and
            (character.isalnum() or character in "._-:")
            else "_"
            for character in value
        )

    suffix = ":" + sanitize(conversation_id)
    prefix = sanitize(f"beam:{tag}")
    if len(suffix) >= 128:
        return ("beam:" + suffix[-123:])[:128]
    return prefix[: 128 - len(suffix)] + suffix


def write_official_results(
    rows: list[dict[str, Any]], out_dir: Path, tag: str
) -> list[str]:
    grouped: dict[str, OrderedDict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if row.get("client_error"):
            continue
        conversation_id = str(row["conversation_id"])
        question_type = str(row["question_type"])
        official = grouped.setdefault(
            conversation_id,
            OrderedDict((kind, []) for kind in QUESTION_TYPES),
        )
        question = copy.deepcopy(row["benchmark_question"])
        question["llm_response"] = row["answer"]
        official[question_type].append(question)
    paths: list[str] = []
    for conversation_id, payload in grouped.items():
        compact = OrderedDict(
            (kind, questions) for kind, questions in payload.items() if questions
        )
        path = out_dir / f"{tag}_beam_{conversation_id}_official.json"
        path.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    return paths


def main() -> int:
    args = parse_args()
    conversations = select_conversations(args)
    questions = select_questions(conversations, args)
    question_counts = Counter(
        question.conversation_id for question in questions
    )
    history_stats = {
        conversation.conversation_id: {
            "messages": len(conversation.messages),
            "characters": sum(
                len(message["content"]) for message in conversation.messages
            ),
            "selected_questions": question_counts[conversation.conversation_id],
        }
        for conversation in conversations
    }
    print(
        json.dumps(
            {
                "data": str(args.data),
                "conversations": len(conversations),
                "questions": len(questions),
                "diagnostic_history_truncation":
                    args.max_history_messages is not None,
                "history": history_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if not questions:
        raise RuntimeError("no probing questions selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.out_dir / f"{args.tag}_beam_results.jsonl"
    audit_path = args.out_dir / f"{args.tag}_beam_requests.jsonl"
    summary_path = args.out_dir / f"{args.tag}_beam_summary.json"
    existing_rows, completed = load_existing(result_path)
    if result_path.exists() and not args.resume and not args.force:
        raise FileExistsError(
            f"result file already exists: {result_path}; use --resume, --force, "
            "or a new --tag"
        )
    if args.force and not args.resume:
        existing_rows, completed = [], set()
        result_path.unlink(missing_ok=True)
        audit_path.unlink(missing_ok=True)

    client = Qw3Client(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        enable_thinking=not args.no_thinking,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    if not client.health():
        raise RuntimeError(f"qw3 endpoint is not healthy: {args.base_url}")

    selected_by_conversation: dict[str, list[BeamQuestion]] = {
        conversation.conversation_id: [
            question
            for question in questions
            if question.conversation_id == conversation.conversation_id
            and question.question_id not in completed
        ]
        for conversation in conversations
    }
    remaining = sum(len(items) for items in selected_by_conversation.values())
    print(
        f"[beam] selected={len(questions)} completed={len(completed)} "
        f"remaining={remaining} results={result_path}",
        flush=True,
    )

    rows = list(existing_rows)
    failures = 0
    cold_ingests = 0
    started_all = time.monotonic()
    file_mode = "a" if args.resume and result_path.exists() else "w"
    audit_mode = "a" if args.resume and audit_path.exists() else "w"
    with result_path.open(file_mode, encoding="utf-8") as result_output, \
            audit_path.open(audit_mode, encoding="utf-8") as audit_output:
        ordinal = len(completed)
        for conversation in conversations:
            pending = selected_by_conversation[conversation.conversation_id]
            if not pending:
                continue
            messages = list(conversation.messages)
            frozen_cache_id = cache_id(args.tag, conversation.conversation_id)
            prefill_started = time.monotonic()
            prefill_result = client.chat(
                messages,
                max_tokens=0,
                enable_thinking=not args.no_thinking,
                extra_body={
                    "kvmem_reselect": "off",
                    "kvmem_cache": {
                        "save": {
                            "id": frozen_cache_id,
                            "scope": "local",
                            "when": "after_request",
                            "ttl_seconds": args.cache_ttl_seconds,
                        }
                    },
                    "kvmem_trace_tag":
                        f"beam-{conversation.conversation_id}-history",
                },
            )
            cold_ingests += 1
            audit_output.write(
                json.dumps(
                    {
                        "conversation_id": conversation.conversation_id,
                        "phase": "history_ingest",
                        "message_count": len(messages),
                        "history_characters": history_stats[
                            conversation.conversation_id
                        ]["characters"],
                        "prompt_tokens": prefill_result.prompt_tokens,
                        "completion_tokens": prefill_result.completion_tokens,
                        "finish_reason": prefill_result.finish_reason,
                        "latency_s": prefill_result.latency_s,
                        "client_error": prefill_result.error,
                        "cache_id": frozen_cache_id,
                        "cache_version": 1,
                        "cache_mode": "save",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            audit_output.flush()
            if (
                prefill_result.error
                or prefill_result.finish_reason != "prefill_only"
                or prefill_result.answer
                or prefill_result.reasoning
                or (prefill_result.completion_tokens or 0) != 0
            ):
                failures += len(pending)
                message = (
                    f"history ingest failed for conversation "
                    f"{conversation.conversation_id}: "
                    f"error={prefill_result.error!r} "
                    f"finish={prefill_result.finish_reason!r}"
                )
                if args.fail_fast:
                    raise RuntimeError(message)
                print(f"[beam] ERROR {message}", flush=True)
                continue
            print(
                f"[beam] ingested conversation={conversation.conversation_id} "
                f"messages={len(messages)} "
                f"tokens={prefill_result.prompt_tokens} "
                f"latency={time.monotonic() - prefill_started:.1f}s",
                flush=True,
            )

            for question in pending:
                ordinal += 1
                final_message, (query_start, query_end) = question_message(
                    question.question
                )
                request_messages = [final_message]
                result = client.chat(
                    request_messages,
                    extra_body={
                        "kvmem_reselect": "force",
                        "kvmem_cache": {
                            "load": {
                                "id": frozen_cache_id,
                                "mode": "frozen",
                                "required": True,
                                "expected_version": 1,
                            }
                        },
                        "kvmem_query_span": {
                            "message_index": 0,
                            "content_start": query_start,
                            "content_end": query_end,
                        },
                        "kvmem_trace_tag": (
                            f"beam-{question.conversation_id}-"
                            f"{question.question_type}-"
                            f"{question.question_index}"
                        ),
                    },
                )
                error = result.error
                if error:
                    failures += 1
                row = {
                    "schema_version": "beam_10m_kvmem.v1",
                    "conversation_id": question.conversation_id,
                    "question_id": question.question_id,
                    "question_type": question.question_type,
                    "question_index": question.question_index,
                    "question": question.question,
                    "gold": question.reference,
                    "rubric": list(question.rubric),
                    "benchmark_question": question.raw,
                    "answer": result.answer,
                    "reasoning": result.reasoning
                        if os.environ.get("QW3_EVAL_STORE_REASONING") == "1"
                        else None,
                    "reasoning_chars": len(result.reasoning),
                    "ttft_s": result.ttft_s,
                    "latency_s": result.latency_s,
                    "finish_reason": result.finish_reason,
                    "truncated": result.truncated,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "client_error": error,
                    "history_messages": len(messages),
                    "history_characters": history_stats[
                        conversation.conversation_id
                    ]["characters"],
                    "diagnostic_history_truncation":
                        args.max_history_messages is not None,
                }
                rows.append(row)
                result_output.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
                result_output.flush()
                audit_output.write(
                    json.dumps(
                        {
                            "conversation_id": question.conversation_id,
                            "question_id": question.question_id,
                            "phase": "probing_question",
                            "message_count": len(request_messages),
                            "query_content_start": query_start,
                            "query_content_end": query_end,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "finish_reason": result.finish_reason,
                            "ttft_s": result.ttft_s,
                            "latency_s": result.latency_s,
                            "client_error": error,
                            "cache_id": frozen_cache_id,
                            "cache_version": 1,
                            "cache_mode": "frozen",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                audit_output.flush()
                print(
                    f"[{ordinal:>3}/{len(questions)}] "
                    f"{question.question_type:<25} "
                    f"finish={result.finish_reason:<8} "
                    f"ttft={result.ttft_s} lat={result.latency_s:.1f}s "
                    f"id={question.question_id}"
                    + (f" error={error}" if error else ""),
                    flush=True,
                )
                if error and args.fail_fast:
                    raise RuntimeError(
                        f"question request failed: {question.question_id}: "
                        f"{error}"
                    )

    official_paths = write_official_results(rows, args.out_dir, args.tag)
    per_type = Counter(
        str(row["question_type"])
        for row in rows
        if not row.get("client_error")
    )
    summary = {
        "schema_version": "beam_10m_kvmem.summary.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data),
        "base_url": args.base_url,
        "model": args.model,
        "tag": args.tag,
        "selected_questions": len(questions),
        "result_rows": len(rows),
        "successful_rows": sum(
            not bool(row.get("client_error")) for row in rows
        ),
        "failures_this_run": failures,
        "cold_history_ingests_this_run": cold_ingests,
        "expected_frozen_cache_loads_this_run": remaining,
        "per_type_successful": {
            question_type: per_type[question_type]
            for question_type in QUESTION_TYPES
        },
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": not args.no_thinking,
        "diagnostic_history_truncation":
            args.max_history_messages is not None,
        "wall_seconds": time.monotonic() - started_all,
        "results": str(result_path),
        "request_audit": str(audit_path),
        "official_result_files": official_paths,
        "official_grading": (
            "Run BEAM's src.evaluation.run_evaluation on each official result "
            "file; this harness intentionally does not replace its type-specific "
            "rubric evaluator."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
