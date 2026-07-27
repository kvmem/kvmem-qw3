#!/usr/bin/env python3
"""Incrementally ingest AgentLongBench one logical turn at a time.

This is an experimental alternative to ``run_agentlongbench_kvmem.py``.  It
keeps the canonical dataset, task instruction, and official evaluator, but
changes how history reaches qw3:

* the dataset's original role boundaries are preserved;
* historical assistant/tool messages are teacher-forced with ``max_tokens=0``;
* before active-capacity pressure, each logical turn is one prefill append;
* after pressure, the first user message in each turn triggers mean-K
  reselection/query replay, and the rest of that turn is prefetched under the
  resulting fixed selected window;
* the final benchmark question is a separate user message and always triggers
  one last reselection before normal decoding.

The script contains no AgentLongBench-specific policy in qw3 itself.  Dataset
turn splitting, pressure timing, and retrieval-query choice all remain in this
evaluation harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_agentlongbench_kvmem as common  # noqa: E402


DEFAULT_DATASET = Path(
    "/data/chaidi/kvmem_eval/data/agentlongbench_512k_normal100/samples.jsonl"
)
DEFAULT_MANIFEST = Path(
    "/home/chaidi/AgentLongBench-Long/results/"
    "agentlongbench_512k_normal100/compact_only_normal100/"
    "manifest/selected_samples.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/data/chaidi/kvmem_eval/results/"
    "agentlongbench_512k_normal100_turn_ingest_k200k"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run role-preserving, per-turn AgentLongBench KVMem ingest"
    )
    parser.add_argument(
        "--benchmark-repo",
        type=Path,
        default=common.DEFAULT_BENCHMARK_REPO,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--benchmark-name",
        default="AgentLongBench-512K-normal100-turn-ingest",
    )
    parser.add_argument("--allow-custom-subset", action="store_true")
    parser.add_argument("--api-base", default="http://127.0.0.1:18087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument(
        "--method",
        default="kvmem_turn_ingest_mean_k_200k_b32_query_replay_immutable_mtp4",
    )
    parser.add_argument(
        "--active-capacity",
        type=int,
        default=237568,
        help=(
            "logical prompt tokens that must already be ingested before a new "
            "turn starts semantic reselection; normally context budget + "
            "generation reserve"
        ),
    )
    parser.add_argument(
        "--round-query",
        choices=("first_user", "whole_round"),
        default="first_user",
        help=(
            "query used for each above-capacity turn reselection; "
            "whole_round is an experimental role-preserving ablation"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--max-sample-sec", type=int, default=7200)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def raw_sample(sample: dict[str, Any]) -> dict[str, Any]:
    raw = sample.get("raw")
    return raw if isinstance(raw, dict) else {}


def sample_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = sample.get("messages") or raw_sample(sample).get("messages") or []
    if not isinstance(messages, list):
        raise RuntimeError("sample messages must be a list")
    return [normalize_message(message) for message in messages]


def normalize_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise RuntimeError("AgentLongBench history contains a non-object message")
    role = str(message.get("role") or "").strip().lower()
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise RuntimeError(f"unsupported AgentLongBench message role: {role!r}")
    normalized: dict[str, Any] = {
        "role": role,
        "content": str(message.get("content") or ""),
    }
    # Preserve the fields that affect Qwen's role/tool transcript rendering.
    if role == "assistant":
        if isinstance(message.get("tool_calls"), list):
            normalized["tool_calls"] = message["tool_calls"]
        if isinstance(message.get("reasoning_content"), str):
            normalized["reasoning_content"] = message["reasoning_content"]
    elif role == "tool":
        for key in ("tool_call_id", "name"):
            if message.get(key) is not None:
                normalized[key] = str(message[key])
    return normalized


def split_history_turns(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """Return non-system prelude and user-led logical turns.

    AgentLongBench normally begins with an assistant guess.  That assistant
    message is prelude.  Every subsequent user observation starts a turn that
    includes all following assistant/tool messages up to the next user.
    """

    non_system = [
        message
        for message in messages
        if message["role"] not in {"system", "developer"}
    ]
    prelude: list[dict[str, Any]] = []
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for message in non_system:
        if message["role"] == "user":
            if current is not None:
                turns.append(current)
            current = [message]
        elif current is None:
            prelude.append(message)
        else:
            current.append(message)
    if current is not None:
        turns.append(current)
    if not turns:
        raise RuntimeError("AgentLongBench sample contains no user-led history turn")
    for index, turn in enumerate(turns):
        if not turn or turn[0]["role"] != "user":
            raise RuntimeError(f"history turn {index} does not begin with a user")
    return prelude, turns


def question_text(sample: dict[str, Any]) -> str:
    question = str(sample.get("question") or raw_sample(sample).get("question") or "")
    if not question:
        raise RuntimeError("AgentLongBench sample has an empty final question")
    return question


def full_content_span(message_index: int, content: str) -> dict[str, int]:
    return {
        "message_index": message_index,
        "content_start": 0,
        "content_end": len(content.encode("utf-8")),
    }


def embedded_content_span(
    message_index: int, content: str, needle: str
) -> dict[str, int]:
    char_start = content.rfind(needle)
    if char_start < 0:
        raise RuntimeError("final question is not present in final user content")
    byte_start = len(content[:char_start].encode("utf-8"))
    return {
        "message_index": message_index,
        "content_start": byte_start,
        "content_end": byte_start + len(needle.encode("utf-8")),
    }


def canonical_task_system(canonical: Any, sample: dict[str, Any]) -> str:
    qtype = canonical.question_type(sample)
    knowledge_label, history_label = canonical.knowledge_history_labels(sample)
    return canonical.build_system_prompt(qtype, history_label, knowledge_label)


def history_sha256(messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SessionClient:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str,
        operation: str,
        max_tokens: int,
        reselect: str,
        prefill_window: str,
        query_span: dict[str, int] | None,
        query_message_range: dict[str, int] | None,
        trace_tag: str,
    ) -> tuple[dict[str, Any], float]:
        payload: dict[str, Any] = {
            "model": self.args.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_tokens": max_tokens,
            "enable_thinking": self.args.enable_thinking,
            "stream": False,
            "kvmem_session_id": session_id,
            "kvmem_session_op": operation,
            "kvmem_reselect": reselect,
            "kvmem_prefill_window": prefill_window,
            "kvmem_trace_tag": trace_tag,
        }
        if query_span is not None:
            payload["kvmem_query_span"] = query_span
        if query_message_range is not None:
            payload["kvmem_query_message_range"] = query_message_range
        if self.args.seed is not None:
            payload["seed"] = self.args.seed
        request = urllib.request.Request(
            self.args.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer local-kvmem-eval",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self.http.open(request, timeout=self.args.timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {text[:4000]}") from exc
        return body, time.perf_counter() - started


def response_fields(
    body: dict[str, Any],
) -> tuple[str, str, str, int, int]:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("chat response contains no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    return (
        str(message.get("content") or ""),
        str(message.get("reasoning_content") or message.get("reasoning") or ""),
        str(choice.get("finish_reason") or ""),
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
    )


def write_audit(handle: Any, row: dict[str, Any]) -> None:
    handle.write(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    handle.flush()


def prefill(
    client: SessionClient,
    audit: Any,
    messages: list[dict[str, Any]],
    *,
    sid: str,
    session_id: str,
    attempt: int,
    turn_index: int | str,
    phase: str,
    operation: str,
    reselect: str,
    prefill_window: str,
    query_span: dict[str, int] | None = None,
    query_message_range: dict[str, int] | None = None,
) -> int:
    trace_tag = (
        f"alb-{sid[:20]}-a{attempt}-t{turn_index}-{phase}"
    )[:128]
    body, latency = client.chat(
        messages,
        session_id=session_id,
        operation=operation,
        max_tokens=0,
        reselect=reselect,
        prefill_window=prefill_window,
        query_span=query_span,
        query_message_range=query_message_range,
        trace_tag=trace_tag,
    )
    answer, reasoning, finish_reason, prompt_tokens, completion_tokens = (
        response_fields(body)
    )
    if (
        answer
        or reasoning
        or completion_tokens != 0
        or finish_reason != "prefill_only"
    ):
        raise RuntimeError(
            "prefill-only contract violated: "
            f"answer_chars={len(answer)} reasoning_chars={len(reasoning)} "
            f"completion_tokens={completion_tokens} "
            f"finish_reason={finish_reason!r}"
        )
    write_audit(
        audit,
        {
            "time": common.now_iso(),
            "stable_sample_id": sid,
            "attempt": attempt,
            "turn_index": turn_index,
            "phase": phase,
            "operation": operation,
            "reselect": reselect,
            "prefill_window": prefill_window,
            "message_count": len(messages),
            "roles": [message["role"] for message in messages],
            "query_span": query_span,
            "query_message_range": query_message_range,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "latency_sec": latency,
            "trace_tag": trace_tag,
        },
    )
    return prompt_tokens


def run_sample(
    args: argparse.Namespace,
    canonical: Any,
    client: SessionClient,
    audit: Any,
    sample: dict[str, Any],
    *,
    attempt: int,
) -> dict[str, Any]:
    sample_started = time.perf_counter()

    def check_sample_deadline() -> None:
        elapsed = time.perf_counter() - sample_started
        if elapsed > args.max_sample_sec:
            raise TimeoutError(
                f"sample exceeded --max-sample-sec={args.max_sample_sec}"
            )

    sid = str(sample["stable_sample_id"])
    session_id = (
        f"alb-{sid[:24]}-{int(time.time() * 1000)}-a{attempt}"
    )
    messages = sample_messages(sample)
    prelude, turns = split_history_turns(messages)
    system = {
        "role": "system",
        "content": canonical_task_system(canonical, sample),
    }
    logical_prompt_tokens = prefill(
        client,
        audit,
        [system, *prelude],
        sid=sid,
        session_id=session_id,
        attempt=attempt,
        turn_index="bootstrap",
        phase="system_prelude",
        operation="start",
        reselect="off",
        prefill_window="pressure",
    )

    semantic_reselections = 0
    first_pressure_turn: int | None = None
    prefill_request_count = 1
    for turn_index, turn in enumerate(turns):
        check_sample_deadline()
        pressure_reached = logical_prompt_tokens >= args.active_capacity
        if not pressure_reached:
            logical_prompt_tokens += prefill(
                client,
                audit,
                turn,
                sid=sid,
                session_id=session_id,
                attempt=attempt,
                turn_index=turn_index,
                phase="whole_turn",
                operation="append",
                reselect="off",
                prefill_window="pressure",
            )
            prefill_request_count += 1
            continue

        if first_pressure_turn is None:
            first_pressure_turn = turn_index
        if args.round_query == "whole_round":
            logical_prompt_tokens += prefill(
                client,
                audit,
                turn,
                sid=sid,
                session_id=session_id,
                attempt=attempt,
                turn_index=turn_index,
                phase="whole_round_retrieval",
                operation="append",
                reselect="force",
                prefill_window="pressure",
                query_message_range={
                    "message_begin": 0,
                    "message_end": len(turn),
                },
            )
            semantic_reselections += 1
            prefill_request_count += 1
            continue

        user = turn[0]
        user_content = str(user.get("content") or "")
        if not user_content:
            raise RuntimeError(
                f"turn {turn_index} has an empty first user retrieval query"
            )
        logical_prompt_tokens += prefill(
            client,
            audit,
            [user],
            sid=sid,
            session_id=session_id,
            attempt=attempt,
            turn_index=turn_index,
            phase="retrieval_query",
            operation="append",
            reselect="force",
            prefill_window="pressure",
            query_span=full_content_span(0, user_content),
        )
        semantic_reselections += 1
        prefill_request_count += 1

        remainder = turn[1:]
        if remainder:
            logical_prompt_tokens += prefill(
                client,
                audit,
                remainder,
                sid=sid,
                session_id=session_id,
                attempt=attempt,
                turn_index=turn_index,
                phase="turn_remainder",
                operation="append",
                reselect="off",
                prefill_window="keep_selected",
            )
            prefill_request_count += 1

    question = question_text(sample)
    check_sample_deadline()
    final_content = f"Question:\n{question}\n\nAnswer:"
    final_span = embedded_content_span(0, final_content, question)
    final_trace_tag = f"alb-{sid[:20]}-a{attempt}-final"[:128]
    final_started = time.perf_counter()
    body, final_latency = client.chat(
        [{"role": "user", "content": final_content}],
        session_id=session_id,
        operation="finish",
        max_tokens=args.max_tokens,
        reselect="force",
        prefill_window="pressure",
        query_span=final_span,
        query_message_range=None,
        trace_tag=final_trace_tag,
    )
    total_latency = time.perf_counter() - final_started
    answer, reasoning, finish_reason, prompt_tokens, completion_tokens = (
        response_fields(body)
    )
    logical_prompt_tokens += prompt_tokens
    semantic_reselections += 1
    write_audit(
        audit,
        {
            "time": common.now_iso(),
            "stable_sample_id": sid,
            "attempt": attempt,
            "turn_index": "final",
            "phase": "final_question",
            "operation": "finish",
            "reselect": "force",
            "prefill_window": "pressure",
            "message_count": 1,
            "roles": ["user"],
            "query_span": final_span,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
            "latency_sec": final_latency,
            "trace_tag": final_trace_tag,
        },
    )
    return {
        "hypothesis": answer.strip(),
        "raw_response": answer,
        "reasoning": reasoning,
        "usage": {
            "prompt_tokens": logical_prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "finish_reason": finish_reason,
        "timing": {
            "ttft_sec": None,
            "total_sec": total_latency,
        },
        "prompt_tokens": logical_prompt_tokens,
        "generation_max_tokens_request": args.max_tokens,
        "history_messages": len(messages),
        "history_turns": len(turns),
        "history_prelude_messages": len(prelude),
        "first_pressure_turn_index": first_pressure_turn,
        "semantic_reselections": semantic_reselections,
        "prefill_request_count": prefill_request_count,
        "history_role_sha256": history_sha256(messages),
        "round_query": args.round_query,
    }


def load_selected_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = common.read_jsonl(args.dataset)
    manifest = common.read_jsonl(args.manifest)
    if len(samples) != len(manifest):
        raise RuntimeError(
            f"samples/manifest length mismatch: {len(samples)} != {len(manifest)}"
        )
    sample_ids = [str(row.get("stable_sample_id") or "") for row in samples]
    manifest_ids = [str(row.get("stable_sample_id") or "") for row in manifest]
    if any(not sid for sid in sample_ids) or sample_ids != manifest_ids:
        raise RuntimeError(
            "samples and manifest must contain identical stable IDs in order"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("samples contain duplicate stable_sample_id values")
    if not args.allow_custom_subset and len(samples) != 100:
        raise RuntimeError(
            f"normal100 turn-ingest expects 100 samples, got {len(samples)}; "
            "pass --allow-custom-subset for a separately validated subset"
        )
    if args.question_id:
        wanted = set(args.question_id)
        samples = [
            sample
            for sample in samples
            if str(sample.get("stable_sample_id")) in wanted
            or str(sample.get("id") or raw_sample(sample).get("id") or "") in wanted
        ]
    elif args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no samples selected")
    return samples


def main() -> None:
    args = parse_args()
    if args.active_capacity <= 0:
        raise RuntimeError("--active-capacity must be positive")
    if args.max_tokens <= 0:
        raise RuntimeError("--max-tokens must be positive for the final decode")
    canonical = common.load_canonical_module(args.benchmark_repo)
    samples = load_selected_samples(args)
    if not common.health_ok(args.api_base):
        raise RuntimeError(f"qw3 server is not healthy at {args.api_base}")

    output = common.paths(args.output_root)
    audit_path = args.output_root / "request_audit.jsonl"
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": args.benchmark_name,
        "method": args.method,
        "prompt_mode": "role_preserving_incremental_turn_ingest",
        "turn_policy": "first_user_message_starts_each_logical_turn",
        "pressure_policy": "prior_logical_tokens_gte_active_capacity",
        "retrieval_query_policy": (
            "full_role_preserving_round"
            if args.round_query == "whole_round"
            else "full_first_user_content"
        ),
        "post_reselection_policy": (
            "whole_round_query_replay"
            if args.round_query == "whole_round"
            else "turn_remainder_keep_selected"
        ),
        "final_query_policy": "separate_user_message_force_reselect",
        "dataset_system_policy": "replace_with_canonical_task_system",
        "dataset": str(args.dataset),
        "canonical_manifest": str(args.manifest),
        "canonical_worker": str(
            args.benchmark_repo
            / "fullcontext"
            / "run_agentlongbench_fullcontext_worker.py"
        ),
        "api_base": args.api_base,
        "model": args.model,
        "active_capacity": args.active_capacity,
        "round_query": args.round_query,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "seed": args.seed,
        "allow_custom_subset": args.allow_custom_subset,
        "selected_samples": len(samples),
        "started_at": common.now_iso(),
    }
    if output["config"].exists():
        previous = json.loads(output["config"].read_text(encoding="utf-8"))
        comparable = tuple(key for key in config if key != "started_at")
        if any(previous.get(key) != config.get(key) for key in comparable):
            raise RuntimeError(
                "output root already contains a different run configuration"
            )
    else:
        common.write_json(output["config"], config)

    answers = common.latest_by_id(common.read_jsonl(output["answers"]))
    evals = common.latest_by_id(common.read_jsonl(output["eval"]))
    manifest_ids = {
        str(row["stable_sample_id"])
        for row in common.read_jsonl(output["manifest"])
        if row.get("stable_sample_id")
    }
    client = SessionClient(args)
    total = len(samples)
    with audit_path.open("a", encoding="utf-8") as audit:
        for index, sample in enumerate(samples, start=1):
            sid = str(sample["stable_sample_id"])
            if sid in evals:
                print(f"[skip] {index}/{total} {sid}", flush=True)
                continue
            answer_row = answers.get(sid)
            if answer_row is None:
                canonical_prompt = canonical.full_context_prompt(sample)
                canonical_prompt_sha256 = hashlib.sha256(
                    canonical_prompt.encode("utf-8")
                ).hexdigest()
                base = {
                    **common.base_row(
                        index,
                        total,
                        sample,
                        args.method,
                        args.benchmark_name,
                    ),
                    "canonical_prompt_sha256": canonical_prompt_sha256,
                    "prompt_mode": "role_preserving_incremental_turn_ingest",
                    "active_capacity": args.active_capacity,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "enable_thinking": args.enable_thinking,
                }
                if sid not in manifest_ids:
                    common.append_jsonl(output["manifest"], base)
                    manifest_ids.add(sid)
                common.append_jsonl(
                    output["status"],
                    {
                        "time": common.now_iso(),
                        "stable_sample_id": sid,
                        "index": index,
                        "status": "started",
                    },
                )
                last_error: Exception | None = None
                for attempt in range(1, args.attempts + 1):
                    sample_started = time.perf_counter()
                    try:
                        result = run_sample(
                            args,
                            canonical,
                            client,
                            audit,
                            sample,
                            attempt=attempt,
                        )
                        answer_row = {
                            **base,
                            **result,
                            "sample_wall_seconds": (
                                time.perf_counter() - sample_started
                            ),
                            "answered_at": common.now_iso(),
                        }
                        common.append_jsonl(output["answers"], answer_row)
                        answers[sid] = answer_row
                        common.append_jsonl(
                            output["status"],
                            {
                                "time": common.now_iso(),
                                "stable_sample_id": sid,
                                "index": index,
                                "status": "answered",
                                "attempt": attempt,
                                "finish_reason": answer_row.get("finish_reason"),
                                "history_turns": answer_row.get("history_turns"),
                                "semantic_reselections": answer_row.get(
                                    "semantic_reselections"
                                ),
                            },
                        )
                        break
                    except Exception as exc:
                        last_error = exc
                        common.append_jsonl(
                            output["status"],
                            {
                                "time": common.now_iso(),
                                "stable_sample_id": sid,
                                "index": index,
                                "status": "answer_failed",
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "error": str(exc)[:4000],
                            },
                        )
                        print(
                            f"[error] {index}/{total} attempt={attempt} "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        if not common.health_ok(args.api_base):
                            raise RuntimeError(
                                "qw3 server stopped during turn-ingest evaluation"
                            ) from exc
                        time.sleep(min(30, 2 * attempt))
                if answer_row is None:
                    raise RuntimeError(
                        f"all attempts failed for {sid}: {last_error}"
                    )

            evaluated = canonical.evaluate_response(
                sample,
                answer_row.get("raw_response")
                or answer_row.get("hypothesis")
                or "",
            )
            eval_row = {
                **{
                    key: answer_row.get(key)
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
                        "canonical_prompt_sha256",
                        "prompt_tokens",
                        "history_turns",
                        "first_pressure_turn_index",
                        "semantic_reselections",
                        "finish_reason",
                    )
                },
                **evaluated,
                "evaluated_at": common.now_iso(),
            }
            common.append_jsonl(output["eval"], eval_row)
            evals[sid] = eval_row
            common.append_jsonl(
                output["status"],
                {
                    "time": common.now_iso(),
                    "stable_sample_id": sid,
                    "index": index,
                    "status": "evaluated",
                    "score": eval_row.get("score"),
                    "correct": eval_row.get("correct"),
                },
            )
            print(
                f"[eval] {index}/{total} score={eval_row.get('score')} "
                f"correct={eval_row.get('correct')} "
                f"turns={answer_row.get('history_turns')} "
                f"reselect={answer_row.get('semantic_reselections')}",
                flush=True,
            )
            common.write_progress(args.output_root, samples, index)

    common.write_final(
        args.output_root,
        samples,
        args.method,
        args.benchmark_name,
    )
    print(f"[complete] results={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
