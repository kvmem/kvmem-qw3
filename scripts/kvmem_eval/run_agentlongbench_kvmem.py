#!/usr/bin/env python3
"""Run canonical AgentLongBench FullContext prompts through qw3 KVMem.

Prompt construction and answer grading are imported from the benchmark's
canonical FullContext worker.  This runner changes only transport: it marks the
embedded final question with qw3's optional kvmem_query_span request field.
All artifacts are append-only/resumable and are written to a new output root.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_BENCHMARK_REPO = Path("/home/chaidi/AgentLongBench_Motivation")
DEFAULT_DATASET = Path("/data/chaidi/kvmem_eval/data/agentlongbench_250/samples.jsonl")
DEFAULT_MANIFEST = Path("/data/chaidi/kvmem_eval/data/agentlongbench_250/manifest.jsonl")
DEFAULT_OUTPUT = Path("/data/chaidi/kvmem_eval/results/agentlongbench_kvmem_32k_b32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentLongBench-250 KVMem evaluator")
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--benchmark-name",
        default="AgentLongBench-500-long250",
        help="artifact label; does not change prompt construction or evaluation",
    )
    parser.add_argument(
        "--allow-custom-subset",
        action="store_true",
        help="allow a validated samples/manifest pair other than canonical long250",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--method", default="kvmem_mean_k_32k_b32")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--context-window", type=int, default=262144)
    parser.add_argument("--context-safety-margin", type=int, default=16)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--max-sample-sec", type=int, default=2400)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--kvmem-retrieval-trace-metadata",
        action="store_true",
        help="send diagnostics-only context span and stable sample trace tag",
    )
    parser.add_argument(
        "--kvmem-round-only",
        action="store_true",
        help=(
            "group the canonical flattened history into complete rounds and "
            "send kvmem_retrieval_group_spans; requires a server launched "
            "with --kvmem-round-retrieval"
        ),
    )
    parser.add_argument(
        "--kvmem-message-expansion",
        action="store_true",
        help=(
            "group the canonical flattened history into complete rendered "
            "messages and send kvmem_retrieval_group_spans; requires "
            "--kvmem-semantic-expansion message on the server"
        ),
    )
    parser.add_argument(
        "--kvmem-inline-refresh",
        choices=("kv_only", "kv_and_state"),
        help=(
            "diagnostic one-shot refresh of the final selected context; "
            "requires QW3_KVMEM_ENABLE_INLINE_REFRESH=1 on the server"
        ),
    )
    parser.add_argument(
        "--kvmem-round-padding",
        type=int,
        default=0,
        metavar="TOKENS",
        help=(
            "experimental token-level newline alignment for round retrieval; "
            "0 disables it, otherwise the server pads every round to this "
            "token multiple (normally the KVMem block size)"
        ),
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_canonical_module(repo: Path) -> Any:
    source = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("agentlongbench_fullcontext", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import canonical worker: {source}")
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
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
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


def paths(root: Path) -> dict[str, Path]:
    return {
        "answers": root / "answers.jsonl",
        "eval": root / "eval.jsonl",
        "manifest": root / "manifest.jsonl",
        "status": root / "status_events.jsonl",
        "progress": root / "progress.json",
        "summary": root / "accuracy_summary.json",
        "validation": root / "validation_report.json",
        "config": root / "run_config.json",
    }


def latest_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["stable_sample_id"]): row
        for row in rows
        if row.get("stable_sample_id")
    }


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health_ok(api_base: str) -> bool:
    url = api_base.removesuffix("/v1").rstrip("/") + "/health"
    try:
        with opener().open(url, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def post_json(url: str, payload: dict[str, Any], timeout: int) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener().open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc


def tokenize_count(api_base: str, prompt: str, timeout: int) -> int:
    url = api_base.removesuffix("/v1").rstrip("/") + "/tokenize"
    result = post_json(url, {"content": prompt}, timeout)
    if isinstance(result.get("count"), int):
        return int(result["count"])
    if isinstance(result.get("tokens"), list):
        return len(result["tokens"])
    raise RuntimeError(f"unexpected /tokenize response: {result}")


def query_byte_span(sample: dict[str, Any], prompt: str) -> tuple[int, int]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    question = str(sample.get("question") or raw.get("question") or "")
    if not question:
        raise RuntimeError("sample has an empty final question")
    suffix = f"Question:\n{question}\n\nAnswer:"
    suffix_start = prompt.rfind(suffix)
    if suffix_start < 0:
        raise RuntimeError("canonical prompt does not contain the expected final question suffix")
    char_start = suffix_start + len("Question:\n")
    char_end = char_start + len(question)
    if prompt[char_start:char_end] != question:
        raise RuntimeError("question character span does not round-trip")
    byte_start = len(prompt[:char_start].encode("utf-8"))
    byte_end = byte_start + len(question.encode("utf-8"))
    return byte_start, byte_end


def history_byte_span(
    canonical: Any, sample: dict[str, Any], prompt: str
) -> tuple[int, int]:
    """Return the exact canonical flattened-history byte span.

    The FullContext worker joins rendered non-system messages with one newline.
    RAG's chunk source uses two newlines; the analysis tool deliberately maps
    both forms through their shared message/text coordinates instead of treating
    their token offsets as interchangeable.
    """
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    messages = sample.get("messages") or raw.get("messages") or []
    lines = [
        canonical.message_to_text(message)
        for message in messages
        if not (isinstance(message, dict) and message.get("role") == "system")
    ]
    history = "\n".join(lines)
    marker = "Full conversation history:\n"
    marker_start = prompt.find(marker)
    if marker_start < 0:
        raise RuntimeError("canonical prompt is missing the history marker")
    char_start = marker_start + len(marker)
    char_end = char_start + len(history)
    if prompt[char_start:char_end] != history:
        raise RuntimeError("flattened history character span does not round-trip")
    if prompt.find(marker, marker_start + 1) >= 0:
        raise RuntimeError("canonical prompt has an ambiguous duplicate history marker")
    byte_start = len(prompt[:char_start].encode("utf-8"))
    byte_end = byte_start + len(history.encode("utf-8"))
    return byte_start, byte_end


def round_byte_spans(
    canonical: Any, sample: dict[str, Any], prompt: str
) -> list[tuple[int, int]]:
    """Partition the canonical flattened history into complete round spans.

    AgentLongBench closes each round with a user feedback message. A round thus
    contains the assistant reasoning/tool call, tool result, candidate answer,
    and the feedback that labels that answer. The first round can contain only
    the initial assistant candidate plus feedback. Newline separators are
    assigned to the following round so the spans form an exact, gap-free
    partition of the history bytes.
    """
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    messages = sample.get("messages") or raw.get("messages") or []
    rendered: list[tuple[dict[str, Any], str]] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            continue
        rendered.append((message, canonical.message_to_text(message)))
    if not rendered:
        raise RuntimeError("sample has no non-system history messages")

    history_begin, history_end = history_byte_span(canonical, sample, prompt)
    history_parts: list[str] = []
    local_cursor = 0
    group_begin = 0
    local_groups: list[tuple[int, int]] = []
    user_messages = 0
    for index, (message, text) in enumerate(rendered):
        if index > 0:
            history_parts.append("\n")
            local_cursor += 1
        history_parts.append(text)
        local_cursor += len(text.encode("utf-8"))
        if isinstance(message, dict) and message.get("role") == "user":
            if local_cursor <= group_begin:
                raise RuntimeError("empty AgentLongBench round")
            local_groups.append((group_begin, local_cursor))
            group_begin = local_cursor
            user_messages += 1

    history = "".join(history_parts)
    expected_bytes = history_end - history_begin
    if len(history.encode("utf-8")) != expected_bytes:
        raise RuntimeError("round reconstruction does not match history byte span")
    if group_begin < local_cursor:
        # Defensive support for a trailing assistant/tool segment. It belongs
        # to the final chronological round rather than becoming a partial
        # independently retrievable unit.
        if not local_groups:
            raise RuntimeError("history contains no user round boundary")
        local_groups[-1] = (local_groups[-1][0], local_cursor)
    if not local_groups or len(local_groups) != user_messages:
        raise RuntimeError("could not derive AgentLongBench round boundaries")
    if local_groups[0][0] != 0 or local_groups[-1][1] != local_cursor:
        raise RuntimeError("round spans do not cover the complete history")
    for previous, current in zip(local_groups, local_groups[1:]):
        if previous[1] != current[0]:
            raise RuntimeError("round spans contain a gap or overlap")
    return [
        (history_begin + begin, history_begin + end)
        for begin, end in local_groups
    ]


def message_byte_spans(
    canonical: Any, sample: dict[str, Any], prompt: str
) -> list[tuple[int, int]]:
    """Partition canonical flattened history into complete message spans.

    Newline separators are assigned to the following message, matching the
    round-span convention. The result is gap-free, ordered, and generic across
    roles; it is derived by the benchmark runner rather than hard-coded in
    KVMem.
    """
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    messages = sample.get("messages") or raw.get("messages") or []
    lines = [
        canonical.message_to_text(message)
        for message in messages
        if not (isinstance(message, dict) and message.get("role") == "system")
    ]
    if not lines:
        raise RuntimeError("sample has no non-system history messages")
    history_begin, history_end = history_byte_span(canonical, sample, prompt)
    cursor = 0
    local_groups: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        begin = cursor
        if index > 0:
            cursor += 1  # the canonical "\n".join separator
        cursor += len(line.encode("utf-8"))
        local_groups.append((begin, cursor))
    if cursor != history_end - history_begin:
        raise RuntimeError("message reconstruction does not match history span")
    return [
        (history_begin + begin, history_begin + end)
        for begin, end in local_groups
    ]


def chat_completion(
    args: argparse.Namespace,
    prompt: str,
    query_span: tuple[int, int],
    context_span: tuple[int, int] | None,
    retrieval_group_spans: list[tuple[int, int]] | None,
    trace_tag: str | None,
    request_max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": request_max_tokens,
        "enable_thinking": args.enable_thinking,
        "stream": True,
        "stream_options": {"include_usage": True},
        "kvmem_query_span": {
            "message_index": 0,
            "content_start": query_span[0],
            "content_end": query_span[1],
        },
    }
    if context_span is not None and trace_tag is not None:
        payload["kvmem_context_span"] = {
            "message_index": 0,
            "content_start": context_span[0],
            "content_end": context_span[1],
        }
        payload["kvmem_trace_tag"] = trace_tag
    if retrieval_group_spans is not None:
        payload["kvmem_retrieval_group_spans"] = [
            {
                "message_index": 0,
                "content_start": begin,
                "content_end": end,
            }
            for begin, end in retrieval_group_spans
        ]
        if args.kvmem_round_padding > 0:
            payload["kvmem_round_padding"] = args.kvmem_round_padding
    if args.kvmem_inline_refresh is not None:
        payload["kvmem_inline_refresh"] = args.kvmem_inline_refresh
    if args.seed is not None:
        payload["seed"] = args.seed
    request = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    content: list[str] = []
    reasoning: list[str] = []
    first_content: float | None = None
    first_reasoning: float | None = None
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    chunks = 0
    try:
        response = opener().open(request, timeout=args.timeout_sec)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:4000]}") from exc
    with response:
        for raw_line in response:
            elapsed = time.perf_counter() - started
            if elapsed > args.max_sample_sec:
                raise TimeoutError(f"sample exceeded {args.max_sample_sec}s")
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(
                    f"qw3 stream error: {event['error']}"
                )
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if not choices:
                continue
            chunks += 1
            choice = choices[0]
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                if first_content is None:
                    first_content = time.perf_counter() - started
                content.append(piece)
            reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning_piece:
                if first_reasoning is None:
                    first_reasoning = time.perf_counter() - started
                reasoning.append(reasoning_piece)
    total = time.perf_counter() - started
    if finish_reason == "error":
        raise RuntimeError(
            "qw3 stream ended with finish_reason=error"
        )
    completion_tokens = usage.get("completion_tokens")
    # TTFT is the first non-empty streamed model token, regardless of whether
    # the API exposes it as reasoning_content or final content. Prefer neither
    # channel: thinking-enabled samples may emit thousands of reasoning tokens
    # before the first visible answer token. The old content-first expression
    # mislabeled that whole decode interval as TTFT.
    first_token_candidates = [
        value
        for value in (first_content, first_reasoning)
        if value is not None
    ]
    ttft = min(first_token_candidates) if first_token_candidates else None
    decode_elapsed = total - (ttft or 0.0)
    return {
        "hypothesis": "".join(content).strip(),
        "raw_response": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "finish_reason": finish_reason,
        "timing": {
            "ttft_sec": ttft,
            "content_ttft_sec": first_content,
            "first_content_sec": first_content,
            "first_reasoning_sec": first_reasoning,
            "total_sec": total,
            "decode_elapsed_after_ttft_sec": decode_elapsed,
            "output_tokens_per_sec_after_ttft": (
                completion_tokens / decode_elapsed
                if isinstance(completion_tokens, int) and decode_elapsed > 0
                else None
            ),
            "stream_chunk_count": chunks,
        },
    }


def base_row(
    index: int,
    total: int,
    sample: dict[str, Any],
    method: str,
    benchmark_name: str,
) -> dict[str, Any]:
    raw = sample.get("raw") if isinstance(sample.get("raw"), dict) else {}
    return {
        "benchmark": benchmark_name,
        "method": method,
        "index": index,
        "total": total,
        "stable_sample_id": sample["stable_sample_id"],
        "official_id": raw.get("id"),
        "sample_id": raw.get("sample_id"),
        "question_type": raw.get("question_type") or sample.get("task_type"),
        "setting": sample.get("setting"),
        "target_length": sample.get("target_length"),
        "actual_length": sample.get("actual_length"),
        "length_bucket": sample.get("length_bucket"),
        "task_type": sample.get("task_type"),
        "source_path": sample.get("source_path"),
        "source_line_index": sample.get("source_line_index"),
        "selection_key": sample.get("selection_key"),
    }


def grouped_metrics(
    evals: list[dict[str, Any]], samples: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    totals = collections.Counter(str(sample.get(key)) for sample in samples)
    counts: collections.Counter[str] = collections.Counter()
    exact: collections.Counter[str] = collections.Counter()
    scores: collections.defaultdict[str, float] = collections.defaultdict(float)
    for row in evals:
        group = str(row.get(key))
        counts[group] += 1
        scores[group] += float(row.get("score") or 0.0)
        if row.get("correct") is True:
            exact[group] += 1
    return {
        group: {
            "total": totals[group],
            "evaluated": counts[group],
            "score_sum": scores[group],
            "official_score_evaluated": scores[group] / counts[group]
            if counts[group]
            else None,
            "strict_score_total": scores[group] / totals[group] if totals[group] else None,
            "exact_correct": exact[group],
            "exact_accuracy_total": exact[group] / totals[group] if totals[group] else None,
        }
        for group in sorted(totals)
    }


def write_progress(root: Path, samples: list[dict[str, Any]], current: int | None) -> None:
    output = paths(root)
    answers = latest_by_id(read_jsonl(output["answers"]))
    evals = list(latest_by_id(read_jsonl(output["eval"])).values())
    score_sum = sum(float(row.get("score") or 0.0) for row in evals)
    exact = sum(row.get("correct") is True for row in evals)
    write_json(
        output["progress"],
        {
            "updated_at": now_iso(),
            "total": len(samples),
            "current_index": current,
            "answers": len(answers),
            "evaluated": len(evals),
            "official_score_evaluated": score_sum / len(evals) if evals else None,
            "strict_score_total": score_sum / len(samples),
            "exact_correct": exact,
            "exact_accuracy_total": exact / len(samples),
        },
    )


def write_final(
    root: Path,
    samples: list[dict[str, Any]],
    method: str,
    benchmark_name: str,
) -> None:
    output = paths(root)
    answer_rows = read_jsonl(output["answers"])
    eval_rows = read_jsonl(output["eval"])
    answers = list(latest_by_id(answer_rows).values())
    evals = list(latest_by_id(eval_rows).values())
    score_sum = sum(float(row.get("score") or 0.0) for row in evals)
    exact = sum(row.get("correct") is True for row in evals)
    total = len(samples)
    summary = {
        "benchmark": benchmark_name,
        "method": method,
        "total": total,
        "answers": len(answers),
        "evaluated": len(evals),
        "score_sum": score_sum,
        "official_score_evaluated": score_sum / len(evals) if evals else None,
        "strict_score_total": score_sum / total,
        "exact_correct": exact,
        "exact_accuracy_total": exact / total,
        "groups": {
            key: grouped_metrics(evals, samples, key)
            for key in ("setting", "target_length", "actual_length", "task_type")
        },
        "updated_at": now_iso(),
    }
    write_json(output["summary"], summary)
    sample_ids = {str(sample["stable_sample_id"]) for sample in samples}
    answer_ids = {str(row["stable_sample_id"]) for row in answers}
    eval_ids = {str(row["stable_sample_id"]) for row in evals}
    validation = {
        "expected_total": total,
        "answers_rows": len(answer_rows),
        "answers_unique": len(answer_ids),
        "eval_rows": len(eval_rows),
        "eval_unique": len(eval_ids),
        "missing_answer_ids": sorted(sample_ids - answer_ids),
        "missing_eval_ids": sorted(sample_ids - eval_ids),
        "extra_answer_ids": sorted(answer_ids - sample_ids),
        "extra_eval_ids": sorted(eval_ids - sample_ids),
        "empty_hypotheses": sum(not str(row.get("hypothesis") or "").strip() for row in answers),
        "finish_reason_length": sum(row.get("finish_reason") == "length" for row in answers),
        "passed": answer_ids == sample_ids and eval_ids == sample_ids,
        "updated_at": now_iso(),
    }
    write_json(output["validation"], validation)
    write_progress(root, samples, total)


def main() -> None:
    args = parse_args()
    if args.kvmem_round_padding < 0:
        raise RuntimeError("--kvmem-round-padding must be >= 0")
    if args.kvmem_round_padding > 0 and not args.kvmem_round_only:
        raise RuntimeError(
            "--kvmem-round-padding requires --kvmem-round-only"
        )
    if args.kvmem_round_only and args.kvmem_message_expansion:
        raise RuntimeError(
            "--kvmem-round-only and --kvmem-message-expansion are "
            "mutually exclusive"
        )
    canonical = load_canonical_module(args.benchmark_repo)
    samples = read_jsonl(args.dataset)
    canonical_manifest = read_jsonl(args.manifest)
    if len(samples) != len(canonical_manifest):
        raise RuntimeError(
            f"samples/manifest length mismatch: samples={len(samples)} "
            f"manifest={len(canonical_manifest)}"
        )
    if not args.allow_custom_subset and len(samples) != 250:
        raise RuntimeError(
            f"full run requires canonical 250 rows; got {len(samples)}. "
            "Pass --allow-custom-subset only for a separately materialized subset."
        )
    sample_ids = [str(row.get("stable_sample_id") or "") for row in samples]
    manifest_ids = [
        str(row.get("stable_sample_id") or "") for row in canonical_manifest
    ]
    if any(not sid for sid in sample_ids) or sample_ids != manifest_ids:
        raise RuntimeError(
            "samples and manifest must contain the same non-empty stable IDs in order"
        )
    if len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("samples contain duplicate stable_sample_id values")
    if args.question_id:
        wanted = set(args.question_id)
        samples = [
            sample
            for sample in samples
            if str(sample.get("stable_sample_id")) in wanted
            or str((sample.get("raw") or {}).get("id")) in wanted
        ]
    elif args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("no samples selected")
    if not health_ok(args.api_base):
        raise RuntimeError(f"qw3 server is not healthy at {args.api_base}")

    output = paths(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark": args.benchmark_name,
        "method": args.method,
        "dataset": str(args.dataset),
        "canonical_manifest": str(args.manifest),
        "canonical_worker": str(
            args.benchmark_repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
        ),
        "api_base": args.api_base,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "context_window": args.context_window,
        "enable_thinking": args.enable_thinking,
        "kvmem_retrieval_trace_metadata": args.kvmem_retrieval_trace_metadata,
        "kvmem_round_only": args.kvmem_round_only,
        "kvmem_message_expansion": args.kvmem_message_expansion,
        "kvmem_round_padding": args.kvmem_round_padding,
        "kvmem_inline_refresh": args.kvmem_inline_refresh,
        "seed": args.seed,
        "allow_custom_subset": args.allow_custom_subset,
        "selected_samples": len(samples),
        "started_at": now_iso(),
    }
    if output["config"].exists():
        previous = json.loads(output["config"].read_text(encoding="utf-8"))
        comparable = (
            "benchmark",
            "method",
            "dataset",
            "canonical_manifest",
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "context_window",
            "enable_thinking",
            "kvmem_retrieval_trace_metadata",
            "kvmem_round_only",
            "kvmem_message_expansion",
            "kvmem_round_padding",
            "kvmem_inline_refresh",
            "seed",
            "allow_custom_subset",
            "selected_samples",
        )
        if any(
            (
                previous.get(key, False)
                if key
                in {
                    "kvmem_retrieval_trace_metadata",
                    "kvmem_round_only",
                    "kvmem_message_expansion",
                    "kvmem_round_padding",
                }
                else previous.get(key)
            )
            != config.get(key)
            for key in comparable
        ):
            raise RuntimeError("output root already contains a different run configuration")
    else:
        write_json(output["config"], config)

    answers = latest_by_id(read_jsonl(output["answers"]))
    evals = latest_by_id(read_jsonl(output["eval"]))
    manifest_ids = {
        str(row["stable_sample_id"])
        for row in read_jsonl(output["manifest"])
        if row.get("stable_sample_id")
    }
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        sid = str(sample["stable_sample_id"])
        if sid in evals:
            print(f"[skip] {index}/{total} {sid}", flush=True)
            continue
        answer = answers.get(sid)
        if answer is None:
            prompt = canonical.full_context_prompt(sample)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            expected_hash = str(sample.get("prompt_sha256") or "")
            if expected_hash and prompt_hash != expected_hash:
                raise RuntimeError(f"prompt hash mismatch for {sid}")
            query_span = query_byte_span(sample, prompt)
            context_span = (
                history_byte_span(canonical, sample, prompt)
                if args.kvmem_retrieval_trace_metadata
                else None
            )
            if args.kvmem_round_only:
                retrieval_group_spans = round_byte_spans(
                    canonical, sample, prompt
                )
                semantic_group_kind = "round"
            elif args.kvmem_message_expansion:
                retrieval_group_spans = message_byte_spans(
                    canonical, sample, prompt
                )
                semantic_group_kind = "message"
            else:
                retrieval_group_spans = None
                semantic_group_kind = None
            prompt_tokens = tokenize_count(args.api_base, prompt, args.timeout_sec)
            request_max_tokens = min(
                args.max_tokens,
                max(1, args.context_window - prompt_tokens - args.context_safety_margin),
            )
            row = {
                **base_row(
                    index,
                    total,
                    sample,
                    args.method,
                    args.benchmark_name,
                ),
                "prompt_sha256": prompt_hash,
                "prompt_tokens": prompt_tokens,
                "prompt_tokenizer": "qw3:/tokenize",
                "prompt_mode": "official_task_prompt_flattened_history",
                "kvmem_query_span_content_bytes": list(query_span),
                "generation_max_tokens_configured": args.max_tokens,
                "generation_max_tokens_request": request_max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "enable_thinking": args.enable_thinking,
                "kvmem_inline_refresh": args.kvmem_inline_refresh,
            }
            if retrieval_group_spans is not None:
                group_lengths = [
                    end - begin for begin, end in retrieval_group_spans
                ]
                row["kvmem_semantic_group_kind"] = semantic_group_kind
                row["kvmem_semantic_groups"] = len(retrieval_group_spans)
                row["kvmem_semantic_group_bytes_min"] = min(group_lengths)
                row["kvmem_semantic_group_bytes_max"] = max(group_lengths)
                if semantic_group_kind == "round":
                    # Preserve historical result-schema fields.
                    row["kvmem_round_groups"] = len(
                        retrieval_group_spans
                    )
                    row["kvmem_round_group_bytes_min"] = min(group_lengths)
                    row["kvmem_round_group_bytes_max"] = max(group_lengths)
                    row["kvmem_round_padding_alignment"] = (
                        args.kvmem_round_padding
                    )
            if context_span is not None:
                row["kvmem_context_span_content_bytes"] = list(context_span)
                row["kvmem_trace_tag"] = sid
            if sid not in manifest_ids:
                append_jsonl(output["manifest"], row)
                manifest_ids.add(sid)
            append_jsonl(
                output["status"],
                {"time": now_iso(), "stable_sample_id": sid, "index": index, "status": "started"},
            )
            print(
                f"[answer] {index}/{total} {sid} prompt={prompt_tokens} "
                f"max={request_max_tokens}",
                flush=True,
            )
            last_error: Exception | None = None
            for attempt in range(1, args.attempts + 1):
                try:
                    result = chat_completion(
                        args,
                        prompt,
                        query_span,
                        context_span,
                        retrieval_group_spans,
                        sid if context_span is not None else None,
                        request_max_tokens,
                    )
                    actual_prompt_tokens = (
                        result.get("usage", {}).get("prompt_tokens")
                        if isinstance(result.get("usage"), dict)
                        else None
                    )
                    if isinstance(actual_prompt_tokens, int):
                        row["actual_prompt_tokens"] = actual_prompt_tokens
                        # This delta includes both any experimental round
                        # padding and the server's chat-template framing.  Do
                        # not label it as padding alone: for the current Qwen
                        # template an unpadded request still has a 10-token
                        # framing delta.
                        row["actual_prompt_tokens_minus_canonical"] = (
                            actual_prompt_tokens - prompt_tokens
                        )
                    answer = {**row, **result, "answered_at": now_iso()}
                    append_jsonl(output["answers"], answer)
                    answers[sid] = answer
                    append_jsonl(
                        output["status"],
                        {
                            "time": now_iso(),
                            "stable_sample_id": sid,
                            "index": index,
                            "status": "answered",
                            "finish_reason": answer.get("finish_reason"),
                        },
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    append_jsonl(
                        output["status"],
                        {
                            "time": now_iso(),
                            "stable_sample_id": sid,
                            "index": index,
                            "status": "answer_failed",
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:4000],
                        },
                    )
                    print(f"[error] attempt={attempt} {type(exc).__name__}: {exc}", flush=True)
                    if not health_ok(args.api_base):
                        raise RuntimeError("qw3 server stopped during evaluation") from exc
                    time.sleep(min(30, 2 * attempt))
            if answer is None:
                raise RuntimeError(f"all attempts failed for {sid}: {last_error}")

        evaluated = canonical.evaluate_response(
            sample, answer.get("raw_response") or answer.get("hypothesis") or ""
        )
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
            "evaluated_at": now_iso(),
        }
        append_jsonl(output["eval"], eval_row)
        evals[sid] = eval_row
        append_jsonl(
            output["status"],
            {
                "time": now_iso(),
                "stable_sample_id": sid,
                "index": index,
                "status": "evaluated",
                "score": eval_row.get("score"),
                "correct": eval_row.get("correct"),
            },
        )
        print(
            f"[eval] {index}/{total} score={eval_row.get('score')} "
            f"correct={eval_row.get('correct')}",
            flush=True,
        )
        write_progress(args.output_root, samples, index)

    write_final(args.output_root, samples, args.method, args.benchmark_name)
    print(f"[complete] results={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
