#!/usr/bin/env python3
"""Run MemoryAgentBench compact/RAG/window baselines with frozen branching.

The official benchmark stores one immutable context and several independent
questions in each parquet row.  This runner preserves that unit of work:

* official context rendering is delegated to
  ``prepare_memoryagentbench_archive_row.py``;
* question-independent compact state and full-history RAG embeddings are built
  once per row;
* every scored question is a fresh Chat Completions request and never contains
  a previous question or answer;
* a deliberately primed, page-aligned *lossless* QW3 prefix-cache entry reuses
  the transformed context.  This is ordinary dense attention, not KVMem sparse
  retrieval.  Only the question-specific RAG suffix is freshly prefetched.

The output ``methods/<method>/rows/*/results.jsonl`` intentionally matches the
layout consumed by ``score_memoryagentbench_official_local.py`` and
``judge_memoryagentbench_special.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Callable

import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_memoryagentbench_archive_row.py"
COMPACT_REFERENCE = Path(
    "/home/chaidi/AgentLongBench-Long/script/compactOnly/"
    "run_two_round_compact_only.py"
)
RAG_REFERENCE = Path(
    "/home/chaidi/AgentLongBench-Long/script/compactRag/"
    "run_three_round_compact_rag.py"
)
WINDOW_REFERENCE = Path(
    "/home/chaidi/AgentLongBench-Long/script/slidingWindow/"
    "run_sliding_window.py"
)
WINDOW_256K_CONFIG_REFERENCE = Path(
    "/home/chaidi/kvmem-efficiency-bench/configs/runs/"
    "qw3_sliding_window.json"
)
PINNED_OFFICIAL_COMMIT = "455306dcabc3842526eb83cd4e225e5d486c5c5d"

SPLITS = (
    "Accurate_Retrieval",
    "Conflict_Resolution",
    "Long_Range_Understanding",
    "Test_Time_Learning",
)
METHODS = ("compact-only", "compact-rag", "sliding-window")
MAX_TOKENS = {
    "ruler": 50,
    "eventqa": 40,
    "longmemeval": 50,
    "fact": 10,
    "infbench": 1200,
    "detective": 2000,
    "icl": 20,
    "recsys": 300,
}


class BaselineError(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise BaselineError(f"reference implementation is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BaselineError(f"cannot import reference implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def source_family(source: str) -> str:
    prefixes = (
        ("ruler_", "ruler"),
        ("eventqa_", "eventqa"),
        ("longmemeval_", "longmemeval"),
        ("factconsolidation_", "fact"),
        ("infbench_", "infbench"),
        ("detective_", "detective"),
        ("icl_", "icl"),
        ("recsys_", "recsys"),
    )
    for prefix, family in prefixes:
        if source.startswith(prefix):
            return family
    raise BaselineError(f"unsupported MemoryAgentBench source: {source}")


@dataclass(frozen=True)
class WorkRow:
    ordinal: int
    split: str
    parquet: Path
    dataset_row: int
    source: str

    @property
    def name(self) -> str:
        return (
            f"{self.ordinal:03d}_{safe_name(self.split)}_"
            f"r{self.dataset_row:03d}_{safe_name(self.source)}"
        )


def enumerate_work(args: argparse.Namespace) -> list[WorkRow]:
    # Keep ordinals canonical across filtered and phased runs.  The KVMem
    # reference run, the >256K phase, and the later <=256K phase must all name
    # one parquet row identically; otherwise resumable output can silently
    # duplicate a row under a different selection-local ordinal.
    selected_splits = set(args.split or SPLITS)
    work: list[WorkRow] = []
    canonical_ordinal = 0
    for split in SPLITS:
        parquet = args.data_dir / f"{split}-00000-of-00001.parquet"
        table = pq.read_table(parquet, columns=["metadata"])
        for row_index, wrapped in enumerate(table.to_pylist()):
            canonical_ordinal += 1
            source = str((wrapped.get("metadata") or {}).get("source") or "")
            if split not in selected_splits:
                continue
            if args.source and source not in args.source:
                continue
            if args.row and row_index not in args.row:
                continue
            work.append(
                WorkRow(canonical_ordinal, split, parquet, row_index, source)
            )

    if (
        args.min_context_tokens_exclusive is not None
        or args.max_context_tokens_inclusive is not None
    ):
        filtered: list[WorkRow] = []
        for row in work:
            summary_path = args.reference_results / "rows" / row.name / "row_summary.json"
            if not summary_path.is_file():
                raise BaselineError(
                    "context-length filtering requires the matching KVMem row "
                    f"summary: {summary_path}"
                )
            summary = read_json(summary_path)
            tokens = int(summary["archive_tokens_unpadded"])
            if (
                args.min_context_tokens_exclusive is not None
                and tokens <= args.min_context_tokens_exclusive
            ):
                continue
            if (
                args.max_context_tokens_inclusive is not None
                and tokens > args.max_context_tokens_inclusive
            ):
                continue
            filtered.append(row)
        work = filtered
    if args.max_contexts is not None:
        work = work[: args.max_contexts]
    return work


class TokenCounter:
    def __init__(self, tokenizer_dir: Path) -> None:
        import transformers
        from transformers import AutoTokenizer

        tokenizer_json = tokenizer_dir / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise BaselineError(f"tokenizer.json is missing: {tokenizer_json}")
        self.path = tokenizer_dir.resolve()
        # Loading tokenizer.json directly is not equivalent to the tokenizer
        # used by recent vLLM releases.  Transformers 5 applies Qwen's updated
        # Unicode pre-tokenization rules at AutoTokenizer construction time;
        # the difference is observable for Hindi/Bengali/Urdu text and can
        # move a nominal 64K prompt across the hard cap.  Run generation with
        # the same Python environment as vLLM and use its AutoTokenizer here.
        self.transformers_version = transformers.__version__
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), local_files_only=True, use_fast=True
        )
        self.calls = 0
        self.seconds = 0.0

    def ids(self, text: str) -> list[int]:
        started = time.perf_counter()
        result = self.tokenizer.encode(text, add_special_tokens=False)
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return list(result)

    def count(self, text: str) -> int:
        return len(self.ids(text))

    def snapshot(self) -> dict[str, Any]:
        return {
            "tokenizer": str(self.path),
            "implementation": type(self.tokenizer).__name__,
            "transformers_version": self.transformers_version,
            "calls": self.calls,
            "seconds": self.seconds,
        }


def qwen_open_user(system: str, content: str) -> str:
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{content}"
    )


QWEN_NO_THINK_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
RAG_VARIABLE_PREFIX = (
    "[Question-specific blocks retrieved from the full original memorized context]"
)


def qwen_chat_no_thinking(system: str, content: str) -> str:
    return qwen_open_user(system, content) + QWEN_NO_THINK_SUFFIX


def qwen_user_only_no_thinking(content: str) -> str:
    return (
        f"<|im_start|>user\n{content}"
        f"{QWEN_NO_THINK_SUFFIX}"
    )


def extract_memorized_context(prefix: str, system_message: str) -> str:
    expected = qwen_open_user(system_message, "")
    if not prefix.startswith(expected):
        raise BaselineError("prepared prefix does not match the official Qwen role layout")
    return prefix[len(expected):]


def prepare_row(args: argparse.Namespace, row: WorkRow) -> dict[str, Any]:
    row_dir = args.workspace / "prepared" / "rows" / row.name
    manifest_path = row_dir / "prepare_manifest.json"
    if not manifest_path.exists():
        row_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(PREPARE_SCRIPT),
            "--parquet",
            str(row.parquet),
            "--row",
            str(row.dataset_row),
            "--out-dir",
            str(row_dir),
            "--timestamp",
            args.timestamp,
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise BaselineError(
                f"MemoryAgentBench preparation failed for {row.name}:\n"
                f"{result.stdout}\n{result.stderr}"
            )
    manifest = read_json(manifest_path)
    if manifest.get("source") != row.source or int(manifest.get("row", -1)) != row.dataset_row:
        raise BaselineError(f"prepared row identity mismatch: {row.name}")
    return manifest


def load_prepared(
    args: argparse.Namespace,
    row: WorkRow,
    *,
    apply_question_limit: bool = True,
) -> dict[str, Any]:
    manifest = prepare_row(args, row)
    prefix = Path(manifest["archive_prefix"]).read_text(encoding="utf-8")
    prep_reference = load_module(PREPARE_SCRIPT, "_mab_baseline_prepare_reference")
    context = extract_memorized_context(prefix, prep_reference.SYSTEM_MESSAGE)
    qa = read_json(Path(manifest["qa_file"]))
    questions = read_json(Path(manifest["questions_file"]))
    if len(qa) != len(questions):
        raise BaselineError(f"question count mismatch in {row.name}")
    total_questions = len(qa)
    all_formatted_questions = list(questions)
    if apply_question_limit and args.question_limit is not None:
        qa = qa[: args.question_limit]
        questions = questions[: args.question_limit]
    return {
        "manifest": manifest,
        "system": prep_reference.SYSTEM_MESSAGE,
        "context": context,
        "qa": qa,
        "formatted_questions": questions,
        # A smoke limit controls generation only. Shared representations must
        # still satisfy every official branch, otherwise smoke answers and the
        # resumed full row can use different Window32K suffixes.
        "all_formatted_questions": all_formatted_questions,
        "total_questions": total_questions,
    }


def align_common_prefix(
    *,
    system: str,
    content: str,
    counter: TokenCounter,
    page_tokens: int,
    max_newlines: int = 4096,
) -> tuple[str, dict[str, int]]:
    """Pad only with newlines so prefix-cache v1 commits the shared prefix.

    QW3 commits the longest page-aligned prefix strictly shorter than a prompt.
    The no-thinking role suffix is nine Qwen tokens, shorter than the 16-token
    KV page.  If the open user prefix is page aligned, a context-only warmup
    therefore commits exactly that open prefix, before the differing question.
    """
    if page_tokens <= 0:
        raise BaselineError("prefix page size must be positive")
    open_text = qwen_open_user(system, content)
    open_tokens = counter.count(open_text)
    tail = open_text[-512:]
    tail_tokens = counter.count(tail)
    chosen: tuple[str, int, int] | None = None
    for newlines in range(max_newlines + 1):
        delta = counter.count(tail + ("\n" * newlines)) - tail_tokens
        if (open_tokens + delta) % page_tokens:
            continue
        candidate = content + ("\n" * newlines)
        exact = counter.count(qwen_open_user(system, candidate))
        if exact % page_tokens == 0:
            chosen = (candidate, newlines, exact)
            break
    if chosen is None:
        raise BaselineError(
            f"cannot align shared prefix within {max_newlines} newline characters"
        )
    candidate, newlines, exact_open = chosen
    full_tokens = counter.count(qwen_chat_no_thinking(system, candidate))
    suffix_tokens = full_tokens - exact_open
    if suffix_tokens <= 0 or suffix_tokens >= page_tokens:
        raise BaselineError(
            "Qwen no-thinking role suffix no longer fits in one KV page: "
            f"suffix={suffix_tokens} page={page_tokens}"
        )
    expected_commit = (full_tokens // page_tokens) * page_tokens
    if expected_commit >= full_tokens:
        expected_commit -= page_tokens
    if expected_commit != exact_open:
        raise BaselineError(
            f"prefix-cache warmup would commit {expected_commit}, expected {exact_open}"
        )
    return candidate, {
        "padding_newline_chars": newlines,
        "open_prefix_tokens": exact_open,
        "warmup_prompt_tokens_local": full_tokens,
        "no_think_suffix_tokens": suffix_tokens,
        "expected_commit_tokens": expected_commit,
        "page_tokens": page_tokens,
    }


def compact_common_content(summary: str, tail: str) -> str:
    if not summary:
        return tail
    return (
        "The earlier memorized context was compressed without seeing any "
        "question or retrieval result.\n\n"
        "[Question-independent checkpoint summary]\n"
        f"{summary.strip()}\n\n"
        "[Uncompressed recent memorized context]\n"
        f"{tail}"
    )


def compact_summary_only_content(summary: str) -> str:
    if not summary.strip():
        raise BaselineError("strict no-tail compact state has an empty summary")
    return (
        "The full memorized context was compressed without seeing any "
        "question or retrieval result.\n\n"
        "[Question-independent full-history summary]\n"
        f"{summary.strip()}"
    )


def find_compact_segment_end(
    *,
    history: str,
    start: int,
    previous_summary: str,
    compact_module: ModuleType,
    counter: TokenCounter,
    input_limit: int,
) -> tuple[int, int]:
    def tokens(end: int) -> int:
        prompt = compact_module.compaction_prompt(
            history[start:end], previous_summary
        )
        return counter.count(qwen_user_only_no_thinking(prompt))

    if tokens(len(history)) <= input_limit:
        return len(history), tokens(len(history))
    lo, hi = start + 1, len(history)
    best = start
    best_tokens = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        value = tokens(mid)
        if value <= input_limit:
            best = mid
            best_tokens = value
            lo = mid + 1
        else:
            hi = mid - 1
    if best <= start:
        raise BaselineError("no context text fits into one compact request")
    return best, best_tokens


def client_from_args(args: argparse.Namespace):
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from client import Qw3Client

    client = Qw3Client(
        base_url=args.api_base,
        model=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=2000,
        enable_thinking=False,
        connect_timeout=30,
        read_timeout=args.read_timeout,
    )
    # Local evaluation must never inherit an HTTP(S) proxy from the shell.
    client._session.trust_env = False
    return client


def checked_chat(
    client: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    attempts: int,
    require_content: bool = True,
    accept_terminal_length: bool = False,
) -> tuple[Any, int]:
    last = None
    for attempt in range(1, attempts + 1):
        result = client.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=False,
            extra_body={"top_k": top_k},
        )
        last = result
        if result.error is None and (
            result.answer
            or not require_content
            or (accept_terminal_length and result.finish_reason == "length")
        ):
            return result, attempt
        if attempt < attempts:
            time.sleep(min(2 ** (attempt - 1), 8))
    raise BaselineError(
        "QW3 request failed after retries: "
        f"error={getattr(last, 'error', None)!r} "
        f"finish={getattr(last, 'finish_reason', None)!r}"
    )


def compact_fingerprint(args: argparse.Namespace, context: str) -> str:
    value = {
        "schema": 1,
        "method": "question-independent-codex-compact",
        "context_sha256": sha256_text(context),
        "model": args.model_name,
        "compact_reference_sha256": sha256_file(COMPACT_REFERENCE),
        "input_limit": args.compaction_input_tokens,
        "summary_max_tokens": args.summary_max_tokens,
        "common_open_limit": args.compact_common_open_tokens,
        "carry_limit": args.carry_max_tokens,
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": args.top_k,
        "thinking": False,
        "full_history": args.compact_full_history,
        "no_raw_tail": args.compact_no_tail,
    }
    return sha256_text(canonical_json(value))


def prepare_compact_state(
    args: argparse.Namespace,
    row: WorkRow,
    prepared: dict[str, Any],
    counter: TokenCounter,
    client: Any,
) -> dict[str, Any]:
    context = prepared["context"]
    system = prepared["system"]
    path = args.workspace / "shared" / "compact" / f"{row.name}.json"
    fingerprint = compact_fingerprint(args, context)
    if path.exists():
        state = read_json(path)
        if state.get("config_sha256") != fingerprint:
            raise BaselineError(f"compact checkpoint fingerprint mismatch: {path}")
        if state.get("status") == "completed":
            return state
    else:
        source_path = None
        if args.source_shared_workspace is not None:
            source_path = (
                args.source_shared_workspace / "shared" / "compact" / f"{row.name}.json"
            )
        if source_path is not None and source_path.is_file():
            source = read_json(source_path)
            if source.get("context_sha256") != sha256_text(context):
                raise BaselineError(
                    f"source compact context mismatch: {source_path}"
                )
            if source.get("question_seen") is not False or source.get("retrieval_seen") is not False:
                raise BaselineError(
                    f"source compact checkpoint leaked query information: {source_path}"
                )
            state = {
                **source,
                "status": "running",
                "config_sha256": fingerprint,
                "imported_from": str(source_path.resolve()),
                "imported_config_sha256": source.get("config_sha256"),
                "imported_round_count": len(source.get("rounds") or []),
                "updated_at": now_iso(),
            }
        else:
            state = {
                "schema_version": 1,
                "status": "running",
                "config_sha256": fingerprint,
                "context_sha256": sha256_text(context),
                "context_chars": len(context),
                "cursor": 0,
                "summary": "",
                "rounds": [],
                "imported_round_count": 0,
                "question_seen": False,
                "retrieval_seen": False,
                "created_at": now_iso(),
            }

    compact_module = load_module(COMPACT_REFERENCE, "_mab_compact_reference")
    cursor = int(state.get("cursor") or 0)
    summary = str(state.get("summary") or "")
    rounds = list(state.get("rounds") or [])
    while True:
        tail = context[cursor:]
        common = (
            compact_summary_only_content(summary)
            if args.compact_full_history and cursor >= len(context)
            else compact_common_content(summary, tail)
        )
        open_tokens = counter.count(qwen_open_user(system, common))
        completed = (
            cursor >= len(context)
            if args.compact_full_history
            else open_tokens <= args.compact_common_open_tokens
        )
        if completed:
            state.update({
                "status": "completed",
                "cursor": cursor,
                "summary": summary,
                "tail_sha256": sha256_text(tail),
                "common_content_sha256": sha256_text(common),
                "common_open_tokens": open_tokens,
                "round_count": len(rounds),
                "completed_at": now_iso(),
            })
            write_json_atomic(path, state)
            return state
        if len(rounds) >= args.max_compact_rounds:
            raise BaselineError(
                f"compact exceeded {args.max_compact_rounds} rounds for {row.name}"
            )
        carry_prompt = compact_module.compaction_prompt("", summary)
        carry_tokens = counter.count(carry_prompt)
        if carry_tokens > args.carry_max_tokens:
            raise BaselineError(
                f"compact carry exceeds limit: {carry_tokens}>{args.carry_max_tokens}"
            )
        selection_started = time.perf_counter()
        end, prompt_tokens_local = find_compact_segment_end(
            history=context,
            start=cursor,
            previous_summary=summary,
            compact_module=compact_module,
            counter=counter,
            input_limit=args.compaction_input_tokens,
        )
        selection_sec = time.perf_counter() - selection_started
        prompt = compact_module.compaction_prompt(context[cursor:end], summary)
        result, accepted_attempt = checked_chat(
            client,
            [{"role": "user", "content": prompt}],
            max_tokens=args.summary_max_tokens,
            temperature=0.0,
            top_p=0.9,
            top_k=args.top_k,
            attempts=args.attempts,
            accept_terminal_length=True,
        )
        if (
            result.prompt_tokens is not None
            and result.prompt_tokens > args.compaction_input_tokens
        ):
            raise BaselineError(
                "server compact prompt exceeded the configured input limit: "
                f"{result.prompt_tokens}>{args.compaction_input_tokens}"
            )
        new_summary = result.answer.strip()
        record = {
            "round_index": len(rounds) + 1,
            "cursor_before": cursor,
            "cursor_after": end,
            "segment_chars": end - cursor,
            "segment_sha256": sha256_text(context[cursor:end]),
            "previous_summary_sha256": sha256_text(summary),
            "prompt_sha256": sha256_text(prompt),
            "prompt_tokens_local": prompt_tokens_local,
            "prompt_tokens_server": result.prompt_tokens,
            "summary": new_summary,
            "summary_sha256": sha256_text(new_summary),
            "summary_tokens_local": counter.count(new_summary),
            "finish_reason": result.finish_reason,
            "summary_truncated_at_limit": result.finish_reason == "length",
            "accepted_attempt": accepted_attempt,
            "selection_sec": selection_sec,
            "ttft_s": result.ttft_s,
            "server_ttft_s": result.server_ttft_s,
            "latency_s": result.latency_s,
            "completion_tokens": result.completion_tokens,
            "completed_at": now_iso(),
            "question_seen": False,
            "retrieval_seen": False,
        }
        rounds.append(record)
        cursor = end
        summary = new_summary
        state.update({
            "cursor": cursor,
            "summary": summary,
            "rounds": rounds,
            "updated_at": now_iso(),
        })
        write_json_atomic(path, state)


def format_rag_blocks(
    blocks: list[dict[str, Any]], *, include_metadata: bool = True
) -> str:
    if not include_metadata:
        return "\n\n---\n\n".join(str(block["text"]) for block in blocks)
    return "\n\n".join(
        (
            f"[Retrieved block rank={block['rank']} "
            f"block_id={block['block_id']}]\n{block['text']}"
        )
        for block in blocks
    )


def rag_question_suffix(
    blocks: list[dict[str, Any]],
    formatted_question: str,
    *,
    include_metadata: bool = True,
) -> str:
    return (
        f"{RAG_VARIABLE_PREFIX}\n"
        f"{format_rag_blocks(blocks, include_metadata=include_metadata)}\n\n"
        f"{formatted_question}"
    )


def select_rag_blocks_for_prompt_budget(
    *,
    retrieval: dict[str, Any],
    system: str,
    aligned_common: str,
    formatted_question: str,
    counter: TokenCounter,
    prompt_limit: int,
    include_metadata: bool,
) -> tuple[list[dict[str, Any]], str, int]:
    """Select the largest top-ranked set that fits the complete prompt cap."""
    candidates = list(retrieval.get("retrieved_blocks") or [])

    def choose(count: int) -> list[dict[str, Any]]:
        return sorted(
            (
                block for block in candidates
                if int(block.get("rank") or 0) <= count
            ),
            key=lambda block: int(block["chunk_index"]),
        )

    def materialize(count: int) -> tuple[list[dict[str, Any]], str, int]:
        blocks = choose(count)
        variable = rag_question_suffix(
            blocks,
            formatted_question,
            include_metadata=include_metadata,
        )
        tokens = counter.count(
            qwen_chat_no_thinking(system, aligned_common + "\n" + variable)
        )
        return blocks, variable, tokens

    best_blocks, best_variable, best_tokens = materialize(0)
    if best_tokens > prompt_limit:
        raise BaselineError(
            "compact summary and question alone exceed strict prompt cap: "
            f"{best_tokens}>{prompt_limit}"
        )
    lo, hi = 1, len(candidates)
    while lo <= hi:
        mid = (lo + hi) // 2
        blocks, variable, tokens = materialize(mid)
        if tokens <= prompt_limit:
            best_blocks, best_variable, best_tokens = blocks, variable, tokens
            lo = mid + 1
        else:
            hi = mid - 1
    return best_blocks, best_variable, best_tokens


def rag_config(args: argparse.Namespace, module: ModuleType) -> dict[str, Any]:
    return {
        "schema": 1,
        "reference_sha256": sha256_file(RAG_REFERENCE),
        "model_path": str(args.rag_model.resolve()),
        "tokenizer_path": str(args.tokenizer_dir.resolve()),
        "model_revision": module.JINA_MODEL_REVISION,
        "code_revision": module.JINA_CODE_REVISION,
        "top_k": args.rag_top_k,
        "block_size": args.rag_block_size,
        "overlap": args.rag_overlap,
        "query": "raw_question",
        "selection_order": "score_desc_chunk_index_tiebreak",
        "prompt_order": "chunk_index_ascending",
    }


def prepare_rag_row(
    args: argparse.Namespace,
    row: WorkRow,
    prepared: dict[str, Any],
    retriever: Any,
    module: ModuleType,
) -> dict[str, Any]:
    import numpy as np

    output = args.workspace / "shared" / "rag" / f"{row.name}.json"
    context = prepared["context"]
    raw_questions = [str(item["raw_question"]) for item in prepared["qa"]]
    config = rag_config(args, module)
    fingerprint = sha256_text(canonical_json({
        "config": config,
        "context_sha256": sha256_text(context),
        "questions_sha256": sha256_text(canonical_json(raw_questions)),
    }))
    if output.exists():
        prior = read_json(output)
        if prior.get("config_sha256") != fingerprint:
            raise BaselineError(f"RAG checkpoint fingerprint mismatch: {output}")
        if prior.get("status") == "completed":
            return prior

    started = time.perf_counter()
    blocks, tokenizer_label = retriever.chunks(
        context, args.rag_block_size, args.rag_overlap
    )
    if not blocks:
        raise BaselineError(f"RAG produced no context blocks for {row.name}")
    all_text = [block["text"] for block in blocks] + raw_questions
    encoded = retriever.model.tokenizer(
        all_text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_lengths = [len(ids) for ids in encoded["input_ids"]]
    if max(input_lengths) > module.JINA_MAX_SEQ_LENGTH:
        raise BaselineError(
            f"Jina input would truncate in {row.name}: max={max(input_lengths)}"
        )
    block_started = time.perf_counter()
    block_embeddings = retriever.model.encode(
        [block["text"] for block in blocks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=args.embedding_batch_size,
    )
    block_sec = time.perf_counter() - block_started
    query_started = time.perf_counter()
    query_embeddings = retriever.model.encode(
        raw_questions,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=args.embedding_batch_size,
    )
    query_sec = time.perf_counter() - query_started
    if not np.isfinite(block_embeddings).all() or not np.isfinite(query_embeddings).all():
        raise BaselineError(f"Jina produced non-finite embeddings for {row.name}")
    effective_top_k = min(args.rag_top_k, len(blocks))
    retrievals = []
    ranking_started = time.perf_counter()
    for question_index, query_embedding in enumerate(query_embeddings):
        scores = block_embeddings @ query_embedding
        ranked = sorted(
            range(len(blocks)), key=lambda index: (-float(scores[index]), index)
        )[:effective_top_k]
        selected = []
        for rank, index in enumerate(ranked, 1):
            selected.append({
                **blocks[index],
                "rank": rank,
                "score": float(scores[index]),
                "retriever": "jina-embedding",
                "embedding_tokens": input_lengths[index],
            })
        selected.sort(key=lambda block: int(block["chunk_index"]))
        retrievals.append({
            "question_index": question_index,
            "raw_question_sha256": sha256_text(raw_questions[question_index]),
            "effective_top_k": effective_top_k,
            "retrieved_tokens_estimate": sum(
                int(block["chunk_tokens"]) for block in selected
            ),
            "retrieved_blocks": selected,
        })
    ranking_sec = time.perf_counter() - ranking_started
    state = {
        "schema_version": 1,
        "status": "completed",
        "config_sha256": fingerprint,
        "config": config,
        "split": row.split,
        "source": row.source,
        "dataset_row": row.dataset_row,
        "context_sha256": sha256_text(context),
        "context_chars": len(context),
        "total_blocks": len(blocks),
        "effective_top_k": effective_top_k,
        "short_context_uses_all_blocks": len(blocks) < args.rag_top_k,
        "chunk_tokenizer": tokenizer_label,
        "embedding_input_tokens_max": max(input_lengths),
        "questions": len(retrievals),
        "timing": {
            "block_embedding_sec": block_sec,
            "query_embedding_sec": query_sec,
            "ranking_sec": ranking_sec,
            "total_sec": time.perf_counter() - started,
        },
        "retrievals": retrievals,
        "completed_at": now_iso(),
    }
    write_json_atomic(output, state)
    return state


def load_rag_state(args: argparse.Namespace, row: WorkRow) -> dict[str, Any]:
    path = args.workspace / "shared" / "rag" / f"{row.name}.json"
    if not path.exists():
        raise BaselineError(
            f"missing RAG state for {row.name}; run prepare-rag before compact methods"
        )
    state = read_json(path)
    if state.get("status") != "completed":
        raise BaselineError(f"incomplete RAG state: {path}")
    return state


def select_fixed_sliding_suffix(
    *,
    system: str,
    context: str,
    formatted_questions: list[str],
    counter: TokenCounter,
    prompt_limit: int,
    alignment_reserve: int,
) -> tuple[str, dict[str, Any]]:
    """Select one recent suffix that fits every independent row question."""
    if not formatted_questions:
        raise BaselineError("cannot select a window for an empty question list")
    started = time.perf_counter()

    target = prompt_limit - alignment_reserve
    if target <= 0:
        raise BaselineError("sliding window alignment reserve consumes the prompt limit")

    def prompt_tokens(start: int, question_index: int) -> int:
        return counter.count(qwen_chat_no_thinking(
            system,
            context[start:] + "\n" + formatted_questions[question_index],
        ))

    def first_fitting_start(question_index: int, lower_bound: int) -> int:
        """Find this question's earliest fitting suffix at/after lower_bound.

        The previous implementation evaluated every question at every binary
        search probe.  For a 100-question MemoryAgentBench row that performs
        roughly 2,000 long tokenizations even though the final boundary is the
        maximum of 100 independent per-question boundaries.  Search only the
        currently active constraint instead.  If another question still
        overflows at that boundary, it becomes the next active constraint.
        Each update moves the shared boundary strictly forward, and the final
        result is exactly max_i(first_fitting_start(i)).
        """
        lo, hi = lower_bound, len(context)
        if prompt_tokens(hi, question_index) > target:
            raise BaselineError(
                "sliding question alone exceeds the complete-prompt limit: "
                f"question={question_index} target={target}"
            )
        while lo < hi:
            mid = (lo + hi) // 2
            if prompt_tokens(mid, question_index) <= target:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # Standalone question length is a cheap, usually exact predictor of which
    # question imposes the tightest suffix boundary.  Correctness does not
    # depend on the heuristic: all questions are checked below, and every
    # violating question advances the boundary to its own exact minimum.
    active = max(
        range(len(formatted_questions)),
        key=lambda index: counter.count(formatted_questions[index]),
    )
    start = first_fitting_start(active, 0)
    constraint_iterations = 1
    while True:
        counts = [
            prompt_tokens(start, index)
            for index in range(len(formatted_questions))
        ]
        overflowing = [
            index for index, value in enumerate(counts) if value > target
        ]
        if not overflowing:
            break
        active = max(overflowing, key=lambda index: counts[index])
        advanced = first_fitting_start(active, start)
        if advanced <= start:
            raise BaselineError(
                "sliding constraint iteration did not advance the window"
            )
        start = advanced
        constraint_iterations += 1

    suffix = context[start:]
    if max(counts) > target:
        raise BaselineError(f"sliding suffix overflow: {max(counts)}>{target}")
    return suffix, {
        "selection": "one fixed recent-character suffix for all row questions",
        "window_start_char": start,
        "window_chars": len(suffix),
        "window_sha256": sha256_text(suffix),
        "min_prompt_tokens_before_alignment": min(counts),
        "max_prompt_tokens_before_alignment": max(counts),
        "prompt_limit": prompt_limit,
        "alignment_reserve": alignment_reserve,
        "constraint_iterations": constraint_iterations,
        "selection_sec": time.perf_counter() - started,
    }


def normalize(text: str) -> str:
    import string

    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def flat_answers(answer: Any) -> list[str]:
    if isinstance(answer, str):
        return [answer]
    if isinstance(answer, list):
        result: list[str] = []
        for item in answer:
            result.extend(flat_answers(item))
        return result
    return [str(answer)]


def simple_metrics(prediction: str, answer: Any) -> dict[str, bool]:
    answers = flat_answers(answer)
    normalized = normalize(prediction)
    return {
        "exact_match": any(normalized == normalize(item) for item in answers),
        "substring_exact_match": any(
            normalize(item) in normalized for item in answers
        ),
    }


def prime_shared_prefix(
    args: argparse.Namespace,
    client: Any,
    counter: TokenCounter,
    system: str,
    common: str,
    variable_prefix: str,
) -> tuple[str, dict[str, Any]]:
    aligned, alignment = align_common_prefix(
        system=system,
        content=common,
        counter=counter,
        page_tokens=args.prefix_page_tokens,
    )
    # Every scored request appends exactly ``"\n" + variable`` to the shared
    # content below. Prime through a non-whitespace prefix shared by every
    # variable in the row.
    #
    # Priming only ``aligned`` is subtly incorrect for byte-level/BPE
    # tokenizers: the final newline token can merge differently with
    # ``<|im_end|>`` in the warmup and with the first question token in the
    # scored request.  A cache entry may then extend one or two tokens past the
    # real common token prefix.  It happened to fall back to the preceding KV
    # page for some rows, but caused a complete prefix-cache miss for others.
    # Qwen's Jinja template applies ``trim`` to message content. A warmup that
    # ends at the separator newline therefore loses the alignment padding on
    # vLLM, even though scored requests retain it as internal whitespace.
    # Ending at a real shared prefix makes the rendered warmup an actual prefix
    # of every scored prompt without changing any scored prompt.
    if not variable_prefix or variable_prefix[-1].isspace():
        raise BaselineError(
            "prefix-cache warmup requires a non-empty, non-whitespace-ending "
            "shared variable prefix"
        )
    warmup_content = aligned + "\n" + variable_prefix
    warmup_open_tokens = counter.count(qwen_open_user(system, warmup_content))
    warmup_prompt_tokens = counter.count(
        qwen_chat_no_thinking(system, warmup_content)
    )
    expected_commit = (
        warmup_prompt_tokens // args.prefix_page_tokens
    ) * args.prefix_page_tokens
    if expected_commit >= warmup_prompt_tokens:
        expected_commit -= args.prefix_page_tokens
    expected_commit -= args.prefix_cache_guard_pages * args.prefix_page_tokens
    if expected_commit <= 0:
        raise BaselineError("prefix-cache warmup has no reusable KV page")
    alignment.update({
        "warmup_shared_separator": "\\n",
        "warmup_shared_variable_prefix_chars": len(variable_prefix),
        "warmup_shared_variable_prefix_sha256": sha256_text(variable_prefix),
        "prefix_cache_guard_pages": args.prefix_cache_guard_pages,
        "warmup_open_tokens": warmup_open_tokens,
        "warmup_prompt_tokens_local": warmup_prompt_tokens,
        "expected_commit_tokens": expected_commit,
    })
    result, attempt = checked_chat(
        client,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": warmup_content},
        ],
        max_tokens=0,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        attempts=args.attempts,
        require_content=False,
    )
    alignment.update({
        "warmup_accepted_attempt": attempt,
        "warmup_prompt_tokens_server": result.prompt_tokens,
        "warmup_completion_tokens": result.completion_tokens,
        "warmup_ttft_s": result.ttft_s,
        "warmup_server_ttft_s": result.server_ttft_s,
        "warmup_engine_ttft_s": result.engine_ttft_s,
        "warmup_latency_s": result.latency_s,
        "warmup_finish_reason": result.finish_reason,
        "warmup_scored": False,
    })
    if result.finish_reason != "prefill_only" or result.completion_tokens not in (None, 0):
        raise BaselineError(
            "dense prefix warmup was not prefill-only: "
            f"finish={result.finish_reason!r} completion={result.completion_tokens!r}"
        )
    if result.prompt_tokens is not None:
        # QW3 tokenizes with the tokenizer embedded in the served GGUF.  Its
        # segmentation can differ slightly from the matching HF fast
        # tokenizer used to construct a strict prompt budget (seven tokens on
        # the current MemoryAgentBench representative).  The server-reported
        # count is authoritative for the KV-page commit.  Do not reject an
        # otherwise valid prefill-only warmup merely because the two tokenizer
        # implementations are not bit-identical; retain the delta explicitly
        # so the result remains auditable.
        local_tokens = int(alignment["warmup_prompt_tokens_local"])
        server_tokens = int(result.prompt_tokens)
        alignment["warmup_prompt_token_delta_server_minus_local"] = (
            server_tokens - local_tokens
        )
        expected_commit = (server_tokens // args.prefix_page_tokens) * (
            args.prefix_page_tokens
        )
        if expected_commit >= server_tokens:
            expected_commit -= args.prefix_page_tokens
        expected_commit -= (
            args.prefix_cache_guard_pages * args.prefix_page_tokens
        )
        if expected_commit <= 0:
            raise BaselineError(
                "server tokenizer leaves no reusable KV page during prefix warmup"
            )
        alignment["expected_commit_tokens_local_estimate"] = alignment[
            "expected_commit_tokens"
        ]
        alignment["expected_commit_tokens"] = expected_commit
    return aligned, alignment


def aggregate_method(method_root: Path) -> dict[str, Any]:
    rows = [
        read_json(path)
        for path in sorted((method_root / "rows").glob("*/row_summary.json"))
    ]
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)
    total_questions = sum(int(row["questions_completed"]) for row in rows)
    summary = {
        "method": method_root.name,
        "contexts_completed": len(rows),
        "questions_completed": total_questions,
        "by_source": {},
    }
    for source, values in sorted(by_source.items()):
        questions = sum(int(row["questions_completed"]) for row in values)
        summary["by_source"][source] = {
            "contexts": len(values),
            "questions": questions,
            "exact_match": (
                sum(int(row["exact_match_count"]) for row in values) / questions
                if questions else None
            ),
            "substring_exact_match": (
                sum(int(row["substring_exact_match_count"]) for row in values)
                / questions if questions else None
            ),
        }
    write_json_atomic(method_root / "progress_summary.json", summary)
    return summary


def method_config(args: argparse.Namespace, method: str) -> dict[str, Any]:
    import importlib.metadata
    import transformers

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = None

    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "method": method,
        "benchmark_semantics": (
            "one immutable context per parquet row; questions are independent "
            "frozen branches; previous questions/answers are never appended"
        ),
        "context_reuse": (
            "QW3 lossless dense prefix cache primed at an exact KV-page boundary"
        ),
        "model": args.model_name,
        "serving_runtime": {
            "vllm_version": vllm_version,
            "transformers_version": transformers.__version__,
            "python": sys.executable,
        },
        "prompt_tokenizer": {
            "path": str(args.tokenizer_dir.resolve()),
            "implementation": "transformers.AutoTokenizer(use_fast=True)",
            "transformers_version": transformers.__version__,
            "must_match_vllm_runtime": True,
        },
        "server_required": {
            "context_tokens": args.context_window,
            "kv_dtype": "fp8",
            "prefill_chunk_tokens": 2048,
            "mtp_chain": args.server_mtp_chain,
            "continuous_batching": True,
            "prefix_cache": True,
            "prefix_cache_commit_guard_pages": args.prefix_cache_guard_pages,
            "parallel": 1,
            "kvmem": False,
        },
        "final_sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "thinking": False,
            "max_tokens": MAX_TOKENS,
            "max_tokens_override": args.answer_max_tokens_override,
        },
        "compact": {
            "reference": str(COMPACT_REFERENCE),
            "reference_sha256": sha256_file(COMPACT_REFERENCE),
            "input_tokens": args.compaction_input_tokens,
            "summary_max_tokens": args.summary_max_tokens,
            "common_open_tokens": args.compact_common_open_tokens,
            "question_seen": False,
            "retrieval_seen": False,
            "shared_between_compact_methods": True,
            "full_history": args.compact_full_history,
            "raw_tail_tokens": 0 if args.compact_no_tail else "remaining_uncompressed_tail",
            "source_shared_workspace": (
                str(args.source_shared_workspace.resolve())
                if args.source_shared_workspace is not None else None
            ),
        },
        "rag": {
            "reference": str(RAG_REFERENCE),
            "reference_sha256": sha256_file(RAG_REFERENCE),
            "top_k": args.rag_top_k,
            "block_size": args.rag_block_size,
            "overlap": args.rag_overlap,
            "corpus": "full original memorized context",
            "index_frequency": "once per parquet row",
            "prepared_question_subset": (
                args.question_limit
                if args.prepare_rag_respect_question_limit else "all"
            ),
            "dynamic_prompt_budget": args.rag_dynamic_budget,
            "prompt_metadata": not args.rag_no_metadata,
        },
        "sliding": {
            "complete_prompt_tokens": args.sliding_prompt_tokens,
            "selection": "one fixed recent suffix fitting every row question",
            "reference": str(WINDOW_REFERENCE),
            "reference_sha256": sha256_file(WINDOW_REFERENCE),
            "reference_256k_config": str(WINDOW_256K_CONFIG_REFERENCE),
            "reference_256k_config_sha256": sha256_file(
                WINDOW_256K_CONFIG_REFERENCE
            ),
            "reused_from_256k_reference": {
                "context_window": 262144,
                "complete_prompt_tokens": args.sliding_prompt_tokens,
                "selection": "largest recent raw-history suffix",
                "prompt_budget_scope": "complete final prompt",
            },
            "comparison_control_overrides": {
                "model_and_kv": "match the current MemoryAgentBench KVMem run",
                "sampling_and_output_limits": "match each official source task",
                "mtp": (
                    f"server MTP chain={args.server_mtp_chain}; utility answers "
                    "branch independently from the same lossless dense prefix"
                ),
            },
        },
        "prompt_limits": {
            "context_window": args.context_window,
            "generation_reserve": args.generation_reserve,
            "compact_rag_final_input": (
                args.strict_final_prompt_tokens
                if args.strict_final_prompt_tokens is not None
                else args.context_window - args.generation_reserve
            ),
        },
        "references": {
            "prepare": str(PREPARE_SCRIPT),
            "prepare_sha256": sha256_file(PREPARE_SCRIPT),
            "official_commit": PINNED_OFFICIAL_COMMIT,
        },
        "selection": {
            "reference_results": str(args.reference_results),
            "min_context_tokens_exclusive": args.min_context_tokens_exclusive,
            "max_context_tokens_inclusive": args.max_context_tokens_inclusive,
        },
    }


def method_config_filename(args: argparse.Namespace) -> str:
    """Keep each resumable length phase's manifest instead of overwriting it."""
    if (
        args.min_context_tokens_exclusive is not None
        and args.max_context_tokens_inclusive is None
    ):
        return "run_config_over256k.json"
    if (
        args.max_context_tokens_inclusive is not None
        and args.min_context_tokens_exclusive is None
    ):
        return "run_config_under256k.json"
    return "run_config_full.json"


def run_method_row(
    args: argparse.Namespace,
    row: WorkRow,
    method: str,
    prepared: dict[str, Any],
    client: Any,
    counter: TokenCounter,
) -> dict[str, Any]:
    method_root = args.workspace / "methods" / method
    row_dir = method_root / "rows" / row.name
    row_dir.mkdir(parents=True, exist_ok=True)
    results_path = row_dir / "results.jsonl"
    summary_path = row_dir / "row_summary.json"
    existing_records = read_jsonl(results_path)
    if any(
        record.get("latency_protocol") != "final_query_boundary_v2"
        for record in existing_records
    ):
        # A v1 row may contain a cold-window or all-history accounting result.
        # Keep the old file on disk as a sidecar and rebuild this method under
        # the publication boundary instead of silently resuming it.
        legacy_path = results_path.with_suffix(".boundary_v1.jsonl")
        if not legacy_path.exists():
            results_path.replace(legacy_path)
        existing_records = []
        summary_path.unlink(missing_ok=True)
    requested_repairs = set(args.repair_question_index or [])
    if summary_path.exists() and not requested_repairs:
        shared_hashes = {
            str(record.get("shared_context_sha256"))
            for record in existing_records
        }
        if len(shared_hashes) <= 1:
            return read_json(summary_path)
        if method != "sliding-window":
            raise BaselineError(
                f"completed row has inconsistent shared contexts: {row.name}"
            )

    compact_state = None
    rag_state = None
    selection: dict[str, Any] = {}
    if method in ("compact-only", "compact-rag"):
        # Requiring precomputed RAG for both compact methods freezes one common
        # experiment manifest and guarantees compact+RAG cannot silently use a
        # different corpus/config.  The compactor itself never reads retrievals.
        rag_state = load_rag_state(args, row)
        compact_state = prepare_compact_state(
            args, row, prepared, counter, client
        )
        summary = str(compact_state.get("summary") or "")
        cursor = int(compact_state.get("cursor") or 0)
        if args.compact_no_tail:
            if cursor != len(prepared["context"]):
                raise BaselineError(
                    f"strict no-tail compact state does not cover full history: "
                    f"{row.name} cursor={cursor}/{len(prepared['context'])}"
                )
            common = compact_summary_only_content(summary)
        else:
            common = compact_common_content(summary, prepared["context"][cursor:])
        selection = {
            "compact_rounds": int(compact_state.get("round_count") or 0),
            "compact_imported_rounds": int(
                compact_state.get("imported_round_count") or 0
            ),
            "compact_cursor": cursor,
            "compact_summary_sha256": sha256_text(summary),
            "compact_common_open_tokens": compact_state["common_open_tokens"],
            "compact_config_sha256": compact_state["config_sha256"],
            "raw_tail_chars": 0 if args.compact_no_tail else len(prepared["context"]) - cursor,
        }
        compact_rounds = list(compact_state.get("rounds") or [])
        imported_round_count = selection["compact_imported_rounds"]
        # The pre-query state already contains every completed maintenance
        # compaction except the final full-window boundary event.  Even when a
        # latency workspace was built from scratch, do not charge all earlier
        # iterative rounds again at final-query arrival.
        maintenance_round_count = max(imported_round_count, len(compact_rounds) - 1)
        maintenance_rounds = compact_rounds[:maintenance_round_count]
        boundary_rounds = compact_rounds[maintenance_round_count:]
        selection.update({
            "latency_protocol": "final_query_boundary_v2",
            "history_maintenance_excluded": True,
            "compact_maintenance_rounds": len(maintenance_rounds),
            "compact_maintenance_total_sec": sum(
                float(item.get("selection_sec") or 0.0)
                + float(item.get("latency_s") or 0.0)
                for item in maintenance_rounds
            ),
            "compact_boundary_rounds": len(boundary_rounds),
            "compact_boundary_selection_sec": sum(
                float(item.get("selection_sec") or 0.0)
                for item in boundary_rounds
            ),
            "compact_boundary_generation_sec": sum(
                float(item.get("latency_s") or 0.0)
                for item in boundary_rounds
            ),
        })
        selection["compact_boundary_total_sec"] = (
            selection["compact_boundary_selection_sec"]
            + selection["compact_boundary_generation_sec"]
        )
    else:
        common, selection = select_fixed_sliding_suffix(
            system=prepared["system"],
            context=prepared["context"],
            formatted_questions=prepared["all_formatted_questions"],
            counter=counter,
            prompt_limit=args.sliding_prompt_tokens,
            alignment_reserve=args.prefix_page_tokens * 2,
        )

    completed = {
        int(record["question_index"]): record for record in existing_records
    }
    expected_shared_hash = sha256_text(common)
    replacements = {
        index for index, record in completed.items()
        if record.get("shared_context_sha256") != expected_shared_hash
    }
    invalid_repairs = sorted(
        index for index in requested_repairs
        if index < 0 or index >= len(prepared["qa"])
    )
    if invalid_repairs:
        raise BaselineError(
            f"repair question indices outside row range for {row.name}: "
            f"{invalid_repairs}; questions={len(prepared['qa'])}"
        )
    replacements.update(requested_repairs)
    if replacements:
        print(
            f"[{row.ordinal}] {method} repairing shared-context mismatch "
            f"{row.name} questions={sorted(replacements)}",
            flush=True,
        )
    if method == "compact-rag":
        shared_variable_prefix = RAG_VARIABLE_PREFIX
    else:
        shared_variable_prefix = os.path.commonprefix(
            prepared["all_formatted_questions"]
        ).rstrip()
        if not shared_variable_prefix:
            raise BaselineError(
                f"{method} questions have no stable shared textual prefix: "
                f"{row.name}"
            )
    reprefill_started = time.perf_counter()
    aligned_common, prefix_cache = prime_shared_prefix(
        args,
        client,
        counter,
        prepared["system"],
        common,
        shared_variable_prefix,
    )
    # This request builds the answer-producing dense KV for the newly compacted
    # summary/common context.  It is query-boundary work for Compact methods,
    # while Sliding's resident window was prepared before the query.
    if method in ("compact-only", "compact-rag"):
        reprefill = time.perf_counter() - reprefill_started
        if reprefill <= 0:
            raise BaselineError(f"missing final-context re-prefill time: {row.name}")
        selection["final_context_reprefill_total_sec"] = float(reprefill)
        selection["final_context_reprefill_included_in_pre_answer"] = True
    else:
        selection["resident_window_warmup_excluded"] = True
    rag_by_question = {}
    if rag_state is not None:
        rag_by_question = {
            int(item["question_index"]): item
            for item in rag_state["retrievals"]
        }
    final_input_limit = (
        args.strict_final_prompt_tokens
        if args.strict_final_prompt_tokens is not None
        else args.context_window - args.generation_reserve
    )
    for question_index, (gold, formatted_question) in enumerate(
        zip(prepared["qa"], prepared["formatted_questions"])
    ):
        if question_index in completed and question_index not in replacements:
            continue
        if method == "compact-rag":
            retrieval = rag_by_question.get(question_index)
            if retrieval is None:
                raise BaselineError(
                    f"missing retrieval q={question_index} for {row.name}"
                )
            materialization_started = time.perf_counter()
            if args.rag_dynamic_budget:
                selected_blocks, variable, prompt_tokens_local = (
                    select_rag_blocks_for_prompt_budget(
                        retrieval=retrieval,
                        system=prepared["system"],
                        aligned_common=aligned_common,
                        formatted_question=formatted_question,
                        counter=counter,
                        prompt_limit=final_input_limit,
                        include_metadata=not args.rag_no_metadata,
                    )
                )
            else:
                selected_blocks = list(retrieval["retrieved_blocks"])
                variable = rag_question_suffix(
                    selected_blocks,
                    formatted_question,
                    include_metadata=not args.rag_no_metadata,
                )
                prompt_tokens_local = counter.count(
                    qwen_chat_no_thinking(
                        prepared["system"], aligned_common + "\n" + variable
                    )
                )
            rag_materialization_sec = time.perf_counter() - materialization_started
        else:
            retrieval = None
            selected_blocks = []
            variable = formatted_question
            rag_materialization_sec = 0.0
        user_content = aligned_common + "\n" + variable
        if method != "compact-rag":
            prompt_tokens_local = counter.count(
                qwen_chat_no_thinking(prepared["system"], user_content)
            )
        limit = (
            args.sliding_prompt_tokens
            if method == "sliding-window" else final_input_limit
        )
        if prompt_tokens_local > limit:
            raise BaselineError(
                f"{method} prompt overflow {row.name} q={question_index}: "
                f"{prompt_tokens_local}>{limit}"
            )
        max_tokens = (
            args.answer_max_tokens_override
            if args.answer_max_tokens_override > 0
            else MAX_TOKENS[source_family(row.source)]
        )
        result, attempt = checked_chat(
            client,
            [
                {"role": "system", "content": prepared["system"]},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            attempts=args.attempts,
            # A final answer that spends its whole generation allowance in
            # reasoning is an accuracy failure, not a transport failure.  Keep
            # the terminal row so one pathological question cannot abort the
            # complete benchmark.  Compaction calls deliberately retain the
            # stricter non-empty-content requirement above.
            accept_terminal_length=True,
        )
        if result.prompt_tokens is not None and result.prompt_tokens > limit:
            raise BaselineError(
                f"server reported {result.prompt_tokens} prompt tokens, limit={limit}"
            )
        metrics = simple_metrics(result.answer, gold["answer"])
        record = {
            "schema_version": 1,
            "latency_protocol": "final_query_boundary_v2",
            "method": method,
            "split": row.split,
            "source": row.source,
            "dataset_row": row.dataset_row,
            "question_index": question_index,
            "qa_pair_id": gold.get("qa_pair_id"),
            "question_type": gold.get("question_type"),
            "question_id": gold.get("question_id"),
            "raw_question": gold["raw_question"],
            "formatted_question_sha256": sha256_text(formatted_question),
            "gold_answer": gold["answer"],
            "keypoints": gold.get("keypoints"),
            "answer": result.answer,
            "reasoning": result.reasoning,
            "finish_reason": result.finish_reason,
            "accepted_attempt": attempt,
            "prompt_tokens_local": prompt_tokens_local,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "max_generation_tokens": max_tokens,
            "ttft_s": result.ttft_s,
            "server_ttft_s": result.server_ttft_s,
            "engine_ttft_s": result.engine_ttft_s,
            "response_ttft_s": result.response_ttft_s,
            "server_request_total_s": result.server_request_total_s,
            "wall_s": result.latency_s,
            "first_part": result.first_part,
            "metrics": metrics,
            "shared_context_sha256": sha256_text(common),
            "aligned_shared_context_sha256": sha256_text(aligned_common),
            "expected_prefix_cache_hit_tokens": prefix_cache[
                "expected_commit_tokens"
            ],
            "previous_question_or_answer_appended": False,
            "retrieval": (
                {
                    "effective_top_k": retrieval["effective_top_k"],
                    "candidate_count": len(retrieval["retrieved_blocks"]),
                    "selected_count": len(selected_blocks),
                    "retrieved_tokens_estimate": sum(
                        int(block["chunk_tokens"]) for block in selected_blocks
                    ),
                    "retrieved_block_ids": [
                        block["block_id"]
                        for block in selected_blocks
                    ],
                    "metadata_headers": not args.rag_no_metadata,
                    "dynamic_prompt_budget": args.rag_dynamic_budget,
                    # Offline prepare-rag freezes corpus/query embeddings and
                    # ranking before the generation server starts.  Preserve
                    # that measured work alongside the per-question prompt
                    # materialization so pre-answer latency can be reproduced
                    # without inferring missing time from wall-clock logs.
                    "index_and_ranking_sec": float(
                        (rag_state.get("timing") or {}).get("total_sec") or 0.0
                    ),
                    "block_embedding_sec": float(
                        (rag_state.get("timing") or {}).get("block_embedding_sec") or 0.0
                    ),
                    "query_embedding_sec": float(
                        (rag_state.get("timing") or {}).get("query_embedding_sec") or 0.0
                    ),
                    "ranking_sec": float(
                        (rag_state.get("timing") or {}).get("ranking_sec") or 0.0
                    ),
                    "history_index_build_sec": (
                        float((rag_state.get("timing") or {}).get("total_sec") or 0.0)
                        - float((rag_state.get("timing") or {}).get("query_embedding_sec") or 0.0)
                        - float((rag_state.get("timing") or {}).get("ranking_sec") or 0.0)
                    ),
                    "query_path_sec": (
                        float((rag_state.get("timing") or {}).get("query_embedding_sec") or 0.0)
                        + float((rag_state.get("timing") or {}).get("ranking_sec") or 0.0)
                        + rag_materialization_sec
                    ),
                    "prompt_materialization_sec": rag_materialization_sec,
                }
                if retrieval is not None else None
            ),
            "completed_at": now_iso(),
        }
        completed[question_index] = record
        if question_index in replacements:
            write_jsonl_atomic(
                results_path,
                [completed[index] for index in sorted(completed)],
            )
        else:
            append_jsonl_fsync(results_path, record)
        print(
            f"[{row.ordinal}] {method} {row.source} row={row.dataset_row} "
            f"q={question_index + 1}/{len(prepared['qa'])} "
            f"EM={int(metrics['exact_match'])} ttft={result.ttft_s} "
            f"wall={result.latency_s:.3f}s",
            flush=True,
        )

    records = [completed[index] for index in range(len(prepared["qa"]))]
    if method == "compact-rag":
        retrieval_rows = [
            record["retrieval"] for record in records
            if isinstance(record.get("retrieval"), dict)
        ]
        selection.update({
            "rag_index_and_ranking_sec": float(
                (rag_state.get("timing") or {}).get("total_sec") or 0.0
            ),
            "rag_block_embedding_sec": float(
                (rag_state.get("timing") or {}).get("block_embedding_sec") or 0.0
            ),
            "rag_query_embedding_sec": float(
                (rag_state.get("timing") or {}).get("query_embedding_sec") or 0.0
            ),
            "rag_ranking_sec": float(
                (rag_state.get("timing") or {}).get("ranking_sec") or 0.0
            ),
            "mean_rag_prompt_materialization_sec": statistics.fmean(
                float(item.get("prompt_materialization_sec") or 0.0)
                for item in retrieval_rows
            ),
        })
    row_summary = {
        "schema_version": 1,
        "method": method,
        "split": row.split,
        "source": row.source,
        "family": source_family(row.source),
        "dataset_row": row.dataset_row,
        "context_chars": len(prepared["context"]),
        "context_sha256": sha256_text(prepared["context"]),
        "questions_expected": len(prepared["qa"]),
        "questions_completed": len(records),
        "exact_match_count": sum(
            int(record["metrics"]["exact_match"]) for record in records
        ),
        "substring_exact_match_count": sum(
            int(record["metrics"]["substring_exact_match"]) for record in records
        ),
        "mean_ttft_s": statistics.fmean(
            record["ttft_s"] for record in records
            if record["ttft_s"] is not None
        ),
        "mean_query_wall_s": statistics.fmean(
            record["wall_s"] for record in records
        ),
        "prefix_cache": prefix_cache,
        "selection": selection,
        "rag_config_sha256": (
            rag_state.get("config_sha256") if rag_state is not None else None
        ),
        "results": str(results_path.resolve()),
        "completed_at": now_iso(),
    }
    # A diagnostic --question-limit is intentionally resumable into the full
    # row.  Do not create the completion sentinel until every official question
    # is present, otherwise a later full run would incorrectly skip the row.
    if len(records) == int(prepared["total_questions"]):
        write_json_atomic(summary_path, row_summary)
    else:
        write_json_atomic(row_dir / "partial_summary.json", row_summary)
    return row_summary


def write_audit(args: argparse.Namespace, work: list[WorkRow]) -> dict[str, Any]:
    old_efficiency = Path("/home/chaidi/kvmem-efficiency-bench")
    report = {
        "created_at": now_iso(),
        "status": "adjustment-required-and-implemented",
        "old_harness": {
            "path": str(old_efficiency),
            "compatible_directly": False,
            "gaps": [
                "AgentLongBench/BEAM one-question sample schema, not MemoryAgentBench rows",
                "one final question per context instead of 3,671 frozen branches",
                "answer_max_tokens fixed to 32 instead of source-specific 10..2000",
                "temperature=0/top_p=1 instead of 0.6/0.95/top_k=20",
                "thinking disabled matches KVMem, but other generation controls do not",
                "RAG recomputes context embeddings in retrieve() for every question",
                "plain requests do not explicitly prime a reusable row-level prefix",
            ],
            "reused_semantics": [
                "Codex question-independent checkpoint compact prompt",
                "Jina v2 small English pinned model and Qwen token chunks",
                "full-history final-only RAG, relevance select then chronological prompt order",
                "complete-prompt recent sliding-window budget",
            ],
        },
        "new_runner": {
            "path": str(Path(__file__).resolve()),
            "contexts": len(work),
            "question_semantics": "independent branches from one immutable row context",
            "context_reuse": "lossless dense QW3 prefix cache; KVMem disabled",
            "previous_answers_appended": False,
            "sampling": "temperature=0.6, top_p=0.95, top_k=20, no thinking",
            "output_limits": MAX_TOKENS,
            "official_scorers_compatible": True,
        },
    }
    write_json_atomic(args.workspace / "baseline_compatibility_audit.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("audit", "prepare", "prepare-rag", "run", "status")
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/"
            "MemoryAgentBench/data"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(
            "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/"
            "Qwen3.6-27B-FP8"
        ),
    )
    parser.add_argument(
        "--rag-model",
        type=Path,
        default=Path(
            "/data/chaidi/kvmem_eval/models/jinaai/"
            "jina-embeddings-v2-small-en"
        ),
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:18090/v1")
    parser.add_argument("--model-name", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--read-timeout", type=float, default=7200)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--server-mtp-chain", type=int, default=4,
        help="record the MTP chain used by the externally managed QW3 server",
    )
    parser.add_argument("--context-window", type=int, default=262144)
    parser.add_argument("--generation-reserve", type=int, default=32768)
    parser.add_argument("--compaction-input-tokens", type=int, default=232000)
    parser.add_argument("--summary-max-tokens", type=int, default=25000)
    parser.add_argument(
        "--answer-max-tokens-override",
        type=int,
        default=0,
        help=(
            "override source-specific answer limits; 0 preserves utility "
            "semantics, while 1 is intended only for TTFT sampling"
        ),
    )
    parser.add_argument("--carry-max-tokens", type=int, default=70000)
    parser.add_argument("--compact-common-open-tokens", type=int, default=192000)
    parser.add_argument("--max-compact-rounds", type=int, default=16)
    parser.add_argument(
        "--compact-full-history",
        action="store_true",
        help="continue question-independent compaction until all history is summarized",
    )
    parser.add_argument(
        "--compact-no-tail",
        action="store_true",
        help="exclude raw recent history from the final compact prompt",
    )
    parser.add_argument(
        "--source-shared-workspace",
        type=Path,
        help="import validated question-independent compact checkpoints from this workspace",
    )
    parser.add_argument(
        "--strict-final-prompt-tokens",
        type=int,
        help="complete final prompt cap for compact methods, including chat template",
    )
    parser.add_argument("--sliding-prompt-tokens", type=int, default=32768)
    parser.add_argument("--prefix-page-tokens", type=int, default=16)
    parser.add_argument(
        "--prefix-cache-guard-pages",
        type=int,
        default=1,
        help=(
            "number of complete KV pages kept behind the natural cache "
            "commit boundary; the server must use the same value"
        ),
    )
    parser.add_argument("--rag-top-k", type=int, default=30)
    parser.add_argument("--rag-block-size", type=int, default=1024)
    parser.add_argument("--rag-overlap", type=int, default=128)
    parser.add_argument(
        "--rag-dynamic-budget",
        action="store_true",
        help="use as many top-ranked RAG candidates as fit the strict final prompt cap",
    )
    parser.add_argument(
        "--rag-no-metadata",
        action="store_true",
        help="materialize retrieved source text without rank/block-id headers",
    )
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--timestamp", default="2026-08-02 00:00:00")
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--source", action="append")
    parser.add_argument("--row", type=int, action="append")
    parser.add_argument("--max-contexts", type=int)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument(
        "--repair-question-index",
        type=int,
        action="append",
        help=(
            "zero-based question index to regenerate in an otherwise "
            "completed row; repeat for multiple indices"
        ),
    )
    parser.add_argument(
        "--prepare-rag-respect-question-limit",
        action="store_true",
        help=(
            "prepare only the diagnostic question subset selected by "
            "--question-limit; intended for independent-query latency runs"
        ),
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path(
            "/data/chaidi/kvmem_eval/results/"
            "memoryagentbench_kvmem_archive_20260802_full"
        ),
        help=(
            "KVMem result root supplying authoritative Qwen token lengths for "
            "phased context-length filters"
        ),
    )
    parser.add_argument("--min-context-tokens-exclusive", type=int)
    parser.add_argument("--max-context-tokens-inclusive", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "run" and args.method is None:
        raise BaselineError("run requires --method")
    if args.question_limit is not None and args.question_limit <= 0:
        raise BaselineError("--question-limit must be positive")
    if args.repair_question_index and args.command != "run":
        raise BaselineError("--repair-question-index is valid only with run")
    if args.repair_question_index and args.max_contexts not in (None, 1):
        raise BaselineError(
            "--repair-question-index requires a single selected row"
        )
    if args.prepare_rag_respect_question_limit and args.question_limit is None:
        raise BaselineError(
            "--prepare-rag-respect-question-limit requires --question-limit"
        )
    if args.answer_max_tokens_override < 0:
        raise BaselineError("--answer-max-tokens-override must be non-negative")
    if args.server_mtp_chain < 0:
        raise BaselineError("--server-mtp-chain must be non-negative")
    if args.prefix_page_tokens <= 0:
        raise BaselineError("--prefix-page-tokens must be positive")
    if not 0 <= args.prefix_cache_guard_pages <= 16:
        raise BaselineError("--prefix-cache-guard-pages must be in [0, 16]")
    if args.rag_overlap < 0 or args.rag_overlap >= args.rag_block_size:
        raise BaselineError("RAG overlap must satisfy 0 <= overlap < block size")
    if args.context_window - args.generation_reserve <= 0:
        raise BaselineError("generation reserve consumes the context window")
    compact_limit = (
        args.strict_final_prompt_tokens
        if args.strict_final_prompt_tokens is not None
        else args.context_window - args.generation_reserve
    )
    if args.strict_final_prompt_tokens is not None and not (
        0 < args.strict_final_prompt_tokens <= args.context_window
    ):
        raise BaselineError(
            "--strict-final-prompt-tokens must be in (0, --context-window]"
        )
    if not args.compact_full_history and args.compact_common_open_tokens >= compact_limit:
        raise BaselineError("compact common prefix leaves no final-query/RAG headroom")
    if args.compact_no_tail and not args.compact_full_history:
        raise BaselineError("--compact-no-tail requires --compact-full-history")
    if args.rag_dynamic_budget and args.strict_final_prompt_tokens is None:
        raise BaselineError(
            "--rag-dynamic-budget requires --strict-final-prompt-tokens"
        )
    if args.rag_no_metadata and not args.rag_dynamic_budget:
        raise BaselineError("--rag-no-metadata requires --rag-dynamic-budget")
    if (
        args.source_shared_workspace is not None
        and not args.source_shared_workspace.is_dir()
    ):
        raise BaselineError(
            f"source shared workspace not found: {args.source_shared_workspace}"
        )
    if (
        args.min_context_tokens_exclusive is not None
        and args.max_context_tokens_inclusive is not None
        and args.min_context_tokens_exclusive >= args.max_context_tokens_inclusive
    ):
        raise BaselineError(
            "context token filters require min-exclusive < max-inclusive"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.workspace = args.workspace.resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    work = enumerate_work(args)
    if args.command == "audit":
        print(json.dumps(write_audit(args, work), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        for row in work:
            prepare_row(args, row)
            print(f"[{row.ordinal}/{len(work)}] prepared {row.source}", flush=True)
        write_audit(args, work)
        return 0
    if args.command == "status":
        status = {"contexts_total": len(work), "methods": {}}
        for method in METHODS:
            root = args.workspace / "methods" / method
            rows = list((root / "rows").glob("*/row_summary.json"))
            results = list((root / "rows").glob("*/results.jsonl"))
            questions = sum(len(read_jsonl(path)) for path in results)
            status["methods"][method] = {
                "contexts_completed": len(rows),
                "questions_written": questions,
            }
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    counter = TokenCounter(args.tokenizer_dir)
    if args.command == "prepare-rag":
        module = load_module(RAG_REFERENCE, "_mab_rag_reference")
        module.configure_rag_parameters(
            args.rag_top_k, args.rag_block_size, args.rag_overlap
        )
        retriever = module.RagRetriever(
            model_path=str(args.rag_model),
            chunk_tokenizer_path=str(args.tokenizer_dir),
            device=args.embedding_device,
            batch_size=args.embedding_batch_size,
        )
        for row in work:
            prepared = load_prepared(
                args,
                row,
                apply_question_limit=args.prepare_rag_respect_question_limit,
            )
            state = prepare_rag_row(args, row, prepared, retriever, module)
            print(
                f"[{row.ordinal}/{len(work)}] RAG {row.source} "
                f"blocks={state['total_blocks']} q={state['questions']} "
                f"time={state['timing']['total_sec']:.3f}s",
                flush=True,
            )
        return 0

    client = client_from_args(args)
    if not client.health():
        raise BaselineError(f"QW3 server is not healthy at {args.api_base}")
    assert args.method is not None
    method_root = args.workspace / "methods" / args.method
    config = method_config(args, args.method)
    # run_config.json remains the convenient "last invocation" pointer, while
    # the phase-specific file is the durable experiment record.
    write_json_atomic(method_root / "run_config.json", config)
    write_json_atomic(method_root / method_config_filename(args), config)
    for row in work:
        lock_path = method_root / "locks" / f"{row.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            # Multiple filtered runners may fill disjoint rows against one
            # max_num_seqs>1 server. If their selections eventually overlap,
            # block here and re-read the completed row instead of racing its
            # JSONL/checkpoint files.
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            prepared = load_prepared(args, row)
            summary = run_method_row(
                args, row, args.method, prepared, client, counter
            )
        aggregate = aggregate_method(method_root)
        print(
            f"[{row.ordinal}/{len(work)}] DONE {args.method} {row.source} "
            f"q={summary['questions_completed']} "
            f"total_contexts={aggregate['contexts_completed']} "
            f"total_questions={aggregate['questions_completed']}",
            flush=True,
        )
    print(json.dumps(aggregate_method(method_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
