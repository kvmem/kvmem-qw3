#!/usr/bin/env python3
"""Run the LongMemEval-M selected-KV + rebuilt-DeltaNet-state control.

The experiment consumes a retrieval dump captured by the exact same binary and
KVMem configuration as the import run, then has two explicit state phases:

1. Export: prefill the exact token IDs from the selected source blocks only and
   save the resulting DeltaNet recurrent/conv state (max_tokens=0).
2. Evaluate: run the original long KVMem request.  After its ordinary mean-k
   selection, qw3 keeps the selected normal-attention KV unchanged, verifies the
   selected source token sequence against the artifact, imports only recurrent /
   conv state, then replays the final query and decodes.

The score dump is authoritative for source block IDs and token IDs.  Import is
collision-safe: both the artifact key and its manifest are bound to the selected
token SHA, and qw3 compares the complete source token vector before changing any
live recurrent state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from typing import Any

import requests

try:
    from .client import Qw3Client
    from .dataset import load_all
    from .judge import DeepSeekJudge
    from .prompt import render_messages
except ImportError:
    from client import Qw3Client  # type: ignore
    from dataset import load_all  # type: ignore
    from judge import DeepSeekJudge  # type: ignore
    from prompt import render_messages  # type: ignore


DEFAULT_DATA = Path("/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # Deliberately required: a historical dump is not a safe default because
    # immutable-K/query-replay changes can alter blocks close to the top-k cutoff.
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--phase", choices=("export", "eval", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--expected-immutable-source-k", type=int, choices=(0, 1))
    parser.add_argument("--expected-block-tokens", type=int)
    parser.add_argument("--expected-budget-blocks", type=int)
    parser.add_argument("--expected-sink-blocks", type=int)
    parser.add_argument("--expected-recent-blocks", type=int)
    parser.add_argument("--expected-method")
    parser.add_argument(
        "--seed-state-prefix",
        help="initialize each dense selected-token export from PREFIX-<question_id>",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def ids_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in read_jsonl(path):
        if row.get("type") == "meta":
            if current is not None:
                snapshots.append(current)
            current = {"meta": row, "blocks": []}
        elif current is None:
            raise RuntimeError("retrieval dump begins with a block row")
        else:
            current["blocks"].append(row)
    if current is not None:
        snapshots.append(current)
    seen: set[str] = set()
    for snapshot in snapshots:
        meta = snapshot["meta"]
        tag = str(meta.get("trace_tag") or "")
        if tag.startswith("lme-"):
            tag = tag[4:]
        if not tag or tag in seen:
            raise RuntimeError(f"missing or duplicate retrieval trace tag {tag!r}")
        seen.add(tag)
        snapshot["question_id"] = tag
        if meta.get("schema_version") != "qw3_kvmem_retrieval_dump.v2":
            raise RuntimeError(f"unsupported retrieval dump schema for {tag}")
        if len(snapshot["blocks"]) != int(meta.get("block_count", -1)):
            raise RuntimeError(f"retrieval block count mismatch for {tag}")
    return snapshots


def prepare(snapshot: dict[str, Any]) -> dict[str, Any]:
    meta = snapshot["meta"]
    question_id = str(snapshot["question_id"])
    block_tokens = int(meta["block_tokens"])
    query_begin = int(meta["query_begin"])
    replay_boundary = (query_begin // block_tokens) * block_tokens
    selected = sorted(
        (row for row in snapshot["blocks"] if int(row.get("sel", 0)) == 1),
        key=lambda row: int(row["p0"]),
    )
    source_blocks = [row for row in selected if int(row["p0"]) < replay_boundary]
    source_tokens: list[int] = []
    source_block_ids: list[int] = []
    source_ranges: list[list[int]] = []
    previous_end = -1
    for block in source_blocks:
        p0 = int(block["p0"])
        nt = int(block["nt"])
        token_ids = block.get("tok")
        if not isinstance(token_ids, list) or len(token_ids) != nt:
            raise RuntimeError(
                f"selected source block lacks exact token IDs: {question_id} "
                f"b={block.get('b')}"
            )
        if p0 < previous_end or p0 + nt > replay_boundary:
            raise RuntimeError(f"invalid selected source ordering for {question_id}")
        previous_end = p0 + nt
        source_block_ids.append(int(block["b"]))
        source_ranges.append([p0, p0 + nt])
        source_tokens.extend(int(value) for value in token_ids)
    if not source_tokens:
        raise RuntimeError(f"no selected source tokens for {question_id}")
    source_token_sha256 = ids_sha256(source_tokens)
    # Bind the on-disk artifact name to the exact selected-token sequence.  A
    # stale state from another retrieval configuration can no longer be resumed
    # merely because it has the same question id.
    key = f"lme-{question_id}-{source_token_sha256[:16]}"
    return {
        "question_id": question_id,
        "key": key,
        "query_span": [int(meta["query_begin"]), int(meta["query_end"])],
        "replay_boundary": replay_boundary,
        "selected_total_blocks": len(selected),
        "selected_source_blocks": len(source_blocks),
        "selected_source_token_count": len(source_tokens),
        "selected_source_token_sha256": source_token_sha256,
        "selected_source_block_ids": source_block_ids,
        "selected_source_ranges": source_ranges,
        "source_tokens": source_tokens,
    }


def validate_dump_config(snapshots: list[dict[str, Any]],
                         args: argparse.Namespace) -> dict[str, Any]:
    if not snapshots:
        raise RuntimeError(f"retrieval dump contains no snapshots: {args.dump_file}")
    fields = {
        "immutable_source_k": args.expected_immutable_source_k,
        "block_tokens": args.expected_block_tokens,
        "budget_blocks": args.expected_budget_blocks,
        "sink": args.expected_sink_blocks,
        "recent": args.expected_recent_blocks,
        "method": args.expected_method,
    }
    reference = snapshots[0]["meta"]
    for name, expected in fields.items():
        values = {snapshot["meta"].get(name) for snapshot in snapshots}
        if len(values) != 1:
            raise RuntimeError(
                f"retrieval dump mixes {name} values: "
                f"{sorted(values, key=lambda value: str(value))}"
            )
        actual = next(iter(values))
        if expected is not None and actual != expected:
            raise RuntimeError(
                f"retrieval dump {name} mismatch: expected={expected!r} "
                f"actual={actual!r} file={args.dump_file}"
            )
    return {name: reference.get(name) for name in fields}


def validate_resume_row(kind: str, row: dict[str, Any],
                        item: dict[str, Any]) -> None:
    expected = {
        "key": item["key"],
        "source_tokens": item["selected_source_token_count"],
        "source_token_sha256": item["selected_source_token_sha256"],
    }
    aliases = {
        "key": ("key", "state_key"),
        "source_tokens": ("source_tokens", "selected_source_tokens"),
        "source_token_sha256": (
            "source_token_sha256", "selected_source_token_sha256"),
    }
    for name, wanted in expected.items():
        actual = next((row[key] for key in aliases[name] if key in row), None)
        if actual != wanted:
            raise RuntimeError(
                f"stale {kind} row for {item['question_id']}: "
                f"{name} expected={wanted!r} actual={actual!r}; use a fresh "
                "output/state directory"
            )


def root_url(api_base: str) -> str:
    root = api_base.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def export_state(session: requests.Session, args: argparse.Namespace,
                 item: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "model": args.model,
        "prompt": item["source_tokens"],
        "max_tokens": 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "kvmem_rebuilt_state_export": item["key"],
        "stream": False,
    }
    seed_key = None
    if args.seed_state_prefix:
        seed_key = f"{args.seed_state_prefix}-{item['question_id']}"
        payload["kvmem_rebuilt_state_seed"] = seed_key
    response = session.post(
        args.api_base.rstrip("/") + "/completions",
        json=payload,
        timeout=(30, args.timeout_sec),
        headers={"Authorization": "Bearer dummy"},
    )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(
            f"state export HTTP {response.status_code} for {item['question_id']}: "
            f"{response.text[:2000]}"
        )
    body = response.json()
    usage = body.get("usage") or {}
    if int(usage.get("prompt_tokens", -1)) != item["selected_source_token_count"]:
        raise RuntimeError(f"state export prompt-token mismatch for {item['question_id']}")
    return {
        "question_id": item["question_id"],
        "key": item["key"],
        "source_tokens": item["selected_source_token_count"],
        "source_token_sha256": item["selected_source_token_sha256"],
        "latency_s": elapsed,
        "finish_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
        "usage": usage,
        "seed_state_key": seed_key,
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("QW3_KVMEM_REBUILT_STATE_DIR") not in (None, str(args.state_dir)):
        raise RuntimeError(
            "local --state-dir differs from QW3_KVMEM_REBUILT_STATE_DIR; "
            "start the server and this script with the same absolute directory"
        )

    snapshots = load_snapshots(args.dump_file)
    if args.limit is not None:
        snapshots = snapshots[: args.limit]
    dump_config = validate_dump_config(snapshots, args)
    items = [prepare(snapshot) for snapshot in snapshots]
    samples = {sample.question_id: sample for sample in load_all(args.data)}
    for item in items:
        if item["question_id"] not in samples:
            raise RuntimeError(f"dataset lacks question {item['question_id']}")

    audit_path = args.output_root / "selection_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for item in items:
            public = {key: value for key, value in item.items() if key != "source_tokens"}
            handle.write(json.dumps(public, separators=(",", ":")) + "\n")
    write_json(
        args.output_root / "config.json",
        {
            "schema_version": "longmemeval_kvmem_rebuilt_state.v1",
            "mode": "fixed_selected_normal_kv_plus_dense_rebuilt_deltanet_state",
            "data": str(args.data),
            "dump_file": str(args.dump_file),
            "dump_file_sha256": file_sha256(args.dump_file),
            "dump_config": dump_config,
            "state_dir": str(args.state_dir),
            "question_ids": [item["question_id"] for item in items],
            "source_token_identity": "complete uint32 vector embedded and compared on import",
            "normal_kv_policy": "ordinary KVMem selection retained byte-for-byte",
            "delta_state_policy": (
                "accumulated history state plus selected-token refresh"
                if args.seed_state_prefix else
                "selected-token rebuild from zero state"
            ),
            "seed_state_prefix": args.seed_state_prefix,
        },
    )

    session = requests.Session()
    session.trust_env = False
    health = session.get(root_url(args.api_base) + "/health", timeout=(10, 10))
    if health.status_code != 200:
        raise RuntimeError(f"qw3 server is not healthy at {args.api_base}")

    export_path = args.output_root / "state_exports.jsonl"
    exported = {str(row["question_id"]): row for row in read_jsonl(export_path)}
    if args.phase in ("export", "all"):
        for position, item in enumerate(items, start=1):
            if args.seed_state_prefix:
                seed_key = f"{args.seed_state_prefix}-{item['question_id']}"
                seed_path = args.state_dir / f"{seed_key}.qw3-deltanet-state"
                if not seed_path.exists() or seed_path.stat().st_size == 0:
                    raise RuntimeError(f"missing accumulated seed state {seed_path}")
            state_path = args.state_dir / f"{item['key']}.qw3-deltanet-state"
            if item["question_id"] in exported and state_path.exists():
                validate_resume_row("state export", exported[item["question_id"]], item)
                print(f"[export-skip] {position}/{len(items)} id={item['question_id']}", flush=True)
                continue
            print(
                f"[export] {position}/{len(items)} id={item['question_id']} "
                f"blocks={item['selected_source_blocks']} "
                f"tokens={item['selected_source_token_count']}",
                flush=True,
            )
            row = export_state(session, args, item)
            if not state_path.exists() or state_path.stat().st_size == 0:
                raise RuntimeError(f"server did not publish state artifact {state_path}")
            row["artifact"] = str(state_path)
            row["artifact_bytes"] = state_path.stat().st_size
            append_jsonl(export_path, row)
            exported[item["question_id"]] = row
            print(
                f"[exported] id={item['question_id']} latency={row['latency_s']:.1f}s "
                f"bytes={row['artifact_bytes']}",
                flush=True,
            )

    if args.phase == "export":
        return
    for item in items:
        state_path = args.state_dir / f"{item['key']}.qw3-deltanet-state"
        if not state_path.exists():
            raise RuntimeError(f"missing rebuilt-state artifact {state_path}")

    client = Qw3Client(
        base_url=args.api_base,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        enable_thinking=True,
        read_timeout=args.timeout_sec,
    )
    judge = None if args.no_judge else DeepSeekJudge()
    baseline_by_id = (
        {str(row["question_id"]): row for row in read_jsonl(args.baseline)}
        if args.baseline else {}
    )
    result_path = args.output_root / "results.jsonl"
    results = {str(row["question_id"]): row for row in read_jsonl(result_path)}
    for position, item in enumerate(items, start=1):
        question_id = item["question_id"]
        if question_id in results:
            validate_resume_row("evaluation", results[question_id], item)
            print(f"[eval-skip] {position}/{len(items)} id={question_id}", flush=True)
            continue
        sample = samples[question_id]
        print(f"[eval] {position}/{len(items)} id={question_id}", flush=True)
        result = client.chat(
            render_messages(sample),
            extra_body={
                "kvmem_rebuilt_state_import": item["key"],
                "kvmem_trace_tag": f"rebuilt-{question_id}",
            },
        )
        verdict = None if result.error or judge is None else judge.judge(sample, result.answer)
        baseline = baseline_by_id.get(question_id, {})
        row = {
            "position": position,
            "question_id": question_id,
            "question_type": sample.question_type,
            "question": sample.question,
            "gold": sample.answer,
            "answer": result.answer,
            "correct": verdict.correct if verdict else None,
            "judge_raw": verdict.raw if verdict else None,
            "judge_error": verdict.error if verdict else None,
            "client_error": result.error,
            "finish_reason": result.finish_reason,
            "truncated": result.truncated,
            "ttft_s": result.ttft_s,
            "latency_s": result.latency_s,
            "reasoning_chars": len(result.reasoning),
            "selected_source_blocks": item["selected_source_blocks"],
            "selected_source_tokens": item["selected_source_token_count"],
            "selected_source_token_sha256": item["selected_source_token_sha256"],
            "state_key": item["key"],
            "baseline_correct": baseline.get("correct"),
            "baseline_answer": baseline.get("answer"),
        }
        append_jsonl(result_path, row)
        results[question_id] = row
        print(
            f"[result] id={question_id} correct={row['correct']} "
            f"ttft={row['ttft_s']} latency={row['latency_s']:.1f}s "
            f"error={row['client_error']}",
            flush=True,
        )

    final = [results[item["question_id"]] for item in items]
    judged = [row for row in final if row.get("correct") is not None]
    correct = sum(row.get("correct") is True for row in judged)
    summary = {
        "schema_version": "longmemeval_kvmem_rebuilt_state.summary.v1",
        "n_samples": len(final),
        "n_judged": len(judged),
        "n_correct": correct,
        "accuracy": 100.0 * correct / len(judged) if judged else None,
        "n_errors": sum(bool(row.get("client_error")) for row in final),
        "mean_ttft_s": (
            sum(float(row["ttft_s"]) for row in final if row.get("ttft_s") is not None)
            / max(1, sum(row.get("ttft_s") is not None for row in final))
        ),
        "results": str(result_path),
        "selection_audit": str(audit_path),
        "state_exports": str(export_path),
    }
    write_json(args.output_root / "summary.json", summary)
    print(f"[complete] correct={correct}/{len(judged)} summary={args.output_root/'summary.json'}")


if __name__ == "__main__":
    main()
