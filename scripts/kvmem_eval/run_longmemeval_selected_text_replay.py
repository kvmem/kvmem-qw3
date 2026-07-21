#!/usr/bin/env python3
"""Dense-prefill the text decoded from exact KVMem-selected LongMemEval blocks.

This is a representation control: the selected source token set and order are
held fixed, but KV tensors are discarded and recomputed from decoded text at
compact contiguous positions.  The raw replay includes the selected system/task
instruction and final query blocks exactly once because both are part of the
actual KVMem-selected window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Any

import requests

try:
    from .dataset import load_all
    from .judge import DeepSeekJudge
except ImportError:
    from dataset import load_all  # type: ignore
    from judge import DeepSeekJudge  # type: ignore


DEFAULT_DATA = Path("/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl")
DEFAULT_BASELINE = Path(
    "/data/chaidi/kvmem_eval/results/"
    "m102_2m_k224k_g32k_bt32_meank_r051_20260719_eval_20260719_063739.jsonl"
)
DEFAULT_TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/Qwen3.6-27B-FP8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--context-window", type=int, default=262_144)
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--timeout-sec", type=float, default=7200.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
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


def ids_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(struct.pack("<i", token_id))
    return digest.hexdigest()


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def covered_tokens(blocks: list[dict[str, Any]], begin: int, end: int) -> int:
    return sum(
        max(
            0,
            min(int(block["p0"]) + int(block["nt"]), end)
            - max(int(block["p0"]), begin),
        )
        for block in blocks
    )


def load_snapshots(path: Path) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in read_jsonl(path):
        if row.get("type") == "meta":
            if current is not None:
                snapshots.append(current)
            current = {"meta": row, "blocks": []}
        elif current is None:
            raise RuntimeError("dump begins with a block row")
        else:
            current["blocks"].append(row)
    if current is not None:
        snapshots.append(current)
    seen: set[str] = set()
    for snapshot in snapshots:
        meta = snapshot["meta"]
        tag = str(meta.get("trace_tag") or "")
        if not tag or tag in seen:
            raise RuntimeError(f"missing or duplicate trace tag {tag!r}")
        seen.add(tag)
        if meta.get("schema_version") != "qw3_kvmem_retrieval_dump.v2":
            raise RuntimeError(f"unsupported dump schema for {tag}")
        if len(snapshot["blocks"]) != int(meta.get("block_count", -1)):
            raise RuntimeError(f"block count mismatch for {tag}")
    return snapshots


def split_reasoning(raw: str) -> tuple[str, str]:
    if "</think>" not in raw:
        return "", raw.strip()
    reasoning, answer = raw.rsplit("</think>", 1)
    return reasoning.strip(), answer.strip()


def server_token_ids(session: requests.Session, api_base: str, text: str, timeout: float) -> list[int]:
    response = session.post(
        api_base.rstrip("/") + "/tokenize",
        json={"content": text, "return_tokens": True},
        timeout=(30, timeout),
        headers={"Authorization": "Bearer dummy"},
    )
    if response.status_code != 200:
        raise RuntimeError(f"tokenize HTTP {response.status_code}: {response.text[:1000]}")
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("/tokenize response does not contain tokens[]")
    return [int(value) for value in tokens]


def complete(
    session: requests.Session, args: argparse.Namespace, prompt: str, max_tokens: int
) -> dict[str, Any]:
    started = time.perf_counter()
    response = session.post(
        args.api_base.rstrip("/") + "/completions",
        json={
            "model": args.model,
            "prompt": prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": max_tokens,
            "thinking_budget": args.thinking_budget,
            "stream": False,
        },
        timeout=(30, args.timeout_sec),
        headers={"Authorization": "Bearer dummy"},
    )
    elapsed = time.perf_counter() - started
    if response.status_code != 200:
        raise RuntimeError(f"completion HTTP {response.status_code}: {response.text[:2000]}")
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("completion response has no choices")
    raw = str(choices[0].get("text") or "")
    reasoning, answer = split_reasoning(raw)
    return {
        "raw_response": raw,
        "reasoning": reasoning,
        "answer": answer,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": body.get("usage"),
        "latency_s": elapsed,
    }


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("run with /home/chaidi/qw3/.venv/bin/python") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    samples = load_all(args.data)
    samples_by_id = {sample.question_id: sample for sample in samples}
    baseline_by_id = {
        str(row["question_id"]): row for row in read_jsonl(args.baseline)
    }
    snapshots = load_snapshots(args.dump_file)
    if args.limit is not None:
        snapshots = snapshots[: args.limit]
    if not snapshots:
        raise RuntimeError("no snapshots selected")
    args.output_root.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    for snapshot in snapshots:
        meta = snapshot["meta"]
        tag = str(meta["trace_tag"])
        # Canonical one-shot captures use the bare question id; transcript
        # replay tags requests as ``lme-<question_id>`` so per-event score dumps
        # cannot collide with unrelated traces. Both identify the same sample.
        question_id = tag[4:] if tag.startswith("lme-") else tag
        sample = samples_by_id.get(question_id)
        baseline = baseline_by_id.get(question_id)
        if sample is None or baseline is None:
            raise RuntimeError(f"dataset/baseline missing trace tag {tag}")
        # `tok` is copied by the executor directly from the exact token vector
        # used for server prefill (`kvmem_trace_prompt_tokens_[p0:p0+nt]`).  It is
        # therefore the authoritative source for this control.  Do not recreate
        # the full three-message prompt with Hugging Face's chat template here:
        # qw3's native renderer and that template differ at consecutive-user
        # message boundaries (25 tokens on the first smoke sample).
        original_prompt_tokens = int(meta.get("prompt_tokens", -1))
        if original_prompt_tokens <= 0:
            raise RuntimeError(f"dump lacks prompt_tokens for {tag}")
        selected = sorted(
            (row for row in snapshot["blocks"] if int(row.get("sel", 0)) == 1),
            key=lambda row: int(row["p0"]),
        )
        if len(selected) != int(meta.get("selected", -1)):
            raise RuntimeError(f"selected block count mismatch for {tag}")
        selected_ids: list[int] = []
        previous_end = -1
        contiguous_runs = 0
        for block in selected:
            p0, nt = int(block["p0"]), int(block["nt"])
            token_ids = block.get("tok")
            if not isinstance(token_ids, list) or len(token_ids) != nt:
                raise RuntimeError(f"selected block lacks exact token IDs: {tag} b={block.get('b')}")
            token_ids = [int(value) for value in token_ids]
            if p0 < previous_end:
                raise RuntimeError(f"selected blocks overlap for {tag}")
            if p0 < 0 or p0 + nt > original_prompt_tokens:
                raise RuntimeError(f"selected block is outside server prompt for {tag}")
            if p0 != previous_end:
                contiguous_runs += 1
            previous_end = p0 + nt
            selected_ids.extend(token_ids)
        q0, q1 = int(meta["query_begin"]), int(meta["query_end"])
        selected_query_tokens = covered_tokens(selected, q0, q1)
        if selected_query_tokens != q1 - q0:
            raise RuntimeError(
                f"final query is not fully selected for {tag}: "
                f"selected={selected_query_tokens} expected={q1-q0}"
            )
        context_begin = int(meta.get("context_begin", 0))
        prefix_tokens = covered_tokens(selected, 0, context_begin)
        if prefix_tokens != context_begin:
            raise RuntimeError(
                f"system/task prefix is not fully selected for {tag}: "
                f"selected={prefix_tokens} expected={context_begin}"
            )
        suffix_tokens = covered_tokens(selected, q1, original_prompt_tokens)
        if suffix_tokens != original_prompt_tokens - q1:
            # Transcript score dumps are taken at the final semantic reselect,
            # immediately after the user content. The server then prefills the
            # fixed seven-token assistant/thinking header without another
            # selection, so it is absent from block_store yet mandatory for the
            # raw /completions control. Reconstruct that deterministic template
            # suffix; ordinary one-shot dumps already contain it and stay exact.
            transcript_suffix = tokenizer.encode(
                "<|im_end|>\n<|im_start|>assistant\n<think>\n",
                add_special_tokens=False,
            )
            missing = original_prompt_tokens - q1 - suffix_tokens
            if tag.startswith("lme-") and suffix_tokens == 0 and \
                    missing == len(transcript_suffix):
                selected_ids.extend(int(value) for value in transcript_suffix)
                suffix_tokens += len(transcript_suffix)
            else:
                raise RuntimeError(
                    f"assistant generation prefix is not fully selected for {tag}"
                )
        replay_prompt = tokenizer.decode(
            selected_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        replay_ids = [
            int(value) for value in tokenizer.encode(replay_prompt, add_special_tokens=False)
        ]
        prepared.append(
            {
                "question_id": question_id,
                "sample": sample,
                "baseline": baseline,
                "original_prompt_tokens": original_prompt_tokens,
                "selected_block_count": len(selected),
                "selected_token_count": len(selected_ids),
                "selected_token_sha256": ids_sha256(selected_ids),
                "selected_query_tokens": selected_query_tokens,
                "query_span": [q0, q1],
                "selected_system_task_prefix_tokens": prefix_tokens,
                "system_task_prefix_span": [0, context_begin],
                "selected_generation_prefix_tokens": suffix_tokens,
                "generation_prefix_span": [q1, original_prompt_tokens],
                "selected_run_count": contiguous_runs,
                "replay_prompt": replay_prompt,
                "replay_prompt_sha256": hashlib.sha256(replay_prompt.encode("utf-8")).hexdigest(),
                "replay_token_count_local": len(replay_ids),
                "replay_token_sha256_local": ids_sha256(replay_ids),
                "retokenization_exact": replay_ids == selected_ids,
                "retokenization_token_delta": len(replay_ids) - len(selected_ids),
                "retokenization_first_mismatch": first_mismatch(selected_ids, replay_ids),
            }
        )

    with (args.output_root / "replay_prompts.jsonl").open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(
                json.dumps(
                    {
                        "question_id": item["question_id"],
                        "replay_prompt_sha256": item["replay_prompt_sha256"],
                        "replay_prompt": item["replay_prompt"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    audit_keys = [
        "question_id",
        "original_prompt_tokens",
        "selected_block_count",
        "selected_token_count",
        "selected_token_sha256",
        "selected_query_tokens",
        "query_span",
        "selected_system_task_prefix_tokens",
        "system_task_prefix_span",
        "selected_generation_prefix_tokens",
        "generation_prefix_span",
        "selected_run_count",
        "replay_prompt_sha256",
        "replay_token_count_local",
        "replay_token_sha256_local",
        "retokenization_exact",
        "retokenization_token_delta",
        "retokenization_first_mismatch",
    ]
    with (args.output_root / "replay_audit.jsonl").open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(
                json.dumps({key: item[key] for key in audit_keys}, separators=(",", ":")) + "\n"
            )
    write_json(
        args.output_root / "replay_config.json",
        {
            "schema_version": "longmemeval_kvmem_selected_text_replay.v1",
            "mode": "decode_exact_selected_tokens_then_dense_raw_prefill",
            "selected_token_source": (
                "QW3 executor copy of the actual server-prefill prompt token vector "
                "at each selected block's [p0,p0+nt) span"
            ),
            "causal_scope": (
                "tests the whole cached-KV representation versus text-prefill difference; "
                "a change is not uniquely attributable to RoPE"
            ),
            "data": str(args.data),
            "baseline": str(args.baseline),
            "dump_file": str(args.dump_file),
            "tokenizer": str(args.tokenizer),
            "question_ids": [item["question_id"] for item in prepared],
            "context_window": args.context_window,
            "max_tokens": args.max_tokens,
            "thinking_budget": args.thinking_budget,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    )
    print(
        f"[prepared] samples={len(prepared)} audit={args.output_root/'replay_audit.jsonl'}",
        flush=True,
    )
    if args.prepare_only:
        return

    session = requests.Session()
    session.trust_env = False
    root = args.api_base.rstrip("/")
    health_root = root[:-3] if root.endswith("/v1") else root
    if session.get(health_root + "/health", timeout=(10, 10)).status_code != 200:
        raise RuntimeError(f"dense server is not healthy at {args.api_base}")
    judge = None if args.no_judge else DeepSeekJudge()
    result_path = args.output_root / "results.jsonl"
    server_token_audit_path = args.output_root / "server_tokenization_audit.jsonl"
    existing = {str(row.get("question_id")): row for row in read_jsonl(result_path)}
    existing_server_audit = {
        str(row.get("question_id")): row
        for row in read_jsonl(server_token_audit_path)
    }

    for position, item in enumerate(prepared, start=1):
        tag = item["question_id"]
        if tag in existing:
            print(f"[skip] {position}/{len(prepared)} id={tag}", flush=True)
            continue
        server_ids = server_token_ids(session, args.api_base, item["replay_prompt"], args.timeout_sec)
        local_ids = [
            int(value)
            for value in tokenizer.encode(item["replay_prompt"], add_special_tokens=False)
        ]
        server_local_mismatch = first_mismatch(local_ids, server_ids)
        server_local_exact = server_local_mismatch is None
        if tag not in existing_server_audit:
            token_audit = {
                "question_id": tag,
                "selected_source_tokens": item["selected_token_count"],
                "decoded_text_tokens_local": len(local_ids),
                "decoded_text_tokens_server": len(server_ids),
                "server_minus_selected": len(server_ids) - item["selected_token_count"],
                "server_minus_local": len(server_ids) - len(local_ids),
                "server_local_exact": server_local_exact,
                "server_local_first_mismatch": server_local_mismatch,
                "server_token_sha256": ids_sha256(server_ids),
                "local_token_sha256": ids_sha256(local_ids),
                "note": (
                    "The dense /completions endpoint uses the server tokenization. "
                    "Differences from source IDs are intrinsic to decode-concatenate-"
                    "retokenize text replay; local/server differences are audited, "
                    "not silently treated as source-token parity."
                ),
            }
            append_jsonl(server_token_audit_path, token_audit)
            existing_server_audit[tag] = token_audit
        if not server_local_exact:
            print(
                f"[tokenizer-audit] id={tag} local={len(local_ids)} "
                f"server={len(server_ids)} first_mismatch={server_local_mismatch}",
                flush=True,
            )
        request_max = min(
            args.max_tokens,
            max(1, args.context_window - len(server_ids) - 16),
        )
        print(
            f"[replay] {position}/{len(prepared)} id={tag} "
            f"selected={item['selected_token_count']} replay={len(server_ids)} max={request_max}",
            flush=True,
        )
        result = complete(session, args, item["replay_prompt"], request_max)
        verdict = judge.judge(item["sample"], result["answer"]) if judge else None
        baseline = item["baseline"]
        row = {
            "position": position,
            "subset_index": baseline.get("subset_index"),
            "question_id": tag,
            "question_type": item["sample"].question_type,
            "question": item["sample"].question,
            "gold": item["sample"].answer,
            "baseline_correct": baseline.get("correct"),
            "baseline_answer": baseline.get("answer"),
            "replay_correct": verdict.correct if verdict else None,
            "replay_judge_raw": verdict.raw if verdict else None,
            "replay_judge_error": verdict.error if verdict else None,
            "replay_answer": result["answer"],
            "replay_reasoning_chars": len(result["reasoning"]),
            "replay_finish_reason": result["finish_reason"],
            "replay_usage": result["usage"],
            "replay_latency_s": result["latency_s"],
            "selected_block_count": item["selected_block_count"],
            "selected_token_count": item["selected_token_count"],
            "selected_run_count": item["selected_run_count"],
            "replay_token_count": len(server_ids),
            "server_minus_selected_token_count": len(server_ids) - item["selected_token_count"],
            "server_local_tokenization_exact": server_local_exact,
            "server_local_first_mismatch": server_local_mismatch,
            "retokenization_token_delta": item["retokenization_token_delta"],
            "generation_max_tokens_request": request_max,
        }
        append_jsonl(result_path, row)
        existing[tag] = row
        print(
            f"[judge] id={tag} baseline={baseline.get('correct')} "
            f"replay={verdict.correct if verdict else None} "
            f"raw={verdict.raw if verdict else None!r} "
            f"error={verdict.error if verdict else None}",
            flush=True,
        )

    final_rows = [existing[item["question_id"]] for item in prepared]
    replay_correct = sum(row.get("replay_correct") is True for row in final_rows)
    judge_errors = sum(bool(row.get("replay_judge_error")) for row in final_rows)
    write_json(
        args.output_root / "summary.json",
        {
            "schema_version": "longmemeval_kvmem_selected_text_replay.summary.v1",
            "n_samples": len(final_rows),
            "baseline_correct": sum(row.get("baseline_correct") is True for row in final_rows),
            "replay_correct": replay_correct,
            "replay_accuracy": 100.0 * replay_correct / len(final_rows),
            "errors_fixed_by_text_replay": replay_correct,
            "judge_errors": judge_errors,
            "interpretation": {
                "replay_improves": (
                    "selected evidence was usable after fresh text prefill; investigate cached-KV "
                    "representation, re-RoPE/window positions, and contextual-state mismatch"
                ),
                "replay_does_not_improve": (
                    "the fixed selected source tokens are insufficient or too noisy; retrieval is "
                    "the leading cause for these samples"
                ),
            },
            "results": str(result_path),
        },
    )
    print(
        f"[complete] replay_correct={replay_correct}/{len(final_rows)} "
        f"summary={args.output_root/'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
