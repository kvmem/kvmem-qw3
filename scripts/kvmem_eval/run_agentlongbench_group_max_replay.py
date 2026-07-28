#!/usr/bin/env python3
"""Dense replay of a group-max re-selection over frozen KVMem block scores.

This is an offline retrieval-policy control.  It does not rescore or rerun the
1M prefill.  For one frozen score dump it:

* maps every original rendered history message to its overlapping 32-token
  KVMem blocks;
* pools each group's score with ``max(block retrieval_score)``;
* greedily admits complete groups under the original block budget;
* pins the original sink and final Question/Answer suffix;
* decodes exactly those source blocks in chronological order and dense-prefills
  the resulting text on a non-KVMem server.

``message-max`` uses one original message per group. ``round-bundle-max`` joins
all messages ending at each ``user: Round N`` feedback, preserving the tool
result-to-round binding that is not present inside the tool message itself.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
import re
import struct
import time
from typing import Any

from transformers import AutoTokenizer

import run_agentlongbench_kvmem as runner
import run_agentlongbench_selected_replay as selected_replay
from analyze_agentlongbench_gold_block_coverage import (
    load_dump,
    message_records,
)


DEFAULT_TOKENIZER = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/models/tokenizers/"
    "Qwen3.6-27B-FP8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kvmem-dump", type=Path, required=True)
    parser.add_argument("--kvmem-eval", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("message-max", "round-bundle-max"),
        required=True,
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def token_sha256(ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in ids:
        digest.update(struct.pack("<i", token_id))
    return digest.hexdigest()


def token_positions_for_chars(
    offset_starts: list[int],
    offset_ends: list[int],
    start: int,
    end: int,
) -> tuple[int, int]:
    left = bisect.bisect_right(offset_ends, start)
    right = bisect.bisect_left(offset_starts, end)
    while left < right and offset_ends[left] <= offset_starts[left]:
        left += 1
    while right > left and offset_ends[right - 1] <= offset_starts[right - 1]:
        right -= 1
    if left >= right:
        raise RuntimeError(f"character span [{start},{end}) maps to no tokens")
    return left, right


def build_groups(
    mode: str,
    records: list[dict[str, Any]],
    prompt_start: int,
    offset_starts: list[int],
    offset_ends: list[int],
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    char_groups: list[dict[str, Any]] = []
    if mode == "message-max":
        for record in records:
            char_groups.append(
                {
                    "label": f"message_{record['message_index']}",
                    "message_indices": [record["message_index"]],
                    "char_start": record["line_start"],
                    "char_end": record["line_end"],
                }
            )
    else:
        pending: list[dict[str, Any]] = []
        for record in records:
            pending.append(record)
            if (
                record["role"] == "user"
                and re.match(r"^Round \d+:", record["content"])
            ):
                match = re.match(r"^Round (\d+):", record["content"])
                assert match is not None
                char_groups.append(
                    {
                        "label": f"round_{match.group(1)}",
                        "message_indices": [
                            item["message_index"] for item in pending
                        ],
                        "char_start": pending[0]["line_start"],
                        "char_end": pending[-1]["line_end"],
                    }
                )
                pending = []
        if pending:
            char_groups.append(
                {
                    "label": "history_tail",
                    "message_indices": [
                        item["message_index"] for item in pending
                    ],
                    "char_start": pending[0]["line_start"],
                    "char_end": pending[-1]["line_end"],
                }
            )

    groups: list[dict[str, Any]] = []
    block_starts = [int(block["p0"]) for block in blocks]
    block_ends = [
        int(block["p0"]) + int(block["nt"]) for block in blocks
    ]
    for group in char_groups:
        token_start, token_end = token_positions_for_chars(
            offset_starts,
            offset_ends,
            prompt_start + int(group["char_start"]),
            prompt_start + int(group["char_end"]),
        )
        block_left = bisect.bisect_right(block_ends, token_start)
        block_right = bisect.bisect_left(block_starts, token_end)
        group_blocks = blocks[block_left:block_right]
        if not group_blocks:
            continue
        groups.append(
            {
                **group,
                "token_start": token_start,
                "token_end": token_end,
                "block_ids": sorted({int(block["b"]) for block in group_blocks}),
                "score": max(float(block.get("rs") or 0.0) for block in group_blocks),
            }
        )
    return groups


def main() -> None:
    args = parse_args()
    if not runner.health_ok(args.api_base):
        raise RuntimeError(f"dense server is not healthy at {args.api_base}")
    snapshots = load_dump(args.kvmem_dump)
    if len(snapshots) != 1:
        raise RuntimeError("group-max replay requires exactly one dump snapshot")
    snapshot = snapshots[0]
    meta = snapshot["meta"]
    sid = str(meta["trace_tag"])
    samples = {
        str(row.get("stable_sample_id")): row
        for row in runner.read_jsonl(args.dataset)
    }
    if sid not in samples:
        raise RuntimeError(f"dataset is missing {sid}")
    sample = samples[sid]
    canonical = runner.load_canonical_module(args.benchmark_repo)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    prompt = canonical.full_context_prompt(sample)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_start = rendered.find(prompt)
    if prompt_start < 0 or rendered.find(prompt, prompt_start + 1) >= 0:
        raise RuntimeError("canonical prompt embedding is ambiguous")
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    rendered_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    offset_starts = [start for start, _ in offsets]
    offset_ends = [end for _, end in offsets]
    if len(rendered_ids) != int(meta["prompt_tokens"]):
        raise RuntimeError(
            f"prompt token parity failed: local={len(rendered_ids)} "
            f"dump={meta['prompt_tokens']}"
        )
    blocks = sorted(snapshot["blocks"], key=lambda block: int(block["p0"]))
    block_map = {int(block["b"]): block for block in blocks}
    records = message_records(canonical, sample, prompt)
    groups = build_groups(
        args.mode,
        records,
        prompt_start,
        offset_starts,
        offset_ends,
        blocks,
    )

    # Preserve the production sink and the complete final Question/Answer suffix.
    sink_count = int(meta.get("sink") or 0)
    selected_ids = {int(block["b"]) for block in blocks[:sink_count]}
    suffix_char = prompt.rfind("\n\nQuestion:\n")
    if suffix_char < 0:
        raise RuntimeError("canonical prompt has no final Question suffix")
    suffix_start, suffix_end = token_positions_for_chars(
        offset_starts,
        offset_ends,
        prompt_start + suffix_char,
        len(rendered),
    )
    selected_ids.update(
        int(block["b"])
        for block in blocks
        if int(block["p0"]) < suffix_end
        and int(block["p0"]) + int(block["nt"]) > suffix_start
    )
    pinned_ids = set(selected_ids)
    budget = int(meta["budget_blocks"])
    ranked_groups = sorted(
        groups, key=lambda group: (-float(group["score"]), group["token_start"])
    )
    admitted: list[dict[str, Any]] = []
    skipped_too_large = 0
    for group in ranked_groups:
        new_ids = set(group["block_ids"]) - selected_ids
        if len(selected_ids) + len(new_ids) <= budget:
            selected_ids.update(new_ids)
            admitted.append(group)
        else:
            skipped_too_large += 1
    if len(selected_ids) > budget:
        raise RuntimeError("pinned groups exceed KVMem block budget")

    selected_blocks = [
        block for block in blocks if int(block["b"]) in selected_ids
    ]
    source_ids: list[int] = []
    for block in selected_blocks:
        p0, nt = int(block["p0"]), int(block["nt"])
        source_ids.extend(rendered_ids[p0 : p0 + nt])
    replay_prompt = tokenizer.decode(
        source_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    local_replay_ids = [
        int(value)
        for value in tokenizer.encode(replay_prompt, add_special_tokens=False)
    ]
    server_replay_ids = (
        local_replay_ids
        if args.prepare_only
        else selected_replay.tokenize_server(args, replay_prompt)
    )
    request_max = min(
        32768, max(1, 262144 - len(server_replay_ids) - 16)
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    runner.write_json(
        args.output_root / "selection_manifest.json",
        {
            "schema_version": "agentlongbench_group_max_replay.v1",
            "stable_sample_id": sid,
            "mode": args.mode,
            "pooling": "max_block_retrieval_score",
            "source_blocks": len(blocks),
            "budget_blocks": budget,
            "selected_blocks": len(selected_blocks),
            "unused_budget_blocks": budget - len(selected_blocks),
            "pinned_blocks": len(pinned_ids),
            "candidate_groups": len(groups),
            "admitted_groups": len(admitted),
            "skipped_too_large": skipped_too_large,
            "selected_group_labels": [group["label"] for group in admitted],
            "selected_block_ids": [int(block["b"]) for block in selected_blocks],
            "selected_source_tokens": len(source_ids),
            "selected_source_token_sha256": token_sha256(source_ids),
            "dense_replay_tokens": len(server_replay_ids),
            "dense_replay_token_sha256": token_sha256(server_replay_ids),
            "retokenization_delta": len(server_replay_ids) - len(source_ids),
            "generation_max_tokens": request_max,
        },
    )
    (args.output_root / "replay_prompt.txt").write_text(
        replay_prompt, encoding="utf-8"
    )
    if args.prepare_only:
        print(
            f"[prepared] mode={args.mode} blocks={len(selected_blocks)}/"
            f"{budget} output={args.output_root}",
            flush=True,
        )
        return

    complete_args = argparse.Namespace(
        api_base=args.api_base,
        model=args.model,
        temperature=0.6,
        top_p=0.95,
        thinking_budget=8192,
        seed=20260722,
        timeout_sec=args.timeout_sec,
        max_sample_sec=args.timeout_sec,
    )
    print(
        f"[answer] {sid} mode={args.mode} groups={len(admitted)}/"
        f"{len(groups)} blocks={len(selected_blocks)}/{budget} "
        f"replay={len(server_replay_ids)} max={request_max}",
        flush=True,
    )
    result = selected_replay.complete(
        complete_args, replay_prompt, request_max
    )
    answer = {
        **runner.base_row(
            1,
            1,
            sample,
            f"kvmem_group_{args.mode}_dense_fp8",
            "AgentLongBench-1M-group-max-control",
        ),
        **result,
        "prompt_mode": f"group_max_{args.mode}_dense_text_replay",
        "prompt_tokens": len(server_replay_ids),
        "answered_at": runner.now_iso(),
    }
    evaluated = canonical.evaluate_response(
        sample, answer.get("raw_response") or answer.get("hypothesis") or ""
    )
    baseline = runner.latest_by_id(runner.read_jsonl(args.kvmem_eval)).get(
        sid, {}
    )
    eval_row = {
        **{
            key: answer.get(key)
            for key in (
                "benchmark",
                "method",
                "index",
                "stable_sample_id",
                "question_type",
                "setting",
                "target_length",
                "actual_length",
                "task_type",
                "source_path",
                "prompt_tokens",
                "finish_reason",
            )
        },
        **evaluated,
        "kvmem_baseline_correct": baseline.get("correct"),
        "kvmem_baseline_score": baseline.get("score"),
        "evaluated_at": runner.now_iso(),
    }
    with (args.output_root / "answers.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
    with (args.output_root / "eval.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(
            json.dumps(eval_row, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
    runner.write_json(
        args.output_root / "summary.json",
        {
            "stable_sample_id": sid,
            "mode": args.mode,
            "correct": eval_row.get("correct"),
            "score": eval_row.get("score"),
            "prediction": eval_row.get("pred_answer"),
            "reference_answer": eval_row.get("reference_answer"),
            "selected_blocks": len(selected_blocks),
            "budget_blocks": budget,
        },
    )
    print(
        f"[complete] mode={args.mode} correct={eval_row.get('correct')} "
        f"score={eval_row.get('score')} output={args.output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
