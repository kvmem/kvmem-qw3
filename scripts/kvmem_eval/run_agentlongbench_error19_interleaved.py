#!/usr/bin/env python3
"""Run error-19 diagnostics as a resumable per-sample pipeline.

For each stable ID, in order:

1. capture the final KVMem selection (or reuse an already captured snapshot);
2. extract that one immutable snapshot and calculate task-grounded coverage;
3. dense-prefill the text decoded from the exact same selected source tokens;
4. for the first few samples, replay message-max and round-bundle-max
   re-selections over the same frozen per-block scores;
5. grade immediately and update combined progress artifacts.

The KVMem and dense servers are expected to remain resident on separate ports.
Only one request is issued at a time, so their model weights coexist in memory
without competing for GPU compute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import run_agentlongbench_kvmem as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dump-file", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--kvmem-eval", type=Path, required=True)
    parser.add_argument("--kvmem-api", default="http://127.0.0.1:8088/v1")
    parser.add_argument("--dense-api", default="http://127.0.0.1:8087/v1")
    parser.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    parser.add_argument(
        "--benchmark-repo", type=Path, default=runner.DEFAULT_BENCHMARK_REPO
    )
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument(
        "--group-max-first",
        type=int,
        default=3,
        help="Run both group-max retrieval controls for the first N samples.",
    )
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(ids) != 19 or len(set(ids)) != 19:
        raise RuntimeError(f"{path} must contain the exact 19 unique IDs")
    return ids


def dump_snapshots(path: Path) -> dict[str, list[str]]:
    snapshots: dict[str, list[str]] = {}
    current_tag: str | None = None
    if not path.exists():
        return snapshots
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid dump JSON {path}:{line_number}: {exc}"
                ) from exc
            if row.get("type") == "meta":
                current_tag = str(row.get("trace_tag") or "")
                if not current_tag or current_tag in snapshots:
                    raise RuntimeError(
                        f"missing/duplicate dump trace tag {current_tag!r}"
                    )
                snapshots[current_tag] = [line]
            elif current_tag is None:
                raise RuntimeError("dump begins with a block row")
            else:
                snapshots[current_tag].append(line)
    for sid, lines in snapshots.items():
        expected = int(json.loads(lines[0])["block_count"]) + 1
        if len(lines) != expected:
            raise RuntimeError(
                f"incomplete dump snapshot for {sid}: rows={len(lines)} "
                f"expected={expected}"
            )
    return snapshots


def capture_payload(
    model: str,
    prompt: str,
    query_span: tuple[int, int],
    context_span: tuple[int, int],
    sid: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 1,
        "enable_thinking": True,
        "thinking_budget": 8192,
        "seed": 20260722,
        "stream": False,
        "kvmem_query_span": {
            "message_index": 0,
            "content_start": query_span[0],
            "content_end": query_span[1],
        },
        "kvmem_context_span": {
            "message_index": 0,
            "content_start": context_span[0],
            "content_end": context_span[1],
        },
        "kvmem_trace_tag": sid,
    }


def run_checked(command: list[str]) -> None:
    print("[exec] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def latest_one(path: Path) -> dict[str, Any]:
    rows = runner.read_jsonl(path)
    if len(rows) != 1:
        raise RuntimeError(f"{path} must contain exactly one row, got {len(rows)}")
    return rows[0]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    tmp.replace(path)


def consolidate(
    args: argparse.Namespace,
    ordered_ids: list[str],
    sample_map: dict[str, dict[str, Any]],
) -> None:
    answers: list[dict[str, Any]] = []
    evals: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for sid in ordered_ids:
        sample_root = args.replay_root / "per_sample" / sid
        if (sample_root / "selected_text/eval.jsonl").exists():
            answers.append(latest_one(sample_root / "selected_text/answers.jsonl"))
            evals.append(latest_one(sample_root / "selected_text/eval.jsonl"))
        if (sample_root / "coverage/per_sample_gold_coverage.jsonl").exists():
            coverage.append(
                latest_one(
                    sample_root / "coverage/per_sample_gold_coverage.jsonl"
                )
            )
    atomic_jsonl(args.replay_root / "answers.jsonl", answers)
    atomic_jsonl(args.replay_root / "eval.jsonl", evals)
    atomic_jsonl(
        args.capture_root / "gold_coverage/per_sample_gold_coverage.jsonl",
        coverage,
    )
    completed_ids = [str(row["stable_sample_id"]) for row in evals]
    group_progress: dict[str, dict[str, int]] = {}
    for mode in ("message-max", "round-bundle-max"):
        summaries = []
        for sid in ordered_ids[: args.group_max_first]:
            path = (
                args.replay_root
                / "per_sample"
                / sid
                / f"group_{mode}"
                / "summary.json"
            )
            if path.exists():
                summaries.append(json.loads(path.read_text(encoding="utf-8")))
        group_progress[mode] = {
            "completed": len(summaries),
            "correct": sum(row.get("correct") is True for row in summaries),
        }
    runner.write_json(
        args.replay_root / "interleaved_progress.json",
        {
            "schema_version": "agentlongbench_error19_interleaved_progress.v1",
            "total": len(ordered_ids),
            "captured": len(dump_snapshots(args.dump_file)),
            "coverage_analyzed": len(coverage),
            "replayed": len(evals),
            "selected_text_correct": sum(
                row.get("correct") is True for row in evals
            ),
            "group_max_first": args.group_max_first,
            "group_max": group_progress,
            "completed_ids": completed_ids,
            "updated_at": runner.now_iso(),
        },
    )
    if len(evals) == len(ordered_ids):
        selected_samples = [sample_map[sid] for sid in ordered_ids]
        runner.write_final(
            args.replay_root,
            selected_samples,
            "kvmem_k224k_exact_selected_tokens_dense_text_fp8",
            "AgentLongBench-1M-DS50-error19-selected-text",
        )


def main() -> None:
    args = parse_args()
    ids = read_ids(args.ids_file)
    samples = runner.read_jsonl(args.dataset)
    sample_map = {str(row.get("stable_sample_id")): row for row in samples}
    missing = [sid for sid in ids if sid not in sample_map]
    if missing:
        raise RuntimeError(f"dataset missing selected ID {missing[0]}")
    if not runner.health_ok(args.kvmem_api):
        raise RuntimeError(f"KVMem server is not healthy at {args.kvmem_api}")
    if not runner.health_ok(args.dense_api):
        raise RuntimeError(f"dense server is not healthy at {args.dense_api}")
    canonical = runner.load_canonical_module(args.benchmark_repo)
    manifest_path = args.capture_root / "capture_manifest.jsonl"
    args.capture_root.mkdir(parents=True, exist_ok=True)
    args.replay_root.mkdir(parents=True, exist_ok=True)
    existing = dump_snapshots(args.dump_file)
    existing_order = list(existing)
    if existing_order != ids[: len(existing_order)]:
        raise RuntimeError(
            "existing dump order is not an exact prefix of the frozen 19 IDs"
        )
    manifest_ids = {
        str(row.get("stable_sample_id"))
        for row in runner.read_jsonl(manifest_path)
    }
    python = sys.executable
    script_root = Path(__file__).resolve().parent

    for index, sid in enumerate(ids, start=1):
        sample = sample_map[sid]
        snapshots = dump_snapshots(args.dump_file)
        if sid not in snapshots:
            prompt = canonical.full_context_prompt(sample)
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            query_span = runner.query_byte_span(sample, prompt)
            context_span = runner.history_byte_span(canonical, sample, prompt)
            prompt_tokens = runner.tokenize_count(
                args.kvmem_api, prompt, args.timeout_sec
            )
            print(
                f"[capture] {index}/19 {sid} prompt={prompt_tokens}",
                flush=True,
            )
            started = time.perf_counter()
            response = runner.post_json(
                args.kvmem_api.rstrip("/") + "/chat/completions",
                capture_payload(
                    args.model, prompt, query_span, context_span, sid
                ),
                args.timeout_sec,
            )
            elapsed = time.perf_counter() - started
            snapshots = dump_snapshots(args.dump_file)
            if sid not in snapshots:
                raise RuntimeError(f"capture completed without dump for {sid}")
            if sid not in manifest_ids:
                choices = response.get("choices") or []
                runner.append_jsonl(
                    manifest_path,
                    {
                        "schema_version":
                            "agentlongbench_selected_kvmem_capture.v1",
                        "stable_sample_id": sid,
                        "delivery_index": index,
                        "prompt_sha256": prompt_hash,
                        "prompt_tokens_unframed": prompt_tokens,
                        "kvmem_query_span_content_bytes": list(query_span),
                        "kvmem_context_span_content_bytes": list(context_span),
                        "elapsed_sec": elapsed,
                        "finish_reason":
                            choices[0].get("finish_reason")
                            if choices
                            else None,
                        "captured_at": runner.now_iso(),
                    },
                )
                manifest_ids.add(sid)
        else:
            print(f"[capture-skip] {index}/19 {sid}", flush=True)

        sample_root = args.replay_root / "per_sample" / sid
        sample_dump = sample_root / "kvmem_retrieval_dump.jsonl"
        sample_dump.parent.mkdir(parents=True, exist_ok=True)
        sample_dump.write_text("".join(snapshots[sid]), encoding="utf-8")

        coverage_root = sample_root / "coverage"
        if not (coverage_root / "gold_coverage_summary.json").exists():
            run_checked(
                [
                    python,
                    str(
                        script_root
                        / "analyze_agentlongbench_gold_block_coverage.py"
                    ),
                    "--dataset",
                    str(args.dataset),
                    "--kvmem-dump",
                    str(sample_dump),
                    "--output-root",
                    str(coverage_root),
                    "--benchmark-repo",
                    str(args.benchmark_repo),
                ]
            )
        else:
            print(f"[coverage-skip] {index}/19 {sid}", flush=True)

        selected_root = sample_root / "selected_text"
        validation_path = selected_root / "validation_report.json"
        replay_done = False
        if validation_path.exists():
            report = json.loads(validation_path.read_text(encoding="utf-8"))
            replay_done = bool(
                report.get("passed")
                and report.get("answers_unique") == 1
                and report.get("eval_unique") == 1
            )
        if not replay_done:
            run_checked(
                [
                    python,
                    str(script_root / "run_agentlongbench_selected_replay.py"),
                    "--dataset",
                    str(args.dataset),
                    "--benchmark-repo",
                    str(args.benchmark_repo),
                    "--kvmem-dump",
                    str(sample_dump),
                    "--kvmem-eval",
                    str(args.kvmem_eval),
                    "--output-root",
                    str(selected_root),
                    "--api-base",
                    args.dense_api,
                    "--model",
                    args.model,
                    "--benchmark-name",
                    "AgentLongBench-1M-DS50-error19-selected-text",
                    "--method",
                    "kvmem_k224k_exact_selected_tokens_dense_text_fp8",
                    "--temperature",
                    "0.6",
                    "--top-p",
                    "0.95",
                    "--max-tokens",
                    "32768",
                    "--thinking-budget",
                    "8192",
                    "--context-window",
                    "262144",
                    "--context-safety-margin",
                    "16",
                    "--timeout-sec",
                    str(args.timeout_sec),
                    "--max-sample-sec",
                    str(args.timeout_sec),
                    "--attempts",
                    "3",
                    "--seed",
                    "20260722",
                ]
            )
        else:
            print(f"[replay-skip] {index}/19 {sid}", flush=True)

        if index <= args.group_max_first:
            for mode in ("message-max", "round-bundle-max"):
                group_root = sample_root / f"group_{mode}"
                summary_path = group_root / "summary.json"
                if not summary_path.exists():
                    run_checked(
                        [
                            python,
                            str(
                                script_root
                                / "run_agentlongbench_group_max_replay.py"
                            ),
                            "--dataset",
                            str(args.dataset),
                            "--benchmark-repo",
                            str(args.benchmark_repo),
                            "--kvmem-dump",
                            str(sample_dump),
                            "--kvmem-eval",
                            str(args.kvmem_eval),
                            "--output-root",
                            str(group_root),
                            "--mode",
                            mode,
                            "--api-base",
                            args.dense_api,
                            "--model",
                            args.model,
                            "--timeout-sec",
                            str(args.timeout_sec),
                        ]
                    )
                else:
                    print(
                        f"[group-skip] {index}/19 {sid} mode={mode}",
                        flush=True,
                    )
        consolidate(args, ids, sample_map)
        eval_row = latest_one(selected_root / "eval.jsonl")
        group_status = []
        if index <= args.group_max_first:
            for mode in ("message-max", "round-bundle-max"):
                summary = json.loads(
                    (
                        sample_root
                        / f"group_{mode}"
                        / "summary.json"
                    ).read_text(encoding="utf-8")
                )
                group_status.append(f"{mode}={summary.get('correct')}")
        print(
            f"[sample-complete] {index}/19 {sid} "
            f"selected_text_correct={eval_row.get('correct')} "
            f"score={eval_row.get('score')} "
            + " ".join(group_status),
            flush=True,
        )

    print(f"[complete] interleaved results={args.replay_root}", flush=True)


if __name__ == "__main__":
    main()
