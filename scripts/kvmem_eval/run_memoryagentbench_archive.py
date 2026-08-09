#!/usr/bin/env python3
"""Run the complete local MemoryAgentBench through durable KVMem archives.

Each benchmark row is one immutable context with one or more questions. The
context is rendered once, tokenized once, and archived once; every question is
then evaluated as an independent frozen branch. Completed rows are durable and
skipped on restart. Temporary raw-K/V archives prefer tmpfs (host memory) and
fall back to the NVMe path only when their exact-token size will not fit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

import pyarrow.parquet as pq


SPLITS = (
    "Accurate_Retrieval",
    "Conflict_Resolution",
    "Long_Range_Understanding",
    "Test_Time_Learning",
)
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
TOKEN_LINE = re.compile(r"\btokens=(\d+)\s+bytes=(\d+)")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, default=root / "build/qw3")
    p.add_argument(
        "--model", type=Path,
        default=root / "models/Qwen3.6-27B-Q8_0.gguf",
    )
    p.add_argument(
        "--data-dir", type=Path,
        default=Path(
            "/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/"
            "MemoryAgentBench/data"
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tmpfs-root", type=Path, default=Path("/dev/shm"))
    p.add_argument("--ssd-root", type=Path,
                   default=Path("/home/chaidi/kca/memoryagentbench_tmp"))
    p.add_argument("--tmpfs-limit-gib", type=float, default=50.0)
    p.add_argument(
        "--archive-storage", choices=("auto", "cpu-only"), default="auto",
        help=(
            "auto falls back from tmpfs to --ssd-root; cpu-only requires "
            "every temporary archive to fit in RAM-backed tmpfs and never "
            "uses the SSD path"
        ),
    )
    p.add_argument("--split", action="append", choices=SPLITS)
    p.add_argument("--source", action="append")
    p.add_argument("--row", type=int, action="append")
    p.add_argument(
        "--selection-manifest", type=Path,
        help="JSONL rows with exact {split,row} context identities",
    )
    p.add_argument("--max-contexts", type=int)
    p.add_argument("--question-limit", type=int,
                   help="per-context diagnostic limit; omit for full evaluation")
    p.add_argument("--block-tokens", type=int, default=512)
    p.add_argument("--budget", type=int, default=204800)
    p.add_argument("--gen-budget", type=int, default=32768)
    p.add_argument("--sink-tokens", type=int, default=2048)
    p.add_argument("--recent-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-ratio", type=float, default=0.50)
    p.add_argument("--cpu-gb-tmpfs", type=float, default=16.0)
    p.add_argument("--cpu-gb-ssd", type=float, default=64.0)
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument(
        "--prefill-window", choices=("pressure", "semantic_chunk"),
        default="pressure",
    )
    p.add_argument("--prefill-semantic-start-tokens", type=int, default=0)
    p.add_argument("--prefill-semantic-query-tokens", type=int, default=0)
    p.add_argument("--immutable-refresh-tokens", type=int, default=1)
    p.add_argument("--ladder-tokens", type=int, default=262144)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--index-placement", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--index-staging-mb", type=int, default=64)
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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
    raise ValueError(source)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def run_logged(cmd: list[str], log_path: Path,
               env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=env, text=True
        )
    if result.returncode:
        tail = "\n".join(
            log_path.read_text(encoding="utf-8", errors="replace")
            .splitlines()[-100:]
        )
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(cmd)}\n{tail}"
        )


def exact_tokenize(args: argparse.Namespace, prefix: Path,
                   token_file: Path, log: Path) -> int:
    cmd = [
        str(args.binary), "tokenize", "--model", str(args.model),
        "--prompt-file", str(prefix), "--token-output", str(token_file),
    ]
    run_logged(cmd, log)
    match = TOKEN_LINE.search(log.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"missing token count in {log}")
    return int(match.group(1))


def available_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def archive_estimate_bytes(ctx_tokens: int, ladder_tokens: int) -> int:
    # Qwen3.6-27B fp8 archive: raw-K + V = 34,816 bytes/token. Each durable
    # recurrent/hidden ladder snapshot is 156,935,744 bytes.
    payload = ctx_tokens * 34_816
    ladder_points = math.ceil(ctx_tokens / max(1, ladder_tokens))
    return payload + ladder_points * 156_935_744 + 256 * 1024 * 1024


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
        out: list[str] = []
        for item in answer:
            out.extend(flat_answers(item))
        return out
    return [str(answer)]


def simple_metrics(prediction: str, answer: Any, family: str) -> dict[str, Any]:
    answers = flat_answers(answer)
    pn = normalize(prediction)
    exact = max((pn == normalize(a) for a in answers), default=False)
    substring = max((normalize(a) in pn for a in answers), default=False)
    result: dict[str, Any] = {
        "exact_match": bool(exact),
        "substring_exact_match": bool(substring),
    }
    if family == "eventqa":
        result["eventqa_recall"] = int(
            bool(answers) and all(a.lower() in prediction.lower() for a in answers)
        )
    if family in ("infbench", "longmemeval", "recsys"):
        result["official_special_metric_pending"] = True
    return result


def aggregate(out_dir: Path) -> dict[str, Any]:
    rows = []
    for path in sorted((out_dir / "rows").glob("*/row_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_questions = 0
    completed_questions = 0
    for row in rows:
        by_source[row["source"]].append(row)
        total_questions += row["questions_expected"]
        completed_questions += row["questions_completed"]
    summary = {
        "contexts_completed": len(rows),
        "questions_completed": completed_questions,
        "questions_expected_in_completed_contexts": total_questions,
        "by_source": {},
    }
    for source, source_rows in sorted(by_source.items()):
        q = sum(row["questions_completed"] for row in source_rows)
        em_num = sum(row["exact_match_count"] for row in source_rows)
        sub_num = sum(row["substring_exact_match_count"] for row in source_rows)
        summary["by_source"][source] = {
            "contexts": len(source_rows),
            "questions": q,
            "exact_match": em_num / q if q else None,
            "substring_exact_match": sub_num / q if q else None,
        }
    (out_dir / "progress_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.prefill_window != "semantic_chunk" and (
        args.prefill_semantic_start_tokens != 0
        or args.prefill_semantic_query_tokens != 0
    ):
        raise ValueError(
            "--prefill-semantic-* requires --prefill-window semantic_chunk"
        )
    if args.prefill_semantic_start_tokens < 0:
        raise ValueError("--prefill-semantic-start-tokens must be >= 0")
    if args.prefill_semantic_query_tokens < 0:
        raise ValueError("--prefill-semantic-query-tokens must be >= 0")
    if args.immutable_refresh_tokens <= 0:
        raise ValueError("--immutable-refresh-tokens must be > 0")
    args.binary = args.binary.resolve()
    args.model = args.model.resolve()
    if args.selection_manifest is not None:
        args.selection_manifest = args.selection_manifest.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows").mkdir(exist_ok=True)
    if args.archive_storage == "auto":
        args.ssd_root.mkdir(parents=True, exist_ok=True)
    selected_splits = args.split or list(SPLITS)
    selected_pairs: set[tuple[str, int]] | None = None
    if args.selection_manifest is not None:
        selected_pairs = set()
        with args.selection_manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                split = str(item["split"])
                row = int(item["row"])
                if split not in SPLITS or row < 0:
                    raise ValueError(
                        f"invalid selection at line {line_number}: {item}"
                    )
                if (split, row) in selected_pairs:
                    raise ValueError(
                        f"duplicate selection at line {line_number}: {item}"
                    )
                selected_pairs.add((split, row))
    run_id = safe_name(args.out_dir.name)
    work: list[tuple[str, Path, int, str]] = []
    for split in selected_splits:
        parquet = args.data_dir / f"{split}-00000-of-00001.parquet"
        table = pq.read_table(parquet, columns=["metadata"])
        for row_index, metadata_wrap in enumerate(table.to_pylist()):
            metadata = metadata_wrap.get("metadata") or {}
            source = metadata.get("source", "")
            if selected_pairs is not None and (split, row_index) not in selected_pairs:
                continue
            if args.source and source not in args.source:
                continue
            if args.row and row_index not in args.row:
                continue
            work.append((split, parquet, row_index, source))
    if selected_pairs is not None:
        found = {(split, row) for split, _, row, _ in work}
        missing = selected_pairs - found
        if missing:
            raise ValueError(
                "selection manifest contains contexts not found under the "
                f"active split/source filters: {sorted(missing)}"
            )
    if args.max_contexts is not None:
        work = work[: args.max_contexts]

    config = vars(args).copy()
    config.update({
        key: str(value) for key, value in config.items()
        if isinstance(value, Path)
    })
    config["splits"] = selected_splits
    config["contexts_selected"] = len(work)
    config["retrieval"] = "key-direction-adaptive"
    config["adaptive_gain_1to2"] = 0.10
    config["adaptive_gain_2to4"] = 0.06
    config["question_format"] = "qwen-user-continuation"
    config["thinking"] = False
    config["kv_dtype"] = "fp8 archive; FP16 index/query"
    (args.out_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    prepare_script = Path(__file__).with_name(
        "prepare_memoryagentbench_archive_row.py"
    )
    env = os.environ.copy()
    env.update({
        "QW3_Q8_BF16_MAIN": "0",
        "QW3_KVMEM_QUERY_REPLAY": "1",
        "QW3_KVMEM_IMMUTABLE_REFRESH_TOKENS":
            str(args.immutable_refresh_tokens),
        "QW3_KVMEM_DROP_PAGE_CACHE": "0",
    })

    for ordinal, (split, parquet, row_index, source) in enumerate(work, 1):
        row_name = f"{ordinal:03d}_{safe_name(split)}_r{row_index:03d}_{safe_name(source)}"
        row_dir = args.out_dir / "rows" / row_name
        row_summary_path = row_dir / "row_summary.json"
        if row_summary_path.exists():
            print(f"[{ordinal}/{len(work)}] SKIP completed {source} row={row_index}",
                  flush=True)
            continue
        row_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{ordinal}/{len(work)}] PREPARE {source} row={row_index}",
              flush=True)
        if not (row_dir / "prepare_manifest.json").exists():
            run_logged(
                [
                    sys.executable, str(prepare_script),
                    "--parquet", str(parquet), "--row", str(row_index),
                    "--out-dir", str(row_dir),
                ],
                row_dir / "prepare.log",
            )
        prep = json.loads(
            (row_dir / "prepare_manifest.json").read_text(encoding="utf-8")
        )
        qa = json.loads(Path(prep["qa_file"]).read_text(encoding="utf-8"))
        questions_file = Path(prep["questions_file"])
        if args.question_limit is not None:
            if args.question_limit <= 0:
                raise ValueError("--question-limit must be > 0")
            qa = qa[: args.question_limit]
            questions = json.loads(questions_file.read_text(encoding="utf-8"))[
                : args.question_limit
            ]
            questions_file = row_dir / "questions.used.json"
            questions_file.write_text(
                json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        token_file = row_dir / "archive_prefix.tokens.bin"
        token_log = row_dir / "tokenize.log"
        if not token_file.exists() or not token_log.exists():
            token_count = exact_tokenize(
                args, Path(prep["archive_prefix"]), token_file, token_log
            )
        else:
            match = TOKEN_LINE.search(token_log.read_text(encoding="utf-8"))
            if not match:
                raise RuntimeError(f"invalid prior token log: {token_log}")
            token_count = int(match.group(1))
        aligned = math.ceil(token_count / args.prefill_chunk) * args.prefill_chunk
        max_tokens = MAX_TOKENS[source_family(source)]
        # Immutable archive mode always uses the bounded tiered page pool. A
        # global 200K budget is larger than many benchmark contexts; passing it
        # verbatim makes (budget + generation reserve) exceed --ctx, disables
        # the pool, and is rejected by immutable-K configuration. Such a row
        # is semantically full-context, so cap its active budget at the aligned
        # archive length and let sink+recent cover that entire short prefix.
        row_budget = min(args.budget, aligned)
        row_sink_tokens = min(args.sink_tokens, row_budget)
        row_recent_tokens = min(
            args.recent_tokens,
            max(0, row_budget - row_sink_tokens),
        )
        if aligned <= args.budget:
            row_recent_tokens = max(0, row_budget - row_sink_tokens)
        ctx = aligned + max(args.gen_budget, max_tokens) + 4096
        estimate = archive_estimate_bytes(ctx, args.ladder_tokens)
        tmpfs_cap = int(args.tmpfs_limit_gib * (1 << 30))
        tmpfs_free = available_bytes(args.tmpfs_root)
        use_tmpfs = estimate <= tmpfs_cap and estimate <= int(tmpfs_free * 0.92)
        if args.archive_storage == "cpu-only" and not use_tmpfs:
            raise RuntimeError(
                "CPU-only archive does not fit RAM-backed tmpfs: "
                f"estimate={estimate/(1<<30):.2f} GiB "
                f"limit={args.tmpfs_limit_gib:.2f} GiB "
                f"free={tmpfs_free/(1<<30):.2f} GiB"
            )
        archive_parent = args.tmpfs_root if use_tmpfs else args.ssd_root
        archive_dir = archive_parent / f"qw3_mab_{run_id}_{row_name}"
        cpu_gb = args.cpu_gb_tmpfs if use_tmpfs else args.cpu_gb_ssd
        print(
            f"[{ordinal}/{len(work)}] TOKENS raw={token_count} aligned={aligned} "
            f"archive_est={estimate/(1<<30):.2f}GiB tier="
            f"{'RAM(tmpfs)' if use_tmpfs else 'NVMe'}",
            flush=True,
        )
        if args.dry_run:
            continue

        manifest = archive_dir / "manifest.json"
        sealed = False
        if manifest.exists():
            sealed = bool(json.loads(manifest.read_text(encoding="utf-8")).get("sealed"))
        if not sealed:
            # Pressure mode uses a cheap mean-K build-time index because its
            # sink+recent policy never consults semantic scores. SemanticChunk
            # instead needs the same Adaptive index used by the live K64
            # experiment: every complete prefill chunk is scored provisionally
            # and then replayed against its selected historical window.
            build_retrieval = (
                "key-direction-adaptive"
                if args.prefill_window == "semantic_chunk" else "mean-k"
            )
            build_cmd = [
                str(args.binary), "archive", "build",
                "--model", str(args.model),
                "--kvmem-archive", str(archive_dir),
                "--archive-token-input", str(token_file),
                "--archive-pad-final-chunk",
                "--archive-ladder-tokens", str(args.ladder_tokens),
                "--ctx", str(ctx),
                "--kvmem-block-tokens", str(args.block_tokens),
                "--kvmem-budget", str(row_budget),
                "--kvmem-gen-budget", str(args.gen_budget),
                "--kvmem-sink-tokens", str(row_sink_tokens),
                "--kvmem-recent-tokens", str(row_recent_tokens),
                "--kvmem-cpu-gb", str(cpu_gb),
                "--kvmem-gpu-memory-ratio", str(args.gpu_memory_ratio),
                "--kvmem-index-placement", args.index_placement,
                "--kvmem-index-staging-mb", str(args.index_staging_mb),
                "--kvmem-query-conditioned",
                "--kvmem-retrieval-method", build_retrieval,
                "--kvmem-opt-stage-out", "on",
                "--kvmem-opt-stage-in", "on",
                "--kvmem-opt-pack", "on",
                "--prefill-chunk", str(args.prefill_chunk),
            ]
            if args.prefill_window == "semantic_chunk":
                build_cmd += [
                    "--archive-prefill-window", "semantic_chunk",
                    "--archive-prefill-semantic-start-tokens",
                    str(args.prefill_semantic_start_tokens),
                    "--archive-prefill-semantic-query-tokens",
                    str(args.prefill_semantic_query_tokens),
                    "--kvmem-adaptive-gain-1to2", "0.10",
                    "--kvmem-adaptive-gain-2to4", "0.06",
                ]
            print(f"[{ordinal}/{len(work)}] BUILD {archive_dir}", flush=True)
            run_logged(build_cmd, row_dir / "build.log", env)
            if args.prefill_window == "semantic_chunk" and aligned > row_budget:
                build_log_text = (row_dir / "build.log").read_text(
                    encoding="utf-8", errors="replace"
                )
                summaries = re.findall(
                    r"native kvmem semantic-chunk \(mtp\): chunks=(\d+)",
                    build_log_text,
                )
                if not summaries or int(summaries[-1]) <= 0:
                    raise RuntimeError(
                        "semantic-chunk archive build did not report any "
                        f"chunk reselections: {row_dir / 'build.log'}"
                    )
                if re.search(r"\bfallback=(?:1|true)\b", build_log_text):
                    raise RuntimeError(
                        "fallback detected during semantic-chunk archive "
                        f"build: {row_dir / 'build.log'}"
                    )

        raw_results = row_dir / "archive_answers.jsonl"
        query_cmd = [
            str(args.binary), "archive", "query",
            "--model", str(args.model),
            "--kvmem-archive", str(archive_dir),
            "--archive-tokens", str(aligned),
            "--archive-questions-json", str(questions_file),
            "--archive-question-format", "qwen-user-continuation",
            "--archive-results-file", str(raw_results),
            "--ctx", str(ctx),
            "--kvmem-block-tokens", str(args.block_tokens),
            "--kvmem-budget", str(row_budget),
            "--kvmem-gen-budget", str(args.gen_budget),
            "--kvmem-sink-tokens", str(row_sink_tokens),
            "--kvmem-recent-tokens", str(row_recent_tokens),
            "--kvmem-cpu-gb", str(cpu_gb),
            "--kvmem-gpu-memory-ratio", str(args.gpu_memory_ratio),
            "--kvmem-index-placement", args.index_placement,
            "--kvmem-index-staging-mb", str(args.index_staging_mb),
            "--kvmem-query-conditioned",
            "--kvmem-retrieval-method", "key-direction-adaptive",
            "--kvmem-adaptive-gain-1to2", "0.10",
            "--kvmem-adaptive-gain-2to4", "0.06",
            "--kvmem-opt-stage-out", "on",
            "--kvmem-opt-stage-in", "on",
            "--kvmem-opt-pack", "on",
            "--prefill-chunk", str(args.prefill_chunk),
            "--temp", str(args.temperature),
            "--top-p", str(args.top_p),
            "--top-k", str(args.top_k),
            "-n", str(max_tokens),
        ]
        print(
            f"[{ordinal}/{len(work)}] QUERY n={len(qa)} "
            f"max_tokens={max_tokens}", flush=True
        )
        query_env = env.copy()
        # PERF_TRACE emits one compact scorer/reselection audit row per
        # question, including fallback status and selected-block count.  The
        # general KVMEM_TRACE also logs raw-K refresh batches and individual
        # tier transitions; on long rows that debug volume materially changes
        # latency and can produce multi-GiB logs without adding evaluator
        # correctness coverage.
        query_env["QW3_KVMEM_PERF_TRACE"] = "1"
        query_env.pop("QW3_KVMEM_TRACE", None)
        run_logged(query_cmd, row_dir / "query.log", query_env)
        query_log_text = (row_dir / "query.log").read_text(
            encoding="utf-8", errors="replace"
        )
        scorer_fallbacks = [
            int(value) for value in re.findall(
                r"\[kvmem-scorer\].*?\bfallback=(\d+)", query_log_text
            )
        ]
        if any(scorer_fallbacks):
            raise RuntimeError(
                f"retrieval scorer fallback detected in {row_dir / 'query.log'}"
            )
        generated = [
            json.loads(line) for line in raw_results.read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]
        if len(generated) != len(qa):
            raise RuntimeError(
                f"row answer count mismatch: {len(generated)} vs {len(qa)}"
            )
        family = source_family(source)
        enriched = []
        for output, gold in zip(generated, qa):
            if output["question_index"] != gold["question_index"]:
                raise RuntimeError("question index mismatch")
            metrics = simple_metrics(output["answer"], gold["answer"], family)
            enriched.append({
                **output,
                "split": split,
                "source": source,
                "dataset_row": row_index,
                "qa_pair_id": gold["qa_pair_id"],
                "question_type": gold.get("question_type"),
                "question_id": gold.get("question_id"),
                "raw_question": gold["raw_question"],
                "gold_answer": gold["answer"],
                "keypoints": gold.get("keypoints"),
                "metrics": metrics,
            })
        enriched_path = row_dir / "results.jsonl"
        enriched_path.write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in enriched),
            encoding="utf-8",
        )
        row_summary = {
            "split": split,
            "source": source,
            "family": family,
            "dataset_row": row_index,
            "context_chars": prep["context_chars"],
            "archive_tokens_unpadded": token_count,
            "archive_tokens": aligned,
            "effective_kvmem_budget": row_budget,
            "effective_sink_tokens": row_sink_tokens,
            "effective_recent_tokens": row_recent_tokens,
            "archive_storage": "tmpfs" if use_tmpfs else "nvme",
            "archive_estimated_gib": estimate / (1 << 30),
            "questions_expected": len(qa),
            "questions_completed": len(enriched),
            "exact_match_count": sum(
                int(x["metrics"]["exact_match"]) for x in enriched
            ),
            "substring_exact_match_count": sum(
                int(x["metrics"]["substring_exact_match"]) for x in enriched
            ),
            "mean_query_wall_s": statistics.fmean(
                x["wall_s"] for x in enriched
            ),
            "max_generation_tokens": max_tokens,
            "scorer_events": len(scorer_fallbacks),
            "scorer_fallbacks": sum(scorer_fallbacks),
            "results": str(enriched_path.resolve()),
            "official_special_metrics_pending": family in (
                "infbench", "longmemeval", "recsys"
            ),
        }
        row_summary_path.write_text(
            json.dumps(row_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = aggregate(args.out_dir)
        print(
            f"[{ordinal}/{len(work)}] DONE {source} q={len(enriched)} "
            f"EM={row_summary['exact_match_count']}/{len(enriched)}; "
            f"total contexts={summary['contexts_completed']} "
            f"questions={summary['questions_completed']}",
            flush=True,
        )
        if not args.keep_archives and archive_dir.exists():
            shutil.rmtree(archive_dir)

    summary = aggregate(args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
