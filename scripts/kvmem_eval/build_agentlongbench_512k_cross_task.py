#!/usr/bin/env python3
"""Materialize a balanced AgentLongBench 512K cross-task diagnostic subset.

The public benchmark archive has a 512K source bucket but no 1M/1024K bucket.
This builder selects two samples for each of the eight official task types and
balances the four AgentLongBench settings (ki-c, ki-v, kf-c, kf-v).  It keeps
the original rows intact and uses the same stable-ID and canonical FullContext
prompt construction as the existing long250 evaluation.

Selection is deterministic: for each selected source member, choose the lowest
hash key among the first N rows.  Limiting the candidate prefix avoids decoding
the entire multi-gigabyte expanded archive merely to construct a small
diagnostic set, while remaining reproducible and independent of answer values.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
from typing import Any


DEFAULT_BENCHMARK_REPO = Path("/home/chaidi/AgentLongBench_Motivation")
DEFAULT_ARCHIVE = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/AgentLongBench/benchmark.tar.gz"
)
DEFAULT_OUTPUT = Path(
    "/data/chaidi/kvmem_eval/data/agentlongbench_512k_cross_task_16"
)
DEFAULT_INSPECT = Path("/home/chaidi/qw3/build/qw3-inspect")
DEFAULT_MODEL = Path("/home/chaidi/qw3/models/Qwen3.6-27B-Q8_0.gguf")
SELECTION_SEED = 20260720


# Two settings per task, balanced to four selected samples per setting.
TASK_SOURCES: tuple[tuple[str, str, tuple[str, str]], ...] = (
    ("Count Frequency(Tool)", "tool_response/count_frequency_tool.jsonl", ("ki-c", "kf-v")),
    ("Find Duplicates(Tool)", "tool_response/find_duplicates_tool.jsonl", ("ki-v", "kf-c")),
    ("Find Target Offsets(Tool)", "tool_response/find_target_offsets_tool.jsonl", ("kf-c", "ki-v")),
    ("Count Correctness(Env)", "env_response/count_correctness_env.jsonl", ("kf-v", "ki-c")),
    ("Count Frequency(Env)", "env_response/count_frequency_env.jsonl", ("ki-c", "kf-v")),
    (
        "Find Round with Largest Value(Env)",
        "env_response/find_round_largest_value_env.jsonl",
        ("ki-v", "kf-c"),
    ),
    ("Weighted Summation(Env)", "env_response/weighted_summation_env.jsonl", ("kf-c", "ki-v")),
    ("Intersection", "final_guess/intersection.jsonl", ("kf-v", "ki-c")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced 16-sample AgentLongBench 512K subset"
    )
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-inspect", type=Path, default=DEFAULT_INSPECT)
    parser.add_argument("--tokenizer-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-exact-tokenization",
        action="store_true",
        help="leave prompt token counts for the runtime /tokenize endpoint",
    )
    parser.add_argument(
        "--candidates-per-source",
        type=int,
        default=8,
        help="deterministically select from this many leading rows per source",
    )
    return parser.parse_args()


def load_fullcontext_module(repo: Path) -> Any:
    source = repo / "fullcontext" / "run_agentlongbench_fullcontext_worker.py"
    spec = importlib.util.spec_from_file_location("agentlongbench_fullcontext", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import canonical worker: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_id(source_path: str, raw: dict[str, Any], line_index: int) -> str:
    material = f"AgentLongBench|{source_path}|{raw.get('id', '')}|{line_index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def selection_key(stable_sample_id: str) -> str:
    material = f"{SELECTION_SEED}|AgentLongBench|{stable_sample_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def task_slug(source_tail: str) -> str:
    return Path(source_tail).stem


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    tmp.replace(path)


def exact_token_count(inspect: Path, model: Path, prompt: str) -> int:
    result = subprocess.run(
        [str(inspect), "--tokenize-stdin", str(model)],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"tokenizer failed with exit {result.returncode}: "
            f"{result.stderr.decode('utf-8', errors='replace')[:2000]}"
        )
    encoded = result.stdout.strip()
    if encoded == b"[]":
        return 0
    if not (encoded.startswith(b"[") and encoded.endswith(b"]")):
        raise RuntimeError("qw3-inspect returned an invalid token array")
    return encoded.count(b",") + 1


def main() -> None:
    args = parse_args()
    if args.candidates_per_source <= 0:
        raise RuntimeError("--candidates-per-source must be positive")

    target_meta: dict[str, tuple[str, str, str]] = {}
    ordered_paths: list[str] = []
    for task_type, source_tail, settings in TASK_SOURCES:
        for setting in settings:
            source_path = f"benchmark/{setting}/512k/{source_tail}"
            target_meta[source_path] = (task_type, source_tail, setting)
            ordered_paths.append(source_path)

    selected: dict[str, tuple[str, int, dict[str, Any]]] = {}
    with tarfile.open(args.archive, "r|gz") as archive:
        for member_info in archive:
            source_path = member_info.name
            if source_path not in target_meta:
                continue
            member = archive.extractfile(member_info)
            if member is None:
                raise RuntimeError(f"archive member is not a file: {source_path}")
            best: tuple[str, int, dict[str, Any]] | None = None
            for line_index, line in enumerate(member):
                if line_index >= args.candidates_per_source:
                    break
                raw = json.loads(line)
                sid = stable_id(source_path, raw, line_index)
                key = selection_key(sid)
                if best is None or key < best[0]:
                    best = (key, line_index, raw)
            if best is None:
                raise RuntimeError(f"source contains no candidate rows: {source_path}")
            selected[source_path] = best

    missing = sorted(set(ordered_paths) - set(selected))
    if missing:
        raise RuntimeError(f"archive is missing selected source members: {missing}")

    canonical = load_fullcontext_module(args.benchmark_repo)
    samples: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for index, source_path in enumerate(ordered_paths, start=1):
        key, line_index, raw = selected[source_path]
        task_type, source_tail, setting = target_meta[source_path]
        raw_task = str(raw.get("question_type") or "")
        if raw_task != task_type:
            raise RuntimeError(
                f"task mismatch for {source_path}:{line_index}: {raw_task!r} != {task_type!r}"
            )
        sid = stable_id(source_path, raw, line_index)
        meta: dict[str, Any] = {
            "dataset": "AgentLongBench",
            "benchmark": "AgentLongBench-512K-cross-task-16",
            "stable_sample_id": sid,
            "selection_key": key,
            "selection_seed": SELECTION_SEED,
            "selection_candidate_prefix": args.candidates_per_source,
            "source_repo": "ign1s/AgentLongBench",
            "source_path": source_path,
            "source_line_index": line_index,
            "setting": setting,
            "length_bucket": "512K",
            "target_length": "512K",
            "actual_length": "512K-source-bucket",
            "task_type": task_type,
            "task_slug": task_slug(source_tail),
            "official_id": raw.get("id"),
            "sample_id": raw.get("sample_id"),
            "question_type": raw_task,
            "index": index,
            "total": len(ordered_paths),
        }
        sample = {**meta, "raw": raw}
        prompt = canonical.full_context_prompt(sample)
        meta["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        meta["prompt_chars"] = len(prompt)
        meta["prompt_utf8_bytes"] = len(prompt.encode("utf-8"))
        if not args.skip_exact_tokenization:
            meta["prompt_tokens_exact"] = exact_token_count(
                args.tokenizer_inspect, args.tokenizer_model, prompt
            )
            meta["prompt_tokenizer_exact"] = (
                f"{args.tokenizer_inspect}:QwenTokenizer"
            )
        sample.update(meta)
        manifest.append(meta)
        samples.append(sample)

    ids = [row["stable_sample_id"] for row in manifest]
    if len(set(ids)) != len(ids):
        raise RuntimeError("selection produced duplicate stable IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "samples.jsonl", samples)
    write_jsonl_atomic(args.output_dir / "manifest.jsonl", manifest)
    validation = {
        "source_archive": str(args.archive),
        "source_bucket": "512k",
        "archive_has_1m_bucket": False,
        "samples": len(samples),
        "tasks": {task: 2 for task, _, _ in TASK_SOURCES},
        "settings": {
            setting: sum(row["setting"] == setting for row in manifest)
            for setting in ("ki-c", "ki-v", "kf-c", "kf-v")
        },
        "prompt_chars_min": min(row["prompt_chars"] for row in manifest),
        "prompt_chars_max": max(row["prompt_chars"] for row in manifest),
        "prompt_tokens_exact_min": min(
            row.get("prompt_tokens_exact", 0) for row in manifest
        ),
        "prompt_tokens_exact_max": max(
            row.get("prompt_tokens_exact", 0) for row in manifest
        ),
        "prompt_tokens_runtime": "rechecked by qw3 /tokenize during evaluation",
        "selection_seed": SELECTION_SEED,
        "selection_candidate_prefix": args.candidates_per_source,
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
