#!/usr/bin/env python3
"""Materialize the frozen ten-sample LongMemEval-M diagnostic set.

The sample JSON lines are copied byte-for-byte from longmemeval_m_102.jsonl.
The companion manifest preserves the original zero-based indices and hashes,
while prior_results.jsonl puts the three existing experiment outcomes side by
side for convenient manual checking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE = Path("/data/chaidi/kvmem_eval/data/longmemeval_m_102.jsonl")
OUTPUT_DIR = Path(
    "/data/chaidi/kvmem_eval/data/longmemeval_m_error10_20260720"
)
BASELINE = Path(
    "/data/chaidi/kvmem_eval/results/"
    "m102_2m_k224k_g32k_bt32_meank_r051_20260719_eval_20260719_063739.jsonl"
)
QUERY_REPLAY = Path(
    "/data/chaidi/kvmem_eval/results/"
    "longmemeval_m_k224k_query_replay10_20260720_eval_20260720_054323.jsonl"
)
DENSE_REPLAY = Path(
    "/data/chaidi/kvmem_eval/results/"
    "longmemeval_m_k224k_selected_text_control10_20260720/results.jsonl"
)
REPLAY_AUDIT = Path(
    "/data/chaidi/kvmem_eval/results/"
    "longmemeval_m_k224k_selected_text_control10_20260720/replay_audit.jsonl"
)

SOURCE_INDICES = (4, 6, 20, 27, 34, 36, 51, 60, 68, 86)
EXPECTED_IDS = (
    "19b5f2b3",
    "1faac195",
    "18dcd5a5",
    "3249768e",
    "06878be2",
    "07b6f563",
    "00ca467f",
    "129d1232",
    "08f4fc43",
    "031748ae",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in read_jsonl(path)}


def write_bytes_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    write_bytes_atomic(path, payload)


def main() -> None:
    wanted = set(SOURCE_INDICES)
    selected_by_index: dict[int, bytes] = {}
    source_digest = hashlib.sha256()
    source_rows = 0
    with SOURCE.open("rb") as handle:
        for source_rows, line in enumerate(handle, start=1):
            source_digest.update(line)
            index = source_rows - 1
            if index in wanted:
                selected_by_index[index] = line
    if source_rows != 102:
        raise RuntimeError(f"expected 102 source rows, found {source_rows}")
    if set(selected_by_index) != wanted:
        raise RuntimeError("one or more requested source indices were not found")

    selected_lines = [selected_by_index[index] for index in SOURCE_INDICES]
    samples = [json.loads(line) for line in selected_lines]
    actual_ids = tuple(str(row.get("question_id")) for row in samples)
    if actual_ids != EXPECTED_IDS:
        raise RuntimeError(
            f"source ID mismatch: actual={actual_ids} expected={EXPECTED_IDS}"
        )

    baseline = by_id(BASELINE)
    query_replay = by_id(QUERY_REPLAY)
    dense_replay = by_id(DENSE_REPLAY)
    audit = by_id(REPLAY_AUDIT)

    manifest_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for subset_index, (source_index, line, sample) in enumerate(
        zip(SOURCE_INDICES, selected_lines, samples)
    ):
        question_id = str(sample["question_id"])
        if question_id not in baseline or question_id not in query_replay:
            raise RuntimeError(f"missing prior KVMem result for {question_id}")
        if question_id not in dense_replay or question_id not in audit:
            raise RuntimeError(f"missing prior dense replay result for {question_id}")
        manifest_rows.append(
            {
                "subset_index": subset_index,
                "source_index": source_index,
                "question_id": question_id,
                "question_type": sample.get("question_type"),
                "question": sample.get("question"),
                "gold": sample.get("answer"),
                "question_date": sample.get("question_date"),
                "source_line_sha256": sha256_bytes(line.rstrip(b"\r\n")),
                "original_prompt_tokens": audit[question_id].get(
                    "original_prompt_tokens"
                ),
            }
        )
        comparison_rows.append(
            {
                "subset_index": subset_index,
                "source_index": source_index,
                "question_id": question_id,
                "question_type": sample.get("question_type"),
                "question": sample.get("question"),
                "gold": sample.get("answer"),
                "original_kvmem": {
                    "correct": baseline[question_id].get("correct"),
                    "answer": baseline[question_id].get("answer"),
                },
                "query_replay": {
                    "correct": query_replay[question_id].get("correct"),
                    "answer": query_replay[question_id].get("answer"),
                },
                "selected_text_dense_replay": {
                    "correct": dense_replay[question_id].get("replay_correct"),
                    "answer": dense_replay[question_id].get("replay_answer"),
                    "selected_block_count": dense_replay[question_id].get(
                        "selected_block_count"
                    ),
                    "selected_token_count": dense_replay[question_id].get(
                        "selected_token_count"
                    ),
                },
            }
        )

    dataset_payload = b"".join(
        line if line.endswith((b"\n", b"\r")) else line + b"\n"
        for line in selected_lines
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = OUTPUT_DIR / "samples.jsonl"
    write_bytes_atomic(dataset_path, dataset_payload)
    write_jsonl_atomic(OUTPUT_DIR / "manifest.jsonl", manifest_rows)
    write_jsonl_atomic(OUTPUT_DIR / "prior_results.jsonl", comparison_rows)
    write_json_atomic(
        OUTPUT_DIR / "validation.json",
        {
            "schema_version": "longmemeval_m_error10.v1",
            "source": str(SOURCE),
            "source_sha256": source_digest.hexdigest(),
            "source_indices_zero_based": list(SOURCE_INDICES),
            "samples": len(samples),
            "question_ids": list(actual_ids),
            "dataset": str(dataset_path),
            "dataset_bytes": len(dataset_payload),
            "dataset_sha256": sha256_bytes(dataset_payload),
            "byte_exact_source_lines": True,
        },
    )
    readme = f"""# LongMemEval-M frozen error-10 subset

This directory contains ten rows copied byte-for-byte from `{SOURCE}`.

- `samples.jsonl`: standalone input dataset in the original schema
- `manifest.jsonl`: new subset index, original source index, ID, gold, and hashes
- `prior_results.jsonl`: original KVMem, query-replay, and dense-replay outputs
- `validation.json`: source/output integrity information

Original zero-based indices: {','.join(str(index) for index in SOURCE_INDICES)}

Use `--use-all` when passing `samples.jsonl` directly to `run_eval.py`; its local
indices are now 0 through 9.  The original indices remain in the manifest.

The matching original-KVMem launcher is:
`/home/chaidi/qw3/scripts/kvmem_eval/run_longmemeval_m_error10_original.sh`.
It uses the same 224K KVMem budget, 32K generation budget, block size 32,
mean-k retrieval, and generation parameters as the original LongMemEval-M run.
"""
    write_bytes_atomic(OUTPUT_DIR / "README.md", readme.encode("utf-8"))
    print(json.dumps(json.loads((OUTPUT_DIR / "validation.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
