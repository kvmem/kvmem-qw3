#!/usr/bin/env python3
"""Seed a new AgentLongBench run from completed rows in compatible runs.

The canonical evaluator can resume within one output root, but it cannot read
completed samples from multiple subset roots.  This utility validates those
rows, rewrites subset-local metadata to the full dataset's index space, and
creates the append-only artifacts that the canonical evaluator will skip.

It intentionally does not create ``run_config.json``.  The evaluator remains
the authority for the destination configuration and writes that file when the
resumed full run starts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--selected-id-source",
        type=Path,
        help=(
            "optional JSONL whose stable_sample_id values define the destination "
            "subset; dataset order remains authoritative"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--method", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def keyed(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = str(row.get("stable_sample_id") or "")
        if not sid:
            raise RuntimeError(f"row without stable_sample_id in {path}")
        if sid in result:
            raise RuntimeError(f"duplicate stable_sample_id {sid} in {path}")
        result[sid] = row
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    tmp.replace(path)


def write_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def rewrite(
    row: dict[str, Any],
    *,
    index: int,
    total: int,
    benchmark_name: str,
    method: str,
    source_root: Path,
) -> dict[str, Any]:
    result = dict(row)
    result["benchmark"] = benchmark_name
    result["method"] = method
    result["index"] = index
    if "total" in result:
        result["total"] = total
    result["resumed_from_output_root"] = str(source_root)
    result["resumed_original_benchmark"] = row.get("benchmark")
    result["resumed_original_method"] = row.get("method")
    return result


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise RuntimeError(
            f"destination must be absent or empty: {args.output_root}"
        )

    samples = read_jsonl(args.dataset)
    if args.selected_id_source is not None:
        selected_rows = read_jsonl(args.selected_id_source)
        selected_ids = [
            str(row.get("stable_sample_id") or "") for row in selected_rows
        ]
        if (
            any(not sid for sid in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)
        ):
            raise RuntimeError(
                "selected ID source contains empty or duplicate stable_sample_id"
            )
        selected_set = set(selected_ids)
        samples = [
            row
            for row in samples
            if str(row.get("stable_sample_id") or "") in selected_set
        ]
        if len(samples) != len(selected_ids):
            found = {str(row.get("stable_sample_id") or "") for row in samples}
            raise RuntimeError(
                "selected ID source does not match dataset: "
                f"missing={sorted(selected_set - found)}"
            )
    sample_ids = [str(row.get("stable_sample_id") or "") for row in samples]
    if any(not sid for sid in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("dataset contains empty or duplicate stable_sample_id")
    index_by_id = {sid: index for index, sid in enumerate(sample_ids, start=1)}
    sample_by_id = {sid: row for sid, row in zip(sample_ids, samples)}

    comparable_config: dict[str, Any] | None = None
    comparable_keys = (
        "dataset",
        "canonical_manifest",
        "model",
        "temperature",
        "top_p",
        "max_tokens",
        "context_window",
        "enable_thinking",
        "kvmem_retrieval_trace_metadata",
        "seed",
        "allow_custom_subset",
    )
    imported: dict[str, dict[str, Any]] = {}
    provenance_sources: list[dict[str, Any]] = []

    for source_root in args.source_root:
        config_path = source_root / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        current = {key: config.get(key) for key in comparable_keys}
        if comparable_config is None:
            comparable_config = current
        elif current != comparable_config:
            raise RuntimeError(
                f"incompatible run configuration in {source_root}: "
                f"{current} != {comparable_config}"
            )
        if Path(str(config.get("dataset"))) != args.dataset:
            raise RuntimeError(
                f"source dataset mismatch in {source_root}: {config.get('dataset')}"
            )

        answers_path = source_root / "answers.jsonl"
        eval_path = source_root / "eval.jsonl"
        manifest_path = source_root / "manifest.jsonl"
        answers = keyed(read_jsonl(answers_path), answers_path)
        evals = keyed(read_jsonl(eval_path), eval_path)
        manifests = keyed(read_jsonl(manifest_path), manifest_path)

        for sid, eval_row in evals.items():
            if sid not in index_by_id:
                raise RuntimeError(f"source row {sid} is absent from destination dataset")
            if sid not in answers or sid not in manifests:
                raise RuntimeError(
                    f"completed source row {sid} lacks answer or manifest data"
                )
            if sid in imported:
                raise RuntimeError(f"completed source row {sid} appears in two roots")
            answer = answers[sid]
            manifest = manifests[sid]
            hashes = {
                str(value)
                for value in (
                    answer.get("prompt_sha256"),
                    eval_row.get("prompt_sha256"),
                    manifest.get("prompt_sha256"),
                    sample_by_id[sid].get("prompt_sha256"),
                )
                if value
            }
            if len(hashes) != 1:
                raise RuntimeError(f"prompt hash mismatch for imported row {sid}: {hashes}")
            imported[sid] = {
                "answer": answer,
                "eval": eval_row,
                "manifest": manifest,
                "source_root": source_root,
            }

        provenance_sources.append(
            {
                "output_root": str(source_root),
                "benchmark": config.get("benchmark"),
                "method": config.get("method"),
                "completed_rows": len(evals),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    total = len(samples)
    ordered_ids = [sid for sid in sample_ids if sid in imported]
    answer_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for sid in ordered_ids:
        item = imported[sid]
        common = {
            "index": index_by_id[sid],
            "total": total,
            "benchmark_name": args.benchmark_name,
            "method": args.method,
            "source_root": item["source_root"],
        }
        answer_rows.append(rewrite(item["answer"], **common))
        eval_rows.append(rewrite(item["eval"], **common))
        manifest_rows.append(rewrite(item["manifest"], **common))
        status_rows.append(
            {
                "time": now_iso(),
                "stable_sample_id": sid,
                "index": index_by_id[sid],
                "status": "imported_completed",
                "source_output_root": str(item["source_root"]),
            }
        )

    write_jsonl(args.output_root / "answers.jsonl", answer_rows)
    write_jsonl(args.output_root / "eval.jsonl", eval_rows)
    write_jsonl(args.output_root / "manifest.jsonl", manifest_rows)
    write_jsonl(args.output_root / "status_events.jsonl", status_rows)
    write_json(
        args.output_root / "resume_seed_provenance.json",
        {
            "dataset": str(args.dataset),
            "selected_id_source": (
                str(args.selected_id_source)
                if args.selected_id_source is not None
                else None
            ),
            "benchmark": args.benchmark_name,
            "method": args.method,
            "total_samples": total,
            "imported_samples": len(ordered_ids),
            "remaining_samples": total - len(ordered_ids),
            "imported_ids_in_dataset_order": ordered_ids,
            "sources": provenance_sources,
            "created_at": now_iso(),
        },
    )
    print(
        f"seeded {len(ordered_ids)}/{total} completed samples into "
        f"{args.output_root}; remaining={total - len(ordered_ids)}"
    )


if __name__ == "__main__":
    main()
