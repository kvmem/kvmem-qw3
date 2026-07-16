#!/usr/bin/env python3
"""Rebuild the canonical AgentLongBench long-context 250-sample subset.

The benchmark repository intentionally does not commit the dataset body.  Its
archived FullContext manifest is nevertheless sufficient to recover the exact
rows from AgentLongBench's benchmark.tar.gz.  This script materializes those
rows in canonical order and verifies both stable IDs and the exact prompt hash
used by the archived four-method comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
from typing import Any


DEFAULT_BENCHMARK_REPO = Path("/home/chaidi/AgentLongBench_Motivation")
DEFAULT_ARCHIVE = Path(
    "/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/AgentLongBench/benchmark.tar.gz"
)
DEFAULT_OUTPUT = Path("/data/chaidi/kvmem_eval/data/agentlongbench_250")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and verify the canonical AgentLongBench long 250 subset"
    )
    parser.add_argument("--benchmark-repo", type=Path, default=DEFAULT_BENCHMARK_REPO)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    source_manifest = args.benchmark_repo / "outputs" / "fullcontext" / "manifest.jsonl"
    manifest = read_jsonl(source_manifest)
    if len(manifest) != 250:
        raise RuntimeError(f"expected 250 manifest rows, found {len(manifest)}")
    ids = [str(row["stable_sample_id"]) for row in manifest]
    if len(set(ids)) != 250:
        raise RuntimeError("canonical manifest contains duplicate stable_sample_id values")

    wanted: dict[str, dict[int, int]] = {}
    for canonical_index, row in enumerate(manifest):
        wanted.setdefault(str(row["source_path"]), {})[
            int(row["source_line_index"])
        ] = canonical_index

    raw_rows: list[dict[str, Any] | None] = [None] * len(manifest)
    # Stream through the gzip member once.  Random seeks in a .tar.gz restart
    # decompression and make this reconstruction many times slower.
    found_paths: set[str] = set()
    with tarfile.open(args.archive, "r|gz") as archive:
        for member_info in archive:
            source_path = member_info.name
            if source_path not in wanted:
                continue
            member = archive.extractfile(member_info)
            if member is None:
                raise RuntimeError(f"archive member is not a file: {source_path}")
            indices = wanted[source_path]
            remaining = set(indices)
            for line_index, line in enumerate(member):
                if line_index not in remaining:
                    continue
                raw_rows[indices[line_index]] = json.loads(line)
                remaining.remove(line_index)
                if not remaining:
                    break
            if remaining:
                raise RuntimeError(
                    f"missing source lines in {source_path}: {sorted(remaining)}"
                )
            found_paths.add(source_path)
    missing_paths = sorted(set(wanted) - found_paths)
    if missing_paths:
        raise RuntimeError(f"missing archive members: {missing_paths}")

    canonical = load_fullcontext_module(args.benchmark_repo)
    samples: list[dict[str, Any]] = []
    prompt_hash_matches = 0
    for index, (meta, raw) in enumerate(zip(manifest, raw_rows), start=1):
        if raw is None:
            raise RuntimeError(f"sample {index} was not reconstructed")
        got_id = stable_id(
            str(meta["source_path"]), raw, int(meta["source_line_index"])
        )
        if got_id != meta["stable_sample_id"]:
            raise RuntimeError(
                f"stable ID mismatch at sample {index}: {got_id} != "
                f"{meta['stable_sample_id']}"
            )
        sample = {**meta, "raw": raw}
        prompt = canonical.full_context_prompt(sample)
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_hash != meta["prompt_sha256"]:
            raise RuntimeError(
                f"prompt hash mismatch at sample {index} {got_id}: "
                f"{prompt_hash} != {meta['prompt_sha256']}"
            )
        prompt_hash_matches += 1
        samples.append(sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "samples.jsonl", samples)
    write_jsonl_atomic(args.output_dir / "manifest.jsonl", manifest)
    summary = {
        "source_archive": str(args.archive),
        "source_manifest": str(source_manifest),
        "samples": len(samples),
        "unique_stable_sample_ids": len(set(ids)),
        "prompt_sha256_matches": prompt_hash_matches,
        "first_stable_sample_id": ids[0],
        "last_stable_sample_id": ids[-1],
    }
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
