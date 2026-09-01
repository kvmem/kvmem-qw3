#!/usr/bin/env python3
"""Create and verify the frozen 20-task stratified DeepSWE sample.

The selector deliberately depends only on the audited public trial table and
Python's documented ``random.Random`` generator.  The generated manifest also
stores the exact task IDs and their digest, so a future runtime/library change
cannot silently alter the benchmark sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO / "benchmark/opencode_swebench/deepswe_v1_1_trials.json"
DEFAULT_OUTPUT = Path(__file__).with_name("stratified20_seed20260831.json")
DEFAULT_SEED = 20_260_831
DEFAULT_QUOTAS = {1: 2, 2: 4, 3: 6, 4: 6, 5: 2}
ANCHORS = ((95.0, 1), (80.0, 2), (60.0, 3), (30.0, 4), (5.0, 5))
ALGORITHM = "pass-rate-strata-contiguous-bins-python-random-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_ids_sha256(task_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(task_ids) + "\n").encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def population(source: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_json(source).get("rows", []):
        task_id = row.get("task_name")
        if task_id and row.get("included_in_score"):
            grouped[str(task_id)].append(row)

    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task_id, rows in grouped.items():
        pass_rate = 100.0 * sum(bool(row.get("passed")) for row in rows) / len(rows)
        level = min(ANCHORS, key=lambda anchor: abs(pass_rate - anchor[0]))[1]
        result[level].append(
            {
                "task_id": task_id,
                "difficulty_level": level,
                "historical_pass_rate_pct": round(pass_rate, 4),
                "public_trials": len(rows),
            }
        )
    for rows in result.values():
        rows.sort(key=lambda row: (row["historical_pass_rate_pct"], row["task_id"]))
    return dict(result)


def generate(source: Path, seed: int, quotas: dict[int, int]) -> dict[str, Any]:
    levels = population(source)
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for level in sorted(quotas):
        rows = levels.get(level, [])
        quota = quotas[level]
        if quota <= 0 or quota > len(rows):
            raise ValueError(
                f"invalid Level {level} quota {quota} for {len(rows)} tasks"
            )
        for bin_index in range(quota):
            lo = bin_index * len(rows) // quota
            hi = (bin_index + 1) * len(rows) // quota
            population_index = rng.randrange(lo, hi)
            selected = dict(rows[population_index])
            selected.update(
                {
                    "historical_reward": None,
                    "selection_bin": bin_index,
                    "selection_bin_start": lo,
                    "selection_bin_stop": hi,
                    "population_index": population_index,
                }
            )
            tasks.append(selected)

    task_ids = [row["task_id"] for row in tasks]
    counts = Counter(
        row["difficulty_level"] for rows in levels.values() for row in rows
    )
    try:
        source_name = str(source.resolve().relative_to(REPO))
    except ValueError:
        source_name = str(source.resolve())
    return {
        "schema_version": 1,
        "description": (
            "Frozen 20-task DeepSWE sample with all five empirical difficulty "
            "levels represented; selection is independent of local KVMem outcomes."
        ),
        "source": source_name,
        "source_sha256": sha256(source),
        "historical_score": {
            "kind": "public_trial_pass_rate",
            "note": "Per-task rates are selection metadata, not local rewards.",
        },
        "selection": {
            "kind": "stratified_difficulty_sample",
            "algorithm": ALGORITHM,
            "algorithm_detail": (
                "Levels ascend 1..5; tasks sort by (historical pass rate, task ID); "
                "each level is partitioned into contiguous integer bins; one index "
                "per bin is drawn by one shared random.Random(seed) stream."
            ),
            "seed": seed,
            "anchors_pct": {str(level): rate for rate, level in ANCHORS},
            "population_counts": {str(level): counts[level] for level in range(1, 6)},
            "quotas": {str(level): quotas[level] for level in sorted(quotas)},
            "task_ids_sha256": task_ids_sha256(task_ids),
        },
        "tasks": tasks,
    }


def quotas_from_manifest(payload: dict[str, Any]) -> dict[int, int]:
    return {
        int(level): int(count)
        for level, count in payload["selection"]["quotas"].items()
    }


def verify(path: Path) -> None:
    frozen = read_json(path)
    selection = frozen.get("selection") or {}
    if selection.get("algorithm") != ALGORITHM:
        raise RuntimeError(
            f"unsupported selection algorithm: {selection.get('algorithm')!r}"
        )
    source = Path(frozen["source"])
    if not source.is_absolute():
        source = REPO / source
    actual_source_sha = sha256(source)
    if actual_source_sha != frozen.get("source_sha256"):
        raise RuntimeError(
            "difficulty source hash mismatch: "
            f"expected {frozen.get('source_sha256')}, got {actual_source_sha}"
        )
    regenerated = generate(
        source,
        int(selection["seed"]),
        quotas_from_manifest(frozen),
    )
    frozen_ids = [row["task_id"] for row in frozen.get("tasks", [])]
    regenerated_ids = [row["task_id"] for row in regenerated["tasks"]]
    if frozen_ids != regenerated_ids:
        raise RuntimeError(
            "frozen task IDs differ from regenerated IDs:\n"
            + json.dumps(
                {"frozen": frozen_ids, "regenerated": regenerated_ids}, indent=2
            )
        )
    expected_ids_sha = selection.get("task_ids_sha256")
    actual_ids_sha = task_ids_sha256(frozen_ids)
    if actual_ids_sha != expected_ids_sha:
        raise RuntimeError(
            f"task ID hash mismatch: expected {expected_ids_sha}, got {actual_ids_sha}"
        )
    print(
        f"OK: {len(frozen_ids)} tasks, seed={selection['seed']}, "
        f"ids_sha256={actual_ids_sha}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check",
        type=Path,
        help="verify an existing frozen manifest instead of generating one",
    )
    args = parser.parse_args()
    if args.check:
        verify(args.check.resolve())
        return 0
    payload = generate(args.source.resolve(), args.seed, DEFAULT_QUOTAS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
