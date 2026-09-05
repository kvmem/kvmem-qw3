#!/usr/bin/env python3
"""Verify all downloaded OSWorld-V2 task classes against the pinned manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
TASK_RE = re.compile(r"(?:^|/)(task_[^/]+\.py)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = SHA256_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


def task_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = TASK_RE.search(value)
    return match.group(1) if match else None


def first_value(values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def collect_hashes(node: Any, result: dict[str, str]) -> None:
    """Accept map and record-list manifests while rejecting conflicting rows."""
    if isinstance(node, dict):
        for key, value in node.items():
            name = task_name(key)
            digest = normalize_hash(value)
            if name and isinstance(value, dict):
                digest = first_value(
                    normalize_hash(value.get(field))
                    for field in ("sha256", "hash", "digest")
                    if field in value
                )
            if name and digest:
                add_hash(result, name, digest)

        name = first_value(
            (
                task_name(node.get(key))
                for key in ("path", "file", "filename", "name", "task")
                if key in node
            )
        )
        digest = first_value(
            (
                normalize_hash(node.get(key))
                for key in ("sha256", "hash", "digest")
                if key in node
            )
        )
        if name and digest:
            add_hash(result, name, digest)

        for value in node.values():
            collect_hashes(value, result)
    elif isinstance(node, list):
        for value in node:
            collect_hashes(value, result)


def add_hash(result: dict[str, str], name: str, digest: str) -> None:
    previous = result.get(name)
    if previous is not None and previous != digest:
        raise ValueError(f"Conflicting hashes for {name}: {previous} != {digest}")
    result[name] = digest


def main() -> None:
    args = parse_args()
    expected_manifest_hash = normalize_hash(args.expected_manifest_sha256)
    if expected_manifest_hash is None:
        raise ValueError("--expected-manifest-sha256 is not a SHA-256 digest")

    actual_manifest_hash = file_sha256(args.manifest)
    if actual_manifest_hash != expected_manifest_hash:
        raise RuntimeError(
            f"Task manifest SHA-256 mismatch: {actual_manifest_hash} "
            f"!= {expected_manifest_hash}"
        )

    with args.manifest.open(encoding="utf-8") as file:
        manifest = json.load(file)
    task_hashes: dict[str, str] = {}
    collect_hashes(manifest, task_hashes)

    if len(task_hashes) != args.expected_task_count:
        raise RuntimeError(
            f"Task manifest contains {len(task_hashes)} recognizable task hashes; "
            f"expected {args.expected_task_count}"
        )

    local_tasks = {path.name: path for path in args.task_dir.glob("task_*.py")}
    if set(local_tasks) != set(task_hashes):
        missing = sorted(set(task_hashes) - set(local_tasks))
        unexpected = sorted(set(local_tasks) - set(task_hashes))
        raise RuntimeError(
            f"Task file set mismatch: missing={missing}, unexpected={unexpected}"
        )

    mismatches = {}
    for name, path in sorted(local_tasks.items()):
        actual_hash = file_sha256(path)
        if actual_hash != task_hashes[name]:
            mismatches[name] = (actual_hash, task_hashes[name])
    if mismatches:
        raise RuntimeError(f"Task SHA-256 mismatches: {mismatches}")

    print(
        f"OSWorld V2 task hashes verified: tasks={len(local_tasks)}, "
        f"manifest_sha256={actual_manifest_hash}"
    )


if __name__ == "__main__":
    main()
