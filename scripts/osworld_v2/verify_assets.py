#!/usr/bin/env python3
"""Build or consume a pinned SHA-256 manifest for all OSWorld V2 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


LFS_POINTER_RE = re.compile(
    rb"\Aversion https://git-lfs\.github\.com/spec/v1\n"
    rb"oid sha256:([0-9a-f]{64})\n"
    rb"size ([0-9]+)\n?\Z"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-checkout", type=Path)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(checkout: Path, expected_commit: str) -> dict:
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise RuntimeError(f"Asset repository commit mismatch: {commit} != {expected_commit}")

    raw_paths = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    names = [item.decode("utf-8") for item in raw_paths.split(b"\0") if item]
    files: dict[str, dict[str, int | str | bool]] = {}
    for name in names:
        path = checkout / name
        content = path.read_bytes()
        pointer = LFS_POINTER_RE.fullmatch(content)
        if pointer:
            digest = pointer.group(1).decode("ascii")
            size = int(pointer.group(2))
            is_lfs = True
        else:
            digest = hashlib.sha256(content).hexdigest()
            size = len(content)
            is_lfs = False
        files[name] = {"sha256": digest, "size": size, "lfs": is_lfs}

    return {
        "schema_version": 1,
        "repository": "xlangai/osworld_v2_assets_gated",
        "revision": "v2026.08.08",
        "commit": commit,
        "file_count": len(files),
        "files": files,
    }


def verify_assets(asset_dir: Path, manifest: dict, expected_commit: str) -> None:
    if manifest.get("commit") != expected_commit:
        raise RuntimeError(
            f"Asset manifest commit mismatch: {manifest.get('commit')} != {expected_commit}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 1061:
        raise RuntimeError("Asset manifest must contain exactly 1061 files")

    actual_names = {
        path.relative_to(asset_dir).as_posix()
        for path in asset_dir.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(asset_dir).parts
    }
    expected_names = set(files)
    if actual_names != expected_names:
        raise RuntimeError(
            f"Asset file set mismatch: missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )

    mismatches = []
    total_bytes = 0
    for index, name in enumerate(sorted(files), start=1):
        metadata = files[name]
        expected_size = metadata.get("size")
        expected_hash = metadata.get("sha256")
        path = asset_dir / name
        actual_size = path.stat().st_size
        total_bytes += actual_size
        if actual_size != expected_size:
            mismatches.append(
                f"{name}: size {actual_size} != {expected_size}"
            )
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            mismatches.append(
                f"{name}: sha256 {actual_hash} != {expected_hash}"
            )
        if index % 100 == 0:
            print(f"asset_hashes_checked={index}/1061", flush=True)
    if mismatches:
        raise RuntimeError("Asset verification failed:\n" + "\n".join(mismatches))
    print(f"OSWorld V2 assets verified: files=1061 bytes={total_bytes}")


def main() -> None:
    args = parse_args()
    if args.git_checkout:
        manifest = build_manifest(args.git_checkout.resolve(), args.expected_commit)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"asset_manifest_written={args.manifest}")
    else:
        with args.manifest.open(encoding="utf-8") as file:
            manifest = json.load(file)
    verify_assets(args.asset_dir.resolve(), manifest, args.expected_commit)


if __name__ == "__main__":
    main()
