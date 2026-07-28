#!/usr/bin/env python3
"""Download selected BEAM-10M conversations through hf-mirror.com.

The official dataset is two Parquet shards.  This downloader keeps the shards
under ``_hf_cache`` and exports the selected rows into the JSON directory layout
consumed by ``run_beam_10m.py``.  Proxy environment variables are explicitly
removed before any network client is created.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import requests

try:
    from .beam_dataset import parse_id_list
except ImportError:
    from beam_dataset import parse_id_list  # type: ignore


HF_MIRROR_ROOT = (
    "https://hf-mirror.com/datasets/Mohammadta/BEAM-10M/resolve/"
    "9b2096193fe74e2837e4713e483351e19817773c/data"
)
SHARDS = (
    (
        "10M-00000-of-00002.parquet",
        153844664,
        "31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5",
    ),
    (
        "10M-00001-of-00002.parquet",
        189980875,
        "a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65",
    ),
)
PROXY_VARIABLES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download official BEAM-10M Parquet from hf-mirror.com and "
            "export selected conversations"
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/chaidi/kvmem_eval/data/beam_10m"),
    )
    parser.add_argument(
        "--conversations",
        default="1",
        help="comma-separated ids and inclusive ranges, e.g. 1 or 1-10",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser.parse_args()


def download(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    force: bool,
    timeout: float,
) -> None:
    if destination.is_file() and not force:
        print(f"[beam-download] exists: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        temporary.unlink()
    print(f"[beam-download] GET {url}", flush=True)
    try:
        with session.get(url, stream=True, timeout=(30.0, timeout)) as response:
            response.raise_for_status()
            expected = int(response.headers.get("content-length") or 0)
            written = 0
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    output.write(chunk)
                    written += len(chunk)
                    if expected:
                        print(
                            f"\r[beam-download] {destination.name}: "
                            f"{written / 2**20:.1f}/"
                            f"{expected / 2**20:.1f} MiB",
                            end="",
                            flush=True,
                        )
            if expected:
                print()
            if expected and written != expected:
                raise RuntimeError(
                    f"short download for {destination}: "
                    f"{written} != {expected}"
                )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def normalize_chat(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("BEAM-10M chat column is not a list")
    result: list[dict[str, Any]] = []
    for index, plan in enumerate(value):
        if not isinstance(plan, dict):
            raise ValueError(f"BEAM-10M chat plan {index} is not an object")
        populated = {
            str(key): batches
            for key, batches in plan.items()
            if batches is not None
        }
        if len(populated) != 1:
            raise ValueError(
                f"BEAM-10M chat plan {index} has "
                f"{len(populated)} populated fields"
            )
        result.append(populated)
    return result


def verify_shard(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"unexpected size for {path}: {actual_size} != {expected_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    print(
        f"[beam-download] verified {path.name}: "
        f"size={actual_size} sha256={actual_sha256}",
        flush=True,
    )


def normalize_questions(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError("BEAM-10M probing_questions is not an object")
    return value


def export_selected(
    shard_paths: list[Path],
    out_dir: Path,
    requested: set[str],
    *,
    force: bool,
) -> set[str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to extract BEAM-10M Parquet. Run this script "
            "as: uv run --no-project --with pyarrow python "
            "scripts/kvmem_eval/download_beam_10m.py ..."
        ) from exc

    exported: set[str] = set()
    for shard_path in shard_paths:
        parquet_file = parquet.ParquetFile(shard_path)
        required_columns = {
            "conversation_id",
            "chat",
            "probing_questions",
        }
        available = set(parquet_file.schema_arrow.names)
        missing = required_columns - available
        if missing:
            raise ValueError(
                f"{shard_path} is missing columns: {sorted(missing)}"
            )
        for batch in parquet_file.iter_batches(
            batch_size=1,
            columns=sorted(required_columns),
        ):
            if exported == requested:
                break
            row = batch.to_pylist()[0]
            conversation_id = str(row["conversation_id"])
            if conversation_id not in requested:
                continue
            conversation_dir = out_dir / conversation_id
            chat_path = conversation_dir / "chat.json"
            question_path = (
                conversation_dir
                / "probing_questions"
                / "probing_questions.json"
            )
            if (
                not force
                and chat_path.is_file()
                and question_path.is_file()
            ):
                print(
                    f"[beam-download] exported files exist: "
                    f"conversation {conversation_id}",
                    flush=True,
                )
                exported.add(conversation_id)
                continue
            question_path.parent.mkdir(parents=True, exist_ok=True)
            chat_path.write_text(
                json.dumps(
                    normalize_chat(row["chat"]),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            question_path.write_text(
                json.dumps(
                    normalize_questions(row["probing_questions"]),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            exported.add(conversation_id)
            print(
                f"[beam-download] exported conversation={conversation_id} "
                f"chat={chat_path.stat().st_size / 2**20:.1f} MiB",
                flush=True,
            )
    return exported


def main() -> int:
    args = parse_args()
    removed = []
    for name in PROXY_VARIABLES:
        if name in os.environ:
            removed.append(name)
            os.environ.pop(name, None)
    print(
        "[beam-download] proxy environment unset: "
        + (", ".join(removed) if removed else "already clear"),
        flush=True,
    )
    ids = parse_id_list(args.conversations)
    invalid = [value for value in ids if not value.isdigit() or not 1 <= int(value) <= 10]
    if invalid:
        raise SystemExit(f"BEAM-10M conversation ids must be 1..10: {invalid}")
    session = requests.Session()
    session.trust_env = False
    cache_dir = args.out_dir / "_hf_cache"
    requested = set(ids)
    exported: set[str] = set()
    for shard, expected_size, expected_sha256 in SHARDS:
        if exported == requested:
            break
        destination = cache_dir / shard
        download(
            session,
            f"{HF_MIRROR_ROOT}/{shard}?download=true",
            destination,
            force=args.force,
            timeout=args.timeout,
        )
        verify_shard(destination, expected_size, expected_sha256)
        exported.update(
            export_selected(
                [destination],
                args.out_dir,
                requested - exported,
                force=args.force,
            )
        )
    missing = requested - exported
    if missing:
        raise RuntimeError(
            f"requested conversations not found in official shards: "
            f"{sorted(missing)}"
        )
    print(
        f"[beam-download] ready: {len(ids)} conversation(s) under "
        f"{args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
