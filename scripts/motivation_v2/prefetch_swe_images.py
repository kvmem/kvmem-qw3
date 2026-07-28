#!/usr/bin/env python3
"""Prefetch SWE-bench Lite base images via a China mirror, then retag to docker.io/swebench/..."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MIRROR = "docker.1ms.run"


def instance_to_swe_image(iid: str) -> str:
    owner, rest = iid.split("__", 1)
    return f"swebench/sweb.eval.x86_64.{owner}_1776_{rest}:latest"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/results/motivation_v2/manifest.jsonl"),
    )
    ap.add_argument("--mirror", default=DEFAULT_MIRROR)
    ap.add_argument("--status", default="pending", help="only prefetch these statuses (comma)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    want = {s.strip() for s in args.status.split(",") if s.strip()}
    rows = []
    for line in args.manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if want and r.get("status") not in want:
            continue
        rows.append(r["instance_id"])
    if args.limit:
        rows = rows[: args.limit]
    ok = fail = skip = 0
    for iid in rows:
        short = instance_to_swe_image(iid)
        official = f"docker.io/{short}"
        mirrored = f"{args.mirror}/{short}"
        # skip if already local under official name
        probe = subprocess.run(
            ["docker", "image", "inspect", official],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            print(f"SKIP {iid} (already have {official})")
            skip += 1
            continue
        print(f"PULL {mirrored}")
        if args.dry_run:
            continue
        p = subprocess.run(["docker", "pull", mirrored])
        if p.returncode != 0:
            print(f"FAIL pull {iid}", file=sys.stderr)
            fail += 1
            continue
        subprocess.check_call(["docker", "tag", mirrored, official])
        # also tag without docker.io prefix for buildx
        subprocess.check_call(["docker", "tag", mirrored, short])
        print(f"OK   {iid} -> {official}")
        ok += 1
    print(f"done ok={ok} skip={skip} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
