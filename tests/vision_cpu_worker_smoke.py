#!/usr/bin/env python3
"""Small real-checkpoint smoke test for the persistent CPU vision worker."""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--worker", default="scripts/qw3_vision_cpu_worker.py", type=Path)
    parser.add_argument("--python", default=".venv/bin/python", type=Path)
    args = parser.parse_args()

    image = Image.new("RGB", (32, 24), color=(20, 80, 160))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    request = {
        "op": "encode",
        "images": [{
            "media_type": "image/png",
            "data": base64.b64encode(encoded.getvalue()).decode(),
        }],
    }

    process = subprocess.Popen(
        [str(args.python), str(args.worker), "--model", str(args.model),
         "--threads", "2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    ready = json.loads(process.stdout.readline())
    if ready != {"ok": True, "event": "ready"}:
        raise RuntimeError(f"worker did not become ready: {ready}")
    process.stdin.write(json.dumps(request).encode() + b"\n")
    process.stdin.flush()
    response = json.loads(process.stdout.readline())
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "worker encode failed"))
    payload = process.stdout.read(response["bytes"])
    if len(payload) != response["bytes"]:
        raise RuntimeError("worker returned a short embedding payload")
    if response["rows"] != sum(response["counts"]) or response["dim"] <= 0:
        raise RuntimeError(f"invalid worker shape: {response}")
    process.stdin.write(b'{"op":"shutdown"}\n')
    process.stdin.flush()
    if process.wait(timeout=10) != 0:
        raise RuntimeError("worker shutdown failed")
    print(
        "vision_cpu_worker_smoke: PASS "
        f"rows={response['rows']} dim={response['dim']} grids={response['grids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
