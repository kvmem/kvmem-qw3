#!/usr/bin/env python3
"""Build llama-server and smoke-verify its MTP speculative path.

Why this exists: the benchmark suite drives llama.cpp via llama-server with
`--spec-type draft-mtp`. That target must be built, and MTP only initializes
against the MTP-enabled GGUF (nextn_predict_layers=1). This script builds the
target and confirms MTP actually initializes before a long sweep wastes time.

Usage (from repo root):
    python3 scripts/bench/build_llama_server.py
    python3 scripts/bench/build_llama_server.py --build-dir /tmp/llama.cpp/build-cuda
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from bench.config import DEFAULT_MODEL, DEFAULT_LLAMA_SERVER


def build(build_dir: Path) -> int:
    if not build_dir.exists():
        print(f"ERROR: build dir not found: {build_dir}\n"
              f"(expected a pre-configured CMake cache with LLAMA_BUILD_SERVER=ON)",
              file=sys.stderr)
        return 2
    jobs = str(os.cpu_count() or 4)
    cmd = ["cmake", "--build", str(build_dir), "--target", "llama-server", "-j", jobs]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def smoke(server: Path, model: Path, port: int) -> int:
    """Reuse probe_llama_mtp_server to assert mtp_initialized=true."""
    import probe_llama_mtp_server as probe  # type: ignore

    argv = [
        "--server", str(server),
        "--model", str(model),
        "-c", "1024", "-n", "16",
        "--port", str(port),
    ]
    print(f"+ smoke: probe_llama_mtp_server {' '.join(argv)}", flush=True)
    rc = probe.main(argv) if _accepts_argv(probe.main) else _run_probe_subprocess(server, model, port)
    return rc


def _accepts_argv(fn) -> bool:
    import inspect
    try:
        return len(inspect.signature(fn).parameters) >= 1
    except (TypeError, ValueError):
        return False


def _run_probe_subprocess(server: Path, model: Path, port: int) -> int:
    cmd = [sys.executable, str(_SCRIPTS_DIR / "probe_llama_mtp_server.py"),
           "--server", str(server), "--model", str(model),
           "-c", "1024", "-n", "16", "--port", str(port)]
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="/tmp/llama.cpp/build-cuda")
    ap.add_argument("--server", default=DEFAULT_LLAMA_SERVER)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=18097)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_build:
        rc = build(Path(args.build_dir))
        if rc != 0:
            print(f"build failed (rc={rc})", file=sys.stderr)
            return rc

    server = Path(args.server)
    if not server.exists():
        print(f"ERROR: llama-server not found after build: {server}", file=sys.stderr)
        return 2
    model = Path(args.model)
    if not model.exists():
        print(f"ERROR: model not found: {model}", file=sys.stderr)
        return 2

    if args.skip_smoke:
        print(f"OK: llama-server present at {server} (smoke skipped)")
        return 0

    rc = smoke(server, model, args.port)
    if rc == 0:
        print("OK: llama-server built and MTP smoke passed (mtp_initialized=true)")
    else:
        print(f"WARN: MTP smoke returned rc={rc} — check model is the MTP-enabled "
              f"GGUF (nextn_predict_layers=1)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
