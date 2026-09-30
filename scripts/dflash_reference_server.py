#!/usr/bin/env python3
"""Launch the pinned SGLang DFlash reference from an isolated environment.

Writes exact arguments/provenance before replacing this process with the
upstream server. The BF16 KV configuration is for reference qualification;
formal QW3 timing requires the same KV dtype and the frozen workload protocol.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REVISION = "9f6fc553322e897a22f30320dae750f4a40a0b84"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference-source", type=Path, required=True)
    p.add_argument("--environment", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--draft", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--port", type=int, default=18083)
    p.add_argument("--max-running", type=int, default=32)
    p.add_argument("--max-total-tokens", type=int, default=131072)
    p.add_argument("--context-length", type=int, default=8192)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--max-mamba-cache-size", type=int)
    p.add_argument("--disable-cuda-graph", action="store_true")
    args = p.parse_args()
    source = args.reference_source.resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"unexpected reference revision: {revision}")
    if subprocess.check_output(["git", "-C", str(source), "diff", "--", "python/sglang"], text=True):
        raise ValueError("reference inference source has local modifications")
    env = os.environ.copy()
    env["CUDA_HOME"] = "/usr/local/cuda-13"
    env["PATH"] = str(args.environment.resolve() / "bin") + ":/usr/local/cuda-13/bin:" + env["PATH"]
    argv = [str(args.environment.resolve() / "bin/python"), "-m", "sglang.launch_server",
        "--model-path", str(args.target.resolve()), "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", str(args.draft.resolve()),
        "--speculative-num-draft-tokens", "8", "--dtype", "bfloat16", "--kv-cache-dtype", "bf16",
        "--attention-backend", "flashinfer", "--speculative-draft-attention-backend", "flashinfer",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--context-length", str(args.context_length), "--max-running-requests", str(args.max_running),
        "--max-total-tokens", str(args.max_total_tokens), "--mem-fraction-static", "0.9",
        "--chunked-prefill-size", "1024", "--cuda-graph-max-bs-decode", str(args.max_running)]
    argv += ["--random-seed", str(args.random_seed)]
    if args.max_mamba_cache_size is not None:
        argv += ["--max-mamba-cache-size", str(args.max_mamba_cache_size)]
    if args.disable_cuda_graph:
        argv += ["--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled"]
    manifest = dict(reference_revision=revision, argv=argv,
        environment_overrides={"CUDA_HOME": env["CUDA_HOME"], "PATH": env["PATH"]},
        target_config_sha256=hashlib.sha256((args.target / "config.json").read_bytes()).hexdigest(),
        draft_config_sha256=hashlib.sha256((args.draft / "config.json").read_bytes()).hexdigest(),
        status="launching", public_performance_gate="not_run")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.execve(argv[0], argv, env)


if __name__ == "__main__":
    main()
