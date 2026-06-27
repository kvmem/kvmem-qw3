#!/usr/bin/env python3
"""Reproduce the kvmem_cpu mid-stream SSE hang and capture server stderr.

Launches exactly the kvmem_cpu config the latency harness uses, sends the
warmup + measured recall prompts sequentially, and on the first hang/failure
dumps the full server stderr to /tmp/kvmem_cpu_server.log so we can see where
the decode path stalls (page-pool exhaustion throw, stage-in stall, etc.).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402
import kvmem_mtp_latency as lat  # noqa: E402

import subprocess


def main() -> int:
    qw3 = Path("./build/qw3")
    model = Path("models/Qwen3.6-27B-Q8_0.gguf")
    ctx, max_tokens, chain = 8192, 256, 4
    facts = 200
    per_req_timeout = 120.0

    sparse = ["--kvmem", "--kvmem-block-tokens", "16", "--kvmem-budget", "512",
              "--kvmem-method", "retrieval", "--kvmem-interval", "16"]
    bound = ["--kvmem-gpu-memory-ratio", "0.0035"]
    flags = sparse + bound + ["--kvmem-cpu-gb", "8"]

    host = "127.0.0.1"
    port = e2e.find_free_port()
    cmd = [
        str(qw3), "serve", "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx), "-n", str(max_tokens), "--temp", "0",
        "--kv-dtype", "fp16", "--prefill-chunk", "2048",
        "--continuous-batching", "--max-active", "1",
        "--native-mtp-speculate", "--mtp-chain", str(chain),
    ] + flags
    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    env["QW3_KVMEM_TIER_TRACE"] = "1"
    env["QW3_KVMEM_TIMING"] = "1"

    print("launching:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    warmup = lat.recall_prompt("WARMUP", facts)
    prompts = [lat.recall_prompt(f"REQ-{i}", facts) for i in range(3)]
    try:
        e2e.wait_for_health(host, port, 180.0)
        print("[health ok] sending warmup ...", flush=True)
        r = lat.stream_chat(host, port, warmup, max_tokens, per_req_timeout)
        print("warmup:", {k: r.get(k) for k in ("ok", "ttft_s", "deltas")},
              flush=True)
        for i, p in enumerate(prompts):
            t0 = time.monotonic()
            r = lat.stream_chat(host, port, p, max_tokens, per_req_timeout)
            dt = time.monotonic() - t0
            print(f"req{i}: ok={r.get('ok')} ttft={r.get('ttft_s')} "
                  f"deltas={r.get('deltas')} wall={dt:.1f}s "
                  f"status={r.get('status')} err={r.get('error')}", flush=True)
            if not r.get("ok"):
                print("FAILED/HUNG on req", i, flush=True)
                break
    finally:
        log = e2e.terminate_server(proc)
        Path("/tmp/kvmem_cpu_server.log").write_text(log, encoding="utf-8")
        print("wrote /tmp/kvmem_cpu_server.log", len(log), "bytes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
