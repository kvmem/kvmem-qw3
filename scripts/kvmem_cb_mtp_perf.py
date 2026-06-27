#!/usr/bin/env python3
"""Throughput sanity check for the fused kvmem + CB + MTP path.

The functional gate (kvmem_cb_mtp_gate.py) proves the four features compose and
stay byte-identical at an identity budget. This driver answers the *separate*
question: did wiring kvmem into the continuous-batching MTP verify path cost us
a lot of decode throughput? A feature that works but halves tok/s is not usable.

It starts one serve per config, fires `--conc` concurrent completion requests,
and reports two numbers parsed from the server's own per-request log line
("native continuous_batch: ... decoded=N decode=Xs (Y tok/s)"):

  * mean per-request decode tok/s (steady-state cost per stream)
  * aggregate decode tok/s = sum(decoded) / batch wall (system throughput)

Configs compared (identical serve flags except the kvmem switch):
  1. plain        : CB + MTP, kvmem OFF                       (baseline)
  2. kvmem_ident  : CB + MTP + kvmem at identity budget       (window overhead,
                    no pruning -> attends the SAME KV as plain, so any delta is
                    pure bookkeeping cost)
  3. kvmem_sparse : CB + MTP + kvmem at a real sparse budget  (window is smaller
                    than the full cache -> attention does LESS work; shows
                    whether kvmem can be a net throughput win at long ctx)

No NSPLIT pinning here: that is for byte-parity, and disabling split-K would
distort throughput. All configs share one clean env so only the kvmem switch
varies.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402

# "decoded=128 decode=1.234s (103.74 tok/s)" -> (128, 103.74)
_DECODE_RE = re.compile(r"decoded=(\d+)\s+decode=[\d.]+s\s+\(([\d.]+)\s+tok/s\)")
# count the verify route actually taken so the perf number is interpretable
_RAGGED_RE = re.compile(r"native continuous_mtp:")


def long_prompt(tag: str, words: int) -> str:
    body = " ".join(["alpha beta gamma delta epsilon"] * (words // 5 + 1))
    return (f"{tag}\nContext passage follows.\n{body}\n"
            f"Now write a detailed multi-paragraph explanation of the passage.")


def run_perf_config(name: str, qw3: Path, model: Path, prompts: List[str],
                    max_tokens: int, ctx: int, chain: int, conc: int,
                    kvmem_flags: List[str], timeout_s: int,
                    extra_env: Optional[Dict[str, str]] = None) -> dict:
    host = "127.0.0.1"
    port = e2e.find_free_port()
    cmd = [
        str(qw3), "serve", "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx), "-n", str(max_tokens), "--temp", "0",
        "--kv-dtype", "fp16", "--prefill-chunk", "2048",
        "--continuous-batching", "--max-active", str(conc),
        "--native-mtp-speculate", "--mtp-chain", str(chain),
    ]
    cmd += kvmem_flags
    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    bodies: List[str] = []
    batch_wall = 0.0
    try:
        e2e.wait_for_health(host, port, min(180.0, float(timeout_s)))

        def fire(p: str):
            return e2e.post_completion(host, port, p, max_tokens, float(timeout_s))

        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            for _, body in ex.map(fire, prompts):
                bodies.append(body)
        batch_wall = time.monotonic() - t0
    finally:
        log = e2e.terminate_server(proc)

    texts = [e2e.parse_completion_text(b) for b in bodies]
    ok_status = all(t != "" for t in texts)
    pairs = [(int(m.group(1)), float(m.group(2)))
             for m in _DECODE_RE.finditer(log)]
    decoded = [d for d, _ in pairs]
    per_req = [r for _, r in pairs]
    total_decoded = sum(decoded)
    mean_tps = sum(per_req) / len(per_req) if per_req else 0.0
    agg_tps = total_decoded / batch_wall if batch_wall > 0 else 0.0
    ragged_steps = len(_RAGGED_RE.findall(log))
    return {
        "name": name,
        "ok_status": ok_status,
        "requests": len(prompts),
        "reqs_with_stats": len(pairs),
        "total_decoded": total_decoded,
        "batch_wall_s": round(batch_wall, 2),
        "mean_per_req_tps": round(mean_tps, 2),
        "agg_tps": round(agg_tps, 2),
        "ragged_verify_steps": ragged_steps,
        "cmd": cmd,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--chain", type=int, default=3)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--prompt-words", type=int, default=1200)
    ap.add_argument("--sparse-budget", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_cb_mtp_perf.json")
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    prompts = [long_prompt(f"REQ-{i}", args.prompt_words)
               for i in range(args.conc)]

    configs = [
        ("plain", []),
        ("kvmem_ident", ["--kvmem", "--kvmem-block-tokens", "16",
                         "--kvmem-budget", "131072", "--kvmem-method",
                         "recency"]),
        ("kvmem_sparse", ["--kvmem", "--kvmem-block-tokens", "16",
                          "--kvmem-budget", str(args.sparse_budget),
                          "--kvmem-method", "recency",
                          "--kvmem-interval", "32"]),
    ]

    results = []
    for cname, flags in configs:
        print(f"[perf] running {cname} ...", flush=True)
        r = run_perf_config(cname, qw3, model, prompts, args.max_tokens,
                            args.ctx, args.chain, args.conc, flags,
                            args.timeout)
        results.append(r)
        print(f"  ok={r['ok_status']} reqs_stats={r['reqs_with_stats']}/"
              f"{r['requests']} total_decoded={r['total_decoded']} "
              f"wall={r['batch_wall_s']}s mean/req={r['mean_per_req_tps']} "
              f"agg={r['agg_tps']} tok/s ragged_steps={r['ragged_verify_steps']}",
              flush=True)

    base = next((r for r in results if r["name"] == "plain"), None)
    print("\n=== throughput vs plain CB+MTP (kvmem OFF) ===")
    print(f"{'config':14s} {'mean/req tok/s':>14s} {'agg tok/s':>10s} "
          f"{'mean Δ%':>9s} {'agg Δ%':>9s}  ok")
    for r in results:
        if base and base["mean_per_req_tps"] > 0:
            md = 100.0 * (r["mean_per_req_tps"] - base["mean_per_req_tps"]) \
                / base["mean_per_req_tps"]
            ad = 100.0 * (r["agg_tps"] - base["agg_tps"]) / base["agg_tps"]
        else:
            md = ad = 0.0
        print(f"{r['name']:14s} {r['mean_per_req_tps']:14.2f} "
              f"{r['agg_tps']:10.2f} {md:+8.1f}% {ad:+8.1f}%  {r['ok_status']}")

    out = {"args": vars(args), "results": results}
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_json}")

    # advisory verdict: identity-budget kvmem should not cost a lot (it attends
    # the same KV). Flag a >15% regression as "efficiency concern".
    verdict = 0
    if base:
        ident = next((r for r in results if r["name"] == "kvmem_ident"), None)
        if ident and base["mean_per_req_tps"] > 0:
            drop = 100.0 * (base["mean_per_req_tps"] - ident["mean_per_req_tps"]) \
                / base["mean_per_req_tps"]
            if drop > 15.0:
                print(f"WARN: kvmem identity budget costs {drop:.1f}% per-req "
                      f"throughput vs plain (>15% concern)")
                verdict = 2
            else:
                print(f"OK: kvmem identity-budget overhead is {drop:.1f}% "
                      f"per-req (<=15%)")
    if not all(r["ok_status"] for r in results):
        print("WARN: a config returned an empty completion")
        verdict = max(verdict, 1)
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
