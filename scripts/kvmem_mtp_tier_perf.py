#!/usr/bin/env python3
"""Throughput of the fused kvmem + CB + MTP path WITH the CPU/NVMe tiers active.

The compatibility gate (kvmem_mtp_compat.py) proves kvmem + CB + MTP compose and
stay byte-identical at an identity budget. kvmem_cb_mtp_perf.py measures the
GPU-resident kvmem overhead. This driver answers the remaining question the user
asked: at MTP chain=4, what does decode throughput look like once the *tiered*
KV memory is switched on -- i.e. when the bounded GPU page pool forces blocks out
to the CPU tier and (on overflow) the NVMe tier, and selected blocks must be
staged back in?

Configs (identical serve flags except the kvmem/tier switches):
  1. plain        : CB + MTP, kvmem OFF                          (baseline)
  2. kvmem_gpu    : CB + MTP + kvmem sparse, all blocks on GPU   (no tier; pure
                    window/selection bookkeeping cost)
  3. kvmem_cpu    : kvmem sparse + bounded GPU pool + CPU tier   (stage-out to
                    pinned host memory; stage-in on recall)
  4. kvmem_nvme   : kvmem sparse + bounded GPU pool + small CPU
                    tier + NVMe backing                          (CPU overflows
                    to NVMe -> exercises the SSD read/write path)

The bounded GPU pool only engages when its block capacity is below the context's
block count, which is driven by --gpu-ratio (a small fraction of device memory).
That is what turns stage-out / stage-in on; with the default 0.50 ratio every
block fits on GPU and the tier path is dormant.

Tier activity is read straight from the server's own QW3_KVMEM_TIER_TRACE lines
so the throughput number is interpretable (you can see how much eviction / recall
actually happened). No NSPLIT pinning here -- that is for byte parity and would
distort throughput.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402

_DECODE_RE = re.compile(r"decoded=(\d+)\s+decode=[\d.]+s\s+\(([\d.]+)\s+tok/s\)")
_RAGGED_RE = re.compile(r"native continuous_mtp:")
_POOL_RE = re.compile(r"bounded_gpu_pool blocks=(\d+) pages=(\d+)")
_STAGE_OUT_RE = re.compile(r"stage_out block=\d+ to=(cpu|nvme)")
_STAGE_IN_CPU_RE = re.compile(r"stage_in block=\d+ from=cpu")
_STAGE_IN_NVME_RE = re.compile(r"stage_in(?:_async_read)? block=\d+ from=nvme")
_CPU_EVICT_RE = re.compile(r"cpu_evict block=\d+ to=nvme")


def recall_prompt(tag: str, n_facts: int) -> str:
    """A long, content-varied report whose tail asks the model to cite scattered
    early/middle facts -- the workload most likely to drive block recall."""
    facts = [
        f"Fact {i}: region {i} shipped {chr(65 + i % 26)} units of commodity "
        f"{i * 7 % 100} during quarter {i % 4 + 1} of year {1900 + i}."
        for i in range(n_facts)
    ]
    return (f"{tag}\nRead the following report carefully.\n" + " ".join(facts) +
            "\nNow write a detailed multi-paragraph analysis. Explicitly cite "
            "facts from the very beginning, the middle, and the end of the "
            "report, comparing the regions named there.")


def run_cfg(name: str, qw3: Path, model: Path, prompts: List[str],
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
    env["QW3_KVMEM_TIER_TRACE"] = "1"
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
    pool = _POOL_RE.search(log)
    return {
        "name": name,
        "ok_status": ok_status,
        "requests": len(prompts),
        "reqs_with_stats": len(pairs),
        "total_decoded": total_decoded,
        "batch_wall_s": round(batch_wall, 2),
        "mean_per_req_tps": round(mean_tps, 2),
        "agg_tps": round(agg_tps, 2),
        "ragged_verify_steps": len(_RAGGED_RE.findall(log)),
        "pool_blocks": int(pool.group(1)) if pool else 0,
        "stage_out": len(_STAGE_OUT_RE.findall(log)),
        "stage_in_cpu": len(_STAGE_IN_CPU_RE.findall(log)),
        "stage_in_nvme": len(_STAGE_IN_NVME_RE.findall(log)),
        "cpu_evict_nvme": len(_CPU_EVICT_RE.findall(log)),
        "cmd": cmd,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--facts", type=int, default=300,
                    help="number of distinct facts in the recall prompt")
    ap.add_argument("--sparse-budget", type=int, default=512)
    ap.add_argument("--interval", type=int, default=16)
    ap.add_argument("--method", default="retrieval")
    ap.add_argument("--gpu-ratio", type=float, default=0.0015,
                    help="GPU memory fraction for the bounded KVMem pool; must be "
                         "small enough that pool blocks < ctx blocks to engage "
                         "the tier path")
    ap.add_argument("--nvme-dir", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_mtp_tier_perf.json")
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    nvme_dir = args.nvme_dir or tempfile.mkdtemp(prefix="qw3_kvmem_nvme_")
    Path(nvme_dir).mkdir(parents=True, exist_ok=True)

    prompts = [recall_prompt(f"REQ-{i}", args.facts) for i in range(args.conc)]

    sparse = ["--kvmem", "--kvmem-block-tokens", "16",
              "--kvmem-budget", str(args.sparse_budget),
              "--kvmem-method", args.method,
              "--kvmem-interval", str(args.interval)]
    # bounded pool: small GPU ratio so the pool cannot hold the whole context.
    bound = ["--kvmem-gpu-memory-ratio", str(args.gpu_ratio)]

    configs = [
        ("plain", []),
        ("kvmem_gpu", sparse + ["--kvmem-gpu-memory-ratio", "0.5"]),
        ("kvmem_cpu", sparse + bound + ["--kvmem-cpu-gb", "8"]),
        # tiny CPU tier forces overflow onto NVMe (cpu_evict to=nvme + nvme reads)
        ("kvmem_nvme", sparse + bound + ["--kvmem-cpu-gb", "0.05",
                                         "--kvmem-nvme-gb", "16",
                                         "--kvmem-nvme-dir", nvme_dir]),
    ]

    results = []
    for cname, flags in configs:
        print(f"[perf] running {cname} ...", flush=True)
        r = run_cfg(cname, qw3, model, prompts, args.max_tokens, args.ctx,
                    args.chain, args.conc, flags, args.timeout)
        results.append(r)
        print(f"  ok={r['ok_status']} stats={r['reqs_with_stats']}/{r['requests']}"
              f" decoded={r['total_decoded']} wall={r['batch_wall_s']}s"
              f" mean/req={r['mean_per_req_tps']} agg={r['agg_tps']} tok/s"
              f" | pool={r['pool_blocks']}blk out={r['stage_out']}"
              f" in_cpu={r['stage_in_cpu']} in_nvme={r['stage_in_nvme']}"
              f" cpu->nvme={r['cpu_evict_nvme']}", flush=True)

    base = next((r for r in results if r["name"] == "plain"), None)
    print(f"\n=== chain={args.chain} conc={args.conc} ctx={args.ctx} "
          f"throughput vs plain CB+MTP (kvmem OFF) ===")
    print(f"{'config':12s} {'mean/req':>9s} {'agg':>8s} {'mean Δ%':>8s} "
          f"{'pool':>6s} {'out':>6s} {'in_cpu':>7s} {'in_nvme':>8s}  ok")
    for r in results:
        md = 0.0
        if base and base["mean_per_req_tps"] > 0:
            md = 100.0 * (r["mean_per_req_tps"] - base["mean_per_req_tps"]) \
                / base["mean_per_req_tps"]
        print(f"{r['name']:12s} {r['mean_per_req_tps']:9.2f} {r['agg_tps']:8.2f} "
              f"{md:+7.1f}% {r['pool_blocks']:6d} {r['stage_out']:6d} "
              f"{r['stage_in_cpu']:7d} {r['stage_in_nvme']:8d}  {r['ok_status']}")

    out = {"args": vars(args), "nvme_dir": nvme_dir, "results": results}
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_json}")

    verdict = 0
    tier_engaged = any(r["name"] in ("kvmem_cpu", "kvmem_nvme") and
                       r["pool_blocks"] > 0 for r in results)
    if not tier_engaged:
        print("WARN: bounded GPU pool never engaged -- lower --gpu-ratio so pool "
              "blocks < ctx blocks, else the tier path is dormant")
        verdict = max(verdict, 1)
    if not all(r["ok_status"] for r in results):
        print("WARN: a config returned an empty completion")
        verdict = max(verdict, 1)
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
