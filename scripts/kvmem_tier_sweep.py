#!/usr/bin/env python3
"""Long-context tiered-offload sweep for the kvmem + CB + MTP path.

Answers the operator question: at a fixed GPU KV-pool cap (50% of the card =
~48 GiB) and a fixed 32K step-level retrieval window, how do throughput and
latency move as the *total* context grows past what the GPU can hold, and how
much KV spills into the CPU (pinned host) and NVMe tiers at each size?

The sweep is a single variable -- context length -- across:

    128K, 256K, 512K, 1M, 1.5M, 2M

Fixed setup for every size:
  * GPU KV-pool cap   -- --kvmem-gpu-memory-ratio 0.5 (~48 GiB of the 95.6 GiB
                         card). NOTE: this caps the *KV pool only*; the 27B Q8
                         weights (~29 GiB) + prefill scratch live ON TOP of it,
                         so the nvidia-smi total footprint is ~weights + pool.
  * retrieval window  -- 32K tokens (--kvmem-block-tokens 16 --kvmem-budget
                         32768 => 2048 blocks selected per reselect interval).
  * decode cadence    -- --kvmem-interval 64 (the default; reselect every 64
                         committed tokens, the throughput-optimal cadence).
  * MTP spec decode   -- --native-mtp-speculate --mtp-chain 4, SINGLE-REQUEST
                         path (NO --continuous-batching). This is deliberate: on
                         the continuous-batching path kvmem reuses the full-ctx CB
                         KV cache as its store (external_kv_cache_ != null), so the
                         bounded GPU pool + CPU/NVMe spill is SKIPPED and nothing
                         tiers. The single-request generate_mtp path is the only
                         one where kvmem allocates its own ratio-capped GPU pool +
                         CPU/NVMe tiers, so it is the only path that can answer the
                         offload question. Requests are issued one at a time.

CPU/NVMe tiers are allocated ONLY at sizes whose KV overflows the 48 GiB GPU
cap (>= 1M). The CPU tier does an eager cudaHostAlloc of its full budget at
server start, so allocating 64 GiB of pinned host at 128K-512K (where nothing
spills) is pure waste -- those sizes run GPU-pool-only and naturally report
cpu=0/nvme=0. At >= 1M we pass --kvmem-cpu-gb 64 + --kvmem-nvme-gb 64 so the
overflow cascades GPU -> CPU -> NVMe.

`plain` (kvmem OFF, full-cache attention) is run only where the full fp16 KV
fits in GPU alongside the weights (<= 512K); at >= 1M plain OOMs (64+ GiB KV +
29 GiB weights > 95.6 GiB) so only the tiered kvmem config runs.

Reported per (ctx, config):
  * throughput -- prefill tok/s + decode tok/s (server `native generate:` line).
  * latency    -- TTFT (client SSE) + TBT mean/p50/p90/p99/max.
  * tier usage -- GPU pool used/cap, CPU tier used/cap, NVMe tier used/cap,
                  from the per-request `[kvmem-tier-usage]` line (authoritative
                  block residency at end of decode).
  * GPU memory -- peak per-process + total device MiB sampled from nvidia-smi
                  while the request runs (cross-checks the pool cap + weights).
  * host RSS   -- peak server VmRSS (cross-checks the pinned CPU tier).

Each config: one small warmup request (warms kernels + cuBLAS autotuner; the
pinned host alloc already happened at startup) then exactly ONE measured
request whose prompt fills ~82% of ctx so the cache reaches the target size and
the tiers engage. reqs=1 because each request does a full prompt prefill and at
2M that is ~1.7M tokens (tens of minutes) -- doubling it buys nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402
import kvmem_mtp_latency as lat  # noqa: E402

_TIER_RE = re.compile(
    r"\[kvmem-tier-usage\] total_blocks=(\d+) block_bytes=(\d+) "
    r"gpu_used=(\d+) gpu_cap=(\d+) gpu_pool=(\d+) "
    r"cpu_used=(\d+) cpu_cap=(\d+) nvme_used=(\d+) nvme_cap=(\d+)")

_MiB = 1024.0 * 1024.0
_GiB = 1024.0 * _MiB


def _gib(nbytes: float) -> float:
    return round(nbytes / _GiB, 3)


def gpu_proc_mib(pid: int) -> int:
    """Per-process GPU memory (MiB) held by `pid`, 0 if not present."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                return int(parts[1])
            except ValueError:
                return 0
    return 0


def gpu_total_mib() -> int:
    """Total device memory used (MiB) across all processes, 0 on error."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0
    line = out.splitlines()[0].strip() if out.splitlines() else ""
    try:
        return int(line)
    except ValueError:
        return 0


def host_rss_mib(pid: int) -> int:
    """Resident set size (MiB) of the server process, 0 on error."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    kb = int(ln.split()[1])
                    return kb // 1024
    except Exception:
        return 0
    return 0


class ResourceSampler(threading.Thread):
    """Polls nvidia-smi + /proc RSS for a server pid; tracks peak usage."""

    def __init__(self, pid: int, period_s: float = 2.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.period = period_s
        self._stop_evt = threading.Event()
        self.peak_gpu_proc = 0
        self.peak_gpu_total = 0
        self.peak_rss = 0
        self.samples = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            gp = gpu_proc_mib(self.pid)
            gt = gpu_total_mib()
            rss = host_rss_mib(self.pid)
            self.peak_gpu_proc = max(self.peak_gpu_proc, gp)
            self.peak_gpu_total = max(self.peak_gpu_total, gt)
            self.peak_rss = max(self.peak_rss, rss)
            self.samples += 1
            self._stop_evt.wait(self.period)

    def stop(self) -> None:
        self._stop_evt.set()


def parse_tier_usage(log: str) -> Optional[Dict[str, int]]:
    """Last [kvmem-tier-usage] line (the measured request, after warmup)."""
    matches = list(_TIER_RE.finditer(log))
    if not matches:
        return None
    g = matches[-1].groups()
    keys = ("total_blocks", "block_bytes", "gpu_used", "gpu_cap", "gpu_pool",
            "cpu_used", "cpu_cap", "nvme_used", "nvme_cap")
    return {k: int(v) for k, v in zip(keys, g)}


def run_one(name: str, qw3: Path, model: Path, ctx: int, chain: int,
            measured_prompt: str, warmup_prompt: str, max_tokens: int,
            warmup_tokens: int, kvmem_flags: List[str], kv_dtype: str,
            prefill_chunk: int, timeout_s: int) -> dict:
    host = "127.0.0.1"
    port = e2e.find_free_port()
    cmd = [
        str(qw3), "serve", "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx), "-n", str(max_tokens), "--temp", "0",
        "--kv-dtype", kv_dtype, "--prefill-chunk", str(prefill_chunk),
        "--native-mtp-speculate", "--mtp-chain", str(chain),
    ] + kvmem_flags
    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    logf = tempfile.NamedTemporaryFile(
        mode="w+", prefix=f"qw3_tier_{name}_{ctx}_", suffix=".log",
        delete=False)
    print(f"  [{name} ctx={ctx}] launching server (timeout {timeout_s}s) ...",
          flush=True)
    t_launch = time.monotonic()
    proc = subprocess.Popen(cmd, text=True, stdout=logf,
                            stderr=subprocess.STDOUT, env=env)
    sampler = ResourceSampler(proc.pid, period_s=2.0)
    result: Dict[str, object] = {"name": name, "ctx": ctx, "ok": False}
    meas = None
    try:
        e2e.wait_for_health(host, port, min(float(timeout_s), 600.0))
        startup_s = time.monotonic() - t_launch
        print(f"  [{name} ctx={ctx}] healthy in {startup_s:.1f}s; warmup ...",
              flush=True)
        sampler.start()
        _ = lat.stream_chat(host, port, warmup_prompt, warmup_tokens,
                            float(timeout_s))
        print(f"  [{name} ctx={ctx}] measured request ...", flush=True)
        t0 = time.monotonic()
        meas = lat.stream_chat(host, port, measured_prompt, max_tokens,
                               float(timeout_s))
        wall = time.monotonic() - t0
        result["wall_s"] = round(wall, 2)
        result["startup_s"] = round(startup_s, 1)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Kill the server FIRST and unconditionally -- a leaked server holds the
        # GPU pool + 64 GiB pinned host and would OOM the next config. Do not let
        # any sampler-cleanup exception skip this.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        try:
            sampler.stop()
            sampler.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        logf.flush()
        logf.close()
        log = Path(logf.name).read_text(encoding="utf-8", errors="replace")

    result["log_path"] = logf.name
    result["peak_gpu_proc_mib"] = sampler.peak_gpu_proc
    result["peak_gpu_total_mib"] = sampler.peak_gpu_total
    result["peak_host_rss_mib"] = sampler.peak_rss

    gen = [m.groups() for m in lat._GEN_RE.finditer(log)]
    if gen:
        g = gen[-1]  # measured request (warmup is earlier)
        prompt_tokens = int(g[0])
        prefill_s = float(g[1])
        decoded = int(g[2])
        decode_s = float(g[3])
        result["prompt_tokens"] = prompt_tokens
        result["prefill_s"] = round(prefill_s, 3)
        result["decoded"] = decoded
        result["decode_s"] = round(decode_s, 3)
        result["prefill_tps"] = round(prompt_tokens / max(prefill_s, 1e-9), 2)
        result["decode_tps"] = round(decoded / max(decode_s, 1e-9), 2)

    if meas and meas.get("ok"):
        tbt = [float(x) for x in meas["tbt_s"]]  # type: ignore[index]
        result["ok"] = True
        result["ttft_s"] = round(float(meas["ttft_s"]), 4)
        result["tbt_mean_ms"] = round(1000 * statistics.mean(tbt), 3) if tbt else 0.0
        result["tbt_p50_ms"] = round(1000 * lat.pctile(tbt, 0.5), 3)
        result["tbt_p90_ms"] = round(1000 * lat.pctile(tbt, 0.9), 3)
        result["tbt_p99_ms"] = round(1000 * lat.pctile(tbt, 0.99), 3)
        result["tbt_max_ms"] = round(1000 * max(tbt), 3) if tbt else 0.0
        result["deltas"] = int(meas["deltas"])
    elif meas:
        result["error"] = (result.get("error") or
                           f"stream status={meas.get('status')} "
                           f"{str(meas.get('error'))[:160]}")

    tier = parse_tier_usage(log)
    if tier:
        result["tier"] = tier
        result["gpu_pool"] = bool(tier["gpu_pool"])
        result["gpu_used_gib"] = _gib(tier["gpu_used"])
        result["gpu_cap_gib"] = _gib(tier["gpu_cap"])
        result["cpu_used_gib"] = _gib(tier["cpu_used"])
        result["cpu_cap_gib"] = _gib(tier["cpu_cap"])
        result["nvme_used_gib"] = _gib(tier["nvme_used"])
        result["nvme_cap_gib"] = _gib(tier["nvme_cap"])
        result["total_kv_gib"] = _gib(tier["total_blocks"] * tier["block_bytes"])
    return result


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx-list", default="131072,262144,524288,1048576,1572864,2097152",
                    help="comma-separated context sizes to sweep")
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="measured-request decode length (TBT sample count)")
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--window", type=int, default=32768,
                    help="retrieval window in tokens (--kvmem-budget)")
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--interval", type=int, default=64)
    ap.add_argument("--method", default="retrieval")
    ap.add_argument("--gpu-ratio", type=float, default=0.5,
                    help="GPU KV-pool cap as a fraction of total device memory")
    ap.add_argument("--cpu-gb", type=float, default=64.0)
    ap.add_argument("--nvme-gb", type=float, default=64.0)
    ap.add_argument("--nvme-dir", default="",
                    help="NVMe tier directory (default: a tmpdir, cleaned up)")
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--fact-fill", type=float, default=0.90,
                    help="fraction of ctx (in TOKENS) to fill with recall facts. "
                         "0.90 with tok-per-fact 35 keeps ~10%% headroom to ctx "
                         "(no HTTP 413, leaves room for the decoded response) while "
                         "still pushing the 2M point past GPU(48)+CPU(64)=112 GiB so "
                         "NVMe engages; lower it for a bigger 413 safety margin.")
    ap.add_argument("--tok-per-fact", type=float, default=35.0,
                    help="measured tokens per recall fact. ~32 at small fact counts, "
                         "rising to ~34-35 at the 2M point (larger fact indices = "
                         "more digits); 35 is the conservative upper bound that keeps "
                         "the generated prompt under ctx at every size.")
    ap.add_argument("--tier-threshold", type=int, default=1048576,
                    help="ctx at/above which CPU+NVMe tiers are allocated")
    ap.add_argument("--plain-max", type=int, default=524288,
                    help="max ctx at which the plain (kvmem-OFF) config is run")
    ap.add_argument("--timeout-base", type=int, default=900)
    ap.add_argument("--timeout-per-ktok", type=float, default=3.6,
                    help="extra timeout seconds per 1K ctx tokens")
    ap.add_argument("--timeout-cap", type=int, default=10800)
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_tier_sweep.json")
    ap.add_argument("--skip-plain", action="store_true")
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    ctx_list = [int(x) for x in args.ctx_list.split(",") if x.strip()]
    own_nvme = not args.nvme_dir
    nvme_dir = args.nvme_dir or tempfile.mkdtemp(prefix="qw3_tier_nvme_")
    Path(nvme_dir).mkdir(parents=True, exist_ok=True)

    sparse = ["--kvmem", "--kvmem-block-tokens", str(args.block_tokens),
              "--kvmem-budget", str(args.window),
              "--kvmem-method", args.method,
              "--kvmem-interval", str(args.interval),
              "--kvmem-gpu-memory-ratio", str(args.gpu_ratio)]

    print(f"=== kvmem tiered-offload sweep ===", flush=True)
    print(f"model={model.name} window={args.window}tok ratio={args.gpu_ratio} "
          f"cpu={args.cpu_gb}GiB nvme={args.nvme_gb}GiB chain={args.chain} "
          f"max_tokens={args.max_tokens} kv={args.kv_dtype}", flush=True)
    print(f"ctx sizes: {[c for c in ctx_list]}", flush=True)
    print(f"nvme_dir={nvme_dir}\n", flush=True)

    warmup_prompt = lat.recall_prompt("WARMUP", 30)
    results: List[dict] = []

    for ctx in ctx_list:
        n_facts = max(1, int(args.fact_fill * ctx / args.tok_per_fact))
        approx_tok = int(n_facts * args.tok_per_fact)
        measured_prompt = lat.recall_prompt("REQ-0", n_facts)
        timeout_s = min(args.timeout_cap,
                        int(args.timeout_base + args.timeout_per_ktok * ctx / 1024))
        print(f"--- ctx={ctx} ({ctx//1024}K) facts={n_facts} "
              f"(~{approx_tok} prompt tok) timeout={timeout_s}s ---", flush=True)

        configs: List[tuple] = []
        kvmem_flags = list(sparse)
        if ctx >= args.tier_threshold:
            kvmem_flags += ["--kvmem-cpu-gb", str(args.cpu_gb),
                            "--kvmem-nvme-gb", str(args.nvme_gb),
                            "--kvmem-nvme-dir", nvme_dir]
        configs.append(("kvmem", kvmem_flags))
        if not args.skip_plain and ctx <= args.plain_max:
            configs.append(("plain", []))

        for cname, flags in configs:
            r = run_one(cname, qw3, model, ctx, args.chain, measured_prompt,
                        warmup_prompt, args.max_tokens, args.warmup_tokens,
                        flags, args.kv_dtype, args.prefill_chunk, timeout_s)
            results.append(r)
            if r.get("ok"):
                tier_s = ""
                if "tier" in r:
                    tier_s = (f" | KV={r['total_kv_gib']}GiB "
                              f"GPU={r['gpu_used_gib']}/{r['gpu_cap_gib']}"
                              f"(pool={int(r['gpu_pool'])}) "
                              f"CPU={r['cpu_used_gib']}/{r['cpu_cap_gib']} "
                              f"NVMe={r['nvme_used_gib']}/{r['nvme_cap_gib']}")
                print(f"  [{cname} ctx={ctx}] OK "
                      f"prefill={r.get('prefill_tps','?')}tok/s "
                      f"decode={r.get('decode_tps','?')}tok/s "
                      f"TTFT={r.get('ttft_s','?')}s "
                      f"TBT={r.get('tbt_mean_ms','?')}/{r.get('tbt_p99_ms','?')}ms "
                      f"GPU_peak={r['peak_gpu_proc_mib']}MiB "
                      f"RSS={r['peak_host_rss_mib']}MiB{tier_s}", flush=True)
            else:
                print(f"  [{cname} ctx={ctx}] FAIL: {r.get('error','?')} "
                      f"(log {r.get('log_path')})", flush=True)
            # Persist incrementally so a crash mid-sweep keeps prior rows.
            Path(args.out_json).write_text(
                json.dumps({"args": vars(args), "nvme_dir": nvme_dir,
                            "results": results}, indent=2), encoding="utf-8")

    print(f"\n=== SUMMARY (window={args.window}tok, GPU cap ratio "
          f"{args.gpu_ratio}) ===", flush=True)
    hdr = (f"{'ctx':>6s} {'cfg':>6s} {'prefill':>9s} {'decode':>8s} "
           f"{'TTFT(s)':>9s} {'TBTmean':>8s} {'TBTp99':>8s} "
           f"{'KV':>7s} {'GPUpool':>8s} {'CPU':>7s} {'NVMe':>7s} "
           f"{'GPUmib':>8s} {'RSSmib':>8s}")
    print(hdr, flush=True)
    for r in results:
        ctxk = f"{r['ctx']//1024}K"
        if r.get("ok"):
            print(f"{ctxk:>6s} {r['name']:>6s} "
                  f"{r.get('prefill_tps',0):9.1f} {r.get('decode_tps',0):8.1f} "
                  f"{r.get('ttft_s',0):9.3f} {r.get('tbt_mean_ms',0):8.2f} "
                  f"{r.get('tbt_p99_ms',0):8.2f} "
                  f"{r.get('gpu_used_gib',0):7.2f} "
                  f"{r.get('gpu_cap_gib',0):8.2f} "
                  f"{r.get('cpu_used_gib',0):7.2f} {r.get('nvme_used_gib',0):7.2f} "
                  f"{r['peak_gpu_proc_mib']:8d} {r['peak_host_rss_mib']:8d}",
                  flush=True)
        else:
            print(f"{ctxk:>6s} {r['name']:>6s}  FAIL: {str(r.get('error'))[:80]}",
                  flush=True)

    Path(args.out_json).write_text(
        json.dumps({"args": vars(args), "nvme_dir": nvme_dir,
                    "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_json}", flush=True)

    if own_nvme:
        try:
            shutil.rmtree(nvme_dir, ignore_errors=True)
            print(f"cleaned nvme_dir {nvme_dir}", flush=True)
        except Exception:
            pass

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
