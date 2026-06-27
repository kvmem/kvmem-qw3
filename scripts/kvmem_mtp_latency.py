#!/usr/bin/env python3
"""Latency + component breakdown for the fused kvmem + CB + MTP path.

kvmem_mtp_tier_perf.py answers "what is steady-state decode throughput once the
tiers are on?". That is necessary but not sufficient: stage-out is front-loaded
into prefill (outside the decode timer) and stage-in is workload-driven, so a
decode-tok/s number alone hides where the tier cost actually lands. This driver
measures the latency a streaming client sees and attributes it to components:

  * TTFT  -- time to first generated token (client-side, SSE). Captures prefill +
             the first window assembly (first retrieval + stage_out + re-RoPE).
  * TBT   -- inter-token latency distribution (mean / p50 / p90 / p99 / max).
             MTP commits tokens in bursts, so the tail is what periodic reselect
             (retrieval + stage_in + stage_out + assemble on the interval) costs.
  * prefill tok/s -- read from the server's own `native continuous_batch:` line.
  * component breakdown -- retrieval (selection scoring + pick_topk), stage_in
             (CPU/NVMe -> GPU), stage_out (GPU -> CPU/NVMe), assemble (re-RoPE),
             read from the server's per-request `[kvmem-timing]` line (emitted
             only under QW3_KVMEM_TIMING, which also adds a device sync to the
             async regions -- so throughput here is NOT comparable to the perf
             harness; run that separately for clean tok/s).

Configs mirror kvmem_mtp_tier_perf.py: plain / kvmem_gpu / kvmem_cpu /
kvmem_nvme. Streaming uses /v1/chat/completions (the only SSE route); thinking
is disabled and temperature pinned to 0 so generation is short and stable. Each
config runs one warmup request (pays the one-time pinned-host cudaHostAlloc) and
then `--reqs` measured requests, strictly sequentially (conc=1) so the per-
request component lines stay ordered and unambiguous.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402

# The CB-MTP per-request summary. The `native continuous_batch:` line reports
# prefill=0.000s on the MTP admit path (a.prefill_s is never assigned there --
# only the stats struct gets it), so parse `native generate:` which carries the
# real per-request prefill_s and decode_s; compute tok/s from prompt/decoded.
_GEN_RE = re.compile(
    r"native generate: prompt_tokens=(\d+) prefill=([\d.]+)s(?: \([\d.]+ tok/s\))? "
    r"decoded=(\d+) decode=([\d.]+)s")
# Two lines per request now: phase=prefill (TTFT-window cost) and phase=decode
# (the per-interval reselect cadence). assemble_ms is itemized into its three
# GPU substeps: pages (window page-table build + H2D sync = "virtual page
# construction"), rerope (re-RoPE of moved blocks), kbar (k̄ recompute).
_TIMING_RE = re.compile(
    r"\[kvmem-timing\] phase=(\w+) request=(\d+) retrieval_ms=([\d.]+) "
    r"stage_in_ms=([\d.]+) stage_out_ms=([\d.]+) assemble_ms=([\d.]+) "
    r"\(pages=([\d.]+) rerope=([\d.]+) kbar=([\d.]+)\) \| retrieval=(\d+) "
    r"stage_in=(\d+)\((\d+)blk\) stage_out=(\d+)\((\d+)blk\) assemble=(\d+)")


def recall_prompt(tag: str, n_facts: int) -> str:
    facts = [
        f"Fact {i}: region {i} shipped {chr(65 + i % 26)} units of commodity "
        f"{i * 7 % 100} during quarter {i % 4 + 1} of year {1900 + i}."
        for i in range(n_facts)
    ]
    return (f"{tag}\nRead the following report carefully.\n" + " ".join(facts) +
            "\nNow write a detailed multi-paragraph analysis. Explicitly cite "
            "facts from the very beginning, the middle, and the end of the "
            "report, comparing the regions named there.")


def stream_chat(host: str, port: int, prompt: str, max_tokens: int,
                timeout_s: float) -> Dict[str, object]:
    """POST a streaming chat completion; return TTFT + per-token arrival deltas.

    TTFT is timed at the first SSE delta carrying non-empty generated text (the
    role chunk and the empty reasoning prelude are skipped). TBT samples are the
    gaps between consecutive text-bearing deltas.
    """
    body = json.dumps({
        "model": "qw3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": True,
    })
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    t0 = time.monotonic()
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    res = conn.getresponse()
    if res.status != 200:
        data = res.read().decode("utf-8", errors="replace")
        conn.close()
        return {"ok": False, "status": res.status, "error": data[:200]}

    ttft: Optional[float] = None
    token_times: List[float] = []
    text_chars = 0
    fp = res.fp
    while True:
        raw = fp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if not piece:
            continue
        now = time.monotonic()
        if ttft is None:
            ttft = now - t0
        token_times.append(now)
        text_chars += len(piece)
    conn.close()
    total = time.monotonic() - t0

    tbt: List[float] = []
    for i in range(1, len(token_times)):
        tbt.append(token_times[i] - token_times[i - 1])
    return {
        "ok": ttft is not None,
        "status": 200,
        "ttft_s": ttft if ttft is not None else 0.0,
        "tbt_s": tbt,
        "deltas": len(token_times),
        "text_chars": text_chars,
        "wall_s": total,
    }


def pctile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def run_cfg(name: str, qw3: Path, model: Path, prompts: List[str],
            warmup_prompt: str, max_tokens: int, ctx: int, chain: int,
            kvmem_flags: List[str], timeout_s: int,
            timing: bool, kv_dtype: str = "fp16") -> dict:
    host = "127.0.0.1"
    port = e2e.find_free_port()
    cmd = [
        str(qw3), "serve", "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx), "-n", str(max_tokens), "--temp", "0",
        "--kv-dtype", kv_dtype, "--prefill-chunk", "2048",
        "--continuous-batching", "--max-active", "1",
        "--native-mtp-speculate", "--mtp-chain", str(chain),
    ]
    cmd += kvmem_flags
    env = os.environ.copy()
    env["QW3_CONTINUOUS_BATCHING_TRACE"] = "1"
    # NOTE: do NOT enable QW3_KVMEM_TIER_TRACE here. It emits one stderr line per
    # block staged (~360 lines/request) which the harness never parses, and an
    # undrained PIPE fills its ~64 KB OS buffer after a couple of requests -- the
    # server then blocks on fprintf(stderr) mid-decode, the stream stalls, and the
    # client readline times out (looks like a tier hang, but is pure pipe back-
    # pressure). Redirecting to a real file (drained to disk, no fixed buffer)
    # makes capture robust regardless of trace volume; the parsed signals
    # (`native generate:` + `[kvmem-timing]`) are one line per request.
    if timing:
        env["QW3_KVMEM_TIMING"] = "1"
    logf = tempfile.NamedTemporaryFile(
        mode="w+", prefix=f"qw3_kvmem_lat_{name}_", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, text=True, stdout=logf,
                            stderr=subprocess.STDOUT, env=env)
    reqs: List[Dict[str, object]] = []
    try:
        e2e.wait_for_health(host, port, min(180.0, float(timeout_s)))
        # Warmup: pay the one-time pinned-host cudaHostAlloc + first selection so
        # the measured requests reflect steady tier behavior, not setup cost.
        _ = stream_chat(host, port, warmup_prompt, max_tokens, float(timeout_s))
        for p in prompts:
            r = stream_chat(host, port, p, max_tokens, float(timeout_s))
            reqs.append(r)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        logf.flush()
        logf.close()
        log = Path(logf.name).read_text(encoding="utf-8", errors="replace")

    # Server-side per-request lines, in order. The warmup is the first of each;
    # measured requests follow. Match them positionally to the client samples.
    gen = [m.groups() for m in _GEN_RE.finditer(log)]
    n_meas = len(prompts)
    gen_meas = gen[-n_meas:] if len(gen) >= n_meas else gen

    prefill_s = [float(g[1]) for g in gen_meas]
    prefill_tps = [int(g[0]) / max(float(g[1]), 1e-9) for g in gen_meas]
    decode_tps = [int(g[2]) / max(float(g[3]), 1e-9) for g in gen_meas]

    all_ttft = [float(r["ttft_s"]) for r in reqs if r.get("ok")]
    all_tbt: List[float] = []
    for r in reqs:
        if r.get("ok"):
            all_tbt.extend([float(x) for x in r["tbt_s"]])  # type: ignore

    # Two timing lines per request (phase=prefill, phase=decode). Parse them all,
    # keep only the last n_meas distinct request ids (warmup is the first id and
    # is excluded), then aggregate each phase separately into per-request means.
    _MS = ("retrieval_ms", "stage_in_ms", "stage_out_ms", "assemble_ms",
           "pages_ms", "rerope_ms", "kbar_ms")
    _CNT = ("retrieval_calls", "stage_in_calls", "stage_in_blocks",
            "stage_out_calls", "stage_out_blocks", "assemble_calls")
    tm_all: List[Dict[str, object]] = []
    for m in _TIMING_RE.finditer(log):
        g = m.groups()
        tm_all.append({
            "phase": g[0], "request": int(g[1]),
            "retrieval_ms": float(g[2]), "stage_in_ms": float(g[3]),
            "stage_out_ms": float(g[4]), "assemble_ms": float(g[5]),
            "pages_ms": float(g[6]), "rerope_ms": float(g[7]),
            "kbar_ms": float(g[8]), "retrieval_calls": int(g[9]),
            "stage_in_calls": int(g[10]), "stage_in_blocks": int(g[11]),
            "stage_out_calls": int(g[12]), "stage_out_blocks": int(g[13]),
            "assemble_calls": int(g[14]),
        })
    seen_ids: List[int] = []
    for d in tm_all:
        if d["request"] not in seen_ids:
            seen_ids.append(d["request"])  # type: ignore[arg-type]
    meas_ids = set(seen_ids[-n_meas:]) if len(seen_ids) >= n_meas else set(seen_ids)

    def agg_phase(phase: str) -> Dict[str, object]:
        rows = [d for d in tm_all
                if d["phase"] == phase and d["request"] in meas_ids]
        nz = max(1, n_meas)
        out: Dict[str, object] = {}
        for k in _MS:
            out[k] = round(sum(float(d[k]) for d in rows) / nz, 3)  # type: ignore
        for k in _CNT:
            out[k] = sum(int(d[k]) for d in rows)  # type: ignore
        return out

    comp_prefill = agg_phase("prefill")
    comp_decode = agg_phase("decode")
    # Whole-request total (back-compat single view): prefill + decode per req.
    comp_total = {k: (round(float(comp_prefill[k]) + float(comp_decode[k]), 3)
                      if k in _MS else int(comp_prefill[k]) + int(comp_decode[k]))
                  for k in list(_MS) + list(_CNT)}

    return {
        "name": name,
        "ok": all(r.get("ok") for r in reqs) and len(reqs) == n_meas,
        "reqs": len(reqs),
        "ttft_mean_s": round(statistics.mean(all_ttft), 4) if all_ttft else 0.0,
        "ttft_p50_s": round(pctile(all_ttft, 0.5), 4),
        "ttft_max_s": round(max(all_ttft), 4) if all_ttft else 0.0,
        "tbt_mean_ms": round(1000 * statistics.mean(all_tbt), 3) if all_tbt else 0.0,
        "tbt_p50_ms": round(1000 * pctile(all_tbt, 0.5), 3),
        "tbt_p90_ms": round(1000 * pctile(all_tbt, 0.9), 3),
        "tbt_p99_ms": round(1000 * pctile(all_tbt, 0.99), 3),
        "tbt_max_ms": round(1000 * max(all_tbt), 3) if all_tbt else 0.0,
        "tbt_samples": len(all_tbt),
        "prefill_tps_mean": round(statistics.mean(prefill_tps), 2) if prefill_tps else 0.0,
        "prefill_s_mean": round(statistics.mean(prefill_s), 3) if prefill_s else 0.0,
        "decode_tps_mean": round(statistics.mean(decode_tps), 2) if decode_tps else 0.0,
        "component_per_req": comp_total,
        "component_prefill": comp_prefill,
        "component_decode": comp_decode,
        "cmd": cmd,
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--reqs", type=int, default=3,
                    help="measured requests per config (sequential, conc=1)")
    ap.add_argument("--facts", type=int, default=200,
                    help="distinct facts per recall prompt (controls prompt len; "
                         "~6.1k prompt tokens at 200, fits the 8192 ctx with the "
                         "256 max-tokens decode -- 300 overflows ctx -> HTTP 413). "
                         "Pass 0 to auto-size facts to ~82%% of --ctx (long-ctx "
                         "sweeps): ~30.1 tok/fact, leaves headroom for the decode "
                         "tail so the prompt does not overflow ctx -> HTTP 413.")
    ap.add_argument("--kv-dtype", default="fp16",
                    help="KV cache dtype (fp16/q8/fp8). Long contexts (>=512K) "
                         "need q8 to fit the full cache in 96 GiB alongside the "
                         "27B weights; note q8/fp8 K disables k-mean retrieval "
                         "scoring (selection falls back to recency-weighted).")
    ap.add_argument("--sparse-budget", type=int, default=512)
    ap.add_argument("--interval", type=int, default=16)
    ap.add_argument("--method", default="retrieval")
    # The bounded GPU page pool engages only when its block capacity is < the
    # ctx in blocks (here 8192/16 = 512), so spilling to the CPU/NVMe tier only
    # happens below that. The pool must also hold one prefill chunk
    # (--prefill-chunk 2048 = 128 pages) plus the selection budget (512 tokens =
    # 32 blocks) at once: the chunk-aware prefill offload reselects (and frees
    # cold pages) before each append, so as long as one reselect leaves room for
    # the next chunk (pool >= ~chunk + budget) prefill never exhausts the pool.
    # 0.0035 * 96 GiB / 1 MiB ~= 344 blocks: above that working set, below ctx
    # (512), so a 200-fact (~380-block) prompt forces steady stage-out without
    # the mid-stream SSE hang the old 0.0015 (~144 blocks) triggered.
    ap.add_argument("--gpu-ratio", type=float, default=0.0035)
    ap.add_argument("--nvme-dir", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-timing", action="store_true",
                    help="skip the component breakdown (QW3_KVMEM_TIMING off); "
                         "use to get latency without the timing device-sync tax")
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_mtp_latency.json")
    ap.add_argument("--configs", default="",
                    help="comma-separated subset of {plain,kvmem_gpu,kvmem_cpu,"
                         "kvmem_nvme} to run (default: all). Use 'plain,kvmem_gpu' "
                         "to isolate the window-vs-full-attention tradeoff at long "
                         "ctx without paying the cpu/nvme tiers' pinned-host alloc.")
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    nvme_dir = args.nvme_dir or tempfile.mkdtemp(prefix="qw3_kvmem_nvme_")
    Path(nvme_dir).mkdir(parents=True, exist_ok=True)

    # Auto-size facts to fill the context for long-ctx sweeps. ~30.1 tok/fact
    # (calibrated: 200 facts -> 6018 prompt tokens); target 82% of ctx so the
    # decode tail + boilerplate stay under ctx (HTTP 413 guard).
    n_facts = args.facts
    if n_facts <= 0:
        n_facts = max(1, int(0.82 * args.ctx / 30.1))
        print(f"[lat] auto-facts: ctx={args.ctx} -> {n_facts} facts "
              f"(~{int(n_facts * 30.1)} prompt tokens)", flush=True)

    prompts = [recall_prompt(f"REQ-{i}", n_facts) for i in range(args.reqs)]
    warmup = recall_prompt("WARMUP", n_facts)

    sparse = ["--kvmem", "--kvmem-block-tokens", "16",
              "--kvmem-budget", str(args.sparse_budget),
              "--kvmem-method", args.method,
              "--kvmem-interval", str(args.interval)]
    bound = ["--kvmem-gpu-memory-ratio", str(args.gpu_ratio)]

    configs = [
        ("plain", []),
        ("kvmem_gpu", sparse + ["--kvmem-gpu-memory-ratio", "0.5"]),
        ("kvmem_cpu", sparse + bound + ["--kvmem-cpu-gb", "8"]),
        ("kvmem_nvme", sparse + bound + ["--kvmem-cpu-gb", "0.05",
                                         "--kvmem-nvme-gb", "16",
                                         "--kvmem-nvme-dir", nvme_dir]),
    ]

    if args.configs:
        want = {c.strip() for c in args.configs.split(",") if c.strip()}
        configs = [(n, f) for (n, f) in configs if n in want]
        if not configs:
            raise SystemExit(f"--configs '{args.configs}' matched no known config")

    timing = not args.no_timing
    results = []
    for cname, flags in configs:
        print(f"[lat] running {cname} ...", flush=True)
        r = run_cfg(cname, qw3, model, prompts, warmup, args.max_tokens,
                    args.ctx, args.chain, flags, args.timeout, timing,
                    args.kv_dtype)
        results.append(r)
        c = r["component_per_req"]
        print(f"  ok={r['ok']} ttft_mean={r['ttft_mean_s']}s "
              f"tbt_mean={r['tbt_mean_ms']}ms p99={r['tbt_p99_ms']}ms "
              f"prefill={r['prefill_tps_mean']}tok/s decode={r['decode_tps_mean']}tok/s",
              flush=True)
        if timing:
            cp, cd = r["component_prefill"], r["component_decode"]
            print(f"     prefill/req: retrieval={cp['retrieval_ms']}ms "
                  f"stage_in={cp['stage_in_ms']}ms({cp['stage_in_blocks']}blk) "
                  f"stage_out={cp['stage_out_ms']}ms({cp['stage_out_blocks']}blk) "
                  f"assemble={cp['assemble_ms']}ms"
                  f"[pages={cp['pages_ms']} rerope={cp['rerope_ms']} "
                  f"kbar={cp['kbar_ms']}]", flush=True)
            print(f"     decode/req:  retrieval={cd['retrieval_ms']}ms "
                  f"stage_in={cd['stage_in_ms']}ms({cd['stage_in_blocks']}blk) "
                  f"stage_out={cd['stage_out_ms']}ms({cd['stage_out_blocks']}blk) "
                  f"assemble={cd['assemble_ms']}ms"
                  f"[pages={cd['pages_ms']} rerope={cd['rerope_ms']} "
                  f"kbar={cd['kbar_ms']}]", flush=True)

    base = next((r for r in results if r["name"] == "plain"), None)
    print(f"\n=== chain={args.chain} ctx={args.ctx} reqs={args.reqs} "
          f"latency vs plain CB+MTP (kvmem OFF) ===")
    hdr = (f"{'config':11s} {'TTFT(s)':>8s} {'dTTFT%':>7s} {'TBT(ms)':>8s} "
           f"{'p99(ms)':>8s} {'prefill':>8s} {'decode':>7s}")
    print(hdr)
    for r in results:
        dt = 0.0
        if base and base["ttft_mean_s"] > 0:
            dt = 100.0 * (r["ttft_mean_s"] - base["ttft_mean_s"]) / base["ttft_mean_s"]
        print(f"{r['name']:11s} {r['ttft_mean_s']:8.4f} {dt:+6.1f}% "
              f"{r['tbt_mean_ms']:8.3f} {r['tbt_p99_ms']:8.3f} "
              f"{r['prefill_tps_mean']:8.1f} {r['decode_tps_mean']:7.1f}")

    if timing:
        # assemble = pages (virtual page-table build + H2D) + rerope (re-RoPE of
        # moved blocks) + kbar (k̄ recompute). retr = kv selection (pick_topk).
        # Split by phase: PREFILL rows are the TTFT-window cost (first selection +
        # prefill-chunk eviction); DECODE rows are the per-interval reselect
        # cadence that taxes decode throughput.
        for phase, key in (("PREFILL (folds into TTFT)", "component_prefill"),
                           ("DECODE (folds into TBT/decode tok/s)",
                            "component_decode")):
            print(f"\n=== per-request kvmem {phase} component breakdown "
                  f"(ms; device-synced, NOT throughput-comparable) ===")
            print(f"{'config':11s} {'retr':>7s} {'stg_in':>7s} {'stg_out':>8s} "
                  f"{'assemble':>9s} {'=pages':>7s} {'+rerope':>8s} {'+kbar':>7s} "
                  f"{'in_blk':>7s} {'out_blk':>8s} {'reselN':>7s}")
            for r in results:
                c = r[key]
                print(f"{r['name']:11s} {c['retrieval_ms']:7.3f} "
                      f"{c['stage_in_ms']:7.3f} {c['stage_out_ms']:8.3f} "
                      f"{c['assemble_ms']:9.3f} {c['pages_ms']:7.3f} "
                      f"{c['rerope_ms']:8.3f} {c['kbar_ms']:7.3f} "
                      f"{c['stage_in_blocks']:7d} {c['stage_out_blocks']:8d} "
                      f"{c['assemble_calls']:7d}")

    out = {"args": vars(args), "nvme_dir": nvme_dir, "results": results}
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_json}")

    verdict = 0
    if not all(r["ok"] for r in results):
        print("WARN: a config did not complete all requests")
        verdict = 1
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
