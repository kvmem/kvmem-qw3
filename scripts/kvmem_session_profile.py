#!/usr/bin/env python3
"""Persistent kvmem context-growth profiler (256K -> 2M).

Launches ONE `qw3 kvmem-session` process that prefills an initial long context
and then keeps growing it across turns (each turn appends a fresh document chunk
to reach the next ladder point, then decodes a short MTP probe). The binary
forces the exact configuration the experiment asks for:

  * --kvmem on, single-request native path (kvmem + MTP is single-request only).
  * --kvmem-update-mode step -> no decode-time reselection.
  * Every turn emits a strictly additive wall-clock hierarchy:
        total = setup + prefill + post-prefill + decode + finalize
    and:
        post-prefill = semantic selection + query replay + other
  * KVMem's retrieval/stage-in/stage-out/assembly counters are nested
    diagnostics. They are intentionally NOT added because I/O and assembly may
    overlap.
  * MTP speculate on (realistic decode throughput).
  * QW3_KVMEM_TIMING on so steps 1-4 are wall-clock attributed.

Unlike kvmem_tier_sweep.py (which restarts a fresh server per ctx size and
re-prefills from scratch), this is a SINGLE persistent process keeping the
kvmem block store + KV cache alive across turns -- true incremental growth and
cross-turn tier persistence, which is what a real agentic session looks like.

This driver just supplies the determinism env + tier sizes + ladder, streams
the process output, samples nvidia-smi / RSS for an independent peak
cross-check, parses the per-turn breakdown + final summary table the binary
prints, and writes a JSON + human summary.

The GPU KV pool is capped at --gpu-ratio of the card; the FULL KV store (every
committed block, retained because a later reselect may pick any of them) spills
GPU -> CPU(pinned host) -> NVMe once it overflows the cap. CPU/NVMe tiers are
allocated ONLY when the largest ladder target is >= --tier-threshold (the CPU
tier eagerly cudaHostAllocs its whole budget at startup, so a 64 GiB pinned
alloc for a tiny smoke ladder is pure waste).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_MiB = 1024.0 * 1024.0
_GiB = 1024.0 * _MiB

# Per-turn breakdown block the binary prints (see print_session_turn).
_HDR_RE = re.compile(
    r"\[kvmem-session\] turn=(\d+) ctx=(\d+)tok \(\+(\d+)\) "
    r"GPU=(\d+)MiB RSS=(\d+)MiB")
_TOTAL_RE = re.compile(r"total native execution.*?=\s*([\d.eE+-]+) ms")
_SETUP_RE = re.compile(r"phase1 setup.*?=\s*([\d.eE+-]+) ms")
_PREFILL_RE = re.compile(
    r"phase2 prefill.*?=\s*([\d.eE+-]+) ms\s*\(([\d.eE+-]+) tok/s\)")
_POST_RE = re.compile(r"phase3 post-prefill.*?=\s*([\d.eE+-]+) ms")
_SEMANTIC_RE = re.compile(r"semantic selection.*?=\s*([\d.eE+-]+) ms")
_QUERY_REPLAY_RE = re.compile(r"query replay.*?=\s*([\d.eE+-]+) ms")
_POST_OTHER_RE = re.compile(r"^\s*other.*?=\s*([\d.eE+-]+) ms", re.MULTILINE)
_DECODE_RE = re.compile(
    r"phase4 decode\s*\(MTP, (\d+) tok\).*?=\s*([\d.eE+-]+) ms\s*"
    r"\(([\d.eE+-]+) tok/s, accept=([\d.eE+-]+)\)")
_FINALIZE_RE = re.compile(r"phase5 finalize.*?=\s*([\d.eE+-]+) ms")
_ACCOUNTING_ERROR_RE = re.compile(
    r"accounting error.*?=\s*([\d.eE+-]+) ms")
_S1_RE = re.compile(r"selection \(top-k retrieval\).*?=\s*([\d.eE+-]+) ms")
_S2_RE = re.compile(r"stage-in.*?=\s*([\d.eE+-]+) ms\s*\((\d+) blk\)")
_S3_RE = re.compile(r"stage-out.*?=\s*([\d.eE+-]+) ms\s*\((\d+) blk\)")
_S4_RE = re.compile(
    r"assemble.*?=\s*([\d.eE+-]+) ms\s*\(pages=([\d.eE+-]+) "
    r"rerope=([\d.eE+-]+) kbar=([\d.eE+-]+)\)")
# In-prefill forced-offload diagnostic (cost is INSIDE step5's wall).
_S5B_RE = re.compile(
    r"in-prefill offload.*?=\s*([\d.eE+-]+) ms\s*\(in=(\d+) out=(\d+)")
# A data row of the additive final summary table: 20 whitespace-separated
# numbers. turn ctx delta total setup prefill post decode finalize error_ms
# pre_tok/s dec_tok/s accept KVgib GPUgib CPUgib NVMEgib pool GPUmib RSSmib
_SUMROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
    r"([\d.eE+-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")
_QUERY_ROW_RE = re.compile(
    r"\[kvmem-session-query\] turn=(\d+) query=(\d+) mode=(\w+) "
    r"base=(\d+) span=\[(\d+),(\d+)\) final=(\d+) "
    r"capture_ms=([\d.eE+-]+) restore_ms=([\d.eE+-]+) "
    r"total_ms=([\d.eE+-]+) semantic_ms=([\d.eE+-]+) "
    r"replay_ms=([\d.eE+-]+) decode_ms=([\d.eE+-]+) "
    r"score_ms=([\d.eE+-]+) stage_in_ms=([\d.eE+-]+) "
    r"stage_out_ms=([\d.eE+-]+) assemble_ms=([\d.eE+-]+) "
    r"stage_in_blocks=(\d+) stage_out_blocks=(\d+) decoded=(\d+)")


def _gib(nbytes: float) -> float:
    return round(nbytes / _GiB, 3)


def gpu_proc_mib(pid: int) -> int:
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


def host_rss_mib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) // 1024
    except Exception:
        return 0
    return 0


class ResourceSampler(threading.Thread):
    """Polls per-process GPU MiB + RSS for the harness pid; tracks peak."""

    def __init__(self, pid: int, period_s: float = 3.0):
        super().__init__(daemon=True)
        self.pid = pid
        self.period = period_s
        self._stop_evt = threading.Event()
        self.peak_gpu = 0
        self.peak_rss = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.peak_gpu = max(self.peak_gpu, gpu_proc_mib(self.pid))
            self.peak_rss = max(self.peak_rss, host_rss_mib(self.pid))
            self._stop_evt.wait(self.period)

    def stop(self) -> None:
        self._stop_evt.set()


def parse_turns(log: str) -> List[dict]:
    """Parse the per-turn [kvmem-session] breakdown blocks into dict rows."""
    turns: List[dict] = []
    # Split the log into per-turn segments at each header line.
    parts = re.split(r"(?=\[kvmem-session\] turn=)", log)
    for seg in parts:
        h = _HDR_RE.search(seg)
        if not h:
            continue
        row: Dict[str, object] = {
            "turn": int(h.group(1)),
            "ctx_tokens": int(h.group(2)),
            "delta_tokens": int(h.group(3)),
            "gpu_mib": int(h.group(4)),
            "rss_mib": int(h.group(5)),
        }
        if (m := _TOTAL_RE.search(seg)):
            row["total_ms"] = float(m.group(1))
        if (m := _SETUP_RE.search(seg)):
            row["setup_ms"] = float(m.group(1))
        if (m := _PREFILL_RE.search(seg)):
            row["prefill_ms"] = float(m.group(1))
            row["prefill_tps"] = float(m.group(2))
        if (m := _POST_RE.search(seg)):
            row["postprefill_ms"] = float(m.group(1))
        if (m := _SEMANTIC_RE.search(seg)):
            row["semantic_ms"] = float(m.group(1))
        if (m := _QUERY_REPLAY_RE.search(seg)):
            row["query_replay_ms"] = float(m.group(1))
        if (m := _POST_OTHER_RE.search(seg)):
            row["post_other_ms"] = float(m.group(1))
        if (m := _DECODE_RE.search(seg)):
            row["decode_probe_tokens"] = int(m.group(1))
            row["decode_ms"] = float(m.group(2))
            row["decode_tps"] = float(m.group(3))
            row["accept"] = float(m.group(4))
        if (m := _FINALIZE_RE.search(seg)):
            row["finalize_ms"] = float(m.group(1))
        if (m := _ACCOUNTING_ERROR_RE.search(seg)):
            row["accounting_error_ms"] = float(m.group(1))
        if (m := _S1_RE.search(seg)):
            row["sel_ms"] = float(m.group(1))
        if (m := _S2_RE.search(seg)):
            row["stage_in_ms"] = float(m.group(1))
            row["stage_in_blocks"] = int(m.group(2))
        if (m := _S3_RE.search(seg)):
            row["stage_out_ms"] = float(m.group(1))
            row["stage_out_blocks"] = int(m.group(2))
        if (m := _S4_RE.search(seg)):
            row["assemble_ms"] = float(m.group(1))
            row["asm_pages_ms"] = float(m.group(2))
            row["asm_rerope_ms"] = float(m.group(3))
            row["asm_kbar_ms"] = float(m.group(4))
        if (m := _S5B_RE.search(seg)):
            row["inprefill_offload_ms"] = float(m.group(1))
            row["inprefill_stage_in_blocks"] = int(m.group(2))
            row["inprefill_stage_out_blocks"] = int(m.group(3))
        top_sum = sum(float(row.get(k, 0.0)) for k in (
            "setup_ms", "prefill_ms", "postprefill_ms",
            "decode_ms", "finalize_ms"))
        row["phase_sum_ms"] = round(top_sum, 6)
        row["phase_error_ms"] = round(
            float(row.get("total_ms", 0.0)) - top_sum, 6)
        post_sum = sum(float(row.get(k, 0.0)) for k in (
            "semantic_ms", "query_replay_ms", "post_other_ms"))
        row["post_sum_ms"] = round(post_sum, 6)
        row["post_error_ms"] = round(
            float(row.get("postprefill_ms", 0.0)) - post_sum, 6)
        turns.append(row)
    return turns


def parse_summary(log: str) -> Dict[int, dict]:
    """Parse the final summary table rows (tier residency) keyed by turn."""
    out: Dict[int, dict] = {}
    after_hdr = False
    for ln in log.splitlines():
        if "kvmem-session SUMMARY" in ln:
            after_hdr = True
            continue
        if not after_hdr:
            continue
        m = _SUMROW_RE.match(ln)
        if not m:
            continue
        g = m.groups()
        out[int(g[0])] = {
            "summary_total_s": float(g[3]),
            "summary_setup_s": float(g[4]),
            "summary_prefill_s": float(g[5]),
            "summary_postprefill_s": float(g[6]),
            "summary_decode_s": float(g[7]),
            "summary_finalize_s": float(g[8]),
            "summary_error_ms": float(g[9]),
            "summary_prefill_tps": float(g[10]),
            "summary_decode_tps": float(g[11]),
            "kv_gib": float(g[13]),
            "gpu_gib": float(g[14]),
            "cpu_gib": float(g[15]),
            "nvme_gib": float(g[16]),
            "pool": int(g[17]),
            "summary_gpu_mib": int(g[18]),
            "summary_rss_mib": int(g[19]),
        }
    return out


def parse_repeated_queries(log: str) -> List[dict]:
    """Parse exact-milestone query probes emitted by kvmem-session."""
    out: List[dict] = []
    for match in _QUERY_ROW_RE.finditer(log):
        g = match.groups()
        out.append({
            "turn": int(g[0]),
            "query": int(g[1]),
            "mode": g[2],
            "base_pos": int(g[3]),
            "query_begin": int(g[4]),
            "query_end": int(g[5]),
            "final_pos": int(g[6]),
            "capture_ms": float(g[7]),
            "restore_ms": float(g[8]),
            "total_ms": float(g[9]),
            "semantic_ms": float(g[10]),
            "query_replay_ms": float(g[11]),
            "decode_ms": float(g[12]),
            "score_ms": float(g[13]),
            "stage_in_ms": float(g[14]),
            "stage_out_ms": float(g[15]),
            "assemble_ms": float(g[16]),
            "stage_in_blocks": int(g[17]),
            "stage_out_blocks": int(g[18]),
            "decoded": int(g[19]),
        })
    return out


def fmt_ladder(targets: List[int]) -> str:
    def one(v: int) -> str:
        if v % (1024 * 1024) == 0:
            return f"{v // (1024 * 1024)}M"
        if v % 1024 == 0:
            return f"{v // 1024}K"
        return str(v)
    return ",".join(one(v) for v in targets)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ladder", default="256K,512K,1M,1.5M,2M",
                    help="comma-separated cumulative ctx targets (K/M suffixes)")
    ap.add_argument("--decode-tokens", type=int, default=256,
                    help="MTP decode probe length per turn")
    ap.add_argument("--query-tokens", type=int, default=32,
                    help="tail tokens used as query per ladder point; 0 "
                         "disables query-conditioned selection/replay")
    ap.add_argument("--repeat-queries", type=int, default=0,
                    help="queries issued from every exact ladder checkpoint; "
                         "0 keeps the legacy path")
    ap.add_argument("--repeat-mode", choices=("frozen", "sequential"),
                    default="frozen",
                    help="restore the same milestone before every query or "
                         "preserve earlier probe turns")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="decode-probe temperature (0=greedy, default). temp>0 "
                         "routes the Qwen3 sampled recipe (top_p=0.95 top_k=20)")
    ap.add_argument("--chain", type=int, default=4, help="MTP chain length")
    ap.add_argument("--window", type=int, default=32768,
                    help="retrieval window in tokens (--kvmem-budget)")
    ap.add_argument("--gen-budget", type=int, default=32768)
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--method", default="retrieval")
    ap.add_argument("--retrieval-method", default="mean-k",
                    choices=("mean-k", "sub-block-mean-k",
                             "key-direction-fixed4",
                             "key-direction-adaptive"))
    ap.add_argument("--subblocks", type=int, default=1)
    ap.add_argument("--adaptive-gain-1to2", type=float, default=0.10)
    ap.add_argument("--adaptive-gain-2to4", type=float, default=0.06)
    ap.add_argument("--index-placement", choices=("gpu", "cpu"),
                    default="gpu")
    ap.add_argument("--index-staging-mb", type=int, default=64)
    ap.add_argument("--gpu-ratio", type=float, default=0.5,
                    help="GPU KV-pool cap as a fraction of total device memory")
    ap.add_argument("--cpu-gb", type=float, default=64.0)
    ap.add_argument("--nvme-gb", type=float, default=96.0)
    ap.add_argument("--nvme-dir", default="",
                    help="NVMe tier directory (default: a tmpdir, cleaned up)")
    ap.add_argument("--kv-dtype", default="fp16")
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--opt-stage-out", choices=("on", "off"), default="on")
    ap.add_argument("--opt-stage-in", choices=("on", "off"), default="on")
    ap.add_argument("--opt-pack", choices=("on", "off"), default="on")
    ap.add_argument("--raw-k-nvme", action="store_true",
                    help="store immutable raw-K authority in NVMe with a "
                         "bounded CPU cache")
    ap.add_argument("--no-immutable-k", action="store_true")
    ap.add_argument("--tier-threshold", type=int, default=1048576,
                    help="max ladder target at/above which CPU+NVMe tiers are "
                         "allocated (below this, GPU-pool-only -- avoids a "
                         "wasted 64 GiB pinned-host alloc for smoke ladders)")
    ap.add_argument("--deterministic", action="store_true", default=True,
                    help="force NSPLIT=1 attention (bit-stable greedy)")
    ap.add_argument("--no-deterministic", dest="deterministic",
                    action="store_false")
    ap.add_argument("--timeout", type=int, default=14400,
                    help="overall process timeout in seconds (default 4h)")
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_session_profile.json")
    args = ap.parse_args(argv)
    if args.repeat_queries < 0:
        raise SystemExit("--repeat-queries must be >= 0")
    if args.repeat_queries > 0 and args.query_tokens <= 0:
        raise SystemExit("--repeat-queries requires --query-tokens > 0")

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    # Parse the ladder client-side too (so we can size tiers + report it); the
    # binary re-parses --session-ladder authoritatively.
    def parse_tok(s: str) -> int:
        s = s.strip()
        mult = 1
        if s and s[-1] in "kKmMgG":
            mult = {"k": 1024, "m": 1024**2, "g": 1024**3}[s[-1].lower()]
            s = s[:-1]
        return int(float(s) * mult)
    targets = [parse_tok(x) for x in args.ladder.split(",") if x.strip()]
    if not targets or any(targets[i] <= targets[i - 1]
                          for i in range(1, len(targets))):
        raise SystemExit(f"ladder must be strictly increasing: {args.ladder}")
    max_target = targets[-1]

    own_nvme = not args.nvme_dir
    nvme_dir = args.nvme_dir or tempfile.mkdtemp(prefix="qw3_session_nvme_")
    Path(nvme_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        str(qw3), "kvmem-session", "--model", str(model),
        "--ctx", str(max_target +
                     args.decode_tokens * len(targets) +
                     args.repeat_queries *
                     (args.query_tokens + args.decode_tokens) + 4096),
        "--session-ladder", fmt_ladder(targets),
        "--session-decode-tokens", str(args.decode_tokens),
        "--session-query-tokens", str(args.query_tokens),
        "--kv-dtype", args.kv_dtype,
        "--prefill-chunk", str(args.prefill_chunk),
        "--native-mtp-speculate", "--mtp-chain", str(args.chain),
        "--kvmem",
        "--kvmem-block-tokens", str(args.block_tokens),
        "--kvmem-budget", str(args.window),
        "--kvmem-gen-budget", str(args.gen_budget),
        "--kvmem-method", args.method,
        "--kvmem-retrieval-method", args.retrieval_method,
        "--kvmem-subblocks", str(args.subblocks),
        "--kvmem-adaptive-gain-1to2", str(args.adaptive_gain_1to2),
        "--kvmem-adaptive-gain-2to4", str(args.adaptive_gain_2to4),
        "--kvmem-index-placement", args.index_placement,
        "--kvmem-index-staging-mb", str(args.index_staging_mb),
        "--kvmem-gpu-memory-ratio", str(args.gpu_ratio),
    ]
    # Keep the default command line byte-compatible with older binaries and
    # already-running ablation queues. The new flags are emitted only when the
    # milestone fan-out is explicitly requested.
    if args.repeat_queries > 0:
        cmd += [
            "--session-repeat-queries", str(args.repeat_queries),
            "--session-repeat-mode", args.repeat_mode,
        ]
    cmd += ["--no-kvmem-immutable-k" if args.no_immutable_k
            else "--kvmem-immutable-k"]
    cmd += [
        "--kvmem-opt-stage-out", args.opt_stage_out,
        "--kvmem-opt-stage-in", args.opt_stage_in,
        "--kvmem-opt-pack", args.opt_pack,
    ]
    if args.temp > 0.0:
        cmd += ["--temp", str(args.temp)]
    tiered = max_target >= args.tier_threshold
    if tiered:
        cmd += ["--kvmem-cpu-gb", str(args.cpu_gb),
                "--kvmem-nvme-gb", str(args.nvme_gb),
                "--kvmem-nvme-dir", nvme_dir]
        if args.raw_k_nvme:
            cmd += ["--kvmem-raw-k-nvme"]

    env = os.environ.copy()
    env["QW3_KVMEM_TIMING"] = "1"
    env["QW3_KVMEM_PERF_TRACE"] = "1"
    # The Q8 model's production launchers use FP16 batch activations. BF16
    # output is not implemented by the CUDA RMSNorm batch kernel.
    env["QW3_Q8_BF16_MAIN"] = "0"
    if args.deterministic:
        env["QW3_FATTN_NSPLIT"] = "1"
        env["QW3_PREFILL_FA2_NSPLIT"] = "1"

    logf = tempfile.NamedTemporaryFile(
        mode="w+", prefix="qw3_kvmem_session_", suffix=".log", delete=False)

    print("=== kvmem context-growth profile (persistent process) ===",
          flush=True)
    print(f"model={model.name} ladder={fmt_ladder(targets)} "
          f"decode_tokens={args.decode_tokens} chain={args.chain} "
          f"query_tokens={args.query_tokens} window={args.window}tok "
          f"repeat_queries={args.repeat_queries} "
          f"repeat_mode={args.repeat_mode} "
          f"gen_budget={args.gen_budget} ratio={args.gpu_ratio} "
          f"kv={args.kv_dtype} retrieval={args.retrieval_method} "
          f"index={args.index_placement} "
          f"opt_stage_out={args.opt_stage_out} "
          f"opt_stage_in={args.opt_stage_in} "
          f"opt_pack={args.opt_pack}",
          flush=True)
    print(f"tiers={'cpu=%g/nvme=%g GiB' % (args.cpu_gb, args.nvme_gb) if tiered else 'GPU-pool-only'} "
          f"nvme_dir={nvme_dir if tiered else '-'}", flush=True)
    print(f"cmd: {' '.join(cmd)}", flush=True)
    print(f"log: {logf.name}\n", flush=True)

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, text=True, stdout=logf,
                            stderr=subprocess.STDOUT, env=env)
    sampler = ResourceSampler(proc.pid)
    sampler.start()
    timed_out = False
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        logf.flush()
        logf.close()

    wall_s = time.monotonic() - t0
    log = Path(logf.name).read_text(encoding="utf-8", errors="replace")
    rc = proc.returncode

    turns = parse_turns(log)
    repeated_queries = parse_repeated_queries(log)
    summary = parse_summary(log)
    for t in turns:
        s = summary.get(int(t["turn"]))
        if s:
            t.update(s)

    print(f"\n--- process exited rc={rc} wall={wall_s/60:.1f}min "
          f"timed_out={timed_out} turns_parsed={len(turns)} ---", flush=True)
    if not turns:
        tail = "\n".join(log.splitlines()[-40:])
        print("NO TURNS PARSED -- last 40 log lines:\n" + tail, flush=True)

    # Human summary table.
    if turns:
        print(f"\n{'turn':>4s} {'ctx':>9s} {'delta':>9s} {'total':>9s} "
              f"{'setup':>8s} {'prefill':>9s} {'post':>8s} "
              f"{'decode':>8s} {'final':>8s} {'err_ms':>8s} "
              f"{'semantic':>9s} {'qreplay':>8s} {'pre_t/s':>8s} "
              f"{'sel':>7s} {'in':>7s} {'out':>7s} {'asm':>7s} "
              f"{'KV':>6s} {'GPU':>6s} {'CPU':>6s} {'NVMe':>6s} {'GPUmib':>8s} "
              f"{'RSSmib':>8s}", flush=True)
        for t in turns:
            print(f"{t.get('turn',0):4d} {t.get('ctx_tokens',0):9d} "
                  f"{t.get('delta_tokens',0):9d} "
                  f"{t.get('total_ms',0)/1000.0:9.3f} "
                  f"{t.get('setup_ms',0)/1000.0:8.3f} "
                  f"{t.get('prefill_ms',0)/1000.0:9.3f} "
                  f"{t.get('postprefill_ms',0)/1000.0:8.3f} "
                  f"{t.get('decode_ms',0)/1000.0:8.3f} "
                  f"{t.get('finalize_ms',0)/1000.0:8.3f} "
                  f"{t.get('phase_error_ms',0):8.3f} "
                  f"{t.get('semantic_ms',0)/1000.0:9.3f} "
                  f"{t.get('query_replay_ms',0)/1000.0:8.3f} "
                  f"{t.get('prefill_tps',0):8.1f} "
                  f"{t.get('sel_ms',0):7.2f} "
                  f"{t.get('stage_in_ms',0):7.2f} {t.get('stage_out_ms',0):7.2f} "
                  f"{t.get('assemble_ms',0):7.2f} "
                  f"{t.get('kv_gib',0):6.2f} {t.get('gpu_gib',0):6.2f} "
                  f"{t.get('cpu_gib',0):6.2f} {t.get('nvme_gib',0):6.2f} "
                  f"{t.get('gpu_mib',0):8d} {t.get('rss_mib',0):8d}",
                  flush=True)

    if repeated_queries:
        print(f"\n{'turn':>4s} {'q':>3s} {'mode':>10s} {'base':>10s} "
              f"{'restore':>9s} {'total':>9s} {'semantic':>9s} "
              f"{'replay':>9s} {'score':>8s} {'stagein':>8s} "
              f"{'stageout':>9s} {'assemble':>9s}", flush=True)
        for q in repeated_queries:
            print(f"{q['turn']:4d} {q['query']:3d} {q['mode']:>10s} "
                  f"{q['base_pos']:10d} {q['restore_ms']:9.3f} "
                  f"{q['total_ms']:9.3f} {q['semantic_ms']:9.3f} "
                  f"{q['query_replay_ms']:9.3f} {q['score_ms']:8.3f} "
                  f"{q['stage_in_ms']:8.3f} {q['stage_out_ms']:9.3f} "
                  f"{q['assemble_ms']:9.3f}", flush=True)

    out = {
        "args": vars(args),
        "cmd": cmd,
        "nvme_dir": nvme_dir if tiered else None,
        "tiered": tiered,
        "ladder_tokens": targets,
        "wall_s": round(wall_s, 2),
        "returncode": rc,
        "timed_out": timed_out,
        "peak_gpu_proc_mib": sampler.peak_gpu,
        "peak_host_rss_mib": sampler.peak_rss,
        "log_path": logf.name,
        "turns": turns,
        "repeated_queries": repeated_queries,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out_json}", flush=True)
    print(f"peak_gpu_proc={sampler.peak_gpu}MiB "
          f"peak_rss={sampler.peak_rss}MiB", flush=True)

    if own_nvme:
        shutil.rmtree(nvme_dir, ignore_errors=True)

    ok = (rc == 0) and (not timed_out) and (len(turns) == len(targets))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
