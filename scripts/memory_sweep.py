#!/usr/bin/env python3
"""Per-prompt-length peak-memory sweep for qw3 vs llama.cpp.

For each prompt length, runs each engine once, polls `nvidia-smi
--query-gpu=memory.used` at ~50 ms while the child process runs, and reports
peak observed - baseline (idle GPU) memory in GiB.

Usage:
    python3 scripts/memory_sweep.py \
        --prompt-tokens "556 2182 4350 8415 16545 33076 65867 131073" \
        --json /tmp/mem_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

# Reuse prompt synthesis + parsers from long_prompt_sweep.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from long_prompt_sweep import (  # type: ignore  # noqa: E402
    make_prompt,
    _QW3_LINE,
    _LLAMA_PROMPT,
    _LLAMA_EVAL,
)


# ---------------------------------------------------------------------------
# nvidia-smi memory polling.

def _smi_query() -> Optional[int]:
    """Returns used memory in MiB, or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            text=True, timeout=2.0,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class MemoryPoller(threading.Thread):
    """Poll memory.used at fixed cadence; expose max observed."""
    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.peak_mib: int = 0
        self.samples: int = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            v = _smi_query()
            if v is not None:
                if v > self.peak_mib:
                    self.peak_mib = v
                self.samples += 1
            self._stop_evt.wait(self.interval_s)

    def stop(self) -> None:
        self._stop_evt.set()


# ---------------------------------------------------------------------------
# Engine runners that capture both throughput and peak memory.

@dataclass
class MemTrial:
    label: str
    ok: bool
    prompt_tokens: int = 0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    peak_mib: int = 0
    error: str = ""


def _run_with_polling(cmd: List[str], env: dict,
                      timeout: float,
                      stdin_dev_null: bool = False
                      ) -> Tuple[subprocess.CompletedProcess, int]:
    poller = MemoryPoller(interval_s=0.05)
    poller.start()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            env=env, timeout=timeout,
            stdin=subprocess.DEVNULL if stdin_dev_null else None,
        )
    finally:
        poller.stop()
        poller.join(timeout=1.0)
    return proc, poller.peak_mib


def run_qw3(args, prompt: str) -> MemTrial:
    pf = Path("/tmp/qw3_mem_prompt.txt")
    pf.write_text(prompt)
    cmd = [
        args.qw3,
        "--backend", "qwen-native",
        "--native-heavy",
        "--native-kernels", "cuda",
        "--native-linear-backend", "auto",
        "--model", args.model,
        "--raw",
        "-c", str(args.ctx),
        "-n", str(args.n_decode),
        "--temp", "0",
        "--seed", "0",
        "--prompt-file", str(pf),
    ]
    env = os.environ.copy()
    try:
        proc, peak = _run_with_polling(cmd, env, args.timeout)
    except subprocess.TimeoutExpired:
        return MemTrial(label="qw3", ok=False, error="timeout")
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return MemTrial(label="qw3", ok=False,
                        error=f"exit {proc.returncode}: {out.strip()[-200:]}")
    m = _QW3_LINE.search(out)
    if not m:
        return MemTrial(label="qw3", ok=False,
                        error=f"no parse; tail={out.strip()[-200:]}")
    return MemTrial(
        label="qw3", ok=True,
        prompt_tokens=int(m.group(1)),
        prefill_tok_s=float(m.group(3)),
        decode_tok_s=float(m.group(6)),
        peak_mib=peak,
    )


def run_llama(args, prompt: str) -> MemTrial:
    pf = Path("/tmp/qw3_mem_prompt.txt")
    pf.write_text(prompt)
    cmd = [
        "timeout", str(int(args.timeout)), args.llama,
        "-m", args.model,
        "-ngl", "99",
        "-c", str(args.ctx),
        "-n", str(args.n_decode),
        "-no-cnv",
        "--simple-io",
        "--no-display-prompt",
        "--temp", "0",
        "--seed", "0",
        "--no-warmup",
        "-f", str(pf),
    ]
    env = os.environ.copy()
    try:
        proc, peak = _run_with_polling(cmd, env, args.timeout + 5.0,
                                       stdin_dev_null=True)
    except subprocess.TimeoutExpired:
        return MemTrial(label="llama.cpp", ok=False, error="timeout")
    err = proc.stderr or ""
    mp = _LLAMA_PROMPT.search(err)
    me = _LLAMA_EVAL.search(err)
    if not (mp and me):
        return MemTrial(label="llama.cpp", ok=False,
                        error=f"exit {proc.returncode}: no perf lines")
    return MemTrial(
        label="llama.cpp", ok=True,
        prompt_tokens=int(mp.group(2)),
        prefill_tok_s=float(mp.group(3)),
        decode_tok_s=float(me.group(3)),
        peak_mib=peak,
    )


# ---------------------------------------------------------------------------
# Driver.

def _wait_for_idle(threshold_mib: int = 2000, max_wait_s: float = 30.0) -> int:
    """Wait until baseline memory drops near idle. Returns observed baseline."""
    deadline = time.time() + max_wait_s
    last = _smi_query() or 0
    while time.time() < deadline:
        cur = _smi_query() or last
        if cur < threshold_mib:
            return cur
        last = cur
        time.sleep(0.5)
    return last


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qw3", default=os.environ.get("QW3", "./build/qw3"))
    p.add_argument("--llama", default=os.environ.get(
        "LLAMA_CLI", "/tmp/llama.cpp/build-cuda/bin/llama-completion"))
    p.add_argument("--model", default=os.environ.get(
        "MODEL",
        "/home/chaidi/AgentSys/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q8_0.gguf"))
    p.add_argument("-n", "--n-decode", type=int, default=32)
    p.add_argument("-c", "--ctx", type=int, default=36864)
    p.add_argument("--prompt-tokens", type=str,
                   default="556 2182 4350 8415 16545 33076 65867 131073")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--json", type=str, default=None)
    args = p.parse_args()

    targets = [int(x) for x in re.split(r"[,\s]+", args.prompt_tokens) if x]

    print(f"qw3        : {args.qw3}")
    print(f"llama.cpp  : {args.llama}")
    print(f"model      : {args.model}")
    print(f"ctx={args.ctx}  n_decode={args.n_decode}")
    print(f"prompt-tokens: {targets}")

    # Establish baseline (other workloads, drivers, etc).
    base = _wait_for_idle()
    print(f"baseline memory.used = {base} MiB")

    cells = []
    for t in targets:
        # T=128K and beyond may need a bigger ctx.
        ctx_for_t = max(args.ctx, t + args.n_decode + 1024)
        ctx_orig = args.ctx
        args.ctx = ctx_for_t

        prompt = make_prompt(t)
        print(f"\n=== target T={t} (ctx={ctx_for_t}) ===")

        # Order: llama first, then qw3, with idle waits between.
        _wait_for_idle(threshold_mib=base + 500)
        ll = run_llama(args, prompt)
        if ll.ok:
            net = ll.peak_mib - base
            print(f"  llama: T={ll.prompt_tokens}  prefill={ll.prefill_tok_s:.0f} tok/s  "
                  f"decode={ll.decode_tok_s:.2f} tok/s  peak={ll.peak_mib} MiB "
                  f"(net {net} MiB = {net/1024:.2f} GiB)")
        else:
            print(f"  llama: FAILED: {ll.error}")

        _wait_for_idle(threshold_mib=base + 500)
        qw = run_qw3(args, prompt)
        if qw.ok:
            net = qw.peak_mib - base
            print(f"  qw3  : T={qw.prompt_tokens}  prefill={qw.prefill_tok_s:.0f} tok/s  "
                  f"decode={qw.decode_tok_s:.2f} tok/s  peak={qw.peak_mib} MiB "
                  f"(net {net} MiB = {net/1024:.2f} GiB)")
        else:
            print(f"  qw3  : FAILED: {qw.error}")

        cells.append({
            "target_T": t,
            "ctx": ctx_for_t,
            "baseline_mib": base,
            "qw3":   asdict(qw),
            "llama": asdict(ll),
        })
        args.ctx = ctx_orig

    print()
    print(f"{'T_qw3':>8} {'T_llama':>8}  "
          f"{'qw3 peak':>10} {'llama peak':>10}  {'qw3 net':>9} {'llama net':>11}")
    print("-" * 70)
    for c in cells:
        q = c["qw3"]; l = c["llama"]
        if not (q["ok"] and l["ok"]):
            print(f"{c['target_T']:>8}: incomplete")
            continue
        qn = (q["peak_mib"] - c["baseline_mib"]) / 1024.0
        ln = (l["peak_mib"] - c["baseline_mib"]) / 1024.0
        print(f"{q['prompt_tokens']:>8} {l['prompt_tokens']:>8}  "
              f"{q['peak_mib']/1024:>9.2f}G {l['peak_mib']/1024:>9.2f}G  "
              f"{qn:>8.2f}G {ln:>10.2f}G")

    if args.json:
        Path(args.json).write_text(json.dumps(cells, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
