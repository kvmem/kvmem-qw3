#!/usr/bin/env python3
"""Three-way sweep: qw3 default (mma-gqa-v2) vs qw3+flashinfer vs llama.cpp.

Mirrors the README methodology: same prompts as long_prompt_sweep.py /
memory_sweep.py, 3 trials per cell (alternating order to spread thermal
drift), median of prefill/decode tok/s, peak VRAM polled at 50 ms.

Output: README-style markdown table.
"""
from __future__ import annotations
import argparse, json, os, re, statistics, subprocess, sys, threading, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from long_prompt_sweep import make_prompt, _QW3_LINE, _LLAMA_PROMPT, _LLAMA_EVAL  # noqa: E402


def _smi_used_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            text=True, timeout=2.0)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class Poller(threading.Thread):
    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_evt = threading.Event()
        self.peak_mib = 0

    def run(self) -> None:
        while not self._stop_evt.is_set():
            v = _smi_used_mib()
            if v is not None and v > self.peak_mib:
                self.peak_mib = v
            self._stop_evt.wait(self.interval_s)

    def cancel(self) -> None:
        self._stop_evt.set()


def _wait_idle(thresh_mib: int = 1500, max_wait_s: float = 30.0) -> int:
    deadline = time.time() + max_wait_s
    last = _smi_used_mib() or 0
    while time.time() < deadline:
        cur = _smi_used_mib() or last
        if cur < thresh_mib:
            return cur
        last = cur
        time.sleep(0.5)
    return last


@dataclass
class Trial:
    label: str
    ok: bool
    prompt_tokens: int = 0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    peak_mib: int = 0
    error: str = ""


def _run_polled(cmd, env, timeout, dev_null_stdin=False):
    p = Poller()
    p.start()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=timeout,
                              stdin=subprocess.DEVNULL if dev_null_stdin else None)
    finally:
        p.cancel(); p.join(timeout=1.0)
    return proc, p.peak_mib


def run_qw3(args, prompt: str, fi: bool) -> Trial:
    pf = Path("/tmp/qw3_fi_bench_prompt.txt")
    pf.write_text(prompt)
    cmd = [
        args.qw3, "--backend", "qwen-native",
        "--native-heavy", "--native-kernels", "cuda",
        "--native-linear-backend", "auto",
        "--model", args.model, "--raw",
        "-c", str(args.ctx), "-n", str(args.n_decode),
        "--temp", "0", "--seed", "0",
        "--prompt-file", str(pf),
    ]
    env = os.environ.copy()
    label = "qw3-fi" if fi else "qw3-default"
    if fi:
        env["QW3_PREFILL_ATTN"] = "flashinfer"
    else:
        env.pop("QW3_PREFILL_ATTN", None)
    try:
        proc, peak = _run_polled(cmd, env, args.timeout)
    except subprocess.TimeoutExpired:
        return Trial(label=label, ok=False, error="timeout")
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return Trial(label=label, ok=False,
                     error=f"exit {proc.returncode}: {out.strip()[-200:]}")
    m = _QW3_LINE.search(out)
    if not m:
        return Trial(label=label, ok=False, error=f"no parse; tail={out.strip()[-200:]}")
    return Trial(label=label, ok=True,
                 prompt_tokens=int(m.group(1)),
                 prefill_tok_s=float(m.group(3)),
                 decode_tok_s=float(m.group(6)),
                 peak_mib=peak)


def run_llama(args, prompt: str) -> Trial:
    pf = Path("/tmp/qw3_fi_bench_prompt.txt")
    pf.write_text(prompt)
    cmd = ["timeout", str(int(args.timeout)), args.llama,
           "-m", args.model, "-ngl", "99",
           "-c", str(args.ctx), "-n", str(args.n_decode),
           "-no-cnv", "--simple-io", "--no-display-prompt",
           "--temp", "0", "--seed", "0", "--no-warmup",
           "-f", str(pf)]
    env = os.environ.copy()
    try:
        proc, peak = _run_polled(cmd, env, args.timeout + 5.0, dev_null_stdin=True)
    except subprocess.TimeoutExpired:
        return Trial(label="llama", ok=False, error="timeout")
    err = proc.stderr or ""
    mp = _LLAMA_PROMPT.search(err); me = _LLAMA_EVAL.search(err)
    if not (mp and me):
        return Trial(label="llama", ok=False,
                     error=f"exit {proc.returncode}: no perf lines")
    return Trial(label="llama", ok=True,
                 prompt_tokens=int(mp.group(2)),
                 prefill_tok_s=float(mp.group(3)),
                 decode_tok_s=float(me.group(3)),
                 peak_mib=peak)


def _med(xs):
    xs = [x for x in xs if x > 0]
    return statistics.median(xs) if xs else 0.0


def _max(xs):
    xs = [x for x in xs if x > 0]
    return max(xs) if xs else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qw3", default=os.environ.get("QW3", "./build/qw3"))
    p.add_argument("--llama", default=os.environ.get(
        "LLAMA_CLI", "/tmp/llama.cpp/build-cuda/bin/llama-completion"))
    p.add_argument("--model", default=os.environ.get(
        "MODEL", "/home/chaidi/qw3/Qwen3.6-27B-Q8_0.gguf"))
    p.add_argument("-n", "--n-decode", type=int, default=32)
    p.add_argument("-c", "--ctx", type=int, default=36864)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--prompt-tokens", type=str,
                   default="556 2182 4350 8415 16545 33076 65867 131073")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--json", default="/tmp/qw3_fi_bench.json")
    args = p.parse_args()

    targets = [int(x) for x in re.split(r"[,\s]+", args.prompt_tokens) if x]

    print(f"qw3        : {args.qw3}")
    print(f"llama.cpp  : {args.llama}")
    print(f"model      : {args.model}")
    print(f"trials={args.trials}  n_decode={args.n_decode}  ctx_default={args.ctx}")
    print(f"targets: {targets}")

    base = _wait_idle()
    print(f"baseline memory.used = {base} MiB\n")

    cells = []
    for T in targets:
        ctx = max(args.ctx, T + args.n_decode + 1024)
        ctx_orig = args.ctx
        args.ctx = ctx
        prompt = make_prompt(T)
        print(f"=== target T={T}  ctx={ctx} ===", flush=True)

        per_cell: dict = {"qw3-default": [], "qw3-fi": [], "llama": []}
        for t in range(args.trials):
            # rotate config order across trials to spread thermal drift
            order = [["qw3-default", "qw3-fi", "llama"],
                     ["qw3-fi", "llama", "qw3-default"],
                     ["llama", "qw3-default", "qw3-fi"]][t % 3]
            for cfg in order:
                _wait_idle(thresh_mib=base + 500)
                if cfg == "qw3-default":
                    r = run_qw3(args, prompt, fi=False)
                elif cfg == "qw3-fi":
                    r = run_qw3(args, prompt, fi=True)
                else:
                    r = run_llama(args, prompt)
                per_cell[cfg].append(r)
                if r.ok:
                    print(f"  [trial {t}] {cfg:<12} prompt={r.prompt_tokens:>6} "
                          f"prefill {r.prefill_tok_s:8.1f} tok/s  "
                          f"decode {r.decode_tok_s:6.2f} tok/s  "
                          f"peak {r.peak_mib/1024:.2f} GiB", flush=True)
                else:
                    print(f"  [trial {t}] {cfg:<12} FAIL: {r.error[:120]}", flush=True)

        def summarize(rs):
            ok = [r for r in rs if r.ok]
            if not ok:
                return None
            return dict(
                T=ok[0].prompt_tokens,
                prefill_med=_med(r.prefill_tok_s for r in ok),
                decode_med=_med(r.decode_tok_s for r in ok),
                peak_mib=_max(r.peak_mib for r in ok),
                n_ok=len(ok),
            )
        cells.append({
            "target_T": T,
            "ctx": ctx,
            "qw3-default": summarize(per_cell["qw3-default"]),
            "qw3-fi": summarize(per_cell["qw3-fi"]),
            "llama": summarize(per_cell["llama"]),
            "trials": {k: [asdict(t) for t in v] for k, v in per_cell.items()},
        })
        Path(args.json).write_text(json.dumps(cells, indent=2))
        print(f"  -> wrote partial {args.json}", flush=True)
        args.ctx = ctx_orig

    # README-style table.
    print()
    print("| Prompt tokens | qw3 prefill | qw3+FI prefill | FI Δ | llama prefill | "
          "qw3 decode | qw3+FI decode | llama decode | qw3 peak | qw3+FI peak | llama peak |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        d = c["qw3-default"]; f = c["qw3-fi"]; l = c["llama"]
        if not (d and f and l):
            print(f"| {c['target_T']} | (incomplete) |"); continue
        T = d["T"]
        fi_pct = (f["prefill_med"] / d["prefill_med"] - 1.0) * 100.0 if d["prefill_med"] else 0.0
        print(f"| {T:>6} | {d['prefill_med']:.0f} tok/s | {f['prefill_med']:.0f} tok/s | "
              f"{fi_pct:+.1f}% | {l['prefill_med']:.0f} tok/s | "
              f"{d['decode_med']:.2f} | {f['decode_med']:.2f} | {l['decode_med']:.2f} | "
              f"{d['peak_mib']/1024:.1f} GiB | {f['peak_mib']/1024:.1f} GiB | "
              f"{l['peak_mib']/1024:.1f} GiB |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
