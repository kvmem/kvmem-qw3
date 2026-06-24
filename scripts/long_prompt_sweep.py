#!/usr/bin/env python3
"""Long-prompt sweep benchmark: qw3 vs llama.cpp.

Alternates qw3 and llama.cpp trial-by-trial to spread thermal drift. Each
(prompt-length, decode-length) cell runs N trials and reports median tok/s.

Usage:
    python3 scripts/long_prompt_sweep.py \\
        --model models/Qwen3.6-27B-Q8_0.gguf \\
        --llama /path/to/llama-completion \\
        --prompt-tokens "4096 8192 16384 32768 65536 131072" \\
        --trials 3 -n 512 --prefill-chunk 2048 \\
        --json /tmp/sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


_PASSAGE = (
    "The book of the generation of Jesus Christ, the son of David, the son of "
    "Abraham. Abraham begat Isaac; and Isaac begat Jacob; and Jacob begat Judas "
    "and his brethren; And Judas begat Phares and Zara of Thamar; and Phares "
    "begat Esrom; and Esrom begat Aram; And Aram begat Aminadab; and Aminadab "
    "begat Naasson; and Naasson begat Salmon; And Salmon begat Booz of Rachab; "
    "and Booz begat Obed of Ruth; and Obed begat Jesse; And Jesse begat David "
    "the king; and David the king begat Solomon of her that had been the wife "
    "of Urias; And Solomon begat Roboam; and Roboam begat Abia; and Abia begat "
    "Asa; And Asa begat Josaphat; and Josaphat begat Joram; and Joram begat "
    "Ozias; And Ozias begat Joatham; and Joatham begat Achaz; and Achaz begat "
    "Ezekias; And Ezekias begat Manasses; and Manasses begat Amon; and Amon "
    "begat Josias; And Josias begat Jechonias and his brethren, about the time "
    "they were carried away to Babylon. "
)


def make_prompt(target_tokens: int) -> str:
    base_tokens = 285
    per_repeat = 270
    n_repeats = max(1, 1 + (target_tokens - base_tokens + per_repeat - 1) // per_repeat)
    base = _PASSAGE * n_repeats
    return f"Summarize the following passage in one paragraph.\n\n{base}\n\nSummary:"


def ctx_for_target(target_tokens: int, n_decode: int, min_ctx: int) -> int:
    need = target_tokens + n_decode + 1024
    if need <= min_ctx:
        return min_ctx
    ctx = 1
    while ctx < need:
        ctx *= 2
    return ctx


_QW3_LINE = re.compile(
    r"\[qw3\] native generate: prompt_tokens=(\d+) "
    r"prefill=([0-9.]+)s \(([0-9.]+) tok/s\) "
    r"decoded=(\d+) decode=([0-9.]+)s \(([0-9.]+) tok/s\)"
)
_TOK_S = r"([0-9.]+|inf|nan)"
_LLAMA_PROMPT = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens?\s*\(\s*[0-9.]+\s*ms per token,\s*"
    + _TOK_S + r"\s*tokens per second\)"
)
_LLAMA_EVAL = re.compile(
    r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs?\s*\(\s*[0-9.]+\s*ms per token,\s*"
    + _TOK_S + r"\s*tokens per second\)"
)


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


@dataclass
class TrialResult:
    label: str
    ok: bool
    prompt_tokens: int = 0
    decoded_tokens: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    peak_mib: float = 0.0
    error: str = ""


@dataclass
class CellSummary:
    variant: str
    target_prompt_tokens: int
    actual_prompt_tokens_qw3: int
    actual_prompt_tokens_llama: int
    qw3_prefill_med: float
    qw3_decode_med: float
    llama_prefill_med: float
    llama_decode_med: float
    qw3_peak_mib_med: float
    qw3_prefill_min: float
    qw3_decode_min: float
    llama_prefill_min: float
    llama_decode_min: float
    qw3_trials: List[TrialResult] = field(default_factory=list)
    llama_trials: List[TrialResult] = field(default_factory=list)


class PeakMemSampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._baseline_mib = 0.0

    def _read_mib(self) -> float:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 0.0
        values = [float(x.strip()) for x in proc.stdout.splitlines() if x.strip()]
        return max(values) if values else 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            used = self._read_mib()
            net = max(0.0, used - self._baseline_mib)
            if net > self.peak_mib:
                self.peak_mib = net
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self.peak_mib = 0.0
        self._baseline_mib = self._read_mib()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_mib


def run_qw3(
    args: argparse.Namespace,
    prompt: str,
    *,
    ctx_size: int,
    extra_args: Sequence[str],
    sample_peak: bool = False,
) -> TrialResult:
    pf = Path("/tmp/qw3_sweep_prompt.txt")
    pf.write_text(prompt)
    cmd = [
        args.qw3,
        "--backend", "qwen-native",
        "--native-heavy",
        "--native-kernels", "cuda",
        "--native-linear-backend", "auto",
        "--model", args.model,
        "--raw",
        "-c", str(ctx_size),
        "-n", str(args.n_decode),
        "--temp", "0",
        "--seed", "0",
        "--prompt-file", str(pf),
    ]
    if args.prefill_chunk is not None:
        if args.prefill_chunk == 0:
            cmd.append("--no-prefill-chunk")
        else:
            cmd.extend(["--prefill-chunk", str(args.prefill_chunk)])
    cmd.extend(extra_args)
    sampler = PeakMemSampler() if sample_peak else None
    if sampler is not None:
        sampler.start()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        return TrialResult(label="qw3", ok=False, error="timeout")
    finally:
        peak = sampler.stop() if sampler is not None else 0.0
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return TrialResult(
            label="qw3",
            ok=False,
            error=f"exit {proc.returncode}: {out.strip()[-200:]}",
            peak_mib=peak,
        )
    m = _QW3_LINE.search(out)
    if not m:
        return TrialResult(
            label="qw3",
            ok=False,
            error=f"no parse; tail={out.strip()[-200:]}",
            peak_mib=peak,
        )
    return TrialResult(
        label="qw3",
        ok=True,
        prompt_tokens=int(m.group(1)),
        prefill_s=float(m.group(2)),
        prefill_tok_s=float(m.group(3)),
        decoded_tokens=int(m.group(4)),
        decode_s=float(m.group(5)),
        decode_tok_s=float(m.group(6)),
        peak_mib=peak,
    )


def run_llama(args: argparse.Namespace, prompt: str, *, ctx_size: int) -> TrialResult:
    pf = Path("/tmp/qw3_sweep_prompt.txt")
    pf.write_text(prompt)
    cmd = [
        "timeout",
        str(int(args.timeout)),
        args.llama,
        "-m",
        args.model,
        "-ngl",
        "99",
        "-c",
        str(ctx_size),
        "-n",
        str(args.n_decode),
        "-no-cnv",
        "--simple-io",
        "--no-display-prompt",
        "--temp",
        "0",
        "--seed",
        "0",
        "--no-warmup",
        "-f",
        str(pf),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=args.timeout + 5.0,
        )
    except subprocess.TimeoutExpired:
        return TrialResult(label="llama.cpp", ok=False, error="timeout")
    err = proc.stderr or ""
    mp = _LLAMA_PROMPT.search(err)
    me = _LLAMA_EVAL.search(err)
    if not (mp and me):
        return TrialResult(
            label="llama.cpp",
            ok=False,
            error=f"exit {proc.returncode}: no perf lines",
        )
    return TrialResult(
        label="llama.cpp",
        ok=True,
        prompt_tokens=int(mp.group(2)),
        prefill_s=float(mp.group(1)) / 1000.0,
        prefill_tok_s=_f(mp.group(3)),
        decoded_tokens=int(me.group(2)),
        decode_s=float(me.group(1)) / 1000.0,
        decode_tok_s=_f(me.group(3)),
    )


def _med(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x > 0]
    return statistics.median(vals) if vals else 0.0


def _min_pos(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x > 0]
    return min(vals) if vals else 0.0


def run_cell(
    args: argparse.Namespace,
    target_tokens: int,
    *,
    variant: str,
    extra_args: Sequence[str],
) -> CellSummary:
    prompt = make_prompt(target_tokens)
    ctx_size = ctx_for_target(target_tokens, args.n_decode, args.ctx)
    qw3_trials: List[TrialResult] = []
    llama_trials: List[TrialResult] = []

    print(
        f"\n[cell] variant={variant} target_prompt_tokens={target_tokens} "
        f"ctx={ctx_size} n_decode={args.n_decode} trials={args.trials}",
        flush=True,
    )

    for t in range(args.trials):
        order = ("llama", "qw3") if t % 2 == 0 else ("qw3", "llama")
        for which in order:
            if which == "qw3":
                r = run_qw3(
                    args,
                    prompt,
                    ctx_size=ctx_size,
                    extra_args=extra_args,
                    sample_peak=args.sample_peak,
                )
                qw3_trials.append(r)
                if r.ok:
                    print(
                        f"  [trial {t}] qw3       prompt={r.prompt_tokens:6d}  "
                        f"prefill {r.prefill_tok_s:8.1f} tok/s  "
                        f"decode {r.decode_tok_s:6.2f} tok/s  "
                        f"peak {r.peak_mib:.1f} MiB",
                        flush=True,
                    )
                else:
                    print(f"  [trial {t}] qw3 FAILED: {r.error}", flush=True)
            else:
                r = run_llama(args, prompt, ctx_size=ctx_size)
                llama_trials.append(r)
                if r.ok:
                    print(
                        f"  [trial {t}] llama.cpp prompt={r.prompt_tokens:6d}  "
                        f"prefill {r.prefill_tok_s:8.1f} tok/s  "
                        f"decode {r.decode_tok_s:6.2f} tok/s",
                        flush=True,
                    )
                else:
                    print(f"  [trial {t}] llama.cpp FAILED: {r.error}", flush=True)

    return CellSummary(
        variant=variant,
        target_prompt_tokens=target_tokens,
        actual_prompt_tokens_qw3=qw3_trials[0].prompt_tokens if qw3_trials and qw3_trials[0].ok else 0,
        actual_prompt_tokens_llama=llama_trials[0].prompt_tokens if llama_trials and llama_trials[0].ok else 0,
        qw3_prefill_med=_med(t.prefill_tok_s for t in qw3_trials if t.ok),
        qw3_decode_med=_med(t.decode_tok_s for t in qw3_trials if t.ok),
        llama_prefill_med=_med(t.prefill_tok_s for t in llama_trials if t.ok),
        llama_decode_med=_med(t.decode_tok_s for t in llama_trials if t.ok),
        qw3_peak_mib_med=_med(t.peak_mib for t in qw3_trials if t.ok),
        qw3_prefill_min=_min_pos(t.prefill_tok_s for t in qw3_trials if t.ok),
        qw3_decode_min=_min_pos(t.decode_tok_s for t in qw3_trials if t.ok),
        llama_prefill_min=_min_pos(t.prefill_tok_s for t in llama_trials if t.ok),
        llama_decode_min=_min_pos(t.decode_tok_s for t in llama_trials if t.ok),
        qw3_trials=qw3_trials,
        llama_trials=llama_trials,
    )


def print_table(cells: List[CellSummary]) -> None:
    print()
    print("=" * 110)
    print("LONG-PROMPT SWEEP — median across trials, tok/s")
    print("=" * 110)
    print(
        f"{'variant':>10}  {'prompt':>8}  | {'qw3 prefill':>12}  {'llama prefill':>14}  "
        f"{'pref %':>8}  | {'qw3 decode':>11}  {'llama decode':>13}  {'dec %':>8}"
    )
    print("-" * 110)
    for c in cells:
        actual = c.actual_prompt_tokens_qw3 or c.actual_prompt_tokens_llama
        pref_pct = (100.0 * c.qw3_prefill_med / c.llama_prefill_med) if c.llama_prefill_med > 0 else 0.0
        dec_pct = (100.0 * c.qw3_decode_med / c.llama_decode_med) if c.llama_decode_med > 0 else 0.0
        print(
            f"{c.variant:>10}  {actual:>8}  | "
            f"{c.qw3_prefill_med:>12.1f}  {c.llama_prefill_med:>14.1f}  {pref_pct:>7.1f}%  | "
            f"{c.qw3_decode_med:>11.2f}  {c.llama_decode_med:>13.2f}  {dec_pct:>7.1f}%"
        )
    print("=" * 110)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qw3", default=os.environ.get("QW3", "./build/qw3"))
    p.add_argument(
        "--llama",
        default=os.environ.get("LLAMA_COMPLETION", os.environ.get("LLAMA_CLI", "")),
    )
    p.add_argument("--model", default=os.environ.get("MODEL", ""))
    p.add_argument("-n", "--n-decode", type=int, default=512)
    p.add_argument("-c", "--ctx", type=int, default=36864)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument(
        "--prompt-tokens",
        type=str,
        default="4096 8192 16384 32768 65536 131072",
    )
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--prefill-chunk", type=int, default=None)
    p.add_argument("--paged-kv", action="store_true")
    p.add_argument("--mtp-chain", type=int, default=0)
    p.add_argument("--variant", default="default")
    p.add_argument("--sample-peak", action="store_true")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--extra-arg", action="append", default=[])
    args = p.parse_args(argv)

    if not args.model:
        raise SystemExit("--model is required (or set MODEL in the environment)")
    if not args.llama:
        raise SystemExit("--llama is required (or set LLAMA_COMPLETION in the environment)")
    if not Path(args.qw3).exists():
        raise SystemExit(f"qw3 binary not found: {args.qw3}")
    if not Path(args.llama).exists():
        raise SystemExit(f"llama-completion not found: {args.llama}")
    if not Path(args.model).exists():
        raise SystemExit(f"model not found: {args.model}")

    extra_args = list(args.extra_arg)
    if args.paged_kv:
        extra_args.append("--paged-kv")
    if args.mtp_chain > 0:
        extra_args.extend(["--native-mtp-speculate", "--mtp-chain", str(args.mtp_chain)])

    targets = [int(x) for x in re.split(r"[,\s]+", args.prompt_tokens) if x]

    print(f"qw3        : {args.qw3}")
    print(f"llama.cpp  : {args.llama}")
    print(f"model      : {args.model}")
    print(f"variant    : {args.variant}")
    print(f"ctx>={args.ctx}  n_decode={args.n_decode}  trials={args.trials}")
    print(f"prompt-tokens: {targets}")

    cells: List[CellSummary] = []
    failures = 0
    for target in targets:
        cell = run_cell(args, target, variant=args.variant, extra_args=extra_args)
        cells.append(cell)
        if not all(t.ok for t in cell.qw3_trials) or not all(t.ok for t in cell.llama_trials):
            failures += 1

    print_table(cells)

    payload = {
        "config": {
            "qw3": args.qw3,
            "llama": args.llama,
            "model": args.model,
            "variant": args.variant,
            "ctx_min": args.ctx,
            "n_decode": args.n_decode,
            "trials": args.trials,
            "prompt_tokens": targets,
            "prefill_chunk": args.prefill_chunk,
            "extra_args": extra_args,
        },
        "cells": [asdict(c) for c in cells],
        "status": {"failed_cells": failures},
    }
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote JSON to {out_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
