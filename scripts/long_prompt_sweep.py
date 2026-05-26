#!/usr/bin/env python3
"""Long-prompt-only sweep benchmark.

Why this exists: short prompts (5-15 tokens) make prefill numbers dominated
by per-process startup overhead — both qw3 and llama.cpp pay fixed costs
(model load, CUDA context, kernel cache warmup, JIT) that swamp the
actual compute. The qw3 process even charges its first prefill with one-shot
lazy init for kernel selection. So short-prompt prefill comparisons read like
noise.

This script runs each (prompt-length, decode-length) cell with the same
prompt content, repeated to fill a target token budget, alternating qw3
and llama.cpp to control for thermal drift. Each cell runs N_TRIALS times
and reports min/median/max so we can see whether the gap is real or just
launch-time variance.

Usage:
    python3 scripts/long_prompt_sweep.py [--n-decode 64] [--ctx 8192]
                                          [--trials 3] [--prompt-tokens "512 1024 2048 4096"]

Defaults match the model loaded in qw3-inspect (32K ctx fine for 4096-token
prompts; we keep ctx at 8192 to avoid llama.cpp warning about over-allocation).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Prompt construction. Pick a passage with no special tokens / no JSON, repeat
# until we hit the target token count (estimated at ~0.75 tokens per word).

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
    """Build a prompt of approximately `target_tokens` tokens (Qwen tokenizer)."""
    # Calibrated empirically against Qwen 3.6 tokenizer: 1 repeat of _PASSAGE
    # plus the wrapper produces ~285 tokens, so each additional repeat is
    # ~270 tokens. Solve `285 + 270 * (n - 1) >= target` for n.
    base_tokens = 285
    per_repeat  = 270
    n_repeats = max(1, 1 + (target_tokens - base_tokens + per_repeat - 1) // per_repeat)
    base = _PASSAGE * n_repeats
    prompt = f"Summarize the following passage in one paragraph.\n\n{base}\n\nSummary:"
    return prompt


# ---------------------------------------------------------------------------
# Output parsing.

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
    error: str = ""


def run_qw3(args, prompt: str, env_extra: Optional[dict] = None) -> TrialResult:
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
        "-p", prompt,
    ]
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        return TrialResult(label="qw3", ok=False, error="timeout")
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return TrialResult(label="qw3", ok=False,
                           error=f"exit {proc.returncode}: {out.strip()[-200:]}")
    m = _QW3_LINE.search(out)
    if not m:
        return TrialResult(label="qw3", ok=False,
                           error=f"no parse; tail={out.strip()[-200:]}")
    return TrialResult(
        label="qw3", ok=True,
        prompt_tokens=int(m.group(1)),
        prefill_s=float(m.group(2)),
        prefill_tok_s=float(m.group(3)),
        decoded_tokens=int(m.group(4)),
        decode_s=float(m.group(5)),
        decode_tok_s=float(m.group(6)),
    )


def run_llama(args, prompt: str) -> TrialResult:
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
        "-p", prompt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              stdin=subprocess.DEVNULL,
                              timeout=args.timeout + 5.0)
    except subprocess.TimeoutExpired:
        return TrialResult(label="llama.cpp", ok=False, error="timeout")
    err = proc.stderr or ""
    mp = _LLAMA_PROMPT.search(err)
    me = _LLAMA_EVAL.search(err)
    if not (mp and me):
        return TrialResult(label="llama.cpp", ok=False,
                           error=f"exit {proc.returncode}: no perf lines")
    return TrialResult(
        label="llama.cpp", ok=True,
        prompt_tokens=int(mp.group(2)),
        prefill_s=float(mp.group(1)) / 1000.0,
        prefill_tok_s=_f(mp.group(3)),
        decoded_tokens=int(me.group(2)),
        decode_s=float(me.group(1)) / 1000.0,
        decode_tok_s=_f(me.group(3)),
    )


# ---------------------------------------------------------------------------
# Sweep driver.

@dataclass
class CellSummary:
    target_prompt_tokens: int
    actual_prompt_tokens_qw3: int
    actual_prompt_tokens_llama: int
    qw3_prefill_med: float
    qw3_decode_med: float
    llama_prefill_med: float
    llama_decode_med: float
    qw3_prefill_min: float
    qw3_decode_min: float
    llama_prefill_min: float
    llama_decode_min: float
    qw3_trials: List[TrialResult]
    llama_trials: List[TrialResult]


def _med(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x > 0]
    return statistics.median(xs) if xs else 0.0


def _min_pos(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x > 0]
    return min(xs) if xs else 0.0


def run_cell(args, target_tokens: int) -> CellSummary:
    prompt = make_prompt(target_tokens)
    qw3_trials: List[TrialResult] = []
    llama_trials: List[TrialResult] = []

    print(f"\n[cell] target_prompt_tokens={target_tokens}, n_decode={args.n_decode}, "
          f"trials={args.trials}", flush=True)

    for t in range(args.trials):
        # Alternate llama / qw3 each trial to spread thermal effects.
        order = ("llama", "qw3") if t % 2 == 0 else ("qw3", "llama")
        for which in order:
            if which == "qw3":
                r = run_qw3(args, prompt)
                qw3_trials.append(r)
                if r.ok:
                    print(f"  [trial {t}] qw3      "
                          f"prompt={r.prompt_tokens:5d}  "
                          f"prefill {r.prefill_tok_s:8.1f} tok/s  "
                          f"decode {r.decode_tok_s:6.2f} tok/s", flush=True)
                else:
                    print(f"  [trial {t}] qw3 FAILED: {r.error}", flush=True)
            else:
                r = run_llama(args, prompt)
                llama_trials.append(r)
                if r.ok:
                    print(f"  [trial {t}] llama.cpp"
                          f" prompt={r.prompt_tokens:5d}  "
                          f"prefill {r.prefill_tok_s:8.1f} tok/s  "
                          f"decode {r.decode_tok_s:6.2f} tok/s", flush=True)
                else:
                    print(f"  [trial {t}] llama.cpp FAILED: {r.error}", flush=True)

    return CellSummary(
        target_prompt_tokens=target_tokens,
        actual_prompt_tokens_qw3=qw3_trials[0].prompt_tokens if qw3_trials and qw3_trials[0].ok else 0,
        actual_prompt_tokens_llama=llama_trials[0].prompt_tokens if llama_trials and llama_trials[0].ok else 0,
        qw3_prefill_med=_med(t.prefill_tok_s for t in qw3_trials if t.ok),
        qw3_decode_med=_med(t.decode_tok_s for t in qw3_trials if t.ok),
        llama_prefill_med=_med(t.prefill_tok_s for t in llama_trials if t.ok),
        llama_decode_med=_med(t.decode_tok_s for t in llama_trials if t.ok),
        qw3_prefill_min=_min_pos(t.prefill_tok_s for t in qw3_trials if t.ok),
        qw3_decode_min=_min_pos(t.decode_tok_s for t in qw3_trials if t.ok),
        llama_prefill_min=_min_pos(t.prefill_tok_s for t in llama_trials if t.ok),
        llama_decode_min=_min_pos(t.decode_tok_s for t in llama_trials if t.ok),
        qw3_trials=qw3_trials,
        llama_trials=llama_trials,
    )


def print_table(cells: List[CellSummary]) -> None:
    print()
    print("=" * 100)
    print("LONG-PROMPT SWEEP — median across trials, tok/s")
    print("=" * 100)
    print(f"{'prompt':>8}  | {'qw3 prefill':>12}  {'llama prefill':>14}  {'pref Δ':>10}  "
          f"| {'qw3 decode':>11}  {'llama decode':>13}  {'dec Δ':>8}")
    print(f"{'tokens':>8}  | {'tok/s':>12}  {'tok/s':>14}  {'tok/s':>10}  "
          f"| {'tok/s':>11}  {'tok/s':>13}  {'tok/s':>8}")
    print("-" * 100)
    for c in cells:
        pref_d = c.qw3_prefill_med - c.llama_prefill_med
        dec_d  = c.qw3_decode_med  - c.llama_decode_med
        actual = c.actual_prompt_tokens_qw3 or c.actual_prompt_tokens_llama
        print(f"{actual:>8}  | "
              f"{c.qw3_prefill_med:>12.1f}  {c.llama_prefill_med:>14.1f}  {pref_d:>+10.1f}  "
              f"| {c.qw3_decode_med:>11.2f}  {c.llama_decode_med:>13.2f}  {dec_d:>+8.2f}")
    print("=" * 100)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--qw3", default=os.environ.get("QW3", "./build/qw3"))
    p.add_argument("--llama", default=os.environ.get(
        "LLAMA_CLI", "/tmp/llama.cpp/build-cuda/bin/llama-completion"))
    p.add_argument("--model", default=os.environ.get(
        "MODEL",
        "/home/chaidi/AgentSys/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q8_0.gguf"))
    p.add_argument("-n", "--n-decode", type=int, default=64)
    p.add_argument("-c", "--ctx", type=int, default=8192)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--prompt-tokens", type=str, default="512 1024 2048 4096",
                   help="space- or comma-separated target prompt token counts")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--json", type=str, default=None,
                   help="optional path to dump per-trial results")
    args = p.parse_args()

    targets = [int(x) for x in re.split(r"[,\s]+", args.prompt_tokens) if x]

    print(f"qw3        : {args.qw3}")
    print(f"llama.cpp  : {args.llama}")
    print(f"model      : {args.model}")
    print(f"ctx={args.ctx}  n_decode={args.n_decode}  trials={args.trials}")
    print(f"prompt-tokens: {targets}")

    cells: List[CellSummary] = []
    for t in targets:
        cells.append(run_cell(args, t))

    print_table(cells)

    if args.json:
        Path(args.json).write_text(json.dumps([asdict(c) for c in cells], indent=2,
                                              default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o)))
        print(f"wrote per-trial JSON to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
