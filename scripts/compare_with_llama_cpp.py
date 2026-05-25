#!/usr/bin/env python3
"""Side-by-side accuracy + efficiency check: qw3 vs llama.cpp.

For each prompt in the test set we run BOTH tools greedily (temp=0) and capture:
  - the generated text (for accuracy)
  - prefill / decode tokens-per-second (for efficiency)
  - prompt-token count

Accuracy metric: longest common prefix of the two generated texts, both as a
character count and as a fraction of the qw3 output length. With greedy
sampling, identical implementations should produce identical text; the first
character of divergence localizes a numerical bug.

Efficiency metric: prefill / decode tok/s for each tool, plus the qw3/llama.cpp
ratio so we can see how close we are to parity.

Designed to be invoked from the local Mac via:
    bash scripts/run_compare.sh
which `rsync`s the script up and executes it on the remote.

Usage on the remote box:
    python3 compare_with_llama_cpp.py [--n 32] [--ctx 4096]
                                       [--qw3 PATH] [--llama PATH] [--model PATH]
                                       [--prompts FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Hello! Briefly tell me what FlashAttention is.",
    "请简要解释什么是注意力机制",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""',
]


# ---------------------------------------------------------------------------
# Output parsing helpers


_QW3_LINE = re.compile(
    r"\[qw3\] native generate: prompt_tokens=(\d+) "
    r"prefill=([0-9.]+)s \(([0-9.]+) tok/s\) "
    r"decoded=(\d+) decode=([0-9.]+)s \(([0-9.]+) tok/s\)"
)

# llama-completion prints performance to stderr looking like:
#   common_perf_print: prompt eval time =     123.45 ms /     5 tokens (   24.69 ms per token,    40.51 tokens per second)
#   common_perf_print:        eval time =     456.78 ms /    16 runs   (   28.55 ms per token,    35.03 tokens per second)
# The tok/s number can be `inf` when the elapsed time rounds to 0.
_TOK_S = r"([0-9.]+|inf|nan)"
_LLAMA_PROMPT = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens?\s*\(\s*[0-9.]+\s*ms per token,\s*"
    + _TOK_S + r"\s*tokens per second\)"
)
_LLAMA_EVAL = re.compile(
    r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs?\s*\(\s*[0-9.]+\s*ms per token,\s*"
    + _TOK_S + r"\s*tokens per second\)"
)


def _to_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return 0.0


@dataclass
class RunResult:
    label: str
    ok: bool
    text: str
    prompt_tokens: int = 0
    decoded_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    prefill_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    wall_seconds: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# qw3 runner


def run_qw3(args, prompt: str) -> RunResult:
    """Invoke qw3 on a single prompt; return parsed metrics + generated text."""
    cmd = [
        args.qw3,
        "--backend", "qwen-native",
        "--native-heavy",
        "--native-kernels", "cuda",
        "--native-linear-backend", "auto",
        "--model", args.model,
        "--raw",
        "-c", str(args.ctx),
        "-n", str(args.n),
        "-p", prompt,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return RunResult(label="qw3", ok=False, text="", error="timeout")
    wall = time.time() - t0

    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        return RunResult(label="qw3", ok=False, text="", wall_seconds=wall,
                         error=f"exit {proc.returncode}: {out.strip()[-200:]}")

    m = _QW3_LINE.search(out)
    if not m:
        return RunResult(label="qw3", ok=False, text="", wall_seconds=wall,
                         error=f"could not parse timing line; tail={out.strip()[-200:]}")

    # qw3 prints the generated text inline before the timing line. Strip both
    # the load + timing lines to get just the model continuation.
    text = out
    text = re.sub(r"\[qw3\] native load:.*?\n", "", text)
    text = _QW3_LINE.sub("", text)
    text = text.rstrip("\n")

    return RunResult(
        label="qw3",
        ok=True,
        text=text,
        prompt_tokens=int(m.group(1)),
        prefill_seconds=float(m.group(2)),
        prefill_tok_s=float(m.group(3)),
        decoded_tokens=int(m.group(4)),
        decode_seconds=float(m.group(5)),
        decode_tok_s=float(m.group(6)),
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# llama.cpp runner


def run_llama(args, prompt: str) -> RunResult:
    """Invoke llama-completion (non-interactive) with greedy sampling.

    llama-cli auto-enables conversation/interactive mode when the model has a
    chat template, which causes the process to block on stdin after generation
    even with `-no-cnv`. llama-completion is the same engine without the
    interactive shell, so it exits cleanly. We still wrap it in `timeout`
    server-side as a belt-and-braces guard.
    """
    cmd = [
        "timeout", str(int(args.timeout)), args.llama,
        "-m", args.model,
        "-ngl", "99",
        "-c", str(args.ctx),
        "-n", str(args.n),
        "-no-cnv",                  # no chat template, raw completion mode
        "--simple-io",
        "--no-display-prompt",       # don't echo prompt back on stdout
        "--temp", "0",                # greedy
        "--seed", "0",
        "--no-warmup",
        "-p", prompt,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                              timeout=args.timeout + 5.0)
    except subprocess.TimeoutExpired:
        return RunResult(label="llama.cpp", ok=False, text="", error="timeout")
    wall = time.time() - t0

    err = proc.stderr or ""
    out = proc.stdout or ""

    # llama-completion exit code is non-zero when wrapped by `timeout` hitting
    # the wall; treat any non-zero with no parse-able timing as an error.
    mp = _LLAMA_PROMPT.search(err)
    me = _LLAMA_EVAL.search(err)

    if proc.returncode != 0 and not (mp and me):
        return RunResult(label="llama.cpp", ok=False, text="", wall_seconds=wall,
                         error=f"exit {proc.returncode}: {err.strip()[-300:] or out.strip()[-300:]}")

    prompt_tokens = int(mp.group(2)) if mp else 0
    prefill_seconds = (float(mp.group(1)) / 1000.0) if mp else 0.0
    prefill_tok_s = _to_float(mp.group(3)) if mp else 0.0

    decoded_tokens = int(me.group(2)) if me else 0
    decode_seconds = (float(me.group(1)) / 1000.0) if me else 0.0
    decode_tok_s = _to_float(me.group(3)) if me else 0.0

    # With --no-display-prompt the stdout is the continuation only.
    text = out.rstrip("\n")

    return RunResult(
        label="llama.cpp",
        ok=mp is not None and me is not None,
        text=text,
        prompt_tokens=prompt_tokens,
        prefill_seconds=prefill_seconds,
        prefill_tok_s=prefill_tok_s,
        decoded_tokens=decoded_tokens,
        decode_seconds=decode_seconds,
        decode_tok_s=decode_tok_s,
        wall_seconds=wall,
        error="" if (mp and me) else "could not parse common_perf_print lines",
    )


# ---------------------------------------------------------------------------
# Comparison


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def render_prompt(p: str, width: int = 64) -> str:
    s = p.replace("\n", "\\n")
    return s if len(s) <= width else s[: width - 1] + "…"


def render_text(t: str, width: int = 64) -> str:
    s = t.replace("\n", "\\n")
    return s if len(s) <= width else s[: width - 1] + "…"


def print_header(args):
    print("=" * 100)
    print(f"qw3 vs llama.cpp accuracy + efficiency comparison")
    print(f"model  : {args.model}")
    print(f"qw3    : {args.qw3}")
    print(f"llama  : {args.llama}")
    print(f"n_tokens={args.n}, ctx={args.ctx}, prompts={len(args.prompts)}")
    print("=" * 100)


def run_pair(args, prompt: str) -> dict:
    q = run_qw3(args, prompt)
    l = run_llama(args, prompt)

    cp_chars = common_prefix_len(q.text, l.text) if q.ok and l.ok else 0
    cp_frac = cp_chars / max(1, len(q.text)) if q.ok else 0.0

    print()
    print("-" * 100)
    print(f"PROMPT   : {render_prompt(prompt, 96)}")
    print("-" * 100)
    if q.ok:
        print(f"qw3      [{q.prompt_tokens}p / {q.decoded_tokens}d]  "
              f"prefill {q.prefill_tok_s:6.2f} tok/s  decode {q.decode_tok_s:6.2f} tok/s  "
              f"wall {q.wall_seconds:5.2f}s")
        print(f"  text  : {render_text(q.text, 90)}")
    else:
        print(f"qw3      ERROR: {q.error}")

    if l.ok:
        print(f"llama.cpp[{l.prompt_tokens}p / {l.decoded_tokens}d]  "
              f"prefill {l.prefill_tok_s:6.2f} tok/s  decode {l.decode_tok_s:6.2f} tok/s  "
              f"wall {l.wall_seconds:5.2f}s")
        print(f"  text  : {render_text(l.text, 90)}")
    else:
        print(f"llama.cpp ERROR: {l.error}")

    if q.ok and l.ok:
        prefill_ratio = (q.prefill_tok_s / l.prefill_tok_s) if l.prefill_tok_s > 0 else 0.0
        decode_ratio = (q.decode_tok_s / l.decode_tok_s) if l.decode_tok_s > 0 else 0.0
        same_first = (q.text[:1] == l.text[:1]) if q.text and l.text else False
        print(f"perf vs llama.cpp:  prefill {prefill_ratio*100:5.1f}%  decode {decode_ratio*100:5.1f}%")
        print(f"accuracy           :  common prefix = {cp_chars} chars ({cp_frac*100:5.1f}% of qw3 output), "
              f"first-char match = {same_first}")
        if cp_chars < min(len(q.text), len(l.text)):
            div = cp_chars
            qsnip = q.text[max(0, div - 10): div + 30].replace("\n", "\\n")
            lsnip = l.text[max(0, div - 10): div + 30].replace("\n", "\\n")
            marker = " " * min(10, div) + "^"
            print(f"divergence at char {div}:")
            print(f"  qw3   : ...{qsnip}")
            print(f"  llama : ...{lsnip}")
            print(f"          {marker}")

    return {
        "prompt": prompt,
        "qw3": asdict(q),
        "llama": asdict(l),
        "common_prefix_chars": cp_chars,
        "common_prefix_frac": cp_frac,
    }


def print_summary(rows: List[dict]):
    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    valid = [r for r in rows if r["qw3"]["ok"] and r["llama"]["ok"]]
    if not valid:
        print("no successful runs to summarize")
        return

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    qpref = mean([r["qw3"]["prefill_tok_s"] for r in valid])
    qdec = mean([r["qw3"]["decode_tok_s"] for r in valid])
    lpref = mean([r["llama"]["prefill_tok_s"] for r in valid])
    ldec = mean([r["llama"]["decode_tok_s"] for r in valid])
    cp = mean([r["common_prefix_chars"] for r in valid])
    cpf = mean([r["common_prefix_frac"] for r in valid])

    print(f"prompts compared        : {len(valid)} / {len(rows)}")
    print(f"qw3      avg prefill    : {qpref:6.2f} tok/s   decode: {qdec:6.2f} tok/s")
    print(f"llama.cpp avg prefill   : {lpref:6.2f} tok/s   decode: {ldec:6.2f} tok/s")
    if lpref > 0 and ldec > 0:
        print(f"qw3 / llama.cpp         : prefill {qpref/lpref*100:5.1f}%   decode {qdec/ldec*100:5.1f}%")
    print(f"avg common-prefix chars : {cp:6.1f}    ({cpf*100:5.1f}% of qw3 output)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qw3", default=os.environ.get("QW3", "./build-cuda/qw3"))
    parser.add_argument("--llama", default=os.environ.get(
        "LLAMA_CLI", "/home/chaidi/qw3/llama.cpp/build-cuda/bin/llama-completion"))
    parser.add_argument("--model", default=os.environ.get(
        "MODEL", "/home/chaidi/AgentSys/unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q8_0.gguf"))
    parser.add_argument("-n", "--n", type=int, default=32, help="tokens to generate")
    parser.add_argument("-c", "--ctx", type=int, default=4096, help="context size")
    parser.add_argument("--prompts", type=str, default=None,
                        help="file with one prompt per line (overrides defaults)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-invocation timeout (seconds)")
    parser.add_argument("--json", type=str, default=None,
                        help="write structured results to this JSON file")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts, "r") as f:
            args.prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
    else:
        args.prompts = list(DEFAULT_PROMPTS)

    print_header(args)
    rows = []
    for p in args.prompts:
        rows.append(run_pair(args, p))

    print_summary(rows)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"\nstructured results written to {args.json}")


if __name__ == "__main__":
    main()
