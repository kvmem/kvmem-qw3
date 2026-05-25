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

# Long-form prompt used for stable prefill timing. Short prompts (<32 tokens)
# are dominated by per-process warmup (cudaMalloc / kernel-JIT / lazy memo
# tables), so the reported prefill tok/s is noisy and pessimistic. With a
# ~1K-token prompt the warmup amortizes out and the matmul/attention compute
# dominates, so we can compare like-for-like with llama.cpp.
#
# Built from the King James Bible (Matthew 1, public domain) doubled to land
# at ~1000 tokens for the Qwen tokenizer.
_LONG_PARA = (
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
    "they were carried away to Babylon: And after they were brought to Babylon, "
    "Jechonias begat Salathiel; and Salathiel begat Zorobabel; And Zorobabel "
    "begat Abiud; and Abiud begat Eliakim; and Eliakim begat Azor; And Azor "
    "begat Sadoc; and Sadoc begat Achim; and Achim begat Eliud; And Eliud "
    "begat Eleazar; and Eleazar begat Matthan; and Matthan begat Jacob; And "
    "Jacob begat Joseph the husband of Mary, of whom was born Jesus, who is "
    "called Christ. So all the generations from Abraham to David are fourteen "
    "generations; and from David until the carrying away into Babylon are "
    "fourteen generations; and from the carrying away into Babylon unto Christ "
    "are fourteen generations. "
)
LONG_PROMPT = (
    "Summarize the following passage in one paragraph.\n\n"
    + _LONG_PARA * 3
    + "\n\nSummary:"
)


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


def common_token_prefix_len(a_tokens: List[int], b_tokens: List[int]) -> int:
    n = min(len(a_tokens), len(b_tokens))
    i = 0
    while i < n and a_tokens[i] == b_tokens[i]:
        i += 1
    return i


def tokenize_with_qw3(args, text: str) -> List[int]:
    """Use qw3 --dump-tokens to get token IDs for `text`. Returns [] on error."""
    cmd = [
        args.qw3,
        "--backend", "qwen-native",
        "--model", args.model,
        "--raw",
        "--dump-tokens",
        "-p", text,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120.0)
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode != 0:
        return []
    # qw3 --dump-tokens emits "tokens=N" then lines of the form
    #   <index>\t<id>\t"<escaped text>"
    # We want the second column (the token id).
    ids: List[int] = []
    for line in (proc.stdout + proc.stderr).splitlines():
        m = re.match(r"^\s*\d+\t(\d+)\t", line)
        if m:
            ids.append(int(m.group(1)))
    return ids


def render_prompt(p: str, width: int = 64) -> str:
    s = p.replace("\n", "\\n")
    return s if len(s) <= width else s[: width - 1] + "…"


def render_text(t: str, width: int = 64) -> str:
    s = t.replace("\n", "\\n")
    return s if len(s) <= width else s[: width - 1] + "…"


def print_header(args):
    print("=" * 100)
    print(f"qw3 vs llama.cpp accuracy + efficiency comparison  (greedy / argmax)")
    print(f"model  : {args.model}")
    print(f"qw3    : {args.qw3}  (always argmax)")
    print(f"llama  : {args.llama}  (--temp 0)")
    print(f"n_tokens={args.n}, ctx={args.ctx}, prompts={len(args.prompts)}")
    if args.token_diff:
        print(f"token-level diff: enabled  (re-tokenizing outputs via qw3 to count matching tokens)")
    print("note: with greedy + identical math, output token sequences should be identical")
    print("=" * 100)


def run_pair(args, prompt: str) -> dict:
    q = run_qw3(args, prompt)
    l = run_llama(args, prompt)

    cp_chars = common_prefix_len(q.text, l.text) if q.ok and l.ok else 0
    cp_frac = cp_chars / max(1, len(q.text)) if q.ok else 0.0

    # Token-level comparison: both engines greedy (qw3 always argmax; llama.cpp
    # gets --temp 0). If our kernels were bit-equivalent the token sequences
    # would match exactly. Number of matching tokens before divergence is a
    # much sharper "is the math right?" signal than character-level prefix.
    q_tokens: List[int] = []
    l_tokens: List[int] = []
    cp_tokens = 0
    cp_tok_frac = 0.0
    if args.token_diff and q.ok and l.ok:
        q_tokens = tokenize_with_qw3(args, q.text)
        l_tokens = tokenize_with_qw3(args, l.text)
        cp_tokens = common_token_prefix_len(q_tokens, l_tokens)
        if q_tokens:
            cp_tok_frac = cp_tokens / len(q_tokens)

    print()
    print("-" * 100)
    print(f"PROMPT   : {render_prompt(prompt, 96)}")
    print("-" * 100)
    if q.ok:
        print(f"qw3      [{q.prompt_tokens}p / {q.decoded_tokens}d]  "
              f"prefill {q.prefill_tok_s:7.2f} tok/s  decode {q.decode_tok_s:6.2f} tok/s  "
              f"wall {q.wall_seconds:5.2f}s")
        print(f"  text  : {render_text(q.text, 90)}")
    else:
        print(f"qw3      ERROR: {q.error}")

    if l.ok:
        print(f"llama.cpp[{l.prompt_tokens}p / {l.decoded_tokens}d]  "
              f"prefill {l.prefill_tok_s:7.2f} tok/s  decode {l.decode_tok_s:6.2f} tok/s  "
              f"wall {l.wall_seconds:5.2f}s")
        print(f"  text  : {render_text(l.text, 90)}")
    else:
        print(f"llama.cpp ERROR: {l.error}")

    if q.ok and l.ok:
        prefill_ratio = (q.prefill_tok_s / l.prefill_tok_s) if l.prefill_tok_s > 0 else 0.0
        decode_ratio = (q.decode_tok_s / l.decode_tok_s) if l.decode_tok_s > 0 else 0.0
        same_first = (q.text[:1] == l.text[:1]) if q.text and l.text else False
        print(f"perf vs llama.cpp:  prefill {prefill_ratio*100:5.1f}%  decode {decode_ratio*100:5.1f}%")
        print(f"accuracy (chars) :  common prefix = {cp_chars} chars ({cp_frac*100:5.1f}% of qw3 output), "
              f"first-char match = {same_first}")
        if args.token_diff:
            ident = (cp_tokens == len(q_tokens) == len(l_tokens))
            print(f"accuracy (tokens):  common prefix = {cp_tokens}/{len(q_tokens)} qw3 tokens "
                  f"({cp_tok_frac*100:5.1f}%), {len(l_tokens)} llama tokens, identical = {ident}")
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
        "common_prefix_tokens": cp_tokens,
        "common_prefix_token_frac": cp_tok_frac,
        "qw3_output_tokens": len(q_tokens),
        "llama_output_tokens": len(l_tokens),
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
    cpt = mean([r["common_prefix_tokens"] for r in valid])
    cptf = mean([r["common_prefix_token_frac"] for r in valid])
    qtok = mean([r["qw3_output_tokens"] for r in valid])

    # Weighted average using the actual prompt-token count: a 1K-token prompt
    # should dominate the prefill summary, otherwise a single short prompt's
    # warmup-noisy tok/s number drags the average around.
    def weighted_mean(values_key, weight_key):
        num = 0.0
        den = 0.0
        for r in valid:
            w = r[weight_key].get("prompt_tokens", 0) or 0
            num += r[weight_key][values_key] * w
            den += w
        return num / den if den > 0 else 0.0

    qpref_w = weighted_mean("prefill_tok_s", "qw3")
    lpref_w = weighted_mean("prefill_tok_s", "llama")

    print(f"prompts compared        : {len(valid)} / {len(rows)}")
    print(f"qw3      avg prefill    : {qpref:7.2f} tok/s   decode: {qdec:6.2f} tok/s")
    print(f"llama.cpp avg prefill   : {lpref:7.2f} tok/s   decode: {ldec:6.2f} tok/s")
    if lpref > 0 and ldec > 0:
        print(f"qw3 / llama.cpp         : prefill {qpref/lpref*100:5.1f}%   decode {qdec/ldec*100:5.1f}%")
    if qpref_w > 0 and lpref_w > 0:
        print(f"prompt-weighted prefill : qw3 {qpref_w:7.2f} tok/s   llama {lpref_w:7.2f} tok/s   "
              f"ratio {qpref_w/lpref_w*100:5.1f}%")
    print(f"avg common-prefix chars : {cp:6.1f}    ({cpf*100:5.1f}% of qw3 output)")
    if any(r["common_prefix_tokens"] for r in valid) or any(r["qw3_output_tokens"] for r in valid):
        print(f"avg common-prefix tokens: {cpt:6.1f}/{qtok:.1f}  ({cptf*100:5.1f}% of qw3 output)")
    n_identical = sum(1 for r in valid
                      if r["common_prefix_tokens"]
                      and r["common_prefix_tokens"] == r["qw3_output_tokens"] == r["llama_output_tokens"])
    if any(r["qw3_output_tokens"] for r in valid):
        print(f"identical token sequences: {n_identical} / {len(valid)}")


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
    parser.add_argument("--long", action="store_true",
                        help="include a ~1K-token prompt for stable prefill timing")
    parser.add_argument("--long-only", action="store_true",
                        help="ONLY run the long prompt (best for prefill benchmarking)")
    parser.add_argument("--token-diff", action="store_true",
                        help="tokenize outputs with qw3 and report token-level common prefix "
                             "(slow: extra qw3 invocations to tokenize)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-invocation timeout (seconds)")
    parser.add_argument("--json", type=str, default=None,
                        help="write structured results to this JSON file")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts, "r") as f:
            args.prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
    elif args.long_only:
        args.prompts = [LONG_PROMPT]
    elif args.long:
        args.prompts = list(DEFAULT_PROMPTS) + [LONG_PROMPT]
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
