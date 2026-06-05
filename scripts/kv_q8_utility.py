"""Utility benchmark: long-context passkey retrieval against a qw3 serve API.

Why this test: KV-cache quantization injects per-row int8 rounding error into
*every cached K/V row*. The error is invisible at short context (tiny cache) and
accumulates as the cache grows. The textbook sensitive probe is needle-in-a-
haystack passkey retrieval: hide a random secret at a controlled depth inside a
long filler context, then ask the model to read it back. A correct retrieval
means attention-over-context survived quantization; a miss means the quant error
corrupted the K/V rows the model needed to attend to.

This version hits a *persistent* qw3 server (OpenAI-compatible /v1/completions,
raw prompt) instead of re-loading ./build/qw3 per cell. The KV dtype is fixed
server-side at launch (qw3 serve --kv-dtype {fp16,q8}); to compare dtypes run two
servers (see scripts/kv_q8_serve_compare.sh) and point --base-url at each.

Usage:
  python3 scripts/kv_q8_utility.py --base-url http://127.0.0.1:8080/v1
                                   [--lens 4000,16000,64000]
                                   [--depths 0.1,0.5,0.9] [--trials 3]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the calibrated filler passage / token-count model from the perf sweep.
from long_prompt_sweep import _PASSAGE  # type: ignore  # noqa: E402
from oai_client import complete  # noqa: E402

# ~270 tokens per filler repeat (calibrated in long_prompt_sweep.make_prompt).
_PER_REPEAT = 270


def make_passkey_prompt(target_tokens: int, depth: float, secret: str) -> str:
    """Filler context with the secret sentence buried at fractional `depth`."""
    n_repeats = max(2, target_tokens // _PER_REPEAT)
    needle = (f" The special access code that you must remember is {secret}. "
              f"Do not forget it: the access code is {secret}. ")
    insert_at = max(1, min(n_repeats - 1, int(round(n_repeats * depth))))
    head = _PASSAGE * insert_at
    tail = _PASSAGE * (n_repeats - insert_at)
    body = head + needle + tail
    return ("Read the following passage carefully. Somewhere inside it is a "
            "special access code.\n\n" + body +
            "\n\nQuestion: What is the special access code mentioned in the "
            "passage above? Answer with only the code.\nAnswer: The access code is")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qw3")
    ap.add_argument("--lens", default="4000,16000,64000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--decode", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    lens = [int(x) for x in args.lens.split(",") if x]
    depths = [float(x) for x in args.depths.split(",") if x]
    rng = random.Random(args.seed)

    print(f"base_url={args.base_url}  model={args.model}")
    print(f"lens={lens}  depths={depths}  trials={args.trials}\n")
    hdr = f"{'len':>7} {'depth':>6} {'hits':>12}"
    print(hdr)
    print("-" * len(hdr))

    hits_total = 0
    n_total = 0
    for L in lens:
        for depth in depths:
            cell = 0
            for _ in range(args.trials):
                secret = str(rng.randint(10000, 99999))
                prompt = make_passkey_prompt(L, depth, secret)
                try:
                    gen = complete(args.base_url, args.model, prompt,
                                   max_tokens=args.decode, temperature=0.0,
                                   seed=0, timeout=args.timeout)
                except Exception:  # noqa: BLE001
                    gen = ""
                hit = secret in gen
                cell += int(hit)
                n_total += 1
                hits_total += int(hit)
            t = args.trials
            print(f"{L:>7} {depth:>6.2f} "
                  f"{cell:>3}/{t:<3}({100*cell/t:>3.0f}%)")

    print("\n=== overall ===")
    print(f"  {hits_total}/{n_total} = {100*hits_total/max(n_total,1):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
