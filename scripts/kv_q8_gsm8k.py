"""Utility benchmark: GSM8K (grade-school math reasoning) against a qw3 serve API.

Why this complements the passkey test: passkey stresses long-context retrieval
(does attention survive over a large quantized cache). GSM8K stresses the other
axis: multi-step arithmetic reasoning at short context, where the model attends
back over its own generated chain-of-thought. If Q8 KV rounding corrupts the
CoT-attention, accuracy drops here even though the cache is small.

This version hits a *persistent* qw3 server (OpenAI-compatible) instead of
re-loading ./build/qw3 per problem. The KV dtype is fixed server-side at launch
(qw3 serve --kv-dtype {fp16,q8}); to compare dtypes, run two servers (see
scripts/kv_q8_serve_compare.sh) and point --base-url at each.

Dataset: GSM8K test split. Expects /tmp/gsm8k_test.jsonl (one {"question","answer"}
JSON per line); pass --data to point elsewhere.

Usage:
  python3 scripts/kv_q8_gsm8k.py --base-url http://127.0.0.1:8080/v1 [--n 40]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oai_client import chat  # noqa: E402

_ANS_RE = re.compile(r"####\s*(-?[\d,]+)")
_NUM_RE = re.compile(r"-?\d[\d,]*")


def gold_answer(answer_field: str) -> str:
    m = _ANS_RE.search(answer_field)
    return m.group(1).replace(",", "") if m else ""


def parse_pred(gen: str) -> str:
    """Extract the predicted integer. Prefer text after a '####', else last number."""
    m = _ANS_RE.search(gen)
    if m:
        return m.group(1).replace(",", "")
    nums = _NUM_RE.findall(gen)
    return nums[-1].replace(",", "") if nums else ""


def make_messages(question: str) -> list:
    instr = ("Solve the math problem. Show brief step-by-step working, then "
             "give the final integer answer on its own after '####'.\n\n"
             f"{question}")
    return [{"role": "user", "content": instr}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qw3")
    ap.add_argument("--data", default="/tmp/gsm8k_test.jsonl")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--decode", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n]

    print(f"base_url={args.base_url}  model={args.model}")
    print(f"n={len(rows)}  decode={args.decode}\n")
    hdr = f"{'#':>3} {'gold':>8} {'pred':>10}"
    print(hdr); print("-" * len(hdr))

    correct = 0
    for i, row in enumerate(rows):
        gold = gold_answer(row["answer"])
        msgs = make_messages(row["question"])
        try:
            gen = chat(args.base_url, args.model, msgs, max_tokens=args.decode,
                       temperature=0.0, seed=0, timeout=args.timeout)
        except Exception as e:  # noqa: BLE001
            gen = ""
            print(f"{i:>3} {gold:>8}  ERR {str(e)[:40]}")
            continue
        pred = parse_pred(gen)
        ok = (pred == gold and gold)
        if ok:
            correct += 1
        print(f"{i:>3} {gold:>8} {'OK ' if ok else 'x  '}{pred:>7}")

    n = len(rows)
    acc = 100 * correct / max(n, 1)
    print("\n=== overall ===")
    print(f"  {correct}/{n} = {acc:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
