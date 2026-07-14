#!/usr/bin/env python3
"""Summarize gold-session block recall across a KVMem score dump.

Snapshots are matched to dataset rows in request order. Gold session character
spans are mapped approximately to token blocks with the same uniform mapping used
by analyze_dump.py. This is intended for fast scorer iteration before expensive
generation/judge runs, not as an exact tokenizer-level metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .analyze_dump import session_char_spans
    from .dataset import load_all
    from .prompt import render_messages
except ImportError:
    from analyze_dump import session_char_spans  # type: ignore
    from dataset import load_all  # type: ignore
    from prompt import render_messages  # type: ignore


def load_snapshots(path: Path) -> list[tuple[dict, list[dict]]]:
    out: list[tuple[dict, list[dict]]] = []
    meta: dict | None = None
    blocks: list[dict] = []
    for raw in path.read_text().splitlines():
        obj = json.loads(raw)
        if obj.get("type") == "meta":
            if meta is not None:
                out.append((meta, blocks))
            meta, blocks = obj, []
        else:
            blocks.append(obj)
    if meta is not None:
        out.append((meta, blocks))
    return out


def answer_blocks(sample, blocks: list[dict]) -> set[int]:
    if not blocks:
        return set()
    total_tokens = blocks[-1]["p0"] + blocks[-1]["nt"]
    msgs = render_messages(sample)
    sys_txt, hist_txt, q_txt = (m["content"] for m in msgs[:3])
    total_chars = len(sys_txt) + len(hist_txt) + len(q_txt)
    hist_prefix = len(sys_txt) + 2
    tok_per_char = total_tokens / max(1, total_chars)
    n = len(blocks)
    bt = max(1, blocks[0]["nt"])

    def to_block(char_pos: int) -> int:
        tok = int(round((hist_prefix + char_pos) * tok_per_char))
        return min(n - 1, max(0, tok // bt))

    wanted = set(sample.answer_session_ids)
    result: set[int] = set()
    for sid, c0, c1 in session_char_spans(sample):
        if sid not in wanted:
            continue
        result.update(range(to_block(c0), to_block(c1) + 1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    samples = load_all(args.data)
    snaps = load_snapshots(args.dump)
    n = min(len(samples), len(snaps), args.limit or len(samples))
    if n == 0:
        raise SystemExit("no matched samples/snapshots")

    any_selected = 0
    best_in_budget = 0
    reciprocal_rank = 0.0
    rows: list[tuple[int, str, int, int, int]] = []
    for i in range(n):
        sample = samples[i]
        meta, blocks = snaps[i]
        gold = answer_blocks(sample, blocks)
        ranked = sorted(blocks, key=lambda b: (b["rs"], b["b"]), reverse=True)
        rank = {b["b"]: r for r, b in enumerate(ranked)}
        ranks = [rank[b] for b in gold if b in rank]
        best_rank = min(ranks) if ranks else len(blocks)
        selected = {b["b"] for b in blocks if b.get("sel")}
        hit = len(gold & selected)
        budget = int(meta["budget_blocks"])
        any_selected += hit > 0
        best_in_budget += best_rank < budget
        reciprocal_rank += 1.0 / (best_rank + 1)
        rows.append((i, sample.question_type, best_rank, hit, len(gold)))

    print(f"matched={n} snapshots={len(snaps)} samples={len(samples)}")
    print(f"answer-session selected(any): {any_selected}/{n} = {any_selected/n:.2%}")
    print(f"best answer block rank<budget: {best_in_budget}/{n} = {best_in_budget/n:.2%}")
    print(f"MRR(best answer block): {reciprocal_rank/n:.4f}")
    for i, qtype, rank, hit, ngold in rows:
        print(f"idx={i:3d} type={qtype:28s} best_rank={rank:4d} "
              f"selected_gold={hit}/{ngold}")


if __name__ == "__main__":
    main()
