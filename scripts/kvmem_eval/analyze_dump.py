#!/usr/bin/env python3
"""Offline analysis of a QW3_KVMEM_DUMP_SCORES score-dump.

Given a LongMemEval-M sample and the per-block retrieval-score dump emitted by the
executor at reselect time, locate the answer session(s) by fractional position in
the rendered prompt, then report:

  * The answer-window block range and its true token positions (H4: <256K vs >256K).
  * The best retrieval_score inside the answer window and its RANK among all blocks,
    vs. the configured budget cutoff -> settles scoring-miss vs. coverage-miss (H1).
  * The recent-window / sink split of the 128 selected blocks and the recent blocks'
    retrieval_scores -> quantifies wasted budget (H3).
  * Budget-sweep + recent-shrink simulation replayed on the dumped scores (H1/H3),
    without re-running the model.
  * Text excerpts of the top-scored blocks, so we can eyeball what scoring latched
    onto (does the ranking pick relevant content at all?).

Char->token mapping is approximate (no local tokenizer): total_tokens/total_chars
is uniform enough over English chat text that a whole-session window (many blocks)
is robust to the drift. In-range vs. extrapolated (256K) is never ambiguous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .dataset import Sample, load_all
    from .prompt import SYSTEM_INSTRUCTION, render_history, render_messages
except ImportError:
    from dataset import Sample, load_all  # type: ignore
    from prompt import SYSTEM_INSTRUCTION, render_history, render_messages  # type: ignore

NATIVE_CTX = 262144  # Qwen3.6-27B native context (256K)


def session_char_spans(sample: Sample) -> list[tuple[str, int, int]]:
    """Reproduce render_history line-by-line, recording [start,end) char offsets of
    each session's rendered text within the history string."""
    spans: list[tuple[str, int, int]] = []
    lines: list[str] = []
    dates = sample.haystack_dates
    ids = sample.haystack_session_ids
    pos = 0  # running char offset (mirrors "\n".join semantics)

    def emit(s: str) -> None:
        nonlocal pos
        # "\n".join adds 1 newline BEFORE every line except the first.
        if lines:
            pos += 1  # the join newline
        lines.append(s)
        pos += len(s)

    for idx, session in enumerate(sample.haystack_sessions):
        sid = ids[idx] if idx < len(ids) else f"_s{idx}"
        date = dates[idx] if idx < len(dates) else ""
        header = f"=== Conversation on {date} ===" if date else f"=== Conversation {idx + 1} ==="
        start = pos + (1 if lines else 0)  # where this session's header char begins
        emit(header)
        for turn in session:
            role = str(turn.get("role", "")).strip().lower()
            content = str(turn.get("content", "")).strip()
            speaker = "User" if role == "user" else "Assistant"
            emit(f"{speaker}: {content}")
        end = pos
        emit("")  # blank line between sessions
        spans.append((sid, start, end))
    return spans


def load_snapshot(dump_path: Path, seq: int | None) -> tuple[dict, list[dict]]:
    """Return (meta, blocks) for the requested seq (or the last snapshot if None)."""
    snaps: list[tuple[dict, list[dict]]] = []
    cur_meta: dict | None = None
    cur_blocks: list[dict] = []
    for line in dump_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("type") == "meta":
            if cur_meta is not None:
                snaps.append((cur_meta, cur_blocks))
            cur_meta = obj
            cur_blocks = []
        else:
            cur_blocks.append(obj)
    if cur_meta is not None:
        snaps.append((cur_meta, cur_blocks))
    if not snaps:
        raise SystemExit(f"no snapshots in {dump_path}")
    if seq is None:
        return snaps[-1]
    for m, b in snaps:
        if m.get("seq") == seq:
            return m, b
    raise SystemExit(f"seq {seq} not found; available: {[m.get('seq') for m, _ in snaps]}")


def simulate_pick(blocks: list[dict], budget: int, sink: int, recent: int) -> set[int]:
    """Replay pick_topk_blocks on dumped retrieval scores (mean-k / Retrieval path):
    keep first `sink` + last `recent` unconditionally, fill the rest by retrieval
    score. Returns the set of selected block ids."""
    n = len(blocks)
    if budget == 0 or n <= budget:
        return {b["b"] for b in blocks}
    kept: set[int] = set()
    for i in range(min(sink, n)):
        if len(kept) >= budget:
            break
        kept.add(blocks[i]["b"])
    for i in range(min(recent, n)):
        if len(kept) >= budget:
            break
        kept.add(blocks[n - 1 - i]["b"])
    if len(kept) < budget:
        rest = [b for b in blocks if b["b"] not in kept]
        # match executor tie-break: higher score first, then higher block id
        rest.sort(key=lambda b: (b["rs"], b["b"]), reverse=True)
        for b in rest[: budget - len(kept)]:
            kept.add(b["b"])
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/"
                                 "LongMemEval-M/longmemeval_m_cleaned.json"))
    ap.add_argument("--index", type=int, required=True, help="subset (use_all) index")
    ap.add_argument("--dump", type=Path,
                    default=Path("/data/chaidi/kvmem_eval/results/scoredump_all.jsonl"))
    ap.add_argument("--seq", type=int, default=None,
                    help="which dump snapshot (default: last)")
    ap.add_argument("--topn", type=int, default=15, help="top scored blocks to excerpt")
    args = ap.parse_args()

    samples = load_all(args.data)
    s = samples[args.index]
    meta, blocks = load_snapshot(args.dump, args.seq)
    blocks.sort(key=lambda b: b["b"])
    n = len(blocks)
    bt = meta["block_tokens"]
    budget = meta["budget_blocks"]
    sink = meta["sink"]
    recent_cfg = meta["recent"]
    recent = recent_cfg
    total_tokens = blocks[-1]["p0"] + blocks[-1]["nt"] if blocks else 0

    # Reconstruct the full prompt text for char->token mapping.
    msgs = render_messages(s)
    sys_txt, hist_txt, q_txt = msgs[0]["content"], msgs[1]["content"], msgs[2]["content"]
    history = render_history(s)  # the raw history (spans are computed against this)
    hist_prefix = len(sys_txt) + 2  # system + a couple template chars (negligible)
    total_chars = len(sys_txt) + len(hist_txt) + len(q_txt)
    tok_per_char = total_tokens / max(1, total_chars)

    def char_to_block(c: int) -> int:
        tok = int(round((hist_prefix + c) * tok_per_char))
        return min(n - 1, max(0, tok // bt))

    print(f"=== sample idx={args.index} qid={s.question_id} type={s.question_type} ===")
    print(f"Q: {s.question}")
    print(f"gold: {s.answer!r}")
    print(f"answer_session_ids: {s.answer_session_ids}")
    mask_lbl = {None: "?", -1: "as-run", 0: "no-mask", 1: "masked"}.get(
        meta.get("mask"), meta.get("mask"))
    print(f"dump: seq={meta['seq']} blocks={n} total_tokens={total_tokens} "
          f"budget={budget} sink={sink} recent_cfg={recent_cfg} recent_eff={recent} "
          f"method={meta['method']} mask={mask_lbl} selected={meta['selected']}")
    print(f"char->tok ratio={tok_per_char:.4f} ({total_chars} chars -> {total_tokens} tok)")

    scores = [b["rs"] for b in blocks]
    order = sorted(range(n), key=lambda i: (scores[i], i), reverse=True)
    rank_of = {order[r]: r for r in range(n)}  # block index (position) -> 0-based rank
    smax = max(scores) if scores else 0.0
    smin = min(scores) if scores else 0.0
    p = lambda q: sorted(scores)[min(n - 1, int(q * n))]
    print(f"score dist: max={smax:.4g} p99={p(0.99):.4g} p90={p(0.90):.4g} "
          f"med={p(0.50):.4g} min={smin:.4g}")

    # Recent-window waste (H3): retrieval scores of the always-kept recent blocks.
    recent_ids = list(range(max(0, n - recent), n))
    recent_scores = [scores[i] for i in recent_ids]
    recent_ranks = [rank_of[i] for i in recent_ids]
    if recent_scores:
        print(f"\n[H3] recent window = last {recent} blocks (ids {recent_ids[0]}..{recent_ids[-1]}): "
              f"score max={max(recent_scores):.4g} med={sorted(recent_scores)[len(recent_scores)//2]:.4g}; "
              f"their ranks span {min(recent_ranks)}..{max(recent_ranks)} (budget cutoff={budget})")
        n_recent_would_qualify = sum(1 for r in recent_ranks if r < budget - sink - recent)
        print(f"      recent blocks that would ALSO rank into the retrieval budget on their own: "
              f"{n_recent_would_qualify}/{recent}")

    # Answer-window analysis (H1/H4).
    spans = session_char_spans(s)
    ans_ids = set(s.answer_session_ids)
    print("\n[answer sessions -> blocks]")
    all_ans_blocks: list[int] = []
    for sid, c0, c1 in spans:
        if sid not in ans_ids:
            continue
        b0, b1 = char_to_block(c0), char_to_block(c1)
        tok0 = int(round((hist_prefix + c0) * tok_per_char))
        tok1 = int(round((hist_prefix + c1) * tok_per_char))
        win = list(range(b0, b1 + 1))
        all_ans_blocks.extend(win)
        best = max(win, key=lambda i: scores[i])
        best_rank = rank_of[best]
        in_range = "IN-RANGE(<256K)" if tok1 < NATIVE_CTX else "EXTRAPOLATED(>256K)"
        n_sel = sum(1 for i in win if blocks[i]["sel"])
        print(f"  sid={sid} chars[{c0}:{c1}] ~tok[{tok0}:{tok1}] {in_range}")
        print(f"    blocks[{b0}:{b1}] ({len(win)} blks)  best_score={scores[best]:.4g} "
              f"@block {best} rank={best_rank}/{n}  "
              f"{'<= IN BUDGET' if best_rank < budget else '>> MISSED (rank>budget)'}")
        print(f"    selected-in-window={n_sel}/{len(win)}  "
              f"window score max={max(scores[i] for i in win):.4g} "
              f"med={sorted(scores[i] for i in win)[len(win)//2]:.4g}")

    # H1 budget sweep + H3 recent-shrink, replayed offline on the dumped scores.
    if all_ans_blocks:
        ans_set = set(all_ans_blocks)
        best_ans = max(all_ans_blocks, key=lambda i: scores[i])
        print(f"\n[H1 budget sweep] answer best block={best_ans} rank={rank_of[best_ans]}/{n} "
              f"score={scores[best_ans]:.4g}")
        sweep_budgets = sorted({128, 256, 512, 1024, 2048, budget})
        for B in sweep_budgets:
            sel = simulate_pick(blocks, B, sink, recent)
            hit = sum(1 for i in ans_set if blocks[i]["b"] in sel)
            print(f"    budget={B*bt} tok ({B} blks): answer blocks selected {hit}/{len(ans_set)}"
                  f"{'  <-- current' if B == budget else ''}")
        print(f"[H3 recent-shrink] budget={budget} blks, vary recent:")
        for R in (32, 16, 8, 4, 0):
            sel = simulate_pick(blocks, budget, sink, R)
            hit = sum(1 for i in ans_set if blocks[i]["b"] in sel)
            print(f"    recent={R}: answer blocks selected {hit}/{len(ans_set)}")

    # Top scored blocks + excerpts (what did scoring latch onto?).
    print(f"\n[top {args.topn} scored blocks]")
    for r in range(min(args.topn, n)):
        i = order[r]
        b = blocks[i]
        c = int(b["p0"] / max(1, total_tokens) * total_chars) - hist_prefix
        c = max(0, min(len(hist_txt) - 1, c))
        excerpt = hist_txt[c:c + 160].replace("\n", " ")
        inans = " *ANS*" if b["b"] in set(all_ans_blocks) else ""
        print(f"  rank{r:>3} blk{b['b']:>4} p0={b['p0']:>7} rs={b['rs']:.4g} "
              f"sel={b['sel']}{inans}  | {excerpt!r}")


if __name__ == "__main__":
    main()
