#!/usr/bin/env python3
"""Compare two QW3_KVMEM_DUMP_SCORES dumps (ratio 0.9 vs 0.5) for one sample.

Answers the sub-cause question for the tiered-path regression:
  * Do the two runs SELECT the same 128 blocks? (selection-set Jaccard + symdiff)
  * Are the gold answer-window blocks selected in each? at what rank?
  * How much do the per-block retrieval scores diverge? (max/median |Δrs|, rank churn)

If selection sets differ / answer missed at 0.5  -> index-scoring divergence under offload.
If selection sets identical                       -> KV stage-in fidelity (bytes), not selection.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_all
from analyze_dump import load_snapshot, session_char_spans
from prompt import render_messages, render_history

NATIVE_CTX = 262144


def snap_by_seq(path: Path, seq: int):
    meta, blocks = None, None
    cm, cb = None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if o.get("type") == "meta":
            if cm is not None and cm.get("seq") == seq:
                return cm, cb
            cm, cb = o, []
        else:
            cb.append(o)
    if cm is not None and cm.get("seq") == seq:
        return cm, cb
    raise SystemExit(f"seq {seq} not found in {path}")


def ans_blocks(sample, meta, blocks):
    bt = meta["block_tokens"]
    n = len(blocks)
    total_tokens = blocks[-1]["p0"] + blocks[-1]["nt"] if blocks else 0
    msgs = render_messages(sample)
    sys_txt, hist_txt, q_txt = msgs[0]["content"], msgs[1]["content"], msgs[2]["content"]
    hist_prefix = len(sys_txt) + 2
    total_chars = len(sys_txt) + len(hist_txt) + len(q_txt)
    tpc = total_tokens / max(1, total_chars)

    def c2b(c):
        tok = int(round((hist_prefix + c) * tpc))
        return min(n - 1, max(0, tok // bt))

    spans = session_char_spans(sample)
    aset = set(sample.answer_session_ids)
    win = []
    for sid, c0, c1 in spans:
        if sid in aset:
            win.extend(range(c2b(c0), c2b(c1) + 1))
    return sorted(set(win))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("/home/chaidi/kvmem_eval/KVMem_Motivation/data/raw/"
                                 "LongMemEval-S/longmemeval_s_cleaned.json"))
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--dump-a", type=Path, required=True, help="ratio 0.9 dump")
    ap.add_argument("--dump-b", type=Path, required=True, help="ratio 0.5 dump")
    args = ap.parse_args()

    s = load_all(args.data)[args.index]
    mA, bA = snap_by_seq(args.dump_a, args.seq)
    mB, bB = snap_by_seq(args.dump_b, args.seq)
    bA.sort(key=lambda b: b["b"]); bB.sort(key=lambda b: b["b"])

    print(f"=== idx={args.index} seq={args.seq} qid={s.question_id} type={s.question_type} ===")
    print(f"Q: {s.question}")
    print(f"gold: {s.answer!r}")
    print(f"A(0.9): blocks={mA['block_count']} sel={mA['selected']} cov={mA['qc_captured_blocks']}/{mA['qc_total_blocks']} idx_ready={mA['index_ready']} mask={mA['mask']}")
    print(f"B(0.5): blocks={mB['block_count']} sel={mB['selected']} cov={mB['qc_captured_blocks']}/{mB['qc_total_blocks']} idx_ready={mB['index_ready']} mask={mB['mask']}")

    selA = {b["b"] for b in bA if b["sel"]}
    selB = {b["b"] for b in bB if b["sel"]}
    inter = selA & selB
    jac = len(inter) / max(1, len(selA | selB))
    print(f"\n[selection] |A|={len(selA)} |B|={len(selB)} shared={len(inter)} "
          f"Jaccard={jac:.3f}  only-in-A={len(selA-selB)}  only-in-B={len(selB-selA)}")

    # score divergence over common block ids
    rsA = {b["b"]: b["rs"] for b in bA}
    rsB = {b["b"]: b["rs"] for b in bB}
    common = sorted(set(rsA) & set(rsB))
    deltas = sorted(abs(rsA[i] - rsB[i]) for i in common)
    if deltas:
        md = deltas[len(deltas)//2]
        print(f"[score Δ] over {len(common)} blocks: max|Δrs|={deltas[-1]:.4g} "
              f"med|Δrs|={md:.4g}")

    # ranks
    def ranks(blocks):
        order = sorted(range(len(blocks)), key=lambda i: (blocks[i]["rs"], blocks[i]["b"]), reverse=True)
        return {blocks[order[r]]["b"]: r for r in range(len(blocks))}
    rkA = ranks(bA); rkB = ranks(bB)

    win = ans_blocks(s, mA, bA)
    print(f"\n[answer window] {len(win)} blocks: {win[0]}..{win[-1]}")
    hdr = f"  {'blk':>5} {'rsA':>10} {'rkA':>5} {'selA':>5} | {'rsB':>10} {'rkB':>5} {'selB':>5}"
    print(hdr)
    budget = mA["budget_blocks"]
    a_hit = b_hit = 0
    for blk in win:
        rA = rkA.get(blk, -1); rB = rkB.get(blk, -1)
        sA = blk in selA; sB = blk in selB
        a_hit += sA; b_hit += sB
        print(f"  {blk:>5} {rsA.get(blk,float('nan')):>10.4g} {rA:>5} {str(sA):>5} | "
              f"{rsB.get(blk,float('nan')):>10.4g} {rB:>5} {str(sB):>5}")
    print(f"\n  answer blocks SELECTED: A(0.9)={a_hit}/{len(win)}   B(0.5)={b_hit}/{len(win)}   (budget={budget})")
    print(f"  best answer-block rank: A={min((rkA.get(b,10**9) for b in win), default=-1)}  "
          f"B={min((rkB.get(b,10**9) for b in win), default=-1)}")


if __name__ == "__main__":
    main()
