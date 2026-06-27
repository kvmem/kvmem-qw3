#!/usr/bin/env python3
"""kvmem x MTP compatibility probe (the dangerous-interaction focus).

The functional gate proves the path doesn't crash and is byte-identical at an
identity budget. This probe asks the sharper question: is MTP *speculation*
still actually doing work when kvmem is on, or has kvmem silently turned the
draft chain into a no-op (every chain bails -> single-token decode -> "works"
but no speedup, and any kvmem-vs-plain throughput number would be a lie)?

The server logs one line per request:

  native mtp_spec_summary: ... drafted=D accepted=A ... acceptance=X ...
      batched_verify_batches=B batched_verify_tokens=T rollbacks=K

That is the ground truth. We compare plain MTP vs kvmem-identity vs kvmem-sparse
on BOTH verify routes:

  * conc=1 -> single verifier route (forward_n_tokens, window self-advances)
  * conc=2 -> ragged batched verify route (the new window-substitution builder)

Compatibility requires, for every kvmem config:
  1. drafted > 0                      (MTP draft chain is live, not bailing)
  2. acceptance > 0                   (verify accepts drafts, MTP is useful)
  3. batched_verify on conc=2         (kvmem rows take the window-aware ragged
                                       route, not a per-row fallback)
  4. identity acceptance == plain     (window == full cache -> identical verify)
  5. identity output == plain         (byte parity, NSPLIT pinned)
  6. sparse output coherent + nonempty (no garbage/empty from a bad window)

NSPLIT is pinned to 1 so plain vs identity is deterministic for the parity
checks; the ragged route is forced via the min-tokens override so conc=2 truly
exercises the batched builder even at short chains.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvmem_e2e_regression as e2e  # noqa: E402

_RE = {
    "drafted": re.compile(r"\bdrafted=(\d+)"),
    "accepted": re.compile(r"\baccepted=(\d+)"),
    "acceptance": re.compile(r"\bacceptance=([\d.]+)"),
    "bv_batches": re.compile(r"\bbatched_verify_batches=(\d+)"),
    "bv_tokens": re.compile(r"\bbatched_verify_tokens=(\d+)"),
    "rollbacks": re.compile(r"\brollbacks=(\d+)"),
}
_SUMMARY = re.compile(r"native mtp_spec_summary:[^\n]*")
_RAGGED_MARK = re.compile(r"native continuous_mtp:")


def long_prompt(tag: str, words: int) -> str:
    body = " ".join(["alpha beta gamma delta epsilon"] * (words // 5 + 1))
    return (f"{tag}\nContext passage follows.\n{body}\n"
            f"Now write a detailed multi-paragraph explanation of the passage.")


def parse_summaries(log: str) -> List[Dict[str, float]]:
    out = []
    for line in _SUMMARY.findall(log):
        rec: Dict[str, float] = {}
        for k, rx in _RE.items():
            m = rx.search(line)
            rec[k] = float(m.group(1)) if m else -1.0
        out.append(rec)
    return out


def run_cfg(name: str, qw3: Path, model: Path, prompts: List[str],
            max_tokens: int, ctx: int, chain: int, conc: int,
            kvmem_flags: List[str], timeout_s: int) -> dict:
    host = "127.0.0.1"
    port = e2e.find_free_port()
    cmd = [
        str(qw3), "serve", "--model", str(model),
        "--host", host, "--port", str(port),
        "--ctx", str(ctx), "-n", str(max_tokens), "--temp", "0",
        "--kv-dtype", "fp16", "--prefill-chunk", "2048",
        "--continuous-batching", "--max-active", str(conc),
        "--native-mtp-speculate", "--mtp-chain", str(chain),
    ] + kvmem_flags
    env = os.environ.copy()
    env.update({
        "QW3_CONTINUOUS_BATCHING_TRACE": "1",
        "QW3_FATTN_NSPLIT": "1",
        "QW3_PREFILL_FA2_NSPLIT": "1",
        "QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS": "1",
    })
    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=env)
    bodies: List[str] = []
    try:
        e2e.wait_for_health(host, port, min(180.0, float(timeout_s)))

        def fire(p: str):
            return e2e.post_completion(host, port, p, max_tokens, float(timeout_s))

        with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
            for _, body in ex.map(fire, prompts):
                bodies.append(body)
    finally:
        log = e2e.terminate_server(proc)

    texts = [e2e.parse_completion_text(b) for b in bodies]
    recs = parse_summaries(log)
    drafted = sum(r["drafted"] for r in recs if r["drafted"] >= 0)
    accepted = sum(r["accepted"] for r in recs if r["accepted"] >= 0)
    bv_batches = sum(r["bv_batches"] for r in recs if r["bv_batches"] >= 0)
    bv_tokens = sum(r["bv_tokens"] for r in recs if r["bv_tokens"] >= 0)
    rollbacks = sum(r["rollbacks"] for r in recs if r["rollbacks"] >= 0)
    acc = accepted / drafted if drafted > 0 else 0.0
    return {
        "name": name,
        "conc": conc,
        "requests": len(prompts),
        "summaries": len(recs),
        "texts": texts,
        "ok_status": all(t != "" for t in texts),
        "drafted": int(drafted),
        "accepted": int(accepted),
        "acceptance": round(acc, 4),
        "bv_batches": int(bv_batches),
        "bv_tokens": int(bv_tokens),
        "rollbacks": int(rollbacks),
        "ragged_marks": len(_RAGGED_MARK.findall(log)),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qw3", default="./build/qw3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--chain", type=int, default=4)
    ap.add_argument("--prompt-words", type=int, default=700)
    ap.add_argument("--sparse-budget", type=int, default=192)
    ap.add_argument("--timeout", type=int, default=500)
    ap.add_argument("--out-json", default="/tmp/qw3_kvmem_mtp_compat.json")
    args = ap.parse_args(argv)

    qw3 = Path(args.qw3)
    model = Path(args.model)
    if not qw3.exists():
        raise SystemExit(f"qw3 binary not found: {qw3}")
    if not model.exists():
        raise SystemExit(f"model not found: {model}")

    ident = ["--kvmem", "--kvmem-block-tokens", "16",
             "--kvmem-budget", "131072", "--kvmem-method", "recency"]
    sparse = ["--kvmem", "--kvmem-block-tokens", "16",
              "--kvmem-budget", str(args.sparse_budget),
              "--kvmem-method", "recency", "--kvmem-sink-blocks", "1",
              "--kvmem-recent-blocks", "2", "--kvmem-interval", "16"]

    results = []
    failures: List[str] = []

    for conc in (1, 2):
        route = "single-verifier" if conc == 1 else "ragged-batched"
        prompts = [long_prompt(f"REQ-{i}", args.prompt_words)
                   for i in range(conc)]
        print(f"\n##### conc={conc} ({route}) #####", flush=True)
        cfgs = [("plain", []), ("kvmem_ident", ident), ("kvmem_sparse", sparse)]
        byname = {}
        for cname, flags in cfgs:
            print(f"[compat] conc={conc} {cname} ...", flush=True)
            r = run_cfg(cname, qw3, model, prompts, args.max_tokens, args.ctx,
                        args.chain, conc, flags, args.timeout)
            results.append(r)
            byname[cname] = r
            print(f"  ok={r['ok_status']} summaries={r['summaries']} "
                  f"drafted={r['drafted']} accepted={r['accepted']} "
                  f"acceptance={r['acceptance']} bv_batches={r['bv_batches']} "
                  f"bv_tokens={r['bv_tokens']} rollbacks={r['rollbacks']} "
                  f"ragged_marks={r['ragged_marks']}", flush=True)

        plain = byname["plain"]
        ident_r = byname["kvmem_ident"]
        sparse_r = byname["kvmem_sparse"]

        # 1+2: MTP must be live under kvmem
        for r in (ident_r, sparse_r):
            if r["drafted"] <= 0:
                failures.append(f"conc={conc} {r['name']}: drafted=0 "
                                "(MTP draft chain dead under kvmem)")
            if r["acceptance"] <= 0.0:
                failures.append(f"conc={conc} {r['name']}: acceptance=0 "
                                "(verify accepts no drafts -> MTP is a no-op)")
        # 3: ragged route actually taken on conc=2 for kvmem rows
        if conc == 2:
            for r in (ident_r, sparse_r):
                if r["bv_batches"] <= 0:
                    failures.append(f"conc={conc} {r['name']}: "
                                    "batched_verify_batches=0 (kvmem row did "
                                    "NOT take the window-aware ragged route)")
        # 4: identity acceptance == plain
        if abs(ident_r["acceptance"] - plain["acceptance"]) > 1e-6:
            failures.append(
                f"conc={conc}: identity acceptance {ident_r['acceptance']} != "
                f"plain {plain['acceptance']} (window should equal full cache)")
        # 5: identity output byte-identical to plain
        if ident_r["texts"] != plain["texts"]:
            failures.append(
                f"conc={conc}: identity output diverged from plain MTP")
        # 6: sparse output coherent (nonempty)
        if not sparse_r["ok_status"]:
            failures.append(f"conc={conc}: sparse produced an empty completion")

    print("\n=== kvmem x MTP compatibility ===")
    print(f"{'route':16s} {'config':13s} {'draft':>6s} {'accept':>7s} "
          f"{'rate':>6s} {'bv_batch':>8s} {'rollbk':>6s}  ok")
    for r in results:
        route = "single" if r["conc"] == 1 else "ragged"
        print(f"{route:16s} {r['name']:13s} {r['drafted']:6d} "
              f"{r['accepted']:7d} {r['acceptance']:6.3f} {r['bv_batches']:8d} "
              f"{r['rollbacks']:6d}  {r['ok_status']}")

    # show one sparse sample so a human can eyeball coherence
    for r in results:
        if r["name"] == "kvmem_sparse" and r["texts"]:
            print(f"\n[sparse conc={r['conc']} sample]: "
                  f"{r['texts'][0][:160]!r}")

    out = {"args": vars(args), "ok": not failures,
           "failures": failures, "results": results}
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
    print(f"\nwrote {args.out_json}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: kvmem x MTP compatible (MTP live, identity-parity, ragged route)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
