#!/usr/bin/env python3
"""Grade a completed KVMem eval run from its per-sample JSONL.

Generation (calling qw3) and grading (calling the DeepSeek judge) are decoupled:
`run_eval.py --no-judge` produces the per-sample JSONL of model answers without
needing a judge key; this script reads that JSONL later and applies the DeepSeek
answer-equivalence judge, emitting the same summary that a judged run_eval.py would
have (overall accuracy + per-type table next to the doc baselines).

Why decouple: model generation over ~110K-token histories is the multi-hour,
server-bound part and is nondeterministic (temp 0.6); judging is cheap, fast, and
the only step that needs DEEPSEEK_API_KEY. Splitting them lets the long run proceed
before the key is available, and lets a run be (re)graded without re-generating.

The judge key still comes strictly from the environment (DEEPSEEK_API_KEY); it is
never read from the JSONL, hardcoded, or written to disk.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .dataset import QUESTION_TYPES, Sample
    from .judge import DeepSeekJudge
except ImportError:  # allow running as a loose module
    from dataset import QUESTION_TYPES, Sample  # type: ignore
    from judge import DeepSeekJudge  # type: ignore


# Overall reference baselines from docs/motivation_experiment_summary_en.md §4.1
DOC_BASELINES: "OrderedDict[str, float]" = OrderedDict([
    ("Full Context (upper bound)", 81.37),
    ("Compact-only", 21.57),
    ("RAG k=22 (~20K)", 73.53),
    ("RAG k=44 (~41K)", 72.55),
])


def sample_from_row(row: dict) -> Sample:
    """Reconstruct the minimal Sample the judge needs from a per-sample JSONL row.

    The judge only reads question_id (for abstention detection), question_type,
    question, and answer (gold); the haystack fields are irrelevant at grading time.
    """
    return Sample(
        question_id=str(row.get("question_id", "")),
        question_type=str(row.get("question_type", "")),
        question=str(row.get("question", "")),
        answer=str(row.get("gold", "")),
        question_date="",
        haystack_dates=[],
        haystack_session_ids=[],
        haystack_sessions=[],
        answer_session_ids=[],
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Grade a KVMem eval JSONL with the DeepSeek judge.")
    ap.add_argument("jsonl", type=Path, help="per-sample JSONL from run_eval.py --no-judge")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write graded outputs (default: alongside the input JSONL)")
    ap.add_argument("--tag", default=None,
                    help="label for output filenames (default: derived from input name)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.jsonl.exists():
        print(f"[grade] ERROR: input JSONL not found: {args.jsonl}")
        return 1
    out_dir = args.out_dir or args.jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (args.jsonl.stem + "_graded")

    rows = []
    with args.jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"[grade] loaded {len(rows)} rows from {args.jsonl}")

    judge = DeepSeekJudge()  # raises if DEEPSEEK_API_KEY is unset
    print(f"[grade] judge: model={judge.model} base_url={judge.base_url}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    graded_path = out_dir / f"{tag}_{ts}.jsonl"
    summary_path = out_dir / f"{tag}_{ts}_summary.json"

    correct_by_type: "OrderedDict[str, int]" = OrderedDict((t, 0) for t in QUESTION_TYPES)
    total_by_type: "OrderedDict[str, int]" = OrderedDict((t, 0) for t in QUESTION_TYPES)
    n_correct = 0
    n_done = 0
    n_error = 0
    n_skipped = 0

    with graded_path.open("w", encoding="utf-8") as out:
        for pos, row in enumerate(rows):
            qtype = str(row.get("question_type", ""))
            # Rows where qw3 itself errored have no answer to grade.
            if row.get("client_error"):
                n_skipped += 1
                row["correct"] = None
                row["judge_raw"] = None
                row["judge_error"] = "skipped: client_error in generation"
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{pos+1:>3}/{len(rows)}] SKIP {qtype:<26} (client_error)")
                continue

            s = sample_from_row(row)
            jr = judge.judge(s, str(row.get("answer", "")))
            total_by_type[qtype] = total_by_type.get(qtype, 0) + 1
            n_done += 1
            verdict = jr.correct
            if jr.error is not None:
                n_error += 1
            elif jr.correct:
                n_correct += 1
                correct_by_type[qtype] = correct_by_type.get(qtype, 0) + 1

            row["correct"] = verdict
            row["judge_raw"] = jr.raw
            row["judge_error"] = jr.error
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            acc = (n_correct / n_done * 100.0) if n_done else 0.0
            v = "Y" if jr.correct else "N"
            print(f"[{pos+1:>3}/{len(rows)}] {qtype:<26} verdict={v} acc={acc:5.1f}% "
                  f"id={row.get('question_id')}"
                  + (f" judge_err={jr.error[:80]}" if jr.error else ""))

    per_type = OrderedDict()
    for t in QUESTION_TYPES:
        tot = total_by_type[t]
        cor = correct_by_type[t]
        per_type[t] = {
            "correct": cor,
            "total": tot,
            "accuracy": (cor / tot * 100.0) if tot else None,
        }
    overall_acc = (n_correct / n_done * 100.0) if n_done else 0.0

    summary = {
        "timestamp_utc": ts,
        "source_jsonl": str(args.jsonl),
        "judge_model": judge.model,
        "n_graded": n_done,
        "n_correct": n_correct,
        "overall_accuracy": overall_acc,
        "n_judge_error": n_error,
        "n_skipped_client_error": n_skipped,
        "per_type": per_type,
        "graded_jsonl_path": str(graded_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== KVMem utility eval — graded summary ===")
    print(f"  source={args.jsonl}")
    print(f"  graded={n_done}  judge_errors={n_error}  skipped(client_error)={n_skipped}")
    print(f"  OVERALL: {n_correct}/{n_done} = {overall_acc:.2f}%")
    print("  per-type:")
    for t in QUESTION_TYPES:
        d = per_type[t]
        a = "n/a" if d["accuracy"] is None else f"{d['accuracy']:5.1f}%"
        print(f"    {t:<28} {d['correct']:>2}/{d['total']:<2}  {a}")
    print("  reference baselines (doc §4.1, same model family):")
    for name, val in DOC_BASELINES.items():
        print(f"    {name:<28} {val:5.2f}%")
    print(f"  graded   : {graded_path}")
    print(f"  summary  : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
