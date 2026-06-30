#!/usr/bin/env python3
"""KVMem utility evaluation orchestrator (external, OpenAI-API-only).

Iterates the deterministic 102-sample LongMemEval-S subset, renders each sample's
full-history QA prompt, sends it to the isolated qw3 serve endpoint over the
OpenAI-compatible HTTP API, grades the answer with the DeepSeek judge, and writes
both a per-sample JSONL and a summary that mirrors
docs/motivation_experiment_summary_en.md §4.1 (overall accuracy + per-type table),
placed next to the doc's Full / Compact / RAG baselines for direct comparison.

This script contains NO qw3-internal logic: qw3 is reached only through
`{base_url}/v1`. Requests are sent strictly sequentially (concurrency = 1) to
match the study and avoid tier/GPU contention. Results are streamed to disk as we
go, so an interrupted run keeps every completed sample.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

try:
    from .client import Qw3Client
    from .dataset import QUESTION_TYPES, build_subset, load_all
    from .judge import DeepSeekJudge
    from .prompt import render_messages
except ImportError:  # allow running as a loose module
    from client import Qw3Client  # type: ignore
    from dataset import QUESTION_TYPES, build_subset, load_all  # type: ignore
    from judge import DeepSeekJudge  # type: ignore
    from prompt import render_messages  # type: ignore


# Overall reference baselines from docs/motivation_experiment_summary_en.md §4.1
# (Qwen3.6-27B family, 102-sample subset). For context in the summary printout.
DOC_BASELINES: "OrderedDict[str, float]" = OrderedDict([
    ("Full Context (upper bound)", 81.37),
    ("Compact-only", 21.57),
    ("RAG k=22 (~20K)", 73.53),
    ("RAG k=44 (~41K)", 72.55),
])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the KVMem LongMemEval utility eval.")
    ap.add_argument("--data", type=Path,
                    default=Path("/home/chaidi/qw3/selected_12_samples.jsonl"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/data/chaidi/kvmem_eval/results"))
    ap.add_argument("--per-type", type=int, default=17)
    ap.add_argument("--use-all", action="store_true",
                    help="run every loaded sample in file order (the provided "
                         "selected_12_samples.jsonl IS the exact 102 subset; skip "
                         "the deterministic build_subset reconstruction)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap total samples (smoke runs); applied after subset build")
    ap.add_argument("--indices", default=None,
                    help="comma-separated subset indices to run (overrides --limit)")
    ap.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="completion budget (doc envelope: max total completion "
                         "was ~7690 tokens); covers thinking + answer")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Qwen3 thinking recipe (temp 0.6 / top-p 0.95) avoids the "
                         "greedy-decode thinking loops seen at temp 0")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--no-thinking", action="store_true",
                    help="disable server-side thinking (default: thinking ON, matches baseline)")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip DeepSeek grading (model-only run; e.g. smoke without a key)")
    ap.add_argument("--read-timeout", type=float, default=3600.0)
    ap.add_argument("--tag", default="kvmem",
                    help="label embedded in the output filenames")
    return ap.parse_args()


def select_samples(subset, args):
    if args.indices:
        idxs = [int(x) for x in args.indices.split(",") if x.strip() != ""]
        return [(i, subset[i]) for i in idxs]
    chosen = list(enumerate(subset))
    if args.limit is not None:
        chosen = chosen[: args.limit]
    return chosen


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_all(args.data)
    subset = loaded if args.use_all else build_subset(loaded, per_type=args.per_type)
    samples = select_samples(subset, args)
    print(f"[run_eval] loaded={len(loaded)} subset={len(subset)} "
          f"running={len(samples)} samples (use_all={args.use_all})")

    client = Qw3Client(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        enable_thinking=not args.no_thinking,
        read_timeout=args.read_timeout,
    )
    if not client.health():
        print(f"[run_eval] ERROR: qw3 endpoint not healthy at {args.base_url}")
        return 1

    judge = None
    if not args.no_judge:
        judge = DeepSeekJudge()  # raises if DEEPSEEK_API_KEY is unset
        print(f"[run_eval] judge: model={judge.model} base_url={judge.base_url}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_path = args.out_dir / f"{args.tag}_eval_{ts}.jsonl"
    summary_path = args.out_dir / f"{args.tag}_eval_{ts}_summary.json"
    # Hypotheses file in the schema the provided evaluate_qa_deepseek.py expects
    # (--hyp-file: one {question_id, hypothesis} object per line). The reference
    # file for that judge is the original samples JSONL (--ref-file).
    hyp_path = args.out_dir / f"{args.tag}_eval_{ts}_hyp.jsonl"
    print(f"[run_eval] writing per-sample -> {jsonl_path}")
    print(f"[run_eval] writing hypotheses -> {hyp_path}")

    correct_by_type: "OrderedDict[str, int]" = OrderedDict((t, 0) for t in QUESTION_TYPES)
    total_by_type: "OrderedDict[str, int]" = OrderedDict((t, 0) for t in QUESTION_TYPES)
    n_correct = 0
    n_done = 0
    n_error = 0
    n_truncated = 0
    ttfts: list[float] = []
    latencies: list[float] = []

    run_t0 = time.monotonic()
    with jsonl_path.open("w", encoding="utf-8") as out, \
            hyp_path.open("w", encoding="utf-8") as hyp_out:
        for pos, (idx, s) in enumerate(samples):
            res = client.chat(render_messages(s))
            total_by_type[s.question_type] = total_by_type.get(s.question_type, 0) + 1

            verdict = None
            judge_raw = None
            judge_err = None
            if res.error:
                n_error += 1
            else:
                if res.truncated:
                    n_truncated += 1
                if res.ttft_s is not None:
                    ttfts.append(res.ttft_s)
                latencies.append(res.latency_s)
                if judge is not None:
                    jr = judge.judge(s, res.answer)
                    verdict = jr.correct
                    judge_raw = jr.raw
                    judge_err = jr.error
                    if jr.correct:
                        n_correct += 1
                        correct_by_type[s.question_type] = correct_by_type.get(s.question_type, 0) + 1
            n_done += 1

            row = {
                "subset_index": idx,
                "question_id": s.question_id,
                "question_type": s.question_type,
                "question": s.question,
                "gold": s.answer,
                "answer": res.answer,
                "reasoning_chars": len(res.reasoning),
                "correct": verdict,
                "judge_raw": judge_raw,
                "judge_error": judge_err,
                "ttft_s": res.ttft_s,
                "latency_s": res.latency_s,
                "finish_reason": res.finish_reason,
                "truncated": res.truncated,
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "client_error": res.error,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            hyp_out.write(json.dumps(
                {"question_id": s.question_id, "hypothesis": res.answer},
                ensure_ascii=False) + "\n")
            hyp_out.flush()

            acc = (n_correct / n_done * 100.0) if n_done else 0.0
            flag = "ERR" if res.error else ("CUT" if res.truncated else "ok ")
            v = "—" if verdict is None else ("Y" if verdict else "N")
            print(f"[{pos+1:>3}/{len(samples)}] {flag} {s.question_type:<26} "
                  f"verdict={v} ttft={res.ttft_s if res.ttft_s is None else round(res.ttft_s,1)}s "
                  f"lat={res.latency_s:6.1f}s acc={acc:5.1f}% "
                  f"id={s.question_id}"
                  + (f" err={res.error[:80]}" if res.error else ""))

    run_wall = time.monotonic() - run_t0

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
        "base_url": args.base_url,
        "model": args.model,
        "enable_thinking": not args.no_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "judged": judge is not None,
        "judge_model": (judge.model if judge is not None else None),
        "n_samples": n_done,
        "n_correct": n_correct,
        "overall_accuracy": overall_acc,
        "n_error": n_error,
        "n_truncated": n_truncated,
        "per_type": per_type,
        "mean_ttft_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else None,
        "run_wall_s": run_wall,
        "jsonl_path": str(jsonl_path),
        "hyp_path": str(hyp_path),
        "ref_path": str(args.data),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- human-readable report -------------------------------------------------
    print("\n=== KVMem utility eval summary ===")
    print(f"  endpoint={args.base_url} model={args.model} thinking={not args.no_thinking}")
    print(f"  samples={n_done}  errors={n_error}  truncated={n_truncated}  "
          f"wall={run_wall/60:.1f} min")
    if judge is not None:
        print(f"  OVERALL: {n_correct}/{n_done} = {overall_acc:.2f}%")
        print("  per-type:")
        for t in QUESTION_TYPES:
            d = per_type[t]
            a = "n/a" if d["accuracy"] is None else f"{d['accuracy']:5.1f}%"
            print(f"    {t:<28} {d['correct']:>2}/{d['total']:<2}  {a}")
        print("  reference baselines (doc §4.1, same model family):")
        for name, val in DOC_BASELINES.items():
            print(f"    {name:<28} {val:5.2f}%")
    else:
        print("  (judge disabled; accuracy not computed)")
    if ttfts:
        print(f"  mean TTFT={summary['mean_ttft_s']:.1f}s  mean latency={summary['mean_latency_s']:.1f}s")
    print(f"  per-sample : {jsonl_path}")
    print(f"  hypotheses : {hyp_path}")
    print(f"  summary    : {summary_path}")
    if args.no_judge:
        print("\n  grade with the provided judge:")
        print(f"    python3 evaluate_qa_deepseek.py --hyp-file {hyp_path} \\")
        print(f"      --ref-file {args.data} --output {args.out_dir}/{args.tag}_eval_{ts}_graded.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
