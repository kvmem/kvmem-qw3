#!/usr/bin/env python3
"""Reproduce LongMemEval accuracy changes under two-request KVMem reuse.

Modes:

* ``oneshot`` sends the canonical system + history + final-question messages.
* ``warm-query`` first sends system + history + a separate short user
  confirmation query.  It then continues the conversation with the warm-up
  assistant response and appends the real question as the newest user message.
  With ``--kvmem-prefix-cache`` on the server, the second request reuses the
  first turn while the last-user query policy sees a short query in both turns.

The two modes intentionally exercise different conversation structures.  The
oneshot arm remains the canonical accuracy reference; the warm-query arm models
an actual two-turn agent/chat session.  The harness records warm-up wall time
separately and reports TTFT only for the final query.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .client import Qw3Client
    from .dataset import load_all
    from .prompt import render_messages
except ImportError:
    from client import Qw3Client  # type: ignore
    from dataset import load_all  # type: ignore
    from prompt import render_messages  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--indices", required=True,
                    help="comma-separated use-all dataset indices")
    ap.add_argument("--mode", choices=("oneshot", "warm-query"), required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8086/v1")
    ap.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--warm-max-tokens", type=int, default=64,
                    help="completion budget for the short warm-up confirmation")
    ap.add_argument(
        "--warm-query",
        default="Please remember the information above and reply with a brief confirmation.",
        help="separate final user message used as the warm-up retrieval query",
    )
    ap.add_argument("--read-timeout", type=float, default=3600.0)
    args = ap.parse_args()

    samples = load_all(args.data)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    client = Qw3Client(
        base_url=args.base_url,
        model=args.model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        enable_thinking=True,
        read_timeout=args.read_timeout,
    )
    if not client.health():
        raise SystemExit(f"server is not healthy: {args.base_url}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for ordinal, idx in enumerate(indices, 1):
            sample = samples[idx]
            messages = render_messages(sample)
            warm = None
            warm_wall_s = 0.0
            final_messages = messages
            if args.mode == "warm-query":
                # Never let the long history be the last user message: under the
                # server's automatic policy that would make the history itself the
                # capped retrieval query.  The explicit confirmation is short and
                # also activates full incremental mean-k index construction.
                warm_user = {"role": "user", "content": args.warm_query}
                warm_messages = [*messages[:2], warm_user]
                t0 = time.monotonic()
                warm = client.chat(
                    warm_messages, max_tokens=args.warm_max_tokens, temperature=0.0,
                    enable_thinking=False,
                )
                warm_wall_s = time.monotonic() - t0
                if warm.error:
                    raise RuntimeError(f"idx={idx} warm-up failed: {warm.error}")
                warm_answer = warm.answer.strip()
                if not warm_answer:
                    raise RuntimeError(f"idx={idx} warm-up returned an empty confirmation")

                # Model a real next turn.  The confirmation response is historical
                # assistant context; the canonical LongMemEval question is appended
                # as the newest user message and is therefore the new query.
                final_messages = [
                    *warm_messages,
                    {"role": "assistant", "content": warm_answer},
                    messages[2],
                ]

            result = client.chat(
                final_messages, max_tokens=args.max_tokens, temperature=0.0,
                top_p=1.0, enable_thinking=True,
            )
            row = {
                "mode": args.mode,
                "ordinal": ordinal,
                "subset_index": idx,
                "question_id": sample.question_id,
                "question_type": sample.question_type,
                "question": sample.question,
                "gold": sample.answer,
                "answer": result.answer,
                "reasoning": result.reasoning,
                "query_ttft_s": result.ttft_s,
                "query_latency_s": result.latency_s,
                "query_finish_reason": result.finish_reason,
                "query_error": result.error,
                "warm_query": args.warm_query if warm else None,
                "warm_answer": warm.answer if warm else None,
                "warm_wall_s": warm_wall_s,
                "warm_ttft_s": warm.ttft_s if warm else None,
                "warm_finish_reason": warm.finish_reason if warm else None,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            preview = result.answer.replace("\n", " ")[:160]
            print(
                f"[{ordinal}/{len(indices)}] idx={idx} {sample.question_type} "
                f"query_ttft={result.ttft_s!r}s warm={warm_wall_s:.1f}s "
                f"answer={preview!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()
