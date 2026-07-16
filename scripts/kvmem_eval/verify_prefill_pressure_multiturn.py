#!/usr/bin/env python3
"""Small API smoke test for deterministic multi-turn prefill pressure.

The server is expected to run with KVMem prefix caching and trace logging.  Two
independent conversations first build an above-budget semantic window, then
append a long suffix followed by a short final user query.  Server-log checks
for the warm hit and ``[bs-prefill-pressure]`` events live in the shell driver.
"""

from __future__ import annotations

import argparse

try:
    from .client import Qw3Client
except ImportError:
    from client import Qw3Client  # type: ignore


def numbered_context(prefix: str, count: int) -> str:
    return "\n".join(
        f"{prefix} record {i:04d}: value-{(i * 7919) % 104729:06d}."
        for i in range(count)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18087/v1")
    ap.add_argument("--model", default="Qwen3.6-27B-Q8_0.gguf")
    ap.add_argument("--base-records", type=int, default=180)
    ap.add_argument("--suffix-records", type=int, default=120)
    args = ap.parse_args()

    client = Qw3Client(
        base_url=args.base_url,
        model=args.model,
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        enable_thinking=False,
        read_timeout=600.0,
    )
    if not client.health():
        raise SystemExit(f"server is not healthy: {args.base_url}")

    for scenario, warm_question in (
        ("alpha", "Which alpha record has value-000000? Reply briefly."),
        ("beta", "Which beta record appears last? Reply briefly."),
    ):
        system = (
            f"Scenario {scenario}. Treat all records below as context.\n"
            + numbered_context(scenario, args.base_records)
        )
        warm_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": warm_question},
        ]
        warm = client.chat(
            warm_messages,
            max_tokens=8,
            temperature=0.0,
            enable_thinking=False,
        )
        if warm.error or not warm.answer.strip():
            raise RuntimeError(
                f"scenario={scenario} warm request failed: "
                f"error={warm.error!r} answer={warm.answer!r}"
            )

        # The final user message is intentionally short.  The large newly
        # appended user/assistant exchange is context, so query detection and
        # prefill-pressure scheduling exercise separate spans.
        suffix = numbered_context(f"{scenario}-suffix", args.suffix_records)
        final_messages = [
            *warm_messages,
            {"role": "assistant", "content": warm.answer.strip()},
            {"role": "user", "content": suffix},
            {"role": "assistant", "content": "Context received."},
            {"role": "user", "content": "What is the scenario name? Reply with one word."},
        ]
        final = client.chat(
            final_messages,
            max_tokens=8,
            temperature=0.0,
            enable_thinking=False,
        )
        if final.error or not final.answer.strip():
            raise RuntimeError(
                f"scenario={scenario} final request failed: "
                f"error={final.error!r} answer={final.answer!r}"
            )
        print(
            f"scenario={scenario} warm={warm.answer.strip()!r} "
            f"final={final.answer.strip()!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
