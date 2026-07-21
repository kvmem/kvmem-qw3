#!/usr/bin/env python3
"""Render a LongMemEval-S sample into OpenAI chat messages.

Mirrors the Full Context layout from docs/motivation_experiment_summary_en.md
(section 3.1):

    System instruction
    Question date
    Full history sessions
    Question

The history goes into one user message and the question into a SEPARATE final
user message, so the question is the very last thing the model reads AND occupies
its own message turn. This matters for query-conditioned KVMem: the server marks
the final user message's token span (`--kvmem-query-conditioned`) and the executor
ranks the 32K decode window by multi-token mean relevance to exactly those
question tokens (the KV-cache analog of RAG embedding the query to rank chunks),
instead of falling back to a recency window.
"""

from __future__ import annotations

from typing import Any

try:
    from .dataset import Sample
except ImportError:  # allow running as a loose module
    from dataset import Sample  # type: ignore


SYSTEM_INSTRUCTION = (
    "You are a helpful personal assistant with long-term memory of your previous "
    "conversations with the user. The conversation history below contains many "
    "dated chat sessions between you (the assistant) and the user. Use the relevant "
    "information from those sessions to answer the user's final question accurately "
    "and concisely. If multiple sessions contain conflicting information, prefer the "
    "most recent one. Today's date is {question_date}."
)


def render_history(sample: Sample) -> str:
    """Render all haystack sessions as a single dated, role-labeled text block."""
    lines: list[str] = []
    dates = sample.haystack_dates
    for idx, session in enumerate(sample.haystack_sessions):
        date = dates[idx] if idx < len(dates) else ""
        header = f"=== Conversation on {date} ===" if date else f"=== Conversation {idx + 1} ==="
        lines.append(header)
        for turn in session:
            role = str(turn.get("role", "")).strip().lower()
            content = str(turn.get("content", "")).strip()
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        lines.append("")  # blank line between sessions
    return "\n".join(lines).rstrip()


def render_messages(sample: Sample) -> list[dict[str, Any]]:
    """Build the OpenAI `messages` array: a system turn, one user turn holding the
    full history, and a SEPARATE final user turn holding only the question. The
    question being its own final message lets the server isolate its token span
    for query-conditioned multi-token block selection."""
    system = SYSTEM_INSTRUCTION.format(question_date=sample.question_date or "unknown")
    history = render_history(sample)
    history_user = (
        f"{history}\n\n"
        f"=== End of conversation history ==="
    )
    question_user = (
        f"Based on the conversations above, answer the following question.\n"
        f"Question (asked on {sample.question_date}): {sample.question}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": history_user},
        {"role": "user", "content": question_user},
    ]


def render_transcript_messages(sample: Sample) -> list[dict[str, Any]]:
    """Render the benchmark history as its original user/assistant turns.

    This is the experimental lifecycle form used to simulate a long-lived chat:
    once the 224K working set plus 32K reserve is full, each subsequent user
    message can trigger re-selection, while the following recorded assistant
    message is teacher-forced prefill rather than newly decoded text. Session
    dates are attached to the first real message of each session, so no synthetic
    user query is introduced solely for metadata.
    """
    messages: list[dict[str, Any]] = [{
        "role": "system",
        "content": SYSTEM_INSTRUCTION.format(
            question_date=sample.question_date or "unknown"),
    }]
    dates = sample.haystack_dates
    for idx, session in enumerate(sample.haystack_sessions):
        date = dates[idx] if idx < len(dates) else ""
        header = (f"=== Conversation on {date} ===" if date
                  else f"=== Conversation {idx + 1} ===")
        first_emitted = True
        for turn in session:
            role = str(turn.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(turn.get("content", "")).strip()
            if first_emitted:
                content = f"{header}\n{content}"
            message = {"role": role, "content": content}
            if first_emitted:
                # Experimental metadata consumed only when
                # kvmem_transcript_replay is requested.  The chat renderer
                # ignores unknown message fields, so the prompt text remains
                # byte-for-byte identical while the server can map independent
                # LongMemEval sessions to exact token boundaries.
                message["kvmem_session_start"] = True
                first_emitted = False
            messages.append(message)
    question_user = (
        "=== End of conversation history ===\n\n"
        "Based on the conversations above, answer the following question.\n"
        f"Question (asked on {sample.question_date}): {sample.question}"
    )
    messages.append({"role": "user", "content": question_user})
    return messages


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    try:
        from .dataset import build_subset, load_all
    except ImportError:
        from dataset import build_subset, load_all  # type: ignore

    ap = argparse.ArgumentParser(description="Preview a rendered prompt.")
    ap.add_argument(
        "--data", type=Path,
        default=Path("/data/chaidi/kvmem_eval/data/longmemeval_s.json"),
    )
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    sub = build_subset(load_all(args.data))
    s = sub[args.index]
    msgs = render_messages(s)
    print(f"[{s.question_id}] type={s.question_type} gold={s.answer!r}")
    print("--- system ---")
    print(msgs[0]["content"])
    print("--- history user (head 800 chars) ---")
    print(msgs[1]["content"][:800])
    print("--- history user (tail 400 chars) ---")
    print(msgs[1]["content"][-400:])
    print(f"--- history user length: {len(msgs[1]['content'])} chars ---")
    print("--- question user (final message) ---")
    print(msgs[2]["content"])
