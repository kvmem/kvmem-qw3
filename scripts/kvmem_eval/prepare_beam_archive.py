#!/usr/bin/env python3
"""Prepare one BEAM conversation for a durable KVMem context archive.

The archive builder accepts already-rendered UTF-8 text.  BEAM stores nested
OpenAI-style messages, so this utility flattens them with ``beam_dataset`` and
renders the same user/assistant subset used by ``qw3_server.cpp`` when a
prefill-only request is made (there is deliberately no trailing assistant
generation header).

The generated corpus can be ingested once and the questions file can then be
reused by every archive-query optimization arm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .beam_dataset import load_conversation
except ImportError:
    from beam_dataset import load_conversation  # type: ignore


ASCII_WS = " \t\n\r\f\v"


def render_history(messages: tuple[dict[str, str], ...]) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported BEAM archive role: {role!r}")
        content = message["content"].strip(ASCII_WS)
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--conversation-id", default="1")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conversation = load_conversation(args.data, args.conversation_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = args.out_dir / "history.qwen-chat.txt"
    questions_path = args.out_dir / "questions.txt"
    metadata_path = args.out_dir / "metadata.json"
    corpus = render_history(conversation.messages)
    corpus_path.write_text(corpus, encoding="utf-8")
    questions_path.write_text(
        "".join(question.question.replace("\n", " ").strip() + "\n"
                for question in conversation.questions),
        encoding="utf-8",
    )
    metadata = {
        "dataset": "BEAM-10M",
        "conversation_id": conversation.conversation_id,
        "messages": len(conversation.messages),
        "questions": len(conversation.questions),
        "rendering": "qw3 prefill-only user/assistant chat subset",
        "corpus_bytes": len(corpus.encode("utf-8")),
        "corpus_path": str(corpus_path),
        "questions_path": str(questions_path),
        "question_ids": [question.question_id
                         for question in conversation.questions],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
