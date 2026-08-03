#!/usr/bin/env python3
"""Render one official MemoryAgentBench row as a frozen Qwen archive prefix.

The official LongContextAgent concatenates one ``memorize`` template per
4096-token sentence chunk, then sends that text and the final query in one
user message.  This tool reproduces the memorized portion and deliberately
leaves the Qwen user turn open.  ``qw3 archive query`` later appends each
question with ``--archive-question-format qwen-user-continuation`` so all
questions branch from the same exact prefix without adding an extra role turn.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import nltk
import pyarrow.parquet as pq
import tiktoken


SYSTEM_MESSAGE = (
    "You are a helpful assistant that can read the context and memorize it "
    "for future retrieval."
)

MEMORIZE = {
    "ruler": "Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the documents I have read: \n{context}\n <Assistant> I have learned the documents and I will answer the question you ask.",
    "longmemeval": "Dialogue between User and Assistant \\n<User> The following context is the conversation between the user and the assistant: \n{context}\n <Assistant> I have memorized the conversation and I will answer the question you ask.",
    "eventqa": "Dialogue between User and Assistant {time_stamp}\\n<User> The following context is the book excerpt: \n{context}\n <Assistant> I have read the book excerpt and I will answer the question you ask.",
    "icl": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the examples I have learned: \n{context}\n <Assistant> I have learned the examples and I will answer the question you ask.",
    "recsys": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the dialogues between a user and recommender system: \n{context}\n <Assistant> I have memorized the dialogues and I will answer the question you ask.",
    "infbench": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
    "detective": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the book I have read: \n{context}\n <Assistant> I have read the book and I will answer the question you ask.",
    "fact": "Dialogue between User and Assistant {time_stamp} \\n<User> The following context is the facts I have learned: \n{context}\n <Assistant> I have learned the facts and I will answer the question you ask.",
}

QUERY = {
    "ruler": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\nQuestion: {question} \n\n Answer:",
    "longmemeval": "The history chats are between you and a user. Based on the relevant chat history, answer the question as concisely as you can, using a single phrase if possible.\n\n {question} \n\n Answer:",
    "eventqa": "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
    "icl": "Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output \"label: {label}\" and nothing else. \n\n{question} \n\n label:",
    "recsys": "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
    "infbench": "You are given a book above and you are tasked to summarize it. \n\n{question} \n\n Now summarize the book.",
    "detective": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
    "fact": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
}


def family(source: str) -> str:
    if source.startswith("ruler_"):
        return "ruler"
    if source.startswith("longmemeval_"):
        return "longmemeval"
    if source.startswith("eventqa_"):
        return "eventqa"
    if source.startswith("icl_"):
        return "icl"
    if source.startswith("recsys_"):
        return "recsys"
    if source.startswith("infbench_"):
        return "infbench"
    if source.startswith("detective_"):
        return "detective"
    if source.startswith("factconsolidation_"):
        return "fact"
    raise ValueError(f"unsupported MemoryAgentBench source: {source}")


def official_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """Match the repository's ``chunk_text_into_sentences`` control flow."""
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    sentences = nltk.sent_tokenize(text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        n = len(enc.encode(sentence, allowed_special={"<|endoftext|>"}))
        if count + n > chunk_size:
            # This intentionally mirrors the upstream implementation, including
            # its behavior for a single over-size first sentence.
            chunks.append(" ".join(current))
            current = [sentence]
            count = n
        else:
            current.append(sentence)
            count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


def indexed(metadata: dict[str, Any], key: str, i: int) -> Any:
    value = metadata.get(key)
    if isinstance(value, list) and i < len(value):
        return value[i]
    return value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, required=True)
    p.add_argument("--row", type=int, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chunk-size", type=int, default=4096)
    p.add_argument(
        "--timestamp",
        default="2026-08-02 00:00:00",
        help="deterministic value for upstream templates that require a timestamp",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    table = pq.read_table(args.parquet)
    if args.row < 0 or args.row >= table.num_rows:
        raise IndexError(f"row {args.row} outside [0,{table.num_rows})")
    row = table.slice(args.row, 1).to_pylist()[0]
    metadata = row.get("metadata") or {}
    source = metadata.get("source", "")
    kind = family(source)
    chunks = official_chunks(row["context"], args.chunk_size)
    memorize_template = MEMORIZE[kind]
    rendered_chunks = [
        memorize_template.format(
            context=chunk,
            time_stamp=args.timestamp,
        )
        for chunk in chunks
    ]
    memorized_context = "\n".join(rendered_chunks).strip()

    # Leave the user turn open. The query command appends `\n{query}`, closes
    # the turn, and opens the assistant turn. Thus the token stream is the same
    # role structure as the official LongContextAgent's context+query request.
    archive_prefix = (
        f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
        f"<|im_start|>user\n{memorized_context}"
    )

    questions = row.get("questions") or []
    answers = row.get("answers") or []
    if not isinstance(questions, list):
        questions = [questions]
    if not isinstance(answers, list):
        answers = [answers]
    if len(questions) != len(answers):
        raise ValueError(
            f"questions/answers mismatch: {len(questions)} vs {len(answers)}"
        )
    query_template = QUERY[kind]
    formatted_questions = [
        query_template.format(question=question, label="{label}")
        for question in questions
    ]
    qa_ids = metadata.get("qa_pair_ids") or [None] * len(questions)
    if not isinstance(qa_ids, list):
        qa_ids = [qa_ids]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = args.out_dir / "archive_prefix.qwen-chat-open-user.txt"
    questions_path = args.out_dir / "questions.json"
    qa_path = args.out_dir / "qa.json"
    prefix_path.write_text(archive_prefix, encoding="utf-8")
    questions_path.write_text(
        json.dumps(formatted_questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    qa = []
    for i, (raw_q, formatted_q, answer) in enumerate(
        zip(questions, formatted_questions, answers)
    ):
        qa.append(
            {
                "question_index": i,
                "qa_pair_id": qa_ids[i] if i < len(qa_ids) else None,
                "raw_question": raw_q,
                "formatted_question": formatted_q,
                "answer": answer,
                "question_date": indexed(metadata, "question_dates", i),
                "question_type": indexed(metadata, "question_types", i),
                "question_id": indexed(metadata, "question_ids", i),
                "previous_event": indexed(metadata, "previous_events", i),
                "keypoints": metadata.get("keypoints"),
            }
        )
    qa_path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_repository_commit": "455306dcabc3842526eb83cd4e225e5d486c5c5d",
        "parquet": str(args.parquet.resolve()),
        "row": args.row,
        "source": source,
        "family": kind,
        "context_chars": len(row["context"]),
        "official_chunk_size": args.chunk_size,
        "official_chunks": len(chunks),
        "timestamp": args.timestamp,
        "questions": len(formatted_questions),
        "archive_prefix": str(prefix_path.resolve()),
        "questions_file": str(questions_path.resolve()),
        "qa_file": str(qa_path.resolve()),
        "prompt_structure": "official one-user context+query; archived user turn left open",
    }
    (args.out_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
