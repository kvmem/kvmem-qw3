#!/usr/bin/env python3
"""BEAM conversation loading and normalization.

The official 10M release stores each conversation as a list of plan objects.
Each plan contains batches, each batch contains turns, and each turn contains
ordinary OpenAI-style user/assistant messages.  This module flattens only those
chat messages, in the exact order used by BEAM's long-context baseline.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


QUESTION_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

ANSWER_FIELDS = (
    "ideal_response",
    "ideal_answer",
    "answer",
    "ideal_summary",
    "expected_compliance",
)


@dataclass(frozen=True)
class BeamQuestion:
    conversation_id: str
    question_type: str
    question_index: int
    question: str
    reference: str
    rubric: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def question_id(self) -> str:
        return (
            f"{self.conversation_id}:{self.question_type}:"
            f"{self.question_index}"
        )


@dataclass(frozen=True)
class BeamConversation:
    conversation_id: str
    messages: tuple[dict[str, str], ...]
    questions: tuple[BeamQuestion, ...]
    raw_questions: dict[str, list[dict[str, Any]]]
    source_dir: Path


def parse_id_list(value: str) -> list[str]:
    """Parse comma-separated ids and inclusive integer ranges."""
    values: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"descending range is not allowed: {part}")
            values.extend(str(index) for index in range(start, end + 1))
        else:
            values.append(part)
    if not values:
        raise ValueError("at least one conversation id is required")
    return values


def _question_object(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, dict):
        raise ValueError(
            f"probing_questions must be an object or encoded object, got "
            f"{type(value).__name__}"
        )
    normalized: dict[str, list[dict[str, Any]]] = {}
    for question_type, questions in value.items():
        if not isinstance(questions, list):
            raise ValueError(
                f"probing question group {question_type!r} is not a list"
            )
        normalized[str(question_type)] = []
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError(
                    f"question in {question_type!r} is not an object"
                )
            normalized[str(question_type)].append(dict(question))
    return normalized


def _reference(question: dict[str, Any]) -> str:
    for field in ANSWER_FIELDS:
        value = question.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    rubric = question.get("rubric")
    if isinstance(rubric, list):
        return "\n".join(str(item) for item in rubric)
    return ""


def normalize_questions(
    conversation_id: str,
    raw_questions: dict[str, list[dict[str, Any]]],
) -> tuple[BeamQuestion, ...]:
    unknown = set(raw_questions) - set(QUESTION_TYPES)
    if unknown:
        raise ValueError(f"unknown BEAM question types: {sorted(unknown)}")
    result: list[BeamQuestion] = []
    for question_type in QUESTION_TYPES:
        for index, raw in enumerate(raw_questions.get(question_type, [])):
            text = str(raw.get("question") or "").strip()
            if not text:
                raise ValueError(
                    f"empty question at {conversation_id}:{question_type}:{index}"
                )
            rubric_value = raw.get("rubric") or []
            if not isinstance(rubric_value, list):
                raise ValueError(
                    f"rubric is not a list at "
                    f"{conversation_id}:{question_type}:{index}"
                )
            result.append(
                BeamQuestion(
                    conversation_id=conversation_id,
                    question_type=question_type,
                    question_index=index,
                    question=text,
                    reference=_reference(raw),
                    rubric=tuple(str(item) for item in rubric_value),
                    raw=dict(raw),
                )
            )
    return tuple(result)


def flatten_10m_chat(data: Any) -> tuple[dict[str, str], ...]:
    """Flatten the official 10M plan/batch/turn structure."""
    if not isinstance(data, list):
        raise ValueError("BEAM 10M chat must be a list of plan objects")
    messages: list[dict[str, str]] = []
    seen_ids: set[int] = set()
    prior_id: int | None = None
    for plan_index, plan in enumerate(data):
        if not isinstance(plan, dict) or len(plan) != 1:
            raise ValueError(
                f"plan {plan_index} must be a single-key object"
            )
        batches = next(iter(plan.values()))
        if batches is None:
            continue
        if not isinstance(batches, list):
            raise ValueError(f"plan {plan_index} batches must be a list")
        for batch_index, batch in enumerate(batches):
            if not isinstance(batch, dict):
                raise ValueError(
                    f"plan {plan_index} batch {batch_index} is not an object"
                )
            turns = batch.get("turns")
            if not isinstance(turns, list):
                raise ValueError(
                    f"plan {plan_index} batch {batch_index} has no turns list"
                )
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, list):
                    raise ValueError(
                        f"plan {plan_index} batch {batch_index} turn "
                        f"{turn_index} is not a list"
                    )
                for raw in turn:
                    if not isinstance(raw, dict):
                        raise ValueError("BEAM message is not an object")
                    role = str(raw.get("role") or "").strip().lower()
                    if role not in {"user", "assistant"}:
                        continue
                    content = str(raw.get("content") or "")
                    messages.append({"role": role, "content": content})
                    raw_id = raw.get("id")
                    if isinstance(raw_id, int):
                        if raw_id in seen_ids:
                            raise ValueError(f"duplicate BEAM chat id: {raw_id}")
                        if prior_id is not None and raw_id <= prior_id:
                            raise ValueError(
                                f"BEAM chat ids are not increasing: "
                                f"{prior_id} then {raw_id}"
                            )
                        seen_ids.add(raw_id)
                        prior_id = raw_id
    if not messages:
        raise ValueError("BEAM conversation contains no user/assistant messages")
    return tuple(messages)


def _conversation_dirs(root: Path) -> Iterator[tuple[str, Path]]:
    if (root / "chat.json").is_file():
        yield root.name, root
        return
    for child in sorted(
        (entry for entry in root.iterdir() if entry.is_dir()),
        key=lambda path: int(path.name) if path.name.isdigit() else path.name,
    ):
        if (child / "chat.json").is_file():
            yield child.name, child


def discover_conversation_ids(root: Path) -> list[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"BEAM data directory not found: {root}")
    return [conversation_id for conversation_id, _ in _conversation_dirs(root)]


def load_conversation(root: Path, conversation_id: str) -> BeamConversation:
    source_dir = (
        root
        if (root / "chat.json").is_file() and root.name == conversation_id
        else root / conversation_id
    )
    chat_path = source_dir / "chat.json"
    questions_path = source_dir / "probing_questions" / "probing_questions.json"
    if not chat_path.is_file():
        raise FileNotFoundError(f"BEAM chat not found: {chat_path}")
    if not questions_path.is_file():
        raise FileNotFoundError(
            f"BEAM probing questions not found: {questions_path}"
        )
    chat = json.loads(chat_path.read_text(encoding="utf-8"))
    raw_questions = _question_object(
        json.loads(questions_path.read_text(encoding="utf-8"))
    )
    return BeamConversation(
        conversation_id=conversation_id,
        messages=flatten_10m_chat(chat),
        questions=normalize_questions(conversation_id, raw_questions),
        raw_questions=raw_questions,
        source_dir=source_dir,
    )
