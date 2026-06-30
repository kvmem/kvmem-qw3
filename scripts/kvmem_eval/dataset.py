#!/usr/bin/env python3
"""LongMemEval-S dataset loading + deterministic 102-sample subset.

The motivation study (docs/motivation_experiment_summary_en.md, section 2.1) uses a
balanced 102-sample subset: 17 samples per question_type across the 6 types. The
study does not publish its exact sample IDs, so we reconstruct the same *recipe*
deterministically: within each question_type, sort by question_id ascending and
take the first N (default 17). This is fully reproducible and independent of any
RNG seed.

Source file: the official `longmemeval_s` JSON from HF dataset
`xiaowu0162/longmemeval` (download via the HF mirror to /data; root / is full).
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The 6 LongMemEval question types, in the order presented in the doc (section 2.1).
QUESTION_TYPES: list[str] = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


@dataclass
class Sample:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    haystack_dates: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, Any]]]
    answer_session_ids: list[str]

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "Sample":
        return cls(
            question_id=str(raw["question_id"]),
            question_type=str(raw["question_type"]),
            question=str(raw["question"]),
            answer=str(raw["answer"]),
            question_date=str(raw.get("question_date", "")),
            haystack_dates=list(raw.get("haystack_dates", [])),
            haystack_session_ids=list(raw.get("haystack_session_ids", [])),
            haystack_sessions=list(raw.get("haystack_sessions", [])),
            answer_session_ids=list(raw.get("answer_session_ids", [])),
        )


def load_all(path: Path) -> list[Sample]:
    """Load samples from a JSON array, JSONL, or concatenated/pretty-printed JSON
    objects. The original HF `longmemeval_s.json` is a JSON array; the provided
    `selected_12_samples.jsonl` is a stream of concatenated (pretty-printed) JSON
    objects, so we fall back to a raw_decode loop that tolerates arbitrary
    whitespace (incl. newlines) between objects."""
    text = path.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
        if not isinstance(raw, list):
            raise ValueError(
                f"expected a JSON list of samples, got {type(raw).__name__}"
            )
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        raw = []
        idx = 0
        n = len(text)
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break
            obj, end = decoder.raw_decode(text, idx)
            raw.append(obj)
            idx = end
    return [Sample.from_raw(r) for r in raw]


def is_abstention(s: Sample) -> bool:
    """Abstention questions (id suffix `_abs`) have no answer in the haystack; the
    gold 'answer' is an instruction to abstain. The motivation study frames answers
    of 'information not available' as failures (section 4 error analysis), i.e. it
    used answerable-only questions, so we exclude abstention by default."""
    return s.question_id.endswith("_abs")


def build_subset(
    samples: list[Sample],
    per_type: int = 17,
    include_abstention: bool = False,
) -> list[Sample]:
    """Deterministic balanced subset: first `per_type` ids (sorted) per question_type.

    Returned order is grouped by QUESTION_TYPES, ids sorted ascending within a type.
    Raises if any type has fewer than `per_type` samples. Abstention questions are
    excluded unless `include_abstention` is set.
    """
    by_type: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        if not include_abstention and is_abstention(s):
            continue
        by_type[s.question_type].append(s)

    subset: list[Sample] = []
    for qtype in QUESTION_TYPES:
        pool = sorted(by_type.get(qtype, []), key=lambda s: s.question_id)
        if len(pool) < per_type:
            raise ValueError(
                f"question_type {qtype!r} has only {len(pool)} samples, "
                f"need {per_type}"
            )
        subset.extend(pool[:per_type])
    return subset


def type_histogram(samples: list[Sample]) -> "OrderedDict[str, int]":
    hist: "OrderedDict[str, int]" = OrderedDict((t, 0) for t in QUESTION_TYPES)
    for s in samples:
        hist[s.question_type] = hist.get(s.question_type, 0) + 1
    return hist


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the LongMemEval-S 102 subset.")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("/data/chaidi/kvmem_eval/data/longmemeval_s.json"),
    )
    ap.add_argument("--per-type", type=int, default=17)
    ap.add_argument("--include-abstention", action="store_true")
    args = ap.parse_args()

    alls = load_all(args.data)
    print(f"loaded {len(alls)} samples; full histogram: {dict(type_histogram(alls))}")
    sub = build_subset(alls, args.per_type, args.include_abstention)
    print(f"subset {len(sub)} samples; histogram: {dict(type_histogram(sub))}")
    for s in sub[:3]:
        print(f"  {s.question_id} [{s.question_type}] sessions={len(s.haystack_sessions)} "
              f"q={s.question[:60]!r}")
