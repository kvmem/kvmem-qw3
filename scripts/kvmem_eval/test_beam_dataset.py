#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from beam_dataset import (
    flatten_10m_chat,
    load_conversation,
    parse_id_list,
)


class BeamDatasetTest(unittest.TestCase):
    def test_parse_id_list(self) -> None:
        self.assertEqual(parse_id_list("1,3-5"), ["1", "3", "4", "5"])
        with self.assertRaises(ValueError):
            parse_id_list("5-3")

    def test_flatten_official_shape(self) -> None:
        chat = [
            {
                "plan-1": [
                    {
                        "batch_number": 1,
                        "turns": [
                            [
                                {"role": "user", "id": 0, "content": "u0"},
                                {
                                    "role": "assistant",
                                    "id": 1,
                                    "content": "a0",
                                },
                            ],
                            [
                                {"role": "user", "id": 2, "content": "u1"},
                            ],
                        ],
                    }
                ]
            },
            {
                "plan-2": [
                    {
                        "batch_number": 1,
                        "turns": [
                            [
                                {
                                    "role": "assistant",
                                    "id": 3,
                                    "content": "a1",
                                }
                            ]
                        ],
                    }
                ]
            },
        ]
        self.assertEqual(
            flatten_10m_chat(chat),
            (
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ),
        )

    def test_load_conversation_and_reference_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            conversation = root / "1"
            questions = conversation / "probing_questions"
            questions.mkdir(parents=True)
            (conversation / "chat.json").write_text(
                json.dumps(
                    [
                        {
                            "plan-1": [
                                {
                                    "turns": [
                                        [
                                            {
                                                "role": "user",
                                                "id": 0,
                                                "content": "hello",
                                            },
                                            {
                                                "role": "assistant",
                                                "id": 1,
                                                "content": "hi",
                                            },
                                        ]
                                    ]
                                }
                            ]
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (questions / "probing_questions.json").write_text(
                json.dumps(
                    {
                        "abstention": [
                            {
                                "question": "unknown?",
                                "ideal_response": "not available",
                                "rubric": ["declines"],
                            }
                        ],
                        "summarization": [
                            {
                                "question": "summary?",
                                "ideal_summary": "brief",
                                "rubric": ["brief"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_conversation(root, "1")
            self.assertEqual(len(loaded.messages), 2)
            self.assertEqual(len(loaded.questions), 2)
            self.assertEqual(loaded.questions[0].reference, "not available")
            self.assertEqual(loaded.questions[1].reference, "brief")


if __name__ == "__main__":
    unittest.main()
