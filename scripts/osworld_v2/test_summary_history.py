#!/usr/bin/env python3
"""Whole-history compaction keeps stable prefixes and commits only valid summaries."""
import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/OSWorld-V2"))
from mm_agents.qwen35vl_agent import Qwen35VLAgent
from mm_agents.qw3_summary_history import SummaryHistory, SUMMARY_INSTRUCTION


def images(messages):
    return sum(p["type"] == "image_url" for m in messages for p in m["content"])


class FakeAgent(Qwen35VLAgent):
    def __init__(self, **kwargs):
        super().__init__(history_mode="summary", context_tokens=32768,
                         summary_trigger_tokens=12000, summary_max_tokens=2048,
                         max_tokens=256, history_n=2, image_max=2, fold_size=1,
                         enable_thinking=False, preserve_thinking=False, **kwargs)
        self.calls = []
        self.bad_summary = False
        self.summary_outputs = []
        self.summary_history.count_text = lambda agent, text: len(text.encode()) // 4 + 1

    def call_llm(self, payload, model):
        self.calls.append(deepcopy(payload))
        text = "\n".join(p.get("text", "") for m in payload["messages"] for p in m["content"])
        summarizing = SUMMARY_INSTRUCTION in text
        self.last_api_metadata = {
            "usage": {"prompt_tokens": len(text.encode()) // 4 + 32 * len(payload["messages"]) + 2040 * images(payload["messages"]),
                      "completion_tokens": 40},
            "finish_reason": "length" if summarizing and self.bad_summary else "stop",
            "attempts": [{"attempt": 1, "status": "success"}],
        }
        if summarizing:
            if self.summary_outputs:
                response, finish = self.summary_outputs.pop(0)
                self.last_api_metadata["finish_reason"] = finish
                return response
            return "State: file /tmp/keep-me.txt exists. Task unfinished. Next: inspect current screenshot."
        return "Action: Observe.\n<tool_call><function=computer_use><parameter=action>screenshot</parameter></function></tool_call>"


class SummaryHistoryTest(unittest.TestCase):
    def setUp(self):
        output = BytesIO()
        Image.new("RGB", (1920, 1088)).save(output, "PNG")
        self.png = output.getvalue()
        self.image_patch = patch("mm_agents.qwen35vl_agent.process_image",
                                 return_value=base64.b64encode(self.png).decode())
        self.image_patch.start()
        self.addCleanup(self.image_patch.stop)

    def test_append_and_multiple_complete_history_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeAgent()
            agent.set_trace_dir(tmp)
            previous = None
            events = []
            for step in range(1, 13):
                agent.predict("Original task: retain exact value 4321.", {"screenshot": self.png})
                messages = agent.calls[-1]["messages"]
                event = agent.last_step_trace["summary_compaction"]
                self.assertFalse(agent.calls[-1]["enable_thinking"])
                self.assertFalse(agent.last_step_trace["compaction"]["triggered"])
                if event:
                    events.append(event)
                    self.assertEqual(images(messages), 1)
                    self.assertEqual(images(agent.calls[-2]["messages"]), event["removed_images"])
                    self.assertEqual(agent.calls[-2]["messages"][-1]["role"], "user")
                    self.assertEqual(agent.calls[-2]["messages"][-1]["content"][0]["text"], "<tool_response>\n")
                    self.assertIn("4321", str(messages))
                    self.assertIn("/tmp/keep-me.txt", str(messages))
                elif previous:
                    self.assertEqual(messages[:len(previous)], previous)
                self.assertLess(agent.last_api_metadata["usage"]["prompt_tokens"],
                                agent.last_step_trace["estimated_prompt_tokens"])
                previous = messages
            self.assertGreaterEqual(len(events), 2)
            self.assertEqual(agent.folded_prefix_k, 0)
            saved = [json.loads(x) for x in (Path(tmp) / "history_compaction.jsonl").read_text().splitlines()]
            self.assertEqual(len(saved), len(events))
            self.assertTrue(all(x["status"] == "success" for x in saved))
            agent.reset()
            self.assertEqual(agent.summary_history.start_step, 1)
            self.assertEqual(agent.summary_history.summary, "")
            self.assertIsNone(agent.summary_history.last_prompt_tokens)

    def test_summary_mode_does_not_apply_small_legacy_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeAgent()
            agent.summary_history.trigger_tokens = 27000
            agent.set_trace_dir(tmp)
            for _ in range(6):
                agent.predict("Observe.", {"screenshot": self.png})
            self.assertEqual(images(agent.calls[-1]["messages"]), 6)
            self.assertEqual(agent.summary_history.start_step, 1)

    def test_truncated_summary_does_not_discard_old_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeAgent()
            agent.set_trace_dir(tmp)
            agent.bad_summary = True
            with self.assertRaisesRegex(ValueError, "complete plain-text summary"):
                for _ in range(10):
                    agent.predict("Keep original.", {"screenshot": self.png})
            self.assertEqual(agent.summary_history.start_step, 1)
            self.assertEqual(agent.summary_history.summary, "")
            self.assertEqual(agent.last_step_trace["status"], "error")
            self.assertEqual(agent.last_step_trace["summary_compaction"]["status"], "error")
            self.assertEqual(len(agent.last_step_trace["summary_compaction"]["attempts"]), 3)

    def test_summary_tool_suffix_is_discarded_without_executing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeAgent()
            agent.set_trace_dir(tmp)
            summary = "Task unfinished. The file /tmp/keep-me.txt exists. Current screen is the editor; inspect it before continuing."
            agent.summary_outputs = [(summary + "\n<tool_call><function=computer_use>delete</function></tool_call>", "tool_calls")]
            for _ in range(10):
                _, actions = agent.predict("Keep original.", {"screenshot": self.png})
                event = agent.last_step_trace["summary_compaction"]
                if event:
                    break
            self.assertEqual(agent.summary_history.summary, summary)
            self.assertEqual(event["status"], "success")
            self.assertTrue(event["attempts"][0]["discarded_tool_or_reasoning_suffix"])
            self.assertNotIn("delete", str(agent.calls[-1]["messages"]))
            self.assertNotIn("delete", str(actions))
            self.assertEqual(agent.calls[-2]["stop"], SummaryHistory.SUMMARY_STOPS)
            self.assertTrue(agent.calls[-2]["response_text_only"])

    def test_invalid_outputs_retry_then_isolate_all_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = FakeAgent()
            agent.set_trace_dir(tmp)
            summary = "Task unfinished. The file /tmp/keep-me.txt exists. Current screen is the editor; inspect it before continuing."
            agent.summary_outputs = [("Action: click the editor.", "tool_calls"),
                                     (summary, "length"), (summary, "stop")]
            for _ in range(10):
                agent.predict("Keep original marker 4321.", {"screenshot": self.png})
                event = agent.last_step_trace["summary_compaction"]
                if event:
                    break
            self.assertEqual([a["status"] for a in event["attempts"]], ["invalid", "invalid", "success"])
            self.assertEqual(event["attempts"][-1]["mode"], "isolated_summary")
            attempts = agent.calls[-4:-1]
            self.assertEqual([images(a["messages"]) for a in attempts], [event["removed_images"]] * 3)
            self.assertIn("4321", str(attempts[-1]["messages"]))
            self.assertEqual(agent.summary_history.summary, summary)
            self.assertEqual(images(agent.calls[-1]["messages"]), 1)

    def test_valid_text_with_native_tool_finish_is_usable(self):
        text = "The user needs to finish the form. Name is Alex; the document is saved on Desktop. Next inspect the newest screenshot."
        clean, discarded = SummaryHistory.clean_summary(text, {"finish_reason": "tool_calls"})
        self.assertEqual(clean, text)
        self.assertTrue(discarded)
        for text in ("<tool_call>click</tool_call>", "<think>unfinished reasoning", "Action: " + "click " * 30):
            with self.assertRaises(ValueError):
                SummaryHistory.clean_summary(text, {"finish_reason": "tool_calls"})

    def test_reserve_validation(self):
        with self.assertRaises(ValueError):
            SummaryHistory(262144, 250000, 4096, 16384)


if __name__ == "__main__":
    unittest.main()
