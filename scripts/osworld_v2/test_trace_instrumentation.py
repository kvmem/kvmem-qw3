#!/usr/bin/env python3
"""Focused tests for OSWorld compaction and latency trace instrumentation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


QW3_ROOT = Path(__file__).resolve().parents[2]
OSWORLD_ROOT = QW3_ROOT / "third_party" / "OSWorld-V2"
sys.path.insert(0, str(OSWORLD_ROOT))
sys.path.insert(0, str(QW3_ROOT / "scripts" / "osworld_v2"))

from analyze_compaction import load_condition  # noqa: E402
from mm_agents.qwen35vl_agent import Qwen35VLAgent  # noqa: E402


def screenshot_bytes() -> bytes:
    image = Image.new("RGB", (1920, 1080), color=(24, 32, 48))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FakeAgent(Qwen35VLAgent):
    def call_llm(self, payload, model):
        step = len(self.screenshots)
        active = sum(
            1
            for message in payload["messages"]
            for part in message.get("content", [])
            if part.get("type") == "image_url"
        )
        self.last_api_metadata = {
            "usage": {
                "prompt_tokens": 1000 + active * 100,
                "completion_tokens": 10,
                "total_tokens": 1010 + active * 100,
            },
            "server_timing": {
                "server_ttft_sec": 0.1,
                "engine_ttft_sec": 0.05,
                "response_ttft_sec": None,
                "request_total_sec": 0.2,
            },
            "attempts": [{"attempt": 1, "status": "success", "wall_time_ms": 200.0}],
            "response_id": f"fake-{step}",
        }
        return (
            "Action: Left click.\n"
            "<tool_call>\n<function=computer_use>\n"
            "<parameter=action>left_click</parameter>\n"
            "<parameter=coordinate>[500, 500]</parameter>\n"
            "</function>\n</tool_call>"
        )


class TraceInstrumentationTest(unittest.TestCase):
    def test_native_tool_response_preserves_reasoning_and_action_text(self):
        message = SimpleNamespace(
            content="Action: Click the center of the OK button.",
            reasoning_content="The attachment dialog is open and the OK button is visible.",
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="computer_use",
                        arguments=json.dumps(
                            {"action": "left_click", "coordinate": [452, 657]}
                        ),
                    )
                )
            ],
        )
        rendered = Qwen35VLAgent._extract_openai_message(message)
        self.assertIn("<think>", rendered)
        self.assertIn("attachment dialog", rendered)
        self.assertIn("Action: Click the center of the OK button.", rendered)
        self.assertEqual(rendered.count("<tool_call>"), 1)
        self.assertIn("[452, 657]", rendered)

        raw = (
            "<think>\nexact plan\n</think>\n\nAction: Exact bytes.\n"
            "<tool_call>x</tool_call>"
        )
        message.model_extra = {"qw3_raw_completion": raw}
        self.assertEqual(Qwen35VLAgent._extract_openai_message(message), raw)

        message.model_extra = {
            "qw3_raw_completion": raw
            + "<|endoftext|><|im_start|>user<|endoftext|>"
        }
        self.assertEqual(Qwen35VLAgent._extract_openai_message(message), raw)

        batched_raw = raw + "\n<tool_call>second</tool_call>"
        message.model_extra = {
            "qw3_raw_completion": batched_raw
            + "<|endoftext|><|im_start|>user\n"
            + "<tool_call>fabricated</tool_call>"
        }
        self.assertEqual(
            Qwen35VLAgent._extract_openai_message(message), batched_raw
        )

    def test_full_action_parser(self):
        agent = Qwen35VLAgent()
        cases = {
            "screenshot": ["pyautogui.screenshot()"],
            "hscroll": ["pyautogui.hscroll(-3)"],
            "triple_click": ["pyautogui.tripleClick(960, 540)"],
            "key_down": ['pyautogui.keyDown("ctrl")'],
            "key_up": ['pyautogui.keyUp("ctrl")'],
            "left_mouse_down": [
                "pyautogui.moveTo(960, 540)",
                "pyautogui.mouseDown(button='left')",
            ],
            "left_mouse_up": [
                "pyautogui.moveTo(960, 540)",
                "pyautogui.mouseUp(button='left')",
            ],
        }
        for action, expected in cases.items():
            params = [f"<parameter=action>{action}</parameter>"]
            if action in {"triple_click", "left_mouse_down", "left_mouse_up"}:
                params.append("<parameter=coordinate>[500, 500]</parameter>")
            if action in {"key_down", "key_up"}:
                params.append('<parameter=keys>["ctrl"]</parameter>')
            if action == "hscroll":
                params.append("<parameter=pixels>-3</parameter>")
            response = (
                "Action: Test.\n<tool_call><function=computer_use>"
                + "".join(params)
                + "</function></tool_call>"
            )
            _, parsed = agent.parse_response(
                response, original_width=1920, original_height=1080
            )
            self.assertEqual(parsed, expected, action)

    def test_folding_trace_and_exact_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            agent = FakeAgent(history_n=100, image_max=4, fold_size=2)
            agent.set_trace_dir(temporary)
            agent.reset()
            observed = []
            for _ in range(7):
                response, actions = agent.predict("Click.", {"screenshot": screenshot_bytes()})
                self.assertIn("tool_call", response)
                self.assertEqual(actions, ["pyautogui.click(960, 540)"])
                observed.append(
                    (
                        agent.last_step_trace["compaction"]["active_image_count"],
                        agent.last_step_trace["compaction"]["triggered"],
                    )
                )
            self.assertEqual(
                observed,
                [(1, False), (2, False), (3, False), (4, False), (3, True), (4, False), (3, True)],
            )
            trace_rows = [
                json.loads(line)
                for line in (Path(temporary) / "agent_step_trace.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(trace_rows), 7)
            self.assertEqual(len(list((Path(temporary) / "agent_inputs").glob("*.png"))), 7)
            self.assertEqual(len(list((Path(temporary) / "agent_messages").glob("*.json"))), 7)

    def test_analyzer_matches_server_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            task_dir = run_dir / "osworld" / "pyautogui" / "screenshot" / "gpt-qw3" / "tasks" / "052"
            task_dir.mkdir(parents=True)
            agent_trace = {
                "compaction": {
                    "triggered": True,
                    "newly_compacted": 2,
                    "folded_prefix_after": 2,
                    "history_start_step": 1,
                    "history_truncated_steps": 0,
                    "active_image_count": 3,
                    "collapsed_steps": [1, 2],
                    "image_steps": [3, 4, 5],
                },
                "api": {
                    "usage": {"prompt_tokens": 7000, "completion_tokens": 50},
                    "server_timing": {
                        "request_total_sec": 3.5,
                        "server_ttft_sec": 2.5,
                        "engine_ttft_sec": 0.2,
                    },
                },
                "timing_ms": {"predict_total": 3510, "api_call": 3500, "image_prepare": 3},
                "request": {"input_file": "agent_inputs/step_0005.png", "message_file": "agent_messages/step_0005.json"},
                "response": {"text": "Action"},
            }
            (task_dir / "step_timing.jsonl").write_text(
                json.dumps(
                    {
                        "step_num": 5,
                        "status": "success",
                        "predict_wall_time_ms": 3511,
                        "agent_trace": agent_trace,
                        "actions": [
                            {
                                "action": "pyautogui.click(1, 1)",
                                "wall_time_ms": 400,
                                "environment_timing_ms": {
                                    "action_dispatch_ms": 100,
                                    "observation_ms": 300,
                                },
                            }
                        ],
                        "action_wall_time_ms": 400,
                        "turn_wall_time_ms": 3911,
                    }
                )
                + "\n"
            )
            (run_dir / "qw3_server.log").write_text(
                "[qw3-serve] multimodal prompt images=3 visual_tokens=6120 vision_cache_hit=0 mandatory_spans=3\n"
                "[qw3] native generate: prompt_tokens=7000 prefill=1.2s (1 tok/s) decoded=50 decode=0.8s (1 tok/s)\n"
                "[qw3-serve] #5 chat chars=10 completion_tokens=50 prompt_tokens=7000 route=plain fallback_reason=disabled 2100.0ms\n"
                "[qw3-server-ttft] rid=5 route=plain stream=0 model_token_seen=1 server_ttft_ms=2500 engine_ttft_ms=200 visible_output_seen=0 response_ttft_ms=-1 request_total_ms=3500\n"
                "[qw3-server-accounting] rid=5 stream=0 total_ms=3500 preprocess_ms=1400 queue_ms=0 engine_ms=2100 response_ms=0 sum_ms=3500 error_ms=0 pre_json_ms=1 pre_validate_ms=1300 pre_render_ms=1 pre_tokenize_ms=2 pre_span_setup_ms=96 pre_sum_ms=1400 pre_error_ms=0 stream_completed=1 terminal_status=success terminal_error=\"\"\n"
            )
            rows, _ = load_condition("compact", run_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["server_rid"], 5)
            self.assertEqual(rows[0]["server_preprocess_ms"], 1400)
            self.assertEqual(rows[0]["active_image_count"], 3)
            self.assertTrue(rows[0]["compaction_triggered"])


if __name__ == "__main__":
    unittest.main()
