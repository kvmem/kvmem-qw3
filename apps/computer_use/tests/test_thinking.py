"""Local reasoning round trips and terminal-only tool delivery."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import aiohttp
from aiohttp import web

from qw3_computer.bridge import create_app
from qw3_computer.codex import CodexClient, write_config
from qw3_computer.protocol import read_sse
from test_bridge import chunk, serve
from test_protocol import request


class ThinkingBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_is_separate_and_tool_never_released_on_truncation(self):
        mode = "stop"
        bodies = []
        async def backend(req):
            bodies.append(await req.json())
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await r.prepare(req)
            await chunk(r, {"reasoning_content": "Local reasoning marker."})
            if mode == "stop":
                await chunk(r, {"content": "Only final text."})
            else:
                await chunk(r, {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                    "function": {"name": "observe", "arguments": "{}"}}]})
            if mode != "disconnect":
                await chunk(r, finish="length" if mode == "length" else "stop")
                await r.write(b"data: [DONE]\n\n")
            return r
        app = web.Application(); app.router.add_post("/v1/chat/completions", backend)
        runner, url = await serve(app)
        bridge, endpoint = await serve(create_app(url + "/v1", allow_thinking=True))
        try:
            async with aiohttp.ClientSession(trust_env=False) as client:
                for mode in ("stop", "length", "disconnect"):
                    body = request() | {"reasoning": {"effort": "medium"}, "max_output_tokens": 100}
                    async with client.post(endpoint + "/v1/responses", json=body) as r:
                        self.assertEqual(r.status, 200)
                        events = [json.loads(s) async for s in read_sse(r.content)]
                    self.assertTrue(bodies[-1]["enable_thinking"])
                    self.assertEqual(bodies[-1]["max_tokens"], 100)
                    self.assertFalse(any(e.get("item", {}).get("type") == "function_call" for e in events))
                    self.assertEqual(events[-1]["type"], {"stop":"response.completed", "length":"response.incomplete", "disconnect":"response.failed"}[mode])
                    if mode == "stop":
                        output = events[-1]["response"]["output"]
                        self.assertEqual([x["type"] for x in output], ["reasoning", "message"])
                        self.assertEqual(output[0]["content"], [{"type":"reasoning_text", "text":"Local reasoning marker."}])
                        self.assertEqual(output[1]["content"][0]["text"], "Only final text.")
                        self.assertEqual([e["output_index"] for e in events if e["type"] == "response.output_item.done"], [0, 1])
                # Even an opt-in bridge must reject thinking in a none request.
                mode = "stop"
                async with client.post(endpoint + "/v1/responses", json=request()) as r:
                    events = [json.loads(s) async for s in read_sse(r.content)]
                self.assertFalse(bodies[-1]["enable_thinking"])
                self.assertEqual(events[-1]["type"], "response.failed")
        finally:
            await bridge.cleanup(); await runner.cleanup()


@unittest.skipUnless(os.environ.get("QW3_TEST_CODEX") == "1", "set QW3_TEST_CODEX=1 for real Codex")
class ThinkingCodexTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_codex_replays_local_reasoning_across_tool_and_user_turns(self):
        calls = []
        async def backend(req):
            body = await req.json(); calls.append(body)
            r = web.StreamResponse(headers={"Content-Type": "text/event-stream"}); await r.prepare(req)
            await chunk(r, {"reasoning_content": f"Local thought marker {len(calls)}."})
            if len(calls) == 1:
                name = next(t["function"]["name"] for t in body["tools"] if t["function"]["name"].endswith("observe"))
                await chunk(r, {"tool_calls": [{"index":0,"id":"thought_observe_1","type":"function",
                    "function":{"name":name,"arguments":"{}"}}]})
                await chunk(r, finish="tool_calls")
            else:
                await chunk(r, {"content": "Observation verified."})
                await chunk(r, finish="stop")
            await r.write(b"data: [DONE]\n\n"); return r
        app = web.Application(); app.router.add_post("/v1/chat/completions", backend)
        runner, url = await serve(app)
        bridge, endpoint = await serve(create_app(url + "/v1", allow_thinking=True))
        with tempfile.TemporaryDirectory(prefix="qw3-thinking-contract-") as directory:
            write_config(directory, endpoint + "/v1", sys.executable,
                         [str(Path(__file__).with_name("fixture_mcp.py").resolve())], thinking=True)
            client = CodexClient(directory)
            async def completed(turn):
                async with asyncio.timeout(30):
                    while True:
                        e = await client.events.get()
                        if e.get("method") == "turn/completed" and e["params"]["turn"]["id"] == turn:
                            return e["params"]["turn"]
                        if e.get("method") == "app/disconnected":
                            self.fail("Codex disconnected")
            try:
                await client.start(); thread = await client.new_thread()
                first = await client.turn(thread, "Observe the fixture, then finish.")
                self.assertEqual((await completed(first))["status"], "completed")
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(c["enable_thinking"] for c in calls))
                thought = next(m for m in calls[1]["messages"] if m.get("reasoning_content"))
                self.assertEqual(thought["reasoning_content"], "Local thought marker 1.")
                self.assertEqual(thought["tool_calls"][0]["id"], "thought_observe_1")
                second = await client.turn(thread, "Retain the observation and finish this follow-up.")
                self.assertEqual((await completed(second))["status"], "completed")
                history = calls[-1]["messages"]
                self.assertIn("Local thought marker 1.", [m.get("reasoning_content") for m in history])
                self.assertIn("Local thought marker 2.", [m.get("reasoning_content") for m in history])
                self.assertEqual(sum(m["role"] == "tool" for m in history), 1)
            finally:
                await client.close(); await bridge.cleanup(); await runner.cleanup()
