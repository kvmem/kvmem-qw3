"""Opt-in: real pinned Codex and MCP; deterministic backend, no GPU claims."""

import asyncio
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from aiohttp import web
from PIL import Image

from qw3_computer.bridge import create_app
from qw3_computer.codex import CodexClient, write_config
from test_bridge import chunk, serve


@unittest.skipUnless(os.environ.get("QW3_TEST_CODEX") == "1", "set QW3_TEST_CODEX=1 for real Codex")
class CodexIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_images_success_error_mcp_rounds_and_interrupt(self):
        calls = []
        waiting, cancelled = asyncio.Event(), asyncio.Event()
        async def backend(request):
            body = await request.json()
            calls.append(body)
            if len(calls) > 3:
                waiting.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            if len(calls) <= 2:
                name = next(t["function"]["name"] for t in body["tools"]
                            if t["function"]["name"].endswith("observe"))
                await chunk(response, {"tool_calls": [{"index": 0, "id": "call_" + str(len(calls)),
                    "type": "function", "function": {"name": name, "arguments": "{}"}}]})
                await chunk(response, finish="tool_calls")
            else:
                await chunk(response, {"content": "Two fixture screenshots received."})
                await chunk(response, finish="stop")
            await response.write(b"data: [DONE]\n\n")
            return response
        app = web.Application()
        app.router.add_post("/v1/chat/completions", backend)
        backend_runner, url = await serve(app)
        bridge_runner, bridge_url = await serve(create_app(url + "/v1"))
        with tempfile.TemporaryDirectory(prefix="qw3-codex-integration-") as runtime:
            write_config(runtime, bridge_url + "/v1", sys.executable,
                         [str(Path(__file__).with_name("fixture_mcp.py").resolve()), "--error-second"])
            client = CodexClient(runtime)
            async def completed(turn):
                while True:
                    event = await asyncio.wait_for(client.events.get(), 40)
                    if event.get("method") == "turn/completed" and event["params"]["turn"]["id"] == turn:
                        return event["params"]["turn"]
                    if event.get("method") == "app/disconnected":
                        self.fail("Codex disconnected")
            try:
                await client.start()
                thread = await client.new_thread()
                image = io.BytesIO()
                Image.new("RGB", (128, 96), "red").save(image, format="PNG")
                data_url = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
                turn = await client.turn(thread, "Observe the fixture twice, then describe the result.", [data_url])
                result = await completed(turn)
                self.assertEqual(result["status"], "completed", result)
                self.assertEqual(len(calls), 3)
                self.assertTrue(any("Two fixture" in i.get("text", "") for i in result["items"]))
                self.assertTrue(any(p.get("type") == "image_url" for m in calls[0]["messages"]
                                    if isinstance(m["content"], list) for p in m["content"]))
                outputs = [m for m in calls[2]["messages"] if m["role"] == "tool"]
                self.assertEqual([m["tool_call_id"] for m in outputs], ["call_1", "call_2"])
                for output in outputs:
                    self.assertTrue(any(p.get("type") == "image_url" for p in output["content"]), output)
                self.assertIn("Action rejected without dispatch", json.dumps(outputs[1]))
                self.assertEqual(calls[1]["messages"], calls[2]["messages"][:len(calls[1]["messages"])])
                turn = await client.turn(thread, "Wait for cancellation.")
                await asyncio.wait_for(waiting.wait(), 10)
                await client.interrupt(thread, turn)
                self.assertEqual((await completed(turn))["status"], "interrupted")
                await asyncio.wait_for(cancelled.wait(), 5)
            except BaseException:
                print((Path(runtime) / "app-server.log").read_text()[-8000:], file=sys.stderr)
                raise
            finally:
                await client.close()
                await bridge_runner.cleanup()
                await backend_runner.cleanup()
