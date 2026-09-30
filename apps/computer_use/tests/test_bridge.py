import asyncio
import json
import unittest

import aiohttp
from aiohttp import web

from qw3_computer.bridge import create_app
from qw3_computer.protocol import read_sse
from test_protocol import request


async def serve(app):
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])


async def chunk(response, delta=None, finish=None):
    data = {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    await response.write(("data: " + json.dumps(data) + "\r\n\r\n").encode())


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mode = "text"
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

        async def backend(request):
            self.entered.set()
            if self.mode == "http_error":
                return web.Response(status=503)
            try:
                if self.mode == "hang":
                    await asyncio.Future()
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                if self.mode in ("tool", "bad_tool", "length"):
                    await chunk(response, {"tool_calls": [{"index": 0, "id": "call_one", "type": "function",
                        "function": {"name": "observe", "arguments": "{"}}]})
                    await chunk(response, {"tool_calls": [{"index": 0, "function": {
                        "arguments": "}" if self.mode != "bad_tool" else "broken"}}]})
                else:
                    await chunk(response, {"content": "你"})
                    await chunk(response, {"content": "好"})
                if self.mode != "truncated":
                    await chunk(response, finish="length" if self.mode == "length" else "stop")
                    await response.write(b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n\n')
                    await response.write(b"data: [DONE]\n\n")
                return response
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        app = web.Application()
        app.router.add_post("/v1/chat/completions", backend)
        self.backend, url = await serve(app)
        self.bridge, self.url = await serve(create_app(url + "/v1"))
        self.client = aiohttp.ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.bridge.cleanup()
        await self.backend.cleanup()

    async def events(self):
        async with self.client.post(self.url + "/v1/responses", json=request()) as response:
            self.assertEqual(response.status, 200)
            return [json.loads(s) async for s in read_sse(response.content)]

    async def test_text_stream_unicode_usage_and_sequence(self):
        events = await self.events()
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        last = events[-1]
        self.assertEqual(last["type"], "response.completed")
        self.assertEqual(last["response"]["output"][0]["content"][0]["text"], "你好")
        self.assertEqual(last["response"]["usage"]["total_tokens"], 12)

    async def test_complete_tool_is_released_once(self):
        self.mode = "tool"
        events = await self.events()
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["call_id"], "call_one")
        self.assertEqual(done[0]["arguments"], "{}")

    async def test_truncated_or_invalid_stream_never_completes(self):
        for mode in ("truncated", "bad_tool"):
            self.mode = mode
            events = await self.events()
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertNotIn("response.completed", [e["type"] for e in events])
            self.assertFalse(any(e.get("item", {}).get("type") == "function_call" for e in events))

    async def test_token_limit_never_releases_action(self):
        self.mode = "length"
        events = await self.events()
        self.assertEqual(events[-1]["type"], "response.incomplete")
        self.assertEqual(events[-1]["response"]["output"], [])

    async def test_http_error_and_invalid_request(self):
        self.mode = "http_error"
        async with self.client.post(self.url + "/v1/responses", json=request()) as response:
            self.assertEqual(response.status, 502)
        async with self.client.post(self.url + "/v1/responses", json=request() | {"store": True}) as response:
            self.assertEqual(response.status, 400)

    async def test_disconnect_during_prefill_cancels_backend_and_releases_slot(self):
        self.mode = "hang"
        pending = asyncio.create_task(self.client.post(self.url + "/v1/responses", json=request()))
        await asyncio.wait_for(self.entered.wait(), 3)
        async with self.client.post(self.url + "/v1/responses", json=request()) as response:
            self.assertEqual(response.status, 409)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(self.cancelled.wait(), 3)
        self.mode = "text"
        self.assertEqual((await self.events())[-1]["type"], "response.completed")
