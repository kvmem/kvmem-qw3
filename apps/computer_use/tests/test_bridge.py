import asyncio
import json
from pathlib import Path
import tempfile
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
        self.backend_bodies = []
        self.mode = "text"
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

        async def backend(request):
            self.backend_bodies.append(await request.json())
            self.backend_authorization = request.headers.get("Authorization")
            self.entered.set()
            if self.mode == "http_error":
                return web.Response(status=503)
            if self.mode == 'payload_error':
                return web.json_response({'error': 'multimodal mandatory spans exceed budget'}, status=413)
            if self.mode == 'long_error':
                return web.Response(text='x' * 5000, status=500)
            try:
                if self.mode == "hang":
                    await asyncio.Future()
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                if self.mode == "slow":
                    await asyncio.sleep(.15)
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
        self.backend_url = url
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
        self.assertNotIn("kvmem_trace_tag", self.backend_bodies[-1])

    async def test_complete_tool_is_released_once(self):
        self.mode = "tool"
        events = await self.events()
        done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["call_id"], "call_one")
        self.assertEqual(done[0]["arguments"], "{}")

    async def test_application_tool_policy_hides_unavailable_tools_but_keeps_old_history(self):
        old = [{"type": "function_call", "name": "read_mcp_resource", "call_id": "old", "arguments": "{}"},
               {"type": "function_call_output", "call_id": "old", "output": "Unsupported MCP method"}]
        body = request(old)
        body["tools"].append({"type": "function", "name": "read_mcp_resource", "parameters": {"type": "object"}})
        with tempfile.TemporaryDirectory() as directory:
            runner, url = await serve(create_app(self.backend_url + "/v1", allowed_tools={"observe"}, trace_dir=Path(directory)))
            try:
                async with self.client.post(url + "/v1/responses", json=body) as response:
                    events = [json.loads(s) async for s in read_sse(response.content)]
                self.assertEqual(events[-1]["type"], "response.completed")
                sent = self.backend_bodies[-1]
                self.assertEqual([t["function"]["name"] for t in sent["tools"]], ["observe"])
                self.assertEqual(sent["messages"][1]["tool_calls"][0]["function"]["name"], "read_mcp_resource")
                policy = json.loads(next(Path(directory).glob("*/tool-policy.json")).read_text())
                self.assertEqual(policy["excluded"], ["read_mcp_resource"])
            finally:
                await runner.cleanup()

    async def test_backend_cannot_release_a_tool_outside_application_policy(self):
        self.mode = "tool"  # Fixture tries to return observe, which is unavailable here.
        runner, url = await serve(create_app(self.backend_url + "/v1", allowed_tools={"different_tool"}))
        try:
            async with self.client.post(url + "/v1/responses", json=request()) as response:
                events = [json.loads(s) async for s in read_sse(response.content)]
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertFalse(any(e["type"] == "response.function_call_arguments.done" for e in events))
            self.assertNotIn("tools", self.backend_bodies[-1])
        finally:
            await runner.cleanup()

    async def test_silent_prefill_keeps_connection_live_without_fake_output(self):
        self.mode = "slow"
        runner, url = await serve(create_app(self.backend_url + "/v1", backend_timeout_seconds=1,
            backend_read_timeout_seconds=.5, heartbeat_seconds=.02))
        try:
            async with self.client.post(url + "/v1/responses", json=request()) as response:
                events = [json.loads(s) async for s in read_sse(response.content)]
            self.assertGreaterEqual(sum(e["type"] == "response.in_progress" for e in events), 2)
            first_output = next(i for i, e in enumerate(events) if e["type"] == "response.output_item.added")
            for event in events[:first_output]:
                self.assertNotIn("usage", event.get("response", {}))
                self.assertEqual(event.get("response", {}).get("output", []), [])
            self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
            self.assertEqual(events[-1]["type"], "response.completed")
            self.assertEqual(events[-1]["response"]["usage"]["input_tokens"], 10)
        finally:
            await runner.cleanup()

    async def test_liveness_does_not_extend_backend_read_deadline(self):
        self.mode = "slow"
        runner, url = await serve(create_app(self.backend_url + "/v1", backend_timeout_seconds=1,
            backend_read_timeout_seconds=.04, heartbeat_seconds=.01))
        try:
            async with self.client.post(url + "/v1/responses", json=request()) as response:
                events = [json.loads(s) async for s in read_sse(response.content)]
            self.assertEqual(events[-1]["type"], "response.failed")
            self.assertFalse(any(e["type"] == "response.output_item.done" for e in events))
            await asyncio.wait_for(self.cancelled.wait(), 3)
        finally:
            await runner.cleanup()

    async def test_disconnect_during_live_wait_cancels_backend_and_releases_slot(self):
        self.mode = "slow"
        runner, url = await serve(create_app(self.backend_url + "/v1", backend_timeout_seconds=1,
            backend_read_timeout_seconds=.5, heartbeat_seconds=.02))
        try:
            response = await self.client.post(url + "/v1/responses", json=request())
            self.assertEqual(response.status, 200)
            response.close()
            await asyncio.wait_for(self.cancelled.wait(), 3)
            self.mode = "text"
            async with self.client.post(url + "/v1/responses", json=request()) as response:
                events = [json.loads(s) async for s in read_sse(response.content)]
            self.assertEqual(events[-1]["type"], "response.completed")
        finally:
            await runner.cleanup()

    async def test_backend_trace_tags_match_retained_requests_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            runner, url = await serve(create_app(self.backend_url + "/v1", trace_dir=Path(directory),
                                                 trace_backend=True))
            try:
                tags = []
                for _ in range(2):
                    async with self.client.post(url + "/v1/responses", json=request()) as response:
                        self.assertEqual(response.status, 200)
                        output = await response.text()
                    body = self.backend_bodies[-1]
                    tag = body["kvmem_trace_tag"]
                    self.assertRegex(tag, r"^[0-9a-f]{32}$")
                    self.assertNotIn(tag, json.dumps(body["messages"]))
                    self.assertNotIn(tag, output)
                    saved = Path(directory) / tag
                    self.assertEqual(json.loads((saved / "request.json").read_text()), body)
                    metrics = json.loads((saved / "metrics.json").read_text())
                    self.assertEqual(metrics["backend_trace_tag"], tag)
                    self.assertEqual(metrics["status"], "completed")
                    tags.append(tag)
                self.assertEqual(len(set(tags)), 2)
                self.assertEqual({p.name for p in Path(directory).iterdir()}, set(tags))
            finally:
                await runner.cleanup()

    async def test_backend_trace_tags_require_retained_local_evidence(self):
        with self.assertRaisesRegex(ValueError, "local trace directory"):
            create_app(self.backend_url + "/v1", trace_backend=True)

    async def test_private_backend_token_is_not_in_model_history_or_output(self):
        token = "private-vm-test-token"
        with tempfile.TemporaryDirectory() as directory:
            runner, url = await serve(create_app(self.backend_url + "/v1", backend_token=token, trace_dir=Path(directory)))
            try:
                async with self.client.post(url + "/v1/responses", json=request()) as response:
                    body = await response.text()
                    self.assertNotIn(token, body)
                self.assertEqual(self.backend_authorization, "Bearer " + token)
                for path in Path(directory).rglob("*.json"):
                    self.assertNotIn(token, path.read_text())
            finally:
                await runner.cleanup()

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

    async def test_backend_error_body_is_retained_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary)
            bridge, url = await serve(create_app(self.backend_url + '/v1', trace_dir=trace))
            try:
                for mode, status, truncated in [('payload_error', 413, False), ('long_error', 500, True)]:
                    self.mode = mode
                    async with self.client.post(url + '/v1/responses', json=request()) as response:
                        self.assertEqual(response.status, 502)
                        error = (await response.json())['error']
                        self.assertEqual(error['upstream_status'], status)
                        self.assertLessEqual(len(error['message']), 4200)
                    metrics = sorted(trace.glob('*/metrics.json'), key=lambda p: p.stat().st_mtime_ns)[-1]
                    receipt = json.loads(metrics.read_text())
                    self.assertEqual(receipt['http_status'], status)
                    self.assertEqual(receipt['backend_error_truncated'], truncated)
                    self.assertLessEqual(len(receipt['backend_error_response']), 4096)
                    if mode == 'payload_error':
                        self.assertIn('mandatory spans exceed budget', receipt['backend_error_response'])
            finally:
                await bridge.cleanup()

    async def test_wire_limit_rejects_oversized_requests_before_forwarding(self):
        from qw3_computer.limits import MAX_MODEL_REQUEST_BYTES
        self.assertEqual(MAX_MODEL_REQUEST_BYTES, 512 * 1024 * 1024)
        for limit in (True, 0, -1, MAX_MODEL_REQUEST_BYTES + 1):
            with self.assertRaises(ValueError):
                create_app(self.backend_url + '/v1', max_request_bytes=limit)
        bridge, url = await serve(create_app(self.backend_url + '/v1', max_request_bytes=1024))
        try:
            async with self.client.post(url + '/v1/responses', json=request([{'role': 'user', 'content': 'x' * 2048}])) as response:
                self.assertEqual(response.status, 413)
            self.assertEqual(self.backend_bodies, [])
            async with self.client.post(url + '/v1/responses', json=request()) as response:
                self.assertEqual(response.status, 200)
                await response.read()
        finally:
            await bridge.cleanup()

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
