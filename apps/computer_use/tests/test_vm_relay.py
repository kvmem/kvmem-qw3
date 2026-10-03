import asyncio
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vm.model_relay import create_app


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.receipts = Path(self.temporary.name) / 'requests.jsonl'
        self.requests = []
        self.cancelled = asyncio.Event()
        upstream = web.Application()

        async def health(request):
            self.requests.append(request.path)
            return web.json_response({"ready": True})

        async def stream(request):
            body = await request.json()
            self.requests.append(body)
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"ok":true}\n\n')
            if body.get('finish'):
                await response.write_eof()
                return response
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

        upstream.router.add_get("/health", health)
        async def models(request):
            return web.json_response({'data':[{'context_window':524288}]})
        upstream.router.add_get('/v1/models',models)
        upstream.router.add_post("/v1/chat/completions", stream)
        self.upstream = TestServer(upstream, handler_cancellation=True)
        await self.upstream.start_server()
        self.client = TestClient(TestServer(create_app(str(self.upstream.make_url("/")).rstrip("/"),
                                                      "127.0.0.1", "test-token", receipt_path=self.receipts), handler_cancellation=True))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.upstream.close()
        self.temporary.cleanup()

    async def test_token_peer_and_route_boundaries(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 401)
        response = await self.client.get("/health", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(response.status, 401)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.receipts.read_text(), '')
        response = await self.client.get("/health", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(await response.json(), {"ready": True})
        response = await self.client.get("/admin", headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status, 404)
        other = TestClient(TestServer(create_app(str(self.upstream.make_url("/")).rstrip("/"),
                                                 "192.0.2.1", "test-token")))
        await other.start_server()
        try:
            response = await other.get("/health", headers={"Authorization": "Bearer test-token"})
            self.assertEqual(response.status, 401)
            self.assertEqual(self.requests, ["/health"])
        finally:
            await other.close()

    async def test_model_capacity_route_requires_same_peer_and_token(self):
        self.assertEqual((await self.client.get('/v1/models')).status,401)
        response=await self.client.get('/v1/models',headers={'Authorization':'Bearer test-token'})
        self.assertEqual(response.status,200)
        self.assertEqual((await response.json())['data'][0]['context_window'],524288)
        self.assertEqual(self.receipts.read_text(),'')

    async def test_wire_limit_rejects_before_forwarding_and_preserves_auth(self):
        from qw3_computer.limits import MAX_MODEL_REQUEST_BYTES
        for limit in (True, 0, -1, MAX_MODEL_REQUEST_BYTES + 1):
            with self.assertRaises(ValueError):
                create_app('http://127.0.0.1:1', '127.0.0.1', 'test-token', max_request_bytes=limit)
        client = TestClient(TestServer(create_app(str(self.upstream.make_url('/')).rstrip('/'),
            '127.0.0.1', 'test-token', max_request_bytes=1024)))
        await client.start_server()
        try:
            body = dict(finish=True, messages=[dict(role='user', content='x' * 2048)])
            async with client.post('/v1/chat/completions', json=body,
                    headers={'Authorization': 'Bearer test-token'}) as response:
                self.assertEqual(response.status, 413)
            self.assertEqual(self.requests, [])
            async with client.post('/v1/chat/completions', json=dict(finish=True),
                    headers={'Authorization': 'Bearer test-token'}) as response:
                self.assertEqual(response.status, 200)
                await response.read()
            async with client.post('/v1/chat/completions', json=body) as response:
                self.assertEqual(response.status, 401)
        finally:
            await client.close()

    async def test_stream_disconnect_closes_upstream(self):
        response = await self.client.post("/v1/chat/completions", json={"stream": True},
                                          headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.content.readline(), b'data: {"ok":true}\n')
        response.close()
        await asyncio.wait_for(self.cancelled.wait(), 2)
        self.assertEqual(self.requests, [{"stream": True}])
        async with asyncio.timeout(2):
            while True:
                rows = [json.loads(line) for line in self.receipts.read_text().splitlines()]
                if rows and rows[-1]['event'] == 'request_terminal':
                    break
                await asyncio.sleep(.01)
        self.assertEqual(rows[-1]['status'], 'cancelled')

    async def test_configured_read_deadline_stops_upstream_and_records_failure(self):
        receipts = Path(self.temporary.name) / 'deadline.jsonl'
        client = TestClient(TestServer(create_app(str(self.upstream.make_url('/')).rstrip('/'),
            '127.0.0.1', 'test-token', receipt_path=receipts,
            backend_timeout_seconds=1, backend_read_timeout_seconds=.04), handler_cancellation=True))
        await client.start_server()
        try:
            async with asyncio.timeout(2):
                response = await client.post('/v1/chat/completions', json={'stream': True},
                    headers={'Authorization': 'Bearer test-token'})
                try:
                    await response.read()
                except aiohttp.ClientError:
                    pass
                await asyncio.wait_for(self.cancelled.wait(), 1)
            rows = [json.loads(line) for line in receipts.read_text().splitlines()]
            self.assertEqual(rows[-1]['event'], 'request_terminal')
            self.assertEqual(rows[-1]['status'], 'error')
            self.assertEqual(rows[-1]['error_type'], 'SocketTimeoutError')
        finally:
            await client.close()

    async def test_receipts_bind_request_and_last_image_without_logging_content(self):
        first, last = b'first image', b'last image'
        def image(value):
            return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(value).decode()}}
        body = {'stream': True, 'finish': True, 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Private test text not to be retained'}, image(first), image(last)]}]}
        raw = json.dumps(body).encode()
        response = await self.client.post('/v1/chat/completions', data=raw,
                                          headers={'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'})
        self.assertEqual(await response.read(), b'data: {"ok":true}\n\n')
        self.assertEqual(self.requests, [body])
        text = self.receipts.read_text()
        self.assertNotIn('Private test text', text)
        self.assertNotIn('test-token', text)
        self.assertNotIn(base64.b64encode(last).decode(), text)
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual([row['event'] for row in rows], ['request_forwarding', 'request_terminal'])
        final = rows[-1]
        self.assertEqual(final['status'], 'complete')
        self.assertEqual(final['wire_sha256'], hashlib.sha256(raw).hexdigest())
        canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
        self.assertEqual(final['canonical_request_sha256'], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(final['latest_image_sha256'], hashlib.sha256(last).hexdigest())
        self.assertLessEqual(final['host_received_monotonic_ns'], final['host_forwarding_monotonic_ns'])
        self.assertLessEqual(final['host_forwarding_monotonic_ns'], final['first_response_byte_monotonic_ns'])
        self.assertLessEqual(final['first_response_byte_monotonic_ns'], final['host_finished_monotonic_ns'])
