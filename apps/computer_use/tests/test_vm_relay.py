import asyncio
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vm.model_relay import create_app


class RelayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.cancelled = asyncio.Event()
        upstream = web.Application()

        async def health(request):
            self.requests.append(request.path)
            return web.json_response({"ready": True})

        async def stream(request):
            self.requests.append(await request.json())
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            await response.write(b'data: {"ok":true}\n\n')
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled.set()

        upstream.router.add_get("/health", health)
        upstream.router.add_post("/v1/chat/completions", stream)
        self.upstream = TestServer(upstream, handler_cancellation=True)
        await self.upstream.start_server()
        self.client = TestClient(TestServer(create_app(str(self.upstream.make_url("/")).rstrip("/"),
                                                      "127.0.0.1", "test-token"), handler_cancellation=True))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.upstream.close()

    async def test_token_peer_and_route_boundaries(self):
        response = await self.client.get("/health")
        self.assertEqual(response.status, 401)
        response = await self.client.get("/health", headers={"Authorization": "Bearer wrong"})
        self.assertEqual(response.status, 401)
        self.assertEqual(self.requests, [])
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

    async def test_stream_disconnect_closes_upstream(self):
        response = await self.client.post("/v1/chat/completions", json={"stream": True},
                                          headers={"Authorization": "Bearer test-token"})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.content.readline(), b'data: {"ok":true}\n')
        response.close()
        await asyncio.wait_for(self.cancelled.wait(), 2)
        self.assertEqual(self.requests, [{"stream": True}])
