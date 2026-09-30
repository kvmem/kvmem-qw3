import asyncio
import base64
import hashlib
import io
import tempfile
import unittest

from aiohttp import web
from PIL import Image

from qw3_computer.desktop import Desktop, DesktopError, DesktopRejected
from test_bridge import serve


class DesktopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sequence = self.effects = 0
        self.paused = True
        self.mode = "normal"
        self.receipts = {}
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "blue").save(buffer, format="PNG")
        self.png = buffer.getvalue()

        async def handle(request):
            if request.path == "/observe":
                return web.json_response(self.observation())
            if request.path == "/control":
                self.paused = (await request.json())["command"] != "resume"
                return web.json_response({"paused": self.paused})
            if request.path.startswith("/actions/"):
                return web.json_response(self.receipts.get(request.path.rsplit("/", 1)[1], {"status": "unknown"}))
            body = await request.json()
            if self.mode == "reject":
                return web.json_response({"delivery": "rejected", "error": "Foreground changed"}, status=409)
            self.effects += 1
            result = {"action_id": body["action_id"], "status": "completed", "observation": self.observation()}
            if self.mode != "unknown":
                self.receipts[body["action_id"]] = {"status": "completed", "result": result}
            self.entered.set()
            if self.mode == "hang":
                await self.release.wait()
            if self.mode in ("lost_response", "unknown"):
                request.transport.close()
            if self.mode == "no_progress":
                result["no_progress"] = True
            return web.json_response(result)

        app = web.Application()
        app.router.add_route("*", "/{path:.*}", handle)
        self.runner, self.url = await serve(app)
        self.desktop = Desktop(self.url, "fixture-token", self.temp.name)
        await self.desktop.start()
        await self.desktop.control("resume")
        await self.desktop.observe()

    def observation(self):
        self.sequence += 1
        return {"desktop_id": "fixture-desktop", "observation_id": str(self.sequence),
                "image": base64.b64encode(self.png).decode(), "sha256": hashlib.sha256(self.png).hexdigest(),
                "width": 8, "height": 8, "paused": self.paused}

    def action(self, id="action_001", **changes):
        return {"desktop_id": "fixture-desktop", "observation_id": self.desktop.latest["observation_id"],
                "action_id": id, "action": {"kind": "type", "text": "hello"}} | changes

    async def asyncTearDown(self):
        self.release.set()
        await self.desktop.close()
        await self.runner.cleanup()
        self.temp.cleanup()

    async def test_duplicate_returns_receipt_without_second_effect(self):
        request = self.action()
        first = await self.desktop.act(request)
        second = await self.desktop.act(request)
        self.assertEqual(first["observation"], second["observation"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(self.effects, 1)
        with self.assertRaisesRegex(DesktopError, "different arguments"):
            await self.desktop.act(request | {"action": {"kind": "key", "key": "ENTER"}})

    async def test_invalid_identity_stale_and_paused_never_dispatch(self):
        for change in ({"desktop_id": "wrong"}, {"observation_id": "old"}, {"action_id": "../../bad"}):
            with self.assertRaises(DesktopError):
                await self.desktop.act(self.action(**change))
        await self.desktop.control("pause")
        with self.assertRaises(DesktopError):
            await self.desktop.act(self.action())
        self.assertEqual(self.effects, 0)

    async def test_known_rejection_does_not_require_uncertain_ack(self):
        self.mode = "reject"
        with self.assertRaises(DesktopRejected):
            await self.desktop.act(self.action())
        self.assertEqual(self.desktop.unresolved(), [])
        self.assertFalse(self.desktop.paused)
        self.assertEqual(self.effects, 0)

    async def test_lost_response_reconciles_after_app_restart_without_replay(self):
        self.mode = "lost_response"
        request = self.action()
        with self.assertRaises(DesktopError):
            await self.desktop.act(request)
        self.assertTrue(self.desktop.paused)
        self.assertEqual(len(self.desktop.unresolved()), 1)
        await self.desktop.close()
        self.desktop = Desktop(self.url, "fixture-token", self.temp.name)
        await self.desktop.start()
        await self.desktop.control("resume")
        self.assertEqual(self.desktop.unresolved(), [])
        self.assertTrue((await self.desktop.act(request))["duplicate"])
        self.assertEqual(self.effects, 1)

    async def test_unknown_delivery_requires_explicit_reconciliation(self):
        self.mode = "unknown"
        request = self.action()
        with self.assertRaises(DesktopError):
            await self.desktop.act(request)
        with self.assertRaisesRegex(DesktopError, "尚未确认"):
            await self.desktop.control("resume")
        self.assertTrue(self.desktop.paused)
        await self.desktop.control("resume", acknowledge_uncertain=True)
        with self.assertRaisesRegex(DesktopError, "user_reconciled"):
            await self.desktop.act(request)
        self.assertEqual(self.effects, 1)

    async def test_cancel_after_effect_pauses_and_retains_action_id(self):
        self.mode = "hang"
        task = asyncio.create_task(self.desktop.act(self.action()))
        await asyncio.wait_for(self.entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.release.set()
        self.assertTrue(self.desktop.paused)
        self.assertTrue(self.paused)
        await self.desktop.control("resume")
        self.assertEqual(self.effects, 1)
        self.assertEqual(self.desktop.unresolved(), [])

    async def test_no_progress_prevents_next_dispatch(self):
        self.mode = "no_progress"
        await self.desktop.act(self.action())
        self.assertTrue(self.desktop.paused)
        with self.assertRaises(DesktopError):
            await self.desktop.act(self.action("action_002"))
        self.assertEqual(self.effects, 1)
