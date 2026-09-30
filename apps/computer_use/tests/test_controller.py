import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

import aiohttp

from qw3_computer.app import Controller, web_app
from test_bridge import serve


class FixtureDesktop:
    def __init__(self):
        self.latest = {"desktop_id": "test", "observation_id": "one", "sha256": "fixture"}
        self.paused = True
        self.session = True
        self.observe = AsyncMock(return_value=self.latest | {"image": "fixture"})
        self.request = AsyncMock(return_value={"files": []})
        self.close = AsyncMock()

    def unresolved(self):
        return []

    async def control(self, command, acknowledge_uncertain=False):
        self.paused = command != "resume"
        return {"paused": self.paused}


class FixtureCodex:
    def __init__(self):
        self.events = asyncio.Queue()
        self.new_thread = AsyncMock(return_value="test-thread")
        self.rpc = AsyncMock()
        self.interrupt = AsyncMock()
        self.close = AsyncMock()

    async def turn(self, thread, text, images):
        for method, status in (("turn/started", "inProgress"), ("turn/completed", "completed")):
            await self.events.put({"method": method, "params": {"threadId": thread,
                                         "turn": {"id": "test-turn", "status": status}}})
        await asyncio.sleep(0)
        return "test-turn"


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.desktop = FixtureDesktop()
        self.controller = Controller(self.temp.name, self.desktop, "unused", 1)
        self.controller.client = FixtureCodex()
        self.controller.ready = True

    async def asyncTearDown(self):
        await self.controller.close()
        self.temp.cleanup()

    async def test_fast_completion_does_not_revert_to_running(self):
        self.controller.consume_task = asyncio.create_task(self.controller.consume())
        state = await self.controller.start_task("test task")
        self.assertEqual(state["status"], "completed")
        self.assertTrue(self.desktop.paused)

    async def test_resume_rpc_failure_keeps_desktop_paused(self):
        self.controller.state.update(status="paused", id="task", thread_id="test-thread", turn_id="old-turn")
        self.controller.client.rpc.side_effect = RuntimeError("thread cannot be loaded")
        with self.assertRaisesRegex(RuntimeError, "cannot be loaded"):
            await self.controller.control("resume")
        self.assertEqual(self.controller.state["status"], "paused")
        self.assertTrue(self.desktop.paused)
        self.assertEqual(json.loads(self.controller.state_path.read_text())["status"], "paused")

    async def test_disconnect_pauses_desktop(self):
        self.controller.state.update(status="running", thread_id="test-thread")
        self.desktop.paused = False
        await self.controller.client.events.put({"method": "app/disconnected"})
        await self.controller.consume()
        self.assertFalse(self.controller.ready)
        self.assertTrue(self.desktop.paused)
        self.assertEqual(self.controller.state["status"], "paused")

    async def test_clean_shutdown_retains_resumable_task(self):
        self.controller.state.update(status="running", id="task", thread_id="test-thread", turn_id="test-turn")
        await self.controller.close()
        state = json.loads(self.controller.state_path.read_text())
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["thread_id"], "test-thread")
        self.assertTrue(self.desktop.paused)

    async def test_app_restart_requires_explicit_resume(self):
        self.controller.state.update(status="running", thread_id="test-thread", id="task")
        self.controller.save()
        restored = Controller(self.temp.name, self.desktop, "unused", 1)
        self.assertEqual(restored.state["status"], "paused")
        self.assertEqual(restored.state["thread_id"], "test-thread")
        self.assertIn("重新启动", restored.state["notice"])

    async def test_foreign_websites_cannot_start_local_tasks(self):
        self.controller.start_task = AsyncMock(return_value={"status": "running"})
        runner, url = await serve(web_app(self.controller))
        try:
            async with aiohttp.ClientSession() as client:
                for headers in ({"Origin": "https://unrelated.example"}, {"Host": "unrelated.example"}):
                    async with client.post(url + "/api/task", json={"prompt": "test"}, headers=headers) as response:
                        self.assertEqual(response.status, 403)
                self.controller.start_task.assert_not_awaited()
                async with client.post(url + "/internal/tool", json={"name": "observe"}) as response:
                    self.assertEqual(response.status, 401)
                async with client.post(url + "/api/task", json={"prompt": "test"}, headers={"Origin": url}) as response:
                    self.assertEqual(response.status, 200)
        finally:
            await runner.cleanup()
