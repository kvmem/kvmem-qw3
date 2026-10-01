import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

import aiohttp

from qw3_computer.app import Controller, web_app
from qw3_computer.desktop import DesktopError, DesktopRejected
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

    async def test_completion_waits_for_file_evidence(self):
        gate = asyncio.Event()
        async def files(*args):
            await gate.wait()
            return {"files": [{"name": "delivered.txt", "content": "ok"}]}
        self.desktop.request = files
        self.controller.consume_task = asyncio.create_task(self.controller.consume())
        await self.controller.start_task("test task")
        self.assertEqual(self.controller.state["status"], "verifying")
        gate.set()
        await asyncio.sleep(0)
        self.assertEqual(self.controller.state["status"], "completed")
        self.assertEqual(self.controller.state["files"][0]["name"], "delivered.txt")

    async def test_resume_rpc_failure_keeps_desktop_paused(self):
        self.controller.state.update(status="paused", id="task", thread_id="test-thread", turn_id="old-turn")
        self.controller.client.rpc.side_effect = RuntimeError("thread cannot be loaded")
        with self.assertRaisesRegex(RuntimeError, "cannot be loaded"):
            await self.controller.control("resume")
        self.assertEqual(self.controller.state["status"], "paused")
        self.assertTrue(self.desktop.paused)
        self.assertEqual(json.loads(self.controller.state_path.read_text())["status"], "paused")

    async def test_missing_result_evidence_does_not_publish_completion(self):
        self.desktop.request.side_effect = DesktopError("Desktop unavailable")
        self.controller.consume_task = asyncio.create_task(self.controller.consume())
        await self.controller.start_task("test task")
        self.assertEqual(self.controller.state["turn_status"], "completed")
        self.assertEqual(self.controller.state["status"], "paused")
        self.assertEqual(self.controller.state["evidence_error"], "Desktop unavailable")

    async def test_disconnect_pauses_desktop(self):
        self.controller.state.update(status="running", thread_id="test-thread")
        self.desktop.paused = False
        await self.controller.client.events.put({"method": "app/disconnected"})
        await self.controller.consume()
        self.assertFalse(self.controller.ready)
        self.assertTrue(self.desktop.paused)
        self.assertEqual(self.controller.state["status"], "paused")

    async def test_known_rejection_returns_fresh_image_without_retry(self):
        self.controller.state["status"] = "running"
        self.desktop.paused = False
        self.desktop.act = AsyncMock(side_effect=DesktopRejected("Foreground changed"))
        self.desktop.observe.return_value = self.desktop.latest | {"observation_id": "fresh", "image": "new-image"}
        result = await self.controller.tool("act", {"action_id": "rejected-001", "action": {"kind": "key", "key": "ENTER"}})
        self.assertTrue(result["isError"])
        metadata = json.loads(result["content"][0]["text"])
        self.assertEqual(metadata["status"], "rejected")
        self.assertFalse(metadata["dispatched"])
        self.assertEqual(metadata["observation"]["observation_id"], "fresh")
        self.assertEqual(result["content"][1]["data"], "new-image")
        self.desktop.act.assert_awaited_once()
        self.desktop.observe.assert_awaited_once()
        self.assertFalse(self.desktop.paused)
        self.assertEqual(self.controller.state["status"], "running")

    async def test_rejected_action_with_failed_refresh_stays_rejected(self):
        self.controller.state["status"] = "running"
        self.desktop.paused = False
        self.desktop.act = AsyncMock(side_effect=DesktopRejected("Foreground changed"))
        self.desktop.observe.side_effect = DesktopError("Observer unavailable")
        with self.assertRaisesRegex(DesktopRejected, "Foreground changed"):
            await self.controller.tool("act", {"action_id": "rejected-002", "action": {"kind": "key", "key": "ENTER"}})
        self.desktop.act.assert_awaited_once()
        self.assertFalse(self.desktop.paused)

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

    async def test_resume_and_restart_keep_task_budget_and_policy(self):
        self.controller.tool_call_limit = 3
        self.controller.output_policy = "task"
        await self.controller.start_task("Update the original workbook in place.")
        await self.controller.tool("observe", {})
        await self.controller.control("pause")
        await self.controller.control("resume")
        await self.controller.tool("observe", {})
        await self.controller.close()
        restored = Controller(self.temp.name, self.desktop, "unused", 1,
                              tool_call_limit=500, output_policy="outputs")
        self.controller = restored
        restored.client = FixtureCodex()
        restored.ready = True
        self.assertEqual(restored.state["tool_calls_used"], 2)
        self.assertEqual(restored.state["tool_call_limit"], 3)
        self.assertEqual(restored.state["output_policy"], "task")
        await restored.control("resume")
        await restored.tool("observe", {})
        with self.assertRaisesRegex(DesktopError, "limit reached"):
            await restored.tool("act", {"action": {"kind": "key", "key": "ENTER"}})
        self.assertTrue(self.desktop.paused)
        self.assertEqual(restored.state["status"], "paused")
        self.assertEqual(restored.state["tool_calls_used"], 3)
        # Exhaustion must be checked before enabling guest input.
        self.desktop.control = AsyncMock(wraps=self.desktop.control)
        with self.assertRaisesRegex(DesktopError, "额度已用尽"):
            await restored.control("resume")
        self.desktop.control.assert_not_awaited()

    async def test_failed_dispatch_is_counted_before_dispatch(self):
        await self.controller.start_task("Test budget persistence")
        async def failing_action(arguments):
            saved = json.loads(self.controller.state_path.read_text())
            self.assertEqual(saved["tool_calls_used"], 1)
            raise DesktopError("Transport failed")
        self.desktop.act = failing_action
        with self.assertRaisesRegex(DesktopError, "Transport failed"):
            await self.controller.tool("act", {})
        self.assertEqual(self.controller.budget_remaining(), 99)

    async def test_concurrent_calls_cannot_overdraw_last_budget_slot(self):
        self.controller.tool_call_limit = 1
        await self.controller.start_task("Test concurrent observation")
        outcomes = await asyncio.gather(self.controller.tool("observe", {}),
                                        self.controller.tool("observe", {}), return_exceptions=True)
        self.assertEqual(sum(isinstance(v, DesktopError) for v in outcomes), 1)
        self.assertEqual(self.controller.state["tool_calls_used"], 1)
        self.assertTrue(self.desktop.paused)

    async def test_legacy_task_cannot_resume_with_an_invented_budget(self):
        self.controller.state = {"status": "paused", "id": "legacy", "thread_id": "thread"}
        self.desktop.control = AsyncMock(wraps=self.desktop.control)
        with self.assertRaisesRegex(DesktopError, "累计调用记录"):
            await self.controller.control("resume")
        self.desktop.control.assert_not_awaited()

    async def test_task_output_policy_does_not_override_in_place_save(self):
        self.controller.output_policy = "task"
        self.controller.client.turn = AsyncMock(return_value="test-turn")
        prompt = r"Save the edited presentation in C:\Users\Qw3Test\Desktop\source.pptx."
        await self.controller.start_task(prompt)
        text = self.controller.client.turn.call_args.args[1]
        self.assertIn(prompt, text)
        self.assertNotIn(r"C:\Qw3Computer\outputs", text)
        await self.controller.control("stop")
        self.controller.output_policy = "outputs"
        await self.controller.start_task("Save a new test note")
        self.assertIn(r"C:\Qw3Computer\outputs", self.controller.client.turn.call_args.args[1])
        self.assertEqual(self.controller.state["tool_calls_used"], 0)

    async def test_invalid_task_limit_is_rejected(self):
        for limit in (0, -1, True, 1.5, 10001):
            with self.assertRaises(ValueError):
                Controller(self.temp.name, self.desktop, "unused", 1, tool_call_limit=limit)

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
