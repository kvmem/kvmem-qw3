import asyncio
import contextlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp

from qw3_computer.app import Controller, web_app
from qw3_computer.desktop import DesktopError, DesktopRejected
from test_bridge import chunk, serve
from test_protocol import request as response_request
from qw3_computer.protocol import read_sse


class FixtureDesktop:
    def __init__(self):
        self.latest = {"desktop_id": "test", "observation_id": "one", "sha256": "fixture"}
        self.paused = True
        self.session = True
        self.observe = AsyncMock(return_value=self.latest | {"image": "fixture"})
        self.request = AsyncMock(return_value={"files": []})
        self.close = AsyncMock()
        self.start = AsyncMock()

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
    async def test_discovered_context_is_bound_to_resume_checkpoint(self):
        from aiohttp import web
        app=web.Application()
        async def metadata(request):return web.json_response({'data':[{'context_window':524288}]})
        app.router.add_get('/v1/models',metadata)
        runner,url=await serve(app)
        try:
            controller=Controller(self.temp.name,self.desktop,url+'/v1',1)
            await controller.discover_backend_context()
            self.assertEqual(controller.model_context_window,524288)
            self.assertEqual(controller.snapshot()['model_context_window'],524288)
            controller.state['model_context_window']=1048576
            with self.assertRaisesRegex(DesktopError,'上下文'):
                controller.require_saved_inference_mode()
        finally:await runner.cleanup()

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

    async def test_changed_thinking_mode_cannot_resume_old_task(self):
        self.controller.state.update(status="paused", id="task", thread_id="test-thread", thinking=False)
        self.controller.thinking = True
        with self.assertRaisesRegex(DesktopError, "推理模式"):
            await self.controller.control("resume")
        self.assertTrue(self.desktop.paused)
        self.controller.client.rpc.assert_not_called()
        self.assertEqual(self.controller.state["status"], "paused")

    async def test_new_task_records_requested_thinking_mode(self):
        self.controller.thinking = True
        self.controller.consume_task = asyncio.create_task(self.controller.consume())
        state = await self.controller.start_task("new thinking task")
        self.assertTrue(state["thinking"])
        self.assertTrue(json.loads(self.controller.state_path.read_text())["thinking"])

    async def test_software_job_start_without_verification_cannot_complete_task(self):
        class SoftwareFixture:
            root = Path("selected-project-workspace")
            pause = AsyncMock()
            close = AsyncMock()
            def snapshot(self, task_id):
                return {"root": str(self.root), "jobs": [{"status": "running"}]}
            def inspect(self, task_id):
                return self.snapshot(task_id)
        self.controller.software = SoftwareFixture()
        self.controller.consume_task = asyncio.create_task(self.controller.consume())
        await self.controller.start_task("Configure the Python project")
        await asyncio.sleep(0)
        self.assertEqual(self.controller.state["turn_status"], "completed")
        self.assertEqual(self.controller.state["status"], "paused")
        self.assertIn("工具校验", self.controller.state["notice"])
        self.assertFalse(self.controller.state["software_verified"])
        self.assertNotIn("evidence_error", self.controller.state)
        from task_driver import terminal
        self.assertEqual(terminal(self.controller.snapshot()), "paused")
        self.controller.software.pause.assert_awaited_once()

    async def test_software_pause_still_runs_when_desktop_executor_is_unreachable(self):
        class SoftwareFixture:
            pause = AsyncMock()
            close = AsyncMock()
        self.controller.software = SoftwareFixture()
        self.controller.state.update(id="install-task", status="running")
        self.desktop.control = AsyncMock(side_effect=DesktopError("executor unavailable"))
        with self.assertRaisesRegex(DesktopError, "unavailable"):
            await self.controller.control("pause")
        self.controller.software.pause.assert_awaited_once_with("install-task")
        self.assertEqual(self.controller.state["status"], "paused")

    async def test_software_scope_cannot_be_added_when_resuming_old_task(self):
        from types import SimpleNamespace
        self.controller.state.update(status="paused", id="task", thread_id="test-thread", thinking=False)
        self.controller.software = SimpleNamespace(root=Path("new-permission"), close=AsyncMock())
        with self.assertRaisesRegex(DesktopError, "系统配置权限"):
            await self.controller.control("resume")
        self.controller.client.rpc.assert_not_called()

    async def test_disabled_software_tool_cannot_dispatch(self):
        self.controller.state["status"] = "running"
        with self.assertRaises(DesktopError):
            await self.controller.tool("software", {"operation": "setup_python"})

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

    async def test_outside_scope_rejection_explains_allowed_focus_recovery_without_dispatching_it(self):
        self.controller.state.update(status='running', id='task')
        self.desktop.paused = False
        self.desktop.act = AsyncMock(side_effect=DesktopRejected('Foreground application is outside the allowed scope'))
        result = await self.controller.tool('act', {'action_id': 'bring-edge-forward', 'action': {'kind': 'key', 'key': 'ALT+TAB'}})
        text = json.loads(result['content'][0]['text'])
        self.assertEqual(text['problem']['code'], 'foreground_outside_scope')
        self.assertIn('open_app', text['problem']['hint'])
        self.assertIn('Preserve unrelated', text['problem']['hint'])
        self.assertFalse(text['dispatched'])
        self.assertEqual(text['recovery']['rejected_calls'], 1)
        self.assertEqual(self.controller.state['status'], 'running')
        self.desktop.act.assert_awaited_once()
        self.assertFalse(self.desktop.paused)

    async def test_rejected_action_with_failed_refresh_stays_rejected(self):
        self.controller.state["status"] = "running"
        self.desktop.paused = False
        self.desktop.act = AsyncMock(side_effect=DesktopRejected("Foreground changed"))
        self.desktop.observe.side_effect = DesktopError("Observer unavailable")
        with self.assertRaisesRegex(DesktopRejected, "Foreground changed"):
            await self.controller.tool("act", {"action_id": "rejected-002", "action": {"kind": "key", "key": "ENTER"}})
        self.desktop.act.assert_awaited_once()
        self.assertFalse(self.desktop.paused)

    async def test_unfocused_launch_is_recoverable_error_not_reissued_or_uncertain(self):
        self.controller.state["status"] = "running"
        self.desktop.paused = False
        self.desktop.act = AsyncMock(return_value={
            "action_id": "launch-001", "status": "completed", "no_progress": False,
            "app_focus": {"verified": False, "reason": "ambiguous_windows"},
            "observation": self.desktop.latest | {"image": "fresh-screen"}})
        result = await self.controller.tool("act", {
            "action_id": "launch-001", "action": {"kind": "open_app", "app": "edge"}})
        self.assertTrue(result["isError"])
        metadata = json.loads(result["content"][0]["text"])
        self.assertEqual(metadata["status"], "completed")
        self.assertEqual(metadata["app_focus"]["reason"], "ambiguous_windows")
        self.assertIn("do not type", metadata["error"])
        self.assertEqual(result["content"][1]["data"], "fresh-screen")
        self.desktop.act.assert_awaited_once()
        self.desktop.observe.assert_not_awaited()
        self.controller.client.interrupt.assert_not_awaited()
        self.assertFalse(self.desktop.paused)
        self.assertEqual(self.controller.state["status"], "running")

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

    async def test_backend_wait_limits_reject_invalid_values_before_start(self):
        for waits in [dict(backend_timeout_seconds=True), dict(backend_timeout_seconds=float('inf')),
                      dict(backend_read_timeout_seconds=float('nan')), dict(heartbeat_seconds=0),
                      dict(backend_read_timeout_seconds=601), dict(backend_timeout_seconds=14401)]:
            with self.subTest(waits=waits), self.assertRaises(ValueError):
                Controller(self.temp.name, self.desktop, "unused", 1, **waits)

    async def test_controller_bridge_honors_explicit_wait_and_read_deadline(self):
        async def delayed_backend(req):
            await req.json()
            response = aiohttp.web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(req)
            await asyncio.sleep(.08)
            await chunk(response, {'content': 'finished'})
            await chunk(response, finish='stop')
            await response.write(b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}\n\n')
            await response.write(b'data: [DONE]\n\n')
            return response
        backend = aiohttp.web.Application()
        backend.router.add_post('/v1/chat/completions', delayed_backend)
        runner, url = await serve(backend)
        try:
            for read_wait, expected in ((.3, 'response.completed'), (.03, 'response.failed')):
                with tempfile.TemporaryDirectory() as directory:
                    desktop = FixtureDesktop()
                    client = FixtureCodex()
                    client.start = AsyncMock()
                    controller = Controller(directory, desktop, url + '/v1', 1,
                        backend_timeout_seconds=.5, backend_read_timeout_seconds=read_wait, heartbeat_seconds=.01)
                    try:
                        with patch('qw3_computer.app.CodexClient', return_value=client):
                            await controller.start()
                        port = controller.bridge.addresses[0][1]
                        async with aiohttp.ClientSession() as session:
                            async with session.post(f'http://127.0.0.1:{port}/v1/responses', json=response_request()) as response:
                                events = [json.loads(s) async for s in read_sse(response.content)]
                        self.assertEqual(events[-1]['type'], expected)
                        self.assertGreaterEqual(sum(e['type'] == 'response.in_progress' for e in events), 2)
                        if expected == 'response.failed':
                            self.assertFalse(any(e['type'] == 'response.output_item.done' for e in events))
                        self.assertTrue(desktop.paused)
                    finally:
                        await controller.close()
        finally:
            await runner.cleanup()

    def completed_review(self):
        self.controller.state.update(id="owned-task", status="completed", thread_id="same-history",
                                     turn_id="previous-turn", turn_status="completed", tool_calls_used=17,
                                     tool_call_limit=50, output_policy="task", messages=["Initial review finished."])
        self.controller.save()

    async def test_followup_preserves_history_identity_and_lifetime_budget(self):
        self.completed_review()
        self.controller.client.turn = AsyncMock(return_value="next-turn")
        result = await self.controller.followup("owned-task", "previous-turn", "event-001", "Review the newly arrived messages.")
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["id"], "owned-task")
        self.assertEqual(result["thread_id"], "same-history")
        self.assertEqual(result["tool_calls_used"], 17)
        self.assertEqual(result["tool_call_limit"], 50)
        self.assertEqual(result["messages"], ["Initial review finished."])
        self.controller.client.new_thread.assert_not_awaited()
        self.controller.client.rpc.assert_awaited_once_with("thread/resume", {"threadId": "same-history"})
        self.assertEqual(result["followups"][0]["delivery"], "accepted")
        self.assertEqual(json.loads(self.controller.state_path.read_text())["tool_calls_used"], 17)

    async def test_followup_is_idempotent_and_refuses_changed_replay(self):
        self.completed_review()
        self.controller.client.turn = AsyncMock(return_value="next-turn")
        args = ("owned-task", "previous-turn", "event-001", "New messages arrived.")
        await self.controller.followup(*args)
        await self.controller.followup(*args)
        self.controller.client.turn.assert_awaited_once()
        with self.assertRaisesRegex(DesktopError, "different content"):
            await self.controller.followup(*args[:3], "Changed instructions")

    async def test_followup_cannot_bypass_pause_failure_budget_or_stale_turn(self):
        for changes in ({"status": "paused"}, {"status": "failed"}, {"status": "running"},
                        {"turn_id": "different-turn"}, {"tool_calls_used": 50},
                        {"evidence_error": "File evidence unavailable"}, {"turn_status": "failed"}):
            self.completed_review()
            self.controller.state.pop("evidence_error", None)
            self.controller.state.update(changes)
            self.desktop.control = AsyncMock(wraps=self.desktop.control)
            with self.assertRaises(DesktopError):
                await self.controller.followup("owned-task", "previous-turn", "event-001", "New messages arrived.")
            self.desktop.control.assert_not_awaited()

    async def test_followup_uncertain_dispatch_is_persisted_without_retry(self):
        self.completed_review()
        self.controller.client.turn = AsyncMock(side_effect=RuntimeError("lost reply"))
        args = ("owned-task", "previous-turn", "event-001", "New messages arrived.")
        with self.assertRaisesRegex(RuntimeError, "lost reply"):
            await self.controller.followup(*args)
        self.assertTrue(self.desktop.paused)
        saved = json.loads(self.controller.state_path.read_text())
        self.assertEqual(saved["status"], "paused")
        self.assertEqual(saved["followups"][0]["delivery"], "uncertain")
        with self.assertRaisesRegex(DesktopError, "uncertain"):
            await self.controller.followup(*args)
        self.controller.client.turn.assert_awaited_once()

    async def test_fast_followup_completion_remains_completed(self):
        self.completed_review()
        self.controller.consume_task = asyncio.create_task(self.controller.consume())
        result = await self.controller.followup("owned-task", "previous-turn", "event-001", "New messages arrived.")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(self.desktop.paused)
        self.assertEqual(result["tool_calls_used"], 17)

    async def test_followup_cannot_acknowledge_uncertain_desktop_actions(self):
        self.completed_review()
        self.desktop.unresolved = lambda: [{"action_id": "unconfirmed"}]
        self.desktop.control = AsyncMock(wraps=self.desktop.control)
        with self.assertRaisesRegex(DesktopError, "verified"):
            await self.controller.followup("owned-task", "previous-turn", "event-001", "New messages arrived.")
        self.desktop.control.assert_not_awaited()

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

    async def test_repeated_validation_errors_pause_without_reset_from_observe(self):
        from qw3_computer.desktop import DesktopValidationError
        await self.controller.start_task('Test bounded GUI recovery')
        self.desktop.act = AsyncMock(side_effect=DesktopValidationError({
            'code':'missing_field','path':'$.action_id','message':'Required field missing',
            'hint':'Move $.action.action_id to $.action_id'}))
        for attempt in range(3):
            result = await self.controller.tool('act', {'action': {'action_id': f'name-{attempt}', 'kind':'click'}})
            payload = json.loads(result['content'][0]['text'])
            self.assertFalse(payload['dispatched'])
            self.assertEqual(payload['error']['path'], '$.action_id')
            if attempt < 2:
                self.assertEqual(self.controller.state['status'], 'running')
                await self.controller.tool('observe', {})
        self.assertEqual(self.controller.state['status'], 'paused')
        self.assertTrue(self.desktop.paused)
        self.assertEqual(self.controller.state['tool_calls_used'], 5)
        self.assertEqual(self.controller.state['gui_recovery']['paused_reason'], 'repeated_rejection')
        self.controller.client.interrupt.assert_awaited_once()
        await self.controller.control('resume')
        self.assertNotIn('gui_recovery', self.controller.state)
        self.assertEqual(self.controller.state['tool_calls_used'], 5)

    async def test_successful_action_clears_rejection_streak(self):
        from qw3_computer.desktop import DesktopValidationError
        await self.controller.start_task('Correct one malformed call')
        self.desktop.act = AsyncMock(side_effect=DesktopValidationError({
            'code':'missing_field','path':'$.action_id','message':'Required field missing','hint':'Use root field'}))
        await self.controller.tool('act', {})
        self.desktop.act = AsyncMock(return_value={'status':'completed','observation':self.desktop.latest | {'image':'fixture'}})
        await self.controller.tool('act', {})
        self.assertNotIn('gui_recovery', self.controller.state)
        self.assertEqual(self.controller.state['status'], 'running')

    async def test_unchanged_observe_loop_pauses_but_changing_loading_does_not(self):
        await self.controller.start_task('Observe progress')
        with patch.object(self.controller, 'recovery_clock', return_value=100):
            await self.controller.tool('observe', {})
        with patch.object(self.controller, 'recovery_clock', return_value=120):
            await self.controller.tool('observe', {})
        self.assertEqual(self.controller.state['status'], 'running')
        self.desktop.observe.return_value = self.desktop.latest | {'image':'fixture','progress_sha256':'loading-progress-2'}
        with patch.object(self.controller, 'recovery_clock', return_value=131):
            await self.controller.tool('observe', {})
        self.assertEqual(self.controller.state['status'], 'running')
        with patch.object(self.controller, 'recovery_clock', return_value=151):
            await self.controller.tool('observe', {})
        with patch.object(self.controller, 'recovery_clock', return_value=162):
            result = await self.controller.tool('observe', {})
        self.assertTrue(result['isError'])
        self.assertEqual(self.controller.state['status'], 'paused')
        self.assertEqual(self.controller.state['gui_recovery']['paused_reason'], 'unchanged_observation_loop')

    async def test_active_software_job_is_not_a_static_desktop_stall(self):
        from types import SimpleNamespace
        await self.controller.start_task('Wait for a real background installation')
        self.controller.software = SimpleNamespace(root=Path('test'),snapshot=lambda task:{'jobs':[{'status':'running'}]},
            close=AsyncMock(),pause=AsyncMock())
        for clock in (100,120,140,160):
            with patch.object(self.controller, 'recovery_clock', return_value=clock):
                await self.controller.tool('observe', {})
        self.assertEqual(self.controller.state['status'], 'running')
        self.controller.client.interrupt.assert_not_called()

    async def test_repeated_known_executor_rejections_are_bounded(self):
        await self.controller.start_task('Leave a blocked foreground safely')
        self.desktop.act = AsyncMock(side_effect=DesktopRejected('Foreground application is outside the allowed scope'))
        for attempt in range(3):
            result = await self.controller.tool('act', {'action_id':f'scope-action-{attempt}'})
            self.assertFalse(json.loads(result['content'][0]['text'])['dispatched'])
        self.assertEqual(self.controller.state['status'], 'paused')
        self.assertEqual(self.controller.state['tool_calls_used'], 3)
        self.assertEqual(self.desktop.act.await_count, 3)
