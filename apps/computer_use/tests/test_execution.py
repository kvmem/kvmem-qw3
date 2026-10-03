import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

from qw3_computer.desktop import DesktopError, DesktopValidationError
from qw3_computer.execution import Execution, check_observation
from test_controller import FixtureDesktop, FixtureCodex
from qw3_computer.app import Controller


class FakeDevice:
    def __init__(self):
        self.task_id = "task-one"
        self.paused = False
        self.latest = {"desktop_id": "dev:session", "observation_id": "1", "sha256": "before",
                       "progress_sha256": "before", "foreground": {"title": "Request", "process": "browser"},
                       "accessibility": {"status": "ok", "truncated": False, "controls": []}}
        self.reads = 0
        self.unresolved_rows = []

    async def observe(self):
        self.reads += 1
        self.latest = self.latest | {"observation_id": str(self.reads)}
        return self.latest

    def unresolved(self):
        return self.unresolved_rows


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.device = FakeDevice()
        self.execution = Execution(self.device, self.path)

    async def asyncTearDown(self):
        self.execution.close()
        self.temp.cleanup()

    async def begin(self, conditions=None):
        return await self.execution.invoke({"operation": "begin_step", "step_id": "save", "goal": "Save request",
            "conditions": conditions or [{"kind": "control_text", "text": "Saved"}]})

    async def test_waits_for_delayed_condition_without_dispatch(self):
        await self.begin()
        async def delayed_update():
            await asyncio.sleep(0.03)
            self.device.latest["accessibility"]["controls"] = [{"name": "Saved"}]
        update = asyncio.create_task(delayed_update())
        result = await self.execution.invoke({"operation": "wait", "step_id": "save", "timeout_seconds": 1})
        await update
        self.assertTrue(result["verified"])
        self.assertFalse(result["input_dispatched"])
        self.assertGreater(result["polls"], 1)
        self.assertEqual(self.execution.steps()[0]["status"], "verified")

    async def test_new_action_ids_cannot_repeat_unchanged_intent(self):
        await self.begin()
        action = {"kind": "click", "x": 90, "y": 40, "button": "left", "clicks": 1}
        for id in ("first", "second"):
            self.execution.before_dispatch({}, {"action": action}, id)
            await self.device.observe()
        with self.assertRaises(DesktopValidationError) as raised:
            self.execution.before_dispatch({}, {"action": action}, "third")
        self.assertEqual(raised.exception.problem["code"], "repeated_intent")
        self.device.latest["progress_sha256"] = "next-page"
        self.execution.before_dispatch({}, {"action": action}, "fourth")

    async def test_wait_current_step_defaults_without_guessing_verified_or_missing_step(self):
        with self.assertRaisesRegex(DesktopError, 'No unique pending'):
            await self.execution.invoke({'operation':'wait'})
        await self.begin()
        self.device.latest['accessibility']['controls']=[{'name':'Saved'}]
        result=await self.execution.invoke({'operation':'wait'})
        self.assertTrue(result['verified'])
        self.assertEqual(result['step_id'],'save')
        with self.assertRaisesRegex(DesktopError, 'No unique pending'):
            await self.execution.invoke({'operation':'wait'})

    async def test_restart_preserves_pending_step_and_input_reservations(self):
        await self.begin()
        action = {"kind": "type", "text": "exact content"}
        for id in ("one", "two"):
            self.execution.before_dispatch({}, {"action": action}, id)
        self.execution.close()
        self.execution = Execution(self.device, self.path)
        self.assertEqual(self.execution.steps()[0]["status"], "pending")
        with self.assertRaises(DesktopValidationError):
            self.execution.before_dispatch({}, {"action": action}, "three")
        self.device.task_id = "another-task"
        self.assertEqual(self.execution.steps(), [])

    async def test_final_condition_rechecked_on_fresh_observation(self):
        await self.begin([{"kind": "foreground_title", "text": "Request"}])
        await self.execution.invoke({"operation": "wait", "step_id": "save", "timeout_seconds": 0})
        result = await self.execution.invoke({"operation": "finish", "goal":"The requested document remains open", "conditions": [{"kind": "foreground_title", "text": "Request"}], "timeout_seconds": 0})
        self.assertTrue(result["verified"])
        self.device.latest["foreground"]["title"] = "Different document"
        self.assertFalse(self.execution.completion(self.device.latest)["verified"])

    async def test_uncertain_input_cannot_be_abandoned_or_finished(self):
        await self.begin()
        self.device.unresolved_rows = [{"action_id": "unknown"}]
        with self.assertRaises(DesktopError):
            await self.execution.invoke({"operation": "abandon_step", "step_id": "save", "reason": "Try again"})
        with self.assertRaises(DesktopError):
            await self.execution.invoke({"operation": "finish", "conditions": [{"kind": "control_text", "text": "Saved"}], "timeout_seconds": 0})

    async def test_inline_step_binds_exact_observation_without_extra_capture(self):
        definition = {"goal":"Save request","conditions":[{"kind":"control_text","text":"Saved"}]}
        self.execution.before_dispatch({"step":definition},{"action":{"kind":"key","key":"CTRL+S"}},"q3_receipt_one")
        self.assertEqual(self.device.reads,0)
        self.assertEqual(self.execution.steps()[0]["before"]["observation_id"],"1")
        self.device.latest['accessibility']['controls']=[{'name':'Saved'}]
        self.assertTrue(self.execution.after_delivery(self.device.latest)['verified'])
        self.execution.before_dispatch({"step":{"goal":"Open next page","conditions":[{"kind":"screen_changed"}]}},
            {"action":{"kind":"key","key":"PAGEDOWN"}},"q3_receipt_two")
        self.assertEqual(len(self.execution.steps()),2)
        self.assertEqual(self.execution.steps()[0]['status'],'verified')

    async def test_inline_step_cannot_silently_replace_pending_plan(self):
        await self.begin()
        request={'step':{'goal':'Another plan','conditions':[{'kind':'screen_changed'}]}}
        with self.assertRaises(DesktopValidationError) as caught:
            self.execution.before_dispatch(request,{'action':{'kind':'key','key':'ESC'}},'q3_other')
        self.assertEqual(self.execution.steps()[0]['status'],'pending')
        checkpoint=caught.exception.problem['checkpoint']
        self.assertEqual(checkpoint['pending_step']['step_id'],'save')
        self.assertEqual(checkpoint['pending_step']['conditions'],self.execution.steps()[0]['conditions'])
        self.assertFalse(checkpoint['input_dispatched'])
        self.assertEqual(checkpoint['uncertain_action_ids'],[])
        # A correction pursuing the existing save conditions needs no copied
        # goal text and does not abandon or invent another step.
        self.execution.before_dispatch({}, {'action':{'kind':'key','key':'TAB'}}, 'q3_continue')
        self.assertEqual(len(self.execution.steps()),1)
        request['step']['recovery_reason']='Close the blocking dialog before trying another save'
        self.execution.before_dispatch(request,{'action':{'kind':'key','key':'ESC'}},'q3_recovery')
        self.assertEqual(self.execution.steps()[0]['status'],'abandoned')

    async def test_final_screen_change_and_empty_conditions_rejected(self):
        with self.assertRaises(DesktopError):
            await self.execution.invoke({"operation": "begin_step", "step_id": "bad", "goal": "Bad", "conditions": []})
        await self.begin([{"kind": "foreground_title", "text": "Request"}])
        await self.execution.invoke({"operation": "wait", "step_id": "save", "timeout_seconds": 0})
        with self.assertRaises(DesktopError):
            await self.execution.invoke({"operation": "finish", "conditions": [{"kind": "screen_changed"}], "timeout_seconds": 0})

    def test_absence_not_proven_by_failed_or_truncated_accessibility(self):
        checks = [{"kind": "control_absent", "text": "Loading"}]
        for status, truncated in (("unavailable", False), ("ok", True)):
            observation = {"accessibility": {"status": status, "truncated": truncated, "controls": []}}
            self.assertFalse(check_observation(checks, observation, {})[0]["passed"])


class VerifiedControllerTests(unittest.IsolatedAsyncioTestCase):
    async def make_checked_controller(self, path):
        desktop = FixtureDesktop()
        desktop.task_id = "manual"
        desktop.paused = False
        desktop.latest["foreground"] = {"title": "Request"}
        desktop.observe.return_value = desktop.latest | {"image": "fixture"}
        controller = Controller(path, desktop, "unused", 1, workflow=True)
        controller.client = FixtureCodex()
        controller.state.update(status="running", id="manual", thread_id="test-thread", turn_id="test-turn")
        checks = [{"kind": "foreground_title", "text": "Request"}]
        await controller.workflow.invoke({"operation": "begin_step", "step_id": "check",
                                          "goal": "Check result", "conditions": checks})
        await controller.workflow.invoke({"operation": "wait", "timeout_seconds": 0})
        return controller, checks

    async def test_checked_finish_ends_task_before_model_narration_and_survives_interrupt(self):
        with tempfile.TemporaryDirectory() as path:
            controller, checks = await self.make_checked_controller(path)
            controller.consume_task = asyncio.create_task(controller.consume())
            try:
                await controller.tool("workflow", {"operation": "finish", "conditions": checks})
                self.assertEqual(controller.state["status"], "completed")
                self.assertTrue(controller.desktop.paused)
                self.assertIn("independent business", controller.state["verification_scope"])
                await controller.finish_interrupt_task
                controller.client.interrupt.assert_awaited_once_with("test-thread", "test-turn")
                await controller.client.events.put({"method": "turn/completed", "params": {
                    "threadId": "test-thread", "turn": {"id": "test-turn", "status": "interrupted"}}})
                await asyncio.sleep(0)
                self.assertEqual(controller.state["status"], "completed")
                self.assertEqual(controller.state["turn_status"], "interrupted")
                self.assertEqual(json.loads(controller.state_path.read_text())["status"], "completed")
            finally:
                await controller.close()

    async def test_failed_final_check_does_not_stop_task(self):
        with tempfile.TemporaryDirectory() as path:
            controller, checks = await self.make_checked_controller(path)
            try:
                await controller.tool("workflow", {"operation": "finish", "conditions": [
                    {"kind": "foreground_title", "text": "Another document"}]})
                self.assertEqual(controller.state["status"], "running")
                self.assertFalse(controller.desktop.paused)
                controller.client.interrupt.assert_not_called()
            finally:
                await controller.close()

    async def test_checked_gui_cannot_finish_unverified_software_job(self):
        with tempfile.TemporaryDirectory() as path:
            controller, checks = await self.make_checked_controller(path)
            class SoftwareFixture:
                pause = AsyncMock()
                close = AsyncMock()
                def snapshot(self, task_id):
                    return {"jobs": [{"status": "completed", "verification": {"passed": False}}]}
            controller.software = SoftwareFixture()
            try:
                await controller.tool("workflow", {"operation": "finish", "conditions": checks})
                self.assertEqual(controller.state["status"], "running")
                self.assertFalse(controller.desktop.paused)
                controller.client.interrupt.assert_not_called()
            finally:
                await controller.close()

    async def test_model_turn_end_without_checked_conditions_stays_paused(self):
        with tempfile.TemporaryDirectory() as path:
            desktop = FixtureDesktop()
            desktop.task_id = "manual"
            controller = Controller(path, desktop, "unused", 1, workflow=True)
            controller.client = FixtureCodex()
            controller.ready = True
            controller.consume_task = asyncio.create_task(controller.consume())
            try:
                state = await controller.start_task("Produce an artifact")
                self.assertEqual(state["status"], "paused")
                self.assertTrue(desktop.paused)
                self.assertNotIn("verification_scope", state)
            finally:
                await controller.close()
