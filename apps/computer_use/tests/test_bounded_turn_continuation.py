"""Run against the freshly built candidate package, not the live controller."""
import asyncio
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock

from bounded_turn_continuation import POLICY, checkpoint, eligibility
import bounded_turn_candidate as builder


Controller = None
package = os.environ.get("QW3_BOUNDED_TEST_PACKAGE")
if package:
    # Keep this candidate out of the normal qw3_computer import cache. The
    # same test process may already be testing the unchanged production app.
    path = Path(package).resolve()
    spec = importlib.util.spec_from_file_location("bounded_turn_test_sdk", path / "__init__.py",
                                                 submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from bounded_turn_test_sdk.app import Controller
    from bounded_turn_test_sdk.desktop import DesktopError


class DesktopFixture:
    def __init__(self):
        self.latest = {"desktop_id": "fixture", "observation_id": "fresh", "sha256": "fixture",
                       "image": "fixture", "width": 32, "height": 24}
        self.paused = True
        self.session = True
        self.pending = []
        self.inputs = []
        self.controls = []
        self.task_id = "task"
        self.observe = AsyncMock(return_value=self.latest)
        self.request = AsyncMock(return_value={"files": []})
        self.close = AsyncMock()

    def unresolved(self):
        return self.pending

    async def control(self, command, acknowledge_uncertain=False):
        self.controls.append(command)
        self.paused = command != "resume"
        return {"paused": self.paused}


class CodexFixture:
    def __init__(self):
        self.events = asyncio.Queue()
        self.turn = AsyncMock(side_effect=["new-1", "new-2"])
        self.interrupt = AsyncMock()
        self.close = AsyncMock()


@unittest.skipUnless(package, "Set QW3_BOUNDED_TEST_PACKAGE to the qualified isolated SDK51 candidate")
class ContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.desktop = DesktopFixture()
        self.controller = Controller(self.tmp.name, self.desktop, "unused", 1, workflow=True)
        self.controller.ready = True
        self.controller.client = CodexFixture()
        self.controller.state = {"id": "task", "thread_id": "thread", "turn_id": "first", "status": "running",
                                 "messages": [], "files": [], "prompt": "Use the visible application",
                                 "started_at": time.time(), "tool_calls_used": 0, "tool_call_limit": 250,
                                 "thinking": False, "execution_mode": "verified", "device_platform": "windows",
                                 "software_root": None, "model_output_tokens": 4096, "model_context_window": 1048576,
                                 "bounded_turn_continuation": checkpoint(self.controller.bounded_turn_owner, 0)}
        self.controller.save()
        self.consumer = asyncio.create_task(self.controller.consume())

    async def asyncTearDown(self):
        self.consumer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.consumer
        await self.controller.close()
        self.tmp.cleanup()

    async def completed(self, *, turn=None, status="completed", error=None):
        expected = turn or self.controller.state.get("turn_id")
        await self.controller.client.events.put({"method": "turn/completed", "params": {
            "threadId": "thread", "turn": {"id": expected,
                                            "status": status, "error": error}}})
        for _ in range(100):
            await asyncio.sleep(.001)
            state = self.controller.state
            handled = state.get("status") in {"paused", "stopped", "completed", "failed"} or (
                state.get("status") == "running" and state.get("turn_id") != expected)
            if self.controller.client.events.empty() and handled:
                return
        self.fail("Controller did not handle the completed turn")

    def saved(self):
        return json.loads(self.controller.state_path.read_text())

    async def test_announcement_turn_continues_same_task_with_durable_reservation_and_no_input(self):
        self.controller.state["gui_recovery"] = {"rejected_calls": 2}
        await self.completed()
        state = self.saved()
        self.assertEqual((state["id"], state["thread_id"], state["turn_id"], state["status"]),
                         ("task", "thread", "new-1", "running"))
        self.assertEqual((state["tool_calls_used"], state["tool_call_limit"]), (0, 250))
        self.assertEqual(state["gui_recovery"], {"rejected_calls": 2})
        self.assertEqual(state["bounded_turn_continuation"]["attempts"][0]["delivery"], "accepted")
        self.assertNotIn("finished_at", state)
        args = self.controller.client.turn.await_args.args
        self.assertEqual(args[0], "thread")
        self.assertIn("Durable checked/pending steps", args[1])
        self.assertEqual(self.desktop.inputs, [])

    async def test_two_more_announcements_exhaust_bound_without_verifying_completion(self):
        await self.completed()
        await self.completed()
        await self.completed()
        state = self.saved()
        self.assertEqual(state["status"], "paused")
        self.assertEqual(self.controller.client.turn.await_count, 2)
        self.assertEqual(len(state["bounded_turn_continuation"]["attempts"]), 2)
        self.assertEqual(state["bounded_turn_last_decision"]["reason"], "additional_turn_limit")
        self.assertTrue(self.desktop.paused)

    async def test_real_tool_call_in_the_turn_prevents_zero_tool_recovery(self):
        await self.controller.tool("observe", {})
        await self.completed()
        self.assertEqual(self.controller.state["tool_calls_used"], 1)
        self.assertEqual(self.controller.state["status"], "paused")
        self.controller.client.turn.assert_not_awaited()

    async def test_user_pause_is_not_overridden_by_completed_notification(self):
        await self.controller.control("pause")
        await self.completed()
        self.assertEqual(self.controller.state["status"], "paused")
        self.assertEqual(self.controller.state["bounded_turn_user_control"], "pause")
        self.controller.client.turn.assert_not_awaited()

    async def test_stop_during_final_observation_wins_over_auto_continuation(self):
        entered, release = asyncio.Event(), asyncio.Event()
        reads = 0
        async def observe():
            nonlocal reads
            reads += 1
            if reads == 1:
                entered.set()
                await release.wait()
            return self.desktop.latest
        self.desktop.observe.side_effect = observe
        event = {"method": "turn/completed", "params": {"threadId": "thread", "turn": {
            "id": "first", "status": "completed"}}}
        await self.controller.client.events.put(event)
        await asyncio.wait_for(entered.wait(), 1)
        await self.controller.control("stop")
        release.set()
        await asyncio.sleep(.02)
        self.assertEqual(self.saved()["status"], "stopped")
        self.controller.client.turn.assert_not_awaited()

    async def test_unknown_receipt_never_resumes_or_replays(self):
        self.desktop.pending = [{"receipt_id": "lost", "status": "uncertain"}]
        await self.completed()
        self.assertEqual(self.saved()["status"], "paused")
        self.assertNotIn("resume", self.desktop.controls)
        self.controller.client.turn.assert_not_awaited()

    async def test_takeover_and_no_progress_pauses_keep_their_recovery_counters(self):
        for change in ({"takeover": {"reason": "needs user input"}},
                       {"gui_recovery": {"paused_reason": "unchanged_observation_loop", "unchanged_observations": 3}}):
            with self.subTest(change=change):
                self.controller.state.update(status="running", turn_id="first", **change)
                await self.completed()
                self.controller.client.turn.assert_not_awaited()
                for key, value in change.items():
                    self.assertEqual(self.saved()[key], value)
                for key in change:
                    self.controller.state.pop(key)

    async def test_model_error_or_interruption_does_not_auto_retry(self):
        for status, error in (("failed", {"message": "upstream"}), ("interrupted", None)):
            with self.subTest(status=status):
                self.controller.state.update(status="running", turn_id="first")
                await self.completed(status=status, error=error)
                self.assertEqual(self.saved()["status"], "failed")
                self.controller.client.turn.assert_not_awaited()

    async def test_checked_completion_does_not_request_a_further_turn(self):
        self.controller.workflow.completion = lambda observation=None: {"verified": True, "verification_scope": "visible"}
        await self.completed()
        self.assertEqual(self.saved()["status"], "completed")
        self.controller.client.turn.assert_not_awaited()

    async def test_exhausted_tool_budget_does_not_gain_a_turn_or_new_budget(self):
        self.controller.state["tool_calls_used"] = 250
        await self.completed()
        self.assertEqual(self.saved()["tool_calls_used"], 250)
        self.controller.client.turn.assert_not_awaited()

    async def test_existing_task_deadline_expiry_does_not_resume_device(self):
        self.controller.state["bounded_turn_continuation"]["deadline_monotonic"] = time.monotonic() - 1
        await self.completed()
        self.assertNotIn("resume", self.desktop.controls)
        self.controller.client.turn.assert_not_awaited()

    async def test_changed_inference_checkpoint_is_not_bypassed(self):
        self.controller.state["model_output_tokens"] = 16384
        await self.completed()
        self.assertEqual(self.saved()["bounded_turn_last_decision"]["reason"], "inference_or_device_checkpoint_changed")
        self.controller.client.turn.assert_not_awaited()

    async def test_lost_turn_reply_stays_uncertain_and_is_not_retried(self):
        self.controller.client.turn.side_effect = TimeoutError("reply lost")
        await self.completed()
        state = self.saved()
        self.assertEqual(state["status"], "paused")
        self.assertEqual(state["bounded_turn_continuation"]["attempts"][0]["delivery"], "uncertain")
        self.assertTrue(self.desktop.paused)
        self.assertFalse(self.controller.ready)
        self.controller.client.close.assert_awaited_once()
        await self.completed(turn="first")  # Late duplicate notification cannot replay it.
        self.assertEqual(self.controller.client.turn.await_count, 1)

    async def test_restart_retains_counter_but_cannot_resume_automatically(self):
        await self.completed()
        restarted = Controller(self.tmp.name, self.desktop, "unused", 1, workflow=True)
        try:
            self.assertEqual(len(restarted.state["bounded_turn_continuation"]["attempts"]), 1)
            restarted.state.update(status="paused", turn_status="completed")
            reason = eligibility(restarted.state, owner=restarted.bounded_turn_owner, origin="running", ready=True,
                                 paused=True, unresolved=[], workflow=True, software=False)
            self.assertEqual(reason, "checkpoint_missing_changed_or_restarted")
        finally:
            await restarted.close()

    async def test_cancelled_delivery_is_durable_and_input_remains_paused(self):
        entered = asyncio.Event()
        async def turn(*args):
            entered.set()
            await asyncio.Event().wait()
        self.controller.client.turn.side_effect = turn
        self.controller.state.update(status="paused", turn_status="completed")
        task = asyncio.create_task(self.controller.continue_unchecked_zero_tool_turn("running"))
        await asyncio.wait_for(entered.wait(), 1)
        self.assertEqual(self.saved()["bounded_turn_continuation"]["attempts"][0]["delivery"], "requesting")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.saved()["bounded_turn_continuation"]["attempts"][0]["delivery"], "uncertain")
        self.assertTrue(self.desktop.paused)

    async def test_delivery_wait_is_bounded_by_the_existing_task_deadline(self):
        async def turn(*args):
            await asyncio.Event().wait()
        self.controller.client.turn.side_effect = turn
        self.controller.state.update(status="paused", turn_status="completed")
        self.controller.state["bounded_turn_continuation"]["deadline_monotonic"] = time.monotonic() + .03
        started = time.monotonic()
        continued = await self.controller.continue_unchecked_zero_tool_turn("running")
        self.assertFalse(continued)
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(self.desktop.paused)
        self.assertEqual(self.saved()["bounded_turn_continuation"]["attempts"][0]["delivery"], "uncertain")

    async def test_device_observation_failure_does_not_request_a_turn(self):
        reads = 0
        async def observe():
            nonlocal reads
            reads += 1
            if reads > 1:
                raise DesktopError("new desktop read failed")
            return self.desktop.latest
        self.desktop.observe.side_effect = observe
        await self.completed()
        self.assertEqual(self.saved()["bounded_turn_continuation"]["attempts"][0]["delivery"], "not_requested")
        self.assertTrue(self.desktop.paused)
        self.controller.client.turn.assert_not_awaited()


class PolicyAndBuildTests(unittest.TestCase):
    def test_changed_or_legacy_controller_is_not_patched(self):
        with self.assertRaises(ValueError):
            builder.patch_app(b"class Controller: pass")

    def test_invalid_checkpoint_counters_and_external_delivery_are_rejected(self):
        state = {"id": "task", "thread_id": "thread", "turn_id": "turn", "status": "paused",
                 "turn_status": "completed", "tool_calls_used": 0, "tool_call_limit": 250,
                 "bounded_turn_continuation": checkpoint("owner", 0, now=0)}
        options = dict(owner="owner", origin="running", ready=True, paused=True, unresolved=[],
                       workflow=True, software=False, now=1)
        self.assertEqual(eligibility(state, **options), "eligible")
        for field, value in (("tool_calls_used", True), ("tool_call_limit", -1),
                             ("followups", [{"delivery": "uncertain"}])):
            with self.subTest(field=field):
                self.assertNotEqual(eligibility(state | {field: value}, **options), "eligible")
        for change in ({"deadline_monotonic": float("nan")}, {"turn_start_tools": True},
                       {"policy": POLICY | {"max_additional_turns": 999}},
                       {"attempts": [{"completed_turn_id": "turn", "delivery": "accepted"}]}):
            with self.subTest(change=change):
                altered = copy.deepcopy(state)
                altered["bounded_turn_continuation"].update(change)
                self.assertNotEqual(eligibility(altered, **options), "eligible")


if __name__ == "__main__":
    unittest.main()
