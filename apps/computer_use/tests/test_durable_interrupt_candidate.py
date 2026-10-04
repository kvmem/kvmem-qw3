import asyncio
import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock

from durable_interrupt_candidate import build_candidate, patch_app_bytes
from qw3_computer.app import Controller as BaselineController
from test_controller import FixtureCodex, FixtureDesktop


SOURCE = Path(__file__).resolve().parents[1] / "qw3_computer" / "app.py"
module = types.ModuleType("qw3_computer._durable_interruption_experiment_test")
module.__package__ = "qw3_computer"
exec(compile(patch_app_bytes(SOURCE.read_bytes()), str(SOURCE), "exec"), module.__dict__)


class InterruptedPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def interrupted_at_device_await(self, controller_class, task_status="paused", turn_status="interrupted"):
        with tempfile.TemporaryDirectory() as temp:
            desktop = FixtureDesktop()
            entered, release = asyncio.Event(), asyncio.Event()
            async def pause(command, *_):
                self.assertEqual(command, "pause")
                desktop.paused = True
                entered.set()
                await release.wait()
            desktop.control = pause
            controller = controller_class(temp, desktop, "unused", 1, model_output_tokens=16384)
            controller.client = FixtureCodex()
            controller.client.turn = AsyncMock()
            controller.state.update(id="original-task", thread_id="original-thread", turn_id="original-turn",
                status=task_status, tool_calls_used=7, tool_call_limit=100, model_output_tokens=16384)
            controller.save()
            consumer = asyncio.create_task(controller.consume())
            try:
                await controller.client.events.put({"method": "turn/completed", "params": {
                    "threadId": "original-thread", "turn": {"id": "original-turn", "status": turn_status}}})
                await asyncio.wait_for(entered.wait(), 2)
                self.assertEqual(controller.state["turn_status"], turn_status)
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer
                saved = json.loads(controller.state_path.read_text())
                restarted = controller_class(temp, desktop, "unused", 1, model_output_tokens=16384)
                self.assertEqual(restarted.state["tool_calls_used"], 7)
                self.assertEqual(restarted.state["tool_call_limit"], 100)
                self.assertEqual(restarted.state["id"], "original-task")
                self.assertTrue(desktop.paused)
                controller.client.turn.assert_not_called()
                return saved, restarted.state
            finally:
                consumer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer

    async def test_unmodified_baseline_loses_interrupt_status_at_this_race(self):
        saved, restarted = await self.interrupted_at_device_await(BaselineController)
        self.assertNotIn("turn_status", saved)
        self.assertNotIn("turn_status", restarted)

    async def test_candidate_persists_interruption_before_device_await_and_restart(self):
        saved, restarted = await self.interrupted_at_device_await(module.Controller)
        self.assertEqual(saved["turn_status"], "interrupted")
        self.assertEqual(restarted["turn_status"], "interrupted")
        self.assertEqual(restarted["status"], "paused")

    async def test_stopped_task_remains_stopped(self):
        saved, restarted = await self.interrupted_at_device_await(module.Controller, "stopped")
        self.assertEqual(saved["turn_status"], "interrupted")
        self.assertEqual(restarted["status"], "stopped")

    async def test_completed_model_turn_is_never_premature_business_completion(self):
        saved, restarted = await self.interrupted_at_device_await(module.Controller, "running", "completed")
        self.assertEqual(saved["status"], "verifying")
        self.assertEqual(saved["turn_status"], "completed")
        self.assertEqual(restarted["status"], "paused")

    async def test_failed_model_turn_does_not_replenish_budget_or_claim_completion(self):
        saved, restarted = await self.interrupted_at_device_await(module.Controller, "running", "failed")
        self.assertEqual(saved["turn_status"], "failed")
        self.assertEqual(restarted["status"], "paused")


class CandidateBuildTests(unittest.TestCase):
    def test_source_is_preserved_only_app_changes_and_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);source = root / "source";source.mkdir()
            (source / "app.py").write_bytes(SOURCE.read_bytes())
            (source / "__init__.py").write_text("# source\n")
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            protocol = root / "protocol.json";protocol.write_text(json.dumps({"framework_sha256": files}))
            result = build_candidate(source, protocol, root / "candidate")
            self.assertEqual(result["changed_package_files"], ["app.py"])
            self.assertEqual((source / "app.py").read_bytes(), SOURCE.read_bytes())
            self.assertFalse(result["real_model_retest"])
            with self.assertRaises(FileExistsError):
                build_candidate(source, protocol, root / "candidate")

    def test_changed_source_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exact qualified"):
            patch_app_bytes(SOURCE.read_bytes() + b"\n# changed\n")


if __name__ == "__main__":
    unittest.main()
