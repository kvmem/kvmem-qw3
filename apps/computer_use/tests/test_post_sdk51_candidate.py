import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from aiohttp import web

import test_desktop as desktop_fixtures
import test_device_gateway as gateway_fixtures
import test_foreground_visual_candidate as visual_fixtures
from foreground_visual_candidate import patch_desktop_bytes as sdk51_desktop
from post_sdk51_candidate import build_candidate, patch_desktop_bytes, patch_gateway_bytes


SOURCE = Path(__file__).resolve().parents[1] / "qw3_computer"
desktop_source = sdk51_desktop((SOURCE / "desktop.py").read_bytes())
desktop = types.ModuleType("qw3_computer._post51_desktop_test")
desktop.__package__ = "qw3_computer"
exec(compile(patch_desktop_bytes(desktop_source), "post51-desktop", "exec"), desktop.__dict__)
gateway = types.ModuleType("qw3_computer._post51_gateway_test")
gateway.__package__ = "qw3_computer"
exec(compile(patch_gateway_bytes((SOURCE / "device_gateway.py").read_bytes()),
             "post51-gateway", "exec"), gateway.__dict__)


class WindowRoleTests(visual_fixtures.VisualForegroundTests):
    async def asyncSetUp(self):
        with patch("test_desktop.Desktop", desktop.Desktop):
            await desktop_fixtures.DesktopTests.asyncSetUp(self)

    async def refused(self, request):
        with self.assertRaises(desktop.DesktopRejected):
            await self.desktop.act(request)
        self.assertEqual(self.effects, 0)
        self.assertEqual(self.desktop.unresolved(), [])

    async def chooser_request(self, action=None, role="file-chooser"):
        request = await self.footer_visual(action)
        # Preserve both captures and their changed external clock. Only the
        # window role is changed in these controlled, synthetic observations.
        self.controls[-1]["kind"] = role
        self.desktop.latest["accessibility"]["controls"][-1]["kind"] = role
        if request["action"]["kind"] in {"key", "type"}:
            request.pop("coordinate_space", None)
        return request

    async def test_file_chooser_visual_footer_is_delivered_once(self):
        request = await self.chooser_request()
        old = self.desktop.latest
        self.assertIsNone(visual_fixtures.module.Desktop.matched_foreground_bounds(self.desktop, old, old))
        result = await self.desktop.act(request)
        self.assertTrue(result["observation_revalidation"]["visual_foreground_context_unchanged"])
        self.assertTrue((await self.desktop.act(request))["duplicate"])
        self.assertEqual(self.effects, 1)

    async def test_dialog_keyboard_uses_full_window_proof(self):
        request = await self.chooser_request({"kind": "key", "key": "CTRL+S"}, "dialog")
        result = await self.desktop.act(request)
        self.assertTrue(result["observation_revalidation"]["foreground_context_unchanged"])
        self.assertFalse(result["observation_revalidation"]["same_view"])
        self.assertEqual(self.effects, 1)

    async def test_chooser_typing_uses_full_window_proof(self):
        request = await self.chooser_request({"kind": "type", "text": "report.txt"})
        result = await self.desktop.act(request)
        self.assertTrue(result["observation_revalidation"]["foreground_context_unchanged"])
        self.assertEqual(self.effects, 1)

    async def test_changed_chooser_body_refuses_keyboard(self):
        request = await self.chooser_request({"kind": "key", "key": "ENTER"})
        self.change_pixel((0, 2))
        await self.refused(request)

    async def test_changed_chooser_filename_refuses_typing(self):
        request = await self.chooser_request({"kind": "type", "text": "report.txt"})
        self.controls[0]["value"] = "unexpected.txt"
        await self.refused(request)

    async def test_same_title_dialog_and_chooser_are_ambiguous(self):
        request = await self.chooser_request({"kind": "key", "key": "ENTER"})
        self.controls.append(self.controls[-1] | {"kind": "dialog"})
        await self.refused(request)

    async def test_clipped_chooser_bounds_do_not_count_as_whole_window(self):
        request = await self.chooser_request({"kind": "key", "key": "ENTER"})
        for controls in (self.controls, self.desktop.latest["accessibility"]["controls"]):
            controls[-1]["width"] = 9
        await self.refused(request)

    async def test_embedded_widget_does_not_count_as_window(self):
        request = await self.chooser_request({"kind": "key", "key": "ENTER"}, "panel")
        await self.refused(request)

    async def test_wrong_chooser_title_does_not_count_as_window(self):
        request = await self.chooser_request({"kind": "key", "key": "ENTER"})
        for controls in (self.controls, self.desktop.latest["accessibility"]["controls"]):
            controls[-1]["name"] = "Background Save As"
        await self.refused(request)


class PhaseJournalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.driver = gateway_fixtures.FakeDriver()
        self.gateway = gateway.Gateway(self.driver, Path(self.temp.name))
        await self.gateway.observe()
        self.gateway.paused = False

    async def asyncTearDown(self):
        await self.gateway.close()
        self.temp.cleanup()

    def action(self):
        return {"action_id": "phase-test", "desktop_id": self.gateway.desktop_id,
                "observation_id": self.gateway.latest["observation_id"],
                "action": {"kind": "click", "x": 8, "y": 8, "clicks": 1, "button": "left"}}

    def phases(self):
        return self.gateway.receipt("phase-test")["phase_evidence"]

    async def test_normal_delivery_remains_unverified_and_duplicate_sends_no_input(self):
        request = self.action()
        result = await self.gateway.act(request)
        before = self.phases()
        self.assertEqual([r["phase"] for r in before], ["reserved", "driver_returned", "post_input_observed"])
        self.assertIsNone(result["verification"]["passed"])
        self.assertTrue(all(r["business_verified"] is False for r in before))
        self.assertTrue((await self.gateway.act(request))["duplicate"])
        self.assertEqual(self.phases(), before)
        self.assertEqual(self.driver.effects, 1)

    async def test_capture_failure_retains_driver_return_but_action_stays_uncertain(self):
        self.driver.mode = "invalid_observation"
        request = self.action()
        with self.assertRaises(RuntimeError):
            await self.gateway.act(request)
        self.assertEqual([r["phase"] for r in self.phases()], ["reserved", "driver_returned"])
        self.assertEqual(self.phases()[-1]["detail"], {"delivered": True})
        self.assertEqual(self.gateway.receipt("phase-test")["status"], "uncertain")
        self.assertIsNone(self.gateway.receipt("phase-test")["result"])
        self.assertTrue(self.gateway.paused)
        with self.assertRaises(gateway.DeviceRejected):
            await self.gateway.act(request)
        self.assertEqual(self.driver.effects, 1)

    async def test_cancel_during_capture_keeps_already_committed_driver_evidence(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.driver.observe

        async def observe():
            if self.driver.effects:
                entered.set()
                await release.wait()
            return await original()

        self.driver.observe = observe
        pending = asyncio.create_task(self.gateway.act(self.action()))
        await asyncio.wait_for(entered.wait(), 2)
        self.assertEqual([r["phase"] for r in self.phases()], ["reserved", "driver_returned"])
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(self.gateway.receipt("phase-test")["status"], "uncertain")
        self.assertTrue(self.gateway.paused)
        self.assertEqual(self.driver.effects, 1)

    async def test_cancel_inside_driver_does_not_invent_return_evidence(self):
        self.driver.mode = "hang"
        pending = asyncio.create_task(self.gateway.act(self.action()))
        await asyncio.wait_for(self.driver.entered.wait(), 2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual([r["phase"] for r in self.phases()], ["reserved"])
        self.assertEqual(self.gateway.receipt("phase-test")["status"], "uncertain")
        self.assertEqual(self.driver.effects, 1)

    async def test_restart_preserves_original_session_and_never_replays_input(self):
        self.driver.mode = "invalid_observation"
        request = self.action()
        with self.assertRaises(RuntimeError):
            await self.gateway.act(request)
        before, old_session = self.phases(), self.gateway.session_id
        await self.gateway.close()
        self.gateway = gateway.Gateway(self.driver, Path(self.temp.name))
        self.assertNotEqual(self.gateway.session_id, old_session)
        self.assertTrue(self.gateway.paused)
        self.assertEqual(self.phases(), before)
        self.assertTrue(all(r["session_id"] == old_session for r in self.phases()))
        with self.assertRaises(gateway.DeviceRejected):
            await self.gateway.act(request)
        self.assertEqual(self.driver.effects, 1)

    async def test_preflight_rejection_has_no_reservation_or_delivery_journal(self):
        self.driver.mode = "preflight_reject"
        with patch("test_device_gateway.DeviceRejected", gateway.DeviceRejected), self.assertRaises(gateway.DeviceRejected):
            await self.gateway.act(self.action())
        self.assertEqual(self.gateway.receipt("phase-test"), {"status": "unknown"})
        self.assertEqual(self.gateway.db.execute("SELECT COUNT(*) FROM action_phases").fetchone()[0], 0)
        self.assertEqual(self.driver.effects, 0)

    async def test_transition_recaptures_once_without_repeating_driver(self):
        self.driver.mode = "transition"
        await self.gateway.act(self.action())
        self.assertEqual(self.phases()[-1]["detail"]["captures"], 2)
        self.assertEqual(self.driver.effects, 1)

    async def test_changed_request_cannot_adopt_existing_phase_evidence(self):
        request = self.action()
        await self.gateway.act(request)
        before = self.phases()
        with self.assertRaises(gateway.DeviceRejected):
            await self.gateway.act(request | {"action": {"kind": "key", "key": "ENTER"}})
        self.assertEqual(self.phases(), before)
        self.assertEqual(self.driver.effects, 1)

    async def test_tampered_phase_request_digest_is_refused(self):
        await self.gateway.act(self.action())
        record = self.phases()[0] | {"request_sha256": "wrong"}
        self.gateway.db.execute("UPDATE action_phases SET evidence=? WHERE id=? AND phase=?",
                                (json.dumps(record), "phase-test", "reserved"))
        self.gateway.db.commit()
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.gateway.receipt("phase-test")
        self.assertEqual(self.driver.effects, 1)

    async def test_legacy_unknown_row_is_not_backfilled_or_reclassified(self):
        request = json.dumps(self.action(), sort_keys=True, ensure_ascii=False)
        self.gateway.db.execute("INSERT INTO actions VALUES (?,?,'uncertain',NULL)", ("phase-test", request))
        self.gateway.db.commit()
        receipt = self.gateway.receipt("phase-test")
        self.assertEqual(receipt["status"], "uncertain")
        self.assertEqual(receipt["phase_evidence"], [])
        self.assertEqual(self.driver.effects, 0)

    async def test_client_reconciliation_ignores_phase_evidence_without_completed_receipt(self):
        # Exercise the HTTP/SQLite boundary. Added diagnostics cannot silently
        # turn an uncertain action into delivery or task acceptance.
        self.driver.mode = "invalid_observation"
        app = gateway.create_app(self.gateway, "fixture-token")
        runner, url = await desktop_fixtures.serve(app)
        with tempfile.TemporaryDirectory() as temp:
            client = desktop.Desktop(url, "fixture-token", temp)
            try:
                await client.start()
                await client.control("resume")
                with self.assertRaises(desktop.DesktopError):
                    await client.act({"action_id": "phase-client", "desktop_id": client.latest["desktop_id"],
                        "observation_id": client.latest["observation_id"], "action": {"kind": "key", "key": "ENTER"}})
                action = client.unresolved()[0]
                remote = self.gateway.receipt(action["receipt_id"])
                self.assertEqual([r["phase"] for r in remote["phase_evidence"]], ["reserved", "driver_returned"])
                report = await client.reconcile_pending()
                self.assertEqual(report["resolved"], [])
                self.assertEqual(len(report["remaining"]), 1)
                self.assertTrue(client.paused)
                self.assertTrue(self.gateway.paused)
                self.assertEqual(self.driver.effects, 1)
            finally:
                await client.close()
                await runner.cleanup()


class BuildTests(unittest.TestCase):
    def test_only_two_private_files_change_and_sources_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            originals = {"desktop.py": desktop_source,
                         "device_gateway.py": (SOURCE / "device_gateway.py").read_bytes(),
                         "__init__.py": b"# qualification fixture\n"}
            for name, contents in originals.items():
                (source / name).write_bytes(contents)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"source_files": {
                "qw3_computer/" + name: hashlib.sha256(data).hexdigest() for name, data in originals.items()}}))
            result = build_candidate(source, manifest, root / "candidate")
            self.assertEqual(set(result["changed_package_files"]), {"desktop.py", "device_gateway.py"})
            self.assertFalse(result["qualified_for_production"])
            self.assertFalse(result["real_model_retest"])
            for name, data in originals.items():
                self.assertEqual((source / name).read_bytes(), data)
            with self.assertRaises(FileExistsError):
                build_candidate(source, manifest, root / "candidate")
            with self.assertRaises(ValueError):
                build_candidate(source, manifest, source / "nested")

    def test_changed_qualified_sources_are_refused(self):
        for function, source in ((patch_desktop_bytes, desktop_source),
                                 (patch_gateway_bytes, (SOURCE / "device_gateway.py").read_bytes())):
            with self.assertRaisesRegex(ValueError, "exact qualified"):
                function(source + b"\n# changed\n")


if __name__ == "__main__":
    unittest.main()
