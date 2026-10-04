import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from foreground_visual_candidate import build_candidate, patch_desktop_bytes
import test_desktop as fixtures


SOURCE = Path(__file__).resolve().parents[1] / "qw3_computer" / "desktop.py"
module = types.ModuleType("qw3_computer._foreground_visual_experiment_test")
module.__package__ = "qw3_computer"
exec(compile(patch_desktop_bytes(SOURCE.read_bytes()), str(SOURCE), "exec"), module.__dict__)


class VisualForegroundTests(unittest.IsolatedAsyncioTestCase):
    # Reuse the established fake HTTP device, including real SQLite receipts.
    observation = fixtures.DesktopTests.observation
    action = fixtures.DesktopTests.action
    observed_controls = fixtures.DesktopTests.observed_controls
    enable_expired_step = fixtures.DesktopTests.enable_expired_step
    aged_context_control = fixtures.DesktopTests.aged_context_control
    change_pixel = fixtures.DesktopTests.change_pixel
    footer_control = fixtures.DesktopTests.footer_control

    async def asyncSetUp(self):
        with patch("test_desktop.Desktop", module.Desktop):
            await fixtures.DesktopTests.asyncSetUp(self)

    async def asyncTearDown(self):
        if hasattr(self, "execution"):
            self.execution.close()
        await fixtures.DesktopTests.asyncTearDown(self)

    async def footer_visual(self, action=None):
        request = await self.footer_control()
        return request | {"coordinate_space": "normalized_1000", "action": action or {
            "kind": "click", "x": 750, "y": 1000, "button": "left", "clicks": 1}}

    async def refused(self, request):
        with self.assertRaises(module.DesktopRejected):
            await self.desktop.act(request)
        self.assertEqual(self.effects, 0)
        self.assertEqual(self.desktop.unresolved(), [])

    async def test_footer_click_ignores_only_external_clock_and_dispatches_once(self):
        request = await self.footer_visual()
        result = await self.desktop.act(request)
        evidence = result["observation_revalidation"]
        self.assertTrue(evidence["visual_foreground_context_unchanged"])
        self.assertIn("every visual point inside the window", evidence["scope"])
        self.assertFalse(evidence["same_view"])
        self.assertTrue(evidence["input_sent"])
        self.assertTrue((await self.desktop.act(request))["duplicate"])
        self.assertEqual(self.effects, 1)
        self.assertEqual((self.last_dispatched["action"]["x"], self.last_dispatched["action"]["y"]), (6, 7))

    async def test_window_body_change_refuses_footer_even_when_target_pixels_unchanged(self):
        request = await self.footer_visual()
        self.change_pixel((0, 2))
        await self.refused(request)

    async def test_footer_change_away_from_target_is_not_masked(self):
        request = await self.footer_visual()
        self.change_pixel((0, 7))
        await self.refused(request)

    async def test_changed_semantic_value_refuses_unchanged_pixels(self):
        request = await self.footer_visual()
        self.controls[0]["value"] = "Changed dialog requirement"
        await self.refused(request)

    async def test_changed_foreground_refuses_unchanged_pixels(self):
        request = await self.footer_visual()
        self.observation_extra["foreground"]["handle"] = "0x99"
        await self.refused(request)

    async def test_changed_device_session_refuses_unchanged_window(self):
        request = await self.footer_visual()
        self.observation_extra["session_id"] = "new-session"
        await self.refused(request)

    async def test_duplicate_frame_evidence_is_not_positive_window_identity(self):
        request = await self.footer_visual()
        self.controls.append(dict(self.controls[-1]))
        await self.refused(request)

    async def test_unknown_accessibility_status_is_not_used_as_window_proof(self):
        request = await self.footer_visual()
        self.desktop.latest["accessibility"]["status"] = "unavailable"
        self.accessibility_status = "unavailable"
        await self.refused(request)

    async def test_desktop_chrome_point_stays_outside_proven_window(self):
        request = await self.footer_visual({"kind": "click", "x": 750, "y": 0,
                                            "button": "left", "clicks": 1})
        await self.refused(request)

    async def test_drag_requires_both_endpoints_inside_window(self):
        request = await self.footer_visual({"kind": "drag", "from_x": 750, "from_y": 1000,
            "to_x": 750, "to_y": 0, "duration_ms": 300, "modifier": "none"})
        await self.refused(request)

    async def test_drag_inside_unchanged_window_is_allowed(self):
        request = await self.footer_visual({"kind": "drag", "from_x": 750, "from_y": 1000,
            "to_x": 500, "to_y": 1000, "duration_ms": 300, "modifier": "none"})
        result = await self.desktop.act(request)
        self.assertTrue(result["observation_revalidation"]["visual_foreground_context_unchanged"])
        self.assertEqual(self.effects, 1)

    async def test_windows_does_not_inherit_linux_window_exception(self):
        request = await self.footer_visual()
        self.observation_extra["capabilities"]["platform"] = "windows"
        await self.refused(request)


class CandidateBuildTests(unittest.TestCase):
    def test_build_is_private_changes_only_desktop_and_never_overwrites(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.mkdir()
            original = SOURCE.read_bytes()
            (source / "desktop.py").write_bytes(original)
            (source / "__init__.py").write_text("# isolated source\n")
            files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            protocol = Path(temp) / "protocol.json"
            protocol.write_text(json.dumps({"framework_sha256": files}))
            output = Path(temp) / "candidate"
            result = build_candidate(source, protocol, output)
            self.assertEqual(result["changed_package_files"], ["desktop.py"])
            self.assertEqual((source / "desktop.py").read_bytes(), original)
            self.assertFalse(result["real_model_retest"])
            self.assertFalse(result["native_windows_qualified"])
            with self.assertRaises(FileExistsError):
                build_candidate(source, protocol, output)

    def test_changed_source_is_rejected_before_build(self):
        with self.assertRaisesRegex(ValueError, "exact qualified"):
            patch_desktop_bytes(SOURCE.read_bytes() + b"\n# changed\n")


if __name__ == "__main__":
    unittest.main()
