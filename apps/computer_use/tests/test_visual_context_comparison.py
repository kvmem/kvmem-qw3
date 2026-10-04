"""Matched image-policy contracts without live input or model inference."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

from test_visual_working_context_candidate import context, observed
from visual_context_comparison_candidate import build_candidate


ROOT = Path(__file__).resolve().parents[1]
TEMP = tempfile.TemporaryDirectory(prefix="qw3-image-comparison-tests-")
OUTPUT = Path(TEMP.name) / "candidate"
BUILD = build_candidate(ROOT / ".runtime/visual-working-context-candidate-03", OUTPUT)
package = types.ModuleType("qw3_comparison_test")
package.__path__ = [str(OUTPUT / "framework/qw3_computer")]
sys.modules[package.__name__] = package
policy = importlib.import_module(package.__name__ + ".visual_context_policy")
history = importlib.import_module(package.__name__ + ".observation_history")
archive = importlib.import_module(package.__name__ + ".visual_request_archive")
codec = importlib.import_module(package.__name__ + ".request_archive")
bridge = importlib.import_module(package.__name__ + ".bridge")
app = importlib.import_module(package.__name__ + ".app")
desktop_module = importlib.import_module(package.__name__ + ".desktop")


class PolicyComparisonTests(unittest.TestCase):
    def test_full_and_recent_have_identical_guidance_tools_parameters_and_reasoning(self):
        source = context(); original = copy.deepcopy(source)
        full, f = policy.select(source, policy="full-retrievable")
        recent, r = policy.select(source)
        self.assertEqual(source, original)
        self.assertEqual(full["messages"][0], recent["messages"][0])
        self.assertEqual(full["messages"][1:], source["messages"])
        self.assertEqual({k:v for k,v in full.items() if k!="messages"}, {k:v for k,v in recent.items() if k!="messages"})
        self.assertEqual(f["retained_image_count"], 7)
        self.assertEqual(f["dropped_images"], 0)
        self.assertEqual(r["retained_image_count"], 3)
        for a, b in zip(full["messages"], recent["messages"]):
            if a["role"] != "tool": self.assertEqual(a, b)
            else: self.assertEqual(a["content"][:2], b["content"][:2])

    def test_both_arms_keep_user_images_and_pinned_reference(self):
        source = context(); pin = history.observation_reference(observed(0))
        full, f = policy.select(source, [pin], "full-retrievable")
        recent, r = policy.select(source, [pin])
        self.assertEqual(full["messages"][2], recent["messages"][2])
        self.assertEqual(full["messages"][4], recent["messages"][4])
        self.assertIn(pin, f["selected"]); self.assertIn(pin, r["selected"])

    def test_missing_pins_fall_back_identically_without_silent_loss(self):
        source = context(); pin = history.observation_reference(observed(77))
        for mode in ("full-retrievable", "recent-observations"):
            view, report = policy.select(source, [pin], mode)
            self.assertIs(view, source)
            self.assertFalse(report["applied"])
            self.assertEqual(report["missing_pins"], [pin])

    def test_absent_history_tool_keeps_both_inputs_unchanged(self):
        source = context(); source["tools"] = []
        for mode in ("full-retrievable", "recent-observations"):
            view, report = policy.select(source, policy=mode)
            self.assertIs(view, source); self.assertFalse(report["applied"])

    def test_invalid_policy_is_rejected(self):
        for mode in ("full", "random", None):
            with self.assertRaises(ValueError): policy.select(context(), policy=mode)

    def test_each_arm_archives_the_same_full_source_and_exact_view(self):
        source = context(); hashes = []
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("full-retrievable", "recent-observations"):
                trace = Path(directory) / mode; trace.mkdir()
                view, report = archive.present_request(source, [], trace, mode)
                stored = json.loads((trace / "request.full.archive.json").read_text())
                restored = codec.decode(stored["tree"], trace.parent / "full-history-blobs")
                self.assertEqual(restored, source); hashes.append(stored["sha256"])
                self.assertEqual(report["presented_sha256"], hashlib.sha256(json.dumps(view, ensure_ascii=False).encode()).hexdigest())
                self.assertEqual(report["policy"], mode)
                self.assertFalse(report["model_latency_improvement_measured"])
            self.assertEqual(*hashes)

    def test_only_declared_private_files_change(self):
        changed = {n for n,sha in BUILD["candidate_files_sha256"].items() if sha != BUILD["source_files_sha256"].get(n)}
        self.assertEqual(changed, {"app.py", "bridge.py", "visual_request_archive.py", "visual_context_policy.py"})
        for name in ("execution.py", "desktop.py", "codex.py", "mcp_proxy.py", "tool_schema.py", "observation_history.py"):
            self.assertEqual(BUILD["candidate_files_sha256"][name], BUILD["source_files_sha256"][name])

    def test_bridge_rejects_policy_without_owned_callback(self):
        with self.assertRaises(ValueError): bridge.create_app("http://127.0.0.1:9/v1", image_policy="full-retrievable")
        with self.assertRaises(ValueError): bridge.create_app("http://127.0.0.1:9/v1", image_policy="invalid")


class ControllerComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.controllers = []

    async def asyncTearDown(self):
        for controller in self.controllers: await controller.close()
        self.temp.cleanup()

    def controller(self, policy="full", workflow=False):
        index = len(self.controllers)
        device = desktop_module.Desktop("http://127.0.0.1:9", "fixture", self.root / str(index) / "device")
        try:
            controller = app.Controller(self.root / str(index) / "app", device, "http://127.0.0.1:9/v1", 0,
                workflow=workflow, image_history=policy, tool_call_limit=10)
        except ValueError:
            device.db.close()
            raise
        self.controllers.append(controller)
        return controller

    async def test_default_full_has_no_history_instrumentation(self):
        controller = self.controller()
        self.assertIsNone(controller.history); self.assertIsNone(controller.workflow)
        self.assertEqual(controller.image_history, "full")

    async def test_both_experimental_arms_require_verified_workflow(self):
        for mode in ("full-retrievable", "recent-observations"):
            with self.assertRaises(ValueError): self.controller(mode)

    async def test_both_experimental_arms_have_the_same_history_and_budget(self):
        for mode in ("full-retrievable", "recent-observations"):
            controller = self.controller(mode, True)
            self.assertIsNotNone(controller.history); self.assertIsNotNone(controller.workflow)
            self.assertEqual(controller.tool_call_limit, 10)

    async def test_resume_cannot_silently_switch_image_policy(self):
        controller = self.controller("full-retrievable", True)
        controller.state["image_history"] = "recent-observations"
        with self.assertRaises(desktop_module.DesktopError): controller.require_saved_inference_mode()


if __name__ == "__main__": unittest.main()
