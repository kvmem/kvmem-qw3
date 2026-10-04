import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from literal_completion_feedback_candidate import build_candidate, digest, patch_execution
from osworld_linux.runner import source_identity
from qw3_computer.execution import check_observation as original_check
from qw3_computer.desktop import DesktopError, DesktopValidationError
from test_execution import FakeDevice


SOURCE = Path(__file__).resolve().parents[1]
temporary = tempfile.TemporaryDirectory()
package = Path(temporary.name) / "literal_feedback_test"
package.mkdir()
(package / "__init__.py").write_text("")
(package / "desktop.py").write_text(
    "from qw3_computer.desktop import DesktopError, DesktopValidationError\n")
(package / "completion_feedback.py").write_bytes((SOURCE / "completion_feedback.py").read_bytes())
(package / "execution.py").write_bytes(patch_execution((SOURCE / "qw3_computer/execution.py").read_bytes()))
spec = importlib.util.spec_from_file_location("literal_feedback_test", package / "__init__.py",
                                            submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
import importlib
execution = importlib.import_module("literal_feedback_test.execution")


def tearDownModule():
    for name in tuple(sys.modules):
        if name == "literal_feedback_test" or name.startswith("literal_feedback_test."):
            del sys.modules[name]
    temporary.cleanup()


def observation():
    return {"observation_id": "now", "progress_sha256": "current",
            "foreground": {"process": "Evince", "title": "prize-items.pdf"},
            "capabilities": {"app_processes": {"pdf": ["Evince"]}},
            "accessibility": {"status": "ok", "truncated": False,
                "controls": [{"name": "page-label-entry", "kind": "text", "value": "2"},
                             {"name": "Save", "kind": "button", "value": None}]}}


class LiteralPredicateTests(unittest.TestCase):
    def compare(self, checks, obs=None, before=None):
        obs = obs or observation(); before = before or {"progress_sha256": "previous"}
        prior = copy.deepcopy((checks, obs, before))
        old = original_check(checks, obs, before)
        new = execution.check_observation(checks, obs, before)
        self.assertEqual((checks, obs, before), prior)
        for old_item, new_item in zip(old, new):
            for key, value in old_item.items():
                self.assertEqual(new_item[key], value)
            self.assertFalse(new_item["comparison_evidence"]["input_delivery_inferred"])
            self.assertFalse(new_item["comparison_evidence"]["independent_business_result_verified"])
        return new

    def test_all_six_predicates_remain_literal(self):
        checks = [{"kind": "control_text", "text": "Save"},
                  {"kind": "control_absent", "text": "Missing"},
                  {"kind": "control_value", "text": "page-label-entry", "value": "2", "control_kind": "text"},
                  {"kind": "foreground_title", "text": "prize-items.pdf"},
                  {"kind": "foreground_app", "text": "pdf"}, {"kind": "screen_changed"}]
        self.assertTrue(all(x["passed"] for x in self.compare(checks)))

    def test_opaque_document_title_is_not_a_control_or_failed_delivery(self):
        result = self.compare([{"kind": "control_text", "text": "Prize Items"}])[0]
        self.assertFalse(result["passed"])
        self.assertEqual(result["matching_controls"], 0)
        self.assertEqual(result["comparison_evidence"]["scope"], "positive_visible_control_name_only")
        self.assertIn("outside that tree", result["hint"])

    def test_partial_and_unavailable_reads_do_not_establish_absence(self):
        for status, truncated in [("ok", True), ("unavailable", False)]:
            obs = observation(); obs["accessibility"].update(status=status, truncated=truncated)
            result = self.compare([{"kind": "control_absent", "text": "Missing"}], obs)[0]
            self.assertFalse(result["passed"])
            self.assertEqual(result["comparison_evidence"]["reason"], "complete_read_not_established")

    def test_duplicate_field_remains_ambiguous_with_original_hint(self):
        obs = observation(); obs["accessibility"]["controls"].append(copy.deepcopy(obs["accessibility"]["controls"][0]))
        result = self.compare([{"kind": "control_value", "text": "page-label-entry", "value": "2"}], obs)[0]
        self.assertFalse(result["passed"])
        self.assertEqual(result["matching_controls"], 2)
        self.assertIn("exact label is ambiguous", result["hint"])

    def test_guessed_process_does_not_become_a_supported_alias(self):
        result = self.compare([{"kind": "foreground_app", "text": "document-viewer"}])[0]
        self.assertFalse(result["passed"])
        self.assertEqual(result["comparison_evidence"]["observed_process_capability_aliases"], ["pdf"])

    def test_save_button_and_visual_transition_are_not_save_proof(self):
        result = self.compare([{"kind": "control_text", "text": "Save"}, {"kind": "screen_changed"}])
        self.assertTrue(all(x["passed"] for x in result))
        self.assertEqual(result[1]["comparison_evidence"]["scope"], "visual_transition_only_no_focus_or_save_proof")

    def test_unknown_new_condition_does_not_enter_the_sdk51_schema(self):
        with self.assertRaises(DesktopError):
            execution.validate_checks([{"kind": "field_value", "field_ref": "guessed", "value": "2"}])

    def test_changed_execution_bytes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "declared SDK51"):
            patch_execution((SOURCE / "qw3_computer/execution.py").read_bytes() + b"\n")


class DurableFlowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.device = FakeDevice()
        self.flow = execution.Execution(self.device, Path(self.temp.name))

    async def asyncTearDown(self):
        self.flow.close(); self.temp.cleanup()

    async def begin(self):
        return await self.flow.invoke({"operation": "begin_step", "step_id": "read", "goal": "Open target page",
            "conditions": [{"kind": "control_text", "text": "Prize Items"}]})

    async def test_wait_stays_pending_without_input_and_returns_scope_feedback(self):
        await self.begin()
        reads_before_wait = self.device.reads
        result = await self.flow.invoke({"operation": "wait", "timeout_seconds": 0})
        self.assertFalse(result["verified"])
        self.assertFalse(result["input_dispatched"])
        self.assertEqual(self.flow.steps()[0]["status"], "pending")
        self.assertEqual(result["conditions"][0]["comparison_evidence"]["scope"], "positive_visible_control_name_only")
        self.assertEqual(result["polls"], 1)
        self.assertEqual(self.device.reads, reads_before_wait + 1)

    async def test_changed_criterion_needs_recovery_and_cannot_replace_uncertain_input(self):
        await self.begin()
        action = {"action": {"kind": "key", "key": "PAGEDOWN"}}
        definition = {"goal": "Open target page", "conditions": [{"kind": "foreground_title", "text": "Request"}]}
        with self.assertRaises(DesktopValidationError):
            self.flow.before_dispatch({"step": definition}, action, "q3_one")
        self.device.unresolved_rows = [{"action_id": "unknown"}]
        definition["recovery_reason"] = "Old criterion was not exposed in native controls"
        with self.assertRaises(DesktopValidationError):
            self.flow.before_dispatch({"step": definition}, action, "q3_two")
        self.assertEqual(self.flow.steps()[0]["status"], "pending")

    async def test_restart_keeps_repeat_intent_reservations(self):
        await self.begin()
        action = {"action": {"kind": "key", "key": "PAGEDOWN"}}
        self.flow.before_dispatch({}, action, "q3_one")
        self.flow.before_dispatch({}, action, "q3_two")
        self.flow.close(); self.flow = execution.Execution(self.device, Path(self.temp.name))
        with self.assertRaises(DesktopValidationError) as raised:
            self.flow.before_dispatch({}, action, "q3_three")
        self.assertEqual(raised.exception.problem["code"], "repeated_intent")

    async def test_completion_is_rechecked_and_not_relaxed_by_feedback(self):
        await self.flow.invoke({"operation": "begin_step", "step_id": "open", "goal": "Open request",
            "conditions": [{"kind": "foreground_title", "text": "Request"}]})
        await self.flow.invoke({"operation": "wait", "timeout_seconds": 0})
        result = await self.flow.invoke({"operation": "finish", "conditions": [{"kind": "foreground_title", "text": "Request"}], "timeout_seconds": 0})
        self.assertTrue(result["verified"])
        self.device.latest["foreground"]["title"] = "Different"
        self.assertFalse(self.flow.completion(self.device.latest)["verified"])


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source = self.root / "source"; self.package = self.source / "qw3_computer"
        shutil.copytree(SOURCE / "qw3_computer", self.package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        files = {p.name: digest(p.read_bytes()) for p in self.package.iterdir()
                 if p.is_file() and p.suffix in {".py", ".html", ".json", ".tex"}}
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"kind": "isolated-qualified-execution-fix-candidate", "files": files,
            "candidate_changes": ["test_fixture_not_native_qualification"]}))
        (self.source / "device_gateway_entry.py").write_text("# scripted fixture, never executed\n")
        (self.source / "observer.py").write_text("# scripted fixture, never executed\n")
        self.acceptance = self.root / "acceptance.json"
        self.acceptance.write_text(json.dumps({"framework_files": source_identity(self.source),
            "canonical_entry_sha256": digest((self.source / "device_gateway_entry.py").read_bytes()),
            "framework_auxiliary_files": {"observer.py": digest((self.source / "observer.py").read_bytes())}}))
        self.output = self.root / "candidate"

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **changes):
        kwargs = {"source_framework": self.source, "source_manifest": self.manifest,
                  "source_acceptance": self.acceptance, "output": self.output,
                  "manifest_sha256": digest(self.manifest.read_bytes()),
                  "acceptance_sha256": digest(self.acceptance.read_bytes())}
        return build_candidate(**(kwargs | changes))

    def test_source_copy_only_changes_feedback_wiring_and_helper(self):
        before = digest((self.package / "execution.py").read_bytes())
        result = self.build()
        self.assertEqual(set(result["changed_package_files"]), {"execution.py", "completion_feedback.py"})
        self.assertEqual(digest((self.package / "execution.py").read_bytes()), before)
        self.assertEqual((self.output / "framework/observer.py").read_bytes(), (self.source / "observer.py").read_bytes())
        self.assertFalse(result["original_predicates_changed"])
        self.assertFalse(result["input_schemas_changed"])
        self.assertFalse(result["native_windows_qualified"])
        self.assertFalse(result["qualified_for_production"])

    def test_wrong_manifest_hash_stops_before_output(self):
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            self.build(manifest_sha256="0" * 64)
        self.assertFalse(self.output.exists())

    def test_auxiliary_corruption_stops_before_output(self):
        (self.source / "observer.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "auxiliary changed"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_package_corruption_stops_before_output(self):
        (self.package / "execution.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "inventory changed"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_existing_output_and_source_nested_output_are_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "outside immutable"):
            self.build(output=self.source / "nested")
        self.build()
        before = (self.output / "build.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.output / "build.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
