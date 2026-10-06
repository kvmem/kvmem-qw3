import unittest

from qw3_computer.desktop_contract import capability_manifest, admit_task, target_facts
from qw3_computer.tool_schema import tools_for_platform


def device(platform="macos"):
    return {"platform": platform, "implemented": True, "actions": ["click", "type", "key", "open_app"],
            "open_apps": ["terminal"], "app_processes": {"terminal": ["com.apple.Terminal"]}, "available_now": True}


class DesktopContractTests(unittest.TestCase):
    def test_manifest_version_is_repeatable_and_changes_with_permissions(self):
        first = capability_manifest(device())
        self.assertEqual(first, capability_manifest(first))
        changed = capability_manifest(device() | {"available_now": False})
        self.assertNotEqual(first["capability_version"], changed["capability_version"])

    def test_implemented_feature_is_not_automatically_qualified(self):
        manifest = capability_manifest(device())
        self.assertTrue(manifest["features"]["ensure_app"]["implemented"])
        self.assertFalse(manifest["features"]["ensure_app"]["qualified_on_platform"])
        self.assertFalse(manifest["features"]["select_window"]["implemented"])
        with self.assertRaisesRegex(ValueError, "not qualified"):
            admit_task(manifest, ["ensure_app"])
        self.assertTrue(admit_task(manifest, ["ensure_app"], qualification_mode=True)["admitted"])

    def test_missing_process_mapping_cannot_advertise_verified_activation(self):
        manifest = capability_manifest(device() | {"app_processes": {}})
        self.assertFalse(manifest["features"]["ensure_app"]["implemented"])
        tools = tools_for_platform("macos", device_capabilities=manifest)
        self.assertNotIn("target", [t["name"] for t in tools])

    def test_qualification_requires_evidence_and_does_not_create_implementation(self):
        manifest = capability_manifest(device() | {"features": {
            "select_window": {"qualified_on_platform": True, "qualification_evidence": ["evidence.json"]},
            "ensure_app": {"qualified_on_platform": True}}})
        for name in ("select_window", "ensure_app"):
            self.assertFalse(manifest["features"][name]["qualified_on_platform"])
        manifest = capability_manifest(device() | {"features": {
            "ensure_app": {"qualified_on_platform": True, "qualification_evidence": ["actual-platform-result.json"]}}})
        self.assertTrue(admit_task(manifest, ["ensure_app"])["admitted"])

    def test_unavailable_features_fail_admission_even_in_qualification_mode(self):
        manifest = capability_manifest(device() | {"available_now": False})
        with self.assertRaisesRegex(ValueError, "unavailable"):
            admit_task(manifest, ["ensure_app"], qualification_mode=True)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            admit_task(capability_manifest(device()), ["select_window"], qualification_mode=True)

    def test_unknown_availability_does_not_become_true(self):
        manifest = capability_manifest(device() | {"available_now": None})
        self.assertIsNone(manifest["features"]["ensure_app"]["available_now"])
        with self.assertRaises(ValueError):
            capability_manifest(device() | {"available_now": 1})
        qualified=capability_manifest(device()|{'available_now':None,'features':{
            'ensure_app':{'qualified_on_platform':True,'qualification_evidence':['device-evidence.json']}}})
        with self.assertRaisesRegex(ValueError,'unknown'):admit_task(qualified,['ensure_app'])

    def test_tool_actions_and_launches_follow_observed_capabilities(self):
        tools = tools_for_platform("macos", True, device_capabilities=device() | {"actions": ["open_app"]})
        branches = tools[1]["inputSchema"]["properties"]["action"]["oneOf"]
        self.assertEqual(len(branches), 1)
        self.assertEqual(branches[0]["properties"]["kind"]["const"], "open_app")
        self.assertEqual(branches[0]["properties"]["app"]["enum"], ["terminal"])
        self.assertEqual(len(tools[1]["inputSchema"]["oneOf"]), 1)
        self.assertEqual(tools[-1]["name"], "target")

    def test_platform_mismatch_does_not_generate_tools(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            tools_for_platform("linux", device_capabilities=device())

    def test_missing_window_or_focus_evidence_remains_unknown(self):
        facts = target_facts({"desktop_id": "device-session", "foreground": {"process": "editor", "title": "Note"}})
        self.assertIsNone(facts["window"]["handle"])
        self.assertIsNone(facts["focused_control"])
        self.assertIsNone(facts["application"]["instance_id"])

    def test_modal_parent_and_scope_are_preserved(self):
        parent = {"pid": 3, "relation": "AXSheets", "title": "Task window"}
        facts = target_facts({"foreground": {"process": "installer", "scope_parent": parent},
                              "accessibility": {"status": "outside_task_scope"}})
        self.assertEqual(facts["owned_modal_parent"], parent)
        self.assertEqual(facts["scope"], "outside_task_scope")
