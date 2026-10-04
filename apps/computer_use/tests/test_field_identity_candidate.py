import asyncio
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import aiohttp

import test_desktop as fixtures
from field_identity import identify_fields, provider_identity_from_attrs, atspi_identity_script
from field_identity_candidate import patch_package_file, build_candidate
from foreground_visual_candidate import patch_desktop_bytes as visual_patch
from linux_dialog_candidate import patch_driver_bytes as dialog_patch
from post_sdk51_candidate import patch_desktop_bytes as role_patch, patch_gateway_bytes


SOURCE = Path(__file__).resolve().parents[1] / "qw3_computer"
SEED = {"desktop.py": role_patch(visual_patch((SOURCE / "desktop.py").read_bytes())),
        "linux_driver.py": dialog_patch((SOURCE / "linux_driver.py").read_bytes()),
        "device_gateway.py": patch_gateway_bytes((SOURCE / "device_gateway.py").read_bytes())}
package_temp = tempfile.TemporaryDirectory()
package = Path(package_temp.name) / "qw3_field_test"
package.mkdir()
for source in SOURCE.iterdir():
    if not source.is_file() or source.suffix not in {".py", ".html", ".json", ".tex"}:
        continue
    data = SEED.get(source.name, source.read_bytes())
    if source.name in ("desktop.py", "linux_driver.py", "execution.py", "tool_schema.py"):
        data = patch_package_file(source.name, data)
    (package / source.name).write_bytes(data)
(package / "field_identity.py").write_bytes(SOURCE.with_name("field_identity.py").read_bytes())
spec = importlib.util.spec_from_file_location("qw3_field_test", package / "__init__.py",
                                            submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
desktop = importlib.import_module("qw3_field_test.desktop")
execution = importlib.import_module("qw3_field_test.execution")
schema = importlib.import_module("qw3_field_test.tool_schema")
linux = importlib.import_module("qw3_field_test.linux_driver")


def tearDownModule():
    for name in tuple(sys.modules):
        if name == "qw3_field_test" or name.startswith("qw3_field_test."):
            del sys.modules[name]
    package_temp.cleanup()


def observation(name="H1", value="H1", **changes):
    result = {"desktop_id": "linux:session", "device_id": "linux", "session_id": "session",
        "display_id": "primary", "observation_id": "old", "width": 1920, "height": 1080,
        "foreground": {"handle": "0x12", "process": "calc", "title": "Book"},
        "capabilities": {"platform": "linux"}, "accessibility": {"status": "ok", "truncated": True,
        "controls": [{"name": name, "value": value, "kind": "text", "editable": True, "x": 74, "y": 169,
            "width": 137, "height": 34, "center_x": 142, "center_y": 186,
            "provider_identity": {"provider": "atspi", "bus_name": ":1.143", "owner_pid": 8944,
                "window_path": "/org/a11y/atspi/accessible/438", "object_path": "/org/a11y/atspi/accessible/444"}}]}}
    return result | changes


class IdentityTests(unittest.TestCase):
    def ref(self, obs=None, task="task"):
        return identify_fields(obs or observation(), task)["accessibility"]["controls"][0].get("field_ref")

    def test_mutable_name_value_and_new_observation_keep_same_native_field_reference(self):
        before = observation()
        after = observation("F1", "F1", observation_id="new")
        original = copy.deepcopy(before)
        self.assertEqual(self.ref(before), self.ref(after))
        self.assertEqual(before, original)

    def test_different_task_never_inherits_reference(self):
        self.assertNotEqual(self.ref(), self.ref(task="other"))

    def test_device_session_display_and_foreground_window_each_invalidate_reference(self):
        original = self.ref()
        for key in ("desktop_id", "device_id", "session_id", "display_id"):
            self.assertNotEqual(original, self.ref(observation(**{key: "changed"})))
        for key in ("handle", "process"):
            obs = observation();obs["foreground"][key] = "changed"
            self.assertNotEqual(original, self.ref(obs))

    def test_native_bus_pid_window_object_and_role_each_invalidate_reference(self):
        original = self.ref()
        for key, value in (("bus_name", ":1.144"), ("owner_pid", 8945),
                           ("window_path", "/new_window"), ("object_path", "/new_object")):
            obs = observation();obs["accessibility"]["controls"][0]["provider_identity"][key] = value
            self.assertNotEqual(original, self.ref(obs))
        obs = observation();obs["accessibility"]["controls"][0]["kind"] = "entry"
        self.assertNotEqual(original, self.ref(obs))

    def test_title_or_layout_change_with_same_native_object_does_not_guess_new_identity(self):
        obs = observation();obs["foreground"]["title"] = "Book modified"
        obs["accessibility"]["controls"][0]["x"] = 100
        self.assertEqual(self.ref(), self.ref(obs))

    def test_absent_native_identity_resource_id_or_coordinates_cannot_mint_reference(self):
        obs = observation();control = obs["accessibility"]["controls"][0]
        control.pop("provider_identity");control["resource_id"] = "shared-id"
        self.assertIsNone(self.ref(obs))

    def test_passive_labels_and_readonly_controls_do_not_mint_editable_field_references(self):
        for editable in (False, None):
            obs = observation();obs["accessibility"]["controls"][0]["editable"] = editable
            self.assertIsNone(self.ref(obs))

    def test_unknown_accessibility_or_missing_context_cannot_mint_reference(self):
        obs = observation();obs["accessibility"]["status"] = "foreground_changed"
        self.assertIsNone(self.ref(obs))
        self.assertIsNone(self.ref(observation(session_id="")))

    def test_duplicate_native_object_even_with_different_role_does_not_mint_reference(self):
        for role in ("text", "entry"):
            obs = observation();obs["accessibility"]["controls"].append(obs["accessibility"]["controls"][0] | {"kind": role})
            self.assertTrue(all("field_ref" not in c for c in identify_fields(obs,"task")["accessibility"]["controls"]))

    def test_nonlinux_metadata_never_inherits_atspi_references(self):
        for platform in ("windows", "android", "macos", "ios"):
            self.assertIsNone(self.ref(observation(capabilities={"platform": platform})))

    def test_injected_old_field_reference_is_removed_when_native_identity_missing(self):
        obs = observation();control = obs["accessibility"]["controls"][0]
        control["field_ref"] = self.ref();control.pop("provider_identity")
        self.assertIsNone(self.ref(obs))

    def test_invalid_provider_attributes_decline_without_unbounded_integer_conversion(self):
        attrs = {"qw3-bus-name": ":1.143", "qw3-owner-pid": "8944",
                 "qw3-object-path": "/object", "qw3-window-path": "/window"}
        self.assertIsNotNone(provider_identity_from_attrs(attrs))
        for key, bad in (("qw3-owner-pid", "9"*5000), ("qw3-owner-pid", "0"),
                         ("qw3-bus-name", "well.known.bus"), ("qw3-object-path", "../../other")):
            self.assertIsNone(provider_identity_from_attrs(attrs | {key: bad}))

    def test_old_exact_name_condition_stays_false_but_explicit_native_reference_can_verify(self):
        before = identify_fields(observation(), "task")
        after = identify_fields(observation("F1", "F1"), "task")
        old = {"kind": "control_value", "text": "H1", "value": "F1", "control_kind": "text"}
        new = {"kind": "field_value", "field_ref": self.ref(), "value": "F1"}
        self.assertFalse(execution.check_observation([old], after, before)[0]["passed"])
        self.assertTrue(execution.check_observation([new], after, before)[0]["passed"])

    def test_changed_native_object_session_or_wrong_value_never_verifies_old_reference(self):
        check = {"kind": "field_value", "field_ref": self.ref(), "value": "F1"}
        for obs in (observation("F1","wrong"), observation("F1","F1",session_id="new")):
            self.assertFalse(execution.check_observation([check], identify_fields(obs,"task"), {})[0]["passed"])
        obs = observation("F1", "F1");obs["accessibility"]["controls"][0]["provider_identity"]["object_path"] = "/replacement"
        self.assertFalse(execution.check_observation([check], identify_fields(obs,"task"), {})[0]["passed"])

    def test_empty_literal_value_supported_explicitly(self):
        check = {"kind": "field_value", "field_ref": self.ref(), "value": ""}
        execution.validate_checks([check])
        self.assertTrue(execution.check_observation([check], identify_fields(observation("", ""),"task"), {})[0]["passed"])

    def test_field_condition_rejects_mixed_labels_unknown_properties_and_invented_format(self):
        check = {"kind": "field_value", "field_ref": self.ref(), "value": "F1"}
        for bad in (check | {"text": "H1"}, check | {"field_ref": "field-anywhere"}, check | {"value": None}):
            with self.assertRaises(desktop.DesktopError):execution.validate_checks([bad])

    def test_model_presentation_keeps_reference_and_literal_value_but_hides_provider_internals(self):
        obs = identify_fields(observation(), "task");control = desktop.model_observation(obs)["accessibility"]["controls"][0]
        self.assertEqual(control["value"], "H1");self.assertEqual(control["field_ref"], self.ref())
        self.assertNotIn("provider_identity", control)
        self.assertIn("provider_identity", obs["accessibility"]["controls"][0])

    def test_source_tree_extension_retains_untrusted_title_as_data(self):
        title = "Save As';raise RuntimeError('injected')#"
        source = linux.foreground_tree_script(title)
        compile(source,"internal-reader","exec")
        self.assertIn(repr(title), source)
        self.assertIn("identity_frame.get_process_id()", source)
        with self.assertRaises(ValueError):atspi_identity_script("unsupported")

    def test_xml_provider_identity_survives_visible_foreground_parser(self):
        xml='''<file-chooser name="Save As"><text name="H1" visible="true" showing="true" enabled="true"
          screencoord="(74,169)" size="(137,34)" qw3-bus-name=":1.143" qw3-owner-pid="8944"
          qw3-object-path="/object" qw3-window-path="/window">H1</text></file-chooser>'''
        result = linux.controls_from_xml(xml,{"title":"Save As"},1920,1080)
        self.assertEqual(result["controls"][0]["provider_identity"]["object_path"], "/object")
        self.assertEqual(result["controls"][0]["value"], "H1")

    def test_public_schema_retains_typed_field_ref_through_actual_converter(self):
        from jsonschema import validate
        from qw3_field_test.protocol import convert_request
        tools = schema.tools_for_platform("linux", workflow=True)
        act = next(t for t in tools if t["name"]=="act")
        args = {"action_id":"edit-field","desktop_id":"device","observation_id":"current",
                "action":{"kind":"type","text":"F1"},"step":{"goal":"Edit field",
                "conditions":[{"kind":"field_value","field_ref":self.ref(),"value":"F1"}]}}
        validate(args,act["inputSchema"])
        converted = convert_request({"stream":True,"input":"Validate the fixture schema","tools":[{"type":"function","name":"mcp__computer__act",
            "parameters":act["inputSchema"]}]},"qw3").body
        properties = converted["tools"][0]["function"]["parameters"]["properties"]["step"]["properties"]["conditions"]["items"]["properties"]
        self.assertEqual(properties["field_ref"]["type"], "string")
        self.assertIn("field_value", properties["kind"]["enum"])


class LiveBoundaryFixtures(unittest.IsolatedAsyncioTestCase):
    observation = fixtures.DesktopTests.observation
    action = fixtures.DesktopTests.action

    async def asyncSetUp(self):
        with patch("test_desktop.Desktop", desktop.Desktop):await fixtures.DesktopTests.asyncSetUp(self)
        obs=observation();self.controls=obs["accessibility"]["controls"]
        self.observation_extra={k:v for k,v in obs.items() if k not in {"accessibility","observation_id","width","height","desktop_id"}}
        self.controls[0].update(x=0,y=0,width=4,height=4,center_x=2,center_y=2)
        await self.desktop.observe()
        self.execution=execution.Execution(self.desktop,Path(self.temp.name))
        self.desktop.execution=self.execution

    async def asyncTearDown(self):
        self.execution.close()
        await fixtures.DesktopTests.asyncTearDown(self)

    async def test_real_http_sqlite_delivery_verifies_changed_name_with_same_object_once(self):
        reference=self.desktop.latest["accessibility"]["controls"][0]["field_ref"]
        self.controls[0].update(name="F1",value="F1")
        request=self.action(step={"goal":"Edit the field","conditions":[{"kind":"field_value","field_ref":reference,"value":"F1"}]})
        first=await self.desktop.act(request)
        self.assertTrue(first["step_check"]["verified"])
        self.assertTrue((await self.desktop.act(request))["duplicate"])
        self.assertEqual(self.effects,1)
        self.assertEqual(self.execution.steps()[0]["status"],"verified")

    async def test_delivered_action_with_replaced_native_object_keeps_step_pending(self):
        reference=self.desktop.latest["accessibility"]["controls"][0]["field_ref"]
        self.controls[0].update(name="F1",value="F1")
        self.controls[0]["provider_identity"]["object_path"]="/replacement"
        result=await self.desktop.act(self.action(step={"goal":"Edit the field","conditions":[{"kind":"field_value","field_ref":reference,"value":"F1"}]}))
        self.assertFalse(result["step_check"]["verified"])
        self.assertEqual(self.effects,1)


class IdleConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_closed_reused_post_connection_fails_baseline_but_new_connector_sends_each_read_once(self):
        # Deterministic peer reproduces a server closing a reused connection
        # after accepting the next POST. This must never trigger POST replay.
        seed = importlib.util.module_from_spec(importlib.util.spec_from_loader("qw3_field_test._seed_linux", loader=None))
        seed.__package__ = "qw3_field_test"
        exec(compile(SEED["linux_driver.py"], "seed-linux-driver", "exec"), seed.__dict__)

        async def exercise(driver_type):
            stats = {"connections": 0, "posts": 0, "closed_reused_posts": 0}
            writers = set()

            async def peer(reader, writer):
                stats["connections"] += 1
                writers.add(writer)
                seen = 0
                try:
                    while True:
                        header = await reader.readuntil(b"\r\n\r\n")
                        length = next(int(line.split(b":",1)[1]) for line in header.split(b"\r\n") if line.lower().startswith(b"content-length:"))
                        body = json.loads(await reader.readexactly(length))
                        self.assertEqual(body["command"][:2], ["python", "-c"])
                        stats["posts"] += 1; seen += 1
                        if seen > 1:
                            stats["closed_reused_posts"] += 1
                            break
                        result = json.dumps({"returncode":0,"output":json.dumps({"handle":"0x12","process":"editor","title":"Note"})}).encode()
                        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(result)).encode() + b"\r\nConnection: keep-alive\r\n\r\n" + result)
                        await writer.drain()
                except asyncio.IncompleteReadError:
                    pass
                finally:
                    writer.close();await writer.wait_closed();writers.discard(writer)

            server = await asyncio.start_server(peer, "127.0.0.1", 0)
            endpoint = "http://127.0.0.1:" + str(server.sockets[0].getsockname()[1])
            driver = driver_type(endpoint,"test-device")
            await driver.start()
            failure = None
            try:
                self.assertEqual((await driver.foreground())["title"], "Note")
                try: await driver.foreground()
                except aiohttp.ClientError as exc: failure = type(exc).__name__
            finally:
                await driver.close();server.close();await server.wait_closed()
                for writer in tuple(writers):writer.close()
                await asyncio.sleep(0)
            return stats, failure

        old, error = await exercise(seed.LinuxDriver)
        self.assertEqual(old["posts"], 2)
        self.assertEqual(old["closed_reused_posts"], 1)
        self.assertIsNotNone(error)
        new, error = await exercise(linux.LinuxDriver)
        self.assertEqual(new["posts"], 2)
        self.assertEqual(new["connections"], 2)
        self.assertEqual(new["closed_reused_posts"], 0)
        self.assertIsNone(error)


class BuildTests(unittest.TestCase):
    def test_build_only_private_inventory_preserves_source_and_prevents_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/"source";source.mkdir();files={}
            for name in ("desktop.py","linux_driver.py","execution.py","tool_schema.py","__init__.py"):
                data=SEED.get(name,(SOURCE/name).read_bytes());(source/name).write_bytes(data)
                files[name]=hashlib.sha256(data).hexdigest()
            seed=root/"seed.json";seed.write_text(json.dumps({"candidate_files_sha256":files}))
            manifest=build_candidate(source,seed,root/"candidate")
            self.assertEqual(len(manifest["changed_package_files"]),5)
            self.assertFalse(manifest["real_model_retest"])
            for name,digest in files.items():self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(),digest)
            with self.assertRaises(FileExistsError):build_candidate(source,seed,root/"candidate")
            with self.assertRaises(ValueError):build_candidate(source,seed,source/"nested")

    def test_changed_source_is_rejected_before_build(self):
        with self.assertRaises(ValueError):patch_package_file("execution.py",(SOURCE/"execution.py").read_bytes()+b"\n# changed\n")


if __name__ == "__main__":unittest.main()
