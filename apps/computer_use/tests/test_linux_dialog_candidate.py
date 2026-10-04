import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest

from linux_dialog_candidate import build_candidate, patch_driver_bytes
from qw3_computer.linux_driver import controls_from_xml as baseline_controls


SOURCE = Path(__file__).resolve().parents[1] / "qw3_computer" / "linux_driver.py"
module = types.ModuleType("qw3_computer._file_chooser_experiment_test")
module.__package__ = "qw3_computer"
exec(compile(patch_driver_bytes(SOURCE.read_bytes()), str(SOURCE), "exec"), module.__dict__)


def window(tag="file-chooser", title="Save As", active="true", button="Save"):
    return f'''<{tag} name="{title}" active="{active}"><push-button name="{button}"
      visible="true" showing="true" enabled="true" screencoord="(10, 20)" size="(30, 10)"/></{tag}>'''


class DialogObservationTests(unittest.TestCase):
    def parse(self, xml):
        return module.controls_from_xml(xml, {"title": "Save As"}, 1920, 1080)

    def test_observed_file_chooser_role_is_excluded_by_baseline_and_supported_by_candidate(self):
        xml=window()
        self.assertEqual(baseline_controls(xml,{"title":"Save As"},1920,1080)["status"],"foreground_unmatched")
        result=self.parse(xml)
        self.assertEqual(result["status"],"ok")
        self.assertEqual([c["name"] for c in result["controls"]],["Save"])

    def test_unmatched_title_does_not_expose_background_chooser(self):
        self.assertEqual(self.parse(window(title="Other Document"))["status"],"foreground_unmatched")

    def test_ambiguous_choosers_are_not_selected_by_position(self):
        self.assertEqual(self.parse('<application>'+window()+window()+'</application>')["status"],"foreground_unmatched")

    def test_unique_active_chooser_disambiguates_exact_title(self):
        xml='<application>'+window(active="false",button="background")+window()+'</application>'
        self.assertEqual([c["name"] for c in self.parse(xml)["controls"]],["Save"])

    def test_frame_and_chooser_title_collision_is_still_ambiguous(self):
        self.assertEqual(self.parse('<application>'+window()+window(tag="frame")+'</application>')["status"],"foreground_unmatched")

    def test_existing_frame_observation_is_identical(self):
        xml=window(tag="frame")
        self.assertEqual(self.parse(xml),baseline_controls(xml,{"title":"Save As"},1920,1080))

    def test_hidden_and_disabled_controls_remain_excluded(self):
        xml=window().replace('enabled="true"','enabled="false"')
        self.assertEqual(self.parse(xml)["controls"],[])

    def test_title_remains_a_literal_and_not_guest_code(self):
        title="Save As'; raise RuntimeError('injected') #"
        source=module.foreground_tree_script(title)
        self.assertIn(repr(title),source)
        compile(source,"bounded-internal-dialog-title-script","exec")


class CandidateBuildTests(unittest.TestCase):
    def test_only_driver_changes_and_output_cannot_overwrite_qualified_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir()
            (source/'linux_driver.py').write_bytes(SOURCE.read_bytes())
            files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
            protocol=root/'protocol.json';protocol.write_text(json.dumps({'framework_sha256':files}))
            manifest=build_candidate(source,protocol,root/'candidate')
            self.assertEqual(manifest['changed_package_files'],['linux_driver.py'])
            self.assertEqual((source/'linux_driver.py').read_bytes(),SOURCE.read_bytes())
            self.assertFalse(manifest['real_model_retest'])
            with self.assertRaises(FileExistsError):build_candidate(source,protocol,root/'candidate')
            with self.assertRaises(ValueError):build_candidate(source,protocol,source/'child')

    def test_changed_source_is_refused(self):
        with self.assertRaisesRegex(ValueError,'exact qualified'):
            patch_driver_bytes(SOURCE.read_bytes()+b'\n# changed\n')


if __name__ == "__main__":
    unittest.main()
