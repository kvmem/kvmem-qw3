"""Safe fault handling must not become a business or ordinary-form pass."""
import json
from pathlib import Path
import tempfile
import unittest

from cross_device_replay import pilot_row


class RecoveryReplayTests(unittest.TestCase):
    def row(self,root,fault=None):
        result={'app_status':'paused','accepted':False,'artifact_passed':False,
                'fault_condition':fault,'fault_response_passed':bool(fault),'fault_accepted':bool(fault),
                'private_fixture_state':{'secret':'must not be exported'}}
        (root/'integration-result.json').write_text(json.dumps(result))
        (root/'pilot-manifest.json').write_text(json.dumps({'case':'linux_browser','video_mode':'vnc'}))
        return pilot_row(root)[1]

    def test_safe_pause_remains_separate_from_normal_form_completion(self):
        with tempfile.TemporaryDirectory() as name:
            row=self.row(Path(name),'no_progress')
            self.assertEqual(row['task'],'linux_browser__no_progress')
            self.assertTrue(row['fault_accepted'])
            self.assertFalse(row['accepted'])
            self.assertFalse(row['artifact_passed'])
            self.assertNotIn('secret',json.dumps(row))

    def test_ordinary_browser_replay_keeps_original_task_identity(self):
        with tempfile.TemporaryDirectory() as name:
            row=self.row(Path(name))
            self.assertEqual(row['task'],'linux_browser')
            self.assertIsNone(row['fault_condition'])

    def test_windows_browser_fault_is_not_labeled_linux(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.row(root,'focus_dialog')
            (root/'pilot-manifest.json').write_text(json.dumps({'case':'windows_browser','video_mode':'vnc'}))
            row=pilot_row(root)[1]
            self.assertEqual(row['platform'],'windows')
            self.assertEqual(row['task'],'windows_browser__focus_dialog')


if __name__=='__main__':unittest.main()
