"""Safe fault handling must not become a business or ordinary-form pass."""
import json
from pathlib import Path
import tempfile
import unittest

from cross_device_replay import pilot_row, matched_rows, export


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

    def matched_fixture(self,root):
        trial={'id':'004-verified-r1','task':'004','arm':'verified'}
        (root/'protocol.json').write_text(json.dumps({'trials':[trial]}))
        directory=root/'attempts'/trial['id'];directory.mkdir(parents=True)
        (directory/'result.json').write_text(json.dumps({'status':'scored','score':0,'passed':False,
            'finished_at':123,'seconds':45,'accepted':False,'unassisted_passed':False,
            'private_request':'secret credential must stay private','video_bytes':0}))
        (directory/'manifest.json').write_text(json.dumps({'task_id':'004','arm':'verified',
            'model_output_tokens':16384,'kvmem':False}))
        return directory

    def test_matched_retest_never_changes_original_formal_denominator_or_exports_requests(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);cohort=root/'development';cohort.mkdir();self.matched_fixture(cohort)
            result=export(None,[],root/'public',18143,matched_campaigns=[cohort])
            self.assertEqual(result['formal_summary']['terminal_attempts'],0)
            self.assertEqual(result['development_osworld_attempts'],1)
            self.assertEqual(result['attempts'][0]['original_score'],0)
            self.assertEqual(result['attempts'][0]['model_output_tokens'],16384)
            self.assertNotIn('secret credential',json.dumps(result))

    def test_incorrect_matched_task_material_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);directory=self.matched_fixture(root)
            (directory/'manifest.json').write_text(json.dumps({'task_id':'051','arm':'verified'}))
            with self.assertRaises(ValueError):list(matched_rows(root))

    def test_matched_identity_cannot_read_outside_declared_attempts(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'protocol.json').write_text(json.dumps({'trials':[{'id':'../private'}]}))
            with self.assertRaises(ValueError):list(matched_rows(root))


if __name__=='__main__':unittest.main()
