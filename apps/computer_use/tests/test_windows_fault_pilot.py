"""Windows recovery evidence and identified scheduler admission stay separate."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fault_browser_pilot import FIELDS,response_grade
from windows_fault_pilot import admitted_child,independent_grade,WindowsRecording
from windows_fault_fixture import identity


class WindowsRecoveryTests(unittest.TestCase):
    def test_model_child_must_match_the_owned_interlude_pid(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'interlude.json'
            path.write_text(json.dumps({'status':'running-separate-pilot','pilot_pid':123}))
            self.assertEqual(admitted_child(path,pid=123)['pilot_pid'],123)
            with self.assertRaises(ValueError):admitted_child(path,pid=124)

    def test_saved_values_without_visible_saved_evidence_do_not_pass(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);directory=root/'windows-fault-final';directory.mkdir()
            (directory/'submissions.jsonl').write_text(json.dumps({'values':FIELDS})+'\n')
            (directory/'fault-state.json').write_text(json.dumps({'condition':'focus_dialog','save_attempts':[{}],
                'saved_count':1,'events':[{'name':'notice_shown'},{'name':'notice_closed'}]}))
            grade=independent_grade({'condition':'focus_dialog'},root)
            self.assertTrue(grade['passed'])
            self.assertFalse(response_grade(grade,{'status':'completed','uncertain_actions':[]})['passed'])

    def test_unknown_input_never_becomes_a_successful_safe_halt(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);directory=root/'windows-fault-final';directory.mkdir()
            (directory/'fault-state.json').write_text(json.dumps({'condition':'no_progress','save_attempts':[{}],
                'saved_count':0,'events':[{'name':'saving_pending_visible'}]}))
            grade=independent_grade({'condition':'no_progress'},root)
            state={'status':'paused','uncertain_actions':[{}],'gui_recovery':{'paused_reason':'unchanged_observation_loop'}}
            self.assertFalse(response_grade(grade,state)['passed'])

    def test_recording_rejects_a_replaced_windows_vm(self):
        recorder=WindowsRecording(Path('/tmp'), 'original-id')
        with patch('windows_fault_pilot.owned_container',return_value={'Id':'other-id','State':{'Running':True}}):
            with self.assertRaises(ValueError):recorder.inspect()

    def test_fixture_names_cannot_escape_the_owned_windows_root(self):
        for value in ('../other-project','bad name','short','x;Stop-Process'):
            with self.assertRaises(ValueError):identity(value)


if __name__=='__main__':unittest.main()
