"""Business checks must reject duplicate writes and unrelated changes."""
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch

from browser_pilot import expected, grade_records
from calc_pilot import grade_bytes, initial_workbook


class BrowserSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.fields = expected('grader-check')

    def record(self, fields=None):
        return json.dumps({'submitted_at': 1, 'values':fields or self.fields}) + '\n'

    def test_no_saved_submission_is_not_completion(self):
        self.assertFalse(grade_records('', self.fields)['passed'])

    def test_correct_single_saved_submission(self):
        self.assertTrue(grade_records(self.record(), self.fields)['passed'])

    def test_repeating_final_save_is_rejected_even_with_identical_data(self):
        self.assertFalse(grade_records(self.record() * 2, self.fields)['passed'])

    def test_correct_visible_fields_with_wrong_delivery_are_rejected(self):
        altered = self.fields | {'delivery':'email'}
        result = grade_records(self.record(altered), self.fields)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['delivery'])

    def test_extra_saved_field_and_malformed_log_cannot_pass(self):
        self.assertFalse(grade_records(self.record(self.fields | {'extra':'wrong'}), self.fields)['passed'])
        self.assertFalse(grade_records('{incomplete', self.fields)['passed'])


class CalcSavedWorkbookTests(unittest.TestCase):
    def setUp(self):
        try:
            from openpyxl import load_workbook
        except ImportError:
            self.skipTest('Use the official OSWorld environment with openpyxl for saved workbook checks')
        self.initial = initial_workbook()
        self.book = load_workbook(io.BytesIO(self.initial))

    def corrected(self):
        self.book['Work']['B3'] = 5
        self.book['Work']['C3'] = 60
        self.book['Summary']['B2'] = '=SUM(Work!D2:D4)'

    def saved(self):
        output = io.BytesIO()
        self.book.save(output)
        return output.getvalue()

    def test_original_or_missing_file_cannot_pass(self):
        self.assertFalse(grade_bytes(self.initial,self.initial)['passed'])
        self.assertFalse(grade_bytes(None,self.initial)['passed'])

    def test_correct_saved_formulas_and_inputs_pass_without_cache_claim(self):
        self.corrected()
        result = grade_bytes(self.saved(), self.initial)
        self.assertTrue(result['passed'])
        self.assertEqual(result['cached_totals'],{'work_total':None,'summary_total':None})

    def test_same_total_but_formula_replaced_by_number_does_not_pass(self):
        self.corrected()
        self.book['Summary']['B2'] = 500
        self.assertFalse(grade_bytes(self.saved(),self.initial)['passed'])

    def test_unrelated_cell_change_does_not_pass(self):
        self.corrected()
        self.book['Summary']['A4'] = 'Changed note'
        result = grade_bytes(self.saved(),self.initial)
        self.assertFalse(result['passed'])
        self.assertFalse(result['checks']['other_cells_and_formulas_preserved'])


class PilotEvidenceTests(unittest.TestCase):
    def test_replaced_container_never_receives_cleanup_signal(self):
        from browser_pilot import stop
        fixture={'platform':'android','container_id':'original-container'}
        with patch('browser_pilot.device_container',return_value='replacement-container'), \
             patch('browser_pilot.subprocess.run') as signal_request:
            with self.assertRaisesRegex(ValueError,'identity changed'):
                stop(fixture)
            signal_request.assert_not_called()

    def test_recording_failure_preserves_verified_business_result_but_not_acceptance(self):
        from platform_pilots import run
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root)/'evidence-check'
            args=SimpleNamespace(output=directory,runtime=Path(root),app='fixture-app',
                                 case='linux_calc',video_mode='live',seconds=60)
            def request(_origin,path,body=None,**_kwargs):
                if path=='/api/task':
                    return {'id':'owned-task','model_context_window':524288}
                return {'id':'owned-task','status':'completed','uncertain_actions':[],
                        'tool_calls_used':3,'model_output_tokens':16384}
            with patch('platform_pilots.request',side_effect=request), \
                 patch('platform_pilots.owned_container',return_value={'Id':'owned-container'}), \
                 patch('platform_pilots.collect_linux_recording',side_effect=TimeoutError('Recorder unavailable')), \
                 patch('calc_pilot.initialize',return_value=({'initial_file':'fixture'},'Edit the workbook')), \
                 patch('calc_pilot.grade',return_value={'passed':True}):
                result=run(args)
            self.assertTrue(result['artifact_passed'])
            self.assertTrue(result['unassisted_passed'])
            self.assertFalse(result['video_collection_passed'])
            self.assertFalse(result['accepted'])
            self.assertTrue(json.loads((directory/'independent-workbook.json').read_text())['passed'])
            self.assertEqual(json.loads((directory/'integration-result.json').read_text()),result)


if __name__ == '__main__':
    unittest.main()
