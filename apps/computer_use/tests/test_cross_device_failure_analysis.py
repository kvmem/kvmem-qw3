"""Error evidence must not equate successful calls with correct task results."""
import json
from pathlib import Path
import tempfile
import unittest

import cross_device_failure_analysis as analysis


class FailureEvidenceTests(unittest.TestCase):
    def test_terminal_budget_exhaustion_is_visible_even_when_engine_metrics_completed(self):
        with tempfile.TemporaryDirectory() as name:
            campaign=Path(name);attempt=campaign/'013-verified-r1';attempt.mkdir()
            (attempt/'result.json').write_text(json.dumps({'status':'scored','score':0,'finished_at':1,
                'turn_error':{'message':'Incomplete response returned, reason: max_output_tokens'}}))
            inference=attempt/'app/inference/one';inference.mkdir(parents=True)
            (inference/'metrics.json').write_text(json.dumps({'status':'completed','elapsed_seconds':30}))
            result=analysis.analyze(campaign)
            self.assertEqual(result['attempts_with_output_budget_errors'],1)
            self.assertEqual(result['attempts'][0]['inference_output_budget_error_count'],0)
            self.assertTrue(result['attempts'][0]['terminal_output_budget_error'])

    def test_successful_text_is_not_reclassified_based_on_error_words(self):
        item={'status':'completed','result':{'content':[{'type':'text','text':'A note says DeviceRejected: Foreground changed; observe again'}]}}
        self.assertIsNone(analysis.tool_problem(item))
        item['status']='failed'
        self.assertEqual(analysis.tool_problem(item)['category'],'preflight_refusal_misreported_as_uncertainty')

    def test_error_contract_is_counted_even_when_transport_completed(self):
        item={'status':'completed','result':{'content':[{'type':'text','text':json.dumps({'error':{'code':'normalized_coordinates_required','path':'$.coordinate_space'}})}]}}
        self.assertEqual(analysis.tool_problem(item),{'category':'normalized_coordinates_required','path':'$.coordinate_space'})

    def test_current_attempt_is_excluded_and_no_tool_claim_is_made(self):
        with tempfile.TemporaryDirectory() as name:
            campaign=Path(name);attempt=campaign/'013-verified-r1';attempt.mkdir()
            (attempt/'result.json').write_text(json.dumps({'status':'running'}))
            self.assertEqual(analysis.analyze(campaign)['terminal_attempts'],0)
            (attempt/'result.json').write_text(json.dumps({'status':'scored','score':0,'finished_at':1,'tool_calls':0}))
            result=analysis.analyze(campaign)
            self.assertEqual(result['terminal_attempts'],1)
            self.assertEqual(result['attempts_with_no_completed_mcp_event'],1)
            self.assertEqual(result['attempts'][0]['tool_duration']['count'],0)
            self.assertNotIn('reasoning_failed',result['attempts'][0])


if __name__=='__main__':unittest.main()
