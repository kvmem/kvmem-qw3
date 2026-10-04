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

    def test_string_and_malformed_actions_do_not_hide_terminal_failure_evidence(self):
        with tempfile.TemporaryDirectory() as name:
            attempt=Path(name);(attempt/'app').mkdir()
            (attempt/'result.json').write_text(json.dumps({'status':'scored','score':0,'finished_at':1}))
            arguments=[{'action':{'kind':'click'}},{'action':json.dumps({'kind':'type_text'})},
                       {'action':'not a JSON object'},None,json.dumps({'action':{'kind':'key'}})]
            events=[{'method':'item/completed','params':{'item':{'type':'mcpToolCall','tool':'act',
                     'arguments':value,'status':'failed','error':'Rejected before input'}}} for value in arguments]
            (attempt/'app/codex-events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            row=analysis.inspect_attempt(attempt)
            self.assertEqual(row['act_call_kinds'],{'click':1,'type_text':1,'unknown':2,'key':1})
            self.assertEqual(row['tool_errors_observed'],5)
            self.assertEqual(row['original_score'],0)
            self.assertNotIn('inputs_delivered',row)


if __name__=='__main__':unittest.main()
