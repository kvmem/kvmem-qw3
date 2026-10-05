"""Offline attribution and privacy checks; no model, guest or grader calls."""
import json
from pathlib import Path
import tempfile
import unittest

import native_pilot_trace_diagnostic as d


class DiagnosticTests(unittest.TestCase):
    def item(self,**values):
        return dict(id='actual-call',tool='act',status='failed',durationMs=7,
                    arguments={'action':{'kind':'type','text':'PRIVATE_ARGUMENT'},'step':{'goal':'PRIVATE_GOAL','conditions':[]}},
                    result={'structuredContent':{'status':'rejected','error':'generic error','problem':{'code':'evidence_refreshed','path':'$.observation_id'}}})|values

    def test_structured_problem_wins_over_generic_error_text(self):
        result=d.tool_summary(self.item());self.assertEqual(result['problem_code'],'evidence_refreshed')

    def test_argument_and_goal_values_are_hashed_not_exported(self):
        output=json.dumps(d.tool_summary(self.item()))
        self.assertNotIn('PRIVATE_ARGUMENT',output);self.assertNotIn('PRIVATE_GOAL',output)
        self.assertIn('argument_sha256',output)

    def test_non_json_workflow_error_is_preserved_and_only_exact_known_literal_classified(self):
        item=self.item(tool='workflow',arguments={'operation':'wait'},result={'structuredContent':None,'content':[{'type':'text','text':'Unknown step_id; inspect workflow status'}]})
        result=d.tool_summary(item)
        self.assertEqual(result['body_parse_error'],'non_json_tool_result');self.assertEqual(result['workflow_error_code'],'unknown_step_id')
        item['result']['content'][0]['text']='PRIVATE_ERROR contains Unknown step_id'
        result=d.tool_summary(item);self.assertIsNone(result['workflow_error_code']);self.assertNotIn('PRIVATE_ERROR',json.dumps(result))

    def test_backend_counts_never_export_reasoning_or_argument_text(self):
        chunk={'choices':[{'finish_reason':'tool_calls','delta':{'reasoning_content':'PRIVATE_REASONING','tool_calls':[{'index':0,'id':'actual-call','function':{'name':'mcp__computer__act','arguments':'{"x":1}'}}]}}]}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'backend.jsonl';path.write_text(json.dumps(chunk)+'\n')
            calls,counts,finishes=d.backend(path)
        self.assertEqual(counts['reasoning_content_characters'],len('PRIVATE_REASONING'))
        self.assertNotIn('PRIVATE_REASONING',json.dumps([counts,finishes]));self.assertEqual(calls[0]['id'],'actual-call')

    def test_same_arguments_from_different_observes_require_exact_actual_call_id(self):
        items=[self.item(id='first',tool='observe',arguments={}),self.item(id='second',tool='observe',arguments={})]
        calls={0:{'id':'second','name':'mcp__computer__observe','arguments':'{}'}}
        result=d.associate(calls,items,{'mcp__computer__observe':{'type':'object'}},set())
        self.assertEqual(result[0]['call_id'],'second')

    def test_argument_mismatch_and_duplicate_association_are_rejected(self):
        call={0:{'id':'actual-call','name':'mcp__computer__act','arguments':'{"x":1}'}}
        item=self.item(arguments={'x':2})
        with self.assertRaises(ValueError):d.associate(call,[item],{'mcp__computer__act':{}},set())
        item['arguments']={'x':1}
        with self.assertRaises(ValueError):d.associate(call,[item],{'mcp__computer__act':{}},{'actual-call'})

    def test_invalid_actual_schema_is_reported_without_repair_or_execution(self):
        call={0:{'id':'actual-call','name':'mcp__computer__act','arguments':'{}'}}
        result=d.associate(call,[self.item(arguments={})],{'mcp__computer__act':{'type':'object','required':['step']}},set())
        self.assertFalse(result[0]['original_schema_valid']);self.assertEqual(result[0]['schema_error_keywords'],['required'])


if __name__=='__main__':unittest.main()
