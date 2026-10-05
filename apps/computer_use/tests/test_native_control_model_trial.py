"""Independent task/profile/wire guards before counting a native model trial."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import native_control_model_trial as t


class TrialEvidenceTests(unittest.TestCase):
    def test_failed_business_grade_is_still_a_terminal_original_trial(self):
        failed=dict(uncertain_actions=0,owned_task_retired=True,app_status='paused',artifact_passed=False,
                    unassisted_passed=False,accepted=False,video_collection_passed=True)
        t.terminal_result(failed)
        self.assertFalse(failed['accepted'])

    def test_unknown_missing_or_unconfirmed_retirement_is_not_terminal(self):
        good=dict(uncertain_actions=0,owned_task_retired=True,app_status='completed',artifact_passed=True,
                  unassisted_passed=True,accepted=True,video_collection_passed=True)
        for mutation in ({'uncertain_actions':1},{'uncertain_actions':False},{'owned_task_retired':False},
                         {'owned_task_stop_unconfirmed':True},{'app_status':'running'},{'accepted':1}):
            with self.subTest(mutation=mutation),self.assertRaises(t.stack.adapter.owned.SafetyHalt):t.terminal_result(good|mutation)

    def test_task_budget_profile_and_permissions_must_be_actual(self):
        good=dict(thinking=True,model_output_tokens=16384,tool_call_limit=100,execution_mode='verified',
                  model_context_window=524288,device_platform='linux',tool_calls_used=17,software_root=None)
        t.task_profile(good)
        for mutation in ({'model_context_window':1048576},{'software_root':'/tmp/work'},
                         {'tool_calls_used':101},{'thinking':False},{'thinking':1},{'execution_mode':'legacy'}):
            with self.subTest(mutation=mutation),self.assertRaises(t.stack.adapter.owned.SafetyHalt):t.task_profile(good|mutation)

    def test_no_model_trace_is_not_a_model_trial(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(t.stack.adapter.owned.SafetyHalt):t.wire_evidence(d)

    def test_wire_permissions_profile_and_terminal_metrics(self):
        good=dict(enable_thinking=True,max_tokens=16384,tools=[{'function':{'name':name}} for name in sorted(t.TOOLS)])
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'1';folder.mkdir();(folder/'metrics.json').write_text('{}')
            for mutation in ({},{'enable_thinking':False},{'max_tokens':4096},
                             {'tools':good['tools']+[{'function':{'name':'shell'}}]}):
                (folder/'request.json').write_text(json.dumps(good|mutation))
                if not mutation:self.assertEqual(len(t.wire_evidence(d)),1)
                else:
                    with self.subTest(mutation=mutation),self.assertRaises(t.stack.adapter.owned.SafetyHalt):t.wire_evidence(d)
            (folder/'metrics.json').unlink();(folder/'request.json').write_text(json.dumps(good))
            with self.assertRaises(t.stack.adapter.owned.SafetyHalt):t.wire_evidence(d)

    def test_actual_wire_roundtrips_existing_lossless_archive(self):
        good=dict(enable_thinking=True,max_tokens=16384,tools=[{'function':{'name':name}} for name in sorted(t.TOOLS)],
                  messages=[{'role':'user','content':'retained screenshot placeholder '*1000}])
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'1';folder.mkdir();(folder/'metrics.json').write_text('{}')
            (folder/'request.json').write_text(json.dumps(good,ensure_ascii=False))
            before=t.wire_evidence(d);self.assertEqual(t.archive.archive_completed(d),1)
            self.assertEqual(t.wire_evidence(d),before)

    def test_unconfirmed_pilot_preserves_vm_even_when_client_has_already_exited(self):
        service=Mock()
        with patch.object(t,'close_stack') as close:
            self.assertFalse(t.cleanup_after_error(service,True,False))
        close.assert_not_called();service.close.assert_not_called()

    def test_post_task_mcp_descendants_are_registered_before_retirement(self):
        service=Mock();identity={'pid':100};old={'pid':101,'command':['codex','app-server']};mcp={'pid':102,'command':['python','mcp_proxy.py']}
        service.record={'services':{'client':{'identity':identity}},'client_descendants':[old]}
        with patch.object(t.stack,'retired',return_value=False),\
             patch.object(t.stack,'descendants',return_value=[old,mcp]) as tree:
            t.close_stack(service)
        tree.assert_called_once_with(identity)
        self.assertEqual(service.save.call_args.kwargs['client_descendants'],[old,mcp]);service.close.assert_called_once()


if __name__=='__main__':unittest.main()
