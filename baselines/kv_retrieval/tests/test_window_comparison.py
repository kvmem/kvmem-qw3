"""Prevent incomplete trials and partial test credit from becoming pass rates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_agent_windows import METHODS, aggregates, apply_terminal_review, infllm_row, jsonl, native_steps, reward_fields


class ComparisonChecks(unittest.TestCase):
    def test_zero_reward_is_valid_and_partial_score_is_not_pass(self):
        summary={'agent_exit':0,'verifier_exit':0,'reward':{'reward':0,'partial':.995}}
        r=reward_fields(summary,'completed')
        self.assertTrue(r['valid_task_outcome'])
        self.assertFalse(r['fully_passed'])
        self.assertEqual(r['score'],0)
        self.assertEqual(r['partial_test_score'],.995)
        self.assertIsNone(reward_fields(summary,'running_agent')['score'])
        self.assertFalse(reward_fields({**summary,'verifier_exit':2},'completed')['valid_task_outcome'])

    def test_paired_statistics_require_every_method(self):
        rows=[{'task_id':task,'method':method,'valid_task_outcome':True,'fully_passed':True,
               'agent_wall_s':120.,'server_request_wall_s':100.}
              for task in ['a','b'] for method in METHODS]
        rows[-1].update(valid_task_outcome=False,fully_passed=None,agent_wall_s=None)
        result=aggregates(rows,['a','b'])
        self.assertEqual(result['complete_four_method_task_ids'],['a'])
        self.assertEqual(result['methods']['infllm_small']['valid_completed_tasks'],1)
        self.assertEqual(result['methods']['infllm_small']['full_pass_rate_among_valid_completed'],1)
        self.assertEqual(result['methods']['infllm_small']['paired_mean_agent_wall_s'],120.)

    def test_live_partial_line_is_not_a_failed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text('{"status":"completed"}\n{"status":')
            self.assertEqual(jsonl(path),[{'status':'completed'}])
            path.write_text('{"status":\n{"status":"completed"}\n')
            with self.assertRaises(json.JSONDecodeError):jsonl(path)

    def test_aborted_trial_preserves_judgment_and_failed_request_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);(run/'server').mkdir()
            (run/'result.json').write_text(json.dumps({'status':'agent_failed','cell_summary':{
                'agent_exit':1,'verifier_exit':0,'reward':{'reward':0,'partial':.87}}}))
            requests=[{'status':'completed','output_tokens':10,'decode_s':1.,'request_wall_s':2.},
                      {'status':'failed','output_tokens':20,'request_wall_s':7.,'error':'bad tool call'}]
            (run/'server/requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            row,steps=infllm_row('task','small',run)
            self.assertFalse(row['valid_task_outcome'])
            self.assertIsNone(row['score'])
            self.assertEqual(row['raw_verifier_reward'],{'reward':0,'partial':.87})
            self.assertEqual(row['request_count'],1)
            self.assertEqual(row['recorded_request_count'],2)
            self.assertEqual(row['server_request_wall_s'],2.)
            self.assertEqual(row['recorded_request_wall_s'],9.)
            self.assertEqual(row['recorded_output_tokens'],30)
            self.assertEqual(row['terminal_error'],'bad tool call')
            self.assertIsNone(steps[-1]['decode_s'])

    def test_reviewed_execution_failure_requires_matching_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)/'task/small/attempt_01';(run/'server').mkdir(parents=True);(run/'cell').mkdir()
            summary={'agent_exit':1,'verifier_exit':0,'reward':{'reward':0,'partial':.87}}
            (run/'result.json').write_text(json.dumps({'status':'agent_failed','agent_started':1,'cell_summary':summary}))
            (run/'cell/cell_summary.json').write_text(json.dumps(summary))
            (run/'server/requests.jsonl').write_text('')
            (run/'client_events.jsonl').write_text('')
            review=Path(directory)/'review.json'
            evidence=['result.json','cell/cell_summary.json','server/requests.jsonl','client_events.jsonl']
            review.write_text(json.dumps({'task_id':'task','arm':str(run),'end_to_end_execution_outcome':'failed',
                'source_hashes':{p:hashlib.sha256((run/p).read_bytes()).hexdigest() for p in evidence}}))
            row,_=infllm_row('task','small',run);apply_terminal_review(row,review)
            self.assertTrue(row['reviewed_execution_failure'])
            self.assertFalse(row['valid_task_outcome'])
            (run/'client_events.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError,'stale evidence'):apply_terminal_review(row,review)
            row['status']='failed'
            with self.assertRaisesRegex(ValueError,'evaluated agent attempt'):apply_terminal_review(row,review)

    def test_terminal_rate_counts_reviewed_failure_without_imputing_pending_tasks(self):
        methods=['kvmem','qw3_compaction','infllm_small']
        rows=[{'task_id':task,'method':method,'valid_task_outcome':task!='pending',
               'fully_passed':True if task!='pending' else None,'agent_wall_s':120.,'server_request_wall_s':100.}
              for task in ['a','b','pending'] for method in methods]
        failed=next(r for r in rows if r['task_id']=='b' and r['method']=='infllm_small')
        failed.update(valid_task_outcome=False,fully_passed=None,reviewed_execution_failure=True)
        result=aggregates(rows,['a','b','pending'],methods);small=result['methods']['infllm_small']
        self.assertEqual(result['complete_paired_task_ids'],['a'])
        self.assertEqual(result['paired_terminal_task_ids'],['a','b'])
        self.assertEqual(small['terminal_agent_trials'],2)
        self.assertEqual(small['reviewed_execution_failures'],1)
        self.assertEqual(small['agent_and_judge_successes'],1)
        self.assertEqual(small['agent_and_judge_success_rate_among_terminal_trials'],.5)
        self.assertEqual(small['paired_agent_and_judge_success_rate'],.5)
        self.assertEqual(small['paired_mean_agent_wall_s'],120.)

    def test_small_only_scope_does_not_require_cancelled_large_arm(self):
        methods=['kvmem','qw3_compaction','infllm_small']
        rows=[{'task_id':'a','method':method,'valid_task_outcome':True,'fully_passed':False,
               'agent_wall_s':120.,'server_request_wall_s':100.} for method in methods]
        result=aggregates(rows,['a'],methods)
        self.assertEqual(result['complete_paired_task_ids'],['a'])
        self.assertEqual(set(result['methods']),set(methods))
        self.assertNotIn('complete_four_method_count',result)
        rows[-1]['valid_task_outcome']=False
        self.assertEqual(aggregates(rows,['a'],methods)['complete_paired_count'],0)

    def test_native_ttft_uses_logged_clock_and_requires_exact_join(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.log'
            generation='native generate: prompt_tokens=100 prefill=0.500s decoded=7 decode=0.2s\n'
            timing='[qw3-server-ttft] rid=1 server_ttft_ms=650 response_ttft_ms=700 request_total_ms=1000\n'
            path.write_text(generation+timing)
            row=native_steps(path)[0]
            self.assertEqual(row['server_first_content_s'],.7)
            self.assertIsNone(row['is_compaction'])
            path.write_text(generation+generation+timing)
            with self.assertRaisesRegex(ValueError,'ambiguous'):native_steps(path)


if __name__=='__main__':unittest.main()
