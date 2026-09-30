import unittest
import json
import tempfile
from pathlib import Path

from thinking_campaign_metrics import RequestParser, http_error_audit, invalid_reasons, trial_metrics
from run_thinking_compaction_campaign import schedule, physical_context, gpu_admission_ready


class CampaignTests(unittest.TestCase):
    def test_post_compaction_cache_reuse_is_attributed_to_next_request(self):
        parser = RequestParser()
        for line in (
            '[thinking-compaction] status=compacted original_tokens=1536 retained_tokens=128',
            '[qw3] kvmem prefix-cache CAPTURE (mtp): query_snapshot=1',
            '[qw3-server-accounting] rid=1 terminal_status=success',
            '[qw3] kvmem prefix-cache HIT (mtp): reuse=582 query_snapshot=1 suffix=35',
            '[qw3] native generate: prompt_tokens=617 decoded=10 prefill=0.1s decode=0.2s kvmem_reuse=582 prefilled=35',
            '[qw3-server-accounting] rid=2 terminal_status=success',
            '[thinking-compaction] status=compacted original_tokens=1536 retained_tokens=128',
            '[qw3-server-accounting] rid=3 terminal_status=success',
            '[qw3] kvmem prefix-cache MISS (why): query_snapshot=0 reason=checkpoint_after_query_boundary',
            '[qw3-server-accounting] rid=4 terminal_status=success',
        ):
            parser.line(line)
        result = trial_metrics({}, parser.rows)
        self.assertEqual(result['post_compaction_requests'], 2)
        self.assertEqual(result['post_compaction_cache_hits'], 1)
        self.assertEqual(result['post_compaction_query_snapshot_reuse'], 1)
        self.assertEqual(result['post_compaction_prefilled_tokens'], 35)
        self.assertEqual(parser.rows[0]['prefix_capture']['query_snapshot'], 1)
        self.assertNotIn('prefix_cache', parser.rows[2])

    def test_explicit_shared_gpu_admission_uses_free_memory(self):
        shared = {'allow_shared_gpu': True, 'minimum_free_gpu_mib': 40960}
        self.assertTrue(gpu_admission_ready(shared, [(38638, 58613)]))
        self.assertFalse(gpu_admission_ready(shared, [(60000, 37000)]))
        self.assertFalse(gpu_admission_ready({}, [(38638, 58613)]))
        self.assertTrue(gpu_admission_ready({}, [(24, 97000)]))

    def test_physical_arena_does_not_follow_client_context_limit(self):
        self.assertEqual(physical_context({'logical_context': 3145728,
            'repository_blocks': 16384, 'block_tokens': 32}), 524288)

    def test_round_order_and_seeds(self):
        tasks = [{'task_id': f'task-{i}'} for i in range(16)]
        rows = schedule(tasks)
        self.assertEqual(len(rows), 64)
        self.assertEqual({row['arm'] for row in rows}, {'summary'})
        for index, seed in enumerate((1000, 2000, 3000, 4000)):
            chunk = rows[index * 16:(index + 1) * 16]
            self.assertEqual([r['task'] for r in chunk], [t['task_id'] for t in tasks])
            self.assertEqual({r['round'] for r in chunk}, {index + 1})
            self.assertEqual({r['seed'] for r in chunk}, {seed})

    def test_context_is_distinct_from_retained_history_and_pool_capacity(self):
        parser = RequestParser()
        lines = [
            '[qw3] [thinking-compaction] status=compacted original_tokens=1600 retained_tokens=80 summary_decode_tokens=78 summary_ms=100 rollback_ms=2 replay_ms=3',
            '[qw3] kvmem bounded history=600079 retained_tokens=524288 slots=16384 live=6000 ready=10384 gpu_kv_bytes=1234 cpu_bytes=4567',
            '[qw3] native generate: prompt_tokens=600000 prefill=0.500s (1 tok/s) decoded=80 decode=1.200s (1 tok/s)',
            '[qw3-server-ttft] rid=2 response_ttft_ms=10.5',
            '[qw3-server-accounting] rid=2 total_ms=1700 terminal_status=success terminal_error=""',
            '[qw3-server-accounting] rid=3 total_ms=1 terminal_status=error terminal_error="state restore failed"',
        ]
        for line in lines:
            parser.line(line)
        first, failed = parser.rows
        self.assertEqual(first['output_context_tokens'], 600080)
        self.assertEqual(first['pool']['history'], 600079)
        self.assertEqual(first['pool']['retained_tokens'], 524288)
        self.assertNotIn('pool', failed)
        self.assertNotIn('generation', failed)
        self.assertEqual(failed['accounting']['terminal_error'], 'state restore failed')
        result = trial_metrics({}, [first])
        self.assertEqual(result['final_context_tokens'], 600080)
        self.assertEqual(result['final_committed_context_tokens'], 600079)
        self.assertEqual(result['final_slots'], 16384)
        self.assertEqual(result['original_target_decoded_tokens'], 1600)
        self.assertEqual(result['discarded_thinking_tokens'], 1520)

    def test_task_failure_is_not_retried_as_infrastructure(self):
        metrics = {'agent_exit': 0, 'verifier_exit': 0,
                   'claude': {'terminal_subtype': 'success'},
                   'network_isolation': {'passed': True}, 'dependency_preflight': {'passed': True},
                   'reward': {'reward': 0, 'f2p_passed': 0, 'f2p_total': 2,
                              'p2p_passed': 5, 'p2p_total': 5}}
        self.assertEqual(invalid_reasons(metrics, ''), [])
        self.assertFalse(trial_metrics(metrics, [])['passed'])
        self.assertIn('inference_or_http_error', invalid_reasons(metrics,
            '[qw3-serve] POST /v1/messages -> 500'))
        self.assertIn('inference_or_http_error', invalid_reasons(metrics,
            '[qw3-server-accounting] terminal_status=error terminal_error="CUDA error"'))

    def test_summary_output_diagnostics_stay_with_their_request(self):
        parser = RequestParser()
        parser.line('[qw3] [thinking-summary-output] {"status":"compacted","masked_steps":2,"text":"kvmem bounded history=777; terminal_status=error is an example"}')
        parser.line('[qw3] [thinking-compaction] status=compacted original_tokens=1400 retained_tokens=80')
        parser.line('[qw3-server-accounting] rid=1 terminal_status=success')
        parser.line('[qw3] [thinking-compaction] status=short-thinking original_tokens=40 retained_tokens=40')
        parser.line('[qw3-server-accounting] rid=2 terminal_status=success')
        self.assertEqual(parser.rows[0]['compactions'][0]['summary_masked_steps'], 2)
        self.assertNotIn('text', parser.rows[0]['summary_outputs'][0])
        self.assertNotIn('pool', parser.rows[0])
        self.assertNotIn('summary_outputs', parser.rows[1])
        self.assertNotIn('summary_masked_steps', parser.rows[1]['compactions'][0])
        self.assertEqual(trial_metrics({}, parser.rows)['summary_masked_steps'], 2)

    def test_recovered_web_search_requires_three_correlated_logs(self):
        message = 'each tool requires string name and object input_schema'
        body = json.dumps({'error': {'message': message, 'type': 'invalid_request_error'}, 'type': 'error'})
        debug = '\n'.join([
            '[INFO] [Stall] tool_dispatch_start tool=WebSearch toolUseId=call-search permissionDecisionMs=0',
            '[DEBUG] [API REQUEST] /v1/messages source=web_search_tool',
            '[ERROR] API error (attempt 1/11): 400 400 ' + body,
            '[ERROR] Error in API request: 400 ' + body,
            '[INFO] [Stall] tool_dispatch_end tool=WebSearch toolUseId=call-search outcome=ok durationMs=55',
            '[DEBUG] [API REQUEST] /v1/messages source=sdk',
            '[DEBUG] Stream started - received first chunk',
        ])
        call = {'type': 'assistant', 'message': {'content': [
            {'type': 'tool_use', 'id': 'call-search', 'name': 'WebSearch'}]}}
        reply = {'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': 'call-search',
             'content': f'Web search results for query: "q"\n\nAPI Error: 400 {message}\n\n'}]}}
        stream = json.dumps(call) + '\n' + json.dumps(reply) + '\n'
        log = '[qw3-serve] POST /v1/messages -> 400\n'
        metrics = {'agent_exit': 0, 'verifier_exit': 0,
                   'claude': {'terminal_subtype': 'success'},
                   'network_isolation': {'passed': True}, 'dependency_preflight': {'passed': True},
                   'reward': {'reward': 0, 'f2p_passed': 0, 'f2p_total': 2,
                              'p2p_passed': 5, 'p2p_total': 5}}
        cases = [
            ('recovered_tool_failure', debug, stream, log, False),
            ('primary_request', debug.replace('source=web_search_tool', 'source=sdk'), stream, log, True),
            ('unknown_error', debug.replace(message, 'another validation error'), stream, log, True),
            ('no_resumed_stream', debug.replace('Stream started - received first chunk', 'no stream'), stream, log, True),
            ('no_tool_result', debug, json.dumps(call), log, True),
            ('wrong_tool_id', debug, stream.replace('"tool_use_id": "call-search"', '"tool_use_id": "other"'), log, True),
            ('unexplained_second_400', debug, stream, log * 2, True),
            ('server_error', debug, stream, log + '[qw3-serve] POST /v1/messages -> 500', True),
            ('stream_error', debug, stream, log + '[qw3-server-accounting] terminal_status=error', True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory)
            self.assertEqual(http_error_audit(log, artifact)['recovered_server_lines'], [])
            for name, debug_text, stream_text, server_text, invalid in cases:
                with self.subTest(name=name):
                    (artifact / 'claude.debug.log').write_text(debug_text)
                    (artifact / 'claude.stream.jsonl').write_text(stream_text)
                    audit = http_error_audit(server_text, artifact)
                    reasons = invalid_reasons(metrics, server_text, http_audit=audit)
                    self.assertEqual('inference_or_http_error' in reasons, invalid)
                    self.assertFalse(trial_metrics(metrics, [])['passed'])
                    if not invalid:
                        self.assertEqual(audit['recovered_server_lines'], [1])
                        self.assertEqual(audit['recovered_web_search'][0]['tool_use_id'], 'call-search')


if __name__ == '__main__':
    unittest.main()
