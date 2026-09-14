import json
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_formal_trace_campaign import blockers, summarize


class FormalTraceTest(unittest.TestCase):
    def test_queued_work_blocks_even_when_gpu_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '123' / 'cmdline'
            path.parent.mkdir()
            path.write_bytes(b'python\0/home/example/scripts/motivation_v3/run_campaign.py\0--out\0somewhere\0')
            with patch('run_formal_trace_campaign.subprocess.check_output', return_value=''), \
                    patch('run_formal_trace_campaign.Path.glob', return_value=[path]):
                self.assertEqual(blockers()['queued_campaign_pids'], [123])

    def test_shared_or_interfered_times_cannot_enter_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = out / 'run'
            run.mkdir()
            (run / 'result.json').write_text(json.dumps({
                'status': 'completed_timing_excluded', 'performance_claim': False}))
            with self.assertRaisesRegex(ValueError, 'ineligible timing'):
                summarize(out, [run])
            self.assertFalse((out / 'summary.csv').exists())

    def test_early_eos_uses_actual_output_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = out / 'run'
            (run / 'server').mkdir(parents=True)
            (run / 'result.json').write_text(json.dumps({
                'status': 'completed_fixed_input_ttft', 'performance_claim': True,
                'method': 'infllm', 'peak_process_gpu_gib': 58}))
            rows = [{'index': i, 'prompt_tokens': 20000 + i * 100,
                     'api_first_content_s': 1, 'request_wall_s': 3,
                     'output_tokens': n, 'stop_reason': 'end_turn'}
                    for i, n in enumerate([12, 8])]
            phases = [{'prefill_s': 1, 'decode_s': 2, 'output_tokens': n} for n in [12, 8]]
            (run / 'measurements.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (run / 'server/requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in phases))
            summarize(out, [run])
            with (out / 'summary.csv').open() as f:
                summaries = list(csv.DictReader(f))
            self.assertEqual([int(r['output_tokens_sum']) for r in summaries], [20, 8])
            phases[1]['output_tokens'] = 64
            (run / 'server/requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in phases))
            with self.assertRaisesRegex(ValueError, 'output-token count mismatch'):
                summarize(out, [run])


if __name__ == '__main__':
    unittest.main()
