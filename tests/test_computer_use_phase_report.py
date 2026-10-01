import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'phase_report', Path(__file__).resolve().parents[1] / 'scripts/computer_use_phase_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

LOG = '''[qw3-serve] multimodal prompt images=1 vision_cache_hits=0 vision_cache_misses=1
[qw3-server-dispatch] rid=1 dispatch=plain speculation_configured=dflash
[qw3] native kvmem query-replay (mtp): checkpoint_ms=4 provisional_ms=1 restore_ms=1 select_ms=2 replay_ms=2
[qw3-native-accounting] trace_tag=example total_ms=20 setup_ms=1 prefill_ms=10 postprefill_ms=2 decode_ms=6 finalize_ms=1
[qw3] native generate: prompt_tokens=100 prefill=0.010s decoded=3 decode=0.006s
[qw3-server-ttft] rid=1 response_ttft_ms=15
[qw3-server-accounting] rid=1 total_ms=25 preprocess_ms=3 queue_ms=1 engine_ms=20 response_ms=1 pre_json_ms=1 pre_validate_ms=1 pre_render_ms=0 pre_tokenize_ms=0 pre_span_setup_ms=1 terminal_status=success
'''


class PhaseReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / 'engine.log'
        self.log.write_text(LOG)

    def metrics(self, **changes):
        path = self.root / 'inference/example'
        path.mkdir(parents=True, exist_ok=True)
        metric = dict(started_at=1, status='completed', backend_trace_tag='example',
                      usage={'input_tokens': 100, 'output_tokens': 3})
        metric.update(changes)
        (path / 'metrics.json').write_text(json.dumps(metric))
        return path.parent

    def test_nested_replay_is_retained_without_double_counting(self):
        requests = report.parse_log(self.log)
        report.correlate(requests, self.metrics())
        self.assertEqual(requests[0]['server']['total_ms'], 25)
        self.assertEqual(requests[0]['native']['prefill_ms'], 10)
        self.assertEqual(requests[0]['query_replay'][0]['select_ms'], 2)
        self.assertTrue(requests[0]['trace_tag_matched'])

    def test_bad_phase_sum_is_rejected(self):
        self.log.write_text(LOG.replace('prefill_ms=10', 'prefill_ms=100'))
        with self.assertRaisesRegex(ValueError, 'Non-additive'):
            report.parse_log(self.log)

    def test_nonfinite_or_negative_timers_are_rejected(self):
        for value in ('NaN', 'Infinity', '-1'):
            with self.subTest(value=value):
                self.log.write_text(LOG.replace('prefill_ms=10', 'prefill_ms=' + value))
                with self.assertRaisesRegex(ValueError, 'Invalid duration'):
                    report.parse_log(self.log)

    def test_nested_duration_cannot_exceed_its_parent(self):
        self.log.write_text(LOG.replace('total_ms=20 setup_ms=1 prefill_ms=10',
                                       'total_ms=30 setup_ms=1 prefill_ms=20'))
        with self.assertRaisesRegex(ValueError, 'exceeds its enclosing'):
            report.parse_log(self.log)

    def test_crossed_request_ids_are_rejected(self):
        self.log.write_text(LOG.replace('rid=1 response_ttft_ms', 'rid=2 response_ttft_ms'))
        with self.assertRaisesRegex(ValueError, 'Unmatched server'):
            report.parse_log(self.log)

    def test_tag_and_token_mismatches_are_rejected(self):
        requests = report.parse_log(self.log)
        for change, error in (({'backend_trace_tag': 'wrong'}, 'trace tags'),
                              ({'usage': {'input_tokens': 101, 'output_tokens': 3}}, 'token counts')):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                report.correlate(requests, self.metrics(**change))

    def test_dense_missing_tag_has_explicit_weaker_attribution(self):
        self.log.write_text(LOG.replace('trace_tag=example', 'trace_tag=-'))
        requests = report.parse_log(self.log)
        report.correlate(requests, self.metrics())
        self.assertFalse(requests[0]['trace_tag_matched'])

    def test_incomplete_request_requires_cancellation_evidence(self):
        self.log.write_text('\n'.join(LOG.splitlines()[:2]) + '\n')
        requests = report.parse_log(self.log)
        self.assertNotIn('native', requests[0])
        with self.assertRaisesRegex(ValueError, 'successful server terminal'):
            report.correlate(requests, self.metrics())
        report.correlate(requests, self.metrics(status='cancelled', usage=None))
        self.assertEqual(requests[0]['bridge_status'], 'cancelled')
        self.assertNotIn('server', requests[0])

    def test_overlapping_requests_are_rejected(self):
        self.log.write_text(LOG.replace('[qw3-native-accounting]',
                                       '[qw3-server-dispatch] rid=2 dispatch=plain\n[qw3-native-accounting]'))
        with self.assertRaisesRegex(ValueError, 'Concurrent or unfinished'):
            report.parse_log(self.log)


if __name__ == '__main__':
    unittest.main()
