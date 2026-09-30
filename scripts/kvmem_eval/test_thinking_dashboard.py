"""Regression checks for live metric attribution and append-only log reads."""
import json
from pathlib import Path
import tempfile
import unittest

from thinking_dashboard import Dashboard, StreamView, Tail, rates, request_metrics


class DashboardTests(unittest.TestCase):
    def test_compaction_counts_original_tokens_and_weighted_time(self):
        long = request_metrics({
            "generation": {"decoded": 433, "decode_sec": 684.163},
            "accounting": {"total_ms": 686470.170887},
            "compactions": [{"status": "compacted", "original_tokens": 28696,
                "retained_tokens": 301, "summary_decode_tokens": 299,
                "summary_ms": 17702.472229, "rollback_ms": 1.07622,
                "replay_ms": 190.384187}],
        })
        self.assertEqual(long["original_tokens"], 28828)
        self.assertAlmostEqual(rates([long])["request_tps"], 41.99454, places=4)
        self.assertAlmostEqual(rates([long])["decode_only_tps"], 43.2678, places=4)
        tiny = {"original_tokens": 100, "wall_s": 1, "decode_s": 1}
        self.assertAlmostEqual(rates([tiny, long])["request_tps"], 28928 / 687.470170887)

    def test_rejected_summary_is_not_discarded_output(self):
        row = request_metrics({"generation": {"decoded": 4000, "decode_sec": 100},
            "compactions": [{"status": "rejected", "original_tokens": 3900,
                             "retained_tokens": 100, "summary_ms": 3000}]})
        self.assertEqual(row["original_tokens"], 4000)
        self.assertEqual(row["compactions"], 0)
        self.assertEqual(row["overhead_s"], 3)
        self.assertIsNone(rates([])["decode_tps"])

    def test_partial_json_line_and_rotation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "log"
            p.write_bytes(b'{"a":')
            tail = Tail(p)
            self.assertEqual(tail.poll(), (True, []))
            with p.open("ab") as stream:
                stream.write(b'1}\n')
            self.assertEqual(tail.poll(), (False, ['{"a":1}']))
            self.assertEqual(tail.poll(), (False, []))
            p.rename(p.with_suffix('.old'))
            p.write_text('new\n')
            self.assertEqual(tail.poll(), (True, ['new']))

    def test_streaming_block_reconciles_with_assistant_snapshot(self):
        s = StreamView()
        for event in [
            {"type": "message_start", "message": {"id": "a", "usage": {"input_tokens": 100}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"thinking": "hello"}},
        ]:
            s.line(json.dumps({"event": event}))
        self.assertTrue(s.active)
        self.assertEqual(s.transcript()[0]["text"], "hello")
        full = {"type": "assistant", "message": {"id": "a", "content": [{"type": "thinking", "thinking": "hello"}]}}
        s.line(json.dumps(full))
        s.line(json.dumps(full))
        s.line(json.dumps({"event": {"type": "message_stop"}}))
        self.assertFalse(s.active)
        self.assertEqual(len(s.transcript()), 1)
        self.assertEqual(s.request_count, 1)

    def test_progress_and_selection_do_not_mix_pending_with_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'manifest.json').write_text(json.dumps({
                'tasks': [{'task_id': 'a'}, {'task_id': 'b'}, {'task_id': 'c'}],
                'total_trials': 3, 'schedule': [{'task': x} for x in 'abc']}))
            (root / 'results.json').write_text(json.dumps([
                {'task': 'a', 'valid': True, 'passed': False, 'reward': 0,
                 'final_context_tokens': 116064, 'total_elapsed_sec': 2799.786,
                 'f2p_passed': 77, 'f2p_total': 80, 'p2p_passed': 116, 'p2p_total': 116}]))
            (root / 'state.json').write_text(json.dumps({'task': 'b', 'phase': 'needs_attention'}))
            (root / 'live.json').write_text(json.dumps({'task': 'b', 'elapsed_sec': 300,
                                                      'updated_at': '2026-09-29T08:00:00Z'}))
            dashboard = Dashboard(root)
            d = dashboard.snapshot()
            self.assertEqual(d['completed'], 1)
            self.assertEqual([t['phase'] for t in d['tasks']], ['failed', 'needs_attention', 'queued'])
            a, b, c = d['tasks']
            self.assertEqual(a['final_context_tokens'], 116064)
            self.assertEqual(a['total_elapsed_sec'], 2799.786)
            self.assertEqual(a['f2p_passed'], 77)
            self.assertIsNone(b['final_context_tokens'])
            self.assertIsNone(b['total_elapsed_sec'])
            self.assertEqual(b['elapsed_sec'], 300)  # stopped trials do not keep ticking
            self.assertIsNone(c['elapsed_sec'])
            self.assertFalse(d['current']['active'])
            with self.assertRaises(ValueError):
                dashboard.snapshot('../../etc/passwd')
            with self.assertRaises(ValueError):
                dashboard.within('/etc/passwd')


if __name__ == '__main__':
    unittest.main()
