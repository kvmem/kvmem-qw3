import unittest

from kvmem_retrieval_diagnostic import EvidenceError, analyze


def request(prompt=181074, budget=131072, gate=True, span=True, success=False):
    rows = ['[qw3-serve] incoming tools=[]']
    if gate:
        rows.append('[qw3-serve] KVMem harness continuation gate harness=generic-tool-client reselect=off headroom_grace=0')
    rows.append(f'[qw3-serve] KVMem request plan stage=draft action=keep-selected-append prompt_tokens={prompt} selection_budget={budget} generation_budget=65536')
    if span:
        rows.append(f'[qw3-serve] kvmem query span [3944,13218) of {prompt} prompt tokens')
    rows.append('[qw3] KVMem request plan stage=final action=pressure-prefill pressure_ingest=1')
    if success:
        rows.append('[qw3] native kvmem query-conditioned: mode=select span=[3944,13218) tokens=9274')
    source = 1416 if prompt > budget else 104
    selected = 1024 if prompt > budget else source
    rows.append(f'[kvmem-scorer] seq=0 position={prompt} requested=mean-k used={"mean-k" if success else "window-profile"} fallback={0 if success else 1} reason={"-" if success else "mean-k:query_span_missing"} query_expected={int(success)} query_captured={int(success)} query_ready={int(success)} indexed_blocks={source if success else 0} source_blocks={source} selected_blocks={selected}')
    return ('\n'.join(rows) + '\n').encode()


class RetrievalDiagnosticTests(unittest.TestCase):
    def test_above_budget_native_gate_and_absent_query(self):
        report = analyze(request())
        self.assertEqual(report['gated_pressure_requests'], 1)
        self.assertEqual(report['above_budget_summary']['window_fallback_events'], 1)
        self.assertEqual(report['above_budget_summary']['identity_window_events'], 0)

    def test_below_budget_identity_fallback_is_not_memory_loss(self):
        report = analyze(request(prompt=13225))
        self.assertEqual(report['gated_pressure_requests'], 0)
        self.assertEqual(report['above_budget_summary']['scorer_events'], 0)
        self.assertEqual(report['summary']['identity_window_events'], 1)

    def test_successful_capture_disproves_gate_explanation(self):
        report = analyze(request(success=True))
        self.assertEqual(report['gated_pressure_requests'], 0)
        self.assertEqual(report['above_budget_summary']['successful_query_events'], 1)
        self.assertEqual(report['requests'][0]['native_query_capture_modes'], ['select'])

    def test_without_gate_evidence_do_not_invent_cause(self):
        self.assertEqual(analyze(request(gate=False))['gated_pressure_requests'], 0)
        self.assertEqual(analyze(request(span=False))['gated_pressure_requests'], 0)

    def test_real_query_metadata_are_row_counts_not_boolean_flags(self):
        data = request(success=True).replace(b'query_expected=1', b'query_expected=256').replace(b'query_captured=1', b'query_captured=256')
        report = analyze(data)
        self.assertEqual(report['summary']['captured_query_events'], 1)
        self.assertEqual(report['summary']['successful_query_events'], 1)

    def test_partial_live_line_is_explicit_and_not_terminal(self):
        data = request() + b'[kvmem-scorer] seq=1'
        with self.assertRaises(EvidenceError):
            analyze(data)
        report = analyze(data, allow_partial=True)
        self.assertTrue(report['incomplete_tail_ignored'])
        self.assertEqual(report['summary']['scorer_events'], 1)
        self.assertFalse(report['business_or_latency_improvement_proven'])

    def test_orphan_or_missing_scorer_metadata_rejected(self):
        event = request().splitlines()[-1] + b'\n'
        with self.assertRaises(EvidenceError):
            analyze(event)
        with self.assertRaises(EvidenceError):
            analyze(request().replace(b'query_ready=0 ', b''))

    def test_duplicate_sequence_or_field_rejected(self):
        data = request()
        with self.assertRaises(EvidenceError):
            analyze(data + data.splitlines()[-1] + b'\n')
        with self.assertRaises(EvidenceError):
            analyze(data.replace(b'query_ready=0', b'query_ready=0 query_ready=1'))

    def test_query_span_must_match_plan(self):
        with self.assertRaises(EvidenceError):
            analyze(request().replace(b'of 181074 prompt', b'of 181075 prompt'))

    def test_new_request_cannot_inherit_previous_gate(self):
        second = request(gate=False).replace(b'seq=0', b'seq=1')
        report = analyze(request() + second)
        self.assertEqual(report['gated_pressure_requests'], 1)
        self.assertIsNone(report['requests'][1]['continuation_gate'])


if __name__ == '__main__':
    unittest.main()
