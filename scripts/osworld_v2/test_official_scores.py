"""Keep native scalar and structured OSWorld evaluations intact in summaries."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from summarize_claude_suite import official_score, write_summary


class OfficialScoreTest(unittest.TestCase):
    def test_legacy_scalar_and_structured_scores(self):
        for score in (0, 0.1111, 0.3853, 1):
            self.assertEqual(official_score(score), score)
            self.assertEqual(official_score({'score': score}), score)

    def test_auxiliary_checks_are_preserved_without_reapplying_penalties(self):
        evaluation = {'score': 0.3853, 'safety': {'total_penalty': 0.2,
            'checks': [{'name': 'example', 'result': 'FAIL', 'penalty': 0.2}]}}
        original = copy.deepcopy(evaluation)
        self.assertEqual(official_score(evaluation), 0.3853)
        self.assertEqual(evaluation, original)

    def test_invalid_scores_are_rejected_without_clamping(self):
        for invalid in (None, True, [], {}, '0.5', {'score': '0.5'}, {'score': True},
                        {'other': 1}, -0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                official_score(invalid)

    def test_summary_accepts_mixed_official_returns_and_retains_partial_credit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'plan.json').write_text(json.dumps({'tasks': ['008']}))
            path = root / 'trials/008/analysis/metrics.json'
            path.parent.mkdir(parents=True)
            metrics = {}
            for arm, score in [('qw3', 1.0), ('kvmem', {'score': 0.3853, 'safety': {'total_penalty': 0.0}})]:
                metrics[arm] = {'result': {'task': '008', 'arm': arm, 'status': 'complete',
                    'steps': 364, 'score': score}, 'native_compactions': 0, 'kvmem_selections': 0,
                    'execution_api_errors': [], 'input_tokens_max': 1}
            path.write_text(json.dumps(metrics))
            original = path.read_bytes()
            summary = write_summary(root)
            self.assertEqual(summary['completed_pairs'], 1)
            self.assertEqual(summary['arms']['qw3']['full_successes'], 1)
            self.assertEqual(summary['arms']['kvmem']['full_successes'], 0)
            self.assertEqual(summary['arms']['kvmem']['mean_official_score'], 0.3853)
            self.assertEqual(path.read_bytes(), original)

    def test_analysis_exposes_numeric_view_without_modifying_official_artifact(self):
        from analyze_claude_pair import analyze
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trial = root / 'full/kvmem'
            trial.mkdir(parents=True)
            evaluation = {'score': 0.3853, 'safety': {'total_penalty': 0.0}}
            original = json.dumps({'task': '008', 'arm': 'kvmem', 'status': 'complete',
                'steps': 0, 'score': evaluation}).encode()
            (trial / 'result.json').write_bytes(original)
            (trial / 'steps.jsonl').write_text(json.dumps({'step': 0}) + '\n')
            with contextlib.redirect_stdout(io.StringIO()):
                analyze(root)
            result = json.loads((root / 'analysis/metrics.json').read_text())['kvmem']['result']
            self.assertEqual(float(result['score']), 0.3853)
            self.assertEqual(result['official_evaluation'], evaluation)
            self.assertEqual((trial / 'result.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
