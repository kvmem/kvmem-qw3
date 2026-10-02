import unittest

from capability_report import DEFINITIONS, DISPLAY_NAMES, matched_pairs


def trial(mode, run, **changes):
    return dict(task='cap11', prompt_sha256='prompt', snapshot='common', timeout_seconds=1800,
        tool_call_limit=340, visual_history='retrievable', client_bundle_sha256='client',
        engine_binary_sha256='engine', shared_runner_identity_sha256='sources',
        backend_timeout_seconds=1800, backend_read_timeout_seconds=1500,
        valid_trial=True, kvmem=mode, run=run, passed=True, strict_success=True,
        task_seconds=100) | changes


class CapabilityReportTests(unittest.TestCase):
    def test_changed_runtime_or_wait_profile_cannot_be_paired(self):
        first = trial('on', 'on-01')
        self.assertEqual(len(matched_pairs([first, trial('off', 'off-01')])), 1)
        for key in ('client_bundle_sha256', 'engine_binary_sha256', 'shared_runner_identity_sha256',
                    'backend_timeout_seconds', 'backend_read_timeout_seconds', 'snapshot', 'prompt_sha256'):
            with self.subTest(key=key):
                self.assertEqual(matched_pairs([first, trial('off', 'off-02', **{key: 'changed'})]), [])

    def test_repetition_keeps_original_failure_and_excludes_uncertain_attempt(self):
        rows = [trial('on', 'on-01', passed=False, strict_success=False),
                trial('off', 'off-01'), trial('on', 'on-02'), trial('off', 'off-02'),
                trial('on', 'uncertain', valid_trial=False)]
        pairs = matched_pairs(rows)
        self.assertEqual(len(pairs), 2)
        self.assertFalse(pairs[0]['on_passed'])
        self.assertIsNone(pairs[0]['speedup'])
        self.assertEqual(pairs[1]['on'], 'on-02')

    def test_both_larger_tasks_have_report_names_and_public_definitions(self):
        for task in ('cap11', 'cap12'):
            self.assertIn(task, DISPLAY_NAMES)
            self.assertIn(task, DEFINITIONS)

    def test_missing_runtime_identity_remains_unpaired(self):
        for key in ('client_bundle_sha256', 'engine_binary_sha256', 'shared_runner_identity_sha256'):
            rows = [trial('on', 'on-01', **{key: None}), trial('off', 'off-01', **{key: None})]
            with self.subTest(key=key): self.assertEqual(matched_pairs(rows), [])


if __name__ == '__main__':
    unittest.main()
