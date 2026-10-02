import json
from pathlib import Path
import tempfile
import unittest

from recovery_diagnostic_guest import Fixture, page
from recovery_diagnostic_run import grade


class RecoveryDiagnosticTests(unittest.TestCase):
    def test_lost_reply_requires_a_real_completed_write_and_is_injected_once(self):
        with tempfile.TemporaryDirectory() as path:
            fixture = Fixture(path, 'uncertain_delivery', 'x' * 32)
            request = {'action_id': 'save-action-001', 'action': {'kind': 'key', 'key': 'ENTER'}}
            self.assertFalse(fixture.should_withhold(request, {'status': 'completed'}))
            fixture.state['saved_count'] = 1
            self.assertFalse(fixture.should_withhold(request, {'status': 'rejected'}))
            self.assertTrue(fixture.should_withhold(request, {'status': 'completed'}))
            self.assertFalse(fixture.should_withhold(request, {'status': 'completed'}))
            self.assertEqual(json.loads((Path(path) / 'state.json').read_text())['withheld']['action_id'], 'save-action-001')

    def test_saved_but_unreconciled_is_handoff_not_unassisted_success(self):
        state = dict(condition='uncertain_delivery', started=True, events=[], saved_count=1,
                     saved={'name':'Aster','reference':'R-204'}, withheld={'action_id':'saved-action'})
        task = dict(outcome='paused_uncertain', desktop_stopped=True,
                    terminal_observation={'uncertain_actions':[{'action_id':'saved-action'}]})
        result = grade(state, task)
        self.assertTrue(result['artifact_correct'])
        self.assertTrue(result['explicit_uncertainty_handoff'])
        self.assertFalse(result['unassisted_success'])
        state['saved_count'] = 2
        self.assertFalse(grade(state, task)['artifact_correct'])

    def test_delays_start_with_task_instead_of_being_consumed_during_setup(self):
        content = page('normal_slow_loading', '/request').decode()
        self.assertIn("fetch('/phase')", content)
        self.assertIn('12000', content)
        self.assertNotIn('Date.now()-start', content)
        self.assertIn('20000', page('delayed_update', '/request').decode())
        self.assertNotIn('Aster', content)
        self.assertNotIn('R-204', content)
