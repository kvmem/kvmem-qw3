"""Exercise budget termination without a model server or desktop VM."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import analyze_claude_pair as analysis
import run_claude_pair as driver


class FakeClient:
    pid = 999999999

    def __init__(self, code=None):
        self.returncode = code

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = -15
        return self.returncode


class BudgetTerminationTest(unittest.TestCase):
    def run_fixture(self, mode):
        def setup(trial, task):
            trial.task = types.SimpleNamespace(instruction='Synthetic control-flow fixture')
            trial.step = 500 if mode == 'action_limit' else 1
            if mode == 'action_limit':
                trial.limit_delivered_ns = 0

            def evaluate():
                self.assertTrue(trial.done)
                self.assertTrue(trial.lock.locked())
                return 0.0

            trial.env = types.SimpleNamespace(evaluate=evaluate, close=lambda: None)

        client = FakeClient(0 if mode == 'normal' else 2 if mode == 'client_error' else None)
        with tempfile.TemporaryDirectory(prefix='qw3-budget-fixture-') as temp:
            args = ['driver', '--output', temp, '--arm', 'qw3', '--task', '052',
                    '--port', '0', '--timeout', '-1' if mode == 'timeout' else '1000']
            with mock.patch.object(sys, 'argv', args), \
                    mock.patch.object(driver.Trial, 'setup', setup), \
                    mock.patch.object(driver.subprocess, 'check_output', return_value='fixture-version'), \
                    mock.patch.object(driver.subprocess, 'Popen', return_value=client), \
                    mock.patch.object(driver.os, 'killpg') as kill, \
                    contextlib.redirect_stdout(io.StringIO()):
                code = driver.main()
            result = json.loads((Path(temp) / 'result.json').read_text())
            if mode in ('timeout', 'action_limit'):
                kill.assert_called_once_with(client.pid, driver.signal.SIGTERM)
                self.assertEqual(result['termination'], mode)
                self.assertGreater(result['budget_stop_ns'], 0)
            else:
                kill.assert_not_called()
            return code, result

    def test_budget_exhaustion_keeps_official_failure_score(self):
        for mode in ('timeout', 'action_limit'):
            with self.subTest(mode=mode):
                code, result = self.run_fixture(mode)
                self.assertEqual(code, 0)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['score'], 0.0)

    def test_client_failure_remains_an_execution_error(self):
        code, result = self.run_fixture('client_error')
        self.assertEqual(code, 1)
        self.assertEqual(result['status'], 'client_error')

    def test_normal_completion(self):
        code, result = self.run_fixture('normal')
        self.assertEqual(code, 0)
        self.assertEqual(result['status'], 'complete')

    def test_errors_before_budget_stop_are_not_hidden(self):
        result = {'termination': 'timeout', 'steps': 10, 'budget_stop_ns': 1000}
        api = {'after_step': 10, 'finished_ns': 999, 'relay_error': 'connection closed'}
        self.assertFalse(analysis.cancelled_after_budget(api, result))
        api['finished_ns'] = 1001
        self.assertTrue(analysis.cancelled_after_budget(api, result))
        self.assertFalse(analysis.cancelled_after_budget({**api, 'after_step': 9}, result))
        self.assertFalse(analysis.cancelled_after_budget({**api, 'relay_error': None, 'status': 500}, result))
        self.assertFalse(analysis.cancelled_after_budget(api, {'termination': 'timeout', 'steps': 10}))


if __name__ == '__main__':
    unittest.main()
