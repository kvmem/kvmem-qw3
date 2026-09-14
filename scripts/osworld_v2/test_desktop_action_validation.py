"""Exercise desktop argument validation without touching a real desktop."""
import json
from http.server import ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import urllib.request

from claude_desktop_mcp import validate_action
import run_claude_pair as driver


class DesktopActionValidationTest(unittest.TestCase):
    def test_recorded_string_keys_are_rejected(self):
        for value in ('[command, w]', '[esc]', '["ctrl", "s"]', 'ctrl'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'keys must be array'):
                driver.action_code({'action': 'key', 'keys': value})

    def test_invalid_types_and_undeclared_fields_are_rejected(self):
        invalid = [[], {}, {'action': 'triple_click'},
            {'action': 'key', 'keys': [1]}, {'action': 'key', 'keys': []},
            {'action': 'key', 'keys': ['']}, {'action': 'key'},
            {'action': 'left_click', 'coordinate': '[1, 2]'},
            {'action': 'left_click', 'coordinate': [True, 2]},
            {'action': 'left_click', 'coordinate': [float('nan'), 2]},
            {'action': 'left_click', 'coordinate': [1000, 2]},
            {'action': 'scroll', 'pixels': 2.5},
            {'action': 'type', 'text': ['abc']},
            {'action': 'terminate', 'status': 'typo'},
            {'action': 'screenshot', 'unexpected': True}]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                driver.action_code(args)

    def test_valid_commands_are_preserved(self):
        cases = [({'action': 'key', 'keys': ['ctrl', 's']}, "pyautogui.hotkey('ctrl', 's')"),
            ({'action': 'key', 'keys': ['enter']}, "pyautogui.press('enter')"),
            ({'action': 'left_click', 'coordinate': [999, 999]}, 'pyautogui.click(1919, 1079)'),
            ({'action': 'type', 'text': 'first\nsecond'}, "pyautogui.typewrite('first')\npyautogui.press('enter')\npyautogui.typewrite('second')"),
            ({'action': 'scroll', 'pixels': -3}, 'pyautogui.scroll(-3)'),
            ({'action': 'terminate', 'status': 'failure'}, 'FAIL')]
        for args, code in cases:
            with self.subTest(args=args):
                self.assertEqual(driver.action_code(args), code)

    def test_http_returns_tool_error_without_executing_or_counting_action(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = driver.Trial.__new__(driver.Trial)
            trial.directory = Path(temp)
            trial.env = mock.Mock()
            trial.step = 0
            trial.max_steps = 500
            server = ThreadingHTTPServer(('127.0.0.1', 0), driver.make_handler(trial))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                request = urllib.request.Request(
                    f'http://127.0.0.1:{server.server_port}/computer',
                    data=json.dumps({'action': 'key', 'keys': '[command, w]'}).encode(),
                    headers={'Content-Type': 'application/json'})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(request, timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    result = json.load(response)
                self.assertTrue(result['isError'])
                self.assertIn('keys must be array', result['content'][0]['text'])
                trial.env.step.assert_not_called()
                self.assertEqual(trial.step, 0)
                self.assertFalse((trial.directory / 'steps.jsonl').exists())
                self.assertTrue((trial.directory / 'tool_errors.jsonl').exists())
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
