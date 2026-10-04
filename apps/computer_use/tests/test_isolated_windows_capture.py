"""A hanging reader must not gain input methods or change delivery code."""
from pathlib import Path
import unittest

from isolated_windows_capture import sources


class IsolatedCaptureTests(unittest.TestCase):
    def test_worker_has_no_input_methods_and_parent_preserves_input_owner(self):
        base=Path(__file__).resolve().parents[1].joinpath('vm/executor.ps1').read_text()
        candidate,reader,identity,executable=sources(base)
        for method in ['SendInput','TypeText','mouse_event','SendKeys','Validate-Action']:
            self.assertNotIn(method,reader)
        self.assertIn('[MTAThread]',reader)
        self.assertIn('worker.Kill()',candidate)
        self.assertIn('worker.WaitForExit(1500)',candidate)
        self.assertIn('ReadToEndAsync()',candidate)
        self.assertEqual(base[:base.index('public sealed class ControlHint')],
                         candidate[:candidate.index('public sealed class ControlHint')])
        def input_code(source):
            return source[source.index('function Validate-Action'):source.index('$listener = [Net.HttpListener]::new()')].split('$script:executorSourceSha256')[0]
        self.assertEqual(input_code(base),input_code(candidate))
        self.assertIn(identity[:16],executable)


if __name__=='__main__':unittest.main()
