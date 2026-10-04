"""The observation candidate must preserve all native input code."""
import unittest
from pathlib import Path

from windows_capture_candidate import generate


class CaptureScopeTests(unittest.TestCase):
    def test_only_capture_and_read_only_health_metadata_change(self):
        base=Path(__file__).resolve().parents[1].joinpath('vm/executor.ps1').read_text()
        candidate=generate(base)
        def input_region(source):
            return source[source.index('function Validate-Action'):source.index('$listener = [Net.HttpListener]::new()')].split('$script:executorSourceSha256')[0]
        self.assertEqual(input_region(base),input_region(candidate))
        self.assertEqual(base[:base.index('public sealed class ControlHint')],
                         candidate[:candidate.index('public sealed class ControlHint')])
        self.assertIn('TryGetCachedPattern(ValuePattern.Pattern',candidate)
        self.assertIn('ControlType.Text',candidate)
        self.assertIn('new PropertyCondition(AutomationElement.IsPasswordProperty, false)',candidate)
        self.assertIn('if (info.IsPassword) continue;',candidate)

    def test_unknown_source_shape_is_rejected_instead_of_guessing_a_patch(self):
        base=Path(__file__).resolve().parents[1].joinpath('vm/executor.ps1').read_text()
        with self.assertRaises(ValueError):generate(base.replace('public bool focused;', 'public bool isFocused;'))


if __name__=='__main__':unittest.main()
