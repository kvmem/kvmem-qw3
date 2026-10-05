import copy
import unittest

from qualify_native_window_evidence import check_sample, probe_script
from test_native_window_evidence import observation


def sample():
    first = observation()['native_window']
    return dict(first=first, last=copy.deepcopy(first), independent=dict(
        window_id=first['window_id'], focus_window_id=first['focus_window_id'],
        reported_pid=first['reported_owner']['pid'], bounds=first['bounds'],
        bounds_provider='xwininfo-absolute-client-origin', legacy_xdotool_bounds=[104, 92, 952, 799]))


class NativeQualificationTests(unittest.TestCase):
    def test_correct_absolute_geometry_retains_legacy_disagreement(self):
        result = check_sample(sample())
        self.assertTrue(result['passed'])
        self.assertTrue(result['legacy_xdotool_geometry_differs'])
        self.assertFalse(result['input_authorized'])

    def test_independent_focus_and_geometry_mismatch_remain_failures(self):
        for key, value in [('window_id', 999), ('focus_window_id', 999), ('reported_pid', 999),
                           ('bounds', [104, 92, 952, 799]), ('bounds_provider', 'guessed')]:
            value_sample = sample()
            value_sample['independent'][key] = value
            self.assertFalse(check_sample(value_sample)['passed'])

    def test_window_changes_between_sources_are_inconclusive(self):
        value = sample()
        value['last']['reported_owner']['start_ticks'] += 1
        self.assertEqual(check_sample(value)['reason'], 'window_or_focus_changed_between_sources')

    def test_missing_native_properties_do_not_qualify(self):
        for properties in [dict(status='unavailable'), dict(focus_within_active=False), dict(wm_class=[])]:
            value = sample()
            value['first'].update(properties)
            value['last'] = copy.deepcopy(value['first'])
            self.assertFalse(check_sample(value)['passed'])

    def test_probe_is_fixed_python_and_uses_raw_focus_without_window_activation(self):
        script = probe_script()
        compile(script, 'fixed-native-window-cross-check', 'exec')
        self.assertIn("['getwindowfocus','-f']", script)
        self.assertIn("['xwininfo','-id'", script)
        self.assertNotIn('windowactivate', script)
        self.assertNotIn('windowfocus', script.replace('getwindowfocus', 'readonly_focus'))


if __name__ == '__main__':
    unittest.main()
