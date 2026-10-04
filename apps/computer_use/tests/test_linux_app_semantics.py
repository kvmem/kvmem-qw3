"""App behavior is only advertised for a measured version/current setting."""
import unittest
from linux_app_semantics import describe,SemanticLinuxDriver
from qw3_computer.linux_driver import LinuxDriver


class AppSemanticsTests(unittest.TestCase):
    def profile(self,version='gedit - Version 41.0',setting='true'):
        return {'version':{'returncode':0,'stdout':version},
                'ensure_trailing_newline':{'returncode':0,'stdout':setting}}

    def test_measured_enabled_setting_describes_saved_buffer_not_user_answer(self):
        value=describe(self.profile())
        self.assertEqual(value['status'],'qualified')
        self.assertTrue(value['ensure_trailing_newline'])
        self.assertIn('extra blank file line',value['save_behavior'])
        self.assertFalse(value['input_transformed'])
        self.assertFalse(value['preferences_changed'])

    def test_disabled_setting_does_not_advertise_automatic_newline(self):
        value=describe(self.profile(setting='false'))
        self.assertFalse(value['ensure_trailing_newline'])
        self.assertNotIn('Saving appends',value['save_behavior'])

    def test_unknown_version_or_setting_has_no_behavior_claim(self):
        for profile in [self.profile(version='gedit - Version 42.0'),self.profile(setting=''),{},
                        self.profile(setting='true\nUnexpected')]:
            self.assertNotIn('save_behavior',describe(profile))

    def test_adapter_does_not_override_input_or_focus_safety(self):
        self.assertIs(SemanticLinuxDriver.dispatch,LinuxDriver.dispatch)
        self.assertIs(SemanticLinuxDriver.preflight,LinuxDriver.preflight)


if __name__=='__main__':unittest.main()
