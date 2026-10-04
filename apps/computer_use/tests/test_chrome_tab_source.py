"""Actual Chrome startup command framing; no browser/VM/input/model."""
import unittest

from qualify_chrome_tab_source import chrome_command_matches_launch


class ChromeStartupCommandTests(unittest.TestCase):
    def setUp(self):
        self.executable='/opt/google/chrome/chrome'
        self.declared=['/usr/bin/google-chrome','--disable-gpu','--user-data-dir=/home/user/.qw3-native-tab-source-owned/profile',
                       '--new-window','file:///home/user/.qw3-native-tab-source-owned/fixture.html']

    def test_actual_retained_single_argv0_contains_the_exact_full_launch(self):
        actual=[' '.join([self.executable,*self.declared[1:]])]
        self.assertTrue(chrome_command_matches_launch(actual,self.declared,self.executable))

    def test_original_and_executable_alias_arg_arrays_are_supported(self):
        for actual in [self.declared,[self.executable,*self.declared[1:]]]:
            self.assertTrue(chrome_command_matches_launch(actual,self.declared,self.executable))

    def test_other_profile_added_flags_or_partial_title_cannot_prove_launch(self):
        actual=' '.join([self.executable,*self.declared[1:]])
        for value in [[actual.replace('/owned/','/unowned/')+' --type=zygote'],
                      [actual.replace('-owned/','-other/')],[actual+' --extra-flag'],[self.executable]]:
            self.assertFalse(chrome_command_matches_launch(value,self.declared,self.executable))

    def test_ambiguous_spaces_or_other_binary_are_not_normalized(self):
        bad=self.declared.copy();bad[-1]='file:///home/user/space path/fixture.html'
        self.assertFalse(chrome_command_matches_launch([' '.join(bad)],bad,self.executable))
        self.assertFalse(chrome_command_matches_launch(self.declared,self.declared,'/bin/sh'))


if __name__=='__main__':unittest.main()
