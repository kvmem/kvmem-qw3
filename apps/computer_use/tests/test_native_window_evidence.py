"""Native evidence must survive source checks without granting desktop input."""
import ast
import copy
import unittest

import native_window_evidence as native


def observation():
    return dict(desktop_id='linux:session', device_id='linux', session_id='session', display_id='primary',
                width=1920, height=1080, rotation=0, capabilities={'platform': 'linux'},
                foreground=dict(handle='0x30000f8', process='Gedit', title='Document - gedit'),
                accessibility=dict(status='foreground_unmatched', controls=[]),
                native_window=dict(provider=native.PROVIDER, status='ok', consistency='two_equal_native_reads',
                    input_sent=False, window_id=0x30000f8, root_id=100, display_name=':0',
                    display_size=[1920, 1080], captured_at=10000, read_milliseconds=50,
                    title='Document - gedit', wm_class=['gedit', 'Gedit'], bounds=[94, 54, 952, 799],
                    same_screen=True, viewable=True, override_redirect=False,
                    reported_owner=dict(pid=9000, uid=1000, start_ticks=12345, executable='/usr/bin/gedit'),
                    focus_window_id=0x30000f9, focus_ancestry=[0x30000f9, 0x30000f8], focus_within_active=True))


class NativeWindowTests(unittest.TestCase):
    def test_missing_accessibility_does_not_hide_positive_native_geometry(self):
        old = observation()
        fresh = copy.deepcopy(old)
        fresh['native_window']['captured_at'] += 100
        proof = native.same_native_window(old, fresh)
        self.assertEqual(proof['bounds'], [94, 54, 1046, 853])
        self.assertFalse(proof['input_authorized'])
        self.assertTrue(proof['pixels_or_semantic_target_not_yet_compared'])

    def test_old_screenshot_without_native_source_has_no_new_proof(self):
        old = observation()
        del old['native_window']
        self.assertIsNone(native.same_native_window(old, observation()))

    def test_session_and_display_changes_invalidate_pair(self):
        for key, value in [('desktop_id', 'other'), ('device_id', 'other'), ('session_id', 'other'),
                           ('display_id', 'other'), ('width', 1280), ('height', 800), ('rotation', 90)]:
            with self.subTest(key=key):
                fresh = observation()
                fresh[key] = value
                self.assertIsNone(native.same_native_window(observation(), fresh))

    def test_reused_pid_and_focus_change_invalidate_pair(self):
        for key, value in [('start_ticks', 67890), ('uid', 1001), ('pid', 9001), ('executable', '/usr/bin/other')]:
            with self.subTest(key=key):
                fresh = observation()
                fresh['native_window']['reported_owner'][key] = value
                self.assertIsNone(native.same_native_window(observation(), fresh))
        fresh = observation()
        fresh['native_window'].update(focus_window_id=0x30000f8, focus_ancestry=[0x30000f8])
        self.assertIsNone(native.same_native_window(observation(), fresh))

    def test_focus_elsewhere_none_root_and_cycles_are_rejected(self):
        for changes in [dict(focus_within_active=False), dict(focus_window_id=1, focus_ancestry=[]),
                        dict(focus_ancestry=[100, 0x30000f8]),
                        dict(focus_ancestry=[0x30000f9, 0x30000f9, 0x30000f8]),
                        dict(focus_window_id=True), dict(focus_ancestry=[0x30000f9])]:
            with self.subTest(changes=changes):
                fresh = observation()
                fresh['native_window'].update(changes)
                self.assertIsNone(native.native_window_context(fresh))

    def test_bad_geometry_and_unmapped_windows_are_rejected(self):
        for changes in [dict(bounds=[-1, 54, 952, 799]), dict(bounds=[94, 54, 0, 799]),
                        dict(bounds=[94, 54, 1920, 799]), dict(bounds=[True, 54, 952, 799]),
                        dict(viewable=False), dict(override_redirect=True), dict(same_screen=False),
                        dict(display_size=[1080, 1920])]:
            with self.subTest(changes=changes):
                fresh = observation()
                fresh['native_window'].update(changes)
                self.assertIsNone(native.native_window_context(fresh))

    def test_invalid_read_status_deadline_and_process_evidence(self):
        for changes in [dict(status='changed'), dict(status='unavailable'), dict(provider='guessed'),
                        dict(read_milliseconds=2000), dict(read_milliseconds=float('nan')),
                        dict(read_milliseconds=True), dict(captured_at=True), dict(input_sent=True),
                        dict(consistency='one_read'), dict(reported_owner={'pid': 9000}),
                        dict(window_id=True), dict(title='Different dialog'), dict(wm_class=['other', 'Other'])]:
            with self.subTest(changes=changes):
                fresh = observation()
                fresh['native_window'].update(changes)
                self.assertIsNone(native.native_window_context(fresh))

    def test_native_window_identity_and_geometry_changes_are_rejected(self):
        fresh = observation()
        fresh['native_window']['bounds'] = [95, 54, 952, 799]
        self.assertIsNone(native.same_native_window(observation(), fresh))
        fresh = observation()
        fresh['native_window']['captured_at'] -= 1
        self.assertIsNone(native.same_native_window(observation(), fresh))
        fresh = observation()
        fresh['foreground']['handle'] = '0xdeadbeef'
        self.assertIsNone(native.same_native_window(observation(), fresh))

    def test_other_platforms_and_missing_context_never_gain_linux_proof(self):
        for value in ['windows', 'android', 'macos']:
            fresh = observation()
            fresh['capabilities']['platform'] = value
            self.assertIsNone(native.native_window_context(fresh))
        fresh = observation()
        del fresh['session_id']
        self.assertIsNone(native.native_window_context(fresh))

    def test_fixed_guest_reader_has_no_input_or_window_mutation_calls(self):
        script = native.guest_script()
        compile(script, 'native-window-guest-reader', 'exec')
        calls = {node.func.attr for node in ast.walk(ast.parse(script))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {'set_input_focus', 'send_event', 'warp_pointer', 'create_window',
                                 'change_property', 'grab_server', 'map', 'configure', 'destroy'})
        self.assertNotIn('pyautogui', script)
        atoms = [node for node in ast.walk(ast.parse(script))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == 'intern_atom']
        self.assertEqual(len(atoms), 3)
        self.assertTrue(all(any(k.arg == 'only_if_exists' and isinstance(k.value, ast.Constant)
                                and k.value.value is True for k in call.keywords) for call in atoms))


if __name__ == '__main__':
    unittest.main()
