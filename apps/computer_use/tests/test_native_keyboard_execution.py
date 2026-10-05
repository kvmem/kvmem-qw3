import ast
import contextlib
import copy
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

import native_keyboard_execution as native
from test_native_window_adapter import small_observation


class DispatchGateTests(unittest.TestCase):
    def run_script(self, states, *, kind='key', text='requested text'):
        expected = small_observation()['native_window']
        inputs, clipboard = [], []
        pyauto = types.ModuleType('pyautogui')
        pyauto.hotkey = lambda *keys:inputs.append(keys)
        subprocess = types.ModuleType('subprocess')
        subprocess.DEVNULL = -3
        subprocess.run = lambda *args,**kwargs:clipboard.append((args,kwargs))
        original = 'import pyautogui;pyautogui.FAILSAFE=False;'
        if kind == 'type':
            original += ("import subprocess;subprocess.run(['xclip','-selection','clipboard'],input="
                         + repr(text.encode()) + ",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=3);pyautogui.hotkey('ctrl','v')")
        else:
            original += "pyautogui.hotkey('ctrl','pagedown')"
        script,nonce = native.guarded_input_script(original,expected,kind)
        tree = ast.parse(script)
        # Replace only the fixed native reader for a deterministic CPU fixture.
        # Actual native X11 reads are qualified separately in the owned VM.
        self.assertEqual(sum(isinstance(n,ast.FunctionDef) and n.name=='read_native_window' for n in tree.body),1)
        tree.body = [n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='read_native_window']
        iterator = iter(copy.deepcopy(states))
        output = io.StringIO()
        with patch.dict(sys.modules,{'pyautogui':pyauto,'subprocess':subprocess}),contextlib.redirect_stdout(output):
            exec(compile(tree,'fixed-native-gate-cpu-fixture','exec'),{'read_native_window':lambda:next(iterator)})
        result = dict(returncode=0,output=output.getvalue())
        return inputs,clipboard,native.confirmed_native_rejection(result,nonce)

    def test_stable_generation_and_raw_focus_sends_one_key_event(self):
        value = small_observation()['native_window']
        inputs,clipboard,rejection = self.run_script([value])
        self.assertEqual(inputs,[('ctrl','pagedown')])
        self.assertEqual(clipboard,[])
        self.assertIsNone(rejection)

    def test_same_window_with_changed_child_focus_sends_no_input(self):
        value = small_observation()['native_window']
        value.update(focus_window_id=value['window_id'],focus_ancestry=[value['window_id']])
        inputs,clipboard,rejection = self.run_script([value])
        self.assertEqual(inputs,[])
        self.assertEqual(clipboard,[])
        self.assertFalse(rejection['input_sent'])

    def test_reused_process_changed_geometry_and_missing_native_read_refuse(self):
        for mutation in [lambda v:v['reported_owner'].update(start_ticks=98765),
                         lambda v:v.update(bounds=[1,2,4,4]),lambda v:v.update(status='unavailable'),
                         lambda v:v.update(captured_at=1),lambda v:v.update(focus_within_active=False)]:
            value = small_observation()['native_window'];mutation(value)
            self.assertEqual(self.run_script([value])[0],[])

    def test_focus_change_during_clipboard_preparation_prevents_paste(self):
        first = small_observation()['native_window'];second=copy.deepcopy(first)
        second.update(focus_window_id=first['window_id'],focus_ancestry=[first['window_id']])
        inputs,clipboard,rejection = self.run_script([first,second],kind='type')
        self.assertEqual(inputs,[])
        self.assertEqual(len(clipboard),1)
        self.assertTrue(rejection['clipboard_prepared'])

    def test_native_rejection_precedes_clipboard_mutation(self):
        value=small_observation()['native_window'];value['status']='changed'
        inputs,clipboard,rejection=self.run_script([value],kind='type')
        self.assertEqual(inputs,[]);self.assertEqual(clipboard,[])
        self.assertFalse(rejection['clipboard_prepared'])

    def test_text_is_data_and_successful_paste_occurs_once(self):
        value=small_observation()['native_window']
        text="'; raise AssertionError('not data')\n$(unexpected)"
        inputs,clipboard,rejection=self.run_script([value,value],kind='type',text=text)
        self.assertEqual(inputs,[('ctrl','v')]);self.assertIsNone(rejection)
        self.assertEqual(clipboard[0][1]['input'],text.encode())

    def test_only_exact_successful_nonce_receipt_proves_non_delivery(self):
        nonce='a'*32
        output=dict(kind=native.REJECTION_KIND,nonce=nonce,input_sent=False,clipboard_prepared=False,
                    reason='native_window_process_or_focus_changed_or_unavailable')
        result=dict(returncode=0,output=json.dumps(output))
        self.assertIsNotNone(native.confirmed_native_rejection(result,nonce))
        for changed in [result|dict(returncode=1),result|dict(status='error'),
                        result|dict(output='Traceback: '+json.dumps(output)),
                        result|dict(output=json.dumps(output|dict(input_sent=True))),
                        result|dict(output=json.dumps(output|dict(nonce='b'*32)))]:
            self.assertIsNone(native.confirmed_native_rejection(changed,nonce))

    def test_missing_native_evidence_preserves_original_script(self):
        original="import pyautogui;pyautogui.FAILSAFE=False;pyautogui.hotkey('enter')"
        self.assertEqual(native.guarded_input_script(original,None,'key'),(original,None))
        with self.assertRaises(ValueError):native.guarded_input_script('unknown input source',small_observation()['native_window'],'key')

    def test_unbound_capture_and_changed_native_before_after_cannot_create_expectation(self):
        for mutation in [lambda v:v.pop('native_window_capture'),
                         lambda v:v['native_window_capture'].update(screenshot_bracket_matched=False),
                         lambda v:v['native_window_capture']['before']['reported_owner'].update(start_ticks=98765)]:
            value=small_observation();mutation(value)
            self.assertIsNone(native.dispatch_expectation(value))
        value=small_observation();expected=native.dispatch_expectation(value)
        value['native_window']['reported_owner']['pid']=100
        self.assertEqual(expected['reported_owner']['pid'],9000)


if __name__ == '__main__':
    unittest.main()
