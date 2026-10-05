import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import native_keyboard_execution as native
import native_window_adapter
from native_keyboard_candidate import build_candidate, patch_file
from test_native_window_adapter import small_observation
import test_desktop as fixtures


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'.runtime/candidate51-retest-20261004/framework'
sys.modules['qw3_computer.native_keyboard_execution']=native
sys.modules['qw3_computer.native_window_adapter']=native_window_adapter
candidate=types.ModuleType('qw3_computer._native_keyboard_candidate_test')
candidate.__package__='qw3_computer'
exec(compile(patch_file('qw3_computer/desktop.py',(SOURCE/'qw3_computer/desktop.py').read_bytes()),
             'isolated-native-keyboard-desktop-test','exec'),candidate.__dict__)
linux_candidate=types.ModuleType('qw3_computer._native_keyboard_driver_test')
linux_candidate.__package__='qw3_computer'
exec(compile(patch_file('qw3_computer/linux_driver.py',(SOURCE/'qw3_computer/linux_driver.py').read_bytes()),
             'isolated-native-keyboard-driver-test','exec'),linux_candidate.__dict__)


class DesktopBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def observation(self):
        return copy.deepcopy(fixtures.DesktopTests.observation(self))
    action=fixtures.DesktopTests.action
    change_pixel=fixtures.DesktopTests.change_pixel
    enable_expired_step=fixtures.DesktopTests.enable_expired_step

    async def asyncSetUp(self):
        with patch('test_desktop.Desktop',candidate.Desktop):
            await fixtures.DesktopTests.asyncSetUp(self)

    async def asyncTearDown(self):
        if hasattr(self,'execution'):self.execution.close()
        await fixtures.DesktopTests.asyncTearDown(self)

    async def prepared(self,kind='key'):
        metadata=small_observation()
        self.controls=[];self.accessibility_status='foreground_unmatched'
        self.observation_extra={key:value for key,value in metadata.items()
                                if key not in {'desktop_id','observation_id','accessibility','width','height'}}
        await self.desktop.observe();await self.enable_expired_step()
        self.change_pixel((0,0))
        return self.action(action=dict(kind='key',key='CTRL+PAGEDOWN') if kind=='key' else dict(kind='type',text='requested content'),
                           step=dict(goal='Perform requested input',conditions=[dict(kind='control_text',text='Saved')]))

    async def refused(self,request):
        with self.assertRaises(candidate.DesktopRejected):await self.desktop.act(request)
        self.assertEqual(self.effects,0);self.assertEqual(self.desktop.unresolved(),[])

    async def test_missing_controls_external_clock_change_sends_key_once_and_does_not_claim_completion(self):
        request=await self.prepared();result=await self.desktop.act(request)
        evidence=result['observation_revalidation']
        self.assertFalse(evidence['same_view']);self.assertFalse(evidence['foreground_context_unchanged'])
        self.assertTrue(evidence['native_keyboard_context_unchanged'])
        self.assertFalse(evidence['native_keyboard_evidence']['accessibility_available'])
        self.assertFalse(result['step_check']['verified'])
        self.assertTrue((await self.desktop.act(request))['duplicate']);self.assertEqual(self.effects,1)

    async def test_paste_uses_same_extra_keyboard_proof(self):
        result=await self.desktop.act(await self.prepared('type'))
        self.assertTrue(result['observation_revalidation']['native_keyboard_context_unchanged'])
        self.assertEqual(self.effects,1)

    async def test_changed_native_child_focus_and_process_generation_refuse(self):
        request=await self.prepared()
        self.observation_extra['native_window']['focus_window_id']=0x30000f8
        self.observation_extra['native_window']['focus_ancestry']=[0x30000f8]
        await self.refused(request)

    async def test_changed_process_generation_refuses(self):
        request=await self.prepared()
        self.observation_extra['native_window']['reported_owner']['start_ticks']=98765
        await self.refused(request)

    async def test_any_client_pixel_change_and_control_semantics_change_refuse(self):
        request=await self.prepared();self.change_pixel((3,3));await self.refused(request)

    async def test_changed_observed_semantic_value_refuses_even_when_client_pixels_match(self):
        request=await self.prepared();self.controls.append(dict(name='Different task state',kind='label'))
        await self.refused(request)

    async def test_no_extra_visual_click_permission(self):
        request=await self.prepared()
        request.update(action=dict(kind='click',x=500,y=500,button='left',clicks=1),coordinate_space='normalized_1000')
        await self.refused(request)

    async def test_no_native_observation_or_session_change_refuses(self):
        request=await self.prepared();self.observation_extra.pop('native_window_capture');await self.refused(request)

    async def test_changed_session_refuses(self):
        request=await self.prepared();self.observation_extra['session_id']='new-session';await self.refused(request)


class CandidateBuildTests(unittest.TestCase):
    def test_real_frozen_inventory_preserves_other_devices_execution_and_runner_sources(self):
        acceptance=SOURCE.parent/'acceptance.json';declared=json.loads(acceptance.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'candidate'
            report=build_candidate(SOURCE,acceptance,output,auxiliary_framework=ROOT)
            files=report['candidate_files_sha256']
            self.assertEqual(files['device_gateway_entry.py'],declared['canonical_entry_sha256'])
            for name in ('qw3_computer/android_driver.py','qw3_computer/device_gateway.py',
                         'qw3_computer/execution.py','qw3_computer/action_validation.py','qw3_computer/codex.py'):
                self.assertEqual(files[name],declared['framework_files'][name])
            for name,digest in report['unchanged_runner_sources_sha256'].items():
                self.assertEqual(digest,declared['framework_files'][name])
                self.assertEqual(hashlib.sha256((output/'source-archive'/name).read_bytes()).hexdigest(),digest)
            self.assertFalse((output/'framework/osworld_linux').exists())
            self.assertFalse(report['real_model_retest']);self.assertFalse(report['frozen_framework_changed'])
            with self.assertRaises(FileExistsError):build_candidate(SOURCE,acceptance,output,auxiliary_framework=ROOT)

    def test_changed_source_and_unsafe_output_refuse(self):
        with self.assertRaises(ValueError):patch_file('qw3_computer/desktop.py',(SOURCE/'qw3_computer/desktop.py').read_bytes()+b'\n')
        with self.assertRaises(ValueError):build_candidate(SOURCE,SOURCE.parent/'acceptance.json',SOURCE/'candidate',auxiliary_framework=ROOT)


class DriverReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def dispatch(self,change_focus=False):
        metadata=small_observation();driver=linux_candidate.LinuxDriver('http://127.0.0.1:9','fixture-device')
        driver.expected_foreground=metadata['foreground']
        driver.expected_native_window=native.dispatch_expectation(metadata)
        actual=copy.deepcopy(metadata['native_window'])
        if change_focus:actual.update(focus_window_id=actual['window_id'],focus_ancestry=[actual['window_id']])
        calls,inputs=[],[]
        pyauto=types.ModuleType('pyautogui');pyauto.hotkey=lambda *keys:inputs.append(keys)
        subprocess=types.ModuleType('subprocess')
        subprocess.check_output=lambda *args,**kwargs:'_NET_ACTIVE_WINDOW(WINDOW): window id # '+metadata['foreground']['handle']

        async def execute(source):
            calls.append(source);tree=ast.parse(source)
            tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='read_native_window']
            stdout=io.StringIO()
            with patch.dict(sys.modules,{'pyautogui':pyauto,'subprocess':subprocess}),contextlib.redirect_stdout(stdout):
                exec(compile(tree,'actual-candidate-driver-cpu-transport','exec'),{'read_native_window':lambda:actual})
            return dict(returncode=0,output=stdout.getvalue())

        driver.execute=execute
        if change_focus:
            with self.assertRaises(linux_candidate.DeviceRejected):await driver.dispatch(dict(kind='key',key='CTRL+PAGEDOWN'))
        else:
            result=await driver.dispatch(dict(kind='key',key='CTRL+PAGEDOWN'))
            self.assertEqual(result,dict(backend='pyautogui',returncode=0))
        self.assertEqual(len(calls),1)
        self.assertEqual(inputs,[] if change_focus else [('ctrl','pagedown')])

    async def test_driver_reports_confirmed_no_input_rejection_without_retry(self):
        await self.dispatch(change_focus=True)

    async def test_driver_keeps_original_input_body_and_sends_one_event(self):
        await self.dispatch()


if __name__=='__main__':
    unittest.main()
