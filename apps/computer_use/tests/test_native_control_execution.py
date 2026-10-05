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

from PIL import Image

from native_control_evidence import native_button, button_metadata_from_attrs
import native_control_execution as native
from test_native_window_adapter import small_observation


def button():
    return dict(name='Save',kind='push-button',x=2,y=2,width=2,height=2,center_x=3,center_y=3,
                value=None,control_id='ctl_0123456789abcdef',
                provider_identity=dict(provider='atspi',bus_name=':1.2',object_path='/org/a11y/atspi/accessible/7',
                                       window_path='/org/a11y/atspi/accessible/1',owner_pid=9000),
                native_button_action=dict(provider='atspi-native-button-action-v1',actions=['click'],
                                          states=dict(enabled=True,sensitive=True,visible=True,showing=True,defunct=False,busy=False)))


def observation():
    value=small_observation()
    value['accessibility']=dict(status='ok',truncated=False,scope='foreground_window',
                               provider='official-osworld-create_atspi_node',controls=[button()])
    return value


class NativeEvidenceTests(unittest.TestCase):
    def compare(self, *, pixel=(5,5), mutate=None, corrupt=False):
        old=observation();fresh=copy.deepcopy(old)
        fresh['observation_id']='fresh-native-button'
        fresh['accessibility']['controls'][0]['control_id']='ctl_fedcba9876543210'
        if mutate:mutate(fresh)
        with tempfile.TemporaryDirectory() as directory:
            runtime=Path(directory);(runtime/'screenshots').mkdir()
            for index,value in enumerate((old,fresh)):
                image=Image.new('RGB',(8,8),'white')
                if index and pixel:image.putpixel(pixel,(0,0,0))
                data=io.BytesIO();image.save(data,format='PNG');raw=data.getvalue()
                value['sha256']=hashlib.sha256(raw).hexdigest()
                (runtime/'screenshots'/(value['sha256']+'.png')).write_bytes(raw)
            if corrupt:(runtime/'screenshots'/(fresh['sha256']+'.png')).write_bytes(b'bad capture')
            return native.native_control_revalidation_evidence(runtime,old,fresh,old['accessibility']['controls'][0])

    def test_internal_pixels_outside_button_admit_only_native_route(self):
        proof=self.compare()
        self.assertEqual(proof['fresh_control_id'],'ctl_fedcba9876543210')
        self.assertFalse(proof['entire_client_pixels_required'])
        self.assertFalse(proof['input_authorized'])
        self.assertTrue(proof['requires_immediate_native_dispatch_check'])

    def test_target_pixel_mutation_and_corrupt_capture_refuse(self):
        self.assertIsNone(self.compare(pixel=(3,3)))
        self.assertIsNone(self.compare(corrupt=True))

    def test_partial_absent_unknown_or_changed_semantics_cannot_admit(self):
        mutations=[lambda v:v['accessibility'].update(truncated=True),
                   lambda v:v['accessibility'].pop('truncated'),
                   lambda v:v['accessibility'].update(status='unavailable'),
                   lambda v:v['accessibility'].update(scope='desktop'),
                   lambda v:v['accessibility'].update(provider='unqualified-reader'),
                   lambda v:v['accessibility']['controls'][0].update(value='Changed document'),
                   lambda v:v['accessibility']['controls'][0]['provider_identity'].update(object_path='/replacement'),
                   lambda v:v['accessibility']['controls'][0]['native_button_action']['states'].update(enabled=False),
                   lambda v:v['accessibility']['controls'].append(copy.deepcopy(v['accessibility']['controls'][0]))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):self.assertIsNone(self.compare(mutate=mutation))

    def test_window_process_focus_session_and_unbound_bracket_refuse(self):
        mutations=[lambda v:v.update(session_id='new-session'),
                   lambda v:v['native_window']['reported_owner'].update(start_ticks=9),
                   lambda v:v['native_window'].update(focus_window_id=v['native_window']['window_id'],focus_ancestry=[v['native_window']['window_id']]),
                   lambda v:v['native_window_capture'].update(screenshot_bracket_matched=False),
                   lambda v:v.pop('native_window_capture')]
        for mutation in mutations:self.assertIsNone(self.compare(mutate=mutation))

    def test_dispatch_requires_positive_owner_identity_and_states(self):
        value=observation();target=value['accessibility']['controls'][0]
        self.assertIsNotNone(native.control_dispatch_expectation(value,target))
        for mutate in [lambda c:c.update(provider_identity=None),
                       lambda c:c['provider_identity'].update(owner_pid=123),
                       lambda c:c['native_button_action']['states'].update(sensitive=False),
                       lambda c:c['native_button_action']['states'].update(defunct=True),
                       lambda c:c['native_button_action'].update(actions=['click','press']),
                       lambda c:c['native_button_action'].update(actions=['open']),
                       lambda c:c.update(kind='text')]:
            control=copy.deepcopy(target);mutate(control)
            self.assertIsNone(native.control_dispatch_expectation(value,control))


class InvocationTests(unittest.TestCase):
    def script(self, *, states=None, accepted=True, invocation_error=None, resolve_error=None):
        value=observation();expected=native.control_dispatch_expectation(value,value['accessibility']['controls'][0])
        script,nonce=native.native_control_script(expected)
        tree=ast.parse(script)
        names={'read_native_window','resolve_guest_button'}
        self.assertEqual(sum(isinstance(n,ast.FunctionDef) and n.name in names for n in tree.body),2)
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in names]
        calls=[];reads=[]
        class Interface:
            nActions=1
            def getName(self,index):return 'click'
            def doAction(self,index):
                calls.append(index)
                if invocation_error:raise invocation_error
                return accepted
        class Node:
            def queryAction(self):return Interface()
        iterator=iter(copy.deepcopy(states if states is not None else [expected['native'],expected['native']]))
        def resolve(expected,pyatspi):
            reads.append(True)
            if resolve_error:raise resolve_error
            return Node()
        namespace={'read_native_window':lambda:next(iterator),'resolve_guest_button':resolve}
        output=io.StringIO()
        with patch.dict(sys.modules,{'pyatspi':types.ModuleType('pyatspi')}),contextlib.redirect_stdout(output):
            try:exec(compile(tree,'native-action-cpu-fixture','exec'),namespace)
            except BaseException as error:return calls,reads,None,error
        return calls,reads,native.checked_native_control_result(dict(returncode=0,output=output.getvalue()),nonce),None

    def test_action_invoked_once_with_two_native_object_reads(self):
        calls,reads,result,error=self.script()
        self.assertEqual(calls,[0]);self.assertEqual(len(reads),2);self.assertIsNone(error)
        self.assertEqual(result[0],'invoked');self.assertTrue(result[1]['invocation_accepted'])

    def test_false_return_is_invocation_attempt_not_confirmed_non_delivery(self):
        calls,_,result,error=self.script(accepted=False)
        self.assertEqual(calls,[0]);self.assertIsNone(error)
        self.assertEqual(result[0],'invoked');self.assertFalse(result[1]['invocation_accepted'])

    def test_exception_after_invocation_stays_uncertain_without_retry(self):
        calls,_,result,error=self.script(invocation_error=RuntimeError('reply lost'))
        self.assertEqual(calls,[0]);self.assertIsNone(result);self.assertIsInstance(error,RuntimeError)

    def test_preinvoke_focus_fault_or_deleted_object_never_calls_action(self):
        old=observation()['native_window'];new=copy.deepcopy(old);new['focus_within_active']=False
        for kwargs in [dict(states=[new]),dict(states=[old,new]),dict(resolve_error=ValueError('object deleted'))]:
            calls,_,result,error=self.script(**kwargs)
            self.assertEqual(calls,[]);self.assertIsNone(error);self.assertEqual(result[0],'rejected')
            self.assertFalse(result[1]['input_sent'])

    def test_traceback_wrong_nonce_or_unstructured_output_is_uncertain(self):
        receipt=dict(kind=native.REJECTION_KIND,nonce='a'*32,input_sent=False,
                     reason='native_button_or_context_changed_or_unavailable',error_kind='ValueError')
        valid=dict(returncode=0,output=json.dumps(receipt))
        self.assertEqual(native.checked_native_control_result(valid,'a'*32)[0],'rejected')
        for raw in [valid|dict(returncode=1),valid|dict(status='error'),valid|dict(output='Traceback '+json.dumps(receipt)),
                    valid|dict(output=json.dumps(receipt|dict(nonce='b'*32))),valid|dict(output=json.dumps(receipt|dict(input_sent=True)))]:
            with self.assertRaises(RuntimeError):native.checked_native_control_result(raw,'a'*32)


class NativeObjectTraversalTests(unittest.TestCase):
    def tree(self):
        expected=native.control_dispatch_expectation(observation(),button())
        app=types.SimpleNamespace(bus_name=':1.2')
        names=('enabled','sensitive','visible','showing','defunct','busy','active')
        pyatspi=types.SimpleNamespace(**{'STATE_'+name.upper():name for name in names},DESKTOP_COORDS=0)
        class State:
            def __init__(self,flags):self.flags=flags
            def contains(self,name):return self.flags.get(name,False)
        class Interface:
            nActions=1
            def getName(self,index):return 'click'
        class Node:
            def __init__(self,path,name,role,flags):
                self.path=path;self.name=name;self.role=role;self.flags=flags;self.app=app
                self.children=[];self.pid=9000;self.bounds=[2,2,2,2]
            def __iter__(self):return iter(self.children)
            def getRoleName(self):return self.role
            def get_process_id(self):return self.pid
            def getState(self):return State(self.flags)
            def queryAction(self):return Interface()
            def queryComponent(self):return types.SimpleNamespace(getExtents=lambda _:types.SimpleNamespace(**dict(zip(('x','y','width','height'),self.bounds))))
        target=Node(expected['button']['identity']['object_path'],'Save','push button',expected['button']['metadata']['states'].copy())
        frame=Node(expected['button']['identity']['window_path'],expected['native']['title'],'frame',dict(active=True))
        frame.children=[target]
        class Application:
            def __iter__(self):return iter([frame])
        # Bind outside a class body to avoid Python's same-name class scope.
        Application.app=app
        pyatspi.Registry=types.SimpleNamespace(getDesktop=lambda _:[Application()])
        return expected,pyatspi,frame,target,Node

    def test_exact_native_object_is_resolved_without_name_or_coordinate_fallback(self):
        expected,pyatspi,frame,target,_=self.tree()
        self.assertIs(native.resolve_guest_button(expected,pyatspi),target)
        target.path='/org/a11y/atspi/accessible/replacement'
        with self.assertRaises(ValueError):native.resolve_guest_button(expected,pyatspi)

    def test_inactive_frame_changed_pid_state_and_geometry_refuse(self):
        for fault in ('frame','pid','state','geometry'):
            expected,pyatspi,frame,target,_=self.tree()
            if fault=='frame':frame.flags['active']=False
            if fault=='pid':frame.pid=9010
            if fault=='state':target.flags['sensitive']=False
            if fault=='geometry':target.bounds=[3,2,2,2]
            with self.assertRaises(ValueError):native.resolve_guest_button(expected,pyatspi)

    def test_duplicate_object_and_partial_deep_tree_refuse(self):
        expected,pyatspi,frame,target,Node=self.tree();frame.children.append(target)
        with self.assertRaises(ValueError):native.resolve_guest_button(expected,pyatspi)
        expected,pyatspi,frame,target,Node=self.tree()
        branch=frame
        for index in range(18):
            child=Node('/depth_'+str(index),'Container','panel',{});branch.children.append(child);branch=child
        with self.assertRaises(ValueError):native.resolve_guest_button(expected,pyatspi)


if __name__=='__main__':unittest.main()
