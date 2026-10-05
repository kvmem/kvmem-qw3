"""Durable Gateway behavior under native invocation replies and lost replies."""
import ast
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import native_control_execution as native
from qw3_computer.device_gateway import Gateway, DeviceRejected
from qw3_computer.device_protocol import capabilities
from test_native_control_execution import observation, button


class Reply:
    status=200
    def __init__(self,session,payload):self.session=session;self.payload=payload
    async def __aenter__(self):return self
    async def __aexit__(self,*_):return False
    async def json(self):
        tree=ast.parse(self.payload['command'][2])
        assignment=next(node for node in tree.body if isinstance(node,ast.Assign)
                        and any(isinstance(target,ast.Name) and target.id=='_nonce' for target in node.targets))
        nonce=ast.literal_eval(assignment.value)
        if self.session.mode=='lost':raise TimeoutError('Reply lost after invocation')
        if self.session.mode=='legacy-marker':
            return dict(returncode=1,status='error',error='QW3_PRE_DISPATCH_FOCUS_REJECT')
        if self.session.mode=='reject':
            receipt=dict(kind=native.REJECTION_KIND,nonce=nonce,input_sent=False,
                         reason='native_button_or_context_changed_or_unavailable',error_kind='ValueError')
        else:
            receipt=dict(kind=native.INVOCATION_KIND,nonce=nonce,invocation_attempted=True,
                         invocation_accepted=self.session.mode!='false',action_index=0,action_name='click')
        return dict(returncode=0,output=json.dumps(receipt))


class Session:
    def __init__(self,mode):self.mode=mode;self.posts=[]
    def post(self,url,*,json):
        self.posts.append(dict(url=url,payload=json))
        return Reply(self,json)


class Driver:
    device_id='native-control-receipt-test'
    endpoint='http://127.0.0.1:1'
    def __init__(self,mode):
        self.session=Session(mode)
        self.capabilities=capabilities('linux')
        self.capabilities=self.capabilities|{'actions':self.capabilities['actions']+['click_control']}
    async def observe(self):
        value=observation();image=io.BytesIO();Image.new('RGB',(8,8),'white').save(image,format='PNG')
        return dict(png=image.getvalue(),foreground=value['foreground'],accessibility=value['accessibility'])
    async def preflight(self,action,value):
        # This suite qualifies receipt/error handling; real identity/preflight
        # and native traversal are separately exercised in the owned VM.
        self.expected=native.control_dispatch_expectation(observation(),button())
    async def dispatch(self,action):
        status,receipt=await native.execute_native_control(self,self.expected)
        if status=='rejected':raise DeviceRejected('Confirmed before invocation')
        return dict(backend='atspi-native-button',native_invocation=receipt,business_completion_verified=False)
    async def close(self):pass


class NativeReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def stack(self,directory,mode):
        driver=Driver(mode);gateway=Gateway(driver,Path(directory));gateway.paused=False
        value=await gateway.observe()
        request=dict(desktop_id=value['desktop_id'],observation_id=value['observation_id'],action_id='native-action',
                     action=dict(kind='click_control',control_id='ctl_0123456789abcdef'))
        return driver,gateway,request

    async def test_success_is_delivery_only_and_duplicate_never_posts_again(self):
        with tempfile.TemporaryDirectory() as directory:
            driver,gateway,request=await self.stack(directory,'accepted')
            try:
                result=await gateway.act(request)
                self.assertEqual(result['delivery'],'delivered')
                self.assertIsNone(result['verification']['passed'])
                self.assertFalse(result['driver_receipt']['business_completion_verified'])
                self.assertTrue((await gateway.act(request))['duplicate'])
                self.assertEqual(len(driver.session.posts),1)
                self.assertEqual(gateway.receipt(request['action_id'])['status'],'completed')
            finally:await gateway.close()

    async def test_exact_preinvoke_receipt_is_rejected_and_reserved(self):
        with tempfile.TemporaryDirectory() as directory:
            driver,gateway,request=await self.stack(directory,'reject')
            try:
                with self.assertRaises(DeviceRejected):await gateway.act(request)
                self.assertEqual(gateway.receipt(request['action_id'])['status'],'rejected')
                self.assertFalse(gateway.paused)
                with self.assertRaises(DeviceRejected):await gateway.act(request)
                self.assertEqual(len(driver.session.posts),1)
            finally:await gateway.close()

    async def test_false_or_lost_reply_pauses_and_never_replays(self):
        for mode in ('false','lost','legacy-marker'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                driver,gateway,request=await self.stack(directory,mode)
                try:
                    with self.assertRaises((RuntimeError,TimeoutError)):await gateway.act(request)
                    self.assertEqual(gateway.receipt(request['action_id'])['status'],'uncertain')
                    self.assertTrue(gateway.paused)
                    with self.assertRaises(DeviceRejected):await gateway.act(request)
                    self.assertEqual(len(driver.session.posts),1)
                finally:await gateway.close()


if __name__=='__main__':unittest.main()
