import asyncio
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from qw3_computer.device_gateway import Gateway, DeviceRejected
from qw3_computer.device_protocol import capabilities
from qw3_computer.linux_driver import controls_from_xml as linux_controls
from qw3_computer.android_driver import controls_from_xml as android_controls


class FakeDriver:
    def __init__(self):
        self.device_id = 'test-device'
        self.capabilities = capabilities('linux')
        self.effects = 0
        self.mode = 'normal'
        self.block = asyncio.Event()
        self.entered = asyncio.Event()
        buffer = io.BytesIO()
        Image.new('RGB', (32,32), 'blue').save(buffer, format='PNG')
        self.png = buffer.getvalue()

    async def observe(self):
        if self.effects and self.mode == 'invalid_observation':
            raise ValueError('invalid capture')
        result={'png': self.png, 'foreground': {'process':'editor','title':'Note'}}
        if self.effects and self.mode == 'transition':
            self.mode='normal'
            result['accessibility']={'status':'foreground_changed','controls':[]}
        return result

    async def preflight(self, action, observation):
        if self.mode == 'preflight_reject':
            raise DeviceRejected('Wrong focus')

    async def dispatch(self, action):
        self.effects += 1
        self.entered.set()
        if self.mode == 'hang': await self.block.wait()
        return {'delivered': True}

    async def close(self): pass


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.driver = FakeDriver()
        self.gateway = Gateway(self.driver, Path(self.temp.name))
        await self.gateway.observe()
        self.gateway.paused = False

    async def asyncTearDown(self):
        await self.gateway.close()
        self.temp.cleanup()

    def action(self, id='action-one'):
        return {'action_id':id,'desktop_id':self.gateway.desktop_id,
                'observation_id':self.gateway.latest['observation_id'],
                'action':{'kind':'click','x':8,'y':8,'clicks':1,'button':'left'}}

    async def test_duplicate_action_receipt_does_not_send_input_twice(self):
        request = self.action()
        first = await self.gateway.act(request)
        second = await self.gateway.act(request)
        self.assertEqual(self.driver.effects, 1)
        self.assertTrue(second['duplicate'])
        self.assertIsNone(first['verification']['passed'])

    async def test_bad_post_input_capture_keeps_uncertain_not_rejected(self):
        self.driver.mode = 'invalid_observation'
        request = self.action()
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            await self.gateway.act(request)
        self.assertEqual(self.gateway.receipt(request['action_id'])['status'], 'uncertain')
        self.assertTrue(self.gateway.paused)
        with self.assertRaises(DeviceRejected): await self.gateway.act(request)
        self.assertEqual(self.driver.effects, 1)

    async def test_cancellation_reserves_identity_and_restart_invalidates_capture(self):
        self.driver.mode = 'hang'
        request = self.action()
        pending = asyncio.create_task(self.gateway.act(request))
        await self.driver.entered.wait()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError): await pending
        old_id = self.gateway.desktop_id
        await self.gateway.close()
        self.gateway = Gateway(self.driver, Path(self.temp.name))
        self.assertNotEqual(old_id, self.gateway.desktop_id)
        self.assertEqual(self.gateway.receipt(request['action_id'])['status'], 'uncertain')
        with self.assertRaises(DeviceRejected): await self.gateway.act(request)
        self.assertEqual(self.driver.effects, 1)

    async def test_preflight_reject_is_confirmed_non_delivery(self):
        self.driver.mode = 'preflight_reject'
        request = self.action()
        with self.assertRaises(DeviceRejected): await self.gateway.act(request)
        self.assertEqual(self.driver.effects, 0)
        self.assertEqual(self.gateway.receipt(request['action_id'])['status'], 'unknown')

    async def test_foreground_transition_recaptures_once_without_repeating_input(self):
        self.driver.mode='transition'
        result=await self.gateway.act(self.action())
        self.assertEqual(result['post_input_captures'],2)
        self.assertEqual(self.driver.effects,1)

    def test_linux_tree_excludes_background_and_offscreen_controls(self):
        xml='''<desktop xmlns:s="state" xmlns:c="component"><frame name="Foreground"><button name="Save" s:visible="true" s:showing="true" s:enabled="true" c:screencoord="(2, 4)" c:size="(10, 8)"/></frame><frame name="Background"><button name="Wrong" s:visible="true" s:showing="true" s:enabled="true" c:screencoord="(2, 4)" c:size="(10, 8)"/></frame></desktop>'''
        result = linux_controls(xml, {'title':'Foreground'}, 32,32)
        self.assertEqual([c['name'] for c in result['controls']], ['Save'])

    def test_android_tree_excludes_password_values_and_hidden_bounds(self):
        xml='''<hierarchy><node text="Secret" password="true" enabled="true" bounds="[0,0][10,10]"/><android.widget.Button text="Save" enabled="true" displayed="true" bounds="[0,0][10,10]"/><node text="Outside" enabled="true" bounds="[80,80][90,90]"/><android.widget.Button text="Hidden" enabled="true" displayed="false" bounds="[0,0][10,10]"/></hierarchy>'''
        result = android_controls(xml,32,32)
        self.assertEqual([c['name'] for c in result['controls']], ['Save'])

    def test_android_keeps_unlabeled_inputs_but_not_passive_container_ids(self):
        xml='''<hierarchy><node resource-id="layout" enabled="true" clickable="false" bounds="[0,0][30,30]"/><node resource-id="search" enabled="true" focusable="true" bounds="[0,0][20,10]"/><node text="5 minutes" enabled="true" checkable="true" checked="true" bounds="[0,10][20,20]"/></hierarchy>'''
        controls=android_controls(xml,32,32)['controls']
        self.assertEqual([c['name'] for c in controls],['search','5 minutes'])
        self.assertTrue(controls[1]['states']['checked'])
