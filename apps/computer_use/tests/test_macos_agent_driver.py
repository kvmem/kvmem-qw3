"""Mac adapter failure-boundary tests using actual HTTP and canonical Gateway."""
import base64
import io
import json
from pathlib import Path
import tempfile
import unittest
import uuid

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from macos_agent_driver import MacDriver
from macos_gateway_entry import create_mac_app
from qw3_computer.device_gateway import Gateway, DeviceRejected


class MacBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.token = 'native-test-token-with-thirty-two-characters'
        (self.root / 'token').write_text(self.token)
        self.native_session = str(uuid.uuid4())
        self.owner = None
        self.paused = True
        self.inputs = 0
        self.controls = 0
        self.mode = 'normal'
        image = io.BytesIO(); Image.new('RGB', (64, 40), 'blue').save(image, format='PNG')
        self.png = base64.b64encode(image.getvalue()).decode()
        native = web.Application()
        native.router.add_route('*', '/{tail:.*}', self.handle)
        self.server = TestServer(native)
        await self.server.start_server()
        self.driver = MacDriver(str(self.server.make_url('')).rstrip('/'), 'mac-test-device',
                                self.root / 'token', self.root / 'native-receipts')
        await self.driver.start()
        self.gateway = Gateway(self.driver, self.root / 'gateway')

    async def asyncTearDown(self):
        await self.gateway.close()
        await self.server.close()
        self.temp.cleanup()

    async def handle(self, request):
        if request.headers.get('Authorization') != 'Bearer ' + self.token:
            return web.json_response({'error': 'Unauthorized'}, status=401)
        body = await request.json() if request.method == 'POST' else {}
        if request.path == '/claim':
            if self.owner and self.owner != body['owner']:
                return web.json_response({'delivery': 'rejected', 'error': 'Owner exists'}, status=409)
            self.owner = body['owner']
        elif request.path == '/release':
            self.owner = None
        elif request.path == '/control':
            self.controls += 1
            if self.mode == 'control-failure':
                return web.json_response({'error': 'Unavailable'}, status=503)
            self.paused = body['command'] != 'resume'
        elif request.path == '/observe':
            return web.json_response({'png': self.png, 'native_capture_id': str(uuid.uuid4()),
                'native_session_id': self.native_session, 'device_paused': self.paused,
                'foreground': {'process': 'com.apple.TextEdit', 'title': 'QW3-test.txt', 'pid': 17},
                'geometry': {'width': 65 if self.mode == 'bad-geometry' else 64, 'height': 40,
                             'point_x': 0, 'point_y': 0, 'point_width': 32, 'point_height': 20},
                'accessibility': {'status': 'ok', 'controls': []}})
        elif request.path == '/preflight':
            if self.mode == 'wrong-focus':
                return web.json_response({'delivery': 'rejected', 'error': 'Foreground changed'}, status=409)
            return web.json_response({'ready': True, 'delivery': 'not_sent'})
        elif request.path == '/act':
            self.inputs += 1
            if self.mode == 'uncertain':
                return web.json_response({'delivery': 'uncertain', 'request_id': body['request_id']})
            return web.json_response({'delivery': 'delivered', 'request_id': body['request_id'], 'status': 'completed'})
        return web.json_response({'protocol': 'qw3-macos-native/1', 'session_id': self.native_session,
                                  'paused': self.paused})

    async def ready(self):
        await self.driver.control('resume')
        self.gateway.paused = False
        await self.gateway.observe()

    def action(self, action_id='mac-action-one'):
        return {'action_id': action_id, 'desktop_id': self.gateway.desktop_id,
                'observation_id': self.gateway.latest['observation_id'],
                'action': {'kind': 'click', 'x': 12, 'y': 8, 'button': 'left', 'clicks': 1}}

    async def test_starts_paused_and_rejects_before_native_input(self):
        await self.gateway.observe()
        with self.assertRaisesRegex(DeviceRejected, 'paused'):
            await self.gateway.act(self.action())
        self.assertEqual(self.inputs, 0)

    async def test_wrong_focus_never_reserves_or_dispatches_input(self):
        await self.ready(); self.mode = 'wrong-focus'
        request = self.action()
        with self.assertRaisesRegex(DeviceRejected, 'Foreground changed'):
            await self.gateway.act(request)
        self.assertEqual(self.inputs, 0)
        self.assertEqual(self.gateway.receipt(request['action_id'])['status'], 'unknown')

    async def test_completed_duplicate_reuses_receipt_without_native_resend(self):
        await self.ready(); request = self.action()
        await self.gateway.act(request)
        result = await self.gateway.act(request)
        self.assertTrue(result['duplicate'])
        self.assertEqual(self.inputs, 1)

    async def test_uncertain_native_result_is_reserved_paused_and_not_retried(self):
        await self.ready(); self.mode = 'uncertain'; request = self.action()
        with self.assertRaisesRegex(RuntimeError, 'uncertain'):
            await self.gateway.act(request)
        self.assertTrue(self.gateway.paused)
        self.assertEqual(self.gateway.receipt(request['action_id'])['status'], 'uncertain')
        with self.assertRaises(DeviceRejected):
            await self.gateway.act(request)
        self.assertEqual(self.inputs, 1)
        records = list((self.root / 'native-receipts').glob('*.json'))
        self.assertEqual(len(records), 1)
        self.assertEqual(json.loads(records[0].read_text())['status'], 'dispatching')

    async def test_real_png_dimensions_must_match_native_geometry(self):
        self.mode = 'bad-geometry'
        with self.assertRaisesRegex(RuntimeError, 'geometry'):
            await self.gateway.observe()
        self.assertIsNone(self.driver.capture)
        self.assertEqual(self.inputs, 0)

    async def test_native_restart_invalidates_capture_without_input(self):
        await self.ready(); self.native_session = str(uuid.uuid4())
        with self.assertRaisesRegex(DeviceRejected, 'session changed'):
            await self.gateway.observe()
        self.assertIsNone(self.driver.capture)
        self.assertEqual(self.inputs, 0)

    async def test_second_native_owner_is_rejected(self):
        another = MacDriver(self.driver.endpoint, 'mac-test-device', self.root / 'token', self.root / 'other')
        with self.assertRaisesRegex(DeviceRejected, 'Owner exists'):
            await another.start()
        await another.close()
        self.assertEqual(self.inputs, 0)

    async def test_remote_resume_failure_leaves_host_paused(self):
        self.mode = 'control-failure'
        client = TestClient(TestServer(create_mac_app(self.gateway, 'gateway-token')))
        await client.start_server()
        try:
            response = await client.post('/control', json={'command': 'resume'},
                                         headers={'Authorization': 'Bearer gateway-token'})
            self.assertEqual(response.status, 503)
            self.assertTrue(self.gateway.paused)
            self.assertEqual(self.inputs, 0)
        finally:
            await client.close()

    async def test_unauthorized_control_never_reaches_native_agent(self):
        client = TestClient(TestServer(create_mac_app(self.gateway, 'gateway-token')))
        await client.start_server()
        try:
            response = await client.post('/control', json={'command': 'resume'})
            self.assertEqual(response.status, 401)
            self.assertEqual(self.controls, 0)
            self.assertTrue(self.gateway.paused)
        finally:
            await client.close()


if __name__ == '__main__':
    unittest.main()
