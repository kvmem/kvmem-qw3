"""Mac adapter failure-boundary tests using actual HTTP and canonical Gateway."""
import base64
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import AsyncMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from macos_agent_driver import MacDriver
from macos_gateway_entry import create_mac_app
from qw3_computer.device_gateway import Gateway, DeviceRejected
from qw3_computer.tool_schema import tools_for_platform
from qw3_computer.desktop import Desktop
from qw3_computer.target_execution import TargetExecution


class MacToolDeclarations(unittest.TestCase):
    def test_advertises_only_the_native_adapters_actions(self):
        for workflow in (False, True):
            tools = tools_for_platform('macos', workflow)
            actions = tools[1]['inputSchema']['properties']['action']['oneOf']
            kinds = {action['properties']['kind']['const'] for action in actions}
            self.assertEqual(kinds, {'click_control', 'click', 'type', 'key', 'scroll', 'open_app'})
            launch = next(action for action in actions if action['properties']['kind']['const'] == 'open_app')
            self.assertEqual(launch['properties']['app']['enum'], ['notepad'])
            keys = next(action for action in actions if action['properties']['kind']['const'] == 'key')
            self.assertNotIn('ANDROID_BACK', keys['properties']['key']['enum'])
            self.assertNotIn('ALT+TAB', keys['properties']['key']['enum'])
            if workflow:
                self.assertEqual(len(tools[1]['inputSchema']['oneOf']), len(actions))

    def test_mac_filter_does_not_change_other_device_declarations(self):
        before = {platform: tools_for_platform(platform, True) for platform in ('windows', 'linux', 'android')}
        tools_for_platform('macos', True)
        self.assertEqual(before, {platform: tools_for_platform(platform, True) for platform in before})

    def test_launch_aliases_follow_explicit_observed_capabilities(self):
        tools = tools_for_platform('macos', macos_apps=['terminal', 'installer'])
        actions = tools[1]['inputSchema']['properties']['action']['oneOf']
        launch = next(a for a in actions if a['properties']['kind']['const'] == 'open_app')
        self.assertEqual(launch['properties']['app']['enum'], ['terminal', 'installer'])
        empty = tools_for_platform('macos', macos_apps=[])
        self.assertNotIn('open_app', [a['properties']['kind']['const'] for a in empty[1]['inputSchema']['properties']['action']['oneOf']])


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
        self.allowed_apps = ['com.apple.TextEdit']
        self.deadline = None
        self.native_features = []
        self.foreground_process = 'com.apple.TextEdit'
        self.capture_reads = 0
        self.state_reads = 0
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
            self.capture_reads += 1
            return web.json_response({'png': self.png, 'native_capture_id': str(uuid.uuid4()),
                'native_session_id': self.native_session, 'device_paused': self.paused,
                'foreground': {'process': self.foreground_process, 'title': 'QW3-test.txt', 'pid': 17},
                'geometry': {'width': 65 if self.mode == 'bad-geometry' else 64, 'height': 40,
                             'point_x': 0, 'point_y': 0, 'point_width': 32, 'point_height': 20},
                'accessibility': {'status': 'ok', 'controls': []}})
        elif request.path == '/state':
            self.state_reads += 1
            return web.json_response({'session_id': self.native_session,
                'foreground': {'process': self.foreground_process, 'title': 'QW3-test.txt', 'pid': 17},
                'foreground_in_scope': True, 'paused': self.paused,
                'ax_trusted': self.mode != 'permission-off', 'screen_capture_allowed': True})
        elif request.path == '/preflight':
            if self.mode == 'wrong-focus':
                return web.json_response({'delivery': 'rejected', 'error': 'Foreground changed'}, status=409)
            return web.json_response({'ready': True, 'delivery': 'not_sent'})
        elif request.path == '/act':
            self.inputs += 1
            if body['action']['kind'] == 'open_app':
                self.foreground_process = {'terminal':'com.apple.Terminal','installer':'com.apple.installer','notepad':'com.apple.TextEdit'}[body['action']['app']]
            if self.mode == 'uncertain':
                return web.json_response({'delivery': 'uncertain', 'request_id': body['request_id']})
            return web.json_response({'delivery': 'delivered', 'request_id': body['request_id'], 'status': 'completed'})
        return web.json_response({'protocol': 'qw3-macos-native/1', 'session_id': self.native_session,
                                  'paused': self.paused, 'allowed_apps': self.allowed_apps,
                                  'scope_expires_at': self.deadline, 'timestamp': time.time(),
                                  'native_features': self.native_features,
                                  'ax_trusted': True, 'screen_capture_allowed': True})

    async def enable_native_foundation(self):
        await self.gateway.close()
        self.native_features = ['lightweight_state','native_app_activation','observation_window_refs','focused_control_identity']
        self.allowed_apps = ['com.apple.TextEdit','com.apple.Terminal','com.apple.installer']
        self.driver = MacDriver(self.driver.endpoint,'mac-test-device',self.root/'token',self.root/'native-v2-receipts')
        await self.driver.start()
        self.gateway = Gateway(self.driver,self.root/'native-v2-gateway')
        await self.ready()

    async def test_native_app_activation_and_lightweight_wait_need_one_input_and_no_post_capture(self):
        await self.enable_native_foundation()
        client=TestClient(TestServer(create_mac_app(self.gateway,'gateway-token')))
        await client.start_server()
        desktop=Desktop(str(client.make_url('')).rstrip('/'),'gateway-token',self.root/'desktop')
        try:
            await desktop.start();await desktop.control('resume')
            captures=self.capture_reads
            with patch('macos_agent_driver.asyncio.create_subprocess_exec') as ssh:
                result=await TargetExecution(desktop).invoke({'operation':'ensure_app','app':'installer',
                    'transition_id':'native-installer','timeout_seconds':2})
                ssh.assert_not_called()
            self.assertTrue(result['verified']);self.assertTrue(result['observe_before_input'])
            self.assertNotIn('image',result['observation'])
            self.assertEqual(self.capture_reads,captures+1)
            self.assertGreater(self.state_reads,0);self.assertEqual(self.inputs,1)
            self.assertIsNone(self.gateway.latest)
            self.assertFalse(desktop.unresolved())
            # Live state must not silently create an actionable capture.
            with self.assertRaises(DeviceRejected):await self.gateway.act({'action_id':'stale-after-state',
                'desktop_id':self.gateway.desktop_id,'observation_id':'old-capture',
                'action':{'kind':'key','key':'ENTER'}})
        finally:
            await desktop.close();await client.close()

    async def test_native_feature_negotiation_only_exposes_implemented_candidates(self):
        await self.enable_native_foundation()
        tools=tools_for_platform('macos',device_capabilities=self.driver.capabilities)
        kinds={b['properties']['kind']['const'] for b in tools[1]['inputSchema']['properties']['action']['oneOf']}
        self.assertTrue({'activate_window','focus_control'}<=kinds)
        self.mode='permission-off'
        await self.gateway.state()
        self.assertFalse(self.driver.capabilities['available_now'])

    async def test_lightweight_state_restart_invalidates_capture_without_input(self):
        await self.enable_native_foundation()
        self.native_session=str(uuid.uuid4())
        with self.assertRaisesRegex(DeviceRejected,'session changed'):await self.gateway.state()
        self.assertIsNone(self.driver.capture);self.assertEqual(self.inputs,0)

    async def enable_activation(self):
        await self.gateway.close()
        self.allowed_apps = ['com.apple.Terminal', 'com.apple.installer']
        connection = self.root / 'activation.json'
        connection.write_text(json.dumps({'endpoint': self.driver.endpoint, 'device_id': 'mac-test-device',
            'user': 'sunny', 'host': '100.77.55.48', 'identity_file': '/tmp/fixture-key', 'known_hosts': '/tmp/fixture-hosts'}))
        self.driver = MacDriver(self.driver.endpoint, 'mac-test-device', self.root / 'token', self.root / 'activation-receipts',
                                activation_connection=connection)
        await self.driver.start()
        self.gateway = Gateway(self.driver, self.root / 'activation-gateway')
        await self.ready()

    async def test_fixed_app_activation_uses_one_command_and_duplicate_does_not_resend(self):
        await self.enable_activation()
        child = AsyncMock(); child.returncode = 0; child.communicate.return_value = (b'', b'')
        request = self.action(); request['action'] = {'kind': 'open_app', 'app': 'installer'}
        with patch('macos_agent_driver.asyncio.create_subprocess_exec', return_value=child) as start:
            result = await self.gateway.act(request)
            self.assertEqual(result['driver_receipt']['transport'], 'fixed-system-app-activation-ssh')
            duplicate = await self.gateway.act(request)
            self.assertTrue(duplicate['duplicate'])
            start.assert_awaited_once()
            self.assertEqual(start.call_args.args[-1], '/usr/bin/open -a /System/Library/CoreServices/Installer.app')
            self.assertEqual(self.inputs, 0)

    async def test_disabled_activation_or_expiring_scope_sends_no_ssh_command(self):
        await self.enable_activation()
        for alias in ('notepad', 'installer'):
            self.deadline = time.time() + 1 if alias == 'installer' else None
            request = self.action('activation-disabled-' + alias)
            request['action'] = {'kind': 'open_app', 'app': alias}
            with patch('macos_agent_driver.asyncio.create_subprocess_exec') as start:
                with self.assertRaises(DeviceRejected): await self.gateway.act(request)
                start.assert_not_called()

    async def test_failed_app_activation_is_uncertain_and_never_retried(self):
        await self.enable_activation()
        child = AsyncMock(); child.returncode = 1; child.communicate.return_value = (b'', b'failed')
        request = self.action('activation-uncertain'); request['action'] = {'kind': 'open_app', 'app': 'terminal'}
        with patch('macos_agent_driver.asyncio.create_subprocess_exec', return_value=child) as start:
            with self.assertRaises(RuntimeError): await self.gateway.act(request)
            self.assertEqual(self.gateway.receipt(request['action_id'])['status'], 'uncertain')
            with self.assertRaises(DeviceRejected): await self.gateway.act(request)
            start.assert_awaited_once()

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
