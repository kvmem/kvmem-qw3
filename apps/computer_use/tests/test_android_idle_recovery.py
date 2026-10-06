import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

import android_idle_recovery as recovery


IDENTITY = {'device_id': 'isolated-android', 'serial': 'emulator-5556',
            'server_instance': 'fixture-server-generation', 'device_boot': 'fixture-boot-generation'}


class Driver:
    device_id, serial = IDENTITY['device_id'], IDENTITY['serial']
    capabilities = {'actions': ['key']}

    def __init__(self):
        self.session_id = 'expired-one'
        self.expiries = 1
        self.creations = self.inputs = 0
        self.creation_error = None
        self.creation_wait = None
        self.input_error = None

    async def observe(self):
        if self.expiries:
            self.expiries -= 1
            raise recovery.AppiumError('invalid session id', 404, 'GET', self.session_id)
        image = io.BytesIO(); Image.new('RGB', (20, 30), 'white').save(image, 'PNG')
        return {'png': image.getvalue(), 'foreground': {'process': 'fixture', 'handle': 'fixture/view'},
                'accessibility': {'status': 'ok', 'controls': []}}

    async def create_session_once(self):
        self.creations += 1
        if self.creation_wait:
            await self.creation_wait.wait()
        if self.creation_error:
            raise self.creation_error
        self.session_id = f'created-{self.creations}'
        return self.session_id

    async def preflight(self, *_):
        pass

    async def dispatch(self, *_):
        self.inputs += 1
        if self.input_error:
            raise self.input_error
        return {'request_completed': True}

    async def close(self):
        pass


class IdleRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.driver = Driver()
        self.read_identity = AsyncMock(return_value=dict(IDENTITY))
        self.gateway = recovery.IdleRecoveryGateway(self.driver, self.temp.name, IDENTITY, self.read_identity)

    async def asyncTearDown(self):
        await self.gateway.close()
        self.temp.cleanup()

    def action(self, observation, action_id='one'):
        return {'action_id': action_id, 'desktop_id': observation['desktop_id'],
                'observation_id': observation['observation_id'],
                'action': {'kind': 'key', 'key': 'ANDROID_HOME'}}

    async def test_paused_expiry_recovers_once_rotates_capture_and_preserves_receipts(self):
        old_epoch = self.gateway.session_id
        self.gateway.db.execute("INSERT INTO actions VALUES ('completed','{}','completed','{}')")
        self.gateway.db.commit()
        observed = await self.gateway.observe()
        self.assertTrue(observed['paused'])
        self.assertNotEqual(observed['session_id'], old_epoch)
        self.assertEqual(self.driver.creations, 1)
        self.assertEqual(self.driver.inputs, 0)
        self.assertEqual(self.gateway.recovery_rows()[0]['status'], 'observed')
        self.assertEqual(self.gateway.receipt('completed')['status'], 'completed')

    async def test_active_device_does_not_reconnect(self):
        self.gateway.paused = False
        with self.assertRaises(recovery.DeviceRejected):
            await self.gateway.observe()
        self.assertEqual(self.driver.creations, 0)

    async def test_unsettled_inputs_prevent_recovery_and_startup(self):
        for status in ('dispatching', 'uncertain', 'unknown'):
            self.gateway.db.execute('DELETE FROM actions')
            self.gateway.db.execute('INSERT INTO actions VALUES (?,?,?,NULL)', ('one', '{}', status))
            self.gateway.db.commit(); self.driver.expiries = 1
            with self.assertRaises(recovery.DeviceRejected):
                await self.gateway.observe()
            with self.assertRaises(recovery.DeviceRejected):
                recovery.require_clean_start(self.temp.name)
        self.assertEqual(self.driver.creations, 0)

    async def test_boot_server_or_selected_device_change_prevents_creation(self):
        for key in IDENTITY:
            changed = dict(IDENTITY); changed[key] = 'changed'
            self.read_identity.return_value = changed; self.driver.expiries = 1
            with self.assertRaises(recovery.DeviceRejected):
                await self.gateway.observe()
        self.assertEqual(self.driver.creations, 0)

    async def test_creation_timeout_is_durable_and_never_reposted(self):
        self.driver.creation_error = TimeoutError('fixture delayed POST')
        old_epoch = self.gateway.session_id
        with self.assertRaises(TimeoutError):
            await self.gateway.observe()
        self.assertEqual(self.gateway.recovery_rows()[0]['status'], 'creation-uncertain')
        self.assertNotEqual(self.gateway.session_id, old_epoch)
        self.assertIsNone(self.gateway.latest)
        self.assertTrue(self.gateway.paused)
        self.driver.expiries = 1
        with self.assertRaises(recovery.DeviceRejected):
            await self.gateway.observe()
        with self.assertRaises(recovery.DeviceRejected):
            recovery.require_clean_start(self.temp.name)
        self.assertEqual(self.driver.creations, 1)

    async def test_second_expiry_in_same_observe_has_no_recursive_creation(self):
        self.driver.expiries = 2
        with self.assertRaises(recovery.AppiumError):
            await self.gateway.observe()
        self.assertEqual(self.driver.creations, 1)
        self.assertEqual(self.gateway.recovery_rows()[0]['status'], 'created')
        self.assertTrue(self.gateway.paused)

    async def test_input_expiry_reserves_unknown_and_never_recovers_or_replays_input(self):
        self.driver.expiries = 0
        observation = await self.gateway.observe()
        self.gateway.paused = False
        self.driver.input_error = recovery.AppiumError('invalid session id', 404, 'POST', self.driver.session_id)
        with self.assertRaises(recovery.AppiumError):
            await self.gateway.act(self.action(observation))
        self.assertEqual(self.gateway.receipt('one')['status'], 'uncertain')
        self.driver.expiries = 1
        with self.assertRaises(recovery.DeviceRejected):
            await self.gateway.observe()
        self.assertEqual(self.driver.inputs, 1)
        self.assertEqual(self.driver.creations, 0)

    async def test_nonexpiry_errors_never_create_sessions(self):
        for error in (TimeoutError(), recovery.AppiumError('unknown error', 500, 'GET', self.driver.session_id)):
            self.driver.observe = AsyncMock(side_effect=error)
            with self.assertRaises(type(error)):
                await self.gateway.observe()
        self.assertEqual(self.driver.creations, 0)

    async def test_http_resume_during_recovery_is_rejected_and_old_capture_stays_stale(self):
        self.driver.expiries = 0
        old_capture = await self.gateway.observe()
        self.driver.expiries = 1; self.driver.creation_wait = asyncio.Event()
        client = TestClient(TestServer(recovery.create_app(self.gateway, 'fixture-token')))
        await client.start_server()
        try:
            headers = {'Authorization': 'Bearer fixture-token'}
            pending = asyncio.create_task(client.get('/observe', headers=headers))
            for _ in range(100):
                if self.gateway.recovering:
                    break
                await asyncio.sleep(.005)
            self.assertTrue(self.gateway.recovering)
            resume = await client.post('/control', headers=headers, json={'command': 'resume'})
            self.assertEqual(resume.status, 409)
            self.driver.creation_wait.set()
            response = await pending
            self.assertEqual(response.status, 200)
            self.assertTrue((await response.json())['paused'])
            resume = await client.post('/control', headers=headers, json={'command': 'resume'})
            self.assertEqual(resume.status, 200)
            stale = await client.post('/act', headers=headers, json=self.action(old_capture))
            self.assertEqual(stale.status, 409)
            self.assertEqual(self.driver.inputs, 0)
            unauthorized = await client.get('/observe')
            self.assertEqual(unauthorized.status, 401)
        finally:
            self.driver.creation_wait.set()
            await client.close()

    async def test_read_identity_failure_does_not_reserve_or_create(self):
        self.read_identity.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            await self.gateway.observe()
        self.assertEqual(self.driver.creations, 0)
        self.assertEqual(self.gateway.recovery_rows(), [])
        self.assertTrue(self.gateway.paused)

    async def test_receipt_database_closes_even_when_driver_shutdown_fails(self):
        self.driver.close = AsyncMock(side_effect=OSError('fixture disconnected shutdown'))
        with self.assertRaises(OSError):
            await self.gateway.close()
        import sqlite3
        with self.assertRaises(sqlite3.ProgrammingError):
            self.gateway.db.execute('SELECT 1')
        self.driver.close = AsyncMock()

    async def test_serve_rechecks_boot_after_ownership_without_starting_changed_device(self):
        self.driver.endpoint = 'http://127.0.0.1:18990'
        self.driver.start = AsyncMock()
        changed = dict(IDENTITY, device_boot='changed-after-lease')
        identity = AsyncMock(side_effect=[dict(IDENTITY), changed])
        lease = Mock(); lease.acquire.return_value = lease
        args = SimpleNamespace(runtime=Path(self.temp.name), driver_endpoint=self.driver.endpoint,
                               device_id=self.driver.device_id, android_serial=self.driver.serial, port=18991)
        with patch.object(recovery, 'IdleRecoveryDriver', return_value=self.driver),\
             patch('qw3_computer.device_lease.DeviceLease', return_value=lease):
            with self.assertRaises(recovery.DeviceRejected):
                await recovery.serve(args, IDENTITY, identity)
        self.driver.start.assert_not_called()
        lease.close.assert_called_once()


class DriverResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_structured_expiry_is_distinct_from_other_http_failures(self):
        async def endpoint(request):
            return web.json_response({'value': {'error': 'invalid session id', 'message': 'fixture'}}, status=404)
        app = web.Application(); app.router.add_get('/session/old/source', endpoint)
        server = TestServer(app); await server.start_server()
        driver = recovery.IdleRecoveryDriver(str(server.make_url('/')).rstrip('/'), IDENTITY['device_id'], IDENTITY['serial'])
        driver.session = recovery.aiohttp.ClientSession(trust_env=False); driver.session_id = 'old'
        try:
            with self.assertRaises(recovery.AppiumError) as context:
                await driver.command('GET', '/source')
            self.assertEqual((context.exception.code, context.exception.method, context.exception.session_id), ('invalid session id', 'GET', 'old'))
        finally:
            await driver.session.close(); await server.close()

    async def test_actual_session_response_must_match_serial_and_platform(self):
        requests = []; response_identity = {'platformName': 'Android', 'udid': IDENTITY['serial']}
        async def endpoint(request):
            requests.append(await request.json())
            return web.json_response({'value': {'sessionId': 'new', 'capabilities': dict(response_identity)}})
        app = web.Application(); app.router.add_post('/session', endpoint)
        server = TestServer(app); await server.start_server()
        driver = recovery.IdleRecoveryDriver(str(server.make_url('/')).rstrip('/'), IDENTITY['device_id'], IDENTITY['serial'])
        driver.session = recovery.aiohttp.ClientSession(trust_env=False)
        try:
            self.assertEqual(await driver.create_session_once(), 'new')
            response_identity['udid'] = 'emulator-5558'
            with self.assertRaises(RuntimeError):
                await driver.create_session_once()
            self.assertEqual(len(requests), 2)
            self.assertTrue(requests[0]['capabilities']['alwaysMatch']['appium:noReset'])
            self.assertFalse(requests[0]['capabilities']['alwaysMatch']['appium:autoLaunch'])
        finally:
            await driver.session.close(); await server.close()


if __name__ == '__main__':
    unittest.main()
