"""Paused Android session recovery without repeating an input or session POST.

This opt-in component reuses the existing driver and gateway. It is not loaded
by the frozen test queues. The embedding runner owns and verifies the concrete
Appium server and emulator boot through read_identity.
"""
import asyncio
import json
from pathlib import Path
import sqlite3
import time
import uuid

import aiohttp
from aiohttp import web

from qw3_computer.android_driver import AndroidDriver
from qw3_computer.device_gateway import DeviceRejected, Gateway, create_app as original_create_app


class AppiumError(RuntimeError):
    def __init__(self, code, status, method, session_id):
        self.code, self.status, self.method, self.session_id = code, status, method, session_id
        super().__init__(f'Appium {method} failed ({status}, {code})')


class IdleRecoveryDriver(AndroidDriver):
    def __init__(self, endpoint, device_id, serial, *, command_timeout=3600, system_port=18221):
        super().__init__(endpoint, device_id, serial)
        if type(command_timeout) is not int or not 1 <= command_timeout <= 3600:
            raise ValueError('Require a bounded Appium command timeout')
        if type(system_port) is not int or not 1024 <= system_port <= 65535:
            raise ValueError('Require an explicitly reserved system port')
        self.command_timeout, self.system_port = command_timeout, system_port

    async def request(self, method, route, body=None):
        async with self.session.request(method, self.endpoint + route, json=body) as response:
            raw = await response.json()
            value = raw.get('value')
            code = value.get('error') if isinstance(value, dict) else None
            if response.status != 200 or code:
                raise AppiumError(code or 'http error', response.status, method, self.session_id)
            return value

    async def create_session_once(self):
        # This POST is deliberately outside every retry loop. A timeout leaves
        # an unresolved creation, even if the server later finishes its work.
        body = {'capabilities': {'alwaysMatch': {
            'platformName': 'Android', 'appium:automationName': 'UiAutomator2',
            'appium:udid': self.serial, 'appium:noReset': True, 'appium:skipUnlock': True,
            'appium:newCommandTimeout': self.command_timeout, 'appium:autoLaunch': False,
            'appium:systemPort': self.system_port}}}
        async with self.session.post(self.endpoint + '/session', json=body,
                                     timeout=aiohttp.ClientTimeout(total=120)) as response:
            raw = await response.json()
            value = raw.get('value', {})
            if response.status != 200 or not isinstance(value, dict) or value.get('error'):
                raise RuntimeError('Appium session creation was not confirmed')
            session_id = value.get('sessionId')
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError('Appium did not return a session identity')
            self.session_id = session_id
            caps = value.get('capabilities', {})
            serial = caps.get('appium:udid', caps.get('udid', caps.get('deviceUDID')))
            if serial != self.serial or str(caps.get('platformName', '')).lower() != 'android':
                raise RuntimeError('Created Appium session does not match the selected emulator')
        return session_id

    async def start(self):
        self.session = aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=15))
        try:
            await self.create_session_once()
        except BaseException:
            await self.session.close()
            raise

    async def close(self):
        try:
            if self.session_id:
                try:
                    await self.command('DELETE', '')
                except AppiumError as error:
                    if error.code != 'invalid session id':
                        raise
        finally:
            await self.session.close()


IDENTITY_FIELDS = {'device_id', 'serial', 'server_instance', 'device_boot'}


def checked_identity(identity, driver):
    if (not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS
            or any(not isinstance(value, str) or not value for value in identity.values())
            or identity['device_id'] != driver.device_id or identity['serial'] != driver.serial):
        raise DeviceRejected('Require the selected emulator boot and owned Appium instance')
    return dict(identity)


def require_clean_start(runtime):
    """Check durable unresolved work before creating any startup session."""
    path = Path(runtime) / 'receipts.sqlite'
    if not path.exists():
        return
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'actions' in tables and db.execute(
                "SELECT 1 FROM actions WHERE status NOT IN ('completed','rejected') LIMIT 1").fetchone():
            raise DeviceRejected('Unsettled input requires reconciliation before reconnecting')
        if 'recoveries' in tables and db.execute(
                "SELECT 1 FROM recoveries WHERE status IN ('creating','creation-uncertain') LIMIT 1").fetchone():
            raise DeviceRejected('Unresolved session creation requires reconciliation before reconnecting')


class IdleRecoveryGateway(Gateway):
    def __init__(self, driver, runtime, expected_identity, read_identity):
        super().__init__(driver, runtime)
        self.expected_identity = checked_identity(expected_identity, driver)
        self.read_identity = read_identity
        self.recovering = False
        self.db.execute('CREATE TABLE IF NOT EXISTS recoveries('
                        'expired_session TEXT PRIMARY KEY,before_epoch TEXT,status TEXT,new_session TEXT,evidence TEXT)')
        self.db.commit()

    def recovery_rows(self):
        return [dict(zip(('expired_session', 'before_epoch', 'status', 'new_session', 'evidence'), row))
                for row in self.db.execute('SELECT * FROM recoveries ORDER BY rowid')]

    async def close(self):
        try:
            await self.driver.close()
        finally:
            self.db.close()

    async def observe(self):
        try:
            return await super().observe()
        except AppiumError as error:
            if error.code != 'invalid session id' or error.method != 'GET':
                raise
            return await self.recover_observation(error)

    async def recover_observation(self, error):
        if (not self.paused or self.recovering or error.session_id != self.driver.session_id
                or not error.session_id):
            raise DeviceRejected('Pause the selected device before recovering an expired session')
        require_clean_start(self.runtime)
        if self.db.execute('SELECT 1 FROM recoveries WHERE expired_session=?', (error.session_id,)).fetchone():
            raise DeviceRejected('This expired session already has a recovery receipt; never repeat its creation')
        # The HTTP caller holds Gateway.lock throughout observation/recovery.
        # Reserve before the identity probe so even concurrent control requests
        # cannot resume this gateway while the probe or POST is in flight.
        self.recovering = True
        before = self.session_id
        try:
            identity = checked_identity(await self.read_identity(), self.driver)
            if identity != self.expected_identity:
                raise DeviceRejected('Emulator boot or owned Appium instance changed; keep paused')
            evidence = {'identity': identity, 'started_at': time.time(), 'input_replayed': False,
                        'reason': 'structured-invalid-session-id-on-read', 'creation_posts': 1}
            self.db.execute("INSERT INTO recoveries VALUES (?,?,'creating',NULL,?)",
                            (error.session_id, before, json.dumps(evidence)))
            self.db.commit()
            # Invalidate all old observations before a possibly ambiguous POST.
            self.session_id = uuid.uuid4().hex
            self.desktop_id = self.driver.device_id + ':' + self.session_id
            self.latest = None
            try:
                new_session = await self.driver.create_session_once()
            except BaseException:
                self.db.execute("UPDATE recoveries SET status='creation-uncertain' WHERE expired_session=?",
                                (error.session_id,))
                self.db.commit()
                raise
            self.db.execute("UPDATE recoveries SET status='created',new_session=? WHERE expired_session=?",
                            (new_session, error.session_id))
            self.db.commit()
            # Call the original implementation once: a second expiry does not
            # recursively create another session in this request.
            observation = await super().observe()
            evidence['finished_at'] = time.time()
            self.db.execute("UPDATE recoveries SET status='observed',evidence=? WHERE expired_session=?",
                            (json.dumps(evidence), error.session_id))
            self.db.commit()
            return observation | {'session_recovery': {'status': 'observed', 'input_replayed': False,
                                                       'previous_gateway_session': before}}
        finally:
            self.paused = True
            self.recovering = False


def create_app(gateway, token):
    app = original_create_app(gateway, token)

    @web.middleware
    async def recovery_control(request, handler):
        if request.path != '/control':
            return await handler(request)
        # Keep the original authentication boundary ahead of this middleware.
        command = (await request.json()).get('command')
        if command != 'resume':
            return await handler(request)
        epoch = gateway.session_id
        if gateway.recovering:
            raise DeviceRejected('Recovery is in progress; resume only after a fresh observation')
        async with gateway.lock:
            if gateway.recovering or epoch != gateway.session_id:
                raise DeviceRejected('Connection changed while waiting; review its fresh observation before resume')
            return await handler(request)

    app.middlewares.append(recovery_control)
    return app


async def serve(args, expected_identity, read_identity):
    """Opt-in runner retaining the existing device lease and gateway endpoints."""
    from qw3_computer.device_lease import DeviceLease
    import secrets
    import signal

    require_clean_start(args.runtime)
    driver = IdleRecoveryDriver(args.driver_endpoint, args.device_id, args.android_serial)
    identity = checked_identity(expected_identity, driver)
    if checked_identity(await read_identity(), driver) != identity:
        raise DeviceRejected('Selected emulator authority changed before startup')
    runtime = Path(args.runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    token_path = runtime / 'token.txt'
    if not token_path.exists():
        token_path.write_text(secrets.token_hex(32)); token_path.chmod(0o600)
    lease = DeviceLease(driver.endpoint + ':' + driver.serial).acquire()
    gateway = runner = None
    driver_started = False
    try:
        require_clean_start(runtime)
        if checked_identity(await read_identity(), driver) != identity:
            raise DeviceRejected('Selected emulator authority changed while acquiring its lease')
        await driver.start()
        driver_started = True
        gateway = IdleRecoveryGateway(driver, runtime, identity, read_identity)
        runner = web.AppRunner(create_app(gateway, token_path.read_text().strip()), handler_cancellation=True)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', args.port).start()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
    finally:
        try:
            if runner is not None:
                await runner.cleanup()
            if gateway is not None:
                await gateway.close()
            elif driver_started:
                await driver.close()
        finally:
            lease.close()
