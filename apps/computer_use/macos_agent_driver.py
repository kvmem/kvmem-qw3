"""Opt-in native Mac driver over an authenticated, SSH-forwarded loopback API.

This module is separate from sources pinned by the restored comparison cohorts.
It exposes no remote shell action. Delivery uncertainty never causes a retry.
"""
import asyncio
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse
import uuid

import aiohttp
from PIL import Image

from qw3_computer.device_gateway import DeviceRejected
from qw3_computer.device_protocol import capabilities


class MacDriver:
    def __init__(self, endpoint, device_id, token_path, runtime, *, activation_connection=None):
        parsed = urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'} or parsed.username:
            raise ValueError('The Mac API must use a local SSH-forwarded HTTP endpoint')
        if not device_id or not parsed.port:
            raise ValueError('A concrete device ID and local endpoint port are required')
        self.endpoint = endpoint.rstrip('/')
        self.device_id = device_id
        # DeviceLease must use the stable concrete Mac identity, not a tunnel port.
        self.serial = device_id
        self.token_path = Path(token_path)
        self.runtime = Path(runtime)
        self.owner = str(uuid.uuid4())
        self.native_session = None
        self.capture = None
        self.session = None
        self.heartbeat_task = None
        self.lease_error = None
        self.activation = None
        self.native_features = set()
        self.launch_apps = {
            'terminal': ('com.apple.Terminal', '/System/Applications/Utilities/Terminal.app'),
            'installer': ('com.apple.installer', '/System/Library/CoreServices/Installer.app'),
        }
        if activation_connection is not None:
            connection = json.loads(Path(activation_connection).read_text())
            if connection['endpoint'].rstrip('/') != self.endpoint or connection['device_id'] != device_id:
                raise ValueError('Mac activation metadata must identify the same native device and endpoint')
            if not re.fullmatch(r'[a-z_][a-z0-9_-]*', connection['user']) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9.:-]*', connection['host']):
                raise ValueError('Invalid owner SSH target')
            self.activation = connection
        self.capabilities = capabilities('macos') | {
            'implemented': True, 'actions': ['click', 'type', 'key', 'scroll', 'open_app'],
            'accessibility': 'AXUIElement', 'transport': 'native-macos-ssh-forwarded-loopback',
            'open_apps': ['notepad'], 'app_processes': {'notepad': ['com.apple.TextEdit']},
            'keyboard_aliases': {'CTRL': 'Command', 'ALT': 'Option'},
            'qualification': 'opt-in candidate; model business success unproven',
            'scope': 'Configured task applications and window title prefixes only',
        }

    async def request(self, method, path, body=None):
        # Exactly one request. Cancellation/timeout on input means uncertain delivery.
        async with self.session.request(method, self.endpoint + path, json=body) as response:
            result = await response.json()
            if response.status == 409 and result.get('delivery') == 'rejected':
                raise DeviceRejected(result.get('error', 'Native Mac rejected before input'))
            if response.status != 200:
                raise RuntimeError('Native Mac request failed: ' + str(result.get('error', response.status)))
            if result.get('delivery') == 'uncertain':
                raise RuntimeError('Native Mac input delivery is uncertain')
            return result

    async def start(self):
        token = self.token_path.read_text().strip()
        if len(token) < 32:
            raise ValueError('Invalid native Mac authentication token')
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.session = aiohttp.ClientSession(trust_env=False,
            headers={'Authorization': 'Bearer ' + token}, timeout=aiohttp.ClientTimeout(total=20))
        try:
            status = await self.request('POST', '/claim', {'owner': self.owner})
            if status.get('protocol') != 'qw3-macos-native/1' or not status.get('paused'):
                raise RuntimeError('Unexpected native protocol or unpaused claim')
            self.native_session = status['session_id']
            self.native_features = set(status.get('native_features', []))
            allowed = status.get('allowed_apps', ['com.apple.TextEdit'])
            aliases = ['notepad'] if 'com.apple.TextEdit' in allowed else []
            processes = {'notepad': ['com.apple.TextEdit']} if aliases else {}
            if self.activation or 'native_app_activation' in self.native_features:
                for alias, (bundle, _) in self.launch_apps.items():
                    if bundle in allowed:
                        aliases.append(alias); processes[alias] = [bundle]
            permissions = [status.get('ax_trusted'), status.get('screen_capture_allowed')]
            self.capabilities.update(open_apps=aliases, app_processes=processes,
                available_now=all(permissions) if all(type(p) is bool for p in permissions) else None,
                allowed_window_prefixes=status.get('allowed_window_prefixes', []),
                scope_expires_at=status.get('scope_expires_at'),
                limitations=['Named app activation does not select an arbitrary document window',
                             'Control focus and multiple displays are not qualified'])
            self.capabilities['lightweight_state'] = 'lightweight_state' in self.native_features
            if 'observation_window_refs' in self.native_features:
                self.capabilities['actions'].append('activate_window')
            if 'focused_control_identity' in self.native_features:
                self.capabilities['actions'].append('focus_control')
            self.capabilities['owned_modal_evidence'] = 'observation_window_refs' in self.native_features
            self.heartbeat_task = asyncio.create_task(self.heartbeat())
        except BaseException:
            await self.session.close()
            raise

    async def heartbeat(self):
        try:
            while True:
                await asyncio.sleep(5)
                status = await self.request('POST', '/heartbeat', {'owner': self.owner})
                if status.get('session_id') != self.native_session:
                    raise RuntimeError('Native Mac session restarted')
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.lease_error = type(exc).__name__
            self.capture = None

    def ensure_session(self, value):
        if self.lease_error:
            raise DeviceRejected('Mac input lease heartbeat failed; reconcile and reconnect while paused')
        if value.get('native_session_id', value.get('session_id')) != self.native_session:
            self.capture = None
            raise DeviceRejected('Native Mac session changed; old captures cannot be used')

    async def control(self, command):
        status = await self.request('POST', '/control', {'owner': self.owner, 'command': command})
        self.ensure_session(status)
        return status

    async def state(self):
        if 'lightweight_state' not in self.native_features:
            raise DeviceRejected('Lightweight state is unavailable on this native version')
        raw = await self.request('GET', '/state')
        self.ensure_session(raw)
        permissions = [raw.get('ax_trusted'), raw.get('screen_capture_allowed')]
        self.capabilities['available_now'] = all(permissions) if all(type(p) is bool for p in permissions) else None
        self.capabilities['allowed_window_prefixes'] = raw.get('allowed_window_prefixes', [])
        self.capabilities['scope_expires_at'] = raw.get('scope_expires_at')
        return {k: raw.get(k) for k in ('foreground', 'focused_control', 'foreground_in_scope', 'timestamp', 'native_session_id')} | {
            'native_session_id': self.native_session, 'device_paused': raw.get('paused')}

    async def observe(self):
        raw = await self.request('GET', '/observe')
        self.ensure_session(raw)
        png = base64.b64decode(raw.pop('png'), validate=True)
        geometry = raw.get('geometry', {})
        with Image.open(io.BytesIO(png)) as image:
            image.load()
            if image.format != 'PNG' or (image.width, image.height) != (geometry.get('width'), geometry.get('height')):
                self.capture = None
                raise RuntimeError('Screenshot pixels differ from the declared native geometry')
            bounds = raw.get('foreground', {}).get('bounds')
            if bounds:
                scale_x = image.width / geometry['point_width']
                scale_y = image.height / geometry['point_height']
                box = (max(0, int((bounds['x'] - geometry['point_x']) * scale_x)),
                       max(0, int((bounds['y'] - geometry['point_y']) * scale_y)),
                       min(image.width, int((bounds['x'] + bounds['width'] - geometry['point_x']) * scale_x)),
                       min(image.height, int((bounds['y'] + bounds['height'] - geometry['point_y']) * scale_y)))
                region = image.crop(box) if box[2] > box[0] and box[3] > box[1] else image
            else:
                box = (0, 0, image.width, image.height)
                region = image
            progress = hashlib.sha256(region.convert('RGB').tobytes()).hexdigest()
        self.capture = {'id': raw['native_capture_id'], 'session': self.native_session,
                        'geometry': geometry, 'foreground': raw['foreground']}
        return raw | {'png': png, 'progress_sha256': progress,
                      'progress_region': {'bounds': list(box), 'coordinate_space': 'pixels',
                                          'scope': 'native_foreground_window_when_available'}}

    def native_arguments(self, action):
        if not self.capture or self.lease_error:
            raise DeviceRejected('No fresh native Mac capture is available')
        return {'owner': self.owner, 'session_id': self.native_session,
                'capture_id': self.capture['id'], 'action': action}

    async def preflight(self, action, observation):
        if observation.get('native_capture_id') != (self.capture or {}).get('id'):
            raise DeviceRejected('The host and native capture identities differ')
        if action['kind'] == 'open_app' and action.get('app') not in self.capabilities['open_apps']:
            raise DeviceRejected('App activation is not enabled for this Mac task')
        await self.request('POST', '/preflight', self.native_arguments(action))

    async def activate_system_app(self, action, request_id):
        alias = action['app']
        if not self.activation or alias not in self.capabilities['open_apps']:
            raise DeviceRejected('App activation is not enabled for this Mac task')
        bundle, path = self.launch_apps[alias]
        # Refresh owner, pause and expiry evidence immediately before the one launch.
        status = await self.request('POST', '/heartbeat', {'owner': self.owner})
        self.ensure_session(status)
        if status.get('paused') or bundle not in status.get('allowed_apps', []):
            raise DeviceRejected('App activation rejected by the current native task scope')
        deadline = status.get('scope_expires_at')
        if deadline is not None and deadline <= status.get('timestamp', time.time()) + 20:
            raise DeviceRejected('Task scope is expiring; no app activation sent')
        c = self.activation
        process = await asyncio.create_subprocess_exec('ssh', '-F', '/dev/null', '-i', c['identity_file'],
            '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10',
            '-o', 'UserKnownHostsFile=' + c['known_hosts'], c['user'] + '@' + c['host'],
            '/usr/bin/open -a ' + path, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            await asyncio.wait_for(process.communicate(), timeout=15)
        except BaseException:
            if process.returncode is None:
                process.kill()
                with contextlib.suppress(Exception): await process.wait()
            raise
        if process.returncode:
            raise RuntimeError('Fixed Mac app activation was not confirmed; do not resend automatically')
        return {'delivery': 'delivered', 'request_id': request_id, 'status': 'completed',
                'transport': 'fixed-system-app-activation-ssh', 'action': action,
                'verification': 'Launch delivery only; verify the resulting foreground', 'native_session_id': self.native_session}

    async def dispatch(self, action):
        arguments = self.native_arguments(action) | {'request_id': str(uuid.uuid4())}
        # Local durable dispatch record binds an uncertain HTTP outcome to the native receipt.
        path = self.runtime / (arguments['request_id'] + '.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump({'request': arguments, 'status': 'dispatching', 'created_at': time.time()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if (action['kind'] == 'open_app' and action.get('app') in self.launch_apps
                    and 'native_app_activation' not in self.native_features):
                result = await self.activate_system_app(action, arguments['request_id'])
            else:
                result = await self.request('POST', '/act', arguments)
            if result.get('delivery') != 'delivered' or result.get('request_id') != arguments['request_id']:
                raise RuntimeError('Native Mac receipt does not confirm this input identity')
        except BaseException:
            self.capture = None
            raise
        self.capture = None
        path.write_text(json.dumps({'request': arguments, 'status': 'completed', 'result': result}))
        return result

    async def close(self):
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.heartbeat_task
        if self.session and not self.session.closed:
            with contextlib.suppress(Exception):
                await self.request('POST', '/release', {'owner': self.owner})
            await self.session.close()
