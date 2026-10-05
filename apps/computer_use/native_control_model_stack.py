"""Private VM, gateway and Codex client for the unchanged native-control study.

This module does not dispatch a task or start an engine. A model trial must bind
an independently registered engine before using the frozen pilot adapter.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from unittest.mock import patch
import uuid

import native_control_pilot_adapter as adapter

ROOT = adapter.queue.ROOT
RUNTIME = ROOT / 'apps/computer_use/.runtime'
CLIENT_PYTHON = ROOT / '.venv/bin/python'
STUDY_SHA = 'b36f465ecd64d9b839d5c8f0b949ac70a66dd2181f097f05e31d3948951445e0'


def port_map(base):
    if type(base) is not int or not 1024 <= base <= 65530:
        raise ValueError('Declare six consecutive unprivileged loopback ports')
    return dict(zip(('server', 'chromium', 'vnc', 'vlc', 'gateway', 'client'), range(base, base + 6)))


def idle_profile(state):
    return (state.get('ready') is True and state.get('status') == 'idle'
            and state.get('desktop_paused') is True and not state.get('uncertain_actions')
            and state.get('thinking') is True and state.get('model_output_tokens') == 16384
            and state.get('tool_call_limit') == 100 and state.get('tool_calls_used') == 0
            and state.get('software') is None and state.get('devices', {}).get('selected') == 'linux-vm')


def retired(identity):
    actual = adapter.owned.process_identity(identity['pid'])
    return actual is None or actual['state'] == 'Z' or not adapter.owned.same_process(identity, actual)


def descendants(identity):
    """Capture descendants of this exact client, without exposing its environment."""
    adapter.queue.require_identity(identity)
    processes = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) <= 1:
            continue
        try:
            current = adapter.owned.process_identity(int(path.name))
            if current and current['uid'] == os.getuid() and current['state'] != 'Z':
                parent = int((path / 'stat').read_text().rpartition(')')[2].split()[1])
                processes[current['pid']] = (parent, current)
        except (OSError, ValueError, UnicodeError, IndexError):
            continue
    selected = {identity['pid']}
    while True:
        added = {pid for pid, (parent, _) in processes.items() if parent in selected} - selected
        if not added:
            break
        selected.update(added)
    return [processes[pid][1] for pid in sorted(selected - {identity['pid']})]


def wait_vm_retired(vm, directory, seconds=30):
    """Poll the same exact container after one close, retaining every reply."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    deadline = time.monotonic() + seconds
    count = 0
    while True:
        response = subprocess.run(['docker', 'inspect', '--type', 'container', vm['container_id']],
                                  capture_output=True, text=True, timeout=10)
        count += 1
        receipt = dict(sequence=count, returncode=response.returncode,
                       stdout=response.stdout, stderr=response.stderr, checked_at=time.time())
        adapter.queue.save(directory / (str(count) + '.json'), receipt)
        if response.returncode != 0:
            if any(text in response.stderr for text in ('No such object', 'No such container')):
                return dict(retired=True, observations=count)
            raise adapter.owned.SafetyHalt('Container state read failed; retirement remains unconfirmed')
        info = json.loads(response.stdout)
        if (len(info) != 1 or info[0]['Id'] != vm['container_id'] or info[0]['Name'] != '/' + vm['name']
                or info[0]['Image'] != adapter.environment.IMAGE
                or info[0]['Config'].get('Labels', {}).get('qw3.goal') != 'cross-device-20261003'):
            raise adapter.owned.SafetyHalt('Container identity changed while waiting for retirement')
        if time.monotonic() >= deadline:
            raise adapter.owned.SafetyHalt('Same container still present after bounded retirement wait')
        time.sleep(.2)


class NativeStack:
    def __init__(self, study, trial_id, output, port_base):
        self.study_path = Path(study).resolve()
        if adapter.owned.digest(self.study_path) != STUDY_SHA:
            raise adapter.owned.SafetyHalt('The prepared eight-trial study changed')
        self.study, self.row, self.framework, self.original = adapter.validate_study(self.study_path, trial_id)
        if Path(adapter.desktop_module.__file__).resolve().parent != self.framework / 'qw3_computer':
            raise adapter.owned.SafetyHalt('Interpreter selected a different private framework')
        self.output = Path(output).resolve()
        if not self.output.is_relative_to(RUNTIME):
            raise ValueError('Use a fresh runtime directory owned by this goal')
        self.ports = port_map(port_base)
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.vm_name = 'qw3-native-control-model-' + uuid.uuid4().hex[:20]
        self.device_id = 'native-pilot-' + uuid.uuid4().hex[:16]
        self.env = self.transport = None
        self.children = {}
        self.record = dict(kind='owned_private_native_control_model_stack', started_at=time.time(),
            study=dict(path=str(self.study_path), sha256=STUDY_SHA), trial_id=trial_id,
            framework=str(self.framework), ports=self.ports, container_name=self.vm_name,
            producer=adapter.owned.process_identity(os.getpid()), services={}, status='prepared',
            task_dispatches_by_stack=0, engine_starts_by_stack=0, source_sha256={})
        for name in ('native_control_model_stack.py', 'native_control_pilot_adapter.py',
                     'osworld_linux/environment.py', 'device_gateway_entry.py'):
            source = ROOT / 'apps/computer_use' / name
            target = self.output / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            self.record['source_sha256'][name] = adapter.owned.digest(source)
        self.save()

    def save(self, **values):
        self.record.update(values)
        adapter.queue.save(self.output / 'stack.json', self.record)

    @property
    def client(self):
        return self.record['services']['client']

    @property
    def gateway(self):
        return self.record['services']['gateway']

    def spawn(self, role, command):
        if role in self.children:
            raise adapter.owned.SafetyHalt('Never replace or retry a service instance')
        pythonpath = os.pathsep.join((str(self.framework), str(ROOT / 'apps/computer_use'),
                                    str(RUNTIME / 'media-qualification')))
        env = dict(os.environ, PYTHONPATH=pythonpath)
        with (self.output / (role + '.log')).open('xb') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self.children[role] = child
        identity = adapter.owned.register_child(child.pid, command)
        runtime = Path(adapter.option(command, '--runtime')).resolve()
        self.record['services'][role] = dict(identity=identity, runtime=str(runtime),
            endpoint='http://127.0.0.1:' + adapter.option(command, '--port'), pythonpath=pythonpath)
        self.save()
        return child

    def ready(self, role, getter, predicate, seconds=120):
        deadline = time.monotonic() + seconds
        child = self.children[role]
        while True:
            if child.poll() is not None:
                raise adapter.owned.SafetyHalt('The same service exited before readiness: ' + role)
            adapter.queue.require_identity(self.record['services'][role]['identity'])
            try:
                value = getter()
            except (OSError, ValueError, KeyError):
                value = None
            if value is not None and predicate(value):
                return value
            if time.monotonic() >= deadline:
                raise adapter.owned.SafetyHalt('Readiness remains unconfirmed; no replacement: ' + role)
            time.sleep(.5)

    def start(self):
        if self.record['status'] != 'prepared':
            raise adapter.owned.SafetyHalt('A stack can start only once')
        if shutil.disk_usage(self.output).free < 50 * 1024**3:
            raise adapter.owned.SafetyHalt('Less than 50 GiB free for a private VM')
        available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        if available < 12 * 1024**2:
            raise adapter.owned.SafetyHalt('Less than 12 GiB available host memory')
        subprocess.run(['docker', 'image', 'inspect', adapter.environment.IMAGE],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=5)
        for port in self.ports.values():
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        stat = adapter.environment.DISK.stat()
        self.seed_before = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        self.save(status='booting', host_mem_available_kib=available)
        with patch.object(adapter.environment, 'PORTS', {k: self.ports[k] for k in ('server', 'chromium', 'vnc', 'vlc')}):
            self.env = adapter.environment.create(self.output, self.vm_name)
        manifest = json.loads((self.output / 'vm.json').read_text())
        vm = dict(name=self.vm_name, container_id=manifest['container_id'],
                  endpoint='http://127.0.0.1:' + str(self.ports['server']))
        self.record['vm'] = vm
        self.transport = adapter.OwnedGuestTransport(vm, self.output,
            dict(administrative_adaptations=[], empty_editor_process_reads=[]))
        self.transport.owned_vm()
        gateway_runtime = self.output / 'gateway'
        command = [str(CLIENT_PYTHON), '-m', 'device_gateway_entry', '--platform', 'linux',
                   '--driver-endpoint', vm['endpoint'], '--device-id', self.device_id,
                   '--runtime', str(gateway_runtime), '--port', str(self.ports['gateway'])]
        self.spawn('gateway', command)
        item = dict(id='linux-vm', label='Private Linux', platform='linux',
                    endpoint='http://127.0.0.1:' + str(self.ports['gateway']),
                    token_file=str(gateway_runtime / 'token.txt'))
        observation = self.ready('gateway', lambda: adapter.queue.gateway_api(item, '/observe'),
            lambda value: value.get('paused') is True and value.get('device_id') == self.device_id)
        registry = self.output / 'registry.json'
        adapter.queue.save(registry, dict(devices=[item]))
        registry.chmod(0o600)
        self.record['registry'] = dict(path=str(registry), sha256=adapter.owned.digest(registry))
        self.record['gateway_initial_observation'] = {k: observation[k] for k in ('device_id', 'session_id')}
        self.save(status='starting-private-codex-client')
        command = [str(CLIENT_PYTHON), '-m', 'qw3_computer.app', '--port', str(self.ports['client']),
                   '--runtime', str(self.output / 'client'), '--device-registry', str(registry),
                   '--selected-device', 'linux-vm', '--backend', 'http://127.0.0.1:18080/v1',
                   '--execution-mode', 'verified', '--thinking', '--model-output-tokens', '16384',
                   '--tool-call-limit', '100', '--output-policy', 'task']
        self.spawn('client', command)
        state = self.ready('client', lambda: adapter.pilot.request(self.client['endpoint'], '/api/state'), idle_profile)
        adapter.queue.save(self.output / 'initial-client-state.json', state)
        self.save(status='ready-idle', client_descendants=descendants(self.client['identity']))
        return state

    def binding(self, engine):
        """No engine-less binding may be used to dispatch the real pilot."""
        if self.record['status'] != 'ready-idle':
            raise adapter.owned.SafetyHalt('Stack is not ready for model binding')
        actual = adapter.owned.register_engine(Path(engine['runtime']), self.original['engine_off_identity'])
        if not adapter.owned.same_process(engine['identity'], actual):
            raise adapter.owned.SafetyHalt('The model engine generation changed')
        return dict(vm=self.record['vm'], client=self.client, gateway=self.gateway,
                    registry=self.record['registry'], engine=engine)

    def close(self):
        if self.record['status'] == 'retired':
            return self.record
        for role, child in self.children.items():
            if role not in self.record['services'] and child.poll() is None:
                raise adapter.owned.SafetyHalt('A live child lacks its full registered identity')
        # Reconcile the actual state before sending any signal. Unknown task or
        # action delivery preserves this stack for audit; no replacement starts.
        client = self.record['services'].get('client')
        gateway = self.record['services'].get('gateway')
        if client and not retired(client['identity']):
            adapter.queue.require_identity(client['identity'])
            adapter.owned.assert_idle(client['endpoint'])
        if gateway:
            adapter.queue.ledger_terminal(Path(gateway['runtime']))
            if not retired(gateway['identity']):
                adapter.queue.require_identity(gateway['identity'])
                item = dict(endpoint=gateway['endpoint'], token_file=Path(gateway['runtime']) / 'token.txt')
                if adapter.queue.gateway_api(item, '/observe').get('paused') is not True:
                    raise adapter.owned.SafetyHalt('Private input device remains active')
        self.save(status='retiring', retirements={})
        for role in ('client', 'gateway'):
            service = self.record['services'].get(role)
            if service:
                self.record['retirements'][role] = adapter.owned.retire_registered(service['identity'])
                self.children[role].wait(timeout=2)
                self.save()
                if role == 'client':
                    deadline = time.monotonic() + 10
                    while not all(retired(row) for row in self.record.get('client_descendants', [])):
                        if time.monotonic() >= deadline:
                            raise adapter.owned.SafetyHalt('A registered Codex descendant remains live')
                        time.sleep(.2)
        if self.env is not None:
            self.transport.owned_vm()
            self.env.close()
            self.record['vm_retirement'] = wait_vm_retired(self.record['vm'], self.output / 'vm-retirement-observations')
            self.record['vm_retired'] = True
        if hasattr(self, 'seed_before'):
            stat = adapter.environment.DISK.stat()
            same = self.seed_before == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            self.record['seed_metadata_unchanged'] = same
            if not same:
                raise adapter.owned.SafetyHalt('Read-only VM seed metadata changed')
        self.save(status='retired', finished_at=time.time())
        return self.record
