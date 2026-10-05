"""Read real Gedit button evidence in a dedicated owned VM, without invoking it."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
from unittest.mock import patch
import uuid
import xml.etree.ElementTree as ET

from native_control_evidence import atspi_button_script, button_metadata_from_attrs, native_button
from field_identity import provider_identity_from_attrs
from osworld_linux import environment
from qw3_computer.linux_driver import foreground_tree_script, attrs, controls_from_xml, FOREGROUND_SCRIPT
from qw3_computer.device_lease import DeviceLease
from qualify_native_keyboard_delivery import PORTS, ENDPOINT, GATEWAY_PORT, owned_vm, guest, digest
import recoverable_linux_recording as recording


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    nonce = uuid.uuid4().hex
    name = 'qw3-native-button-probe-' + nonce[:12]
    env = lease = recorder = None
    cid = None
    report = dict(kind='readonly_real_gedit_native_button_interface_probe', status='preparing',
                  started_at=time.time(), model_requests=0, action_invocations=0,
                  setup_keyboard_events=0, existing_devices_changed=False,
                  producer=recording.owned.process_identity(os.getpid()),
                  recipe_sha256=digest(__file__), helper_sha256=digest(Path(__file__).with_name('native_control_evidence.py')))

    def save(**values):
        report.update(values)
        p = output / 'state.tmp'
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        p.replace(output / 'result.json')

    save()
    try:
        if shutil.disk_usage(output).free < 50 * 1024**3:
            raise RuntimeError('Less than50GiB free for dedicated VM')
        subprocess.run(['docker', 'image', 'inspect', environment.IMAGE], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True, timeout=5)
        for port in [*PORTS.values(), GATEWAY_PORT]:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', port))
        lease = DeviceLease(ENDPOINT + ':desktop').acquire()
        seed = environment.DISK.stat()
        seed_before = (seed.st_dev, seed.st_ino, seed.st_size, seed.st_mtime_ns)
        save(status='booting-owned-vm', container_name=name)
        print('Starting dedicated Gedit native-interface probe', flush=True)
        with patch.object(environment, 'PORTS', PORTS):
            env = environment.create(output, name)
        cid = json.loads((output / 'vm.json').read_text())['container_id']
        info = owned_vm(name, cid)
        mounts = [v for v in info['Mounts'] if v['Destination'] == '/System.qcow2']
        if len(mounts) != 1 or mounts[0]['RW'] or mounts[0]['Source'] != str(environment.DISK):
            raise RuntimeError('Unexpected dedicated VM seed mount')
        save(container_id=cid, status='launching-gedit', seed_read_only_mount_verified=True)
        recorder = recording.RecoverableLinuxRecording(output, cid, name=name,
                    profile=recording.RecordingProfile(duration_seconds=240))
        recorder.start()
        launch = """import json,subprocess,time
child=subprocess.Popen(['gedit','--standalone'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
print(json.dumps(dict(pid=child.pid,action_invocations=0)))
"""
        save(gedit_launch=guest(launch))
        deadline = time.monotonic() + 30
        while True:
            owned_vm(name, cid)
            front = guest(FOREGROUND_SCRIPT)
            if front['process'] == 'Gedit':
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Gedit did not become foreground; no button invoked')
            time.sleep(.2)
        # Fixed diagnostic initialization only, separate from native invocation.
        # One edit enables Save, matching the failed real task's unsaved state.
        save(status='initializing-unsaved-document', setup_keyboard_events=1)
        initialized = guest("import json,pyautogui;pyautogui.write('QW3 native button qualification',interval=.02);print(json.dumps(dict(fixed_setup_edit=True)))")
        time.sleep(.3)
        front = guest(FOREGROUND_SCRIPT)
        source = atspi_button_script(foreground_tree_script(front['title']))
        (output / 'readonly-native-reader.py').write_text(source)
        tree = guest(source, seconds=6)
        if tree.get('status') != 'ok':
            raise RuntimeError('Unique foreground AT-SPI frame unavailable')
        (output / 'native-tree.xml').write_text(tree['AT'])
        nodes = []
        for node in ET.fromstring(tree['AT']).iter():
            a = attrs(node)
            if node.tag.rsplit('}', 1)[-1] != 'push-button':
                continue
            nodes.append(dict(name=a.get('name'), identity=provider_identity_from_attrs(a),
                              native_button_action=button_metadata_from_attrs(a), attrs=a))
        controls = controls_from_xml(tree['AT'], front, 1920, 1080)
        (output / 'original-controls.json').write_text(json.dumps(controls, ensure_ascii=False, indent=2)+'\n')
        save(status='passed', passed=True, foreground=front, setup_initialization=initialized,
             controls=controls, native_button_nodes=nodes,
             positive_native_save_buttons=[v for v in nodes if v['name']=='Save' and native_button(dict(v,kind='push-button',provider_identity=v['identity'])) is not None],
             action_invocations=0, model_requests=0)
        if len(report['positive_native_save_buttons']) != 1:
            raise RuntimeError('Exactly one positive native Save button was not established')
    except BaseException as error:
        save(status='failed', passed=False, error_kind=type(error).__name__, error=str(error))
    finally:
        if recorder is not None:
            try:
                save(recording=recorder.finish('continuous.mp4' if report.get('passed') else 'interrupted.mp4'),video_collection_passed=True)
            except Exception as error:
                save(recording_error=str(error),video_collection_passed=False,passed=False)
        if env is not None:
            try:
                owned_vm(name,cid)
                env.close()
                save(dedicated_vm_retired=subprocess.run(['docker','inspect',cid],capture_output=True,timeout=5).returncode!=0)
            except Exception as error:
                save(vm_cleanup_error=str(error),dedicated_vm_retired=False,passed=False)
        if lease is not None:
            lease.close()
        if 'seed_before' in locals():
            value=environment.DISK.stat()
            save(seed_metadata_unchanged=seed_before==(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns))
        save(finished_at=time.time())
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=run(a.output.resolve())
    print(json.dumps({k:result.get(k) for k in ('status','passed','action_invocations','model_requests','dedicated_vm_retired','video_collection_passed','error')}))
    if not result.get('passed'):
        raise SystemExit(1)
