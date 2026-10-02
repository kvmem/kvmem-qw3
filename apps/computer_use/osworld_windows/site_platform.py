"""Verify the qualified private-site deployment before a Windows attempt.

Image tags can move, and using the base compose file can silently restore old
frontends. Bind trials to actual image IDs, routing mounts and native evidence.
"""
import json
from pathlib import Path

from .source import ROOT, sha256

PLATFORM = ROOT / 'apps/computer_use/osworld_windows/offline_site_platform.json'


def verify(containers, manifest=PLATFORM, *, root=ROOT):
    manifest, root = Path(manifest), Path(root)
    frozen = json.loads(manifest.read_text())
    if frozen.get('kind') != 'qualified_windows_private_site_platform' or frozen.get('qualified') is not True:
        raise ValueError('Private-site platform is not qualified')
    for relative, digest in frozen['files'].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != digest:
            raise ValueError('Qualified site input or evidence changed: '+relative)
    actual = {c['Name'].lstrip('/'): c for c in containers}
    if len(actual) != len(containers) or set(actual) != set(frozen['containers']):
        raise ValueError('Private-site container set differs from qualification')
    for name, expected in frozen['containers'].items():
        value = actual[name]
        if value['Image'] != expected['image'] or value['State']['Running'] is not True:
            raise ValueError('Private-site image changed or stopped: '+name)
        labels = value['Config'].get('Labels') or {}
        if (labels.get('com.docker.compose.project') != 'qw3-osworld-windows'
                or labels.get('com.docker.compose.service') != expected['service']):
            raise ValueError('Private-site container identity differs: '+name)
        mounts = sorted([dict(source=m['Source'], destination=m['Destination'], rw=m['RW'])
                         for m in value['Mounts'] if m['Type'] == 'bind'], key=lambda m: m['destination'])
        if mounts != expected['bind_mounts']:
            raise ValueError('Private-site routing or mounted inputs differ: '+name)
    evidence = json.loads((root / frozen['windows_qualification']).read_text())
    if (evidence.get('passed') is not True or evidence.get('final_clean_restore') is not True
            or evidence.get('render', {}).get('passed') is not True):
        raise ValueError('Windows offline rendering qualification is incomplete')
    return dict(manifest=str(manifest), sha256=sha256(manifest),
                container_images={name: v['image'] for name, v in frozen['containers'].items()},
                windows_qualification=frozen['windows_qualification'], qualified=True,
                scope='Common test platform; not an agent strategy optimization or model success')


async def qualify_runtime(command, manifest=PLATFORM):
    frozen = json.loads(Path(manifest).read_text())
    names = list(frozen['containers'])
    containers = json.loads(await command('docker', 'inspect', *names))
    return verify(containers, manifest)
