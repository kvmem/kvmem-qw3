"""Install the selected qualified client after each clean snapshot restore.

The same snapshot can therefore run a frozen baseline or a frozen candidate;
installation verifies bytes and supplies no task answers or model instructions.
"""
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

from .source import sha256


def checked_path(name):
    if not isinstance(name, str):
        raise ValueError('Unsafe native bundle path')
    path = PurePosixPath(name)
    if (not isinstance(name, str) or not name or path.is_absolute() or '..' in path.parts
            or '\\' in name or ':' in name or str(path) != name):
        raise ValueError('Unsafe native bundle path')
    return path


def verify_bundle(bundle):
    bundle = Path(bundle)
    archive = json.loads((bundle / 'archive.json').read_text())
    qualification = json.loads((bundle / 'qualification.json').read_text())
    if not re.fullmatch(r'[a-f0-9]{64}', archive['sha256']):
        raise ValueError('Invalid bundle archive identity')
    native = qualification.get('native_result', {})
    if (qualification.get('passed') is not True or native.get('passed') is not True
            or native.get('process_exit_code') != 0
            or qualification.get('bundle_sha256') != archive['sha256']
            or native.get('bundle_sha256') != archive['sha256']):
        raise ValueError('Bundle has no matching successful Windows contract qualification')
    zip_path = Path(archive['path'])
    if zip_path.stat().st_size != archive['bytes'] or sha256(zip_path) != archive['sha256']:
        raise ValueError('Qualified bundle archive changed')
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('source_files'):
        raise ValueError('Bundle has no installed source inventory')
    with zipfile.ZipFile(zip_path) as reader:
        names = set()
        for entry in reader.infolist():
            name = entry.filename.rstrip('/')
            checked_path(name)
            if name.casefold() in names or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Duplicate or symlink native bundle member')
            names.add(name.casefold())
        if reader.read('manifest.json') != manifest_path.read_bytes():
            raise ValueError('Archived native manifest changed')
        import hashlib
        for name, digest in manifest['source_files'].items():
            checked_path(name)
            if (not re.fullmatch(r'[a-f0-9]{64}', digest)
                    or sha256(bundle / 'app' / name) != digest
                    or hashlib.sha256(reader.read('app/' + name)).hexdigest() != digest):
                raise ValueError('Native bundle source changed: ' + name)
    return dict(archive=archive, manifest_sha256=sha256(manifest_path),
                qualification_sha256=sha256(bundle / 'qualification.json'),
                source_files=len(manifest['source_files']))


def prepare(bundle, run, share, output):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,60}', run):
        raise ValueError('Invalid native installation run')
    evidence = verify_bundle(bundle)
    stage = Path(share) / (run + '-bundle')
    stage.mkdir(exist_ok=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    archive = evidence['archive']
    shutil.copyfile(archive['path'], stage / 'client.zip')
    if sha256(stage / 'client.zip') != archive['sha256']:
        raise ValueError('Staged bundle differs from qualified archive')
    script = r'''$ErrorActionPreference='Stop'
$stage='\\host.lan\Data\RUN-bundle'
$result=@{kind='frozen_native_bundle_install';passed=$false;bundle_sha256='ARCHIVE';manifest_sha256='MANIFEST';fresh_install=$false}
try {
 if ((Get-FileHash "$stage\client.zip" -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'ARCHIVE') {throw 'Staged bundle archive changed'}
 $destination='C:\Qw3Computer\clients\CLIENT'
 if (!(Test-Path $destination)) {
  New-Item -ItemType Directory $destination | Out-Null
  Expand-Archive -LiteralPath "$stage\client.zip" -DestinationPath $destination
  $result.fresh_install=$true
 }
 if ((Get-FileHash "$destination\manifest.json" -Algorithm SHA256).Hash.ToLowerInvariant() -cne 'MANIFEST') {throw 'Installed bundle manifest changed'}
 $manifest=Get-Content "$destination\manifest.json" -Raw | ConvertFrom-Json
 foreach($entry in $manifest.source_files.PSObject.Properties) {
  if ((Get-FileHash (Join-Path "$destination\app" $entry.Name) -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry.Value) {throw "Installed source changed: $($entry.Name)"}
 }
 $result.directory=$destination;$result.source_files=@($manifest.source_files.PSObject.Properties).Count
 $result.passed=$true
} catch {$result.error=$_.Exception.ToString()}
$result | ConvertTo-Json -Depth 5 | Set-Content '\\host.lan\Data\RUN-bundle-install.json' -Encoding UTF8
'''.replace('RUN', run).replace('ARCHIVE', archive['sha256']).replace('MANIFEST', evidence['manifest_sha256']).replace('CLIENT', archive['sha256'][:12])
    (output / 'install.ps1').write_text(script)
    (output / 'manifest.json').write_text(json.dumps(evidence, indent=2))
    return evidence
