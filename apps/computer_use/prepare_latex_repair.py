"""Prepare/reset independent LaTeX missing-package inputs in the owned, idle Windows VM."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from osworld_windows.bundle import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reset-from', type=Path)
    args = parser.parse_args()
    app = Path(__file__).resolve().parent
    share = app / '.runtime/windows/shared'
    output = args.output.resolve(); identity = output.name
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,40}', identity):
        raise ValueError('Use a new short fixture identity')
    archive = verify_bundle(args.bundle)['archive']
    output.mkdir(parents=True, exist_ok=False)
    with open('/tmp/qw3-osworld-vm.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        vm = json.loads(subprocess.check_output(['docker', 'inspect', 'qw3-computer-windows']))[0]
        if vm['Config']['Labels'].get('com.qw3.computer-use.owner') != 'codex-goal-20260930':
            raise ValueError('VM ownership mismatch')
        token = (share / 'executor-token.txt').read_text().strip()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(urllib.request.Request('http://127.0.0.1:18082/health', headers={'Authorization': 'Bearer '+token}), timeout=5) as response:
            if not json.load(response)['paused']:
                raise ValueError('VM is not idle')
        source = app / 'latex_repair_fixture_guest.py'
        staged = share / (identity+'.py')
        shutil.copyfile(source, staged); shutil.copyfile(source, output / source.name)
        workspace = 'C:\\Qw3Computer\\'+identity
        reset_arg = ''
        if args.reset_from:
            original = json.loads(args.reset_from.read_text())
            workspace = original['workspace']
            if (not PureWindowsPath(workspace).is_relative_to('C:/Qw3Computer')
                    or not re.fullmatch(r'C:\\Qw3Computer\\[a-zA-Z0-9_-]{8,60}', workspace)):
                raise ValueError('Unexpected fixture workspace')
            shutil.copyfile(args.reset_from, share / (identity+'-original.json'))
            reset_arg = " --reset-from '\\\\host.lan\\Data\\"+identity+"-original.json'"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        script = f'''$ErrorActionPreference='Stop'
$source='\\\\host.lan\\Data\\{identity}.py'
if ((Get-FileHash $source -Algorithm SHA256).Hash.ToLowerInvariant() -cne '{source_hash}') {{throw 'Fixture source changed'}}
& 'C:\\Qw3Computer\\clients\\{archive['sha256'][:12]}\\python\\python.exe' -X utf8 $source --workspace '{workspace}' --output '\\\\host.lan\\Data\\{identity}-result'{reset_arg} *> '\\\\host.lan\\Data\\{identity}.log'
@{{exit_code=$LASTEXITCODE;finished=[DateTime]::UtcNow.ToString('o')}} | ConvertTo-Json | Set-Content '\\\\host.lan\\Data\\{identity}-exit.json' -Encoding UTF8
'''
        (output / 'prepare.ps1').write_text(script)
        subprocess.run([sys.executable, str(app / 'vm/provision.py'), str(output / 'prepare.ps1')], check=True)
        deadline = time.monotonic()+1200
        while not (share / (identity+'-exit.json')).exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Fixture worker unresolved; inspect before another attempt')
            time.sleep(1)
        receipt = json.loads((share / (identity+'-exit.json')).read_text(encoding='utf-8-sig'))
        result = json.loads((share / (identity+'-result/result.json')).read_text())
        result.update(process_exit=receipt, fixture_source_sha256=source_hash, bundle_sha256=archive['sha256'])
        (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({'status': result['status'], 'exit': receipt['exit_code']}))
        if result['status'] != 'ready_fault_injected' or receipt['exit_code'] != 0:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
