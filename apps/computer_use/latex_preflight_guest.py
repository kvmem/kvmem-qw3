"""Scripted MiKTeX command qualification; never counted as a model product trial."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import winreg
import zipfile

from qw3_computer.software import process_environment


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def paths():
    result = {}
    for label, hive, key in [('user', winreg.HKEY_CURRENT_USER, 'Environment'),
                             ('machine', winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment')]:
        with winreg.OpenKey(hive, key) as handle:
            try:
                result[label] = winreg.QueryValueEx(handle, 'Path')
            except FileNotFoundError:
                result[label] = None
    return result


async def run(args):
    args.workspace.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'kind': 'scripted_latex_command_qualification', 'model_used': False,
              'counted_as_product_trial': False, 'mode': args.mode,
              'status': 'preparing', 'started_at': time.time(), 'commands': []}

    async def command(label, argv, timeout=90):
        receipt = {'label': label, 'argv': list(map(str, argv)), 'started_at': time.time()}
        report['commands'].append(receipt)
        with (args.output / (label+'.stdout.log')).open('wb') as stdout, (args.output / (label+'.stderr.log')).open('wb') as stderr:
            process = await asyncio.create_subprocess_exec(*map(str, argv), cwd=args.workspace, env=process_environment(),
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            receipt['pid'] = process.pid; save(args.output/'result.json', report)
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except BaseException:
                # A timeout is unresolved installation, not proof of rollback.
                report['needs_reconciliation'] = True
                killer = await asyncio.create_subprocess_exec(str(Path(os.environ['SYSTEMROOT'])/'System32/taskkill.exe'),
                    '/PID', str(process.pid), '/T', '/F', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                await asyncio.wait_for(killer.wait(), 15)
                receipt['termination_exit_code'] = killer.returncode
                await asyncio.wait_for(process.wait(), 15)
                raise
            finally:
                receipt.update(exit_code=process.returncode, finished_at=time.time())
                save(args.output/'result.json', report)
        if process.returncode != 0:
            raise RuntimeError(f'{label} exited with {process.returncode}')
        return receipt

    try:
        manifest = json.loads((args.assets/'sources.json').read_text())
        report['assets'] = []
        for asset in manifest['assets']:
            source = args.assets/asset['filename']
            if source.name != asset['filename'] or digest(source) != asset['sha256']:
                raise ValueError('Official asset hash mismatch')
            target = args.workspace/source.name
            shutil.copyfile(source, target)
            if digest(target) != asset['sha256']:
                raise ValueError('Staged asset changed')
            report['assets'].append({'file': source.name, 'sha256': asset['sha256'], 'bytes': source.stat().st_size})
        archive = args.workspace/'miktexsetup-5.5.0+1763023-x64.zip'
        with zipfile.ZipFile(archive) as reader:
            if reader.namelist() != ['miktexsetup_standalone.exe']:
                raise ValueError('Unexpected setup archive contents')
            reader.extractall(args.workspace)
        before = paths(); report['registry_paths_before'] = before
        report['candidate_fonts'] = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in (Path(os.environ['SYSTEMROOT'])/'Fonts').iterdir()
            if p.is_file() and any(n in p.name.lower() for n in ('simsun', 'simhei', 'msyh', 'noto', 'deng', 'simfang', 'simkai'))]
        setup = args.workspace/'miktexsetup_standalone.exe'
        basic = args.workspace/'basic-miktex-25.12-x64.exe'
        powershell = Path(os.environ['SYSTEMROOT'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
        metadata = args.workspace/'metadata.ps1'
        metadata.write_text("$ErrorActionPreference='Stop'\nGet-ChildItem -LiteralPath $PSScriptRoot -Filter '*.exe' | ForEach-Object {\n"
            "$s=Get-AuthenticodeSignature $_.FullName; [PSCustomObject]@{file=$_.Name;version=$_.VersionInfo.FileVersion;"
            "signature=[string]$s.Status;signer=$s.SignerCertificate.Subject;thumbprint=$s.SignerCertificate.Thumbprint}\n"
            "} | ConvertTo-Json -Depth 4\n", encoding='utf-8')
        await command('asset-metadata', [powershell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', metadata])
        await command('setup-version', [setup, '--version'])
        await command('setup-help', [setup, '--help'])
        install = args.workspace/'portable'
        # The GUI setup parser accepts --portable[=DIR]. Portable and the
        # --user-* roots are mutually exclusive, even in unattended mode.
        # An elevated VM console otherwise inherits the installer's shared
        # default, which conflicts with portable mode during component setup.
        argv = [basic, '--unattended', '--private', '--portable='+str(install),
                '--no-registry', '--no-additional-roots']
        if args.mode == 'dry-run':
            argv.append('--dry-run')
        await command('basic-'+args.mode, argv, timeout=600 if args.mode == 'install' else 120)
        if args.mode == 'install':
            bins = install/'texmfs/install/miktex/bin/x64'
            await command('xelatex-version', [bins/'xelatex.exe', '--version'])
            await command('miktex-version', [bins/'miktex.exe', '--version'])
        report['registry_paths_after'] = paths()
        report['path_unchanged'] = before == report['registry_paths_after']
        report['workspace_files'] = [str(p.relative_to(args.workspace)) for p in args.workspace.rglob('*') if p.is_file()]
        report['install_root_absent'] = not install.exists()
        report.update(status='complete', passed=all(c['exit_code']==0 for c in report['commands'])
                      and report['path_unchanged'] and (args.mode=='install' or report['install_root_absent']))
    except Exception as exc:
        report.update(status='failed', passed=False, error=f'{type(exc).__name__}: {exc}')
    finally:
        if 'registry_paths_before' in report:
            report['registry_paths_after'] = paths()
            report['path_unchanged'] = report['registry_paths_before'] == report['registry_paths_after']
        report['finished_at'] = time.time(); save(args.output/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['dry-run', 'install'], default='dry-run',
                        help='dry-run is checked for writes: the tested Basic installer still extracts packages')
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({k:result.get(k) for k in ('status','passed','error')}))
    raise SystemExit(0 if result.get('passed') else 1)
