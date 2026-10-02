"""Prepare a real missing-class fault after successful managed LaTeX compilation.

This scripted prerequisite is never an agent repair trial. Retain every failed
setup/reset and refuse another mutation when prior work is unresolved.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import time

from qw3_computer.software import Software, process_environment, sha256
from qw3_computer.software_latex import FLAGS


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def tree(path):
    return {str(p): sha256(p) for p in path.rglob('*') if p.is_file()}


def probe(software, name, output):
    output.mkdir(parents=True, exist_ok=False)
    compiler = software.path('latex', name, 'texmfs', 'install', 'miktex', 'bin', 'x64', 'xelatex.exe')
    result = subprocess.run([str(compiler), *FLAGS, '--output-directory=' + str(output), 'main.tex'],
        cwd=software.path('projects', name), env=process_environment(), stdin=subprocess.DEVNULL,
        capture_output=True, timeout=180)
    (output / 'stdout.log').write_bytes(result.stdout)
    (output / 'stderr.log').write_bytes(result.stderr)
    pdf = output / 'main.pdf'
    text = result.stdout.decode('utf-8', errors='replace')
    return {'exit_code': result.returncode, 'pdf_present': pdf.is_file(),
            'pdf_sha256': sha256(pdf) if pdf.is_file() else None,
            'missing_class': "File `ctexart.cls' not found" in text, 'output': str(output)}


def ctex_installed(software, output):
    executable = software.path('latex', 'latex-repair', 'texmfs', 'install', 'miktex', 'bin', 'x64', 'miktex.exe')
    result = subprocess.run([str(executable), 'packages', 'list', '--template', '{id}|{isInstalled}\n'],
        cwd=software.path('projects', 'latex-repair'), env=process_environment(), stdin=subprocess.DEVNULL,
        capture_output=True, timeout=60)
    output.write_bytes(result.stdout + result.stderr)
    states = result.stdout.decode('utf-8', errors='replace').splitlines()
    return result.returncode == 0 and 'ctex|true' in states


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.reset_from.read_text(encoding='utf-8-sig')) if args.reset_from else None
    if original:
        if original['status'] != 'ready_fault_injected' or Path(original['workspace']) != args.workspace:
            raise ValueError('Reset belongs to a different LaTeX fixture workspace')
    elif args.workspace.exists():
        raise ValueError('LaTeX repair fixture workspace already exists')
    software = Software(args.workspace)
    report = {'kind': 'scripted_latex_repair_fixture', 'model_used': False, 'agent_success_claimed': False,
              'status': 'preparing', 'started_at': time.time(), 'jobs': [], 'workspace': str(args.workspace),
              'guest_pid': os.getpid(), 'fixture_source_sha256': sha256(Path(__file__))}
    save(args.output / 'result.json', report)
    try:
        if any(j['status'] not in {'completed', 'failed', 'cancelled'} for j in software.jobs.values()):
            raise ValueError('Previous software action is unresolved')
        if original:
            report['reset_from_sha256'] = sha256(args.reset_from)
            if original['unrelated_files'] != tree(software.path('projects', 'unrelated-latex')):
                raise ValueError('Unrelated project differs from the frozen input')
            if any(sha256(software.path('projects', 'latex-repair', n)) != value
                   for n, value in original['source_sha256'].items()):
                raise ValueError('Source differs from the frozen input')
        else:
            for name in ('latex-repair', 'unrelated-latex'):
                job = await software.invoke('fixture-setup', {'operation': 'setup_latex',
                    'action_id': 'fixture-' + name, 'project': name, 'profile': 'latex-chinese-math'})
                await software.tasks[job['id']]
                state = software.jobs[job['id']]; report['jobs'].append(state.copy())
                save(args.output / 'result.json', report)
                if state['status'] != 'completed' or not state.get('verification', {}).get('passed'):
                    raise ValueError('Healthy prerequisite setup failed; no fault injected')
        target = software.path('projects', 'latex-repair')
        reference = software.path('projects', 'unrelated-latex')
        report.update(target_project=str(target), reference_project=str(reference),
            source_sha256={'main.tex': sha256(target / 'main.tex')}, unrelated_files=tree(reference))
        for name, label in (('latex-repair', 'target'), ('unrelated-latex', 'reference')):
            facts = probe(software, name, args.output / (label + '-healthy'))
            report[label + '_healthy_probe'] = facts
            if facts['exit_code'] != 0 or not facts['pdf_present']:
                raise ValueError('Both projects must compile before fault injection')
        candidates = list(software.path('latex', 'latex-repair', 'texmfs', 'install', 'tex').rglob('ctexart.cls'))
        if len(candidates) != 1:
            raise ValueError('Expected exactly one installed CTeX article class')
        broken = software.path(*candidates[0].relative_to(software.root).parts)
        original_sha = sha256(broken)
        if not ctex_installed(software, args.output / 'package-state-before.log'):
            raise ValueError('CTeX package metadata is absent before fault injection')
        if original and original_sha != original['fault']['original_sha256']:
            raise ValueError('Restored class differs from the original fault input')
        backup = software.path('fixture-backups', args.output.name)
        backup.mkdir(parents=True, exist_ok=False)
        broken.rename(backup / 'ctexart.cls')
        report['fault'] = {'kind': 'missing_class_with_package_metadata_present', 'file': str(broken),
                           'original_sha256': original_sha, 'backup': str(backup / 'ctexart.cls')}
        report['fault']['package_metadata_preserved'] = ctex_installed(software, args.output / 'package-state-after.log')
        if not report['fault']['package_metadata_preserved']:
            raise ValueError('CTeX package metadata changed during fault injection')
        report['target_broken_probe'] = probe(software, 'latex-repair', args.output / 'target-broken')
        report['reference_after_injection_probe'] = probe(software, 'unrelated-latex', args.output / 'reference-after')
        failed = report['target_broken_probe']; healthy = report['reference_after_injection_probe']
        if failed['exit_code'] == 0 or failed['pdf_present'] or not failed['missing_class']:
            raise ValueError('Injected failure is not the intended missing class')
        if healthy['exit_code'] != 0 or not healthy['pdf_present']:
            raise ValueError('Unrelated project no longer compiles')
        if report['unrelated_files'] != tree(reference) or sha256(target / 'main.tex') != report['source_sha256']['main.tex']:
            raise ValueError('Fault preparation changed project files')
        report['status'] = 'ready_fault_injected'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        await software.close()
        report['finished_at'] = time.time()
        save(args.output / 'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reset-from', type=Path)
    result = asyncio.run(run(parser.parse_args()))
    print(json.dumps({'status': result['status']}))
    raise SystemExit(0 if result['status'] == 'ready_fault_injected' else 1)
