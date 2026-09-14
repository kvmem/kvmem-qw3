"""Wait for an existing campaign, then build a fully audited review snapshot.

This never launches/retries collection or marks the research goal complete.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def process_identity(pid):
    try:
        # The command name may itself contain spaces or parentheses.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    script = Path(__file__).resolve().parent
    state_path = args.campaign / 'finalization.json'
    state = {'started_at': time.time(), 'status': 'waiting',
             'wait_pid': args.wait_pid, 'completed_commands': [],
             'human_review_required': True, 'goal_complete': False}
    # Do not silently replace an existing attempt or run two finalizers.
    with state_path.open('x') as stream:
        json.dump(state, stream, indent=2)

    def save():
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)

    try:
        if args.wait_pid:
            identity = process_identity(args.wait_pid)
            state['process_start_ticks'] = identity
            save()
            while identity is not None and process_identity(args.wait_pid) == identity:
                time.sleep(30)
        campaign = json.loads((args.campaign / 'campaign.json').read_text())
        if campaign['status'] != 'finished':
            raise RuntimeError('Campaign did not finish successfully; no automatic retry')
        state['status'] = 'auditing_and_analyzing'
        save()
        common = ['--manifest', str(args.manifest), '--measurements',
                  str(args.campaign / 'measurements')]
        commands = [
            ['audit_source_units.py', '--manifest', str(args.manifest), '--out',
             str(args.campaign / 'source_units_audit.json')],
            ['audit_new.py', *common, '--out', str(args.campaign / 'integrity_audit.json'),
             '--require-complete'],
            ['analyze_new.py', *common, '--out', str(args.campaign / 'analysis')],
            ['boundary_support_controls.py', '--analysis',
             str(args.campaign / 'analysis/analysis.json'), '--out', str(args.campaign / 'analysis')],
            ['schedule_diagnostics.py', *common, '--out', str(args.campaign / 'analysis')],
            ['write_report.py', '--campaign', str(args.campaign), '--manifest',
             str(args.manifest), '--legacy', str(args.legacy)],
        ]
        manifest = json.loads(args.manifest.read_text())
        auxiliary_ids = manifest.get('cohort_amendment', {}).get('auxiliary_replicates', [])
        if auxiliary_ids:
            auxiliary = args.campaign / 'auxiliary_manifest.json'
            aux = json.loads(auxiliary.read_text())
            assert {r['id'] for r in aux['trajectories']} == set(auxiliary_ids)
            assert not set(auxiliary_ids) & {r['id'] for r in manifest['trajectories']}
            commands[2:2] = [
                ['audit_source_units.py', '--manifest', str(auxiliary), '--out',
                 str(args.campaign / 'auxiliary_source_units_audit.json')],
                ['audit_new.py', '--manifest', str(auxiliary), '--measurements',
                 str(args.campaign / 'auxiliary_measurements'), '--out',
                 str(args.campaign / 'auxiliary_integrity_audit.json'), '--require-complete'],
            ]
            commands.insert(-1, ['compare_auxiliary.py', '--campaign', str(args.campaign)])
            commands.insert(-1, ['summarize_resources.py', '--campaign', str(args.campaign)])
        commands.append(['write_paper_draft.py', '--campaign', str(args.campaign),
                         '--manifest', str(args.manifest)])
        for name, *options in commands:
            subprocess.run([sys.executable, str(script / name), *options], check=True)
            state['completed_commands'].append(name)
            save()
        archive = args.campaign / 'final_analysis_snapshot'
        archive.mkdir(exist_ok=False)
        for source in sorted(script.glob('*.py')):
            shutil.copy2(source, archive / source.name)
        shutil.copy2(args.manifest, archive / 'manifest.json')
        if auxiliary_ids:
            shutil.copy2(auxiliary, archive / 'auxiliary_manifest.json')
        for name, reference in manifest.get('collection_manifests', {}).items():
            source = Path(reference['path'])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == reference['sha256']
            shutil.copy2(source, archive / f'collection_manifest_{name}.json')
        digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(archive.iterdir()) if p.is_file()}
        (archive / 'sha256.json').write_text(json.dumps(digests, indent=2) + '\n')
        state.update(status='ready_for_human_review', finished_at=time.time(),
                     snapshot=str(archive))
        save()
    except BaseException as error:
        state.update(status='failed', error=str(error), finished_at=time.time())
        save()
        raise


if __name__ == '__main__':
    main()
