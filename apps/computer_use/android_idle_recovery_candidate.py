"""Add opt-in recovery to a full immutable framework; no existing file edits."""
import argparse
import json
from pathlib import Path
import shutil
import time

import kvmem_osworld_queue as owned

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / 'apps/computer_use/.runtime'
BASE_SHA = 'a07266aed63d31f470aa2b5fbcd60fc8def54a519b907e1eeb8fb1a4c8343c8a'


def inventory(framework):
    return {str(path.relative_to(framework)): owned.digest(path) for path in framework.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts}


def build(args):
    base_path = args.base.resolve(); base = json.loads(base_path.read_text()); framework = base_path.parent / 'framework'
    if owned.digest(base_path) != BASE_SHA or inventory(framework) != base['candidate_files_sha256']:
        raise ValueError('Require the unchanged full native-button framework')
    qualification = json.loads(args.cpu.read_text())
    if not qualification.get('passed') or qualification.get('tests_run') != 14:
        raise ValueError('Require the final recovery qualification')
    for name, digest in qualification['source_sha256'].items():
        if owned.digest(ROOT / 'apps/computer_use' / name) != digest:
            raise ValueError('Qualified recovery source changed: ' + name)
        if name.startswith('qw3_computer/') and base['candidate_files_sha256'].get(name) != digest:
            raise ValueError('Qualified gateway dependency differs from the original framework')
    output = args.output.resolve()
    if not output.is_relative_to(RUNTIME):
        raise ValueError('Use a fresh goal-owned candidate')
    output.mkdir(mode=0o700, exist_ok=False)
    target = output / 'framework'
    shutil.copytree(framework, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    addition = target / 'android_idle_recovery.py'
    if addition.exists():
        raise ValueError('Original framework already contains the opt-in module')
    shutil.copyfile(ROOT / 'apps/computer_use/android_idle_recovery.py', addition)
    after = inventory(target)
    if after != base['candidate_files_sha256'] | {'android_idle_recovery.py': owned.digest(addition)}:
        raise ValueError('An existing framework file changed')
    record = {'kind': 'full_framework_with_opt_in_android_idle_recovery', 'prepared_at': time.time(),
              'base_build': {'path': str(base_path), 'sha256': BASE_SHA},
              'cpu_qualification': {'path': str(args.cpu.resolve()), 'sha256': owned.digest(args.cpu)},
              'added_files': ['android_idle_recovery.py'], 'changed_existing_files': [],
              'candidate_files_sha256': after, 'production_installed': False,
              'original_queue_or_client_changed': False, 'model_requests': 0, 'device_inputs': 0,
              'entry': 'Opt-in android_idle_recovery.serve(args, expected_identity, read_identity). Existing default gateway is unchanged.'}
    (output / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'cpu', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    result = build(parser.parse_args())
    print(json.dumps({key: result[key] for key in ('added_files', 'changed_existing_files', 'production_installed')}))
