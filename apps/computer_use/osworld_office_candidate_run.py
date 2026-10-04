"""Qualify auxiliary read-only Office adapters before the unchanged runner."""
import hashlib
import json
from unittest.mock import patch

import osworld_candidate_run as base


def verify_auxiliary(args):
    manifest=json.loads(args.acceptance_manifest.read_text())
    files=manifest.get('framework_auxiliary_files')
    if not isinstance(files,dict) or set(files)!={'bounded_linux_observation.py','linux_app_semantics.py'}:
        raise ValueError('Declare the exact two read-only observation helpers')
    framework=args.framework.resolve()
    for name,expected in files.items():
        source=(framework/name).resolve()
        if not source.is_relative_to(framework) or not source.is_file():
            raise ValueError('Auxiliary observation source is missing or outside the frozen framework')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen auxiliary observation source changed: '+name)


def main():
    original=base.admit
    def admit(args):
        verify_auxiliary(args)
        return original(args)
    # Separate entry and in-process qualification only. The frozen task
    # runner, initializer, evaluator and comparison budgets are untouched.
    with patch.object(base,'admit',admit):base.main()


if __name__=='__main__':main()
