#!/usr/bin/env python3
"""Finish existing CPU preparation, then inference, grading, and CSV export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_qwen35_longmemeval import OUT, ROOT, environment, save


def process_identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-pid', required=True, type=int)
    args = parser.parse_args()
    status_path = OUT / 'campaign_status.json'
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        assert process_identity(previous['pid']) != previous.get('process_identity'), 'Campaign already running'
    state = {'pid': os.getpid(), 'process_identity': process_identity(os.getpid()),
             'started_at': time.time(), 'status': 'waiting_for_cpu_preparation'}
    save(status_path, state)
    try:
        preparation = json.loads((OUT / 'status.json').read_text())
        assert preparation['pid'] == args.prepare_pid
        identity = process_identity(args.prepare_pid)
        deadline = time.monotonic() + 3600
        while identity is not None and process_identity(args.prepare_pid) == identity:
            preparation = json.loads((OUT / 'status.json').read_text())
            if preparation['status'] in ('prepared', 'needs_attention'):
                break
            assert time.monotonic() < deadline, 'Preparation exceeded one hour'
            time.sleep(5)
        # Prepared status is written only after the embedding worker has closed.
        preparation = json.loads((OUT / 'status.json').read_text())
        assert preparation['status'] == 'prepared', preparation
        for phase, script in (
            ('inference', 'run_qwen35_longmemeval.py'),
            ('grading', 'score_qwen35_longmemeval.py'),
            ('export', 'report_qwen35_longmemeval.py'),
        ):
            state.update(status=phase, phase_started_at=time.time())
            save(status_path, state)
            with (OUT / f'campaign_{phase}.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name(script))],
                                         cwd=ROOT, env=environment(), stdout=log, stderr=log)
                state.update(child_pid=child.pid)
                save(status_path, state)
                code = child.wait()
            assert code == 0, f'{phase} exited {code}; inspect campaign_{phase}.log'
        report = json.loads((OUT / 'report' / 'summary.json').read_text())
        planned = len(json.loads((OUT / 'inference_samples.json').read_text()))
        assert report['complete'] and report['scored_pairs'] == planned
        state.update(status='complete', finished_at=time.time())
    except BaseException as error:
        state.update(status='needs_attention', error=repr(error), finished_at=time.time())
        raise
    finally:
        save(status_path, state)


if __name__ == '__main__':
    main()
