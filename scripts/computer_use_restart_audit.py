#!/usr/bin/env python3
"""Audit retained Windows native crash/resume evidence without rerunning tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def read(path):
    return json.loads(path.read_text())


def audit_restart(run, binary_sha256):
    summary = read(run / 'summary.json')
    assert summary['status'] == 'completed' and summary['passed'] and len(summary['trials']) == 2
    assert {t['kvmem'] for t in summary['trials']} == {'on', 'off'}
    checks = []
    for trial in summary['trials']:
        native = trial['native_result']
        task = native['task']
        assert trial['passed'] and native['driver_exit_code'] == 0 and task['passed']
        # These are deliberately killed app return codes, not taskkill's return code.
        assert task['crash_exit_code'] == task['cleanup_exit_code'] == 1
        assert not task.get('error') and not task.get('cleanup_error')
        assert task['before_crash']['id'] == task['restored']['id'] == task['task_id']
        assert task['before_crash']['thread_id'] and task['before_crash']['thread_id'] == task['restored']['thread_id']
        assert task['restored']['status'] == 'paused' and task['restored']['desktop_paused'] and task['paused_without_actions_seconds'] >= 2
        assert task['first-start_pid'] != task['restart_pid']
        actual = task['verified_file']
        assert actual == trial['verified_file'] and actual in task['state']['files']
        raw = actual['content'].encode()
        assert len(raw) == actual['size'] and hashlib.sha256(raw).hexdigest() == actual['sha256']
        directory = Path(trial['directory'])
        expected = summary['task_definitions'][trial['case']]
        assert actual['name'] == expected['filename'] and actual['content'] == expected['content']
        assert read(directory / 'guest/restart/result.json') == task
        assert native['bundle_sha256'] == summary['bundle_sha256']
        for label, pid in [('crash', task['first-start_pid']), ('cleanup', task['restart_pid'])]:
            lines = (directory / 'guest/restart' / (label + '-taskkill.log')).read_text().splitlines()
            assert lines and all(line.startswith('SUCCESS:') for line in lines)
            assert any(f'with PID {pid} ' in line for line in lines)
        with sqlite3.connect((directory / 'guest/restart/app/desktop/desktop.sqlite').as_uri() + '?mode=ro', uri=True) as db:
            rows = list(db.execute('SELECT request,status FROM actions'))
        assert rows and not any(status in ('dispatching', 'uncertain') for _, status in rows)
        actions = [json.loads(body)['action'] for body, status in rows if status == 'completed']
        assert {'kind': 'key', 'key': 'CTRL+C'} in actions and {'kind': 'key', 'key': 'CTRL+V'} in actions
        assert not any(a.get('kind') == 'type' and actual['content'] in a.get('text', '') for a in actions)
        engine = read(directory / 'engine/manifest.json')
        assert engine['guided_refresh'] == ('both' if trial['kvmem'] == 'on' else 'off') and engine['binary_sha256'] == binary_sha256
        speculation = re.findall(r'native dflash_spec_summary: enabled=true .*?drafted=(\d+) accepted=(\d+).*?fallback=(\w+)', (directory / 'engine/engine.log').read_text())
        assert speculation and sum(int(x[1]) for x in speculation) > 0 and all(x[2] == 'false' for x in speculation)
        checks.append(dict(kvmem=trial['kvmem'], passed=True, seconds=task['elapsed_seconds'], file_sha256=actual['sha256'], completed_actions=len(actions)))
    result = dict(kind='native_restart_retained_evidence_audit', passed=True, checks=checks)
    result['source_summary_sha256'] = hashlib.sha256((run / 'summary.json').read_bytes()).hexdigest()
    result['termination_scope'] = 'The driver checks taskkill exit zero; crash/cleanup_exit_code 1 records deliberately terminated application processes. Successful retained taskkill logs and exact root PIDs are independently checked.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_restart(args.run, args.binary_sha256)
    result['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
