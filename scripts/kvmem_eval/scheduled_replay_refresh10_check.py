#!/usr/bin/env python3
"""Dispatch the user-authorized half-hour check to the existing Codex task."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/data/chaidi/qw3-experiments/replay_refresh_agent10_20260922')
STATE = ROOT / 'monitor'
TIMER = 'qw3-replay-refresh10-monitor.timer'

def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)

def main():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'dispatch.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (STATE / 'all_complete.json').exists():
            subprocess.run(['systemctl', '--user', 'disable', '--now', TIMER], check=True)
            return
        if (STATE / 'paused').exists():
            return
        config = json.loads((STATE / 'config.json').read_text())
        prompt = (STATE / 'check_prompt.txt').read_text()
        result = subprocess.run([
            '/home/chaidi/.local/bin/codex', 'queue', '--thread', config['thread_id'],
            '--message', prompt,
        ], cwd='/home/chaidi/qw3', capture_output=True, text=True, timeout=90)
        record = dict(dispatched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      thread_id=config['thread_id'], exit_code=result.returncode,
                      stdout=result.stdout[-4000:], stderr=result.stderr[-4000:])
        save(STATE / 'last_dispatch.json', record)
        with (STATE / 'dispatches.jsonl').open('a') as history:
            history.write(json.dumps(record, ensure_ascii=False) + '\n')
        if result.returncode:
            raise RuntimeError('Codex queue failed; inspect monitor/last_dispatch.json')

if __name__ == '__main__':
    main()
