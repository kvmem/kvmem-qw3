#!/usr/bin/env python3
"""Queue the authorized 30-minute benchmark check in the existing Codex task."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/home/chaidi/qw3')
STATE = ROOT / 'benchmark/memory_benchmark_monitor_20260915'
TIMER = 'qw3-memory-benchmark-monitor.timer'


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
        # The shared app server serializes queued turns for this existing task.
        # Do not create another session or start an independent GPU worker here.
        result = subprocess.run([
            '/home/chaidi/.local/bin/codex', 'queue',
            '--thread', config['thread_id'], '--message', prompt,
        ], cwd=ROOT, capture_output=True, text=True, timeout=90)
        record = {
            'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'thread_id': config['thread_id'],
            'dispatch_exit_code': result.returncode,
            'dispatch_stdout': result.stdout[-4000:],
            'dispatch_stderr': result.stderr[-4000:],
        }
        tmp = STATE / 'last_dispatch.tmp'
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        os.replace(tmp, STATE / 'last_dispatch.json')
        if result.returncode:
            raise RuntimeError('Codex queue failed; inspect last_dispatch.json')


if __name__ == '__main__':
    main()
