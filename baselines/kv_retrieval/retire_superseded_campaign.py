"""Retire our stopped old scheduler after its current InfLLM arm exits.

Never signal the arm, model server, process group, or unrelated experiments.
The parent was stopped at the user's request to prevent further baseline reruns.
PID start times guard against PID reuse during a long active agent request.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[19]), fields[0]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--parent', type=int, required=True)
    parser.add_argument('--child', type=int, required=True)
    args = parser.parse_args()
    parent = identity(args.parent)
    child = identity(args.child)
    assert parent is not None and parent[1] == 'T', 'parent must already be stopped'
    assert child is not None
    command = Path(f'/proc/{args.parent}/cmdline').read_bytes()
    assert b'run_pilot_campaign.py' in command and str(args.campaign).encode() in command
    output = args.campaign / 'retirement.json'
    record = dict(status='waiting_for_current_infllm_arm', parent_pid=args.parent,
                  child_pid=args.child, parent_start_ticks=parent[0],
                  child_start_ticks=child[0], started=time.time(),
                  reason='User requests new InfLLM runs only, aligned to existing long-running agent experiments')

    def save():
        output.write_text(json.dumps(record, indent=2) + '\n')

    save()
    while True:
        now_parent, now_child = identity(args.parent), identity(args.child)
        if now_parent is None or now_parent[0] != parent[0]:
            record['status'] = 'parent_already_exited'
            break
        if now_child is None or now_child[0] != child[0] or now_child[1] == 'Z':
            # SIGTERM is pending while stopped. CONT lets the installed handler
            # cleanly exit; the completed child can no longer be terminated.
            os.kill(args.parent, signal.SIGTERM)
            os.kill(args.parent, signal.SIGCONT)
            record['status'] = 'retirement_signal_sent_after_arm_exit'
            break
        assert now_parent[1] == 'T', 'old scheduler resumed unexpectedly'
        time.sleep(10)
    record['finished'] = time.time()
    save()


if __name__ == '__main__':
    main()
