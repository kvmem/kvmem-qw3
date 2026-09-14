"""Sample GPU process utilization outside the frozen model/request path.

Starts partway through the first trial: missing earlier samples are unknown,
never zero. SM utilization is a sampled activity metric, not kernel occupancy
or a correction factor for shared-GPU timings.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from observe_window_campaign import process_identity


def parse_pmon(text):
    columns = None
    rows = []
    for line in text.splitlines():
        if line.lstrip().startswith('# gpu '):
            columns = line.lstrip()[1:].split()
        elif line.strip() and not line.lstrip().startswith('#'):
            if columns is None:
                raise ValueError('pmon data has no column header')
            parts = line.split(maxsplit=len(columns)-1)
            if len(parts) != len(columns):
                raise ValueError('pmon column mismatch')
            row = {}
            for key, value in zip(columns, parts):
                row[key] = None if value == '-' else (int(value) if value.isdigit() else value)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    identity = process_identity(args.pid)
    if not identity or 'run_window_campaign.py' not in identity['command'] or str(campaign) not in identity['command']:
        raise ValueError('expected the verified live campaign process')
    started = identity['starttime_ticks']
    output = campaign/'gpu_process_utilization.jsonl'
    while True:
        live = process_identity(args.pid)
        if not live or live['state'] == 'Z' or live['starttime_ticks'] != started:
            return
        state = json.loads((campaign/'state.json').read_text())
        if state['status'] in ['completed', 'failed']:
            return
        row = {'observed_at': time.time(), 'campaign_pid': args.pid,
               'task': state.get('task'), 'variant': state.get('variant'),
               'arm': state.get('arm'), 'campaign_status': state['status']}
        try:
            sample = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'],
                                    capture_output=True, text=True, timeout=15, check=True)
            row.update(raw_pmon=sample.stdout, processes=parse_pmon(sample.stdout))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row['sampling_error'] = repr(exc)
        row['sampling_finished_at'] = time.time()
        with output.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        time.sleep(60)


if __name__ == '__main__':
    main()
