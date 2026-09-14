"""Read-only host-memory sampling alongside a frozen full-task campaign.

PSS/RSS measure the model server and its child processes, not just KV. Sampled
peaks are lower bounds; a process VmHWM can include model loading. No CUDA calls
or expensive tensor traversal are added to the generation path.
"""
import argparse
import json
from pathlib import Path
import time


def kb_fields(path):
    result = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'kB':
            result[parts[0].rstrip(':')] = int(parts[1])
    return result


def processes(pid):
    todo, seen, rows = [pid], set(), []
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        root = Path('/proc') / str(pid)
        try:
            fields = root.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            status = kb_fields(root / 'status')
            smaps = kb_fields(root / 'smaps_rollup')
            rows.append({'pid': pid, 'starttime_ticks': int(fields[19]),
                'rss_kib': smaps['Rss'], 'pss_kib': smaps['Pss'],
                'pss_anon_kib': smaps.get('Pss_Anon'), 'vm_hwm_kib': status.get('VmHWM'),
                'swap_kib': status.get('VmSwap')})
            for task in (root / 'task').glob('*'):
                try:
                    todo.extend(map(int, (task / 'children').read_text().split()))
                except OSError:
                    pass
        except (OSError, ValueError):
            continue
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args()
    while True:
        state = json.loads((a.campaign / 'state.json').read_text())
        row = {'at': time.time(), 'campaign_status': state['status'],
            'host': kb_fields(Path('/proc/meminfo'))}
        if state.get('arm'):
            arm = Path(state['arm'])
            result = arm / 'result.json'
            if result.exists():
                result = json.loads(result.read_text())
                if result.get('server_pid') and result['status'] in {'starting', 'running_agent'}:
                    row.update(arm=str(arm), model_processes=processes(result['server_pid']))
                    with (arm / 'host_memory_samples.jsonl').open('a') as f:
                        f.write(json.dumps(row) + '\n')
        if state['status'] in {'completed', 'failed'}:
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
