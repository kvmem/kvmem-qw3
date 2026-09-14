"""Summarize observed collection cost, separating device and process memory samples."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    args = parser.parse_args()
    root = args.campaign
    combined = json.loads((root / 'campaign.json').read_text())
    assert combined['status'] == 'finished'
    origins = [Path(combined[k]) for k in ['original_campaign', 'supplement_campaign']]
    domains = defaultdict(list)
    server_pids = set()
    server_memory = []
    request_count = 0
    for origin in origins:
        state = json.loads((origin / 'campaign.json').read_text())
        assert state['status'] == 'finished'
        for p in origin.glob('measurements/*/step_*.json'):
            d = json.loads(p.read_text())
            assert d['status'] == 'collected'
            domains[d['domain']].append(d['elapsed_sec'])
            request_count += 1
        for p in sorted(origin.glob('server_*/gpu_samples.csv')):
            samples = list(csv.DictReader(p.open()))
            peak = max(samples, key=lambda r: float(r[' memory.used [MiB]'].split()[0]))
            server_pids.add((p.parent / 'server.pid').read_text().strip())
            server_memory.append(dict(source=str(p), samples=len(samples),
                                      device_peak_MiB=float(peak[' memory.used [MiB]'].split()[0]),
                                      local_peak_timestamp=peak['timestamp']))
    process_path = root.parent / 'gpu_process_samples.jsonl'
    processes = {}
    process_start = None
    if process_path.exists():
        for line in process_path.read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue  # An active monitor may still be writing its last record.
            record = json.loads(line)
            if process_start is None:
                process_start = record['timestamp']
            for fields in csv.reader(record['processes_csv'].splitlines()):
                if not fields:
                    continue
                pid, memory, name = [v.strip() for v in fields]
                if not memory.isdigit():
                    continue
                item = processes.setdefault(pid, dict(name=name, peak_MiB=0, samples=0,
                                                      collection_server=pid in server_pids))
                item['peak_MiB'] = max(item['peak_MiB'], int(memory))
                item['samples'] += 1
    result = dict(as_of=time.time(), total_collected_calls_including_auxiliary=request_count,
                  collection_wall_hours=(combined['finished_at']-combined['started_at'])/3600,
                  summed_observed_request_hours=sum(sum(v) for v in domains.values())/3600,
                  domains={k:dict(calls=len(v), observed_mean_request_seconds=sum(v)/len(v))
                           for k, v in domains.items()},
                  device_memory_samples=server_memory,
                  process_samples_since=process_start, process_sample_source=str(process_path),
                  processes=processes,
                  scope='Observed instrumented recorded-input collection cost, not serving performance. '
                  'Device-wide historical memory peaks cannot be attributed to this collection alone. '
                  'Process monitoring began during batch 3, so its observed peak is not a full-run '
                  'peak or a proven minimum GPU capacity. Final analysis/review time is excluded.')
    assert request_count == 969
    out = root / 'analysis'
    (out / 'resource_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    device_peak = max(r['device_peak_MiB'] for r in server_memory)/1024
    own_peaks = [r['peak_MiB']/1024 for r in processes.values() if r['collection_server']]
    own_peak = f'{max(own_peaks):.2f} GiB' if own_peaks else '未获得'
    other_count = sum(not r['collection_server'] for r in processes.values())
    lines = ['# 采集时间与显存观察', '',
             f"采集共 {request_count} 次调用（含 12 次辅助尝试），从首次采集启动到补充采集完成约 "
             f"{result['collection_wall_hours']:.2f} 小时。请求记录耗时合计 "
             f"{result['summed_observed_request_hours']:.2f} 小时；完整分析和审图时间另计。", '',
             '模型为 Qwen3.8-27B-Q8_0，视觉模块 BF16，KV FP16，64K 上下文上限。', '',
             f'整卡监控观察到的峰值为 {device_peak:.2f} GiB；第三批后期开始的分进程监控中，'
             f'采集服务观察到的峰值为 {own_peak}。两者监控区间和统计对象不同。'
             f'分进程记录还观察到 {other_count} 个其他 GPU 进程，整卡数值包含同期其他测试。'
             '分进程监控开始前的区间无法事后归属，因此不能由这些数字推导最低显存要求。', '',
             '这些时间包含 attention 旁路记录开销，也未证明全程独占 GPU。它们用于规划采集资源，'
             '不能作为系统检索开销或服务吞吐量的实验结果。', '',
             '逐批显存记录、进程观察起点和各类请求耗时见 `resource_summary.json`。']
    (out / 'resource_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'collected_calls_including_auxiliary': request_count,
                      'collection_wall_hours': result['collection_wall_hours']}))


if __name__ == '__main__':
    main()
