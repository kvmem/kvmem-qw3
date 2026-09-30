#!/usr/bin/env python3
"""Resumable summary-only DeepSWE campaign, all samples before the next seed."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import thinking_campaign_metrics as metrics_lib

REPO = Path('/home/chaidi/qw3')
HARNESS = REPO / 'benchmark/claude_deepswe_ab'
SEEDS = (1000, 2000, 3000, 4000)
CONDITION = 'claude_kvmem_128k_gen64k'


def load_batch(directory=HARNESS):
    spec = importlib.util.spec_from_file_location('thinking_campaign_batch', directory / 'batch.py')
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    batch.HERE = directory
    batch.ANALYZE = directory / 'analyze_arm.py'
    return batch


def schedule(tasks, seeds=SEEDS):
    return [{"round": r, "seed": seed, "sample_index": i, "task": task['task_id'], "arm": 'summary'}
            for r, seed in enumerate(seeds, 1) for i, task in enumerate(tasks, 1)]


def trial_dir(root, item):
    return root / f"round-{item['round']}" / item['task']


def physical_context(manifest):
    # Bounded KVMem --ctx sizes the arena, while the client history limit is
    # independent. The HTTP engine admits logical histories beyond that arena.
    return manifest['repository_blocks'] * manifest['block_tokens']


def gpu_queue_blockers(root, batch):
    # Optional operational dependency on an already-running foreign campaign.
    # Wait for its supervisor, not a transient GPU-free gap between its trials.
    owners = batch.read_json(root / 'gpu_wait_owners.json', {}).get('owners', [])
    return [owner for owner in owners if batch._process_start_ticks(owner['pid']) == owner['start_ticks']]


def gpu_admission_ready(manifest, usage):
    if manifest.get('allow_shared_gpu', False):
        return all(free >= manifest.get('minimum_free_gpu_mib', 40960) for used, free in usage)
    return all(used < 1000 for used, free in usage)


def wait_for_gpu(root, manifest, batch, complete):
    while True:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free',
            '--format=csv,noheader,nounits'], text=True)
        usage = [tuple(int(x.strip()) for x in line.split(',')) for line in output.splitlines()]
        blockers = [] if manifest.get('allow_shared_gpu', False) else gpu_queue_blockers(root, batch)
        if usage and not blockers and gpu_admission_ready(manifest, usage):
            return
        state(batch, root, 'waiting_for_gpu', completed_trials=complete, total_trials=manifest['total_trials'],
              waiting_owners=blockers, gpu_usage_mib=usage,
              allow_shared_gpu=manifest.get('allow_shared_gpu', False))
        time.sleep(10)


def state(batch, root, phase, **data):
    batch.atomic_json(root / 'state.json', {
        'phase': phase, 'updated_at': batch.utc_now(), 'supervisor_pid': os.getpid(), **data})


def historical_baselines(root, tasks, batch):
    recordings = HARNESS / 'recordings'
    catalogs = [
        ('kvmem', 'kvmem3m_128gen64_full113_claude_pass4_pro_reuse_passmajor_s1000_4000_20260829'),
        ('kvmem', 'kvmem3m_128gen64_full113_claude_pass4_taskmajor_s1000_4000_20260826'),
        ('dense', 'dense256_claude_auto_compact_first16_pass4_s1000_4000_20260901')]
    selected = {}
    configurations = {}
    for family, name in catalogs:
        directory = recordings / name
        configurations[name] = batch.read_json(directory / 'manifest.json', {})
        summary = batch.read_json(directory / 'summary.json', {})
        for row in summary.get('rows', []):
            task = row['task_id']
            if task not in {t['task_id'] for t in tasks}:
                continue
            for seed, previous in row.get('results', {}).items():
                key = (family, task, int(seed))
                path = Path(previous.get('metrics', '/nonexistent'))
                if key in selected or not previous.get('valid') or not path.is_file():
                    continue
                metrics = batch.read_json(path, {})
                parser = metrics_lib.RequestParser()
                parser.poll(path.parent / 'qw3_server.timed.log')
                selected[key] = {'family': family, 'task': task, 'seed': int(seed), 'source': name,
                    'metrics_path': str(path), 'metrics_sha256': batch.sha256(path),
                    'historical_valid': previous['valid'], **metrics_lib.trial_metrics(metrics, parser.rows)}
    data = {'selection_rule': 'Fixed catalog priority, valid recorded trials only; never select by reward or elapsed time.',
            'comparison_limit': 'Historical reference, not an isolated summary ablation: binary, repository design, CPU backing and guided-query cap differ.',
            'configurations': configurations, 'rows': list(selected.values())}
    batch.atomic_json(root / 'historical-baselines.json', data)


def freeze(args):
    batch = load_batch()
    root = args.experiment.resolve()
    if (root / 'manifest.json').exists():
        return root
    ids = [x.strip() for x in args.tasks.read_text().splitlines() if x.strip()]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError('expected the 16 unique historical task IDs')
    tasks = []
    for task_id in ids:
        task = batch.task_metadata(task_id)
        task['verifier_image'] = f'qw3-deepswe-{task_id}-verifier:v1.1'
        task['instruction_sha256'] = batch.sha256(Path(task['instruction']))
        task['images'] = {name: json.loads(subprocess.check_output(['docker', 'image', 'inspect', name]))[0]['Id']
                          for name in (task['agent_image'], task['verifier_image'])}
        tasks.append(task)
    validation = args.validated_pilot.resolve()
    validated = batch.read_json(validation / 'manifest.json')
    validation_results = validated.get('validation_results', {
        'http_baseline': 'http-oracle-baseline/result.json',
        'http_summary': 'http-oracle-summary/result.json',
        'output_27b': 'output-27b-transaction/result.json'})
    if not {'http_baseline', 'http_summary', 'output_27b'} <= validation_results.keys():
        raise ValueError('required prelaunch validation missing')
    for name, path in validation_results.items():
        if not batch.read_json(validation / path, {}).get('passed'):
            raise ValueError(f'prelaunch validation missing: {name}')
    oracle = (validation / 'state-oracle.log').read_text()
    if not all(f'boundary={b} recurrent=exact' in oracle for b in (63, 511, 1535)):
        raise ValueError('state oracle validation missing')
    (root / 'bin').mkdir(parents=True)
    (root / 'harness').mkdir()
    (root / 'setup').mkdir()
    shutil.copy2(validation / 'bin/qw3', root / 'bin/qw3')
    if batch.sha256(root / 'bin/qw3') != validated['binaries']['qw3']:
        raise ValueError('validated binary identity changed')
    shutil.copy2(validation / 'source.tar.gz', root / 'model-source.tar.gz')
    for source in (Path(__file__), Path(metrics_lib.__file__), HARNESS / 'batch.py',
                   HARNESS / 'analyze_arm.py', HARNESS / 'service_watchdog.py'):
        shutil.copy2(source, root / 'harness' / source.name)
    for source in (batch.SERVER, batch.CELL_RUNNER, batch.AGENT_RUNNER):
        shutil.copy2(source, root / 'setup' / source.name)
    shutil.copy2(args.tasks, root / 'tasks.txt')
    historical_baselines(root, tasks, batch)
    flags = ['--kvmem-index-placement', 'cpu', '--kvmem-repository-blocks', '16384',
             '--kvmem-repository-gc', 'delete', '--thinking-summary', '--thinking-summary-min-tokens', '1024']
    wrapper = root / 'setup/binary-wrapper.sh'
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(root / 'bin/qw3')) + ' "$@" ' + shlex.join(flags) + '\n')
    wrapper.chmod(0o755)
    fixed = [root / 'bin/qw3', root / 'model-source.tar.gz', root / 'tasks.txt', root / 'historical-baselines.json',
             *list((root / 'harness').iterdir()), *list((root / 'setup').iterdir())]
    manifest = {
        'schema': 'qw3.thinking_summary.first16.pass4.v1', 'created_at': batch.utc_now(),
        'tasks': tasks, 'seeds': list(args.seeds), 'arms': ['summary'], 'order': 'round-then-sample',
        'schedule': schedule(tasks, args.seeds), 'total_trials': len(tasks) * len(args.seeds),
        'logical_context': 3145728, 'physical_context': 524288, 'repository_blocks': 16384, 'block_tokens': 32,
        'selection_budget': 131072, 'generation_reserve': 65536, 'cpu_gb': 48,
        'summary_min_tokens': 1024, 'summary_mode': 'natural', 'summary_fixed_cap': None,
        'summary_output_policy': 'Paused-task memory note + Summary: prefill + constrained greedy plain-text generation; full output diagnostics; invalid output preserves original reasoning.',
        'sampling': validated['sampling'], 'port': args.port,
        'allow_shared_gpu': args.allow_shared_gpu, 'minimum_free_gpu_mib': 40960,
        'model': str(batch.MODEL), 'model_size': batch.MODEL.stat().st_size,
        'model_mtime_ns': batch.MODEL.stat().st_mtime_ns,
        'model_sha256': batch.sha256(batch.MODEL),
        'claude_binary': str(batch.CLAUDE), 'claude_sha256': batch.sha256(batch.CLAUDE),
        'files': {str(path.relative_to(root)): batch.sha256(path) for path in fixed},
        'validation_source': str(validation), 'binary_sha256': validated['binaries']['qw3'],
        'cache_reuse_policy': validated.get('cache_reuse_policy'),
        'validation_sha256': {name: batch.sha256(validation / name)
            for name in ('state-oracle.log', *validation_results.values())},
        'baseline_policy': 'Reuse historical baselines as references; do not launch baseline trials.',
        'failure_policy': 'Task reward=0 is a completed outcome; infrastructure/inference failures pause for repair, retaining all attempts.',
        'metrics_policy': 'Final pool is sampled before teardown. Logical context, retained prefix tokens, fixed capacity and live KV occupancy are distinct. Peaks are sampled at request boundaries.'}
    batch.atomic_json(root / 'manifest.json', manifest)
    state(batch, root, 'prepared', total_trials=manifest['total_trials'], completed_trials=0)
    return root


def verify(root, manifest, batch):
    for name, expected in manifest['files'].items():
        if batch.sha256(root / name) != expected:
            raise RuntimeError(f'frozen campaign file changed: {name}')
    model = Path(manifest['model'])
    if (model.stat().st_size, model.stat().st_mtime_ns) != (manifest['model_size'], manifest['model_mtime_ns']):
        raise RuntimeError('model file changed')
    for name, identity in manifest.get('dflash_files', {}).items():
        actual = Path(name).stat()
        if (actual.st_size, actual.st_mtime_ns) != (identity['size'], identity['mtime_ns']):
            raise RuntimeError(f'DFlash model file changed: {name}')


def worker(attempt):
    root = next(p for p in attempt.parents if (p / 'harness/batch.py').exists())
    batch = load_batch(root / 'harness')
    info = batch.read_json(attempt / 'service.json')
    info['log'] = Path(info['log'])
    task = batch.read_json(attempt / 'manifest.json')['tasks'][0]
    return batch.run_arm(attempt, task, CONDITION, SimpleNamespace(**info))


def stop_worker(process, artifact):
    # Signal the shell too, so its owned-container trap runs before exit.
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    owner = hashlib.sha256(str(artifact).encode()).hexdigest()[:16]
    for role in ('agent', 'verifier'):
        cidfile = artifact / f'{role}.container.cid'
        if not cidfile.exists():
            continue
        cid = cidfile.read_text().strip()
        result = subprocess.run(['docker', 'inspect', cid], capture_output=True, text=True)
        if result.returncode:
            continue
        labels = json.loads(result.stdout)[0]['Config'].get('Labels') or {}
        if labels.get('com.qw3.deepswe.run-owner') == owner and labels.get('com.qw3.deepswe.role') == role:
            subprocess.run(['docker', 'stop', '--timeout', '5', cid], stdout=subprocess.DEVNULL, timeout=20)


def system_sample(elapsed):
    result = {'elapsed_sec': elapsed, 'time': time.time()}
    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, timeout=5)
    result['gpu_processes'] = []
    for line in gpu.stdout.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            pid, mib = map(int, parts)
            item = {'pid': pid, 'gpu_allocated_bytes': mib * 1024**2}
            try:
                status = Path(f'/proc/{pid}/status').read_text()
                for field in ('VmRSS', 'VmHWM'):
                    import re
                    match = re.search(rf'^{field}:\s+(\d+)', status, re.M)
                    item[field + '_bytes'] = int(match[1]) * 1024 if match else None
            except OSError:
                pass
            result['gpu_processes'].append(item)
    result['artifact_disk_free_bytes'] = shutil.disk_usage('/data').free
    return result


def run_trial(root, manifest, item, batch):
    task = next(t for t in manifest['tasks'] if t['task_id'] == item['task'])
    for image, expected in task['images'].items():
        if json.loads(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]['Id'] != expected:
            raise RuntimeError(f'task image changed: {image}')
    if batch.sha256(Path(task['instruction'])) != task['instruction_sha256']:
        raise RuntimeError('task instruction changed')
    trial = trial_dir(root, item)
    trial.mkdir(parents=True, exist_ok=True)
    number = 1 + max((int(p.name.split('-')[-1]) for p in trial.glob('attempt-*')), default=0)
    attempt = trial / f'attempt-{number:02d}'
    attempt.mkdir()
    (attempt / 'setup').symlink_to(root / 'setup', target_is_directory=True)
    batch.atomic_json(attempt / 'manifest.json', {
        'tasks': [task], 'seed': item['seed'], 'sampling': manifest['sampling'],
        'claude_model_id': 'Qwen3.8-27B[1m]', 'kvmem': {'context': manifest['logical_context']},
        'execution': {'agent_container_memory': '16g'}, 'thinking_arm': 'summary'})
    artifact = attempt / 'runs' / item['task'] / CONDITION
    parser = metrics_lib.RequestParser()
    service = process = None
    started = time.monotonic()
    failure = None
    os.environ['QW3_BINARY_PATH'] = str(root / 'setup/binary-wrapper.sh')
    state(batch, root, 'starting_trial', **item, attempt=number, artifact=str(artifact))
    with batch.DockerNetwork(attempt, f"r{item['round']}-s{item['sample_index']}-a{number}",
            allow_wan=batch.network_policy_for_task(task) == 'docker_bridge_wan_audited') as network:
        try:
            service = batch.Service(attempt, CONDITION, 0, manifest['port'], item['seed'], manifest['sampling'],
                bind_host=network.gateway, agent_network=network.name, context=physical_context(manifest),
                kvmem_cpu_gb=manifest['cpu_gb'], kvmem_guided_query_tokens=512)
            service.wait_ready()
            batch.atomic_json(attempt / 'service.json', {key: str(getattr(service, key)) if key == 'log' else getattr(service, key)
                for key in ('bind_host', 'port', 'agent_network', 'log')})
            # Ignore startup/warmup output; every logged request below belongs to this agent.
            parser.offset = service.log.stat().st_size
            started = time.monotonic()
            with (attempt / 'worker.log').open('w') as log, (attempt / 'requests.jsonl').open('w') as trace, \
                    (attempt / 'resources.jsonl').open('w') as resources:
                process = subprocess.Popen([sys.executable, str(root / 'harness' / Path(__file__).name),
                    '--run-cell', str(attempt)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state(batch, root, 'running_trial', **item, attempt=number, artifact=str(artifact), worker_pid=process.pid)
                sampled_at = -30.0
                while True:
                    elapsed = time.monotonic() - started
                    for row in parser.poll(service.log, elapsed):
                        trace.write(json.dumps(row) + '\n')
                        if row['accounting'].get('terminal_status') == 'error':
                            failure = 'inference stream error: ' + str(row['accounting'].get('terminal_error'))
                    trace.flush()
                    if elapsed - sampled_at >= 10:
                        resources.write(json.dumps(system_sample(elapsed)) + '\n'); resources.flush()
                        sampled_at = elapsed
                        batch.atomic_json(root / 'live.json', {**item, 'attempt': number, 'elapsed_sec': elapsed,
                            'updated_at': batch.utc_now(), 'requests': len(parser.rows),
                            'last_request': parser.rows[-1] if parser.rows else None,
                            'artifact': str(artifact), 'worker_pid': process.pid})
                    if process.poll() is not None or failure:
                        break
                    if service.process.poll() is not None:
                        failure = 'inference service exited'; break
                    time.sleep(2)
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
            raise
        finally:
            stop_worker(process, artifact)
            if service is not None:
                for row in parser.poll(service.log, time.monotonic() - started):
                    with (attempt / 'requests.jsonl').open('a') as trace:
                        trace.write(json.dumps(row) + '\n')
                service.stop()
            if failure:
                batch.atomic_json(attempt / 'interruption.json', {'reason': failure, 'at': batch.utc_now()})
    text = (artifact / 'qw3_server.timed.log').read_text(errors='replace') if (artifact / 'qw3_server.timed.log').exists() else ''
    metrics = batch.read_json(artifact / 'metrics.json', {})
    http_audit = metrics_lib.http_error_audit(text, artifact)
    reasons = metrics_lib.invalid_reasons(metrics, text, http_audit=http_audit)
    if failure:
        reasons.append(failure)
    if not parser.rows or not all(row.get('pool') for row in parser.rows if row.get('generation')):
        reasons.append('missing_request_pool_metrics')
    metrics_lib.write_csv(attempt / 'requests.csv', metrics_lib.request_csv(parser.rows))
    result = {**item, 'attempt': number, 'binary_sha256': manifest['binary_sha256'],
              'shared_gpu_allowed': manifest.get('allow_shared_gpu', False),
              'valid': not reasons, 'error_reasons': reasons, 'artifact': str(artifact),
              'http_error_audit': http_audit,
              'http_error_count': len(http_audit['http_errors']),
              'recovered_auxiliary_http_error_count': len(http_audit['recovered_web_search']),
              'request_trace': str(attempt / 'requests.jsonl'), 'resource_trace': str(attempt / 'resources.jsonl'),
              'finished_at': batch.utc_now(), **metrics_lib.trial_metrics(metrics, parser.rows)}
    batch.atomic_json(attempt / 'result.json', result)
    if reasons:
        raise RuntimeError('trial needs repair: ' + '; '.join(reasons))
    batch.atomic_json(trial / 'result.json', result)
    return result


def publish(root, manifest, batch):
    rows = [batch.read_json(trial_dir(root, item) / 'result.json') for item in manifest['schedule']]
    rows = [r for r in rows if r and r.get('valid')]
    sample_rows, _ = metrics_lib.summarize_rows(rows, ['summary'])
    batch.atomic_json(root / 'results.json', rows)
    metrics_lib.write_csv(root / 'results.csv', [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
    metrics_lib.write_csv(root / 'samples.csv', sample_rows)
    references = batch.read_json(root / 'historical-baselines.json', {}).get('rows', [])
    comparisons = []
    for row in rows:
        for ref in references:
            if (row['task'], row['seed']) != (ref['task'], ref['seed']):
                continue
            comparison = {key: row[key] for key in ('round', 'task', 'seed')}
            comparison.update({'reference_family': ref['family'], 'reference_source': ref['source'],
                'summary_passed': row['passed'], 'reference_passed': ref['passed'],
                'controlled_ablation': False, 'both_passed': row['passed'] and ref['passed']})
            for key in ('agent_elapsed_sec', 'request_wall_sec', 'final_context_tokens', 'final_live',
                        'final_ready', 'final_gpu_kv_bytes', 'final_cpu_bytes'):
                a, b = ref.get(key), row.get(key)
                comparison[key + '_reference'] = a
                comparison[key + '_summary'] = b
                comparison[key + '_delta'] = b - a if a is not None and b is not None else None
            comparisons.append(comparison)
    metrics_lib.write_csv(root / 'historical-comparison.csv', comparisons)
    report = [f"# Thinking 摘要：{len(manifest['tasks'])} 样本 × {len(manifest['seeds'])} 轮", '',
              f"完成 {len(rows)}/{manifest['total_trials']}；任务通过 {sum(r['passed'] for r in rows)}。", '',
              '顺序：每轮依次执行完整样本，seed 为 ' + ' / '.join(map(str, manifest['seeds'])) + '。只跑摘要组。', '',
              '默认 thinking ≥1024 tokens 才摘要，摘要自然结束。逻辑 context 上限 3M，物理仓库 512K tokens。', '',
              '执行端：' + manifest.get('speculative_method', 'mtp') + '；摘要上下文：' +
                  manifest.get('summary_context', '原有摘要策略，见 manifest') + '。', '',
              '历史 baseline 仅作参照；旧二进制、仓库、CPU backing 和 guided-query 参数存在差异，不能将全部差值归因于摘要。', '',
              ('用户授权与其他实验共享 GPU；耗时包含资源竞争，resources.jsonl 每10秒记录所有 GPU 进程占用，不能按独占运行解释。'
               if manifest.get('allow_shared_gpu', False) else '本批次按独占 GPU 准入。'), '',
              '最终 context 是末次生成的 prompt + 可见 decoded tokens；committed context 是实际 forward 的历史位置。',
              'KV live/ready/GPU/CPU/index 在每次请求完成时记录；峰值是请求边界采样峰值。资源曲线每10秒采样，包含模型权重的 GPU 总分配不能当作 KV bytes。', '',
              '| 样本 | 已完成轮次 | 通过次数 | 平均耗时 (s) | 平均最终 context |',
              '| --- | ---: | ---: | ---: | ---: |']
    for sample in sample_rows:
        def number(value):
            return f'{value:.1f}' if value is not None else '—'
        report.append(f"| {sample['task']} | {sample['completed_rounds']} | {sample['successes']} | "
                      f"{number(sample['agent_elapsed_sec_mean'])} | {number(sample['final_context_tokens_mean'])} |")
    report += ['', '逐 trial 指标：results.csv；按样本汇总：samples.csv；历史参照：historical-comparison.csv。',
               f"已核实由工具结果返回、随后主推理继续的 WebSearch 辅助请求拒绝：{sum(r.get('recovered_auxiliary_http_error_count', 0) for r in rows)} 次；证据保存在 trial 的 http_error_audit，仍计入 HTTP 错误统计。",
               '每个 attempt 保留 requests.jsonl / requests.csv（KV/context曲线）、resources.jsonl（资源采样）及原始日志。',
               '任务 reward=0 是有效结果，不通过重试筛选成功。基础设施/推理错误保留 attempt 并暂停修复。']
    (root / 'report.md').write_text('\n'.join(report) + '\n')
    batch.atomic_json(root / 'summary.json', {'updated_at': batch.utc_now(), 'completed_trials': len(rows),
        'total_trials': manifest['total_trials'], 'successes': sum(r['passed'] for r in rows), 'samples': sample_rows,
        'completed_rounds': [r for r in range(1, len(manifest['seeds']) + 1)
                             if sum(x['round'] == r for x in rows) == len(manifest['tasks'])]})
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--tasks', type=Path, default=HARNESS / 'tasks_first16_full113.txt')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--validated-pilot', type=Path, default=REPO / 'benchmark/thinking_compaction_natural_1k_deepswe_20260922')
    parser.add_argument('--port', type=int, default=18973)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-shared-gpu', action='store_true',
                        help='User-authorized concurrency; require 40 GiB free before each trial')
    parser.add_argument('--run-cell', type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('--seeds must be unique')
    if args.run_cell:
        return worker(args.run_cell.resolve())
    if args.experiment is None:
        parser.error('--experiment is required')
    root = freeze(args)
    if args.prepare_only:
        print(root); return 0
    batch = load_batch(root / 'harness')
    manifest = batch.read_json(root / 'manifest.json')
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch.publish_supervisor_owner(root)
        verify(root, manifest, batch)
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            complete = publish(root, manifest, batch)
            for item in manifest['schedule']:
                existing = batch.read_json(trial_dir(root, item) / 'result.json', {})
                if existing.get('valid'):
                    continue
                verify(root, manifest, batch)
                batch.wait_for_disk_headroom(root, minimum_free_bytes=16 * 1024**3)
                wait_for_gpu(root, manifest, batch, complete)
                run_trial(root, manifest, item, batch)
                complete = publish(root, manifest, batch)
            state(batch, root, 'finished', completed_trials=complete, total_trials=manifest['total_trials'])
        except BaseException as error:
            publish(root, manifest, batch)
            current = batch.read_json(root / 'state.json', {})
            state(batch, root, 'stopped' if isinstance(error, KeyboardInterrupt) else 'needs_attention', reason=str(error),
                  **{key: current[key] for key in ('round', 'task', 'seed', 'attempt', 'artifact') if key in current})
            raise
        finally:
            batch.publish_supervisor_owner(root, 'stopped')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
