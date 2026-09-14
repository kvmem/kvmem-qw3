"""Exclusive-GPU paired TTFT and bounded decode on fixed text history.

This supplements complete shared-GPU agent trials with controlled timing. It
does not execute a compaction policy, score tasks, or measure long decoding.
Queued campaigns must finish as well as release their current GPU allocation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

from summarize_pilot import native_phases

ROOT = Path(__file__).resolve().parent


def blockers():
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader"], text=True).strip()
    queued = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except OSError:
            continue
        scripts = [x.decode(errors='replace') for x in argv[1:]]
        if any(s.endswith(('/run_pilot_campaign.py', '/motivation_v3/run_campaign.py',
                           '/motivation_v3/finish_corrected_campaign.py')) for s in scripts):
            queued.append(int(path.parent.name))
    return {"gpu": gpu, "queued_campaign_pids": queued} if gpu or queued else {}


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def summarize(out, runs):
    records, summaries = [], []
    for run in runs:
        result = json.loads((run / 'result.json').read_text())
        if result['status'] != 'completed_fixed_input_ttft' or not result['performance_claim']:
            raise ValueError('ineligible timing result: ' + str(run))
        rows = [json.loads(l) for l in (run / 'measurements.jsonl').read_text().splitlines()]
        if result['method'] == 'infllm':
            phases = [json.loads(l) for l in (run / 'server/requests.jsonl').read_text().splitlines()]
            phases = [{'engine_prefill_s': p['prefill_s'], 'engine_decode_s': p['decode_s'],
                       'output_tokens': p['output_tokens'], 'guided_query_s': 0,
                       'query_replay_s': 0} for p in phases]
        else:
            phases = [{'engine_prefill_s': p['prefill_ms'] / 1000, 'engine_decode_s': p['decode_ms'] / 1000,
                       'output_tokens': int(p['decoded']), 'guided_query_s': p.get('post_guided_query_ms', 0) / 1000,
                       'query_replay_s': p.get('post_query_replay_ms', 0) / 1000}
                      for p in native_phases(run / 'server.log')]
        if len(phases) != len(rows):
            raise ValueError('request / engine phase count mismatch: ' + str(run))
        for r, phase in zip(rows, phases):
            if r.get('output_tokens') is not None and r['output_tokens'] != phase['output_tokens']:
                raise ValueError('wire / engine output-token count mismatch: ' + str(run))
            records.append({'run': run.name, 'method': result['method'],
                'request_index': r['index'], 'prompt_tokens': r['prompt_tokens'],
                'initial_request': r is rows[0], 'api_ttft_s': r['api_first_content_s'],
                'request_wall_s': r['request_wall_s'], 'stop_reason': r.get('stop_reason'), **phase})
        for label, selected in [('all', rows), ('after_initial_request', rows[1:])]:
            values = [r['api_first_content_s'] for r in selected if r['api_first_content_s'] is not None]
            if not values:
                raise ValueError('no measured content tokens: ' + str(run))
            summaries.append({'run': run.name, 'method': result['method'], 'scope': label,
                'n': len(values), 'missing_ttft': len(selected) - len(values),
                'mean_s': statistics.mean(values), 'p50_s': percentile(values, .5),
                'p95_s': percentile(values, .95), 'p99_s': percentile(values, .99),
                'max_s': max(values), 'total_s': sum(values),
                'peak_process_gpu_gib': result['peak_process_gpu_gib'],
                'output_tokens_sum': sum(p['output_tokens'] for p in (phases if label == 'all' else phases[1:])),
                'engine_decode_sum_s': sum(p['engine_decode_s'] for p in (phases if label == 'all' else phases[1:])),
                'decode_scope': 'up to 64 naturally generated tokens per request; first sampled token originates in prefill'})
    for name, rows in [('requests.csv', records), ('summary.csv', summaries)]:
        with (out / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--wait', action='store_true', help='Wait for both active GPU owners and known queued campaigns')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if not plan.get('frozen'):
        raise ValueError('requires the frozen pilot resource configuration')
    for name, expected in plan['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen source changed: ' + name)
    commands = []
    # Reverse the method order on the second repetition. Each arm starts its own
    # model process and receives exactly the same 62 observed request prefixes.
    # Natural EOS is retained. Measure actual output counts; do not assume every
    # method uses the full output cap or produces an identical continuation.
    for repeat, order in enumerate([['infllm', 'native_kvmem'], ['native_kvmem', 'infllm']], 1):
        for method in order:
            setting = plan['methods'][method]
            run = a.out.resolve() / f'repeat_{repeat}_{method}'
            command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'run_fixed_trace.py'),
                '--formal-ttft', '--output-tokens', '64', '--trace', str(a.trace.resolve()), '--out', str(run), '--method', method,
                '--gpu-process-cap-gib', str(plan['gpu_process_cap_gib']),
                '--required-free-gpu-gib', str(setting['minimum_free_gpu_gib'])]
            if method == 'infllm':
                command += ['--infllm-config', str(ROOT / setting['config']),
                    '--infllm-max-gpu-gib', str(setting['allocator_cap_gib']),
                    '--infllm-context', str(setting['context_tokens'])]
            else:
                command += ['--native-context', str(setting['context_tokens']),
                    '--native-cpu-gb', str(plan['cpu_kv_cap_gib']),
                    '--native-gpu-memory-ratio', str(setting['gpu_planning_ratio']),
                    '--kvmem-budget', str(setting['selection_budget']),
                    '--kvmem-gen-budget', str(setting['generation_budget']),
                    '--kvmem-middecode-trigger', str(setting['middecode_trigger_tokens'])]
            commands.append({'run': str(run), 'command': command})
    if a.dry_run:
        print(json.dumps({'blockers': blockers(), 'runs': commands}, indent=2))
        return
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'plan.json').write_bytes(a.plan.read_bytes())
    state = {'status': 'waiting', 'started': time.time(), 'completed': [], 'runs': commands,
        'claim_scope': 'fixed-input TTFT and up-to-64-token greedy generation; same 60 GiB monitored ceiling; no agent/compaction/long-decode claim',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'measurement_sha256': hashlib.sha256((ROOT / 'run_fixed_trace.py').read_bytes()).hexdigest()}
    def save():
        temp = a.out / 'state.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(a.out / 'state.json')
    child = None
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for row in commands:
            while (busy := blockers()):
                state.update(status='waiting_for_exclusive_gpu', blockers=busy)
                save()
                if not a.wait:
                    raise RuntimeError('GPU or queued campaign still busy')
                time.sleep(10)
            if hashlib.sha256((ROOT / 'run_fixed_trace.py').read_bytes()).hexdigest() != state['measurement_sha256']:
                raise ValueError('measurement source changed while waiting')
            for name, expected in plan['source_sha256'].items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                    raise ValueError('frozen source changed while waiting: ' + name)
            weights = json.loads((ROOT / plan['model_manifest']).read_text())
            for name, expected in weights['files'].items():
                st = (Path(weights['model']) / name).stat()
                if (st.st_size, st.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
                    raise ValueError('model changed since SHA256 audit: ' + name)
            state.update(status='running', current=row['run'], blockers={})
            save()
            env = dict(os.environ)
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                env.pop(key, None)
            env['QW3_KVMEM_SCRATCH_RESERVE_MIB'] = str(plan['methods']['native_kvmem']['scratch_reserve_mib'])
            with (a.out / (Path(row['run']).name + '.controller.log')).open('w') as log:
                child = subprocess.Popen(row['command'], cwd=ROOT.parents[1], env=env, stdout=log, stderr=log)
                state['child_pid'] = child.pid
                save()
                if child.wait():
                    raise RuntimeError('fixed trace failed: ' + row['run'])
            state['completed'].append(row['run'])
            summarize(a.out, [Path(x) for x in state['completed']])
            save()
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
