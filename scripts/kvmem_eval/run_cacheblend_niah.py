#!/usr/bin/env python3
"""Matched CacheBlend on/off diagnostic using an existing frozen RULER campaign.

Keeps the reference model, KV budgets, guided query, MTP and blind ingestion.
This is a selected-cohort diagnostic, not an official RULER leaderboard run.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import requests


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--binary', type=Path, default=Path('build/qw3'))
    ap.add_argument('--length', type=int, default=65536)
    ap.add_argument('--depths', type=int, nargs='+', default=[10, 50, 90])
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--seeds', type=int, nargs='+', help='Dataset and request seeds; defaults to --seed')
    ap.add_argument('--port', type=int, default=19643)
    ap.add_argument('--ratio', type=float, default=.10)
    ap.add_argument('--max-recompute', type=int, default=2**32 - 1)
    ap.add_argument('--max-output', type=int, default=8192)
    ap.add_argument('--thinking-budget', type=int, default=4096)
    ap.add_argument('--arms', nargs='+', choices=['off', 'on'], default=['off', 'on'])
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--extend-cohort', action='store_true', help='With --resume, add frozen cases without dropping existing ones')
    args = ap.parse_args()
    assert not args.extend_cohort or args.resume, '--extend-cohort requires --resume'
    args.seeds = sorted(set(args.seeds or [args.seed]))
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference / 'harness'))
    spec = importlib.util.spec_from_file_location('ruler_reference', args.reference / 'harness/ruler_campaign.py')
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    cohort = [r for r in json.loads((args.reference / 'schedule.json').read_text())
              if r['input_tokens'] == args.length and r['depth'] in args.depths and r['seed'] in args.seeds]
    assert len(cohort) == len(ref.TASKS) * len(set(args.depths)) * len(args.seeds)
    assert len({r['trial_id'] for r in cohort}) == len(cohort)
    binary = args.binary.resolve()
    config_path = args.out / 'config.json'
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         'binary_sha256': ref.sha(binary), 'tasks': ref.TASKS, 'trials_per_arm': len(cohort),
         'timing_note': 'Check resources.jsonl for other GPU processes; overlapping wall times are not an isolated speed comparison.'}
    if config_path.exists():
        assert args.resume, 'existing campaign requires --resume'
        previous = json.loads(config_path.read_text())
        for key in ['reference', 'length', 'seed', 'ratio', 'max_recompute', 'max_output', 'thinking_budget', 'arms']:
            assert config[key] == previous[key], f'resume changed {key}'
        if args.extend_cohort:
            old_cohort = json.loads((args.out / 'cohort.json').read_text())
            new_by_id = {row['id']: row for row in cohort}
            assert all(new_by_id.get(row['id']) == row for row in old_cohort), 'extension changed or dropped frozen cases'
        else:
            assert args.depths == previous['depths'], 'resume changed depths'
            assert args.seeds == (previous.get('seeds') or [previous['seed']]), 'resume changed seeds'
        save(args.out / f'config-before-resume-{time.time_ns()}.json', previous)
    save(config_path, config)
    save(args.out / 'cohort.json', cohort)
    results = json.loads((args.out / 'results.json').read_text()) if args.resume and (args.out / 'results.json').exists() else []
    completed = {(r['arm'], r['id']) for r in results}
    assert len(completed) == len(results), 'duplicate results'
    assert completed <= {(arm, row['id']) for arm in args.arms for row in cohort}, 'results outside cohort'
    http = requests.Session()
    http.trust_env = False
    base = f'http://127.0.0.1:{args.port}'
    for arm in args.arms:
        if all((arm, row['id']) in completed for row in cohort): continue
        dest = args.out / arm
        dest.mkdir(exist_ok=True)
        log = dest / 'server.log'
        cmd = json.loads((args.reference / 'on/server-command.json').read_text())
        cmd[0] = str(binary)
        for flag, value in [('--port', args.port), ('--thinking-budget', args.thinking_budget)]:
            cmd[cmd.index(flag) + 1] = str(value)
        env = {k: v for k, v in os.environ.items() if not k.startswith('QW3_')}
        env.update(json.loads((args.reference / 'on/server-env.json').read_text()))
        env.update(QW3_KVMEM_BLEND_RATIO=str(args.ratio if arm == 'on' else 0),
                   QW3_KVMEM_BLEND_MAX_TOKENS=str(args.max_recompute), QW3_KVMEM_PREFIX_CACHE_TRACE='1',
                   QW3_KVMEM_GUIDED_MTP='1', QW3_KVMEM_TRACE='1')
        save(dest / 'command.json', cmd)
        save(dest / 'env.json', {k: v for k, v in env.items() if k.startswith('QW3_')})
        save(dest / f'binary-identity-{time.time_ns()}.json', {'path': str(binary), 'sha256': ref.sha(binary)})
        with log.open('a') as stream:
            server = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None: raise RuntimeError(f'{arm} server exited; see {log}')
                try:
                    if http.get(base + '/health', timeout=2).ok: break
                except requests.RequestException:
                    pass
                if time.monotonic() >= deadline: raise TimeoutError('server startup')
                time.sleep(1)
            for row in cohort:
                if (arm, row['id']) in completed: continue
                trial = dest / row['id']
                if trial.exists():
                    assert args.resume and not (trial / 'result.json').exists()
                    failed = dest / 'failed-attempts'
                    failed.mkdir(exist_ok=True)
                    trial.rename(failed / (row['id'] + '-' + str(time.time_ns())))
                trial.mkdir(exist_ok=False)
                src = Path(row['case_dir']) / 'tokens.npy'
                assert ref.sha(src) == row['tokens_sha256']
                tokens = np.load(src).tolist()
                assert len(tokens) == row['input_tokens']
                start_log = log.stat().st_size
                with (args.out / 'resources.jsonl').open('a') as f:
                    resources = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                                          '--format=csv,noheader'], text=True)
                    f.write(json.dumps({'arm': arm, 'trial': row['id'], 'server_pid': server.pid,
                                        'gpu_processes': resources}) + '\n')
                save(args.out / 'state.json', {'phase': 'running', 'arm': arm, 'trial': row['id'],
                     'completed': len(results), 'total': len(cohort) * len(args.arms), 'server_pid': server.pid})

                def request(input_tokens, tag, maximum=0, query=None):
                    payload = {'model': Path(cmd[cmd.index('--model') + 1]).name, 'prompt': input_tokens,
                               'max_tokens': maximum, 'temperature': 1., 'top_p': .95, 'top_k': 20,
                               'seed': row['seed'], 'enable_thinking': maximum > 0,
                               'thinking_budget': args.thinking_budget, 'thinking_summary': False,
                               'kvmem_trace_tag': row['id'] + '-' + tag, 'kvmem_reselect': 'auto' if query else 'off'}
                    if query: payload['kvmem_query_token_span'] = query
                    begin = time.monotonic()
                    response = http.post(base + '/v1/completions', json=payload, timeout=(10, 7200))
                    elapsed = time.monotonic() - begin
                    if not response.ok: raise RuntimeError(response.text[:4000])
                    result = response.json()
                    assert result['usage']['prompt_tokens'] == len(input_tokens)
                    if maximum == 0:
                        assert result['usage']['completion_tokens'] == 0
                        assert result['choices'][0]['finish_reason'] == 'prefill_only'
                    save(trial / (tag + '.json'), {'wall_seconds': elapsed, 'response': result})
                    return result, elapsed

                request([100, 101, 102, row['seed']], 'reset')
                boundary = row['query_begin'] // 32 * 32
                for i, end in enumerate(list(range(65536, boundary, 65536)) + [boundary]):
                    request(tokens[:end], f'ingest-{i}')
                response, elapsed = request(tokens, 'answer', args.max_output,
                                           {'begin': row['query_begin'], 'end': row['query_end']})
                with log.open('rb') as f:
                    f.seek(start_log)
                    logs = f.read().decode(errors='replace')
                (trial / 'server.log').write_text(logs)
                blends = [dict((k, float(v)) for k, v in re.findall(r'(\w+)=([\d.]+)', line))
                          for line in logs.splitlines() if '[kvmem-blend]' in line]
                if arm == 'on':
                    assert any(x.get('tokens', 0) > 0 for x in blends), 'blend never executed'
                    for event in blends:
                        assert event['tokens'] <= event['window_tokens'] * args.ratio
                        assert event.get('missing_source_tokens', 0) == 0
                else: assert not blends, 'disabled arm executed blend'
                assert 'native kvmem query replay:' in logs, 'query replay did not run'
                assert '[thinking-compaction]' not in logs
                result = {**{k: row[k] for k in ['id', 'task', 'depth', 'input_tokens', 'seed']},
                          'arm': arm, 'wall_seconds': elapsed, 'blend': blends,
                          'usage': response['usage'], 'finish_reason': response['choices'][0]['finish_reason'],
                          **ref.score(args.reference, response['choices'][0]['text'], row['answers'])}
                save(trial / 'result.json', result)
                results.append(result)
                save(args.out / 'results.json', results)
                print(json.dumps({k: result[k] for k in ['arm', 'task', 'depth', 'ruler_score', 'finish_reason', 'wall_seconds']}), flush=True)
        except BaseException as error:
            save(args.out / 'state.json', {'phase': 'failed', 'arm': arm, 'completed': len(results), 'error': repr(error)})
            raise
        finally:
            server.terminate()
            try: server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    groups = []
    for arm in args.arms:
        for task in ref.TASKS:
            rows = [r for r in results if r['arm'] == arm and r['task'] == task]
            groups.append({'arm': arm, 'task': task, 'count': len(rows),
                           'score': sum(r['ruler_score'] for r in rows) / len(rows),
                           'all_correct': sum(r['all_answers_found'] for r in rows),
                           'mean_answer_seconds': sum(r['wall_seconds'] for r in rows) / len(rows)})
    save(args.out / 'summary.json', groups)
    save(args.out / 'state.json', {'phase': 'complete', 'completed': len(results)})


if __name__ == '__main__':
    main()
