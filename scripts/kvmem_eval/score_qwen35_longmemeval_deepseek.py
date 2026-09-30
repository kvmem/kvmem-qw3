#!/usr/bin/env python3
"""Grade completed paired outputs through the official DeepSeek API, without a GPU.

Credentials are read with getpass (or from DEEPSEEK_API_KEY), never persisted.
The local judge and its reports are preserved. --watch grades newly finished
pairs until the fixed experiment plan is complete.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'benchmark/longmemeval_qwen35_9b_matched102_20260915'
OFFICIAL = Path('/home/chaidi/kvmem_eval/KVMem_Motivation/scripts/official/evaluate_qa.py')
API_BASE = 'https://api.deepseek.com'
PARAMETERS = {'model': 'deepseek-flash', 'thinking': {'type': 'disabled'},
              'temperature': 0, 'max_tokens': 16, 'stream': False}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prompt_function():
    functions = [n for n in ast.parse(OFFICIAL.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == 'get_anscheck_prompt']
    assert len(functions) == 1
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(OFFICIAL), 'exec'), namespace)
    return namespace['get_anscheck_prompt']


def verdict(raw):
    choice = raw['choices'][0]
    value = (choice['message'].get('content') or '').strip().lower().strip('.!')
    if choice['finish_reason'] != 'stop' or value not in ('yes', 'no'):
        raise RuntimeError('Judge returned an incomplete or non-binary verdict; inspect saved response')
    return value == 'yes'


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Client:
    def __init__(self, key):
        self.key = key

    def request(self, endpoint, body=None):
        # Do not forward credentials through a proxy or to a redirected host.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirects())
        request = urllib.request.Request(API_BASE + endpoint,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        started = time.monotonic()
        for attempt in range(4):
            try:
                with opener.open(request, timeout=60) as response:
                    result = json.load(response)
                return result, {'wall_sec': time.monotonic() - started, 'attempts': attempt + 1}
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f'DeepSeek HTTP {code}; no response headers or credentials logged') from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise RuntimeError('DeepSeek network request failed after four attempts') from None
            time.sleep(2 ** attempt)


def jobs_snapshot(out, samples):
    jobs = []
    for sample in samples:
        pair = []
        for policy in ('kvmem', 'compact_rag'):
            path = out / 'trials' / sample['question_id'] / policy / 'result.json'
            if not path.exists():
                break
            result = json.loads(path.read_text())
            if result.get('status') != 'completed':
                break
            pair.append({'question_id': sample['question_id'], 'policy': policy,
                         'prediction_sha256': sha(path), 'answer': result['answer']})
        if len(pair) == 2:
            jobs.extend(pair)
    return sorted(jobs, key=lambda j: digest('blind-judge-20260915:' + j['question_id'] + ':' + j['policy']))


def assess(client, check, ref, answer, path):
    prompt = check(ref['question_type'], ref['question'], ref['answer'], answer,
                   abstention=ref['question_id'].endswith('_abs'))
    body = {**PARAMETERS, 'messages': [{'role': 'user', 'content': prompt}]}
    save(path.with_suffix('.request.json'), body)  # Intentionally excludes HTTP headers.
    raw, timing = client.request('/chat/completions', body)
    save(path.with_suffix('.response.json'), {'response': raw, **timing})
    model = raw.get('model', '').lower()
    if model != 'deepseek-flash' and not model.startswith('deepseek-v4.1-flash'):
        raise RuntimeError(f'Unexpected returned judge model: {model}')
    return verdict(raw), raw, digest(body)


def methods(rows):
    result = []
    for policy in ('kvmem', 'compact_rag'):
        arm = [r for r in rows if r['policy'] == policy]
        types = {}
        for row in arm:
            group = types.setdefault(row['question_type'], {'n': 0, 'correct': 0})
            group['n'] += 1
            group['correct'] += int(row['correct'])
        correct = sum(r['correct'] for r in arm)
        result.append({'policy': policy, 'n': len(arm), 'correct': correct,
                       'accuracy': correct / len(arm) if arm else None, 'by_type': types})
    return result


def export(out, root, batch, rows, planned):
    comparison = []
    for row in rows:
        path = out / 'judge/scores' / f"{row['question_id']}_{row['policy']}.json"
        if path.exists():
            local = json.loads(path.read_text())
            assert row['prediction_sha256'] == local['prediction_sha256']
            comparison.append({**row, 'local_correct': local['correct'],
                               'judges_agree': row['correct'] == local['correct']})
    paired_local = {r['question_id'] for r in comparison
                    if sum(s['question_id'] == r['question_id'] for s in comparison) == 2}
    overlap = [r for r in comparison if r['question_id'] in paired_local]
    summary = {'judge': 'DeepSeek V4.1 Flash (official API)', 'parameters': PARAMETERS,
        'scored_pairs': len(rows) // 2, 'planned_pairs': planned,
        'complete': len(rows) == 2 * planned, 'methods': methods(rows),
        'response_models': sorted({r['response_model'] for r in rows}),
        'local_judge_overlap': {'paired_questions': len(paired_local),
            'deepseek': methods(overlap),
            'local_27b': methods([{**r, 'correct': r['local_correct']} for r in overlap]),
            'changed_verdicts': [{k: r[k] for k in ('question_id', 'policy', 'correct', 'local_correct')}
                                 for r in overlap if not r['judges_agree']]},
        'notes': ['Official LongMemEval per-type prompts, custom judge; not an official leaderboard score.',
                  'Interim completed subset; type proportions may be unbalanced.',
                  'Only questions, reference answers/rubrics and final hypotheses are sent to the judge.',
                  'Method names, full histories and thinking traces are not sent.']}
    fields = ['question_id', 'question_type', 'policy', 'correct', 'raw_judge_response',
              'response_model', 'prediction_sha256', 'hypothesis']
    for destination in (batch, root / 'report'):
        destination.mkdir(parents=True, exist_ok=True)
        save(destination / 'summary.json', summary)
        save(destination / 'scored_results.json', rows)
        save(destination / 'local_judge_comparison.json', comparison)
        with (destination / 'paired_results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({'scored_pairs': len(rows) // 2, 'methods': summary['methods'],
                      'snapshot': str(batch)}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prompt-key', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.workers <= 4
    out = args.output_root.resolve()
    root = out / 'judge_deepseek_v41_flash'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = getpass.getpass('DeepSeek API key (hidden): ') if args.prompt_key else os.environ.get('DEEPSEEK_API_KEY', '')
    if not key.strip():
        raise RuntimeError('No DeepSeek API key supplied')
    client = Client(key.strip())
    refs = {r['question_id']: r for r in json.loads((out / 'references.json').read_text())}
    samples = json.loads((out / 'inference_samples.json').read_text())
    fixtures = [{k: r[k] for k in ('fixture', 'hypothesis', 'expected')}
                for r in json.loads((out / 'judge/calibration.json').read_text())['cases']]
    identity = {'script': sha(Path(__file__)), 'official_prompt': sha(OFFICIAL),
                'references': sha(out / 'references.json'), 'api_base': API_BASE,
                'parameters': PARAMETERS, 'calibration_fixtures_sha256': digest(fixtures)}
    if (root / 'identity.json').exists():
        assert json.loads((root / 'identity.json').read_text()) == identity
    else:
        save(root / 'identity.json', identity)
    status = {'status': 'connecting', 'pid': os.getpid(), 'started_at': time.time(),
              'planned_pairs': len(samples), 'watch': args.watch}
    save(root / 'status.json', status)
    try:
        models, _ = client.request('/models')
        available = [m['id'] for m in models['data']]
        assert PARAMETERS['model'] in available, 'Requested model not available on this API account'
        save(root / 'model_availability.json', {'checked_at': time.time(), 'available_ids': available,
            'version_documentation': 'https://api-docs.deepseek.com/news/news260910/'})
        print('Connected to official DeepSeek API; deepseek-flash is available', flush=True)
        check = prompt_function()
        if not (root / 'calibration.json').exists():
            status.update(status='calibrating')
            save(root / 'status.json', status)
            cases = []
            for i, fixture in enumerate(fixtures):
                label, raw, _ = assess(client, check, fixture['fixture'], fixture['hypothesis'],
                                       root / 'calibration' / f'case_{i:02d}')
                cases.append({**fixture, 'observed': label, 'response_model': raw['model']})
            save(root / 'calibration.json', {'cases': cases,
                 'passed': all(c['observed'] == c['expected'] for c in cases)})
        assert json.loads((root / 'calibration.json').read_text())['passed'], 'DeepSeek calibration failed'
        print('Calibration passed; scoring paired outputs', flush=True)
        previous = None
        while True:
            jobs = jobs_snapshot(out, samples)
            signature = digest(jobs)
            if jobs and signature != previous:
                batch = root / 'runs' / f"{time.time_ns()}_{len(jobs)//2}_pairs"
                save(batch / 'selection.json', {'created_at': time.time(), 'planned_pairs': len(samples),
                    'predictions': [{k: j[k] for k in ('question_id', 'policy', 'prediction_sha256')} for j in jobs]})
                status.update(status='grading', selected_pairs=len(jobs)//2, snapshot=str(batch))
                save(root / 'status.json', status)

                def grade(job):
                    qid, policy = job['question_id'], job['policy']
                    path = out / 'trials' / qid / policy / 'result.json'
                    score_path = root / 'scores' / f'{qid}_{policy}.json'
                    if score_path.exists():
                        row = json.loads(score_path.read_text())
                        assert row['prediction_sha256'] == job['prediction_sha256'] == sha(path)
                        return row
                    label, raw, request_sha = assess(client, check, refs[qid], job['answer'],
                        batch / 'requests' / digest(qid + ':' + policy))
                    assert sha(path) == job['prediction_sha256'], 'Prediction changed during grading'
                    row = {'question_id': qid, 'question_type': refs[qid]['question_type'],
                        'policy': policy, 'correct': label, 'hypothesis': job['answer'],
                        'prediction_sha256': job['prediction_sha256'], 'request_sha256': request_sha,
                        'response_model': raw['model'], 'response_id': raw.get('id'),
                        'system_fingerprint': raw.get('system_fingerprint'),
                        'raw_judge_response': raw['choices'][0]['message']['content'], 'usage': raw.get('usage')}
                    save(score_path, row)
                    return row

                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    rows = list(pool.map(grade, jobs))
                summary = export(out, root, batch, rows, len(samples))
                previous = signature
                status.update(scored_pairs=summary['scored_pairs'], updated_at=time.time())
            if len(jobs) == len(samples) * 2 or not args.watch:
                status.update(status='complete' if len(jobs) == len(samples) * 2 else 'snapshot_complete',
                              finished_at=time.time())
                break
            status.update(status='waiting_for_completed_pairs')
            save(root / 'status.json', status)
            inference = json.loads((out / 'status.json').read_text())
            if inference.get('status') == 'needs_attention':
                raise RuntimeError('Inference needs attention; completed scores preserved')
            time.sleep(30)
    except Exception as exc:
        status.update(status='needs_attention', error=str(exc).replace(client.key, '[REDACTED]'),
                      finished_at=time.time())
        print(status['error'], file=sys.stderr, flush=True)
        return 1
    finally:
        save(root / 'status.json', status)
    return 0


if __name__ == '__main__':
    sys.exit(main())
