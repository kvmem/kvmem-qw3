#!/usr/bin/env python3
"""Blind local-27B grading using the official LongMemEval prompt function.

This is explicitly a local-judge pilot, not an official leaderboard score.
"""
import argparse
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from run_qwen35_longmemeval import OUT, TOKENIZER, AutoTokenizer, generate, ids, save, server, sha, template

OFFICIAL = Path('/home/chaidi/kvmem_eval/KVMem_Motivation/scripts/official/evaluate_qa.py')


def prompt_function():
    tree = ast.parse(OFFICIAL.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'get_anscheck_prompt']
    assert len(functions) == 1
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(OFFICIAL), 'exec'), namespace)
    return namespace['get_anscheck_prompt']


def verdict(text):
    value = text.strip().lower().strip('.!')
    if value not in ('yes', 'no'):
        raise RuntimeError(f'Invalid local judge answer: {text!r}')
    return value == 'yes'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    refs = {r['question_id']: r for r in json.loads((OUT / 'references.json').read_text())}
    samples = json.loads((OUT / 'inference_samples.json').read_text())
    if args.limit:
        samples = samples[:args.limit]
    jobs = []
    for sample in samples:
        for policy in ('kvmem', 'compact_rag'):
            path = OUT / 'trials' / sample['question_id'] / policy / 'result.json'
            assert path.exists(), f'Missing paired output: {path}'
            result = json.loads(path.read_text())
            assert result['status'] == 'completed'
            jobs.append((sample['question_id'], policy, path, result))
    jobs.sort(key=lambda row: hashlib.sha256(('blind-judge-20260915:' + row[0] + ':' + row[1]).encode()).hexdigest())
    root = OUT / 'judge'
    tok = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    check = prompt_function()
    identity = {'script': sha(Path(__file__)), 'official_prompt_source': sha(OFFICIAL),
                'references': sha(OUT / 'references.json')}
    if (root / 'identity.json').exists():
        assert json.loads((root / 'identity.json').read_text()) == identity
    else:
        save(root / 'identity.json', identity)
    run = root / f'run_{int(time.time())}'
    status = {'status': 'running', 'started_at': time.time(), 'pairs': len(samples)}
    save(root / 'status.json', status)
    rows = []
    try:
        with server(run, 'judge') as base:
            def assess(ref, response, directory, name):
                prompt = check(ref['question_type'], ref['question'], ref['answer'], response,
                               abstention=ref['question_id'].endswith('_abs'))
                rendered = ids(tok, template(tok, [{'role': 'user', 'content': prompt}], thinking=False))
                assert len(rendered) <= 16000
                result = generate(base, rendered, directory, name, thinking=False, max_tokens=16)
                return verdict(result['answer']), result

            calibration_path = root / 'calibration.json'
            if not calibration_path.exists():
                calibration = []
                selected = {}
                for ref in refs.values():
                    key = 'abstention' if ref['question_id'].endswith('_abs') else ref['question_type']
                    selected.setdefault(key, ref)
                # Preference references are rubrics, not valid assistant replies.
                # Use an independent fixture with an actual personalized answer.
                selected['single-session-preference'] = {
                    'question_id': 'calibration_preference',
                    'question_type': 'single-session-preference',
                    'question': 'Can you suggest something I could cook for dinner?',
                    'answer': 'The user is vegetarian and likes spicy food. Suggest a spicy vegetarian meal; do not suggest meat or fish.',
                }
                for group, ref in selected.items():
                    positive = ('The conversation does not contain the information needed to answer this question.'
                                if group == 'abstention' else ref['answer'])
                    negative = ('The answer is 987654321, and the record states it unambiguously.'
                                if group == 'abstention' else 'I do not know the answer.')
                    if group == 'single-session-preference':
                        positive = 'Since you are vegetarian and enjoy spicy food, try a chickpea and spinach curry with extra chili, served over rice.'
                        negative = 'Try a beef steak with chicken gravy.'
                    for expected, answer in ((True, positive), (False, negative)):
                        got, result = assess(ref, answer, run / 'calibration', f'{group}_{int(expected)}')
                        calibration.append({'group': group, 'expected': expected, 'observed': got,
                                            'fixture': ref, 'hypothesis': answer,
                                            'response': result['answer']})
                save(calibration_path, {'cases': calibration, 'passed': all(r['expected'] == r['observed'] for r in calibration)})
            assert json.loads(calibration_path.read_text())['passed'], 'Local judge calibration failed'
            for qid, policy, path, result in jobs:
                destination = root / 'scores' / f'{qid}_{policy}.json'
                if destination.exists():
                    row = json.loads(destination.read_text())
                    assert row['prediction_sha256'] == sha(path)
                else:
                    correct, scored = assess(refs[qid], result['answer'], run / 'answers', f'{qid}_{policy}')
                    row = {'question_id': qid, 'question_type': refs[qid]['question_type'],
                           'policy': policy, 'correct': correct, 'raw_judge_response': scored['answer'],
                           'prediction_sha256': sha(path), 'hypothesis': result['answer']}
                    save(destination, row)
                rows.append(row)
        summary = []
        for policy in ('kvmem', 'compact_rag'):
            group = [r for r in rows if r['policy'] == policy]
            answers = [r for _, p, _, r in jobs if p == policy]
            types = defaultdict(list)
            for row in group:
                types['abstention' if row['question_id'].endswith('_abs') else row['question_type']].append(row['correct'])
            summary.append({'policy': policy, 'questions': len(group), 'correct': sum(r['correct'] for r in group),
                'accuracy': statistics.mean(r['correct'] for r in group),
                'mean_answer_request_sec': statistics.mean(r['request_wall_sec'] for r in answers),
                'mean_final_query_total_sec': statistics.mean(r['final_query_total_sec'] for r in answers),
                'by_stratum': {k: {'correct': sum(v), 'n': len(v)} for k, v in types.items()}})
        save(OUT / 'scored_results.json', rows)
        save(OUT / 'summary.json', {'judge': 'local Qwen3.8-27B Q8, official per-type prompt, temperature zero',
             'is_official_leaderboard_score': False, 'questions': len(samples), 'methods': summary,
             'notes': ['Fixed balanced pilot, not the natural full-test distribution.',
                       'Final-query time excludes previously built history memory/index; phase costs remain in raw results.']})
        status.update(status='complete', finished_at=time.time())
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        status.update(status='needs_attention', error=repr(error))
        raise
    finally:
        save(root / 'status.json', status)


if __name__ == '__main__':
    main()
