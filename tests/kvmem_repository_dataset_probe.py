#!/usr/bin/env python3
"""Bounded-GC retention experiment on fixed MemoryAgentBench QA data.

This is a separate 4B ablation, not a replacement for the locked 9B/27B runs.
The original documents, first 16 questions of rows 0 and 1, prompt templates,
and official substring metric are frozen before inference. Documents arrive
incrementally, followed by append-only question/answer turns using actual replies. Off/shadow retain the
whole corpus, while delete uses a smaller physical repository with identical
active/generation budgets. Report accuracy losses as well as gains.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--parquet', type=Path, required=True)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18501)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts/kvmem_eval'))
    from prepare_memoryagentbench_archive_row import official_chunks, MEMORIZE, QUERY, SYSTEM_MESSAGE
    sys.path.insert(0, str(args.official_repo))
    from utils.eval_other_utils import post_process
    head = subprocess.check_output(['git', '-C', str(args.official_repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != '455306dcabc3842526eb83cd4e225e5d486c5c5d':
        raise RuntimeError('official evaluator revision changed')
    args.out.mkdir(parents=True, exist_ok=False)
    # Immutable executable: later builds cannot alter a running matrix.
    binary = args.out / 'qw3-frozen'
    shutil.copy2(args.binary, binary)
    shutil.copy2(__file__, args.out / 'driver.py')
    corpus = pq.read_table(args.parquet).slice(0, 2).to_pylist()
    rows = []
    for index, row in enumerate(corpus):
        source = row['metadata']['source']
        if not source.startswith('ruler_qa'):
            raise RuntimeError('canonical QA rows changed')
        chunks = [MEMORIZE['ruler'].format(context=x, time_stamp='2026-08-02 00:00:00')
                  for x in official_chunks(row['context'], 2048)]
        questions = [{'index': q, 'id': row['metadata']['qa_pair_ids'][q],
                      'text': QUERY['ruler'].format(question=row['questions'][q]),
                      'answers': row['answers'][q]} for q in range(16)]
        rows.append({'row': index, 'source': source, 'chunks': chunks, 'questions': questions})
    (args.out / 'dataset.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (args.out / 'reference_lock.json').write_text(json.dumps({
        'parquet': str(args.parquet), 'parquet_sha256': sha(args.parquet),
        'dataset_sha256': sha(args.out / 'dataset.json'), 'official_commit': head,
        'evaluator_sha256': sha(args.official_repo / 'utils/eval_other_utils.py'),
        'template_sha256': sha(root / 'scripts/kvmem_eval/prepare_memoryagentbench_archive_row.py'),
        'binary_sha256': sha(binary), 'driver_sha256': sha(__file__),
        'scope': 'separate 4B online-retention ablation; not the official archive protocol or 9B/27B benchmark',
        'active_tokens': 16384, 'generation_tokens': 8192, 'block_tokens': 32,
        'sampling': 'greedy, thinking disabled', 'output_tokens': 128,
        'ordering': 'rows 0,1; first 16 questions as append-only conversation; actual answers retained; no ground truth in prompts; no retries for wrong answers',
        'protocol': 'online-retention-v2; continued dialogue, not frozen independent archive queries'
    }, indent=2))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f'http://127.0.0.1:{args.port}'

    def request(path, body=None):
        req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
        with opener.open(req, timeout=600) as response:
            return json.load(response)

    summary = []
    for row in rows:
        for mode, capacity in [('off', 524288), ('shadow', 524288), ('delete', 65536)]:
            if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                raise RuntimeError('GPU occupied; do not overlap inference')
            out = args.out / f'row{row["row"]}' / mode
            out.mkdir(parents=True)
            cmd = [str(binary.resolve()), 'serve', '--model', str(args.model.resolve()),
                '--host', '127.0.0.1', '--port', str(args.port), '--ctx', str(capacity),
                '--prefill-chunk', '512', '--paged-kv', '--kv-page-size', '16', '--kv-dtype', 'fp8',
                '--mtp-chain', '4', '--kvmem', '--kvmem-repository-blocks', str(capacity // 32),
                '--kvmem-repository-gc', mode, '--kvmem-block-tokens', '32',
                '--kvmem-budget', '16384', '--kvmem-gen-budget', '8192',
                '--kvmem-sink-blocks', '16', '--kvmem-recent-blocks', '32',
                '--kvmem-index-placement', 'cpu', '--kvmem-cpu-gb', '24',
                '--kvmem-gpu-memory-ratio', '0.10', '--kvmem-query-conditioned',
                '--kvmem-guided-reselect', 'both', '--kvmem-guided-query-tokens', '64',
                '--kvmem-middecode-trigger-tokens', '4096']
            (out / 'command.json').write_text(json.dumps(cmd, indent=2))
            env = {k: v for k, v in os.environ.items() if not k.startswith('QW3_')
                   and k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
            env.update(QW3_KVMEM_PREFIX_CACHE_TRACE='1', QW3_KVMEM_GUIDED_MTP='1')
            with (out / 'server.log').open('w') as log:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    (out / 'owned_pid.json').write_text(json.dumps({'pid': proc.pid, 'start': time.time()}))
                    deadline = time.monotonic() + 120
                    while True:
                        if proc.poll() is not None: raise RuntimeError('server startup failed')
                        try:
                            model = request('/v1/models')['data'][0]['id']
                            break
                        except OSError:
                            if time.monotonic() >= deadline: raise RuntimeError('server startup timed out')
                            time.sleep(.25)
                    messages = []
                    body = {'model': model, 'system': SYSTEM_MESSAGE, 'temperature': 0,
                            'thinking': {'type': 'disabled'}, 'messages': messages}
                    answers = []
                    with (out / 'responses.jsonl').open('w') as response_log:
                        for i, chunk in enumerate(row['chunks']):
                            messages.append({'role': 'user', 'content': chunk})
                            body['max_tokens'] = 1
                            response = request('/v1/messages', body)
                            response_log.write(json.dumps({'ingest': i, 'response': response}) + '\n'); response_log.flush()
                            if mode != 'delete' and response['usage']['input_tokens'] + 128 > capacity:
                                raise RuntimeError('reference does not retain full input')
                            messages.append({'role': 'assistant', 'content': 'I have learned the documents.'})
                            (out / 'status.json').write_text(json.dumps({'ingested': i + 1, 'chunks': len(row['chunks']), 'answers': 0}))
                        for q in row['questions']:
                            messages.append({'role': 'user', 'content': q['text']})
                            body['max_tokens'] = 128
                            response = request('/v1/messages', body)
                            response_log.write(json.dumps({'question': q['id'], 'response': response}) + '\n'); response_log.flush()
                            answer = ''.join(x.get('text', '') for x in response['content'] if x['type'] == 'text')
                            metrics, info = post_process({'output': answer}, q['answers'], {'sub_dataset': row['source']})
                            answers.append(dict(q, answer=answer, metrics=metrics, stop_reason=response['stop_reason']))
                            (out / 'answers.json').write_text(json.dumps(answers, indent=2, default=lambda x: x.item()))
                            # Preserve the actual reply to keep a real append-only
                            # dialogue. Replacing old questions can force cold
                            # rebuild and reread forgotten evidence, masking GC loss.
                            messages.append({'role': 'assistant', 'content': response['content']})
                            (out / 'status.json').write_text(json.dumps({'ingested': len(row['chunks']), 'answers': len(answers)}))
                    result = {'row': row['row'], 'source': row['source'], 'mode': mode,
                              'correct': sum(bool(x['metrics']['substring_exact_match']) for x in answers),
                              'questions': len(answers), 'complete': True}
                    summary.append(result)
                    (args.out / 'results.json').write_text(json.dumps(summary, indent=2))
                    print(json.dumps(result), flush=True)
                finally:
                    proc.terminate()
                    try: proc.wait(timeout=20)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()


if __name__ == '__main__':
    main()
