#!/usr/bin/env python3
"""Frozen 9B policy comparison; no old reader answers are reused."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import run_qwen35_longmemeval as b

ROOT = b.ROOT
OUT = ROOT / 'benchmark/longmemeval_aligned_9b_20260915'
PREVIOUS = ROOT / 'benchmark/longmemeval_qwen35_9b_matched102_20260915'
MODELS = {'9b': ROOT / 'models/Qwen3.5-9B-Q8_0-with-MTP.gguf'}
COMMON = {
    'kv_dtype': 'fp8', 'logical_capacity': 262144, 'active_prompt_budget': 32768,
    'generation_reserve': 32768, 'answer_max_tokens': 32768, 'thinking_budget': 4096,
    'answer_sampling': {'temperature': 1.0, 'top_p': .95, 'top_k': 20,
        'min_p': 0.0, 'presence_penalty': 1.5, 'repetition_penalty': 1.0, 'seed': 1000},
    'summary_sampling': {'temperature': 0.0, 'top_p': .95, 'top_k': 20,
        'min_p': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'seed': 1000},
    'mtp_chain': 4, 'prefill_chunk': 2048,
    'kvmem': {'block_tokens': 32, 'sink_tokens': 256, 'recent_tokens': 0,
        'retrieval': 'mean-k', 'index_placement': 'gpu', 'gpu_memory_ratio': 1.0,
        'cpu_gb': 16, 'nvme_gb': 0, 'gc': False,
        'history_ingest': 'question-blind raw-token session',
        'query_span': 'literal question tokens', 'query_replay': 'full'},
    'compact_rag': {'embedder': 'jina-embeddings-v2-small-en', 'device': 'cpu',
        'embedding_threads': 8, 'chunk_tokens': 1024, 'overlap': 128,
        'summary_new_history_tokens': 24576, 'summary_prompt_tokens': 32768,
        'summary_max_tokens': 2048, 'summary_thinking': False,
        'uncompressed_tail': False, 'retrieval': 'rank, fit total 32K, chronological order'},
    'judge': {'model': 'deepseek-flash', 'version': 'V4.1 Flash',
        'thinking': False, 'temperature': 0, 'max_tokens': 16},
    'timing': 'fresh server per arm; loading excluded from service phase costs',
}
ORIGINAL_GENERATE = b.generate
IDENTITY = None


def generate(base, prompt, directory, name, *, thinking=True, max_tokens=None, extra=None):
    limit = COMMON['answer_max_tokens'] if max_tokens is None else max_tokens
    if name == 'answer':
        assert limit == 32768 and thinking
    elif name.startswith('compact_'):
        assert limit == 2048 and not thinking
    sampling = COMMON['answer_sampling'] if thinking else COMMON['summary_sampling']
    return ORIGINAL_GENERATE(base, prompt, directory, name, thinking=thinking,
        max_tokens=limit, extra={**sampling, **(extra or {})})


def server_command(model, policy, port):
    command = [str(b.BINARY), 'serve', '--model', str(model), '--host', '127.0.0.1',
        '--port', str(port), '--ctx', '262144', '--kv-dtype', 'fp8',
        '--prefill-chunk', '2048', '--temp', '1.0', '--top-p', '0.95', '--top-k', '20',
        '--seed', '1000', '--thinking-budget', '4096', '--mtp-chain', '4',
        '--native-mtp-speculate', '--enable-thinking']
    if policy == 'kvmem':
        command += ['--kvmem', '--kvmem-prefix-cache', '--kvmem-block-tokens', '32',
            '--kvmem-budget', '32768', '--kvmem-prefill-budget', '32768',
            '--kvmem-gen-budget', '32768', '--kvmem-sink-tokens', '256',
            '--kvmem-recent-tokens', '0', '--kvmem-method', 'retrieval',
            '--kvmem-retrieval-method', 'mean-k', '--kvmem-index-placement', 'gpu',
            '--kvmem-update-mode', 'step', '--kvmem-query-conditioned', '--kvmem-immutable-k',
            '--kvmem-gpu-memory-ratio', '1.0', '--kvmem-cpu-gb', '16',
            '--no-kvmem-raw-k-nvme', '--kvmem-nvme-gb', '0', '--kvmem-opt-stage-out', 'on',
            '--kvmem-opt-stage-in', 'on', '--kvmem-opt-pack', 'on']
    else:
        assert policy == 'compact_rag'
        command += ['--continuous-batching', '--prefix-cache', '--max-active', '1',
                    '--max-pending', '8', '--mtp-batched-draft', '--mtp-paged-prefix']
    return command


def gpu_pids():
    return subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).split()


@contextlib.contextmanager
def server(directory, policy):
    assert not gpu_pids(), 'Another GPU process is active'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = server_command(b.MODEL, policy, port)
    for path in (b.MODEL, b.BINARY):
        stat = path.stat()
        recorded = IDENTITY['files'][str(path)]
        assert [stat.st_size, stat.st_mtime_ns] == recorded['stat'], 'Frozen model/binary changed'
    b.save(directory / 'server_manifest.json', {'command': command, 'started_at': time.time(),
        'binary_sha256': IDENTITY['files'][str(b.BINARY)]['sha256'],
        'model_sha256': IDENTITY['files'][str(b.MODEL)]['sha256'], 'common_parameters': COMMON})
    with (directory / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=b.environment(), stdout=log,
                                stderr=log, start_new_session=True)
        try:
            base = f'http://127.0.0.1:{port}'
            for _ in range(120):
                assert proc.poll() is None, f'Server exited: {directory}'
                try:
                    with b.HTTP.open(base + '/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Server readiness')
            yield base
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            for _ in range(120):
                if str(proc.pid) not in gpu_pids():
                    break
                time.sleep(.5)


def prepare():
    global IDENTITY
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'common': COMMON, 'models': {k: str(v) for k, v in MODELS.items()},
        'sample_description': 'Exact archived 102 questions, 17 per type; both 9B policies freshly generated.',
        'scope': '9B KVMem versus 9B compaction+RAG only; no 27B inference requested.',
        'protocol': 'Shared current protocol, not a claim to reproduce the incomplete old RAG settings.',
        'official_sampling_sources': ['https://huggingface.co/Qwen/Qwen3.5-9B'],
        'sampling_note': 'Both methods use the 9B general-thinking recipe, including presence_penalty=1.5.',
        'index_reuse': 'Only model-independent Jina indices reused; their measured CPU preparation time retains original provenance.'}
    if (OUT / 'plan.json').exists():
        assert json.loads((OUT / 'plan.json').read_text()) == plan
    else:
        b.save(OUT / 'plan.json', plan)
    for tag in MODELS:
        target = OUT / tag
        target.mkdir(exist_ok=True)
        for filename in ('inference_samples.json', 'references.json', 'sample_ids.txt'):
            src, dst = PREVIOUS / filename, target / filename
            if not dst.exists():
                shutil.copy2(src, dst)
            assert b.sha(src) == b.sha(dst)
        b.save(target / 'plan.json', {**plan, 'reader_model': tag})
        if not (target / 'indices').exists():
            (target / 'indices').symlink_to(PREVIOUS / 'indices', target_is_directory=True)
        # Reuse only calibration fixtures, never any local or external candidate scores.
        if not (target / 'judge/calibration.json').exists():
            b.save(target / 'judge/calibration.json', json.loads((PREVIOUS / 'judge/calibration.json').read_text()))
        b.save(target / 'status.json', {'status': 'queued', 'planned_questions_this_invocation': 102})
    files = [Path(__file__).resolve(), Path(b.__file__).resolve(), b.RAG_WORKER, b.BINARY,
             *MODELS.values(), b.TOKENIZER / 'tokenizer.json', b.TOKENIZER / 'tokenizer_config.json',
             OUT / 'plan.json', OUT / '9b/inference_samples.json', OUT / '9b/references.json']
    IDENTITY = {'files': {str(p): {'sha256': b.sha(p), 'stat': [p.stat().st_size, p.stat().st_mtime_ns]}
                           for p in files}}
    if (OUT / 'identity.json').exists():
        assert json.loads((OUT / 'identity.json').read_text()) == IDENTITY
    else:
        b.save(OUT / 'identity.json', IDENTITY)
    tok = b.AutoTokenizer.from_pretrained(b.TOKENIZER, local_files_only=True)
    for tag, model in MODELS.items():
        b.MODEL = model
        b.save(OUT / tag / 'raw_token_vocabulary_audit.json', b.validate_raw_token_vocabulary(tok))
    commands = [server_command(MODELS['9b'], policy, 1234) for policy in ('kvmem', 'compact_rag')]
    # Only the policy-specific suffix may differ between the two server launches.
    common_end = commands[0].index('--enable-thinking') + 1
    assert commands[0][:common_end] == commands[1][:common_end]
    b.save(OUT / 'alignment_preflight.json', {'passed': True,
        'common_server_args_equal_between_policies': True, 'same_input_files': True,
        'same_hf_token_id_mapping': True, 'answer_output_cap': 32768,
        'shared_protocol': COMMON})
    return tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    tok = prepare()
    if args.prepare_only:
        print('9B paired-policy parameter/vocabulary preflight passed', flush=True)
        return
    state = {'status': 'waiting_for_previous_run', 'pid': os.getpid(), 'started_at': time.time()}
    b.save(OUT / 'status.json', state)
    while not (OUT / 'gpu_ready.json').exists():
        time.sleep(2)
    assert not gpu_pids()
    samples = json.loads((OUT / '9b/inference_samples.json').read_text())
    if args.limit:
        samples = samples[:args.limit]
    b.generate, b.server = generate, server
    b.OUT = OUT
    embed = None
    try:
        embed = b.Embeddings()
        for i, sample in enumerate(samples):
            qid = sample['question_id']
            for tag in (list(MODELS) if i % 2 == 0 else list(reversed(MODELS))):
                b.OUT, b.MODEL = OUT / tag, MODELS[tag]
                for policy in (('kvmem', 'compact_rag') if i % 2 == 0 else ('compact_rag', 'kvmem')):
                    directory = b.OUT / 'trials' / qid / policy
                    if (directory / 'result.json').exists():
                        continue
                    assert not directory.exists(), f'Incomplete arm requires review: {directory}'
                    directory.mkdir(parents=True)
                    state.update(status='running', model=tag, question_id=qid, policy=policy, ordinal=i)
                    b.save(OUT / 'status.json', state)
                    b.save(b.OUT / 'status.json', state)
                    result = (b.run_kvmem(sample, tok, directory) if policy == 'kvmem' else
                              b.run_compact_rag(sample, tok, directory, embed))
                    result.update(question_id=qid, question_type=sample['question_type'],
                                  policy=policy, model=MODELS[tag].name, status='completed')
                    b.save(directory / 'result.json', result)
                    print(json.dumps({'model': tag, 'question_id': qid, 'policy': policy,
                        'finish_reason': result['finish_reason'], 'output_tokens': result['usage']['completion_tokens']}), flush=True)
        state.update(status='inference_complete', finished_at=time.time())
        for tag in MODELS:
            b.save(OUT / tag / 'status.json', state)
    except BaseException as exc:
        state.update(status='needs_attention', error=repr(exc), finished_at=time.time())
        raise
    finally:
        if embed:
            embed.close()
        b.save(OUT / 'status.json', state)


if __name__ == '__main__':
    main()
