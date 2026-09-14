"""Deferred same-weight qw3/llama.cpp prefill check; requires exclusive GPU.

This is an FP16-KV, non-speculative numerical diagnostic, not pilot throughput.
llama.cpp receives the exact token IDs dumped by qw3, avoiding template or
tokenizer differences. Results require interpretation; no tolerance is inferred
from the run and no free-running completion is treated as an oracle.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path('/home/chaidi/qw3')
HERE = ROOT / 'benchmark/qwen35_small_20260914'
OUT = HERE / 'reference_logits_audit'
QW3 = HERE / 'frozen/qw3'
LLAMA = Path('/home/chaidi/llama.cpp/build/bin/llama-server')
PORT = 18561
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def wait_gpu():
    deadline = time.monotonic() + 60
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                  '--format=csv,noheader'], text=True).strip():
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU is occupied; no concurrent diagnostic admitted')
        time.sleep(.5)


def api(path, payload=None):
    request = urllib.request.Request('http://127.0.0.1:' + str(PORT) + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with OPENER.open(request, timeout=180) as response:
        return json.load(response)


def compare(native, response):
    rows = response.get('completion_probabilities') or response.get('probs')
    if not rows or len(rows) != 1 or not rows[0].get('top_logprobs'):
        raise RuntimeError('Missing reference raw first-token log probabilities')
    reference = {x['id']: x['logprob'] for x in rows[0]['top_logprobs']}
    actual = {x['id']: x['logit'] for x in native['top']}
    assert all(math.isfinite(x) for x in [*reference.values(), *actual.values()])
    common = sorted(actual.keys() & reference.keys())
    if len(common) < 2:
        raise RuntimeError('Too few common top tokens; inspect model/token bindings')
    # A logit and a log probability differ by a shared log-normalizer. Remove
    # only that scalar, reporting overlap separately so missing tokens cannot
    # make the numerical difference appear smaller without being disclosed.
    offsets = [actual[x] - reference[x] for x in common]
    offset = sum(offsets) / len(offsets)
    errors = [x - offset for x in offsets]
    ref_best = max(reference, key=reference.get)
    return dict(qw3_top1=native['argmax_token'], llama_top1=ref_best,
        top1_agrees=native['argmax_token'] == ref_best,
        common_top_tokens=len(common), qw3_top_tokens=len(actual),
        llama_top_tokens=len(reference),
        centered_common_logit_rmse=math.sqrt(sum(x*x for x in errors)/len(errors)),
        centered_common_logit_max_abs=max(map(abs, errors)),
        qw3_top1_margin=sorted(actual.values(), reverse=True)[0] - sorted(actual.values(), reverse=True)[1],
        llama_top1_margin=sorted(reference.values(), reverse=True)[0] - sorted(reference.values(), reverse=True)[1])


def main():
    OUT.mkdir(exist_ok=True)
    cases = {
        'short': 'The capital of France is',
        'code': 'def unique_in_order(values):\n    """Return unique values, preserving their first occurrence."""\n    seen = set()\n',
        'chat': '<|im_start|>user\nReturn the result of 17 * 19.\n<|im_end|>\n<|im_start|>assistant\n<think>\n',
        'history_1k': ''.join(f'Observation {i}: service worker_{i % 7} returned status=200.\n' for i in range(64)) + 'The observations indicate',
        'history_2k': ''.join(f'Observation {i}: service worker_{i % 7} returned status=200.\n' for i in range(128)) + 'The observations indicate',
    }
    models = {s: ROOT / f'models/Qwen3.5-{s}-Q8_0-with-MTP.gguf' for s in ('9B', '4B')}
    models['27B'] = ROOT / 'models/Qwen3.8-27B-Q8_0.gguf'
    identity = dict(qw3_sha256=sha(QW3), llama_sha256=sha(LLAMA),
        models={s: dict(path=str(p), sha256=sha(p)) for s, p in models.items()},
        cases=cases, script_sha256=sha(Path(__file__)), kv='FP16', mtp=0,
        qw3_prefill='sequential per-token logits diagnostic',
        llama_prefill='batched, 512 tokens per microbatch',
        note='Numerical diagnostic only; formal pilot remains FP8 KV, MTP=4.')
    if (OUT/'identity.json').exists():
        assert json.loads((OUT/'identity.json').read_text()) == identity, 'Cached audit identity changed'
    save(OUT/'identity.json', identity)
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')
           and not k.startswith(('QW3_', 'LLAMA_ARG_'))}
    state = dict(status='running', pid=os.getpid(), started_at=time.time())
    results = []
    save(OUT/'status.json', state)
    try:
        for size, model in models.items():
            for name, prompt in cases.items():
                directory = OUT/size/name
                directory.mkdir(parents=True, exist_ok=True)
                prompt_path = directory/'prompt.txt'
                prompt_path.write_text(prompt)
                dump = directory/'qw3_logits.jsonl'
                if not (directory/'qw3_run.json').exists():
                    wait_gpu()
                    command = [str(QW3), '--model', str(model), '--raw', '--kv-dtype', 'fp16',
                        '--mtp-chain','0','--temp','0','--top-k','1','--seed','1000',
                        '--prefill-chunk','512','-c','16384','-n','1',
                        '--prompt-file',str(prompt_path),'--dump-logits',str(dump),
                        '--dump-logits-top-k','32']
                    state.update(model=size, case=name, engine='qw3'); save(OUT/'status.json',state)
                    with (directory/'qw3.log').open('w') as log:
                        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=log,timeout=600,check=True)
                    save(directory/'qw3_run.json',dict(command=command,returncode=0))
            wait_gpu()
            command = [str(LLAMA),'-m',str(model),'-ngl','99','-c','16384','-np','1',
                '-b','512','-ub','512','-ctk','f16','-ctv','f16','-fa','on',
                '--no-warmup','--host','127.0.0.1','--port',str(PORT)]
            state.update(model=size, engine='llama.cpp'); save(OUT/'status.json', state)
            with (OUT/size/'llama.log').open('a') as log:
                proc = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=log)
                save(OUT/size/'llama_run.json',dict(command=command,pid=proc.pid))
                try:
                    deadline=time.monotonic()+180
                    while True:
                        if proc.poll() is not None: raise RuntimeError('llama.cpp server exited before readiness')
                        try:
                            if api('/health').get('status') == 'ok': break
                        except (urllib.error.URLError,TimeoutError): pass
                        if time.monotonic()>=deadline: raise TimeoutError('llama.cpp readiness')
                        time.sleep(.5)
                    for name in cases:
                        directory=OUT/size/name
                        records=[json.loads(x) for x in (directory/'qw3_logits.jsonl').read_text().splitlines()]
                        tokens=next(x['tokens'] for x in records if x['event']=='prompt')
                        native=next(x for x in records if x['event']=='step'
                                    and x['phase']=='prefill' and x['step']==len(tokens)-1)
                        assert native['input_token']==tokens[-1]
                        assert 0 < len(tokens) < 16000
                        payload=dict(prompt=tokens,n_predict=1,temperature=-1,n_probs=128,
                            return_tokens=True,cache_prompt=False,post_sampling_probs=False,seed=1000)
                        save(directory/'llama_request.json',payload)
                        response=api('/completion',payload)
                        save(directory/'llama_response.json',response)
                        assert response.get('tokens_evaluated')==len(tokens) and not response.get('truncated'),response
                        results.append(dict(model=size,case=name,prompt_tokens=len(tokens),**compare(native,response)))
                        save(OUT/'comparison.json',results)
                finally:
                    proc.terminate()
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill();proc.wait()
        state.update(status='complete_pending_interpretation',finished_at=time.time())
    except BaseException as error:
        state.update(status='needs_attention',error=repr(error));raise
    finally:
        save(OUT/'status.json',state)


if __name__ == '__main__':
    main()
