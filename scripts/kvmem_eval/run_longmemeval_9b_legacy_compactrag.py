#!/usr/bin/env python3
"""Run the archived 27B summary/RAG/prompt code with the 9B reader.

The legacy package is an immutable git export. Valid responses retain the
original pipeline; invalid generations are terminal sample failures, never
reasoning-as-summary fallbacks. Legacy sampling is the default; the explicit
Qwen3.5 policy keeps the pipeline but uses its recommended thinking sampler.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'benchmark/longmemeval_9b_legacy27b_compactrag_20260915'
DATA = OUT / '9b'
SOURCE = OUT / 'legacy_source'
OLD = Path('/home/chaidi/kvmem_eval/KVMem_Motivation')
BINARY = ROOT / 'benchmark/qwen35_small_20260914/frozen/qw3'
MODEL = ROOT / 'models/Qwen3.5-9B-Q8_0-with-MTP.gguf'
TOKENIZER = ROOT / 'models/Qwen3.5-9B-BF16'
OLD_TOKENIZER = OLD / 'models/tokenizers/Qwen3.6-27B-FP8'
EMBED = Path('/data/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2'
             '/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SAMPLING_POLICY = 'legacy'
RECOMMENDED_SAMPLING = {'temperature':1.0, 'top_p':.95, 'top_k':20,
                        'min_p':0.0, 'presence_penalty':1.5, 'repetition_penalty':1.0}


def validate_request(payload):
    assert 'max_tokens' not in payload
    assert payload['chat_template_kwargs'] == {'enable_thinking':True}
    if SAMPLING_POLICY == 'qwen35-thinking':
        for key, value in RECOMMENDED_SAMPLING.items():
            assert payload[key] == value, key
    else:
        assert payload['temperature'] == 0


class ModelGenerationFailure(Exception):
    def __init__(self, reason, stage, response_path):
        super().__init__(reason)
        self.reason, self.stage, self.response_path = reason, stage, response_path


def checked_content(response, stage, path):
    """Do not mistake a broken SSE stream for a model failure or a summary."""
    finish = response.get('finish_reason')
    if finish not in ('stop', 'length'):
        raise RuntimeError(f'Incomplete/abnormal transport response: {path} ({finish})')
    content = response['raw']['choices'][0]['message'].get('content') or ''
    if stage == 'summary' and finish == 'length':
        raise ModelGenerationFailure('summary_output_limit', stage, path)
    if not content.strip():
        raise ModelGenerationFailure(stage + '_empty_final_content', stage, path)
    # A nonempty length-terminated final answer is retained and judged, with
    # its finish reason visible. A truncated memory must never feed another call.
    return content


def previous_model_failure(directory):
    for path in sorted(directory.glob('*.response.json')):
        stage = path.name.split('_')[0]
        if stage not in ('summary', 'answer'):
            continue
        response = json.loads(path.read_text())
        if response.get('finish_reason') not in ('stop', 'length'):
            continue  # Interrupted attempts are not terminal model failures.
        try:
            checked_content(response, stage, path)
        except ModelGenerationFailure as exc:
            return exc
    return None


def failure_result(sample, failure):
    response = json.loads(failure.response_path.read_text())
    return {'status':'failed', 'terminal_failure':True, 'failure_kind':'model_generation',
        'failure_reason':failure.reason, 'failure_stage':failure.stage,
        'failure_response':str(failure.response_path.resolve()),
        'failure_response_sha256':sha(failure.response_path),
        'question_id':sample['question_id'], 'question_type':sample['question_type'],
        'model':'Qwen3.5-9B-Q8', 'policy':'compact_rag', 'protocol':'legacy27b',
        'answer':'', 'finish_reason':response['finish_reason'],
        'failed_request_usage':{'prompt_tokens':response['input_tokens'],
                               'completion_tokens':response['output_tokens']},
        'failed_request_wall_sec':response['wall_sec'],
        'answer_generated':False, 'protocol_revision':'failure-handling-v2',
        'sampling_policy':SAMPLING_POLICY,
        'note':'Count as unsolved in the full cohort. No reasoning fallback, parameter change, or accuracy-driven retry.'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_legacy():
    spec = importlib.util.spec_from_file_location('legacy_lme', SOURCE / 'scripts/common/__init__.py',
             submodule_search_locations=[str(SOURCE / 'scripts/common')])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from legacy_lme import runner, compact, retrieval, tokenization, llm_client
    return runner, compact, retrieval, tokenization, llm_client


def config(base):
    c = json.loads((SOURCE / 'configs/exp2_addttft_retrieval_k22.yaml').read_text())
    c.update(input_path=str(DATA / 'inference_samples.json'),
             summary_output_path=str(DATA / 'compact_summaries.jsonl'),
             output_path=str(DATA / 'retrieval_outputs.jsonl'),
             retrieval_output_path=str(DATA / 'retrieval_top22_sessions.jsonl'))
    c['llm'].update(api_base=base + '/v1', model_name='Qwen3.5-9B-Q8', api_key_env=None)
    c['tokenizer']['name'] = str(TOKENIZER)
    c['compact']['compact_prompt_path'] = str(SOURCE / 'docs/codex_compact_sources/prompt.md')
    c['retrieval']['embedding_model'] = str(EMBED)
    if SAMPLING_POLICY == 'qwen35-thinking':
        c['llm']['temperature'] = RECOMMENDED_SAMPLING['temperature']
        # The legacy summary runner overrides llm.temperature from this field.
        c['compact']['summary_temperature'] = RECOMMENDED_SAMPLING['temperature']
        c['llm']['extra_body'].update({k:v for k,v in RECOMMENDED_SAMPLING.items()
                                     if k != 'temperature'})
    return c


def command(port):
    # Omit both output and thinking caps, as did the old RAG client/service
    # launch. The hard remaining context is the native server's output limit.
    return [str(BINARY), 'serve', '--model', str(MODEL), '--host', '127.0.0.1',
            '--port', str(port), '--ctx', '262144', '--kv-dtype', 'fp8',
            '--prefill-chunk', '2048', '--temp', '0', '--top-p', '.95',
            '--top-k', '20', '--seed', '0', '--enable-thinking',
            '--mtp-chain', '4', '--native-mtp-speculate', '--continuous-batching',
            '--prefix-cache', '--max-active', '1', '--max-pending', '8',
            '--mtp-batched-draft', '--mtp-paged-prefix']


def preflight():
    manifest = json.loads((OUT / 'legacy_source_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        assert sha(SOURCE / name) == digest, f'Legacy source changed: {name}'
    runner, compact, retrieval, tokenization, client = load_legacy()
    import torch
    torch.set_num_threads(8)
    assert not torch.cuda.is_available(), 'Retriever must stay on CPU'
    a = tokenization._load_hf_tokenizer(str(OLD_TOKENIZER))
    b = tokenization._load_hf_tokenizer(str(TOKENIZER))
    assert a.get_vocab() == b.get_vocab()
    sample = [{'role': 'user', 'content': 'History\nQuestion: example\nAnswer:'}]
    assert a.apply_chat_template(sample, tokenize=False, add_generation_prompt=True,
            enable_thinking=True) == b.apply_chat_template(sample, tokenize=False,
            add_generation_prompt=True, enable_thinking=True)
    c = config('http://127.0.0.1:1')
    original = json.loads((SOURCE / 'configs/exp2_addttft_retrieval_k22.yaml').read_text())
    assert c['llm']['max_tokens'] is None
    expected_extra = dict(original['llm']['extra_body'])
    if SAMPLING_POLICY == 'qwen35-thinking':
        expected_extra.update({k:v for k,v in RECOMMENDED_SAMPLING.items() if k != 'temperature'})
    assert c['llm']['extra_body'] == expected_extra
    expected_compact = dict(original['compact'])
    if SAMPLING_POLICY == 'qwen35-thinking':
        expected_compact['summary_temperature'] = RECOMMENDED_SAMPLING['temperature']
    assert {k:v for k,v in c['compact'].items() if k != 'compact_prompt_path'} == {
        k:v for k,v in expected_compact.items() if k != 'compact_prompt_path'}
    assert {k:v for k,v in c['retrieval'].items() if k != 'embedding_model'} == {
        k:v for k,v in original['retrieval'].items() if k != 'embedding_model'}
    payload = client.create_llm_client(c)._base_payload('a complete legacy prompt')
    validate_request(payload)
    summary_config = {**c, 'llm':{**c['llm'], 'max_tokens':None,
                                'temperature':c['compact']['summary_temperature']}}
    validate_request(client.create_llm_client(summary_config)._base_payload('a summary prompt'))
    assert 'max_tokens' not in payload and payload['messages'] == [
        {'role': 'user', 'content': 'a complete legacy prompt'}]
    embed = retrieval._load_embedding_model(str(EMBED))
    assert str(embed.device) == 'cpu' and embed.max_seq_length == 256
    # Preserve the original embedder's default 256-token encoding truncation.
    # Altering it to fit the 1024-token retrieval chunks would change the method.
    import sentence_transformers, transformers
    save(OUT / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'sentence_transformers': sentence_transformers.__version__,
        'embedder': str(EMBED), 'embedder_max_seq_length': embed.max_seq_length,
        'embedder_device': str(embed.device), 'threads': torch.get_num_threads()})
    samples = {s['question_id']:s for s in json.loads((DATA / 'inference_samples.json').read_text())}
    audit_path = OUT / 'legacy_boundary_audit.json'
    if not audit_path.exists():
        archived = [json.loads(l) for l in (OLD / 'outputs/exp1/compact_summaries.jsonl').read_text().splitlines()]
        checks = []
        for old in archived:
            assert old['compact_rounds'] == 1
            x = compact.build_compacted_context(samples[old['question_id']], 100000, 4000,
                    str(TOKENIZER), summarize_fn=lambda text: old['compact_summary'])
            old_events = [v for v in old['compact_events'] if v['event'] == 'overflow_compact']
            new_events = [v for v in x['compact_events'] if v['event'] == 'overflow_compact']
            assert x['remaining_sessions'] == old['remaining_sessions'], old['question_id']
            assert x['compact_rounds'] == old['compact_rounds']
            for u,v in zip(old_events,new_events):
                for key in ['overflow_at_session_index','compacted_session_indices',
                            'summary_source_tokens','trial_tokens_when_overflow']:
                    assert u[key] == v[key], (old['question_id'],key)
            checks.append({'question_id':old['question_id'], 'passed':True,
                           'overflow_session_indices':[v['overflow_at_session_index'] for v in new_events]})
            print('Archived boundary verified:', old['question_id'], flush=True)
        save(audit_path, {'passed':True, 'checks':checks,
            'note':'Saved summaries used only as fixed fixtures for boundary validation, never for 9B inference.'})
    assert json.loads(audit_path.read_text())['passed']
    files = [Path(__file__), BINARY, MODEL, TOKENIZER/'tokenizer.json',
             TOKENIZER/'tokenizer_config.json', DATA/'inference_samples.json']
    files += [p for p in EMBED.rglob('*') if p.is_file() and p.name != 'README.md']
    identity_path = OUT/'identity_v2.json'
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        for p in files:
            assert [p.stat().st_size,p.stat().st_mtime_ns] == identity[str(p)]['stat'], str(p)
    else:
        save(identity_path, {str(p):{'sha256':sha(p), 'stat':[p.stat().st_size,p.stat().st_mtime_ns]} for p in files})
    save(OUT/'plan_v2.json', {'scope':'Rerun only 9B Compaction+RAG using archived 27B pipeline.',
        'failure_policy':'Reject truncated/empty summaries; never use reasoning fallback; count terminal model failures as unsolved and continue next sample. Transport errors require attention.',
        'legacy_revision':manifest['revision'], 'samples':102, 'compact_threshold_tokens':100000,
        'sampling_policy':SAMPLING_POLICY,
        'summary_and_answer_temperature':c['llm']['temperature'], 'summary_and_answer_thinking':True,
        'sampling_overrides':RECOMMENDED_SAMPLING if SAMPLING_POLICY == 'qwen35-thinking' else {},
        'max_tokens':'omitted; native server uses remaining 262144-token context',
        'thinking_budget':'uncapped', 'uncompressed_tail':True, 'retrieval_top_k':22,
        'retrieval_order':'descending similarity (legacy order)', 'retrieval_chunk_size':1024,
        'retrieval_chunk_overlap':128, 'embedder':'all-MiniLM-L6-v2',
        'final_input_cap':'No added 32K cap; retain legacy assembly.',
        'server_command_template':command(0), 'model_independent_token_ids_equal':True,
        'limitations':['Old service was vLLM; current reader uses existing 9B Q8 qw3, FP8 KV and MTP=4.',
          'Old exact service defaults and embedder revision were not archived; do not claim engine-level model-only control.',
          'Existing KVMem results are references, with their own already frozen prompt/sampling protocol.']})
    return runner, client, b


@contextlib.contextmanager
def server():
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
    assert not active, f'GPU already in use by {active}'
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port = s.getsockname()[1]
    cmd = command(port)
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')
           and not k.startswith('QW3_') and k != 'CUDA_VISIBLE_DEVICES'}
    with (OUT/'server.log').open('a') as log:
        proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
        save(OUT/'server_manifest.json', {'pid':proc.pid, 'command':cmd, 'started_at':time.time()})
        base = f'http://127.0.0.1:{port}'
        try:
            for _ in range(120):
                assert proc.poll() is None, 'Native server exited during startup'
                try:
                    with HTTP.open(base+'/health',timeout=1) as res:
                        if res.status == 200:break
                except OSError:time.sleep(1)
            else:raise TimeoutError('Server startup')
            yield base
        finally:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()


def main():
    global OUT, DATA, SAMPLING_POLICY
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output-root',type=Path,default=OUT)
    parser.add_argument('--sampling-policy',choices=['legacy','qwen35-thinking'],default='legacy')
    args = parser.parse_args()
    OUT = args.output_root.resolve(); DATA = OUT/'9b'; SAMPLING_POLICY = args.sampling_policy
    OUT.mkdir(parents=True,exist_ok=True)
    # A sampler correction defines a new cohort, never a retry-until-correct mix.
    for result_path in (DATA/'trials').glob('*/compact_rag/result.json'):
        recorded_policy = json.loads(result_path.read_text()).get('sampling_policy','legacy')
        if recorded_policy != SAMPLING_POLICY:
            raise RuntimeError('Sampling policy differs from existing results; use a fresh output root')
    lock = (OUT/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = {'status':'preflight','pid':os.getpid(),'started_at':time.time(),'planned_samples':102}
    def status(**kw):
        state.update(**kw); save(OUT/'status.json',state);save(DATA/'status.json',state)
    def counts():
        rows = [json.loads(p.read_text()) for p in (DATA/'trials').glob('*/compact_rag/result.json')]
        return {'processed_samples':len(rows),
                'completed_samples':sum(r['status']=='completed' for r in rows),
                'failed_samples':sum(r['status']=='failed' for r in rows)}
    status()
    try:
        runner, client, tok = preflight()
        if args.prepare_only:
            status(status='prepared');return
        current = {}
        old_generate = client.OpenAICompatibleLLMClient.generate
        old_request = client.OpenAICompatibleLLMClient._request
        def recorded_request(self,payload):
            validate_request(payload)
            save(current['directory']/(current['name']+'.request.json'),payload)
            return old_request(self,payload)
        def recorded_generate(self,prompt,metadata=None):
            idx = len(list(current['directory'].glob(current['stage']+'_*.request.json')))
            current['name'] = f"{current['stage']}_{idx:02d}"
            messages = [{'role':'user','content':prompt}]
            rendered = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            expected_count = len(tok.encode(rendered))
            current['directory'].mkdir(parents=True,exist_ok=True)
            (current['directory']/(current['name']+'.prompt.txt')).write_text(prompt)
            t = time.perf_counter()
            result = old_generate(self,prompt,metadata)
            wall = time.perf_counter()-t
            response_path = current['directory']/(current['name']+'.response.json')
            response = {
                'text':result.text, 'input_tokens':result.input_tokens,'output_tokens':result.output_tokens,
                'finish_reason':result.finish_reason,'wall_sec':wall,'timing':result.timing,'raw':result.raw,
                'hf_prompt_tokens':expected_count,'native_prompt_tokens':result.input_tokens,
                'prompt_token_count_matches':expected_count==result.input_tokens}
            save(response_path,response)
            current['responses'].append({'stage':current['stage'],'wall_sec':wall,
                'finish_reason':result.finish_reason,'token_count_matches':expected_count==result.input_tokens})
            result.text = checked_content(response,current['stage'],response_path)
            return result
        client.OpenAICompatibleLLMClient._request = recorded_request
        client.OpenAICompatibleLLMClient.generate = recorded_generate
        with server() as base:
            c = config(base);save(OUT/'effective_config.json',c)
            samples = json.loads((DATA/'inference_samples.json').read_text())
            for i,sample in enumerate(samples):
                qid = sample['question_id'];directory = DATA/'trials'/qid/'compact_rag'
                if (directory/'result.json').exists():continue
                current.update(directory=directory,stage='summary',responses=[])
                status(status='running',ordinal=i,question_id=qid,phase='summary',**counts())
                one = {**c,'question_ids':[qid]}
                try:
                    previous = previous_model_failure(directory)
                    if previous:
                        raise previous
                    runner.make_compact_summaries(one)
                    current['stage'] = 'answer';status(phase='retrieval_and_answer')
                    _, rows = runner.run_retrieval(one)
                except ModelGenerationFailure as failure:
                    save(directory/'result.json',failure_result(sample,failure))
                    status(phase='sample_failed',**counts())
                    print(json.dumps({'question_id':qid,'status':'failed','reason':failure.reason}),flush=True)
                    continue
                row = next(r for r in rows if r['question_id']==qid)
                summaries = [json.loads(l) for l in (DATA/'compact_summaries.jsonl').read_text().splitlines()]
                summary = next(r for r in summaries if r['question_id']==qid)
                result = {'status':'completed','model':'Qwen3.5-9B-Q8','question_id':qid,
                    'question_type':sample['question_type'],'policy':'compact_rag','protocol':'legacy27b',
                    'sampling_policy':SAMPLING_POLICY,
                    'answer':row['hypothesis'],'finish_reason':row['finish_reason'],
                    'usage':{'prompt_tokens':row['input_tokens'],'completion_tokens':row['output_tokens']},
                    'request_wall_sec':row['llm_elapsed_sec'], 'final_query_total_sec':row['end_to_end_elapsed_sec'],
                    'timing_note':'Legacy end-to-end field includes history embedding during retrieval.',
                    'compaction_rounds':summary['compact_rounds'],
                    'compaction_request_sec':sum(r['wall_sec'] for r in current['responses'] if r['stage']=='summary'),
                    'legacy_row':row}
                save(directory/'result.json',result)
                save(directory/'protocol_audit.json',{'requests':current['responses'],
                    'legacy_source_manifest':str(OUT/'legacy_source_manifest.json')})
                status(phase='sample_complete',**counts())
                print(json.dumps({'completed':i+1,'question_id':qid,'finish_reason':row['finish_reason'],
                    'prompt_tokens':row['input_tokens'],'output_tokens':row['output_tokens']}),flush=True)
        status(status='inference_complete',finished_at=time.time(),**counts())
    except BaseException as exc:
        status(status='needs_attention',error=repr(exc),finished_at=time.time())
        raise


if __name__ == '__main__':
    main()
