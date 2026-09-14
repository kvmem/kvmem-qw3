"""Export independently auditable throughput rows and matched-model summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
FORMAL = HERE/'formal'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric(line):
    return {k: float(v) for k, v in re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?)', line)}


def export(partial=False):
    state = json.loads((FORMAL/'status.json').read_text())
    if not partial:
        assert state['status'] == 'complete' and len(state['completed']) == 24
    rows, steps = [], []
    for path in sorted(FORMAL.glob('opt_*.json')):
        d = json.loads(path.read_text())
        if d['status'] != 'complete':
            continue
        a = d['arguments']
        method = 'InfiniGen' if a['scheme'] == 'infinigen' else f'FlexGen {a["kv"].upper()} KV'
        count = a['batch']*(a['gen']-1)
        assert len(d['costs_s']) == a['gen'] and len(d['calls']) == 3
        assert d['calls'][1]['role'] == 'decode_warmup'
        row = dict(family='OPT', method=method, model='OPT-6.7B', weight_dtype='FP16', kv_dtype='FP16', mtp=0,
                   history_tokens=a['prompt'], active_policy='selected KV <=400 in accelerated layers' if a['scheme']=='infinigen' else 'full history',
                   batch_size=a['batch'], rep=int(path.stem.rsplit('_r',1)[1]), threads=a['threads'],
                   output_tokens=a['batch']*a['gen'], timed_output_tokens=count, decode_s=d['decode_s'],
                   decode_tokens_per_s=count/d['decode_s'], prefill_s=d['prefill_s'],
                   timing_scope='original synchronized generate timer, excluding prefill and decode warmup',
                   torch_peak_gib=d['torch_peak_gib'], result_file=str(path), result_sha256=sha(path))
        rows.append(row)
        for index, seconds in enumerate(d['costs_s'][1:], 1):
            steps.append(dict(family='OPT', method=method, batch_size=a['batch'], history_tokens=a['prompt'], rep=row['rep'],
                              decode_step=index, step_s=seconds, outputs_in_step=a['batch'], source=str(path)))
    for path in sorted(FORMAL.glob('mistral_*.json')):
        d = json.loads(path.read_text())
        if d['status'] != 'complete' and not partial:
            continue
        method = 'InfLLM' if d['arguments']['method']=='infllm' else 'HF SDPA GPU KV'
        for r in d['results']:
            count = r['generated_tokens']-1
            assert count == len(r['step_s']) == 127
            row = dict(family='Mistral', method=method, model='Mistral-7B-Instruct-v0.2', weight_dtype='BF16', kv_dtype='BF16', mtp=0,
                       history_tokens=r['prompt_tokens'], active_policy='4096 local + 128 init + 16x128 retrieved' if method=='InfLLM' else 'full history',
                       batch_size=1, rep=r['rep'], threads=r['threads'], output_tokens=r['generated_tokens'], timed_output_tokens=count,
                       decode_s=r['decode_s'], decode_tokens_per_s=count/r['decode_s'], prefill_s=r['prefill_s'],
                       timing_scope='continuous synchronized decode loop, including selection/transfer/sampling/history append',
                       torch_peak_gib=r['torch_peak_gib'], result_file=str(path), result_sha256=sha(path),
                       input_sha256=r['input_ids_sha256'], output_ids_sha256=hashlib.sha256(json.dumps(r['output_ids']).encode()).hexdigest())
            rows.append(row)
            for index, seconds in enumerate(r['step_s'], 1):
                steps.append(dict(family='Mistral', method=method, batch_size=1, history_tokens=r['prompt_tokens'], rep=r['rep'],
                                  decode_step=index, step_s=seconds, outputs_in_step=1, source=str(path)))
    for path in sorted(FORMAL.glob('qwen_*/result.json')):
        d = json.loads(path.read_text())
        if d['status'] != 'complete':
            continue
        log = path.parent/'server.log'
        timings = [numeric(line) for line in log.read_text().splitlines() if '[qw3-server-ttft]' in line]
        assert len(timings)==len(d['results'])
        for r, timing in zip(d['results'], timings):
            if r['warmup']:
                continue
            assert timing['model_token_seen']==1 and timing['request_total_ms']>timing['server_ttft_ms']
            assert r['usage']['completion_tokens']==128
            # The no-MTP native "decode" log includes post-prefill selection.
            # Do not silently treat that timer as pure decode. First model
            # token -> server completion removes prefill/initial selection;
            # it conservatively retains final cache bookkeeping/serialization.
            seconds = (timing['request_total_ms']-timing['server_ttft_ms'])/1000
            phase = r['phases']
            method = 'KVMem' if 'kvmem' in path.parent.name else 'qw3 GPU KV'
            row = dict(family='Qwen', method=method, model='Qwen3.8-27B', weight_dtype='Q8_0', kv_dtype='FP8', mtp=d['mtp_chain'],
                       history_tokens=r['prompt_tokens'], active_policy='128K selection + 128K generation reserve' if method=='KVMem' else 'full history',
                       batch_size=1, rep=r['rep'], threads='native defaults', output_tokens=128, timed_output_tokens=127,
                       decode_s=seconds, decode_tokens_per_s=127/seconds, prefill_s=phase['prefill_ms']/1000,
                       timing_scope='first model token to server completion; includes final bookkeeping/serialization',
                       native_reported_decode_s=phase['decode_ms']/1000,
                       native_reported_decode_scope='includes post-prefill maintenance on plain no-MTP path' if d['mtp_chain']==0 and method=='KVMem' else 'native decode timer',
                       server_ttft_ms=timing['server_ttft_ms'], server_request_total_ms=timing['request_total_ms'],
                       private_guided_query_ms=phase.get('post_guided_query_ms','not separately logged'),
                       query_replay_ms=phase.get('post_query_replay_ms','not separately logged'),
                       mtp_drafted=phase.get('mtp_drafted',0), mtp_accepted=phase.get('mtp_accepted',0),
                       result_file=str(path), result_sha256=sha(path), server_log_sha256=sha(log))
            rows.append(row)
    if not partial:
        assert len(rows)==84, len(rows)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ['family','method','model','mtp','batch_size','history_tokens'])
        groups.setdefault(key, []).append(row)
    summaries=[]
    for key, values in sorted(groups.items()):
        if not partial:
            assert len(values)==3 and {x['rep'] for x in values}=={0,1,2}, key
        rates=[v['decode_tokens_per_s'] for v in values]
        summaries.append(dict(zip(['family','method','model','mtp','batch_size','history_tokens'],key),
                              n=len(values), median_tokens_per_s=statistics.median(rates), min_tokens_per_s=min(rates), max_tokens_per_s=max(rates),
                              pooled_tokens_per_s=sum(x['timed_output_tokens'] for x in values)/sum(x['decode_s'] for x in values),
                              median_prefill_s=statistics.median(x['prefill_s'] for x in values), timing_scope=values[0]['timing_scope']))
    destination=HERE/('partial_export' if partial else 'export')
    destination.mkdir(exist_ok=True)
    for name, records in [('runs',rows),('summary',summaries),('decode_steps',steps)]:
        with (destination/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in records))))
            writer.writeheader();writer.writerows(records)
    audit={'complete':not partial,'runs':len(rows),'groups':len(summaries),
           'native_interval_correction':'No-MTP plain native decode log includes initial reselection. Use first model token to server completion and 127 remaining outputs; preserve original reported timer separately. This includes final bookkeeping and is not a kernel-only timer.',
           'cross_model_speedup_claim':False,'accuracy_claim':False,'author_sources_unchanged':{},'frozen_inputs_passed':True}
    for p,h in json.loads((FORMAL/'frozen.json').read_text()).items():
        assert sha(Path(p))==h,p
    import subprocess
    for name in ['InfiniGen','InfLLM']:
        source=HERE.parent/'upstream'/name
        dirty=subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True)
        assert not dirty
        audit['author_sources_unchanged'][name]=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    (destination/'audit.json').write_text(json.dumps(audit,indent=2))
    (destination/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(audit,indent=2))
    return summaries


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true')
    export(p.parse_args().partial)
