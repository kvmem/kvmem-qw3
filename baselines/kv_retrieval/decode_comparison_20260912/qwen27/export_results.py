"""Export matched Qwen model results, retaining method and timing boundaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(line):
    return {k:float(v) for k,v in re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?)',line)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());rows=[];steps=[]
    if not a.partial:
        state=json.loads((Path(plan['output'])/'status.json').read_text())
        assert state['status']=='complete' and len(state['completed'])==len(plan['jobs'])
    for job in plan['jobs']:
        path=Path(job['result'])
        if not path.exists():
            assert a.partial;continue
        data=json.loads(path.read_text())
        if data['status']!='complete':
            assert a.partial;continue
        native=path.name=='result.json'
        timings=[]
        if native:
            original_plan=json.loads((HERE.parents[3]/'benchmark/osworld_v2/claudecode_remaining_20260912/plan.json').read_text())
            assert data['binary_sha256']==original_plan['binary_sha256']
            assert digest(Path(data['command'][0]))==data['binary_sha256']
            log=path.parent/'server.log'
            timings=[numbers(s) for s in log.read_text().splitlines() if '[qw3-server-ttft]' in s]
            assert len(timings)==len(data['results'])
        for index,r in enumerate(data['results']):
            if r.get('warmup'):continue
            count=(r['usage']['completion_tokens'] if native else r['generated_tokens'])-1
            assert count>0
            row=dict(model='Qwen3.8-27B',weight_dtype='BF16',kv_storage_dtype='FP8 E4M3FN',mtp=0,batch=1,
                     method=job['name'],history_tokens=r['prompt_tokens'],rep=r['rep'],timed_output_tokens=count,
                     input_ids_sha256=r['input_ids_sha256'],source=str(path),source_sha256=digest(path),
                     cpu_affinity=json.dumps(data['cpu_affinity']),
                     configuration=json.dumps(job.get('configuration',{}),sort_keys=True))
            if native:
                t=timings[index]
                assert t['model_token_seen']==1 and t['request_total_ms']>t['server_ttft_ms']
                seconds=(t['request_total_ms']-t['server_ttft_ms'])/1000
                row.update(backend='native qw3',decode_s=seconds,decode_tokens_per_s=count/seconds,
                           prefill_s=r['phases']['prefill_ms']/1000,server_ttft_s=t['server_ttft_ms']/1000,
                           native_logged_decode_s=r['phases']['decode_ms']/1000,
                           request_wall_s=r['request_wall_s'],
                           timing='first model token to server completion; includes final bookkeeping and serialization',
                           native_log_sha256=digest(log))
            else:
                assert data['no_qw3_projection_library_loaded'] and len(r['step_s'])==count
                row.update(backend='HF/FLA/PyTorch',decode_s=r['decode_s'],decode_tokens_per_s=count/r['decode_s'],
                           prefill_s=r['prefill_s'],initial_offload_s=r['initial_offload_s'],graph_setup_s=r['graph_setup_s'],
                           timing='continuous decode including selection/transfers/sampling; excludes prefill and initial offload',
                           threads=r['threads'],peak_torch_allocated_gib=r['peak_torch_allocated_gib'],
                           mean_selected_tokens_per_head=r.get('mean_selected_tokens_per_head'),
                           h2d_mib_per_forward=r.get('h2d_mib_per_forward'),
                           output_ids_sha256=hashlib.sha256(json.dumps(r['output_ids']).encode()).hexdigest())
                for step,seconds in enumerate(r['step_s'],1):
                    steps.append(dict(method=job['name'],history_tokens=r['prompt_tokens'],rep=r['rep'],decode_step=step,seconds=seconds,source=str(path)))
            rows.append(row)
    groups={}
    for r in rows:groups.setdefault((r['method'],r['history_tokens'],r['timed_output_tokens']),[]).append(r)
    if not a.partial:
        assert len(rows)==plan['expected_rows'],len(rows)
        assert len(groups)==plan['expected_groups'],len(groups)
        for values in groups.values():
            assert sorted(r['rep'] for r in values)==list(range(len(values)))
    summaries=[]
    for (method,length,outputs),values in sorted(groups.items()):
        rates=[r['decode_tokens_per_s'] for r in values]
        summaries.append(dict(method=method,history_tokens=length,timed_output_tokens=outputs,n=len(values),
                              median_tokens_per_s=statistics.median(rates),min_tokens_per_s=min(rates),max_tokens_per_s=max(rates),
                              pooled_tokens_per_s=sum(r['timed_output_tokens'] for r in values)/sum(r['decode_s'] for r in values),
                              median_prefill_s=statistics.median(r['prefill_s'] for r in values),backend=values[0]['backend']))
    # Every arm must use the same exact input at a given history length.
    for length in {r['history_tokens'] for r in rows}:
        assert len({r['input_ids_sha256'] for r in rows if r['history_tokens']==length})==1,length
    for path,expected in plan.get('source_sha256',{}).items():assert digest(Path(path))==expected,path
    a.out.mkdir(exist_ok=True,parents=True)
    for name,values in [('runs',rows),('summary',summaries),('decode_steps',steps)]:
        keys=sorted(set().union(*(r.keys() for r in values)))
        with (a.out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2))
    audit={'status':'partial' if a.partial else 'complete','rows':len(rows),'groups':len(summaries),
           'same_input_ids_per_length':True,'plan_sha256':digest(a.plan),'frozen_source_hashes_passed':True,
           'same_original_bf16_weights':json.loads((HERE/'weight_identity_audit.json').read_text()),
           'limitations':['External arms are Qwen ports of original methods, not untouched upstream end-to-end implementations.',
                          'Repeated identical 240K native KVMem prompts retract generated output; two no_reusable_checkpoint cold prefills were observed. Raw prefill and request wall times remain exported.',
                          'HF BF16 residual and native FP32 residual, and FP8 rounding position, differ.',
                          'Native first-token-to-completion interval includes final bookkeeping; HF loop ends after final sampled token.',
                          'Small attention budgets differ. Matched-cap measurements are separate.',
                          'Fixed-prefix decode throughput does not establish online agent accuracy or summary-stall latency.']}
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'rows':len(rows),'groups':len(summaries),'status':audit['status']}))


if __name__=='__main__':main()
