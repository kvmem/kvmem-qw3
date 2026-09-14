"""Export all repetitions and summarize diagnostics without mixing timing scopes."""
from pathlib import Path
import collections
import csv
import json
import statistics

HERE=Path(__file__).resolve().parent


def write_csv(name, rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def merge_intervals(intervals):
    merged=[]
    for start,end in sorted(intervals):
        if end<=start:
            continue
        if merged and start<=merged[-1][1]:
            merged[-1][1]=max(end,merged[-1][1])
        else:
            merged.append([start,end])
    return sum(end-start for start,end in merged)


def summarize(groups):
    rows=[]
    for key,values in sorted(groups.items()):
        method,length,mtp,mode,threads=key
        rates=[r['decode_tokens_per_s'] for r in values]
        rows.append(dict(method=method,prompt_tokens=length,mtp=mtp,mode=mode,threads=threads,
            n=len(values),median_tps=statistics.median(rates),min_tps=min(rates),max_tps=max(rates),
            median_ms_per_useful_output=statistics.median(1000/rate for rate in rates),
            all_outputs_exact=all(r.get('exact_output_vs_prior_fresh_prefill',r.get('exact_output_vs_prior',False)) for r in values)))
    return rows


def main():
    groups=collections.defaultdict(list);trials=[];profiles=[];profile_ranges=[];mtp_rows=[]
    statuses={};checks=collections.Counter()
    for method in ('infinigen','infllm'):
        source=HERE/f'formal_{method}.json'
        if not source.exists():
            continue
        data=json.loads(source.read_text());statuses[method]=data['status']
        for group in ('tuning','captures','measurements','profiles'):
            for r in data[group]:
                checks[group+'_checked']+=1
                checks[group+'_exact']+=bool(r['exact_output_vs_prior_fresh_prefill'])
        for r in data['measurements']:
            groups[(method,r['prompt_tokens'],r['mtp'],r['mode'],r['threads'])].append(r)
            row={k:r[k] for k in ('prompt_tokens','mtp','mode','threads','rep','outputs','decode_s','decode_tokens_per_s','exact_output_vs_prior_fresh_prefill')}
            row.update(method=method,source=source.name)
            trials.append(row)
            batches=r['batches']
            if r['mtp']:
                sums={k:sum(b.get(k,0) for b in batches) for k in ('drafted','accepted','emitted','draft_s','verify_s','commit_prime_s','total_s')}
                mtp_rows.append(dict(method=method,prompt_tokens=r['prompt_tokens'],mode=r['mode'],rep=r['rep'],
                    acceptance=sums['accepted']/sums['drafted'],batches=len(batches),
                    verified_positions=sums['drafted']+len(batches),useful_outputs=r['outputs']-1,**sums))
        for r in data['profiles']:
            p=r['profile'];useful=r['outputs']-1
            row=dict(method=method,prompt_tokens=r['prompt_tokens'],mtp=r['mtp'],mode=r['mode'],
                threads=r['threads'],instrumented_tps=r['decode_tokens_per_s'],decode_s=r['decode_s'],useful_outputs=useful)
            if method=='infinigen':
                forecasts=p['forecasts'];row['forecasts']=len(forecasts)
                row['forecasts_per_useful_output']=len(forecasts)/useful
                row['kv_and_lookup_h2d_mib_per_useful_output']=sum(x['h2d_bytes'] for x in forecasts)/useful/2**20
                row['mean_selected_tokens_per_head']=statistics.mean(x['selected_tokens_per_head'] for x in forecasts)
                for field in ('query_enqueue_host_s','selector_and_indices_sync_host_s','sort_host_s',
                              'dedup_host_s','cpu_gather_host_s','h2d_submit_host_s','worker_host_s'):
                    row[field.removesuffix('_s')+'_ms_per_useful_output']=sum(x[field] for x in forecasts)*1000/useful
                row['consumer_host_wait_ms_per_useful_output']=sum(p['consumer_host_wait_s'])*1000/useful
                byphase=collections.defaultdict(list)
                for span in p['gpu_stream_ranges']:
                    byphase[span['phase']].append((span['start_ms'],span['end_ms']))
                for phase,spans in byphase.items():
                    profile_ranges.append(dict(method=method,prompt_tokens=r['prompt_tokens'],mtp=r['mtp'],mode=r['mode'],
                        phase=phase,n=len(spans),sum_ms_per_useful_output=sum(b-a for a,b in spans)/useful,
                        union_ms_per_useful_output=merge_intervals(spans)/useful,
                        warning='CUDA event stream spans include host enqueue gaps and overlap other phases; not exclusive kernel time.'))
            else:
                cache=p['infllm_cache'];row.update(cache)
                for key in ('in_attention_buffer','gpu_resident','cpu_only'):
                    row[key+'_fraction']=cache.get(key,0)/cache['selected_blocks']
            profiles.append(row)
    summary=summarize(groups)
    write_csv('throughput_summary.csv',summary);write_csv('throughput_repetitions.csv',trials)
    write_csv('profile_host_and_traffic.csv',profiles);write_csv('profile_gpu_stream_spans.csv',profile_ranges)
    write_csv('mtp_batches_summary.csv',mtp_rows)
    thread=[];gil=[];reuse=[];pooled=collections.defaultdict(list)
    for key,values in groups.items():
        if key[3] in ('full','optimized_both'):
            pooled[key].extend(values)
    if (HERE/'thread_confirmation.json').exists():
        data=json.loads((HERE/'thread_confirmation.json').read_text());statuses['thread_confirmation']=data['status']
        tg=collections.defaultdict(list)
        for r in data['results']:
            tg[('infinigen',r['prompt_tokens'],r['mtp'],r['mode'],r['threads'])].append(r)
            pooled[('infinigen',r['prompt_tokens'],r['mtp'],r['mode'],r['threads'])].append(r)
        thread=summarize(tg);write_csv('thread_confirmation_summary.csv',thread)
        gg=collections.defaultdict(list)
        for r in data.get('gil_diagnostic',[]):gg[(r['mode'],r['gil_switch_s'])].append(r)
        for (mode,interval),values in sorted(gg.items()):
            rates=[r['decode_tokens_per_s'] for r in values]
            gil.append(dict(mode=mode,gil_switch_s=interval,n=len(values),median_tps=statistics.median(rates),
                min_tps=min(rates),max_tps=max(rates),all_outputs_exact=all(r['exact_output_vs_prior'] for r in values)))
        write_csv('gil_diagnostic_summary.csv',gil)
        reuse=[{k:v for k,v in r.items() if k!='layers'} for r in data['historical_row_reuse']]
        write_csv('immutable_kv_reuse_opportunity.csv',reuse)
    index={(r['method'],r['prompt_tokens'],r['mtp'],r['mode']):r for r in summary}
    gaps=[]
    for length in (32768,245760):
        for mtp in (0,4):
            keys=[('infinigen',length,mtp,'full'),('infllm',length,mtp,'full')]
            if not all(k in index and index[k]['n']==3 for k in keys):continue
            full,infllm=[index[k]['median_ms_per_useful_output'] for k in keys]
            for mode in ('optimized_bookkeeping','optimized_both'):
                opt=index.get(('infinigen',length,mtp,mode))
                if opt is None or opt['n']!=3:continue
                after=opt['median_ms_per_useful_output']
                gaps.append(dict(prompt_tokens=length,mtp=mtp,mode=mode,before_ms=full,after_ms=after,
                    infllm_ms=infllm,saved_ms=full-after,remaining_gap_ms=after-infllm,
                    observed_gap_removed_fraction=(full-after)/(full-infllm) if full>infllm else None,
                    scope='Controlled implementation gain on these prefixes; remainder is not proven intrinsic method cost.'))
    write_csv('observed_gap_attribution.csv',gaps)
    pooled_summary=summarize(pooled)
    write_csv('all_controlled_repetitions_summary.csv',pooled_summary)
    pooled_index={(r['method'],r['prompt_tokens'],r['mtp'],r['mode'],r['threads']):r for r in pooled_summary}
    pooled_gaps=[]
    for length in (32768,245760):
        for mtp in (0,4):
            keys=[('infinigen',length,mtp,'full',8),('infinigen',length,mtp,'optimized_both',8),('infllm',length,mtp,'full',2)]
            if not all(k in pooled_index for k in keys):continue
            base,opt,ref=[pooled_index[k] for k in keys]
            a,b,c=[r['median_ms_per_useful_output'] for r in (base,opt,ref)]
            pooled_gaps.append(dict(prompt_tokens=length,mtp=mtp,before_n=base['n'],after_n=opt['n'],ref_n=ref['n'],
                before_ms=a,after_ms=b,infllm_ms=c,saved_ms=a-b,remaining_gap_ms=b-c,
                observed_gap_removed_fraction=(a-b)/(a-c) if a>c else None))
    write_csv('all_controlled_gap_attribution.csv',pooled_gaps)
    result=dict(statuses=statuses,correctness=checks,throughput=summary,threads=thread,gil=gil,
        profiles=profiles,reuse=reuse,gaps=gaps,pooled_throughput=pooled_summary,pooled_gaps=pooled_gaps,
        caveats=['Different method working sets: InfLLM 6272 approximate tokens, InfiniGen cap 2048 per query head.',
            'Three repeated timings per condition reuse an exact restored initial state; one fixed prefix per length.',
            'Oracle query/IDs and full GPU history are diagnostic interventions, not deployable baseline scores.',
            'Profile events perturb timing; overlapping host/stream durations must not be added as wall-time shares.',
            'InfiniGen and InfLLM are Qwen/Q8/FP8/MTP ports in their external frameworks, not untouched author model configurations.',
            'Native KVMem engine has different kernels, residual precision and working set; its full ratio is not pure retrieval-method gain.'])
    (HERE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    def table(rows, fields):
        lines=['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']
        for row in rows:
            vals=[]
            for key in fields:
                value=row.get(key,'')
                vals.append(f'{value:.4f}' if isinstance(value,float) else str(value))
            lines.append('| '+' | '.join(vals)+' |')
        return '\n'.join(lines)
    sections=[
        '# Complete measured data',
        'Generated by `analyze.py`. All throughput is useful decode token/s; load, prefill and exact initial-state restoration are excluded. Instrumented profiles are separate. Status: '+str(statuses),
        '## Uninstrumented formal repetitions (median and full range)',
        table(summary,['method','prompt_tokens','mtp','mode','threads','n','median_tps','min_tps','max_tps','all_outputs_exact']),
        '## All matching controls, including the confirmation campaign',
        'All uninstrumented 128-output observations at each exact configuration are retained, including slow repetitions. GIL interval trials, 64-output tuning and oracle/capture trials are excluded. InfiniGen has six observations at 8 threads when confirmation is complete; InfLLM has three.',
        table(pooled_summary,['method','prompt_tokens','mtp','mode','threads','n','median_tps','min_tps','max_tps']),
        table(pooled_gaps,['prompt_tokens','mtp','before_n','after_n','ref_n','before_ms','after_ms','infllm_ms','observed_gap_removed_fraction']),
        '## Thread confirmation (paired 2 versus 8 threads, same 128 outputs)',
        table(thread,['prompt_tokens','mtp','mode','threads','n','median_tps','min_tps','max_tps']),
        '## Python thread handoff diagnostic (32K, MTP=4, same arithmetic)',
        table(gil,['mode','gil_switch_s','n','median_tps','min_tps','max_tps']),
        '## Observed InfiniGen versus InfLLM delay gap removed by exact implementation changes',
        'Using **ms per useful output**, not subtracting token/s. `(before - after) / (before - InfLLM)`. The remaining delay is not established as intrinsic method cost. No confidence interval or cross-task generalization is claimed.',
        table(gaps,['prompt_tokens','mtp','mode','before_ms','after_ms','infllm_ms','saved_ms','remaining_gap_ms','observed_gap_removed_fraction']),
        '## Instrumented host spans and H2D traffic',
        'Host spans overlap GPU activity and other threads. Do not sum these columns into exclusive wall-time percentages.',
        table([r for r in profiles if r['method']=='infinigen'],['prompt_tokens','mtp','mode','instrumented_tps',
            'query_enqueue_host_ms_per_useful_output','selector_and_indices_sync_host_ms_per_useful_output',
            'sort_host_ms_per_useful_output','dedup_host_ms_per_useful_output','cpu_gather_host_ms_per_useful_output',
            'consumer_host_wait_ms_per_useful_output','kv_and_lookup_h2d_mib_per_useful_output']),
        '## InfLLM observed cache hits',
        table([r for r in profiles if r['method']=='infllm'],['prompt_tokens','mtp','selected_blocks',
            'in_attention_buffer_fraction','gpu_resident_fraction','cpu_only_fraction']),
        '## InfiniGen immutable-row temporal locality',
        'Potential previous-selection KV reuse, measured after the first token of each layer. **No cache has been implemented and no speedup is implied by this percentage.** Draft tail rows are excluded.',
        table(reuse,['prompt_tokens','selected_immutable_rows_after_first','also_selected_previous_token','reuse_fraction']),
        '## Interpretation boundaries',
        '\n'.join('- '+x for x in result['caveats']),
    ]
    (HERE/'MEASUREMENTS.md').write_text('\n\n'.join(sections)+'\n')
    print(json.dumps({'statuses':statuses,'formal_rows':len(trials),'correctness':checks,'summary':summary},indent=2))


if __name__=='__main__':main()
