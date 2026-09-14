"""Keep decode, new-query request, diagnostics and profiles in separate tables."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import re
import statistics

HERE=Path(__file__).resolve().parent

def csv_out(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    raw=[];profiles=[];states={};checks=[]
    native_log_cache={}
    def request_log(path,rid):
        if path not in native_log_cache:
            blocks={};pending=[]
            for line in path.read_text(errors='replace').splitlines():
                pending.append(line)
                if '[qw3-server-ttft]' in line:
                    match=re.search(r'\brid=(\d+)',line)
                    if match:blocks[int(match[1])]=pending;pending=[]
            native_log_cache[path]=blocks
        return native_log_cache[path][int(rid)]
    for path in sorted((HERE/'native').glob('*/result.json')):
        d=json.loads(path.read_text());states[str(path.relative_to(HERE))]=d['status']
        method=path.parent.name.split('_mtp')[0]
        for r in d['results']:
            if r['warmup']:continue
            mtp=r.get('mtp_summaries',[{}]);m=mtp[0].get('values',{}) if mtp else {}
            raw.append(dict(scenario='steady_decode',method=method,history_tokens=r['prompt_tokens'],mtp=d['mtp_chain'],
                mode='native',rep=r['rep'],outputs=128,timed_outputs=127,
                decode_tokens_per_s=r['decode_tokens_per_s'],decode_s=r['decode_interval_s'],
                input_ids_sha256=r['input_ids_sha256'],output_text_sha256=r['output_text_sha256'],
                mtp_drafted=m.get('drafted'),mtp_accepted=m.get('accepted'),mtp_batches=m.get('batches'),
                mtp_verified_positions=m.get('batched_verify_tokens'),mtp_draft_s=m.get('draft_s'),
                mtp_verify_s=m.get('verify_s'),mtp_restore_s=m.get('restore_s'),source=str(path)))
    external_paths=sorted(HERE.glob('external_*.json'))
    if (HERE/'infllm_chunk_query.json').exists():external_paths.append(HERE/'infllm_chunk_query.json')
    for path in external_paths:
        d=json.loads(path.read_text());method=d['method'];states[path.name]=d['status']
        for group,scenario in [('measurements','steady_decode'),('new_queries','new_query'),('diagnostics','diagnostic')]:
            if path.name=='external_infllm.json' and group=='new_queries':scenario='query_serial_append_diagnostic'
            for r in d[group]:
                n=r.get('history_tokens',r['prompt_tokens']);m=r['batches'];mtp=r['mtp']
                out=dict(scenario=scenario,method=method,history_tokens=n,mtp=mtp,mode=r.get('mode','online_query'),
                    rep=r['rep'],outputs=r['outputs'],timed_outputs=r['outputs']-1,
                    decode_tokens_per_s=r['decode_tokens_per_s'],decode_s=r['decode_s'],
                    exact_output=r.get('exact_output_vs_reference',r.get('exact_output_vs_serial_query')),
                    output_ids_sha256=hashlib.sha256(json.dumps(r['output_ids']).encode()).hexdigest(),
                    input_ids_sha256=r.get('input_ids_sha256',r.get('prefix_input_ids_sha256')),source=str(path))
                if mtp:
                    out.update(mtp_drafted=sum(x['drafted'] for x in m),mtp_accepted=sum(x['accepted'] for x in m),
                        mtp_batches=len(m),mtp_verified_positions=sum(x['drafted']+1 for x in m),
                        mtp_draft_s=sum(x['draft_s'] for x in m),mtp_verify_s=sum(x['verify_s'] for x in m),
                        mtp_commit_prime_s=sum(x['commit_prime_s'] for x in m))
                if group=='new_queries':
                    out.update(new_query_tokens=r['new_query_tokens'],prompt_tokens=r['prompt_tokens'],
                        query_ttft_s=r['query_ttft_s'],request_s=r['request_s'],
                        effective_tokens_per_s=r['effective_tokens_per_s'],timing_boundary='in_process_request',
                        input_hash_scope='prepared_history',query_append=r['query_append'])
                raw.append(out)
                if out['exact_output'] is not None:checks.append(bool(out['exact_output']))
        for r in d['profiles']:
            p=r['profile'];n=r['outputs']-1
            row=dict(method=method,history_tokens=r['prompt_tokens'],mtp=r['mtp'],mode=r['mode'],
                     instrumented_decode_s=r['decode_s'],useful_outputs=n)
            if method=='infinigen':
                ff=p['forecasts'];row.update(forecasts=len(ff),forecasts_per_output=len(ff)/n,
                    mean_selected_tokens_per_head=statistics.mean(x['selected_tokens_per_head'] for x in ff),
                    logical_h2d_mib_per_output=sum(x['h2d_bytes'] for x in ff)/n/2**20,
                    consumer_host_wait_ms_per_output=sum(p['consumer_host_wait_s'])*1000/n)
                for key in ['worker_host_s','query_enqueue_host_s','selector_and_indices_sync_host_s',
                            'sort_host_s','dedup_host_s','cpu_gather_host_s','h2d_submit_host_s']:
                    row[key+'_ms_per_output']=sum(x[key] for x in ff)*1000/n
            else:
                row.update(p['infllm_cache']);row.update(p.get('infllm_extra',{}))
                row['gpu_resident_fraction']=row.get('gpu_resident',0)/max(row.get('selected_blocks',0),1)
                row['cpu_only_fraction']=row.get('cpu_only',0)/max(row.get('selected_blocks',0),1)
                row['logical_miss_payload_mib_per_output']=row.get('miss_source_payload_bytes',0)/n/2**20
            profiles.append(row)
    for path in sorted([*HERE.glob('query_*/result.json'),*HERE.glob('guided_query_*/result.json')]):
        d=json.loads(path.read_text());states[str(path.relative_to(HERE))]=d['status']
        for r in d['results']:
            m=r['mtp_stats'];t=r['server_timing'];phase=r['phases']
            child=Path(d['repetition_sources'][r['rep']]) if 'repetition_sources' in d else path
            logs=request_log(child.parent/'server.log',t['rid'])
            replay_count=sum(any(s in line for s in ['native kvmem query replay:',
                'native kvmem query replay (plain):','native kvmem query-replay (plain): span=']) for line in logs)
            guided_events=[]
            for line in logs:
                if 'native kvmem guided query:' not in line:continue
                mode=re.search(r'\bmode=([\w-]+)',line)
                tokens=re.search(r'\bquery_tokens=(\d+)',line)
                if mode and tokens:guided_events.append((mode.group(1),int(tokens.group(1))))
            raw.append(dict(scenario='new_query_guided_supplement' if d.get('guided_query_requested') else 'new_query',
                method=d['method'],history_tokens=r['history_tokens'],mtp=d['mtp'],
                mode='native',rep=r['rep'],outputs=128,timed_outputs=127,
                decode_tokens_per_s=r['decode_tokens_per_s'],decode_s=r['decode_s'],new_query_tokens=r['new_query_tokens'],
                prompt_tokens=r['prompt_tokens'],query_ttft_s=r['query_ttft_s'],request_s=t['request_total_ms']/1000,
                effective_tokens_per_s=128000/t['request_total_ms'],client_request_s=r['request_s'],
                client_effective_tokens_per_s=r['effective_tokens_per_s'],timing_boundary='native_server_request',
                input_hash_scope='history_plus_new_query',
                input_ids_sha256=r['input_ids_sha256'],output_text_sha256=r['output_text_sha256'],
                mtp_drafted=m.get('drafted'),mtp_accepted=m.get('accepted'),mtp_batches=m.get('batches'),
                mtp_verified_positions=m.get('batched_verify_tokens'),
                semantic_select_log_count=r['semantic_select_log_count'],query_replay_log_count=replay_count,
                semantic_ms=phase.get('post_semantic_ms'),query_replay_ms=phase.get('post_query_replay_ms'),
                guided_query_ms=phase.get('post_guided_query_ms'),
                guided_query_generated_tokens=sum(n for mode,n in guided_events if not mode.startswith('fallback')) if guided_events else None,
                guided_query_used_tokens=sum(n for mode,n in guided_events) if guided_events else None,
                guided_query_mode=','.join(mode for mode,_ in guided_events) if guided_events else None,
                guided_prompt_ms=phase.get('post_guided_prompt_ms'),
                guided_decode_ms=phase.get('post_guided_decode_ms'),
                guided_state_ms=phase.get('post_guided_state_ms'),
                native_phase_accounting_available='post_semantic_ms' in phase,
                native_total_ms=phase.get('total_ms'),native_decode_ms=phase.get('decode_ms'),
                setup_ms=phase.get('setup_ms'),postprefill_ms=phase.get('postprefill_ms'),
                finalize_ms=phase.get('finalize_ms'),
                prefill_ms=phase.get('prefill_ms',0),fallback_log_count=len(r['checkpoint_fallback_logs']),
                new_query_not_cached=r['new_query_not_cached'],source=str(path)))
    native_serial={(r['method'],r['history_tokens'],r['rep']):r['output_text_sha256'] for r in raw
                   if r['scenario']=='steady_decode' and r['method'] in ('qw3','kvmem') and r['mtp']==0}
    for r in raw:
        if r['scenario']=='steady_decode' and r['method'] in ('qw3','kvmem') and r['mtp']==4:
            r['native_output_text_match_mtp0']=r['output_text_sha256']==native_serial[(r['method'],r['history_tokens'],r['rep'])]
    csv_out('measurements.csv',raw);csv_out('host_traffic_profiles.csv',profiles)
    csv_out('guided_query_supplement.csv',[r for r in raw if r['scenario']=='new_query_guided_supplement'])
    csv_out('kvmem_query_phases.csv',[r for r in raw if r['method']=='kvmem' and r['scenario'] in ['new_query','new_query_guided_supplement'] and r['native_phase_accounting_available']])
    grouped=collections.defaultdict(list)
    for r in raw:grouped[(r['scenario'],r['method'],r['history_tokens'],r['mtp'],r['mode'])].append(r)
    summaries=[]
    for key,rows in sorted(grouped.items()):
        scenario,method,length,mtp,mode=key
        row=dict(scenario=scenario,method=method,history_tokens=length,context_label=f'{length//1024}K',mtp=mtp,mode=mode,n=len(rows))
        for field in ['decode_tokens_per_s','decode_s','effective_tokens_per_s','request_s','query_ttft_s','semantic_ms','query_replay_ms','guided_query_ms']:
            values=[r[field] for r in rows if r.get(field) is not None]
            if values:row.update({f'median_{field}':statistics.median(values),f'min_{field}':min(values),f'max_{field}':max(values)})
        if mtp and all(r.get('mtp_drafted') for r in rows):
            row['pooled_mtp_acceptance']=sum(r['mtp_accepted'] for r in rows)/sum(r['mtp_drafted'] for r in rows)
        summaries.append(row)
    csv_out('summary.csv',summaries)
    csv_out('supplement_240k.csv',[r for r in summaries if r['history_tokens']==245760])
    attribution=[]
    for n in [32768,131072]:
        def one(scenario,method,mode=None):
            matches=[r for r in summaries if r['scenario']==scenario and r['method']==method and r['history_tokens']==n
                     and r['mtp']==0 and (mode is None or r['mode']==mode) and r['n']==3]
            return matches[0] if len(matches)==1 else None
        native=one('steady_decode','kvmem')
        for method,gpu_mode,oracle_mode in [('infllm','gpu_source','oracle_ids_gpu_source'),
                                             ('infinigen','optimized_both_gpu_source','oracle_ids_gpu_source_fast')]:
            base=one('steady_decode',method);gpu=one('diagnostic',method,gpu_mode);oracle=one('diagnostic',method,oracle_mode)
            if not all([native,base,gpu,oracle]):continue
            t0,tg,to,tn=[r['median_decode_s'] for r in [base,gpu,oracle,native]]
            attribution.append(dict(method=method,history_tokens=n,mtp=0,useful_outputs=127,
                normal_s=t0,gpu_kv_source_s=tg,oracle_ids_gpu_source_s=to,kvmem_native_s=tn,
                observed_gap_s=t0-tn,gpu_source_change_s=t0-tg,oracle_gpu_change_s=t0-to,
                fraction_of_observed_gap_removed=(t0-to)/(t0-tn),remaining_gap_s=to-tn,
                interpretation='Same-implementation counterfactual with oracle selections and extra GPU memory; not an intrinsic algorithm share or additive profiler breakdown. GPU-source-only retains index synchronization.'))
    csv_out('diagnostic_gap_attribution.csv',attribution)
    for scenario in ['steady_decode','new_query']:
        for mtp in [0,4]:
            rows=[]
            for n in [8192,16384,32768,65536,131072]:
                row=dict(history_tokens=n,context_label=f'{n//1024}K')
                for method in ['qw3','kvmem','infllm','infinigen']:
                    matches=[r for r in summaries if r['scenario']==scenario and r['method']==method and r['history_tokens']==n and r['mtp']==mtp]
                    if matches:
                        assert len(matches)==1
                        row[method]=matches[0]['median_'+('effective_tokens_per_s' if scenario=='new_query' else 'decode_tokens_per_s')]
                if len(row)>2:rows.append(row)
            csv_out(f'{scenario}_mtp{mtp}.csv',rows)
            if scenario=='new_query':
                matched=[]
                for n in [8192,16384,32768,65536,131072]:
                    row=dict(history_tokens=n,context_label=f'{n//1024}K')
                    for method in ['qw3','kvmem','infllm','infinigen']:
                        values=[r['decode_tokens_per_s'] for r in raw if r['scenario']=='new_query' and r['method']==method
                                and r['history_tokens']==n and r['mtp']==mtp]
                        if values:row[method]=statistics.median(values)
                    if len(row)>2:matched.append(row)
                csv_out(f'new_query_decode_mtp{mtp}.csv',matched)
    # Same prompt IDs, despite independently implemented adapters.
    identity_errors=[]
    for n in [8192,16384,32768,65536,131072]:
        signatures={r['input_ids_sha256'] for r in raw if r['scenario']=='steady_decode' and r['history_tokens']==n}
        if len(signatures)>1:identity_errors.append(n)
    assert not identity_errors,identity_errors
    inputs=json.loads((HERE.parent/'decode_comparison_20260912/inputs.json').read_text())
    suffix=inputs['suffix'];old=suffix['before']+suffix['query']+suffix['after']
    query=json.loads((HERE/'new_query.json').read_text())['input_ids']
    for r in raw:
        if r['scenario'] not in ['new_query','new_query_guided_supplement']:continue
        n=r['history_tokens'];expected=inputs['evaluation']['input_ids'][:n-len(old)]+old
        if r['input_hash_scope']=='history_plus_new_query':expected=expected+query
        assert hashlib.sha256(json.dumps(expected).encode()).hexdigest()==r['input_ids_sha256'],r
        assert r['prompt_tokens']==n+len(query),r
    result=dict(statuses=states,measurement_rows=len(raw),internal_exact_checks=len(checks),all_internal_checks_exact=all(checks),
        main_prompt_identity_passed=not identity_errors,new_query_identity_passed=True,profiles=len(profiles))
    (HERE/'analysis_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
