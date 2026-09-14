"""Final completeness audit; no failed or slow repetitions may be dropped."""
from pathlib import Path
import collections
import csv
import json

HERE=Path(__file__).resolve().parent

def main():
    status=json.loads((HERE/'analysis_status.json').read_text())
    assert all(s=='complete' for s in status['statuses'].values()),status
    assert status['all_internal_checks_exact'] and status['main_prompt_identity_passed'] and status['new_query_identity_passed']
    rows=list(csv.DictReader((HERE/'measurements.csv').open()))
    counts=collections.Counter(r['scenario'] for r in rows)
    assert counts==dict(steady_decode=132,new_query=144,diagnostic=60,query_serial_append_diagnostic=36,new_query_guided_supplement=2),counts
    for scenario in ['steady_decode','new_query']:
        for method in ['qw3','kvmem','infllm','infinigen']:
            for mtp in [0,4]:
                for n in [8192,16384,32768,65536,131072]:
                    selected=[r for r in rows if r['scenario']==scenario and r['method']==method and int(r['mtp'])==mtp and int(r['history_tokens'])==n]
                    assert sorted(int(r['rep']) for r in selected)==[0,1,2],(scenario,method,mtp,n)
    native_queries=[r for r in rows if r['scenario']=='new_query' and r['method'] in ['qw3','kvmem']]
    assert len(native_queries)==72 and all(r['new_query_not_cached']=='True' for r in native_queries)
    guided=[r for r in rows if r['scenario']=='new_query_guided_supplement']
    assert sorted(int(r['history_tokens']) for r in guided)==[131072,245760]
    assert all(float(r['guided_query_ms'])>0 and float(r['query_replay_ms'])>0 for r in guided),guided
    assert {int(r['history_tokens']):r['guided_query_mode'] for r in guided}=={131072:'direct',245760:'fallback-original'}
    assert all(int(r['fallback_log_count'])==0 for r in native_queries+guided)
    traces=json.loads((HERE/'trace_analysis.json').read_text())
    assert {(r['method'],r['mtp']) for r in traces}=={(m,t) for m in ['qw3','kvmem','infllm','infinigen'] for t in [0,4]}
    profile=list(csv.DictReader((HERE/'host_traffic_profiles.csv').open()))
    assert len(profile)==24,len(profile)
    result=dict(status='complete',formal_main_grid_rows=240,total_rows=len(rows),scenario_counts=dict(counts),
                external_equivalence_checks=status['internal_exact_checks'],external_equivalence_passed=True,
                input_identity_passed=True,native_new_query_cache_guard_passed=True,trace_count=len(traces),
                native_new_query_checkpoint_fallbacks=0,
                guided_outcomes=[dict(history_tokens=int(r['history_tokens']),mode=r['guided_query_mode'],
                                     generated_tokens=int(r['guided_query_generated_tokens']),
                                     used_query_tokens=int(r['guided_query_used_tokens'])) for r in guided],
                native_mtp_output_text_mismatches=[dict(method=r['method'],history_tokens=int(r['history_tokens']),rep=int(r['rep']))
                    for r in rows if r.get('native_output_text_match_mtp0')=='False'],
                note='Three repetitions for every formal cell. Diagnostic and instrumented runs are separate. Native MTP byte differences are disclosed, not filtered.')
    (HERE/'final_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
