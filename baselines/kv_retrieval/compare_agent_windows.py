"""Join real InfLLM trials to the frozen historical cohort without imputing outcomes.

Read-only with respect to all runs. Native and adapted server clocks are kept
explicit; shared/unknown GPU load never produces a claimed algorithm speedup.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

ROOT=Path(__file__).resolve().parent
METHODS=['kvmem','qw3_compaction','infllm_large','infllm_small']
GENERATE=re.compile(r'native generate: prompt_tokens=(\d+) prefill=([\d.]+)s.*?decoded=(\d+) decode=([\d.]+)s')
NUMBERS=re.compile(r'\b([A-Za-z_]\w*)=(-?\d+(?:\.\d+)?)')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(path)


def jsonl(path):
    if not path.exists():return []
    data=path.read_text().splitlines()
    result=[]
    for index,line in enumerate(data):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if index!=len(data)-1:raise
    return result


def stat(values):
    values=sorted(float(x) for x in values if x is not None and math.isfinite(x))
    if not values:return None
    def quantile(q):
        index=(len(values)-1)*q
        lo=int(index);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(index-lo)
    return {'n':len(values),'mean':statistics.mean(values),'median':quantile(.5),
            'p95':quantile(.95),'p99':quantile(.99),'max':values[-1]}


def reward_fields(summary,terminal_status):
    reward=summary.get('reward') if summary else None
    if not isinstance(reward,dict):
        reward={'reward':reward} if isinstance(reward,(int,float)) else {}
    score=reward.get('reward')
    valid=(terminal_status=='completed' and summary.get('agent_exit')==0
           and summary.get('verifier_exit')==0 and isinstance(score,(int,float))
           and math.isfinite(score) and 0<=score<=1)
    return {'valid_task_outcome':valid,'score':score if valid else None,
            'fully_passed':score==1 if valid else None,
            'partial_test_score':reward.get('partial') if valid else None,
            'f2p_passed':reward.get('f2p_passed') if valid else None,
            'f2p_total':reward.get('f2p_total') if valid else None,
            'p2p_passed':reward.get('p2p_passed') if valid else None,
            'p2p_total':reward.get('p2p_total') if valid else None,
            'agent_exit':summary.get('agent_exit'),'verifier_exit':summary.get('verifier_exit')}


def native_steps(path):
    rows=[];pending=[]
    for line_no,line in enumerate(path.read_text().splitlines(),1):
        generated=GENERATE.search(line)
        if generated:
            pending.append({'prompt_tokens':int(generated[1]),'prefill_s':float(generated[2]),
                'output_tokens':int(generated[3]),'decode_s':float(generated[4]),'generate_log_line':line_no})
        if '[qw3-server-ttft]' not in line:continue
        values={k:float(v) for k,v in NUMBERS.findall(line)}
        if len(pending)!=1:
            raise ValueError(f'ambiguous native generation/TTFT alignment at {path}:{line_no}, pending={len(pending)}')
        row=pending.pop()
        for dest,key in [('server_first_model_s','server_ttft_ms'),('server_first_content_s','response_ttft_ms'),('server_request_wall_s','request_total_ms')]:
            v=values.get(key)
            row[dest]=v/1000 if v is not None and v>=0 else None
        row.update(request_index=len(rows)+1,request_id=int(values['rid']),ttft_log_line=line_no,
                   ttft_clock='native server request-arrival -> first streamed content, includes reasoning',
                   server_first_non_reasoning_content_s=None,
                   is_compaction=None)
        rows.append(row)
    if pending:raise ValueError(f'unpaired native generate records: {path}')
    return rows


def historical_rows(audit):
    trials=[];steps=[]
    for old in audit['rows']:
        run=Path(old['path']);metrics=read_json(run/'metrics.json');summary=read_json(run/'cell_summary.json')
        if hashlib.sha256((run/'cell_summary.json').read_bytes()).hexdigest()!=old['summary_sha256']:
            raise ValueError(f'historical summary changed: {run}')
        method='kvmem' if old['method']=='kvmem' else 'qw3_compaction'
        manifest_path=run.parents[2]/'manifest.json'
        manifest=read_json(manifest_path)
        batch_path=run.parents[5]/'manifest.json'
        batch=read_json(batch_path) if batch_path.exists() else {}
        server,claude=metrics['server'],metrics['claude']
        rows=native_steps(run/'server/qw3_server.log')
        if len(rows)!=server['request_count']:
            raise ValueError(f'historical request-count mismatch: {run}')
        terminal='completed' if summary['agent_exit']==0 and summary['verifier_exit']==0 else 'requires_review'
        row={'task_id':old['task_id'],'seed':old['seed'],'method':method,'run':str(run),'status':terminal,
            **reward_fields(summary,terminal),'agent_wall_s':metrics['agent_elapsed_sec'],
            'total_cell_wall_s':metrics['total_elapsed_sec'],'server_request_wall_s':server['request_wall_sec'],
            'prefill_s':server['prefill_sec'],'decode_s':server['decode_sec'],'output_tokens':server['decoded_tokens'],
            'decode_tps':server['decoded_tokens']/server['decode_sec'] if server['decode_sec']>0 else None,
            'token_frontend_s':None,
            'decode_timing_scope':'native backend-reported decode; frontend portion not separately reported',
            'request_count':server['request_count'],'max_prompt_tokens':server['max_prompt_tokens_server'],
            'compactions':claude['compactions'],'compaction_total_s':claude['compaction_total_sec'],
            'kvmem_reselections':server['kvmem_actual_reselections'],
            'tool_errors_reported':claude['tool_errors'],'api_errors_reported':claude['api_errors'],
            'server_first_content_s':stat(r['server_first_content_s'] for r in rows),
            'server_first_non_reasoning_content_s':None,
            'gpu_node_label':batch.get('execution',{}).get('node_name'),
            'gpu_exclusive_claim':batch.get('execution',{}).get('gpu_exclusive'),
            'gpu_exclusivity_verified':False,'formal_algorithm_speedup_eligible':False,
            'mtp_chain':int(old['observed_config']['mtp_chain']),'historical_sampling':manifest.get('sampling'),
            'agent_memory_label':batch.get('execution',{}).get('agent_container_memory'),
            'metrics_sha256':hashlib.sha256((run/'metrics.json').read_bytes()).hexdigest(),
            'server_log_sha256':hashlib.sha256((run/'server/qw3_server.log').read_bytes()).hexdigest(),
            'metric_scope':'complete_trial','peak_process_gpu_gib':None,'peak_model_tree_pss_gib':None}
        trials.append(row)
        steps.extend({'task_id':row['task_id'],'seed':row['seed'],'method':method,**r,'source':str(run/'server/qw3_server.log')} for r in rows)
    return trials,steps


def infllm_row(task_id,variant,run):
    result=read_json(run/'result.json')
    summary=result.get('cell_summary') or {}
    rows=jsonl(run/'server/requests.jsonl')
    complete=[r for r in rows if r.get('status')=='completed']
    failed=[r for r in rows if r.get('status')!='completed']
    agent=read_json(run/'cell/agent_summary.json') if (run/'cell/agent_summary.json').exists() else {}
    resource=jsonl(run/'resource_samples.jsonl');host=jsonl(run/'host_memory_samples.jsonl')
    peaks=[r['gpu_processes_gib'].get(str(result['server_pid']),0) for r in resource] if result.get('server_pid') else []
    method='infllm_'+variant
    total_decode=sum(r.get('decode_s',0) for r in complete)
    total_output=sum(r.get('output_tokens',0) for r in complete)
    frontend=sum(r['token_frontend_s'] for r in complete) if complete and all('token_frontend_s' in r for r in complete) else None
    row={'task_id':task_id,'seed':1000,'method':method,'run':str(run),'status':result['status'],
        **reward_fields(summary,result['status']),
        'agent_wall_s':agent.get('elapsed_sec'),'total_cell_wall_s':summary.get('elapsed_sec'),
        'server_request_wall_s':sum(r.get('request_wall_s',0) for r in complete),
        'prefill_s':sum(r.get('prefill_s',0) for r in complete),'decode_s':total_decode,
        'output_tokens':total_output,'decode_tps':total_output/total_decode if total_decode else None,
        'token_frontend_s':frontend,
        'decode_timing_scope':'adapted sampling/model/streaming loop wall time; includes measured token frontend',
        'request_count':len(complete),'failed_model_requests':sum(r.get('status')!='completed' for r in rows),
        'recorded_request_count':len(rows),
        'recorded_request_wall_s':sum(r.get('request_wall_s',0) for r in rows),
        'recorded_output_tokens':sum(r.get('output_tokens',0) for r in rows),
        'raw_verifier_reward':summary.get('reward'),
        'terminal_error':(result.get('error') or result.get('resource_error')
                          or (failed[-1].get('error') if failed else None)),
        'max_prompt_tokens':max((r.get('prompt_tokens',0) for r in rows),default=None),
        'unexpected_cold_prefills':sum(r.get('unexpected_cold_prefill',False) for r in complete),
        'active_capacity_tokens':196608 if variant=='large' else 6272,
        'server_first_content_s':stat(r.get('api_first_content_s') for r in complete),
        'server_first_non_reasoning_content_s':stat(r.get('api_first_action_s') for r in complete),
        'peak_process_gpu_gib':max(peaks,default=None),
        'peak_model_tree_pss_gib':max((sum(p['pss_kib'] for p in r.get('model_processes',[]))/2**20 for r in host),default=None),
        'metric_scope':'complete_trial' if result['status']=='completed' else 'completed_requests_only; trial incomplete',
        'shared_gpu':result.get('shared_gpu'),'gpu_exclusivity_verified':False,
        'formal_algorithm_speedup_eligible':False,'mtp_chain':0,
        'resource_error':result.get('resource_error'),'error':result.get('error')}
    steps=[]
    for i,r in enumerate(rows,1):
        steps.append({'task_id':task_id,'seed':1000,'method':method,'request_index':i,
            **{k:r.get(k) for k in ['request_id','status','prompt_tokens','output_tokens','resume_origin','reuse_reason','cold_prefill','unexpected_cold_prefill','prefill_s','decode_s','token_frontend_s']},
            'server_first_model_s':r.get('first_model_token_s'),'server_first_content_s':r.get('api_first_content_s'),
            'server_first_non_reasoning_content_s':r.get('api_first_action_s'),
            'server_request_wall_s':r.get('request_wall_s'),'is_compaction':False,
            'ttft_clock':'adapted server request-arrival -> first streamed content, includes reasoning',
            'source':str(run/'server/requests.jsonl')})
    return row,steps


def apply_terminal_review(row,path):
    """Accept only a reviewed execution failure tied to the unchanged raw trial."""
    if not path.exists():return
    review=read_json(path)
    if review.get('end_to_end_execution_outcome')!='failed':return
    run=Path(row['run'])
    if review.get('arm')!=str(run):return
    raw_result=read_json(run/'result.json')
    if (review.get('task_id')!=row['task_id'] or review.get('arm')!=str(run)
            or row['status']!='agent_failed' or not raw_result.get('agent_started')
            or row.get('agent_exit') in (None,0) or row.get('verifier_exit')!=0
            or not isinstance(row.get('raw_verifier_reward'),dict)):
        raise ValueError('terminal failure review does not match an evaluated agent attempt')
    for relative in ['result.json','cell/cell_summary.json','server/requests.jsonl','client_events.jsonl']:
        if review.get('source_hashes',{}).get(relative)!=hashlib.sha256((run/relative).read_bytes()).hexdigest():
            raise ValueError('terminal failure review has stale evidence: '+relative)
    reward=row['raw_verifier_reward'].get('reward')
    if not isinstance(reward,(int,float)) or not math.isfinite(reward) or not 0<=reward<=1:
        raise ValueError('terminal failure review has no valid raw verifier score')
    row.update(reviewed_execution_failure=True,terminal_review_path=str(path))


def aggregates(trials,tasks,methods=None):
    methods=METHODS if methods is None else methods
    paired=[task for task in tasks if all(any(r['task_id']==task and r['method']==method and r['valid_task_outcome'] for r in trials) for method in methods)]
    result={'compared_methods':methods,'complete_paired_task_ids':paired,'complete_paired_count':len(paired),'methods':{}}
    def terminal(r):return r['valid_task_outcome'] or r.get('reviewed_execution_failure',False)
    terminal_paired=[task for task in tasks if all(any(r['task_id']==task and r['method']==method and terminal(r)
                     for r in trials) for method in methods)]
    result.update(paired_terminal_task_ids=terminal_paired,paired_terminal_count=len(terminal_paired))
    if methods==METHODS:
        result.update(complete_four_method_task_ids=paired,complete_four_method_count=len(paired))
    for method in methods:
        rows=[r for r in trials if r['method']==method]
        valid=[r for r in rows if r['valid_task_outcome']]
        selected=[r for r in valid if r['task_id'] in paired]
        terminal_rows=[r for r in rows if terminal(r)]
        paired_terminal_rows=[r for r in terminal_rows if r['task_id'] in terminal_paired]
        successes=sum(r['valid_task_outcome'] and r['fully_passed'] for r in terminal_rows)
        result['methods'][method]={'planned_tasks':len(tasks),'valid_completed_tasks':len(valid),
            'terminal_agent_trials':len(terminal_rows),
            'reviewed_execution_failures':sum(bool(r.get('reviewed_execution_failure')) for r in terminal_rows),
            'agent_and_judge_successes':successes,
            'agent_and_judge_success_rate_among_terminal_trials':successes/len(terminal_rows) if terminal_rows else None,
            'paired_agent_and_judge_success_rate':statistics.mean(bool(r['valid_task_outcome'] and r['fully_passed'])
                for r in paired_terminal_rows) if paired_terminal_rows else None,
            'full_passes':sum(r['fully_passed'] for r in valid),
            'full_pass_rate_among_valid_completed':statistics.mean(r['fully_passed'] for r in valid) if valid else None,
            'paired_full_pass_rate':statistics.mean(r['fully_passed'] for r in selected) if selected else None,
            'paired_mean_agent_wall_s':statistics.mean(r['agent_wall_s'] for r in selected) if selected else None,
            'paired_mean_request_wall_s':statistics.mean(r['server_request_wall_s'] for r in selected) if selected else None}
    return result


def csv_rows(path,rows):
    flat=[{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows]
    if not flat:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader();writer.writerows(flat)


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();a.campaign=a.campaign.resolve();out=a.out or a.campaign/'comparison';out.mkdir(parents=True,exist_ok=True)
    plan=read_json(a.campaign/'plan.json');audit=read_json(ROOT/plan['historical_audit'])
    trials,steps=historical_rows(audit)
    tasks=[r['task_id'] for r in read_json(ROOT/plan['task_manifest'])['tasks']]
    variants=[v for v in ['large','small'] if v in plan['variants']]
    if not variants:raise ValueError('campaign has no supported variants')
    methods=['kvmem','qw3_compaction']+['infllm_'+v for v in variants]
    for task in tasks:
        for variant in variants:
            attempts=sorted(p for p in (a.campaign/task/variant).glob('attempt_*') if p.is_dir())
            results=[r for r in attempts if (r/'result.json').exists()]
            if results:
                row,requests=infllm_row(task,variant,results[-1]);steps.extend(requests)
                apply_terminal_review(row,a.campaign/'terminal_reviews'/f'{task}.json')
                row['attempts']=[str(r) for r in attempts];trials.append(row)
            else:
                trials.append({'task_id':task,'seed':1000,'method':'infllm_'+variant,'status':'pending',
                               'valid_task_outcome':False,'score':None,'fully_passed':None})
    report={'created_at':time.time(),'cohort':tasks,'seed':1000,'trials':trials,'aggregates':aggregates(trials,tasks,methods),
        'scope':'Only complete paired tasks for all selected methods enter paired averages. Partial live trials never count as zero scores.',
        'selected_variants':variants,
        'timing_limitations':plan['comparison_limitations'],
        'historical_load_note':'Ordinary qw3 root manifest explicitly allows nonexclusive GPU; actual historical load is not verified.',
        'memory_note':'InfLLM process GPU memory and model-tree PSS peaks are maxima of sampled observations '
                    '(approximately 2-second GPU and 10-second host-memory intervals), not guaranteed instantaneous peaks. '
                    'Reused native trial metrics do not provide these measurements; native memory fields remain unknown. '
                    'Configured allocator/container limits are not measured memory consumption.',
        'ttft_note':'Both server first-content clocks include reasoning. Neither includes client pre-API compaction waiting. '
                    'Native per-request compaction labels are unknown here; task-level compaction counts come from Claude events.',
        'non_reasoning_clock_note':'InfLLM server_first_non_reasoning_content_s comes from api_first_action_s: '
                    'server request-arrival -> first text delta or tool-use block start. It excludes thinking content, '
                    'but is not the time a complete tool call is available or executed. The native server logs used '
                    'here do not expose this separate clock; leave native values missing, not zero or inferred.',
        'tool_error_note':'Model-issued shell commands can legitimately fail during diagnosis. Tool error count alone is not an infrastructure exclusion.',
        'terminal_outcome_note':'Agent-and-judge success requires normal agent completion and a full verifier pass. '
                    'Reviewed terminal execution failures remain in this denominator as failures. Running/pending '
                    'tasks and startup failures before the agent begins are not terminal agent outcomes. Counts '
                    'may cover different terminal task subsets until the campaign finishes; paired terminal '
                    'statistics identify their common task IDs. This is an end-to-end system outcome, not isolated retrieval accuracy.'}
    write_json(out/'comparison.json',report);csv_rows(out/'trials.csv',trials);csv_rows(out/'server_steps.csv',steps)
    lines=['# InfLLM DeepSWE comparison','',f'Generated at Unix time {report["created_at"]:.3f}. Seed 1000; four original tasks. Selected InfLLM windows: {", ".join(variants)}.',
           '', '| Task | Method | Status | Full pass | Partial test score | Agent min | Model requests |',
           '|---|---|---|---:|---:|---:|---:|']
    def fmt(v,digits=3):return '—' if v is None else f'{v:.{digits}f}'
    for task in tasks:
        for method in methods:
            r=next(r for r in trials if r['task_id']==task and r['method']==method)
            lines.append(f'| {task} | {method} | {r["status"]} | {fmt(r.get("fully_passed"),0)} | {fmt(r.get("partial_test_score"))} | {fmt(r["agent_wall_s"]/60 if r.get("agent_wall_s") is not None else None,1)} | {r.get("request_count","—")} |')
    ordered=[next(r for r in trials if r['task_id']==task and r['method']==method)
             for task in tasks for method in methods]
    lines += ['', '## Terminal agent outcomes', '', report['terminal_outcome_note'], '',
              '| Method | Terminal agent trials | Normal completions | Reviewed execution failures | Agent-and-judge successes |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        totals=report['aggregates']['methods'][method]
        lines.append(f'| {method} | {totals["terminal_agent_trials"]} | {totals["valid_completed_tasks"]} | '
                     f'{totals["reviewed_execution_failures"]} | {totals["agent_and_judge_successes"]} |')
    lines += ['', '## Observed efficiency', '',
              'Complete rows cover the full trial; incomplete rows cover only completed model requests. '
              'In-flight requests are not yet included. These are observations, not paired speedup estimates.', '',
              '| Task | Method | Measurement scope | Request min | Prefill s | Decoded K tokens | Decode tok/s |',
              '|---|---|---|---:|---:|---:|---:|']
    for r in ordered:
        lines.append(f'| {r["task_id"]} | {r["method"]} | {r.get("metric_scope","pending")} | '
                     f'{fmt(r["server_request_wall_s"]/60 if r.get("server_request_wall_s") is not None else None,1)} | '
                     f'{fmt(r.get("prefill_s"),1)} | '
                     f'{fmt(r["output_tokens"]/1000 if r.get("output_tokens") is not None else None,1)} | '
                     f'{fmt(r.get("decode_tps"),2)} |')
    lines += ['', 'Decode tok/s is total output tokens divided by total reported decode time. '
              'The adapted InfLLM decode clock includes sampling and streaming/frontend work; '
              'the native clock is backend-reported decode time. It is not a GPU-only compute rate.', '',
              '## Observed memory', '', report['memory_note'], '',
              '| Task | Method | Status | Sampled peak process GPU GiB | Sampled peak model-tree PSS GiB |',
              '|---|---|---|---:|---:|']
    for r in ordered:
        lines.append(f'| {r["task_id"]} | {r["method"]} | {r["status"]} | '
                     f'{fmt(r.get("peak_process_gpu_gib"),2)} | {fmt(r.get("peak_model_tree_pss_gib"),2)} |')
    lines += ['', 'Running-trial memory peaks cover observations so far and can increase.']
    failed_trials=[r for r in ordered if r['status'] in {'agent_failed','verifier_failed',
                   'resource_failed','infrastructure_failed','failed'}]
    if failed_trials:
        lines += ['', '## Trials ending with errors', '',
                  'These terminated trials remain explicit in the results. Their original post-termination '
                  'verifier scores describe the artifact at interruption; they are not normally completed '
                  'agent trajectories and are excluded from complete-pair means. Recorded request time and '
                  'tokens below include failed requests; missing failed-request decode timing is not imputed.', '',
                  '| Task | Status | Raw verifier score | Raw partial score | All recorded request min | All recorded output K tokens | Error |',
                  '|---|---|---:|---:|---:|---:|---|']
        for r in failed_trials:
            reward=r.get('raw_verifier_reward')
            reward=reward if isinstance(reward,dict) else {'reward':reward}
            error=str(r.get('terminal_error') or f'agent_exit={r.get("agent_exit")}, verifier_exit={r.get("verifier_exit")}').replace('|','\\|').replace('\n',' ')
            lines.append(f'| {r["task_id"]} | {r["status"]} | {fmt(reward.get("reward"),0)} | '
                         f'{fmt(reward.get("partial"))} | '
                         f'{fmt(r["recorded_request_wall_s"]/60 if r.get("recorded_request_wall_s") is not None else None,1)} | '
                         f'{fmt(r["recorded_output_tokens"]/1000 if r.get("recorded_output_tokens") is not None else None,1)} | {error} |')
    lines += ['', '## Comparison limits', '']
    lines.extend('- '+note for note in report['timing_limitations'])
    lines += ['',report['scope'],'',report['historical_load_note'],'',report['ttft_note'],'',report['tool_error_note'],'',
              report['non_reasoning_clock_note'],'',
              'These observations do not establish an algorithm speedup: the adapters, MTP support and GPU load differ. '
              'All reported task outcomes come from independently generated live trajectories.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'out':str(out),'trials':len(trials),'server_step_records':len(steps),
                      'complete_paired_tasks':report['aggregates']['complete_paired_count']}))


if __name__=='__main__':main()
