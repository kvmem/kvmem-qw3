#!/usr/bin/env python3
"""Analyze a native Claude Code pair without treating a summary token as an action."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import statistics
from summarize_claude_suite import official_score

def jsonl(path):
    rows=[]
    if path.exists():
        with path.open() as source:
            for line in source:
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
    return rows

def stats(values):
    values=sorted(x for x in values if x is not None)
    if not values:return None
    position=.95*(len(values)-1);lower=int(position);fraction=position-lower
    p95=values[lower]*(1-fraction)+values[min(lower+1,len(values)-1)]*fraction
    return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'p95':p95,'max':max(values)}

def cancelled_after_budget(api,result):
    if (result.get('termination') not in ('action_limit','timeout','episode_end') or
        api['after_step']<result.get('steps',float('inf')) or not api.get('relay_error')):
        return False
    if result.get('termination') == 'episode_end':
        ended = result.get('episode_ended_ns')
        if ended is None or api.get('started_ns', -1) < ended:
            return False
    stopped=result.get('budget_stop_ns')
    # Earlier action-limit recordings predate the explicit cancellation clock.
    if stopped is None:return result.get('termination')=='action_limit'
    return (api.get('finished_ns') or 0)>=stopped

def server_records(path):
    """Join serial native phase records to their enclosing HTTP request ID."""
    records={};pending={};wrapper={}
    if not path.exists():return records
    with path.open() as source:
        for line in source:
            numbers={k:float(v) for k,v in re.findall(r'\b([a-zA-Z_][a-zA-Z_0-9]*)=(-?\d+(?:\.\d+)?)',line)}
            if '[qw3-anthropic-preprocess]' in line and numbers.get('stream')==1:
                wrapper={f'native_wrapper_{k}':v for k,v in numbers.items()}
            elif '[qw3-native-accounting]' in line:
                pending={f'native_{k}':v for k,v in numbers.items()}
            elif 'native generate: prompt_tokens=' in line:
                pending.update({k:numbers[k] for k in ('prompt_tokens','kvmem_reuse','prefilled') if k in numbers})
                pending.update({k:v for k,v in numbers.items() if k.startswith('kvmem_middecode_')})
            elif 'native continuous_batch: request=' in line:
                pending.update(native_prefill_ms=numbers['prefill']*1000,native_decode_ms=numbers['decode']*1000,
                    reused_tokens=numbers['reused_tokens'],prefilled=numbers['prefill_tokens'])
            elif '[qw3-server-ttft]' in line:
                rid=int(numbers['rid']);records.setdefault(rid,{}).update(wrapper);wrapper={}
                records[rid].update(pending);pending={}
                records[rid].update({k:v for k,v in numbers.items() if k!='rid'})
            elif '[qw3-server-accounting]' in line:
                rid=int(numbers['rid']);records.setdefault(rid,{}).update({k:v for k,v in numbers.items() if k!='rid'})
    return records

def prefill_selection(api):
    return api.get('server',{}).get('native_post_semantic_ms',0)>1

def middecode_refreshes(api):
    return int(api.get('server',{}).get('kvmem_middecode_refreshes',0))

def selection_request(api):
    return prefill_selection(api) or middecode_refreshes(api)>0

def analyze(root):
    combined=[];summaries={}
    for arm in ['qw3','kvmem']:
        p=root/'full'/arm
        if not (p/'steps.jsonl').exists():continue
        steps=jsonl(p/'steps.jsonl');apis=jsonl(p/'api.jsonl');events=jsonl(p/'claude.stream.jsonl')
        boundaries=[e for e in events if e.get('subtype')=='compact_boundary']
        tool_errors=[{'tool_use_id':c.get('tool_use_id'),'content':c.get('content')}
            for e in events if e.get('type')=='user'
            for c in e.get('message',{}).get('content',[]) if isinstance(c,dict) and c.get('type')=='tool_result' and c.get('is_error')]
        result=json.loads((p/'result.json').read_text())
        if isinstance(result.get('score'),dict):
            # This is a derived view. Preserve the complete official return and
            # expose its numeric score for report readers already running in a
            # long-lived suite. Never rewrite the original trial result.json.
            evaluation=result['score']
            result={**result,'score':official_score(evaluation),'official_evaluation':evaluation}
        resource_marker=p.resolve().parents[1]/'gitlab_stop_record.json'
        resource_changes=[]
        if resource_marker.exists():
            change=json.loads(resource_marker.read_text())
            if arm in change.get('arms',['kvmem']):resource_changes.append(change)
        server=server_records(p/'qw3_server.log')
        # In this isolated, serial suite every generation has one server rid.
        # Token-count requests do not enter the generation sequence.
        generated=sorted((a for a in apis if a['path'].split('?')[0]=='/v1/messages'),key=lambda a:a['id'])
        for rid,a in enumerate(generated,1):
            phases=server.get(rid,{})
            usage_tokens=a.get('input_usage',{}).get('input_tokens')
            budget_cancelled=cancelled_after_budget(a,result)
            if 'prompt_tokens' in phases and phases['prompt_tokens']!=usage_tokens and not (usage_tokens is None and budget_cancelled):
                raise ValueError(f'{arm}: server/API token mismatch at rid={rid}')
            a['server']=phases
        # Identify native response message IDs so a compact-boundary event can
        # be attached to the next real action-producing API response.
        message_ids={};usage_rows=[]
        for a in apis:
            response_file=p/f"response_{a['id']:04d}.sse.gz"
            if response_file.exists():
                for line in gzip.decompress(response_file.read_bytes()).splitlines():
                    if line.startswith(b'data: '):
                        try:e=json.loads(line[6:])
                        except json.JSONDecodeError:continue
                        if e.get('type')=='message_start':message_ids[e['message']['id']]=a['id'];break
            usage_rows.append({'api_id':a['id'],'after_step':a['after_step'],'input_tokens':a.get('input_usage',{}).get('input_tokens'),
                'output_tokens':a.get('output_usage',{}).get('output_tokens'),'ttft_ms':a.get('ttft_ms'),'stop_reason':a.get('stop_reason')})
        compact_next_api=set();pending=None;compactions=[]
        for e in events:
            if e.get('subtype')=='compact_boundary':pending=e['compact_metadata']
            if pending is not None and e.get('type')=='assistant':
                api_id=message_ids.get(e.get('message',{}).get('id'))
                if api_id is not None:
                    compact_next_api.add(api_id)
                    # In this serial trace the summary is the generation just
                    # before the first assistant response after compact_boundary.
                    # Claude Code may consume a summary ending in tool_use too;
                    # filtering for end_turn can silently reuse an older summary.
                    prior=[a for a in generated if a['id']<api_id]
                    summary=prior[-1] if prior else None
                    compactions.append({'metadata':pending,'next_action_api_id':api_id,'summary_api':summary})
                    pending=None
        rows=[]
        for previous,s in zip(steps,steps[1:]):
            ready=previous['observation_ready_ns'];start=s['started_ns']
            calls=[a for a in apis if a['started_ns']>=ready and a.get('first_content_ns',start+1)<=start and a.get('stop_reason')=='tool_use']
            a=calls[-1] if calls else None
            # Maintenance may occur after first content or in an earlier,
            # non-action response. Attribute all model calls in this decision
            # interval, not just the final action-producing call.
            decision_calls=[call for call in generated if ready<=call['started_ns']<start]
            selection_calls_before_action=[call for call in decision_calls if selection_request(call)]
            row={'arm':arm,'step':s['step'],'action':s['action']['action'],'decision_wait_ms':s['decision_wait_ms'],
                'observation_to_action_response_first_content_ms':None if a is None else (a['first_content_ns']-ready)/1e6,
                'observation_to_first_tool_arguments_ms':None if a is None or 'first_tool_args_ns' not in a else (a['first_tool_args_ns']-ready)/1e6,
                'api_ttft_ms':None if a is None else a.get('ttft_ms'),'api_id':None if a is None else a['id'],
                'input_tokens':None if a is None else a.get('input_usage',{}).get('input_tokens'),
                'native_compaction_before_action':any(call['id'] in compact_next_api for call in decision_calls),
                'kvmem_selection_before_action':bool(selection_calls_before_action),
                'kvmem_prefill_selection_before_action':any(prefill_selection(call) for call in decision_calls),
                'kvmem_middecode_selection_before_action':any(middecode_refreshes(call)>0 for call in decision_calls),
                'kvmem_selection_api_ids':[call['id'] for call in selection_calls_before_action],
                'kvmem_middecode_refreshes_before_action':sum(middecode_refreshes(call) for call in decision_calls),
                'kvmem_middecode_maintenance_ms':sum(
                    call.get('server',{}).get('kvmem_middecode_query_ms',0)+
                    call.get('server',{}).get('kvmem_middecode_reselect_ms',0)
                    for call in decision_calls if middecode_refreshes(call)>0)}
            for key in ('preprocess_ms','engine_ttft_ms','native_setup_ms','native_prefill_ms','native_post_semantic_ms',
                        'native_post_query_replay_ms','native_post_guided_query_ms','native_postprefill_ms',
                        'pre_json_ms','pre_validate_ms','server_ttft_ms','response_ttft_ms',
                        'native_wrapper_parse_ms','native_wrapper_convert_ms','native_wrapper_bridge_ms',
                        'native_wrapper_handler_ms','native_wrapper_total_ms'):
                row[key]=None if a is None else a.get('server',{}).get(key)
            for key in ('request_body_read_ms','request_forward_ms'):
                row[key]=None if a is None else a.get(key)
            row['observation_to_api_start_ms']=None if a is None else (a['started_ns']-ready)/1e6
            row['api_outside_inner_server_ttft_ms']=None if a is None or row['response_ttft_ms'] is None else a['ttft_ms']-row['response_ttft_ms']
            rows.append(row)
        combined.extend(rows)
        for compact in compactions:
            compact['next_action']=next((r for r in rows if r['api_id']==compact['next_action_api_id']),None)
        selection_calls=[a for a in generated if selection_request(a)]
        regular=[r for r in rows if not r['native_compaction_before_action'] and not r['kvmem_selection_before_action']]
        maintenance=[r for r in rows if r['native_compaction_before_action'] or r['kvmem_selection_before_action']]
        api_errors=[a for a in apis if a.get('status',200)>=400 or a.get('stream_error') or a.get('relay_error')]
        post_episode_errors=[a for a in api_errors if cancelled_after_budget(a,result)]
        summaries[arm]={'result':result,'native_compactions':len(boundaries),'compact_boundaries':boundaries,
            'compaction_details':compactions,'kvmem_selections':len(selection_calls),'selection_apis':selection_calls,
            'kvmem_prefill_selection_requests':sum(prefill_selection(a) for a in generated),
            'kvmem_middecode_selection_requests':sum(middecode_refreshes(a)>0 for a in generated),
            'kvmem_middecode_refreshes':sum(middecode_refreshes(a) for a in generated),
            'kvmem_selection_events':sum(int(prefill_selection(a))+middecode_refreshes(a) for a in generated),
            'selection_count_definition':'kvmem_selections counts requests with prefill selection or middecode refresh; events counts individual selections.',
            'middecode_timing_definition':'Per-action kvmem_middecode_maintenance_ms sums recorded query and reselect timers across the decision interval; includes query subphases once and excludes uninstrumented final MTP rebasing. Full action wait includes the whole pause; TTFT may precede it.',
            'maintenance_action_wait_ms':stats(r['decision_wait_ms'] for r in maintenance),
            'maintenance_step_ttft_ms':stats(r['observation_to_action_response_first_content_ms'] for r in maintenance),
            'decision_wait_ms':stats(r['decision_wait_ms'] for r in rows),'step_ttft_ms':stats(r['observation_to_action_response_first_content_ms'] for r in rows),
            'regular_decision_wait_ms':stats(r['decision_wait_ms'] for r in regular),'regular_step_ttft_ms':stats(r['observation_to_action_response_first_content_ms'] for r in regular),
            'api_ttft_ms':stats(a.get('ttft_ms') for a in apis),'input_tokens_max':max((a.get('input_usage',{}).get('input_tokens',0) for a in apis),default=0),
            'api_total_ms_sum':sum(a.get('total_ms',0) for a in apis),
            'input_tokens_sum':sum(a.get('input_usage',{}).get('input_tokens',0) for a in apis),
            'output_tokens_sum':sum(a.get('output_usage',{}).get('output_tokens',0) for a in apis),
            'api_errors':api_errors,
            'execution_api_errors':[a for a in api_errors if a not in post_episode_errors],
            'post_episode_api_errors':post_episode_errors,
            'tool_errors':tool_errors,
            'resource_changes':resource_changes,
            'api_usage':usage_rows}
        resource_samples=[r for r in jsonl(root/'resources.jsonl') if r.get('arm')==arm and r.get('stage')==arm and r.get('server_memory_kib')]
        if resource_samples:
            summaries[arm]['resource_samples']={
                'n':len(resource_samples),
                'min_host_available_gib':min(r['mem_available_bytes'] for r in resource_samples)/2**30,
                'peak_server_rss_gib':max(r['server_memory_kib']['VmRSS'] for r in resource_samples)/2**20,
                'peak_server_hwm_gib':max(r['server_memory_kib']['VmHWM'] for r in resource_samples)/2**20,
                'peak_server_swap_gib':max(r['server_memory_kib'].get('VmSwap',0) for r in resource_samples)/2**20}
    out=root/'analysis';out.mkdir(exist_ok=True)
    if len(summaries)==2:
        common_steps=min(max(r['step'] for r in combined if r['arm']==arm) for arm in summaries)
        for arm,summary in summaries.items():
            common=[r for r in combined if r['arm']==arm and r['step']<=common_steps]
            summary['common_action_window']={'max_step':common_steps,
                'step_ttft_ms':stats(r['observation_to_action_response_first_content_ms'] for r in common),
                'decision_wait_ms':stats(r['decision_wait_ms'] for r in common)}
            regular=[r for r in combined if r['arm']==arm and not r['native_compaction_before_action'] and not r['kvmem_selection_before_action']]
            summary['regular_phase_windows']={name:{key:stats(r[key] for r in part) for key in
                ('api_ttft_ms','preprocess_ms','engine_ttft_ms','pre_json_ms','pre_validate_ms','api_outside_inner_server_ttft_ms','observation_to_api_start_ms',
                 'native_wrapper_parse_ms','native_wrapper_convert_ms','native_wrapper_bridge_ms','native_wrapper_total_ms',
                 'request_body_read_ms','request_forward_ms')}
                for name,part in [('first50',regular[:50]),('last50',regular[-50:])]}
    (out/'metrics.json').write_text(json.dumps(summaries,indent=2)+'\n')
    if combined:
        with (out/'step_latency.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
    if len(summaries)==2:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
        for arm,color in [('qw3','#326bb0'),('kvmem','#008878')]:
            rs=[r for r in combined if r['arm']==arm]
            for ax,key,title in zip(axes,['observation_to_action_response_first_content_ms','decision_wait_ms','input_tokens'],
                ['Observation to first content of action response (includes preceding compaction)','Observation to next desktop action dispatch','Full request history tokens (before KVMem selection)']):
                valid=[r for r in rs if r[key] is not None]
                ax.plot([r['step'] for r in valid],[r[key]/(1000 if key!='input_tokens' else 1) for r in valid],label=arm,color=color,lw=1)
                compact=[r for r in valid if r['native_compaction_before_action']]
                if compact:ax.scatter([r['step'] for r in compact],[r[key]/(1000 if key!='input_tokens' else 1) for r in compact],color='#ce6132',marker='D',s=35,zorder=5,label='Native compaction')
                for flag,marker,label in [('kvmem_prefill_selection_before_action','^','KVMem prefill selection'),
                                          ('kvmem_middecode_selection_before_action','s','KVMem middecode selection')]:
                    selected=[r for r in valid if r[flag]]
                    if selected:ax.scatter([r['step'] for r in selected],[r[key]/(1000 if key!='input_tokens' else 1) for r in selected],color='#7143a8',marker=marker,s=25,zorder=5,label=label)
                ax.set_title(title,loc='left',fontsize=10);ax.grid(alpha=.2);ax.set_ylabel('seconds' if key!='input_tokens' else 'tokens')
        axes[0].legend();axes[-1].set_xlabel('Desktop action step (trajectories may differ)')
        plotted_changes=set()
        for summary in summaries.values():
            for change in summary['resource_changes']:
                change_key=(change['step'],change.get('requested_at'))
                if change_key in plotted_changes:continue
                plotted_changes.add(change_key)
                for ax in axes:ax.axvline(change['step'],color='#777777',ls=':',lw=.9)
                axes[-1].annotate('GitLab stop requested',xy=(change['step'],.93),xycoords=('data','axes fraction'),
                    xytext=(4,0),textcoords='offset points',rotation=90,ha='left',va='top',fontsize=8,color='#555555',annotation_clip=True)
        task=summaries['qw3']['result']['task']
        fig.suptitle(f'OSWorld {task} — Claude Code 2.1.148 / Qwen3.8-27B / FP8 KV',fontsize=13)
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'latency_comparison.{ext}',dpi=180)
        for ax in axes:ax.set_xlim(0,common_steps+1)
        fig.suptitle(f'OSWorld {task} — first {common_steps} actions of each trajectory / Claude Code 2.1.148',fontsize=13)
        for ext in ['png','pdf','svg']:fig.savefig(out/f'latency_comparison_common_window.{ext}',dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
        for arm,color in [('qw3','#326bb0'),('kvmem','#008878')]:
            rs=[r for r in combined if r['arm']==arm and not r['native_compaction_before_action'] and not r['kvmem_selection_before_action']]
            for ax,key in zip(axes,['observation_to_action_response_first_content_ms','decision_wait_ms']):
                valid=[r for r in rs if r[key] is not None]
                ax.scatter([r['step'] for r in valid],[r[key]/1000 for r in valid],s=6,alpha=.5,label=arm,color=color)
                ax.grid(alpha=.2);ax.set_ylabel('seconds');ax.legend()
        axes[0].set_title('First content of action response — maintenance steps excluded',loc='left')
        axes[1].set_title('Next desktop action dispatch — maintenance steps excluded',loc='left')
        axes[-1].set_xlabel('Desktop action step (trajectories may differ)');fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(out/f'latency_regular_steps.{ext}',dpi=180)
        plt.close(fig)
    print(json.dumps({k:{x:y for x,y in v.items() if x not in ('api_usage','compact_boundaries','selection_apis','compaction_details')} for k,v in summaries.items()},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);analyze(parser.parse_args().root)
