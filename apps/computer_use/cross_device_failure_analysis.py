"""Summarize observed tool failures without attributing unmeasured model skills.

Only terminal attempt receipts, completed MCP events and inference metrics are
read. Screenshots, requests, evaluator answers and credentials are not exported.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def distribution(values):
    values=sorted(values)
    return {'count':len(values),'p50_seconds':statistics.median(values) if values else None,
            'p95_seconds':values[max(0,math.ceil(len(values)*.95)-1)] if values else None,
            'p95_method':'nearest_rank'}


def numeric(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def completed_tools(path):
    if not path.exists():return []
    items=[]
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            item=row.get('params',{}).get('item',{})
            if row.get('method')=='item/completed' and item.get('type')=='mcpToolCall':items.append(item)
    return items


def tool_problem(item):
    parts=[part.get('text','') for part in (item.get('result') or {}).get('content',[]) if part.get('type')=='text']
    structured=[]
    for part in parts:
        try:body=json.loads(part)
        except ValueError:continue
        if isinstance(body,dict):structured.append(body)
    problem=next((body['problem'] for body in structured if isinstance(body.get('problem'),dict)),None)
    if problem is None:
        problem=next((body['error'] for body in structured if isinstance(body.get('error'),dict)),None)
    failed=item.get('status')=='failed' or bool(item.get('error')) or problem is not None
    if not failed:return None
    if problem:
        value={'category':problem.get('code','structured_tool_error'),'path':problem.get('path')}
        if 'Observation expired; observe again' in problem.get('message',''):
            value['signal']='expired_observation'
        return value
    text='\n'.join(parts)
    if 'DeviceRejected: Foreground changed; observe again' in text:
        return {'category':'preflight_refusal_misreported_as_uncertainty','path':None}
    if 'Observation expired; observe again' in text:
        return {'category':'observation_expired','path':None}
    return {'category':'other_tool_error','path':None}


def action_kind(item):
    """Count an event's declared kind without inferring validity or delivery."""
    arguments=item.get('arguments')
    if isinstance(arguments,str):
        try:arguments=json.loads(arguments)
        except ValueError:return 'unknown'
    if not isinstance(arguments,dict):return 'unknown'
    action=arguments.get('action')
    if isinstance(action,str):
        try:action=json.loads(action)
        except ValueError:return 'unknown'
    if not isinstance(action,dict) or not isinstance(action.get('kind'),str):return 'unknown'
    return action['kind']


def inspect_attempt(directory):
    result_path=directory/'result.json'
    result=json.loads(result_path.read_text())
    if not result.get('finished_at'):return None
    items=completed_tools(directory/'app/codex-events.jsonl')
    problems=[value for item in items if (value:=tool_problem(item))]
    timings=[item['durationMs']/1000 for item in items if numeric(item.get('durationMs'))]
    inputs=Counter(action_kind(item) for item in items if item.get('tool')=='act')
    inference=[]
    for path in sorted(directory.joinpath('app/inference').glob('*/metrics.json')):
        inference.append(json.loads(path.read_text()))
    errors=Counter()
    output_budget_hits=0
    for value in inference:
        error=value.get('error')
        if error:
            errors[error.get('code','error') if isinstance(error,dict) else 'unstructured_error']+=1
            if 'max_output_tokens' in json.dumps(error):output_budget_hits+=1
    status=result.get('status')
    terminal_budget_error='max_output_tokens' in json.dumps(result.get('turn_error'))
    return {'attempt':directory.name,'original_status':status,'original_score':result.get('score'),
        'result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'agent_status':result.get('agent_status'),'seconds':result.get('seconds'),
        'tool_calls_reported':result.get('tool_calls'),'completed_mcp_events':len(items),
        'tool_errors_observed':len(problems),'tool_problem_categories':dict(Counter(p['category'] for p in problems)),
        'tool_problem_signals':dict(Counter(p['signal'] for p in problems if p.get('signal'))),
        'tool_problem_paths':dict(Counter(p['path'] for p in problems if p['path'])),
        'act_call_kinds':dict(inputs),'tool_duration':distribution(timings),
        'inference_requests':len(inference),'inference_error_codes':dict(errors),
        'inference_output_budget_error_count':output_budget_hits,
        'terminal_output_budget_error':terminal_budget_error,
        'inference_total_duration':distribution([v['elapsed_seconds'] for v in inference if numeric(v.get('elapsed_seconds'))]),
        'first_model_delta':distribution([v['first_model_delta_seconds'] for v in inference if numeric(v.get('first_model_delta_seconds'))]),
        'uncertain_actions_reported':result.get('uncertain_actions')}


def analyze(campaign):
    rows=[row for path in sorted(campaign.glob('*/result.json')) if (row:=inspect_attempt(path.parent))]
    categories=Counter()
    signals=Counter()
    for row in rows:
        categories.update(row['tool_problem_categories'])
        signals.update(row['tool_problem_signals'])
    return {'kind':'observed_failure_evidence_not_model_skill_attribution','created_at':time.time(),
        'terminal_attempts':len(rows),'tool_problem_categories':dict(categories),
        'tool_problem_signals':dict(signals),
        'attempts_with_output_budget_errors':sum(row['terminal_output_budget_error'] or row['inference_output_budget_error_count']>0 for row in rows),
        'attempts_with_no_completed_mcp_event':sum(row['completed_mcp_events']==0 for row in rows),
        'attempts':rows,'limits':[
            'A completed MCP event is not proof of successful input or verified business progress.',
            'Counts describe observed error returns, not mutually exclusive root causes.',
            'A numeric original score does not establish complete task materials.',
            'Absence of tool errors does not establish correct localization, perception or reasoning.',
            'First model delta may be thinking text; it is not time to the first executable action.',
            'A completed inference metric can still end in a Codex max_output_tokens error; both evidence sources are checked.',
            'Tool duration includes observation/condition wait when the tool does so.',
            'Zero scores alone cannot separate visual interpretation, localization and planning.',
            'Only terminal attempts are included; earlier interrupted records remain separately retained.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.campaign)
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in {'attempts','limits'}},ensure_ascii=False))
