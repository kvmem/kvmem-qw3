"""Read-only scope-limited diagnosis of a terminal original OSWorld trial.

Exports timings, wire/receipt counts and rejection codes. It never exports
model prompts, screenshots, credentials, arbitrary tool arguments or grader
answers. Inference elapsed time includes image processing, context and shared
GPU effects; it is not a measurement of pure thinking time.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import statistics
import time

import kvmem_osworld_queue as owned


def body(item):
    result=item.get('result')
    if not isinstance(result,dict):raise ValueError('Tool result missing')
    value=result.get('structuredContent')
    if value is None:
        texts=[c['text'] for c in result['content'] if c['type']=='text']
        if len(texts)!=1:raise ValueError('Ambiguous tool result')
        value=json.loads(texts[0])
    if not isinstance(value,dict):raise ValueError('Unstructured tool result')
    return value


def read_database(path,query):
    db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    try:return db.execute(query).fetchall()
    finally:db.close()


def summarize(attempt):
    attempt=Path(attempt).resolve();result_path=attempt/'result.json';task_path=attempt/'app/task.json'
    result=json.loads(result_path.read_text());task=json.loads(task_path.read_text())
    if result.get('status')!='scored' or task.get('status') not in ('completed','stopped','failed','paused'):
        raise ValueError('Read only an independently scored terminal task')
    if type(result.get('seconds')) not in (int,float) or result['seconds']<=0:
        raise ValueError('Require the recorded actual task duration')
    rows=[]
    for line in (attempt/'app/codex-events.jsonl').read_text().splitlines():
        event=json.loads(line);item=event.get('params',{}).get('item',{})
        if event.get('method')!='item/completed' or item.get('type')!='mcpToolCall':continue
        value=body(item);error=value.get('error');problem=value.get('problem')
        code=error.get('code') if isinstance(error,dict) else None
        if code is None and isinstance(problem,dict):code=problem.get('code')
        args=item.get('arguments',{})
        rows.append(dict(tool=item['tool'],kind=args.get('action',{}).get('kind'),
            wire_status=item['status'],body_status=value.get('status'),rejection_code=code,
            duration_ms=item['durationMs'],step_supplied='step' in args))
    if len(rows)!=result['tool_calls'] or len(rows)!=task['tool_calls_used']:
        raise ValueError('Actual wire inventory differs from the terminal task count')
    metrics_paths=sorted((attempt/'app/inference').glob('*/metrics.json'))
    metrics=[json.loads(p.read_text()) for p in metrics_paths]
    if not metrics or any(m.get('status') not in ('completed','cancelled','failed') for m in metrics):
        raise ValueError('Model request metrics are missing or not terminal')
    seconds=sum(m.get('elapsed_seconds',0) for m in metrics)
    inputs=[m['usage']['input_tokens'] for m in metrics if m.get('usage',{}).get('input_tokens') is not None]
    desktop_path=attempt/'app/desktop/desktop.sqlite';gateway_path=attempt/'gateway/receipts.sqlite'
    desktop_counts=dict(read_database(desktop_path,'SELECT status,count(*) FROM actions GROUP BY status'))
    gateway_counts=dict(read_database(gateway_path,'SELECT status,count(*) FROM actions GROUP BY status'))
    revalidations=read_database(desktop_path,'SELECT evidence FROM revalidations')
    ages=[json.loads(row[0])['age_milliseconds'] for row in revalidations]
    files=[result_path,task_path,attempt/'app/codex-events.jsonl',desktop_path,gateway_path,*metrics_paths]
    return dict(kind='readonly_actual_terminal_trial_timing_and_rejection_diagnostic',checked_at=time.time(),
        attempt=str(attempt),sources=[dict(path=str(p),sha256=owned.digest(p)) for p in files],
        original_score=result.get('original_evaluation'),passed=result['passed'],stop_reason=result['stop_reason'],
        task_seconds=result['seconds'],tool_calls=len(rows),tool_seconds=sum(r['duration_ms'] for r in rows)/1000,
        model_requests=len(metrics),model_request_status_counts=dict(Counter(m['status'] for m in metrics)),
        model_elapsed_seconds=seconds,model_elapsed_to_task_ratio=seconds/result['seconds'],
        completed_request_input_token_range=[min(inputs),max(inputs)] if inputs else None,
        tool_body_status_counts=dict(Counter(r['body_status'] for r in rows)),
        rejection_code_counts=dict(Counter(r['rejection_code'] for r in rows if r['rejection_code'])),
        desktop_action_counts=desktop_counts,gateway_action_counts=gateway_counts,
        revalidation_count=len(revalidations),revalidation_age_range_ms=[min(ages),max(ages)] if ages else None,
        median_revalidation_age_ms=statistics.median(ages) if ages else None,
        tool_summaries=rows,new_model_requests=0,new_device_inputs=0,sources_changed=False,
        limits=['Single full-history arm; opposite arm and repeat are not yet available.',
                'Inference elapsed time includes visual/context/generation and GPU sharing. It is not pure thinking time.',
                'Evidence refresh is a confirmed refusal, not proof that the refusal was mistaken.',
                'Completed delivery receipts do not demonstrate correct application state or business success.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=summarize(args.attempt)
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('original_score','task_seconds','tool_calls','model_elapsed_seconds',
        'model_elapsed_to_task_ratio','rejection_code_counts','gateway_action_counts')},ensure_ascii=False))
