"""Read terminal native pilots; export counts/identities, never model text.

Backend calls are associated by exact emitted ID/name/arguments, not timestamps
alone. No tool dispatcher, model client, task initializer or grader runs here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import shutil
import time

from jsonschema import Draft202012Validator
from osworld_linux import archive


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return dict(path=str(Path(path).resolve()),sha256=digest(path))
def require(value,reason):
    if not value:raise ValueError(reason)


def tool_result(item):
    result=item.get('result')
    if not isinstance(result,dict):return None,'missing_tool_result'
    structured=result.get('structuredContent')
    if isinstance(structured,dict):return structured,None
    texts=[c.get('text') for c in result.get('content',[]) if c.get('type')=='text']
    if len(texts)!=1 or not isinstance(texts[0],str):return None,'ambiguous_tool_result'
    try:value=json.loads(texts[0])
    except ValueError:return None,'non_json_tool_result'
    if not isinstance(value,dict):return None,'non_object_tool_result'
    return value,None


def tool_summary(item):
    value,parse_error=tool_result(item);value=value or {}
    args=item.get('arguments',{});require(isinstance(args,dict),'Original tool arguments are not an object')
    problem=next((value[k] for k in ('problem','error') if isinstance(value.get(k),dict)),{})
    # Match an original literal program error, never infer from arbitrary text.
    workflow_error=None
    if item['tool']=='workflow' and parse_error=='non_json_tool_result':
        texts=[c.get('text') for c in item.get('result',{}).get('content',[]) if c.get('type')=='text']
        if texts==['Unknown step_id; inspect workflow status']:workflow_error='unknown_step_id'
    check=value.get('step_check') or {}
    if not isinstance(check,dict):check={}
    supplied=args.get('step') or {};supplied=supplied if isinstance(supplied,dict) else {}
    conditions=supplied.get('conditions',[])
    return dict(call_id=item['id'],tool=item['tool'],wire_status=item['status'],
        action_kind=args.get('action',{}).get('kind'),workflow_operation=args.get('operation'),
        step_supplied=isinstance(args.get('step'),dict),condition_kinds=[c.get('kind') for c in conditions if isinstance(c,dict)],
        argument_sha256=hashlib.sha256(json.dumps(args,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        body_status=value.get('status'),dispatched=value.get('dispatched'),delivery=value.get('delivery'),
        problem_code=problem.get('code'),problem_path=problem.get('path'),body_parse_error=parse_error,
        workflow_error_code=workflow_error,
        duration_ms=item.get('durationMs'),step_verified=check.get('verified'),
        condition_passes=[c.get('passed') for c in check.get('conditions',[]) if isinstance(c,dict)])


def backend(path):
    calls={};counts=Counter();finishes=Counter()
    for line in Path(path).read_text().splitlines():
        chunk=json.loads(line)
        for choice in chunk.get('choices',[]):
            if choice.get('finish_reason') is not None:finishes[choice['finish_reason']]+=1
            delta=choice.get('delta',{})
            for key in ('reasoning_content','content'):
                if isinstance(delta.get(key),str):counts[key+'_characters']+=len(delta[key])
            for part in delta.get('tool_calls',[]):
                index=part.get('index',0);require(type(index) is int and index>=0,'Backend tool index is invalid')
                call=calls.setdefault(index,dict(id='',name='',arguments=''))
                if part.get('id'):
                    require(call['id'] in ('',part['id']),'Backend changed tool call identity');call['id']=part['id']
                function=part.get('function',{})
                for key in ('name','arguments'):
                    value=function.get(key,'');require(isinstance(value,str),'Backend tool delta is invalid');call[key]+=value
                counts['tool_call_delta_events']+=1
    return calls,dict(counts),dict(finishes)


def associate(calls,items,schemas,seen):
    associations=[]
    for call in calls.values():
        arguments=json.loads(call['arguments'])
        matches=[i for i in items if i['id']==call['id'] and call['name'].endswith('__'+i['tool']) and i['arguments']==arguments]
        require(len(matches)==1,'Actual backend call identity/name/arguments do not uniquely match the original tool event')
        require(call['id'] not in seen,'Backend call was associated twice');seen.add(call['id'])
        require(call['name'] in schemas,'Backend generated an undeclared tool')
        errors=list(Draft202012Validator(schemas[call['name']]).iter_errors(arguments))
        associations.append(dict(call_id=call['id'],exact_original_arguments=True,original_schema_valid=not errors,
            schema_error_keywords=sorted({e.validator for e in errors})))
    return associations


def scoped_receipts(path,task_id):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    try:
        db.execute('BEGIN')
        rows=db.execute('SELECT status,count(*) FROM actions WHERE task=? GROUP BY status',(task_id,)).fetchall()
        count=db.execute('SELECT count(*) FROM revalidations r JOIN observations o ON o.id=r.source WHERE o.task=?',(task_id,)).fetchone()[0]
        return dict(actions=dict(rows),revalidations=count,readonly_transaction=True,live_database_not_hashed_as_immutable=True)
    finally:db.close()


def diagnose_pilot(protocol,spec,row,attempt,metrics_index):
    result_path=attempt/'integration-result.json';result=read(result_path)
    require(result==row['result'] and digest(result_path)==row['result_sha256'],'Original independent business result changed')
    require(result.get('owned_task_retired') is True and type(result.get('uncertain_actions')) is int and result['uncertain_actions']==0,
            'Task or input retirement is not independently confirmed')
    task_path=attempt/'task.json';task=read(task_path);manifest_path=attempt/'pilot-manifest.json';manifest=read(manifest_path)
    require(task['id']==manifest['task_id'] and task['status'] in ('completed','stopped','failed','paused'),'Actual task identity/status changed')
    require(manifest['deadline_seconds']==spec['seconds'] and manifest['app_runtime']==protocol['app_runtime'],'Original native budget/runtime changed')
    events_path=attempt/'codex-events.jsonl'
    items=[e['params']['item'] for e in map(json.loads,events_path.read_text().splitlines())
           if e.get('method')=='item/completed' and e.get('params',{}).get('item',{}).get('type')=='mcpToolCall']
    require(len(items)==result['tool_calls']==task['tool_calls_used'],'Completed original tool inventory differs from task/result')
    require(len({i['id'] for i in items})==len(items),'Original tool call IDs are not unique')
    summaries=[tool_summary(item) for item in items]
    progress_path=attempt/'progress.jsonl';progress=[json.loads(s) for s in progress_path.read_text().splitlines()]
    end=max(p['at'] for p in progress);start=task['started_at']
    selected=[(p,m) for p,m in metrics_index if start<=m['started_at']<=end]
    require(selected,'No actual model metrics in the original native interval')
    records=[];seen=set()
    for path,metric in sorted(selected,key=lambda pair:pair[1]['started_at']):
        require(metric['status'] in ('completed','cancelled','failed'),'Actual model request remains nonterminal')
        require(type(metric.get('elapsed_seconds')) in (int,float) and metric['elapsed_seconds']>=0,'Actual model elapsed time is missing')
        data=archive.read_request(path.parent);request=json.loads(data)
        require(request.get('enable_thinking') is True and request.get('max_tokens')==16384,'Actual model profile changed')
        schemas={t['function']['name']:t['function']['parameters'] for t in request['tools']}
        require(set(schemas)=={'mcp__computer__observe','mcp__computer__act','mcp__computer__workflow'},'Actual tool permissions changed')
        backend_path=path.parent/'backend.jsonl';calls,counts,finishes=backend(backend_path)
        associations=associate(calls,items,schemas,seen) if calls else []
        require(metric['status']=='completed' or not calls,'Incomplete backend tool output cannot be attributed as a completed event')
        records.append(dict(trace=path.parent.name,metrics=ref(path),backend=ref(backend_path),
            actual_request_sha256=archive.digest(data),started_at=metric['started_at'],status=metric['status'],
            elapsed_seconds=metric['elapsed_seconds'],first_model_delta_seconds=metric.get('first_model_delta_seconds'),
            usage=metric.get('usage'),stream_counts=counts,finish_reasons=finishes,tool_associations=associations))
    require(seen=={i['id'] for i in items},'Some actual original tool events lack exact backend attribution')
    serial=all(b['started_at']>=a['started_at']+a['elapsed_seconds']-.01 for a,b in zip(records,records[1:]))
    elapsed=sum(m['elapsed_seconds'] for m in records)
    before_delta=sum(m['first_model_delta_seconds'] for m in records if type(m.get('first_model_delta_seconds')) in (int,float))
    outputs=[m['usage']['output_tokens'] for m in records if isinstance(m.get('usage'),dict) and 'output_tokens' in m['usage']]
    inputs=[m['usage']['input_tokens'] for m in records if isinstance(m.get('usage'),dict) and 'input_tokens' in m['usage']]
    database=Path(protocol['app_runtime'])/'devices/linux-vm/desktop.sqlite'
    receipt=scoped_receipts(database,task['id'])
    completed=sum(s['body_status']=='completed' and s['delivery']=='delivered' for s in summaries if s['tool']=='act')
    require(receipt['actions'].get('completed',0)==completed and set(receipt['actions'])<={'completed','rejected'},'Scoped input receipts disagree with actual delivered tool results')
    return dict(trial=spec['id'],case=spec['case'],sources=[ref(p) for p in (result_path,task_path,manifest_path,events_path,progress_path)],
        original_accepted=result['accepted'],original_artifact_passed=result['artifact_passed'],task_seconds=result['seconds'],
        tool_calls=len(items),tool_seconds=sum(s['duration_ms'] for s in summaries)/1000,delivered_inputs=completed,
        body_status_counts=dict(Counter(s['body_status'] for s in summaries)),problem_code_counts=dict(Counter(s['problem_code'] for s in summaries if s['problem_code'])),
        problem_path_counts=dict(Counter(s['problem_path'] for s in summaries if s['problem_path'])),
        unstructured_tool_result_count=sum(s['body_parse_error'] is not None for s in summaries),
        workflow_operation_counts=dict(Counter(s['workflow_operation'] for s in summaries if s['workflow_operation'])),
        workflow_error_counts=dict(Counter(s['workflow_error_code'] for s in summaries if s['workflow_error_code'])),
        scoped_receipt_counts=receipt,model_request_count=len(records),model_status_counts=dict(Counter(m['status'] for m in records)),
        all_model_requests_serial=serial,model_elapsed_seconds=elapsed,model_elapsed_to_task_ratio=elapsed/result['seconds'],
        before_first_delta_seconds=before_delta,before_first_delta_to_model_elapsed_ratio=before_delta/elapsed,
        output_tokens_max=max(outputs) if outputs else None,output_tokens_median=statistics.median(outputs) if outputs else None,
        input_token_range=[min(inputs),max(inputs)] if inputs else None,
        tool_summaries=summaries,model_requests=records)


def diagnose(protocol_path,queue_path,output):
    protocol_path,queue_path,output=map(lambda p:Path(p).resolve(),(protocol_path,queue_path,output))
    protocol=read(protocol_path);state=read(queue_path)
    require(state['status']=='completed' and state['protocol_sha256']==digest(protocol_path),'Require the unchanged actual completed SDK queue')
    require([r['id'] for r in state['trials']]==[r['id'] for r in protocol['trials']],'Require the full original SDK inventory')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    shutil.copyfile(__file__,output/'recipe-source.py')
    report=dict(kind='readonly_original_native_pilot_exact_wire_and_contract_diagnostic',checked_at=time.time(),passed=False,
        protocol=ref(protocol_path),queue_snapshot=ref(queue_path),recipe=ref(output/'recipe-source.py'),results=[],model_calls_dispatched=0,device_inputs=0,
        reasoning_text_exported=False,arbitrary_tool_argument_values_exported=False,original_grades_changed=False,business_improvement_proven=False)
    try:
        metrics=[(p,read(p)) for p in (Path(protocol['app_runtime'])/'inference').glob('*/metrics.json')]
        for spec,row in zip(protocol['trials'],state['trials']):
            if spec['kind']=='normal-pilot':report['results'].append(diagnose_pilot(protocol,spec,row,protocol_path.parent/'attempts'/spec['id'],metrics))
        require(len(report['results'])==4,'Require both frozen text and Calc repeats')
        report.update(passed=True,limits=[
            'Four failed known development tasks; this is not a new task, optimization comparison or general model score.',
            'Time before the first model delta includes queueing, image/context processing and backend work; it is not pure prefill or thinking.',
            'Character counts do not measure reasoning tokens. Request usage measures total output.',
            'A rejected input is not automatically a mistaken refusal. Delivered input does not establish task correctness.',
            'No runtime, model profile, task, initializer, grader or fixed queue was changed.'])
    except BaseException as error:report.update(error_kind=type(error).__name__,error=str(error));raise
    finally:(output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','queue','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();r=diagnose(args.protocol,args.queue,args.output)
    print(json.dumps([dict(trial=x['trial'],model_fraction=x['model_elapsed_to_task_ratio'],before_delta_fraction=x['before_first_delta_to_model_elapsed_ratio'],
        output_max=x['output_tokens_max'],problems=x['problem_code_counts'],delivered=x['delivered_inputs']) for x in r['results']]))
