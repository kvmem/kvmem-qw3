"""Read-only image-policy measurement on terminal traces; no inference replay."""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from osworld_linux.archive import read_request
from visual_working_context_candidate import HISTORY_TOOLS


def diagnose(candidate, campaign, output, queue_path=None):
    candidate,campaign,output=map(lambda p:Path(p).resolve(),(candidate,campaign,output))
    build=json.loads((candidate/'build.json').read_text())
    package=candidate/'framework/qw3_computer'
    if build.get('kind')!='isolated_visual_working_context_candidate':raise ValueError('Declare candidate')
    for name,digest in build['candidate_files_sha256'].items():
        if hashlib.sha256((package/name).read_bytes()).hexdigest()!=digest:raise ValueError('Candidate changed: '+name)
    sys.path.insert(0,str(package.parent));visual=importlib.import_module('qw3_computer.visual_working_context')
    queue_path=Path(queue_path).resolve() if queue_path else campaign/'queue.json'
    if queue_path not in (campaign/'queue.json',campaign/'continued_queue.json'):
        raise ValueError('Use this campaign original or continued terminal queue')
    queue_bytes=queue_path.read_bytes();queue=json.loads(queue_bytes);items=[]
    if queue['protocol_sha256']!=hashlib.sha256((campaign/'protocol.json').read_bytes()).hexdigest():
        raise ValueError('Frozen campaign protocol changed')
    for trial in queue['trials']:
        identity=trial['id']
        if Path(identity).name!=identity:raise ValueError('Bad trial identity')
        attempt=campaign/'attempts'/identity
        result_bytes=(attempt/'result.json').read_bytes();result=json.loads(result_bytes)
        if not result.get('finished_at') or result!=trial['result']:raise ValueError('Only the exact recorded terminal attempt is eligible')
        traces=[]
        for trace in (attempt/'app/inference').iterdir():
            if trace.is_dir() and (trace/'metrics.json').is_file():
                metrics=json.loads((trace/'metrics.json').read_text());traces.append((metrics['started_at'],trace))
        trace=max(traces,key=lambda row:row[0])[1]
        source_bytes=read_request(trace);source=json.loads(source_bytes);before=copy.deepcopy(source)
        # Existing frozen requests did not offer the new tool. Simulate only
        # the future tool admission; this is not a request dispatched to QW3.
        admitted=source|{'tools':source.get('tools',[])+[{'type':'function','function':{
            'name':'mcp__computer__history','description':HISTORY_TOOLS[0]['description'],
            'parameters':HISTORY_TOOLS[0]['inputSchema']}}]}
        policies=[]
        plain,report=visual.working_context(admitted)
        policies.append(report|{'pin_fixture':'none','presented_json_bytes':len(json.dumps(plain,ensure_ascii=False).encode())})
        calls={};eligible=[]
        for message in admitted['messages']:
            for call in message.get('tool_calls',[]):calls[call['id']]=call.get('function',{}).get('name')
            item=visual.eligible_observation(message,calls)
            if item:eligible.append(item['reference'])
        if eligible:
            pinned,pin_report=visual.working_context(admitted,[eligible[0]])
            policies.append(pin_report|{'pin_fixture':'first-eligible-observation-for-boundary-check-only',
                'presented_json_bytes':len(json.dumps(pinned,ensure_ascii=False).encode())})
        if source!=before:raise ValueError('Original request mutated')
        if read_request(trace)!=source_bytes:raise ValueError('Original retained source changed')
        if (attempt/'result.json').read_bytes()!=result_bytes:raise ValueError('Terminal business result changed')
        items.append({'trial':identity,'terminal_result_sha256':hashlib.sha256(result_bytes).hexdigest(),
            'original_score':result.get('score'),'original_uncertain_actions':result.get('uncertain_actions'),
            'trace':str(trace),'original_request_sha256':hashlib.sha256(source_bytes).hexdigest(),
            'original_request_bytes':len(source_bytes),'hypothetical_admitted_bytes':len(json.dumps(admitted,ensure_ascii=False).encode()),
            'tool_admission_simulated':True,'original_request_and_result_unchanged':True,'policies':policies})
    report={'kind':'read_only_terminal_image_context_diagnostic','at':time.time(),'passed':True,
        'candidate_build_sha256':hashlib.sha256((candidate/'build.json').read_bytes()).hexdigest(),
        'recipe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'campaign_protocol_sha256':queue['protocol_sha256'],'queue_snapshot_sha256':hashlib.sha256(queue_bytes).hexdigest(),
        'queue_path':str(queue_path),'terminal_trials':len(items),'items':items,
        'model_requests_dispatched':0,'device_inputs':0,'latency_improvement_measured':False,
        'success_improvement_measured':False,'model_input_token_reduction_measured':False,
        'limits':['Hypothetical image presentation on existing histories; no new QW3 decisions or actions.',
                  'Historical read tool and system guidance would change the harness; this is not a KVMem isolation test.',
                  'First eligible observation pin is a boundary fixture, not an inferred task reference.',
                  'Textual history remains; missing pinned source causes full-input fallback rather than silent loss.']}
    with output.open('x',encoding='utf-8') as file:file.write(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','campaign','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--queue',type=Path,help='Explicit original or continued queue in the same campaign')
    args=p.parse_args();result=diagnose(args.candidate,args.campaign,args.output,args.queue)
    print(json.dumps({'passed':True,'terminal_trials':result['terminal_trials'],'model_requests_dispatched':0,
                      'image_counts':[{ 'trial':item['trial'],'source':item['policies'][0]['source_image_count'],
                        'presented':item['policies'][0]['retained_image_count']} for item in result['items']]}))


if __name__=='__main__':main()
