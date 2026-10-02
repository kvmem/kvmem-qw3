"""Frozen P2 screenshot/text by actual thinking on/off, two repetitions per cell."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import random
import re
import time
from urllib.parse import urlparse

import aiohttp
from PIL import Image

from compare_replay import monitor
from qw3_computer.protocol import read_sse
from visual_diagnostic_run import digest,image_item,save

CATEGORIES={'calculation','revision','selection','workflow'}


def strict_equal(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(expected,dict):return set(actual)==set(expected) and all(strict_equal(actual[k],v) for k,v in expected.items())
    if isinstance(expected,list):return len(actual)==len(expected) and all(strict_equal(a,b) for a,b in zip(actual,expected))
    return actual==expected


def grade(sample,answer):
    try:
        value=answer.strip()
        if value.startswith('```json\n') and value.endswith('\n```'):value=value[8:-4]
        parsed=json.loads(value)
        if not isinstance(parsed,dict):raise ValueError('Expected a JSON object')
        expected=sample['expected']
        return {'passed':strict_equal(parsed,expected),'format_valid':set(parsed)==set(expected),'answer':parsed,
                'field_matches':{k:k in parsed and strict_equal(parsed[k],v) for k,v in expected.items()}}
    except (ValueError,TypeError) as exc:return {'passed':False,'format_valid':False,'error':str(exc)}


def request_for(sample,root,representation,thinking,max_tokens):
    intro='请仅依据以下材料处理任务，不操作软件，不使用外部知识补齐条件。\n'+sample['question']+'\n只在最终回答输出符合以下结构的 JSON：\n'+json.dumps(sample['schema'],ensure_ascii=False)
    content=[{'type':'text','text':intro}]
    if representation=='text':content.append({'type':'text','text':sample['equivalent_text']})
    elif representation=='screenshot':
        with Image.open(root/sample['image']) as image:content.append(image_item(image.convert('RGB')))
    else:raise ValueError('Unknown representation')
    return {'model':'qw3','messages':[{'role':'user','content':content}], 'temperature':0,'top_p':1,
            'enable_thinking':thinking,'max_tokens':max_tokens,'stream':True,'stream_options':{'include_usage':True}}


def validate(corpus,root):
    samples=corpus['samples']
    if len(samples)!=12 or len({s['id'] for s in samples})!=12 or Counter(s['category'] for s in samples)!=Counter({k:3 for k in CATEGORIES}):
        raise ValueError('Expected twelve unique samples, three per category')
    for category in CATEGORIES:
        if Counter(s['split'] for s in samples if s['category']==category)!=Counter(development=2,heldout=1):raise ValueError('Incorrect frozen split')
    for s in samples:
        if not re.fullmatch(r'[a-z0-9-]{1,60}',s['id']):raise ValueError('Unsafe sample identity')
        path=(root/s['image']).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path)!=s['image_sha256']:raise ValueError('Screenshot identity mismatch')
        if s['equivalent_text']!=s['title']+'\n'+'\n'.join(s['facts']):raise ValueError('Equivalent text changed')
        if not s['ground_truth_reviewed'] or not s['screenshot_reviewed'] or not s['equivalence_reviewed']:raise ValueError('Unreviewed case')
        if not s.get('expected') or not s.get('question') or not s.get('schema'):raise ValueError('Missing grading definition')
    if corpus['max_tokens']!=4096 or corpus['repetitions']!=2:raise ValueError('Unfrozen output budget or repetition count')
    return samples


async def run(args):
    frozen=args.corpus.read_bytes();corpus=json.loads(frozen);samples=validate(corpus,args.corpus.parent)
    engine=json.loads((args.engine/'manifest.json').read_text())
    actual=[v.decode() for v in Path(f"/proc/{engine['pid']}/cmdline").read_bytes().split(b'\0') if v]
    port=int(engine['command'][engine['command'].index('--port')+1]);endpoint=urlparse(args.endpoint)
    if actual!=engine['command'] or engine['kvmem']!='off' or endpoint.scheme!='http' or endpoint.hostname not in ('127.0.0.1','localhost') or endpoint.port!=port or endpoint.path:
        raise ValueError('Owned local KVMem-off engine required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    order=[(s,p,t,r) for s in samples for p in ('screenshot','text') for t in (False,True) for r in (1,2)]
    random.Random(20261005).shuffle(order)
    report={'kind':'P2_reasoning_2x2','status':'running','started_at':time.time(),'planned_requests':96,'trials':[],
            'engine':engine,'corpus_sha256':digest(args.corpus),'source_sha256':digest(Path(__file__)),
            'max_tokens':4096,'temperature':0,'actual_task_success_claimed':False,
            'order':[{'sample':s['id'],'representation':p,'thinking':t,'repetition':r} for s,p,t,r in order]}
    save(out/'result.json',report)
    stopped=asyncio.Event();sampler=asyncio.create_task(monitor(engine['pid'],out/'resources.jsonl',stopped))
    try:
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=180,sock_read=90)) as client:
            for index,(sample,representation,thinking,repetition) in enumerate(order):
                directory=out/f'{index:02d}-{sample["id"]}';directory.mkdir()
                trial={'sample':sample['id'],'category':sample['category'],'split':sample['split'],'representation':representation,
                       'thinking':thinking,'repetition':repetition,'started_at':time.time()}
                answer='';reasoning='';done=False;started=time.monotonic()
                try:
                    if args.corpus.read_bytes()!=frozen or digest(args.corpus.parent/sample['image'])!=sample['image_sha256']:
                        raise ValueError('Frozen inputs changed; no request dispatched')
                    body=request_for(sample,args.corpus.parent,representation,thinking,4096);save(directory/'request.json',body)
                    trial['request_sha256']=digest(directory/'request.json')
                    async with client.post(args.endpoint+'/v1/chat/completions',json=body) as response:
                        response.raise_for_status()
                        with (directory/'stream.jsonl').open('w',encoding='utf-8') as stream:
                            async for data in read_sse(response.content):
                                if data=='[DONE]':done=True;break
                                stream.write(data+'\n');stream.flush();chunk=json.loads(data)
                                if chunk.get('usage'):trial['usage']=chunk['usage']
                                for choice in chunk.get('choices',[]):
                                    delta=choice.get('delta',{})
                                    if delta.get('content') or delta.get('reasoning_content'):trial.setdefault('first_generated_text_seconds',time.monotonic()-started)
                                    if delta.get('content'):trial.setdefault('first_final_text_seconds',time.monotonic()-started)
                                    answer+=delta.get('content') or '';reasoning+=delta.get('reasoning_content') or ''
                                    if choice.get('finish_reason'):trial['finish_reason']=choice['finish_reason']
                    if not done:raise ValueError('Incomplete stream')
                    if bool(reasoning)!=thinking:raise ValueError('Actual reasoning stream disagrees with requested mode')
                    trial.update(status='complete',answer=answer,reasoning_characters=len(reasoning),score=grade(sample,answer))
                    if trial.get('finish_reason')=='length':trial['score']['passed']=False;trial['budget_exhausted']=True
                except Exception as exc:trial.update(status='infrastructure_or_mode_error',answer=answer,error=f'{type(exc).__name__}: {exc}')
                trial['elapsed_seconds']=time.monotonic()-started
                save(directory/'result.json',trial);report['trials'].append(trial);save(out/'result.json',report)
                print(json.dumps({'completed':len(report['trials']),'sample':sample['id'],'thinking':thinking,'representation':representation,'status':trial['status'],'passed':trial.get('score',{}).get('passed')}),flush=True)
                if trial['status']!='complete':report['status']='halted';break
            else:report['status']='complete'
    finally:
        stopped.set();await sampler
        report['finished_at']=time.time();save(out/'result.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','engine','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--endpoint',default='http://127.0.0.1:18080')
    result=asyncio.run(run(parser.parse_args()));print(json.dumps({'status':result['status'],'completed':len(result['trials'])}))
