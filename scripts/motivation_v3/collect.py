"""Serial, resumable attention collection from a frozen replay manifest."""
import argparse
import base64
import copy
import gzip
import hashlib
import json
import time
from pathlib import Path
from collections import defaultdict
import requests
from trace_io import read_request, save_segment
from prepare_dataset import clean_message, dialogue_messages

BASH_TOOL={'type':'function','function':{'name':'bash','description':'Execute a bash command.',
    'parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}
QUERY_TOOL={'type':'function','function':{'name':'query','description':'Query candidate entities using conditions.',
    'parameters':{'type':'object','properties':{'conditions':{'type':'array','items':{'type':'object'}}},'required':['conditions']}}}


def read_source(row,cache):
    path=row['source']
    if path not in cache:
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['source_sha256']:
            raise ValueError('source checksum changed: '+path)
        cache[path]=json.loads(raw) if not path.endswith('.jsonl') else raw.decode().splitlines()
    data=cache[path]
    kind=row['source_kind']
    if kind=='mini_swe_replay':return data['messages']
    if kind=='longmemeval_conversation_replay':return dialogue_messages(data[row['source_row']])
    if kind=='agentlongbench_replay':return json.loads(data[row['source_line']])['messages']
    return None


def frame_messages(row,frame,source,image_cache):
    if row['domain']!='gui':
        return [clean_message(m) for m in source[:frame['cut']]]
    raw=Path(frame['message_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=frame['message_sha256']:
        raise ValueError('GUI message checksum changed')
    messages=json.loads(raw)
    for m in messages:
        if not isinstance(m.get('content'),list):continue
        for block in m['content']:
            if block.get('type')!='image_url':continue
            meta=block['image_url'];h=meta['sha256']
            if h not in image_cache:
                raw=Path(row['image_map'][h]).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('GUI image checksum mismatch')
                image_cache[h]='data:image/png;base64,'+base64.b64encode(raw).decode()
            block['image_url']={'url':image_cache[h]}
    return messages


def json_gz(path,data):
    with gzip.open(path,'wt',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False)


def lcp(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def post(session,url,payload,timeout):
    r=session.post(url,json=payload,timeout=timeout)
    if r.status_code!=200:raise RuntimeError(f'HTTP {r.status_code}: {r.text[:1000]}')
    return r.json()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--endpoint',default='http://127.0.0.1:18190')
    ap.add_argument('--trace',type=Path,required=True);ap.add_argument('--input-trace',type=Path,required=True)
    ap.add_argument('--only',action='append');ap.add_argument('--steps',type=int)
    ap.add_argument('--timeout',type=int,default=900);ap.add_argument('--limit-requests',type=int)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());cache={};session=requests.Session();session.trust_env=False
    deadline=time.monotonic()+300
    while True:
        try:
            ready=session.get(args.endpoint+'/v1/models',timeout=3)
            if ready.status_code==200:break
        except requests.RequestException:pass
        if time.monotonic()>deadline:raise RuntimeError('Probe did not become ready; no requests dispatched')
        time.sleep(2)
    domains=defaultdict(list)
    for r in manifest['trajectories']:
        if not args.only or r['id'] in args.only:domains[r['domain']].append(r)
    ordered=[]
    while any(domains.values()):
        for rows in domains.values():
            if rows:ordered.append(rows.pop(0))
    emitted=0;failures=0
    for row in ordered:
        out=args.out/row['id'];out.mkdir(exist_ok=True)
        image_cache={};source=read_source(row,cache)
        frames=row['frames'][:args.steps] if args.steps else row['frames']
        previous=None
        for index,frame in enumerate(frames):
            path=out/f'step_{index:03d}.json'
            if path.exists():
                record=json.loads(path.read_text())
                if record['status']=='collected':
                    with gzip.open(record['input_tokens_path'],'rt') as f:previnput=json.load(f)
                    previous=(record,previnput,frame)
                    continue
                # A failed request is retained and requires deliberate retry.
                previous=None
                continue
            if args.limit_requests is not None and emitted>=args.limit_requests:
                print('request limit reached',flush=True);return
            if args.trace.exists() and args.trace.stat().st_size>30*1024**3:
                raise RuntimeError('30 GiB raw trace cap reached; archive before continuing')
            if __import__('shutil').disk_usage(args.out).free<40*1024**3:
                raise RuntimeError('Less than 40 GiB disk free; collection stopped without deleting data')
            payload={'model':manifest['model'],'messages':frame_messages(row,frame,source,image_cache),
                'max_tokens':manifest['max_decode_tokens'],'temperature':0,'seed':manifest['seed'],
                'enable_thinking':True,'thinking_budget':manifest['thinking_budget'],'stream':False}
            if row['domain']=='code':payload['tools']=[BASH_TOOL]
            if row['domain']=='tool_state':payload['tools']=[QUERY_TOOL]
            json_gz(out/f'request_{index:03d}.json.gz',payload)
            start=args.trace.stat().st_size if args.trace.exists() else 0
            input_start=args.input_trace.stat().st_size if args.input_trace.exists() else 0
            record=dict(trajectory_id=row['id'],domain=row['domain'],step=index,
                source_step=frame['source_step'],status='running',trace_start=start,
                trace_path=str(args.trace.resolve()),input_trace_path=str(args.input_trace.resolve()),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                started_at=time.time(),replay=True,closed_loop=False)
            path.write_text(json.dumps(record,indent=2)+'\n')
            t=time.monotonic()
            try:
                response=post(session,args.endpoint+'/v1/chat/completions',payload,args.timeout)
                json_gz(out/f'response_{index:03d}.json.gz',response)
                stop=args.trace.stat().st_size
                with args.input_trace.open('rb') as f:
                    f.seek(input_start);inputs=[json.loads(s) for s in f if s.strip()]
                if len(inputs)!=1:raise ValueError(f'expected one native input record, got {len(inputs)}')
                inp=inputs[0]
                if inp['append_base']!=0:raise ValueError('collection requires full-input requests')
                seg=read_request(args.trace,start,stop);prompt=len(inp['tokens'])
                prefill=int((seg['positions']<prompt).sum());decode=int((seg['positions']>=prompt).sum())
                if prefill<1:raise ValueError('prefill attention missing')
                if len(seg['positions'])>1 and not (seg['positions'][1:]-seg['positions'][:-1]==1).all():
                    raise ValueError('attention positions are not consecutive')
                info=save_segment(out/f'attention_{index:03d}.npz',seg)
                tokenpath=out/f'input_tokens_{index:03d}.json.gz';json_gz(tokenpath,inp)
                record.update(info,status='collected',prompt_tokens=prompt,
                    api_prompt_tokens=response.get('usage',{}).get('prompt_tokens'),
                    completion_tokens=response.get('usage',{}).get('completion_tokens'),
                    prefill_samples=prefill,decode_samples=decode,
                    input_tokens_path=str(tokenpath.resolve()),
                    finish_reason=response['choices'][0].get('finish_reason'),
                    prefix_identity_verified=False,trace_end=stop)
                if previous and frame['source_step']==previous[2]['source_step']+1:
                    prev,old,oldframe=previous
                    common=lcp(old['tokens'],inp['tokens'])
                    if old['mrope'] and inp['mrope']:common=min(common,lcp(old['mrope'],inp['mrope']))
                    # For images, verify the retained image sequence as well as
                    # placeholder token IDs and positional encoding coordinates.
                    oldimages=oldframe.get('image_sha256',[]);newimages=frame.get('image_sha256',[])
                    pixels_ok=not oldimages or oldimages==newimages[:len(oldimages)]
                    if pixels_ok:
                        record.update(common_prefix_tokens=common,history_end_tokens=common,
                                      prefix_identity_verified=True)
                    else:
                        record['prefix_exclusion_reason']='image prefix changed or compacted'
                previous=(record,inp,frame)
                print(f"{row['id']} {index+1}/{len(row['frames'])}: prompt={prompt} prefill={prefill} decode={decode} {time.monotonic()-t:.1f}s",flush=True)
            except Exception as e:
                record.update(status='failed',error=str(e));previous=None;failures+=1
                print(f"FAILED {row['id']} step {index}: {e}",flush=True)
                # A timeout can leave an active server request; never dispatch
                # another request into an ambiguous trace interval.
                if isinstance(e,requests.exceptions.Timeout):
                    record['elapsed_sec']=time.monotonic()-t
                    path.write_text(json.dumps(record,indent=2)+'\n');raise
            record['elapsed_sec']=time.monotonic()-t;record['finished_at']=time.time()
            path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            emitted+=1
            if failures>=5:raise RuntimeError('Five failed requests; stopping to diagnose')
        cache.clear()
    print('collection pass finished',flush=True)


if __name__=='__main__':main()
