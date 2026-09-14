#!/usr/bin/env python3
"""Unscored GPU A/B for lossless transport and decode-tail replay checkpoints.

Keeps model/KV/selection/output settings identical between arms. The test uses
an intentionally length-limited response and edits only its rendered tail; it
is a regression fixture, not an OSWorld success-rate measurement.
"""
import argparse, copy, json, os, pathlib, re, subprocess, time, urllib.error, urllib.request


def run(args, arm):
    out = args.out / arm
    out.mkdir(parents=True, exist_ok=True)
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(json.loads(args.settings.read_text()))
    enabled = arm == 'optimized'
    env.update(QW3_KVMEM_PREFIX_CACHE_TRACE='1', QW3_MESSAGE_PREFIX_CACHE=str(int(enabled)),
               QW3_SERVER_TOKEN_CACHE=str(int(enabled)), QW3_VISION_CPU_CACHE_MIB='8192' if enabled else '0',
               QW3_KVMEM_DECODE_CHECKPOINT_TOKENS='1024' if enabled else '0')
    command = json.loads(args.command_json.read_text())
    command[0] = str(args.binary.resolve())
    port = int(command[command.index('--port')+1])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def post(path, body=None):
        data = None if body is None else json.dumps(body,ensure_ascii=False).encode()
        req = urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,
            headers={'Content-Type':'application/json','User-Agent':'opencode/1.18.18',
                     'x-opencode-session':'latency-tail-replay-regression'})
        with opener.open(req,timeout=900) as r: return json.load(r)
    rows=[]; ack=None; previous=[]
    def call(label, messages, max_tokens):
        nonlocal ack,previous
        count=0
        if enabled and ack:
            for a,b in zip(previous,messages):
                if a != b:break
                count+=1
        payload=dict(model='gpt-qw3',messages=messages[count:],temperature=0,seed=0,
                     max_tokens=max_tokens,enable_thinking=False,preserve_thinking=False)
        if enabled:
            payload['qw3_cache_messages']=True
            if count:payload['qw3_messages_prefix']={'id':ack['id'],'count':count}
        log_start=log_path.stat().st_size
        started=time.monotonic();res=post('/v1/chat/completions',payload);elapsed=time.monotonic()-started
        if enabled:
            ack=res.get('qw3_messages_cache');assert ack and ack['count']==len(messages),res.keys()
            previous=copy.deepcopy(messages)
        row=dict(label=label,wall_sec=elapsed,usage=res['usage'],timing=res.get('timing'),
                 prefix_messages=count,wire_bytes=len(json.dumps(payload).encode()),
                 response=res,log_start=log_start,log_end=log_path.stat().st_size)
        rows.append(row)
        (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
        return res['choices'][0]['message']
    def tool(messages, ident, content):
        messages.extend([{'role':'assistant','content':'I will inspect the latest record.',
                          'tool_calls':[{'id':ident,'type':'function','function':
                                        {'name':'read_file','arguments':'{"path":"record.log"}'}}]},
                         {'role':'tool','tool_call_id':ident,'content':content}])
    log_path=out/'server.log'
    with log_path.open('w') as log:
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        (out/'server.pid').write_text(str(proc.pid)+'\n')
        try:
            for i in range(180):
                if proc.poll() is not None:raise RuntimeError('server failed; see server.log')
                try:post('/health');break
                except OSError:time.sleep(1)
            else:raise RuntimeError('startup timeout')
            unit='Observation: the checkpoint state remains stable.\n'
            unit_count=post('/tokenize',{'content':unit*1000})['count']
            ballast=unit*max(1,args.history_tokens*1000//unit_count)
            messages=[{'role':'system','content':'You are a coding agent. Follow the instruction at the end of the latest tool result.'},
                      {'role':'user','content':'Read each appended record and follow its final instruction. Keep exact text in the conversation.'}]
            tool(messages,'history',ballast+'\nFor this turn reply exactly OK.')
            messages.append(call('prepare',messages,8))
            messages.append({'role':'user','content':
                'This is a deterministic decode-length regression, not a tool task. '
                'Do not call or mention any tool. Produce exactly 10000 numbered '
                'observations in plain text before stopping. Every observation must '
                'repeat the words CHECKPOINT STATE STABLE. Do not abbreviate. Start with line 1.'})
            long_prefix=copy.deepcopy(messages)
            generated=call('long_decode',messages,args.max_tokens)
            assert rows[-1]['usage']['completion_tokens'] >= args.max_tokens-10,'fixture did not reach output cap'
            raw=generated.get('qw3_raw_completion') or generated['content']
            assert len(raw)>1000
            # Emulate a client rendering a slightly shortened response tail.
            rewritten=raw[:-40]
            messages.append({'role':'assistant','content':rewritten,'qw3_raw_completion':rewritten})
            tool(messages,'verify','CHECKPOINT_STATUS=ready. Reply exactly OK.')
            messages.append(call('tail_rewrite',messages,8))
            tool(messages,'append','FINAL_CHECKPOINT=durable. Reply exactly OK.')
            messages.append(call('short_append',messages,8))
            # Rewrite farther back into the same long output, then branch. Any
            # checkpoint beyond this new LCP must be rejected/pruned.
            branch=copy.deepcopy(long_prefix)
            branch.append({'role':'assistant','content':raw[:len(raw)//2],
                           'qw3_raw_completion':raw[:len(raw)//2]})
            tool(branch,'branch','The earlier continuation was discarded. Reply exactly OK.')
            call('earlier_branch',branch,8)
            if enabled:
                try:post('/v1/chat/completions',dict(model='gpt-qw3',messages=[],max_tokens=1,qw3_cache_messages=True,
                         qw3_messages_prefix={'id':'missing-regression-handle','count':1}))
                except urllib.error.HTTPError as exc:
                    error=json.load(exc);assert exc.code==409 and error['error']['code']=='qw3_messages_cache_miss'
                else:raise AssertionError('unknown prefix handle was accepted')
            call('unrelated', [{'role':'user','content':'What is 2 plus 2? Reply with the number only.'}],8)
            text=log_path.read_text()
            target=next(x for x in rows if x['label']=='tail_rewrite')
            segment=text.encode()[target['log_start']:target['log_end']].decode(errors='replace')
            hit=re.search(r'prefix-cache HIT \(mtp\): reuse=(\d+) ckpt=(\w+) D=(\d+).*prompt=(\d+) suffix=(\d+)',segment)
            assert hit, 'tail rewrite fell back to full cold prefill'
            reuse,kind,D,prompt,suffix=hit.groups();assert int(reuse)<=int(D)
            if enabled:
                assert 'prefix-cache REPLAY-R:' in segment,'decode checkpoint was not reused'
                assert int(reuse)%32==0 and int(suffix)<2048,(reuse,D,suffix)
                for row in rows:
                    chunk=text.encode()[row['log_start']:row['log_end']].decode(errors='replace')
                    for match in re.finditer(r'REPLAY-R: R=(\d+) D=(\d+)',chunk):
                        assert int(match[1])<=int(match[2]) and int(match[1])%32==0
            report=dict(ok=True,scored_trial=False,arm=arm,rows=rows,
                        tail_replay=dict(reuse=int(reuse),D=int(D),suffix=int(suffix)),
                        binary=str(args.binary.resolve()))
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            return report
        finally:
            proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--binary',type=pathlib.Path,required=True)
    ap.add_argument('--settings',type=pathlib.Path,required=True)
    ap.add_argument('--command-json',type=pathlib.Path,required=True)
    ap.add_argument('--out',type=pathlib.Path,required=True)
    ap.add_argument('--arm',choices=['baseline','optimized','both'],default='both')
    ap.add_argument('--history-tokens',type=int,default=132000)
    ap.add_argument('--max-tokens',type=int,default=4096)
    args=ap.parse_args(); reports=[]
    for arm in (['baseline','optimized'] if args.arm=='both' else [args.arm]):reports.append(run(args,arm))
    if len(reports)==2:
        comparisons=[]
        for a,b in zip(reports[0]['rows'],reports[1]['rows']):
            exact=a['response']['choices']==b['response']['choices']
            # Initial generation is identical until any checkpoint is restored.
            if a['label'] in ('prepare','long_decode'):assert exact,(a['label'],'output changed before replay')
            comparisons.append(dict(label=a['label'],baseline_ttft=a['timing']['server_ttft_sec'],
                optimized_ttft=b['timing']['server_ttft_sec'],response_equal=exact,
                prompt_tokens_equal=a['usage']['prompt_tokens']==b['usage']['prompt_tokens']))
        (args.out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
if __name__=='__main__':main()
