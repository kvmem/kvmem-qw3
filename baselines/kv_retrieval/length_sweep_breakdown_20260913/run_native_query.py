"""Measure a real appended query on the frozen native server, including refresh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import urllib.error

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=BASE.parents[1]

def save(path,d):
    path.with_suffix('.tmp').write_text(json.dumps(d,indent=2)+'\n')
    path.with_suffix('.tmp').replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['qw3','kvmem'],required=True)
    p.add_argument('--mtp',type=int,choices=[0,4],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--lengths',nargs='+',type=int,default=[8192,16384,32768,65536,131072,245760])
    p.add_argument('--reps',type=int,default=3);p.add_argument('--port',type=int,default=18540)
    p.add_argument('--trace',action='store_true');p.add_argument('--guided-query',action='store_true');a=p.parse_args()
    assert not a.guided_query or (a.method=='kvmem' and a.mtp==4 and min(a.lengths)>=131072)
    trace_target=a.lengths[-1] if a.trace else None
    if a.trace:
        assert a.reps==1 and len(a.lengths)==1
        # Reproduce the formal ladder's checkpoint/cache retention state, but
        # capture only its requested last query. Earlier calls are untimed here.
        a.lengths=[n for n in [8192,16384,32768,65536,131072,245760] if n<=trace_target]
    assert not a.out.exists();a.out.mkdir(parents=True)
    if a.reps > 1:
        # Repeat from a fresh server, so a prior repetition cannot cache the new
        # query itself. Model/history preparation remains outside query timing.
        aggregate=dict(status='running',started=time.time(),method=a.method,mtp=a.mtp,
                       results=[],preparations=[],repetition_sources=[],
                       repetition_policy='fresh server per repetition; same ordered history lengths and same appended query')
        save(a.out/'result.json',aggregate)
        try:
            for rep in range(a.reps):
                target=a.out/f'rep_{rep}'
                cmd=[sys.executable,str(Path(__file__).resolve()),'--method',a.method,'--mtp',str(a.mtp),
                     '--out',str(target),'--lengths',*[str(n) for n in a.lengths],
                     '--reps','1','--port',str(a.port)]
                if a.guided_query:cmd.append('--guided-query')
                subprocess.run(cmd,check=True)
                data=json.loads((target/'result.json').read_text())
                assert data['status']=='complete'
                for key in ['results','preparations']:
                    for row in data[key]:
                        row['rep']=rep;aggregate[key].append(row)
                aggregate['repetition_sources'].append(str(target/'result.json'))
                save(a.out/'result.json',aggregate)
            aggregate['status']='complete'
        except BaseException as e:
            aggregate.update(status='error',error=repr(e));raise
        finally:
            aggregate['ended']=time.time();save(a.out/'result.json',aggregate)
        return
    assert json.loads((HERE/'pause.json').read_text())['status']=='paused'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    os.sched_setaffinity(0,set(range(16))|set(range(32,48)))
    plan=json.loads((ROOT/'benchmark/osworld_v2/claudecode_remaining_20260912/plan.json').read_text())
    arm=plan['arms'][a.method];cmd=list(arm['command'])
    for flag in ['--vision-model','--vision-device']:
        i=cmd.index(flag);del cmd[i:i+2]
    for flag,value in [('--port',str(a.port)),('--temp','0'),('-n','128'),('--mtp-chain',str(a.mtp))]:
        cmd[cmd.index(flag)+1]=value
    # The CLI rejects its automatic guided-refresh option with MTP=0. The raw
    # completions main grid uses the explicit user query (not that chat-harness
    # option) in both arms. Full explicit-query replay remains enabled.
    if a.method=='kvmem' and a.mtp==0:cmd[cmd.index('--kvmem-guided-reselect')+1]='off'
    env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy') and not k.startswith('QW3_')}
    env.update(arm['environment'])
    if a.mtp==0:env['QW3_KVMEM_GUIDED_MTP']='0'
    data=json.loads((BASE/'decode_comparison_20260912/inputs.json').read_text())
    source=data['evaluation']['input_ids'];s=data['suffix'];old_query=s['before']+s['query']+s['after']
    new_query=json.loads((HERE/'new_query.json').read_text());query=new_query['input_ids']
    report=dict(status='starting',started=time.time(),method=a.method,mtp=a.mtp,command=cmd,results=[],preparations=[],
        guided_query_requested=a.guided_query,
        trace_target_history_tokens=trace_target,
        trace_prelude_lengths=a.lengths[:-1] if a.trace else [],
        binary_sha256=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest(),environment=arm['environment'],
        effective_environment={k:v for k,v in env.items() if k.startswith('QW3_')},
        new_query_tokens=len(query),new_query_ids=query,
        timing='Effective throughput = 128 outputs / query request wall, including new-query prefill, selection/load/replay and decode; initial history preparation separate.')
    save(a.out/'result.json',report)
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}));base=f'http://127.0.0.1:{a.port}'
    nvtx=None
    if a.trace:
        import torch
        nvtx=torch.cuda.nvtx
    def request(ids,span,max_tokens,trace=False,measured_query=False):
        body=dict(model='qw3',prompt=ids,max_tokens=max_tokens,temperature=0,ignore_eos=True,stream=False,seed=0)
        if a.method=='kvmem':body['kvmem_query_token_span']=dict(begin=span[0],end=span[1])
        if a.guided_query and measured_query:
            body.update(kvmem_query_guided_thinking_max_tokens=0,kvmem_query_guided_query_max_tokens=256,
                        kvmem_query_guided_direct=True)
        req=urllib.request.Request(base+'/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        offset=(a.out/'server.log').stat().st_size
        if trace:nvtx.range_push('measured_query')
        tick=time.perf_counter()
        try:
            with http.open(req,timeout=1200) as response:result=json.load(response)
        except urllib.error.HTTPError as error:
            detail=error.read().decode(errors='replace')
            save(a.out/'http_error.json',dict(status=error.code,body=detail))
            raise RuntimeError(f'HTTP {error.code}: {detail}') from error
        finally:
            elapsed=time.perf_counter()-tick
            if trace:nvtx.range_pop()
        with (a.out/'server.log').open('rb') as f:f.seek(offset);lines=f.read().decode(errors='replace').splitlines()
        phases={};timing={};mtp={}
        for line in lines:
            nums={k:float(v) for k,v in re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)=(-?\d+(?:\.\d+)?)',line)}
            if '[qw3-native-accounting]' in line:phases.update(nums)
            elif 'native generate: prompt_tokens=' in line or 'native continuous_batch: request=' in line:
                phases.update(nums);phases.setdefault('prefill_ms',nums.get('prefill',0)*1000)
                phases.setdefault('decode_ms',nums.get('decode',0)*1000)
            elif '[qw3-server-ttft]' in line:timing=nums
            elif 'native mtp_spec_summary:' in line:mtp=nums
        assert result['usage']['completion_tokens']==max_tokens
        return dict(request_s=elapsed,usage=result['usage'],phases=phases,server_timing=timing,mtp_stats=mtp,
            output_text=result['choices'][0]['text'],
            output_text_sha256=hashlib.sha256(result['choices'][0]['text'].encode()).hexdigest(),
            input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            semantic_select_log_count=sum('mode=select' in line or 'semantic-chunk' in line for line in lines),
            query_replay_log_count=sum(any(s in line for s in ['native kvmem query replay:',
                'native kvmem query replay (plain):','native kvmem query-replay (plain): span=']) for line in lines),
            checkpoint_fallback_logs=[line for line in lines if any(k in line for k in ['cold_prefill','no_reusable_checkpoint','checkpoint_after_query_boundary'])])
    server=None
    try:
        with (a.out/'server.log').open('x') as log:
            server=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,start_new_session=True)
            report['server_pid']=server.pid;save(a.out/'result.json',report)
            for _ in range(240):
                if server.poll() is not None:raise RuntimeError('Server exited')
                try:
                    with http.open(base+'/health',timeout=1):break
                except OSError:time.sleep(1)
            else:raise TimeoutError('Server readiness')
            # Exercise decode/MTP before preparing the measured histories. This
            # excludes first-use kernel/graph setup without caching the new query.
            warm=source[:1024-len(old_query)]+old_query
            report['warmup']=request(warm,(1024-len(s['query'])-len(s['after']),1024-len(s['after'])),128)
            save(a.out/'result.json',report)
            for length in a.lengths:
                history=source[:length-len(old_query)]+old_query
                old_span=(length-len(s['query'])-len(s['after']),length-len(s['after']))
                report['status']=f'prepare_{length}';save(a.out/'result.json',report)
                initial=request(history,old_span,1)
                initial.update(history_tokens=length);report['preparations'].append(initial);save(a.out/'result.json',report)
                for rep in range(a.reps):
                    tracing=a.trace and length==trace_target
                    row=request(history+query,(length+new_query['query_begin_offset'],length+new_query['query_end_offset']),128,tracing,True)
                    t=row['server_timing'];decode=(t['request_total_ms']-t['server_ttft_ms'])/1000
                    row.update(history_tokens=length,prompt_tokens=length+len(query),rep=rep,mtp=a.mtp,
                        new_query_tokens=len(query),outputs=128,effective_tokens_per_s=128/row['request_s'],
                        query_ttft_s=t['server_ttft_ms']/1000,decode_s=decode,decode_tokens_per_s=127/decode,instrumented=tracing)
                    reused=row['phases'].get('reused_tokens',row['phases'].get('kvmem_reuse',0))
                    assert reused <= length, ('new query unexpectedly reused',reused,length)
                    row['new_query_not_cached']=True
                    report['results'].append(row);save(a.out/'result.json',report)
                    print(json.dumps({k:row[k] for k in ['history_tokens','rep','mtp','effective_tokens_per_s','decode_tokens_per_s','query_ttft_s','semantic_select_log_count','query_replay_log_count']}),flush=True)
        report['status']='complete'
    except BaseException as e:
        report.update(status='error',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        report['ended']=time.time();save(a.out/'result.json',report)

if __name__=='__main__':main()
