#!/usr/bin/env python3
"""256K physical-pool regression: append screenshots from 100K through 160K.

Owns a local temporary server; stops it before comparing selected warm outputs
with cold starts. Also checks cancellation during prefill and decode, capacity
rejection, and immediate reuse after the cancelled request has drained.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--qw3', default='./build/qw3')
    ap.add_argument('--model', default='models/Qwen3.8-27B-Q8_0.gguf')
    ap.add_argument('--vision-model', default='models/Qwen3.8-27B-BF16')
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=18081)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    base = f'http://127.0.0.1:{args.port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    env = {k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    env.update(QW3_PREFIX_CACHE_TRACE='1', QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES='1',
               QW3_PREFIX_CACHE_MAX_ENTRIES='4', QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES='16384',
               QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES='16384')
    command = [args.qw3, 'serve', '--model', args.model, '--vision-model', args.vision_model,
               '--vision-device', 'cuda', '--host','127.0.0.1', '--port',str(args.port),
               '--ctx','262144','--kv-dtype','fp16','--prefill-chunk','2048','--mtp-chain','4',
               '--continuous-batching','--max-active','1','--temp','0','--thinking-budget','0','-n','16384']
    proc = None; log_handle = None
    def request(path, body=None, timeout=600, headers=None):
        req = urllib.request.Request(base+path, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json', **(headers or {})})
        with opener.open(req, timeout=timeout) as r: return json.load(r)
    def stop():
        nonlocal proc, log_handle
        if proc:
            proc.terminate()
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            proc = None
        if log_handle: log_handle.close(); log_handle = None
    def start(label, warm):
        nonlocal proc, log_handle
        stop(); log_handle = (out/f'{label}.log').open('w')
        proc = subprocess.Popen(command+(['--prefix-cache'] if warm else []),
                                env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        for _ in range(180):
            if proc.poll() is not None: raise RuntimeError(f'server died: {label}')
            try:
                request('/health', timeout=1); return
            except Exception: time.sleep(1)
        raise RuntimeError('server startup timeout')
    def result_text(r): return r['choices'][0]['message']
    results=[]; parity=[]; cancellations=[]; cold_cases=[]
    try:
        start('warm',True)
        unit='The recorded application state is unchanged.\n'
        count=request('/tokenize',{'content':unit*1000})['count']
        filler=unit*(100000*1000//count)
        image_paths = sorted(Path(args.image).parent.glob('step_*.png')) or [Path(args.image)]
        messages=[{'role':'system','content':'You are checking a screenshot history. Reply with exactly OK and nothing else.'},
                  {'role':'user','content':filler}]
        crossed=False
        for turn in range(42):
            # Use an actual saved OSWorld screenshot; each new attachment is a
            # distinct observation even when a desktop is visually unchanged.
            data='data:image/png;base64,'+base64.b64encode(image_paths[turn % len(image_paths)].read_bytes()).decode()
            messages.append({'role':'user','content':[{'type':'text','text':f'Observation {turn}. Reply OK.'},
                             {'type':'image_url','image_url':{'url':data}}]})
            payload={'model':'gpt-qw3','messages':messages,'max_tokens':16384,'temperature':0,
                     'enable_thinking':False,'preserve_thinking':True,'seed':0}
            t=time.monotonic(); r=request('/v1/chat/completions',payload)
            row={'turn':turn,'prompt_tokens':r['usage']['prompt_tokens'],'timing':r.get('timing'),
                 'wall_sec':time.monotonic()-t,'message':result_text(r)}
            results.append(row); print(json.dumps(row),flush=True)
            if turn==0 or (not crossed and row['prompt_tokens']>131072) or row['prompt_tokens']>163840:
                cold_cases.append((turn,json.loads(json.dumps(payload)),result_text(r)))
                crossed = crossed or row['prompt_tokens']>131072
            (out/'progress.json').write_text(json.dumps(results,indent=2))
            if row['prompt_tokens']>163840: break
            messages.append({'role':'assistant','content':r['choices'][0]['message'].get('content') or 'OK'})
        assert results[-1]['prompt_tokens']>163840, 'did not cross 160K'
        log=(out/'warm.log').read_text()
        summaries=re.findall(r'native continuous_batch:[^\n]+',log)
        assert len(summaries)==len(results)
        for line in summaries[1:]:
            assert int(re.search(r'reused_tokens=(\d+)',line)[1])>0,line
            assert int(re.search(r'prefill_tokens=(\d+)',line)[1])<6000,line
            assert int(re.search(r'kv_seq_len=(\d+)',line)[1])>0,line
        assert 'prefix_cache evict' in log, 'entry cap did not exercise shared-page eviction'
        assert all(r['timing']['server_ttft_sec']<10 for r in results[1:]), 'warm TTFT regression'
        # A different 170K prompt ensures a cold prefill is in flight at timeout.
        for phase in ['prefill','decode']:
            rid=uuid.uuid4().hex
            body={'model':'gpt-qw3','prompt':('Different cold input.\n'*34000 if phase=='prefill' else 'Count forever: 1, 2, 3,'),
                  'max_tokens':16384,'temperature':0,'ignore_eos':True,'raw_prompt':True}
            finished=[]
            def send():
                try: finished.append(request('/v1/completions',body,timeout=0.8,
                                             headers={'x-qw3-request-id':rid}))
                except Exception as exc: finished.append(type(exc).__name__)
            worker=threading.Thread(target=send);worker.start()
            if phase == 'prefill':
                time.sleep(0.3)
                try:
                    request('/v1/completions',body,timeout=2)
                    raise AssertionError('expected capacity rejection while original is active')
                except urllib.error.HTTPError as exc:
                    assert exc.code == 429, exc.code
            worker.join(timeout=5)
            assert not worker.is_alive()
            assert finished and isinstance(finished[0], str), 'request finished before timeout'
            t=time.monotonic()
            while time.monotonic()-t<20:
                ack=request('/v1/requests/'+rid+'/cancel',{})
                if ack.get('completed'): break
                time.sleep(0.1)
            else: raise AssertionError('cancellation did not drain '+phase)
            elapsed=time.monotonic()-t
            retry=request('/v1/completions',{'model':'gpt-qw3','prompt':'Reply OK','max_tokens':1,'temperature':0})
            cancellations.append({'phase':phase,'client_result':finished,'drain_sec':elapsed,'retry_usage':retry['usage']})
            print(json.dumps(cancellations[-1]),flush=True)
        stop()
        for turn,payload,expected in cold_cases:
            start(f'cold_{turn}',False)
            r=request('/v1/chat/completions',payload)
            parity.append({'turn':turn,'identical':result_text(r)==expected,'cold':result_text(r),'warm':expected})
            assert parity[-1]['identical'],parity[-1]
            print(json.dumps(parity[-1]),flush=True)
        report={'ok':True,'context_tokens':262144,'physical_slots':262144,'mtp':4,
                'turns':results,'parity':parity,'cancellation':cancellations}
        (out/'report.json').write_text(json.dumps(report,indent=2))
    finally: stop()

if __name__=='__main__': main()
