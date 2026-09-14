"""Confirm the tuning result with 128 outputs, paired thread counts and repeats."""
from pathlib import Path
import os
import json
import subprocess
import time
import sys
import numpy as np
from types import SimpleNamespace
from probe import Harness, HERE, PRIOR, previous, torch


@torch.inference_mode()
def main():
    out=HERE/'thread_confirmation.json'
    assert not out.exists()
    assert json.loads((HERE/'pause.json').read_text())['status']=='paused'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    os.sched_setaffinity(0,set(range(16))|set(range(32,48)))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
    previous.enable_conv()
    prior=json.loads((PRIOR/'formal/infinigen.json').read_text())['results']
    expected={r['prompt_tokens']:r['output_ids'] for r in prior if r['mtp']==0}
    chosen=json.loads((HERE/'formal_infinigen.json').read_text())['chosen_threads']
    state={'status':'loading','started':time.time(),'results':[],'chosen_threads':chosen,'historical_row_reuse':[],
        'gil_diagnostic':[],'default_gil_switch_s':sys.getswitchinterval()}
    previous.save(out,state)
    h=Harness(SimpleNamespace(method='infinigen',window='small',tokens=128,graphs=True))
    try:
        for length in (32768,245760):
            state['status']=f'preparing_{length}';previous.save(out,state)
            h.prepare(length)
            for rep in range(3):
                for mtp in ((0,4) if rep%2==0 else (4,0)):
                    order=list(dict.fromkeys((2,chosen[str(mtp)])))
                    if rep%2:order.reverse()
                    for n in order:
                        torch.set_num_threads(n)
                        h.backend.executor.submit(torch.set_num_threads,n).result()
                        actual=h.backend.executor.submit(torch.get_num_threads).result()
                        assert actual==n
                        for mode in (('full','optimized_both') if rep%2==0 else ('optimized_both','full')):
                            row=h.measure_mode(length,mtp,128,mode)
                            assert row['output_ids']==expected[length]
                            row.update(threads=n,worker_threads=actual,rep=rep,exact_output_vs_prior=True)
                            state['results'].append(row);previous.save(out,state)
                            print(json.dumps({k:row[k] for k in ('prompt_tokens','mtp','mode','threads','rep','decode_tokens_per_s')}),flush=True)
            # Separate untimed-in-analysis capture; only immutable prompt KV
            # is eligible, so the estimate never reuses a rewritten draft tail.
            capture=h.measure_mode(length,0,128,capture=True)
            assert capture['output_ids']==expected[length]
            selected_total=reused_total=0; layers=[]
            for idx, records in h.controller.tapes[(length,0)].items():
                s=h.prepared[1].layers[idx]
                mapped=np.arange(s.qh)//(s.qh//s.kh)
                last=None;selected=reused=0
                for record in records:
                    ids=record['ids'][:,0].numpy()
                    unique=np.unique((ids*s.kh+mapped).reshape(-1))
                    unique=unique[unique<length*s.kh]
                    if last is not None:
                        selected+=len(unique)
                        reused+=len(np.intersect1d(unique,last,assume_unique=True))
                    last=unique
                selected_total+=selected;reused_total+=reused
                layers.append({'layer':idx,'selected_immutable_rows_after_first':selected,
                    'also_selected_previous_token':reused,'reuse_fraction':reused/selected if selected else None})
            state['historical_row_reuse'].append({'prompt_tokens':length,'mtp':0,'output_tokens':128,
                'selected_immutable_rows_after_first':selected_total,'also_selected_previous_token':reused_total,
                'reuse_fraction':reused_total/selected_total,'layers':layers,
                'scope':'Opportunity for a last-selected-row GPU cache; not a measured cache implementation or a throughput claim.'})
            previous.save(out,state)
            if length == 32768:
                # MTP runs showed substantial variation without output changes.
                # Isolate Python thread handoff; do not change target arithmetic.
                n=chosen['4']
                torch.set_num_threads(n)
                h.backend.executor.submit(torch.set_num_threads,n).result()
                state['status']='gil_handoff_diagnostic';previous.save(out,state)
                original_interval=sys.getswitchinterval()
                try:
                    intervals=[original_interval,0.001,0.0001]
                    for rep in range(3):
                        for interval in intervals[rep:]+intervals[:rep]:
                            sys.setswitchinterval(interval)
                            for mode in (('full','optimized_both') if rep%2==0 else ('optimized_both','full')):
                                row=h.measure_mode(length,4,128,mode)
                                assert row['output_ids']==expected[length]
                                row.update(threads=n,rep=rep,gil_switch_s=interval,exact_output_vs_prior=True)
                                state['gil_diagnostic'].append(row);previous.save(out,state)
                                print(json.dumps({k:row[k] for k in ('prompt_tokens','mtp','mode','threads','rep','gil_switch_s','decode_tokens_per_s')}),flush=True)
                finally:
                    sys.setswitchinterval(original_interval)
        state['status']='complete'
    except BaseException as exc:
        state.update(status='error',error=repr(exc));raise
    finally:
        h.dispose();h.backend.close();state['ended']=time.time();previous.save(out,state)


if __name__=='__main__':main()
