"""Thread tuning, exact-output oracle diagnostics, and separate profiles."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from types import SimpleNamespace
from probe import Harness, HERE, PRIOR, torch, previous


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['infinigen','infllm'], default='infinigen')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--trace-mtp', type=int, choices=[0,4])
    parser.add_argument('--trace-mode', default='full')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    assert not args.out.exists()
    assert json.loads((HERE/'pause.json').read_text())['status'] == 'paused'
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    os.sched_setaffinity(0, set(range(16)) | set(range(32,48)))
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); torch.manual_seed(0)
    previous.enable_conv()
    config = SimpleNamespace(method=args.method, window='small', tokens=128, graphs=True)
    report = {'status':'loading', 'started':time.time(), 'method':args.method,
        'tuning':[], 'measurements':[], 'profiles':[], 'captures':[],
        'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
        'timing':'127 useful outputs / full decode wall; reset/prefill excluded; oracle variants diagnostic only'}
    save(args.out, report)
    h = None
    reference_rows = json.loads((PRIOR/'formal'/f'{args.method}.json').read_text())['results']
    expected = {r['prompt_tokens']:r['output_ids'] for r in reference_rows if r['mtp']==0}

    def threads(n):
        torch.set_num_threads(n)
        if h.backend:
            h.backend.executor.submit(torch.set_num_threads, n).result()
            actual = h.backend.executor.submit(torch.get_num_threads).result()
            assert actual == n, (n, actual)
        else:
            actual = None
        return actual

    def measure(length, mtp, n, mode='full', outputs=128, capture=False, profile=False, rep=0):
        actual = threads(n)
        row = h.measure_mode(length,mtp,outputs,mode,capture,profile)
        row.update(threads=n, worker_threads=actual, rep=rep,
            exact_output_vs_prior_fresh_prefill=row['output_ids']==expected[length][:outputs])
        assert row['exact_output_vs_prior_fresh_prefill'], (length,mtp,mode,n,'output mismatch')
        print(json.dumps({k:row[k] for k in ('prompt_tokens','mtp','mode','threads','rep','decode_tokens_per_s','capture','instrumented','exact_output_vs_prior_fresh_prefill')}),flush=True)
        return row

    try:
        h = Harness(config)
        report.update(model=h.manifest, mtp_mapping=h.predictor.mapping)
        if args.trace_mtp is not None:
            h.prepare(32768)
            measure(32768,args.trace_mtp,args.threads,args.trace_mode,outputs=32)
            report['worker_native_tid'] = h.backend.executor.submit(__import__('threading').get_native_id).result()
            gc_events=[]
            gc_starts=[]
            def gc_event(phase, info):
                if phase=='start':
                    gc_starts.append((time.perf_counter(),info['generation']))
                    torch.cuda.nvtx.range_push('python_gc_gen'+str(info['generation']))
                else:
                    start,generation=gc_starts.pop()
                    gc_events.append(dict(generation=generation,host_s=time.perf_counter()-start,
                        collected=info.get('collected'),uncollectable=info.get('uncollectable')))
                    torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            gc.callbacks.append(gc_event)
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push('measured_decode_with_initial_reset')
            row = measure(32768,args.trace_mtp,args.threads,args.trace_mode,outputs=32)
            torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            gc.callbacks.remove(gc_event)
            report['python_gc_events']=gc_events
            report['measurements'].append(row)
            report['status']='complete'; save(args.out,report)
            return
        if args.smoke:
            h.prepare(8192)
            for mtp in (0,4):
                report['captures'].append(measure(8192,mtp,2,capture=True))
                for mode in ('full','oracle_query','oracle_ids','gpu_source','oracle_ids_gpu_source',
                             'optimized_q_only','optimized_bookkeeping','optimized_both'):
                    report['measurements'].append(measure(8192,mtp,2,mode))
                    save(args.out,report)
                report['profiles'].append(measure(8192,mtp,2,profile=True))
            report['status']='complete'; save(args.out,report)
            return
        chosen = {0:args.threads,4:args.threads}
        lengths = (32768,245760)
        for length in lengths:
            report['status']=f'preparing_{length}';save(args.out,report)
            h.prepare(length)
            if length==32768 and args.method=='infinigen':
                for rep,order in enumerate(((2,4,8),(8,4,2))):
                    for n in order:
                        for mtp in ((0,4) if rep==0 else (4,0)):
                            report['tuning'].append(measure(length,mtp,n,outputs=64,rep=rep))
                            save(args.out,report)
                for mtp in (0,4):
                    medians={n:statistics.median(r['decode_tokens_per_s'] for r in report['tuning']
                        if r['threads']==n and r['mtp']==mtp) for n in (2,4,8)}
                    chosen[mtp]=max(medians,key=medians.get)
                report['chosen_threads']=chosen;save(args.out,report)
            for mtp in (0,4):
                n=chosen[mtp]
                if args.method=='infinigen':
                    report['captures'].append(measure(length,mtp,n,capture=True))
                    modes=['full','oracle_query','oracle_ids','gpu_source','oracle_ids_gpu_source',
                           'optimized_q_only','optimized_bookkeeping','optimized_both']
                else:
                    modes=['full']
                for rep in range(3):
                    order=modes[rep:]+modes[:rep]
                    for mode in order:
                        report['status']=f'measuring_{length}_mtp{mtp}_{mode}_rep{rep}'
                        save(args.out,report)
                        report['measurements'].append(measure(length,mtp,n,mode,rep=rep))
                        save(args.out,report)
                report['profiles'].append(measure(length,mtp,n,profile=True))
                if args.method=='infinigen':
                    report['profiles'].append(measure(length,mtp,n,'optimized_both',profile=True))
                save(args.out,report)
        report['status']='complete'
    except BaseException as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if h:
            h.dispose()
            if h.backend:
                h.backend.close()
        report['ended']=time.time();save(args.out,report)


if __name__=='__main__':
    main()
