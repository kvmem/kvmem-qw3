"""Batch-one decode timing with fixed input IDs, isolated from prefill/setup."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'infinigen/InfiniGen'))
import torch
from alignment.q8_model import load_q8_model
from alignment.fp8_kv import install_fp8_storage
from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import install, make_cache, forward_text
from qwen.backend import Backend, Config
from qwen.optimized import OptimizedBackend
from qwen.full_offload_control import FullOffloadBackend
from qwen.run_throughput import validate_model
from fastpaths import enable_conv, install_fp8_views, install_graph_dispatch, prepare_graphs, release_cache_graphs


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def prompt(data, length):
    suffix = data['suffix']
    extra = suffix['before'] + suffix['query'] + suffix['after']
    history = data['source']['input_ids']
    if length + 512 > len(history):
        raise ValueError('Insufficient real history; do not pad')
    ids = history[:length-len(extra)] + extra
    begin = length-len(suffix['query'])-len(suffix['after'])
    return ids, (begin, begin+len(suffix['query']))


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['dense_gpu', 'dense_cpu', 'infllm', 'infinigen', 'infinigen_dedup'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[8192])
    p.add_argument('--source', choices=['tuning','evaluation'], default='tuning')
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--thread-grid', type=int, nargs='+')
    p.add_argument('--no-fast-conv', action='store_true')
    p.add_argument('--fp8-views', action='store_true')
    p.add_argument('--sync-global', action='store_true')
    p.add_argument('--cached-blocks', type=int, default=32)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--teacher-force', action='store_true')
    p.add_argument('--graphs', action='store_true')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Exclusive GPU required')
    torch.set_num_threads(args.threads)
    if not args.no_fast_conv:
        enable_conv()
    if args.method == 'infllm':
        (install_fp8_views if args.fp8_views else install_fp8_storage)()
    raw = (HERE/'inputs.json').read_bytes(); data=json.loads(raw)
    data['source']=data[args.source]
    report={'status':'loading','args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'input_sha256':hashlib.sha256(raw).hexdigest(),'results':[], 'batch_size':1,
        'timing':'separate prefill; continuous decode loop including greedy sampling; ignore EOS for bounded throughput only',
        'model_accuracy_claim':False,'source_files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),HERE/'fastpaths.py',BASE/'infinigen/InfiniGen/qwen/backend.py',BASE/'infinigen/InfiniGen/qwen/optimized.py']}}
    save(args.out,report)
    backend=None
    try:
        model, manifest=load_q8_model(BASE.parents[1]/'models/Qwen3.8-27B-BF16',BASE.parents[1]/'models/Qwen3.8-27B-Q8_0.gguf')
        report['model']=manifest
        if args.graphs:
            install_graph_dispatch(model)
        if args.method=='infllm':
            install(model)
            cfg=InfLLMConfig(max_cached_block=args.cached_blocks,async_global_stream=not args.sync_global)
        else:
            backend_class={'infinigen_dedup':OptimizedBackend,'dense_cpu':FullOffloadBackend}.get(args.method,Backend)
            backend=backend_class(model)
            rotations=BASE/'infinigen/InfiniGen/qwen/results/exclusive_20260912_v5/repeat_01.rotations.pt'
            backend.rotations={i:t.cuda() for i,t in torch.load(rotations,weights_only=True).items()}
            if args.validate and args.method != 'dense_cpu':
                report['validation']=validate_model(backend,torch.tensor([data['source']['input_ids'][:333]],device='cuda'))
                save(args.out,report)
        # Warmup is its own prefix/run. Formal rows exclude its cost entirely.
        cases=[(args.threads,1024)]+[(threads,length) for threads in (args.thread_grid or [args.threads]) for length in args.lengths]
        for run_index,(threads,length) in enumerate(cases):
            torch.set_num_threads(threads)
            ids, query_span=prompt(data,length)
            n=16 if run_index==0 else args.tokens
            cache=make_cache(model.config,cfg) if args.method=='infllm' else backend.cache(Config(capacity=length+n+8,dense=args.method=='dense_gpu'))
            def forward(tokens):
                return forward_text(model,tokens,cache,logits_to_keep=1).logits if args.method=='infllm' else backend.forward(tokens,cache)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            for pos in range(0,length,1024):
                logits=forward(torch.tensor([ids[pos:pos+1024]],device='cuda'))
            torch.cuda.synchronize();prefill_s=time.perf_counter()-start
            start=time.perf_counter()
            graph_checks=prepare_graphs(model,cache) if args.graphs else []
            graph_setup_s=time.perf_counter()-start
            start=time.perf_counter()
            if backend:backend.finalize(cache)
            torch.cuda.synchronize();finalize_s=time.perf_counter()-start
            generated=[];consumed=[];step_times=[]
            continuation=data['source']['input_ids'][length:length+n]
            torch.cuda.synchronize();start=time.perf_counter()
            for index in range(n):
                step=time.perf_counter()
                token=int(logits[0,-1].argmax())
                generated.append(token)
                if index+1<n:
                    actual=continuation[index] if args.teacher_force else token
                    consumed.append(actual)
                    logits=forward(torch.tensor([[actual]],device='cuda'))
                step_times.append(time.perf_counter()-step)
            torch.cuda.synchronize();decode_s=time.perf_counter()-start
            row={'history_tokens':length,'output_tokens':n,'decode_forwards':n-1,'prefill_s':prefill_s,
                'finalize_s':finalize_s,'decode_s':decode_s,'output_tokens_per_s':n/decode_s,
                'decode_forwards_per_s':(n-1)/decode_s,'host_step_s':step_times,
                'predicted_token_ids':generated,'consumed_token_ids':consumed,
                'mode':'teacher_forced' if args.teacher_force else 'greedy_ignore_eos',
                'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'query_span':query_span, 'threads':threads, 'graph_setup_s':graph_setup_s, 'graph_checks':graph_checks}
            if backend:
                rows=[dict(layer=i,**r) for i in backend.modules for r in cache.layers[i].metrics]
                row['forecasts']=rows
                row['h2d_payload_mib_per_forward']=sum(r['h2d_bytes'] for r in rows)/max(1,n-1)/2**20
            else:
                managers=[x.manager for x in cache.layers if hasattr(x,'manager')]
                row['global_blocks']=[m.num_global_block for m in managers]
                row['gpu_cached_blocks']=[len(m.cached_blocks[0]) for m in managers]
                row['config']=asdict(cfg)
                row['working_kv_dtype']=str(managers[0].local_k.dtype)
                row['gpu_history_cache_dtype']=str(managers[0].cuda_cache.data.dtype)
                del managers
            if run_index==0:report['warmup']=row
            else:report['results'].append(row)
            report['status']='running';save(args.out,report)
            print(json.dumps({'method':args.method,'warmup':run_index==0,**{k:row[k] for k in ('history_tokens','prefill_s','decode_forwards_per_s')}}),flush=True)
            release_cache_graphs(model)
            del cache,logits
            gc.collect();torch.cuda.empty_cache()
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        if backend is not None:backend.close()
        save(args.out,report)


if __name__=='__main__':main()
