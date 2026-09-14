"""Matched BF16 Qwen model; original-method adapters, no qw3 compute library."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
BASE = PARENT.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(PARENT), str(BASE), str(BASE/'infinigen/InfiniGen')]
import torch
from transformers import AutoTokenizer
from run_model_smoke import load_text_model
from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import install, make_cache, forward_text
from qwen.backend import Backend, Config
from qwen.optimized import OptimizedBackend
from qwen.full_offload_control import FullOffloadBackend
from qwen.run_throughput import validate_model
from fastpaths import (enable_conv, install_fp8_views, install_graph_dispatch,
                       prepare_graphs, release_cache_graphs)


def save(path, report):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2)+'\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['infllm', 'infinigen', 'dense_gpu', 'dense_cpu'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', nargs='+', type=int, default=[8192, 32768, 122880, 245760])
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--thread-grid', nargs='+', type=int)
    p.add_argument('--source', choices=['tuning', 'evaluation'], default='evaluation')
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--graphs', action='store_true')
    p.add_argument('--no-history-warmup', action='store_true')
    p.add_argument('--async-global', action='store_true')
    p.add_argument('--infllm-window', choices=['small', '128k'], default='small')
    p.add_argument('--infinigen-kv-budget', type=int, default=2048)
    p.add_argument('--calibration', type=Path, default=HERE/'bf16_rotations.pt')
    p.add_argument('--calibrate-only', action='store_true')
    p.add_argument('--validate', action='store_true')
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    assert a.tokens >= 2 and a.reps >= 1
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    enable_conv()
    if a.method == 'infllm':
        install_fp8_views()
    inputs = json.loads((PARENT/'inputs.json').read_text())
    history = inputs[a.source]['input_ids']
    suffix = inputs['suffix']
    extra = suffix['before'] + suffix['query'] + suffix['after']
    report = {'status': 'loading', 'started': time.time(), 'results': [],
              'args': {k:str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
              'cpu_affinity': sorted(os.sched_getaffinity(0)), 'batch_size': 1,
              'qw3_compute_backend': False, 'weight_dtype': 'BF16', 'kv_storage': 'FP8 E4M3FN',
              'residual_dtype': 'BF16 HF/FLA (native qw3 residual differs)',
              'input_file_sha256': digest(PARENT/'inputs.json'),
              'timing': '127 full decode forwards for 128 outputs, first greedy token before timer; includes retrieval, transfers, sampling and state updates; load/calibration/prefill/initial selection excluded',
              'accuracy_claim': False, 'source_sha256': {str(f):digest(f) for f in
               [Path(__file__), BASE/'run_model_smoke.py', PARENT/'fastpaths.py',
                BASE/'infinigen/InfiniGen/qwen/backend.py', BASE/'infinigen/InfiniGen/qwen/optimized.py',
                BASE/'infinigen/InfiniGen/qwen/full_offload_control.py', BASE/'kv_baselines/infllm_adapter.py',
                BASE/'kv_baselines/infllm_hybrid.py']}}
    save(a.out, report)
    backend = None
    try:
        tick = time.perf_counter()
        model, report['model'] = load_text_model(ROOT/'models/Qwen3.8-27B-BF16')
        report['load_s'] = time.perf_counter()-tick
        if a.graphs:
            install_graph_dispatch(model)
        if a.method == 'infllm':
            install(model)
            topk = 16 if a.infllm_window == 'small' else 991
            cfg = InfLLMConfig(topk=topk, max_cached_block=max(32,topk),
                               async_global_stream=a.async_global)
            report['infllm_configuration'] = asdict(cfg)
        else:
            cls = {'infinigen':OptimizedBackend, 'dense_cpu':FullOffloadBackend}.get(a.method,Backend)
            backend = cls(model)
            if a.method == 'infinigen':
                pg19 = BASE/'infinigen/InfiniGen/accuracy/setup/pg19_firstbook.txt'
                report['calibration_source'] = {'file':str(pg19), 'sha256':digest(pg19), 'tokens':2048}
                if not a.calibration.exists():
                    report['status'] = 'calibrating'; save(a.out,report)
                    tokenizer = AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.8-27B-BF16',local_files_only=True)
                    ids = tokenizer.encode(pg19.read_text(),add_special_tokens=False)[:2048]
                    calibration = backend.cache(Config(capacity=2048,dense=True,collect_calibration=True))
                    tick = time.perf_counter()
                    backend.forward(torch.tensor([ids],device='cuda'),calibration)
                    backend.calibrate(calibration)
                    torch.cuda.synchronize()
                    report['calibration_s'] = time.perf_counter()-tick
                    torch.save({i:t.cpu() for i,t in backend.rotations.items()},a.calibration)
                    del calibration,tokenizer
                    gc.collect(); torch.cuda.empty_cache()
                    save(a.out,report)
                else:
                    backend.rotations = {i:t.cuda() for i,t in torch.load(a.calibration,weights_only=True).items()}
                report['calibration_sha256'] = digest(a.calibration)
                assert set(backend.rotations) == set(backend.modules)
                if a.validate:
                    report['validation'] = validate_model(backend,torch.tensor([history[:333]],device='cuda'))
                    save(a.out,report)
        if a.calibrate_only:
            assert a.method=='infinigen' and report.get('validation',{}).get('status')=='passed'
            report['status']='complete'
            return

        def measure(length, outputs):
            assert len(extra) < length <= len(history)
            ids = history[:length-len(extra)] + extra
            cache = make_cache(model.config,cfg) if a.method=='infllm' else backend.cache(
                Config(capacity=length+outputs+8,max_num_kv=a.infinigen_kv_budget,dense=a.method=='dense_gpu'))
            def forward(tokens):
                return forward_text(model,tokens,cache,logits_to_keep=1).logits if a.method=='infllm' else backend.forward(tokens,cache)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            for pos in range(0,length,a.chunk):
                logits = forward(torch.tensor([ids[pos:pos+a.chunk]],device='cuda'))
            torch.cuda.synchronize(); prefill_s=time.perf_counter()-tick
            tick=time.perf_counter()
            checks=prepare_graphs(model,cache) if a.graphs else []
            graph_s=time.perf_counter()-tick
            tick=time.perf_counter()
            if backend: backend.finalize(cache)
            torch.cuda.synchronize(); finalize_s=time.perf_counter()-tick
            token=int(logits[0,-1].argmax())
            generated=[token]; steps=[]
            torch.cuda.synchronize(); tick=time.perf_counter()
            for _ in range(outputs-1):
                step=time.perf_counter()
                logits=forward(torch.tensor([[token]],device='cuda'))
                token=int(logits[0,-1].argmax())
                generated.append(token)
                steps.append(time.perf_counter()-step)
            torch.cuda.synchronize(); decode_s=time.perf_counter()-tick
            row={'prompt_tokens':length,'generated_tokens':outputs,'decode_forwards':outputs-1,
                 'decode_s':decode_s,'decode_tokens_per_s':(outputs-1)/decode_s,'step_s':steps,
                 'prefill_s':prefill_s,'initial_offload_s':finalize_s,'graph_setup_s':graph_s,
                 'graph_checks':checks,'output_ids':generated,'threads':torch.get_num_threads(),
                 'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                 'peak_torch_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                 'peak_torch_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
            if backend:
                stats=[dict(layer=i,**s) for i in backend.modules for s in cache.layers[i].metrics]
                row['forecasts']=stats
                row['h2d_mib_per_forward']=sum(s['h2d_bytes'] for s in stats)/(outputs-1)/2**20
                if stats:
                    row['mean_selected_tokens_per_head']=sum(s['selected_tokens_per_head'] for s in stats)/len(stats)
            else:
                managers=[x.manager for x in cache.layers if hasattr(x,'manager')]
                row['history_blocks_per_layer']=[m.num_global_block for m in managers]
                row['resident_blocks_per_layer']=[len(m.cached_blocks[0]) for m in managers]
                row['history_cache_dtype']=str(managers[0].cuda_cache.data.dtype)
                del managers
            release_cache_graphs(model)
            del cache,logits
            gc.collect();torch.cuda.empty_cache()
            return row

        report['warmup']=measure(1024,16);save(a.out,report)
        for threads in a.thread_grid or [a.threads]:
            torch.set_num_threads(threads)
            for length in a.lengths:
                if not a.no_history_warmup:
                    report.setdefault('history_warmups',[]).append(measure(length,16));save(a.out,report)
                for rep in range(a.reps):
                    report['status']=f'measuring_{length}_threads{threads}_rep{rep}';save(a.out,report)
                    row=measure(length,a.tokens);row['rep']=rep
                    report['results'].append(row);save(a.out,report)
                    print(json.dumps({k:row[k] for k in ['prompt_tokens','threads','rep','decode_tokens_per_s','prefill_s']}),flush=True)
        assert 'libaligned_q8.so' not in Path('/proc/self/maps').read_text()
        report['no_qw3_projection_library_loaded']=True
        report['status']='complete'
    except BaseException as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if backend: backend.close()
        report['ended']=time.time();save(a.out,report)


if __name__=='__main__':
    main()
