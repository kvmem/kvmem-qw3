"""Bounded original FlexiCache component checks; never a full baseline result.

Compile mode uses no GPU. Numerical mode requires spare memory and keeps
allocations tiny; the full model, task runner and head profiles are not involved.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def source():
    from kv_baselines.upstream import verify_source
    pin = verify_source('FlexiCache')
    return ROOT / 'upstream/FlexiCache', pin


def load_cuda(name):
    from torch.utils.cpp_extension import load
    upstream, _ = source()
    path = upstream / 'vllm/v1/flexicache/kernels/cuda' / (name + '.cu')
    build = ROOT / 'compat_build/flexicache' / name
    build.mkdir(parents=True, exist_ok=True)
    return load(name='kvbaseline_flexicache_' + name, sources=[str(path)],
        extra_cuda_cflags=['-O3'], extra_cflags=['-O3'],
        build_directory=str(build), verbose=False)


def load_scores():
    upstream, _ = source()
    path = upstream / 'vllm/attention/ops/flexi_cache_triton_kernels.py'
    spec = importlib.util.spec_from_file_location('_kvbaseline_flexicache_kernels', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def score_checks():
    import torch
    module = load_scores()
    gen = torch.Generator().manual_seed(73)
    heads, group, dim, pages, block, mm_block = 4, 6, 256, 8, 32, 128
    query_cpu = torch.randn(1, heads * group, dim, generator=gen).half()
    keys_cpu = torch.randn(heads, pages, block, dim, generator=gen).half()
    low, high = keys_cpu.amin(2), keys_cpu.amax(2)
    mm_cpu = torch.zeros(heads, mm_block, 2, dim, dtype=torch.float16)
    mm_cpu[:, :pages, 0] = low
    mm_cpu[:, :pages, 1] = high
    query = query_cpu.cuda()
    mm = mm_cpu.cuda()
    table = torch.arange(heads, dtype=torch.int32, device='cuda').view(1, heads, 1)
    scores = torch.full((1, heads, pages), -17., dtype=torch.float32, device='cuda')
    lengths = torch.tensor([pages * block - 3], dtype=torch.int32, device='cuda')
    unstable = torch.tensor([1 << 1], dtype=torch.uint32, device='cuda')
    steps = torch.tensor([1], dtype=torch.int32, device='cuda')
    starts = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
    args = [scores, query, mm, table, lengths, unstable, steps, 4,
        scores.stride(0), scores.stride(1), query.stride(0), query.stride(1),
        *mm.stride(), *table.stride()]
    meta = dict(filter_by_query_len=True, query_start_len_ptr=starts,
        queries_per_kv=group, BLOCK_SIZE=block, HEAD_SIZE=dim, HEAD_SIZE_PAD=dim,
        MINMAX_CACHE_BLOCK_SIZE=mm_block, PAGES_PER_TB=16)
    original_error = None
    try:
        module.kernel_compute_block_scores_minmax[(1, heads, 1)](*args, **meta)
        torch.cuda.synchronize()
    except Exception as error:
        original_error = str(error)
    # Author's existing padded implementation handles non-power-of-two GQA.
    padded = module.kernel_compute_block_scores_minmax_qwen32
    meta['queries_per_kv_padded'] = 8
    padded[(1, heads, 1)](*args, **meta)
    torch.cuda.synchronize()
    q = query_cpu.float().reshape(heads, group, dim)
    reference = torch.maximum(q[:, :, None] * low.float()[:, None],
        q[:, :, None] * high.float()[:, None]).sum(-1).amax(1)
    reference[:, -1] = torch.finfo(torch.float32).max
    observed = scores.cpu()[0]
    torch.testing.assert_close(observed, reference, atol=2e-4, rtol=2e-5)
    max_error = (observed[:, :-1] - reference[:, :-1]).abs().max().item()
    # Between reranks only the profiled unstable head may update its scores.
    scores.fill_(-17)
    steps.fill_(2)
    padded[(1, heads, 1)](*args, **meta)
    torch.cuda.synchronize()
    observed = scores.cpu()[0]
    torch.testing.assert_close(observed[1], reference[1], atol=2e-4, rtol=2e-5)
    assert torch.all(observed[[0, 2, 3]] == -17)
    steps.fill_(4)
    padded[(1, heads, 1)](*args, **meta)
    torch.cuda.synchronize()
    torch.testing.assert_close(scores.cpu()[0], reference, atol=2e-4, rtol=2e-5)
    # The original scoring kernel deliberately does not rank multi-token prefill.
    scores.fill_(-17)
    starts[1] = 2
    padded[(1, heads, 1)](*args, **meta)
    torch.cuda.synchronize()
    assert torch.all(scores.cpu() == -17)
    return {'head_size': dim, 'gqa_group': group, 'original_dispatch_error': original_error,
        'padded_original_kernel': 'passed', 'max_absolute_score_error': max_error,
        'stable_head_cadence': 'passed', 'unstable_head_every_step': 'passed',
        'prefill_gate': 'passed', 'is_full_model_baseline': False}


def transfer_checks():
    import torch
    h2d = load_cuda('h2d_copy_blocks')
    d2h = load_cuda('kv_d2h_tx_gpu_mapped')
    tables = load_cuda('block_table_h2d_dirty')
    generator = torch.Generator().manual_seed(73)
    host = torch.randn(2, 32, 32, 256, generator=generator).half().pin_memory()
    device = torch.zeros_like(host, device='cuda')
    src_cpu = torch.tensor([17, 3, 25], dtype=torch.int32)
    dst_cpu = torch.tensor([0, 9, 7], dtype=torch.int32)
    src, dst = src_cpu.cuda(), dst_cpu.cuda()
    main = torch.cuda.current_stream()
    background = torch.cuda.Stream()
    background.wait_stream(main)
    h2d.register_cpu_kv_cache_pinned(host)
    with torch.cuda.stream(background):
        h2d.kv_cache_h2d_transfer(host, device, src, dst, chunk_pairs=2, grid_blocks=2)
    main.wait_stream(background)
    expected = torch.zeros_like(host)
    expected[:, dst_cpu.long()] = host[:, src_cpu.long()]
    assert torch.equal(device.cpu(), expected)
    host_back = torch.zeros_like(host, pin_memory=True)
    background.wait_stream(main)
    with torch.cuda.stream(background):
        d2h.kv_cache_d2h_transfer(device, host_back, dst, src, chunk_pairs=2, grid_blocks=2)
    background.synchronize()
    expected_back = torch.zeros_like(host)
    expected_back[:, src_cpu.long()] = host[:, src_cpu.long()]
    assert torch.equal(host_back, expected_back)
    host_table = torch.arange(2 * 2 * 4 * 16, dtype=torch.int32).reshape(2, 2, 4, 16).pin_memory()
    device_table = torch.full_like(host_table, -1, device='cuda')
    dirty = torch.tensor([[3, 11], [5, 5]], dtype=torch.int32, device='cuda')
    background.wait_stream(main)
    with torch.cuda.stream(background):
        tables.block_table_h2d_dirty(host_table, device_table, dirty, 2, 2, 4, 16, 2)
    main.wait_stream(background)
    expected_table = torch.full_like(host_table, -1)
    expected_table[0, :, :, 3:11] = host_table[0, :, :, 3:11]
    assert torch.equal(device_table.cpu(), expected_table)
    return {'pinned_uva_h2d': 'bitwise passed', 'pinned_uva_d2h': 'bitwise passed',
        'background_stream_ordering': 'passed', 'chunked_transfer': 'passed',
        'dirty_table_untouched_ranges': 'passed', 'head_size': 256,
        'full_model_baseline': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['compile', 'scores', 'transfers'], required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('preserve raw component results')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    upstream, pin = source()
    report = {'started': time.time(), 'source': pin, 'mode': a.mode,
        'status': 'running', 'full_baseline_result': False, 'performance_claim': False,
        'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        a.out.write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        import torch
        if a.mode == 'compile':
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
                raise ValueError('compile mode requires CUDA_VISIBLE_DEVICES empty')
            report['compiled'] = []
            for name in ['block_table_h2d_dirty', 'kv_d2h_tx_gpu_mapped', 'h2d_copy_blocks', 'topk_swap_map']:
                load_cuda(name)
                report['compiled'].append(name)
                assert not torch.cuda.is_initialized(), 'compilation must not initialize GPU'
                save()
                print(json.dumps({'compiled': name}), flush=True)
        else:
            free = float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip()) / 1024
            if free < 2:
                raise RuntimeError('requires at least 2 GiB free for tiny component check')
            torch.cuda.set_per_process_memory_fraction(.25 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
            report['shared_gpu_correctness_only'] = True
            report[a.mode] = score_checks() if a.mode == 'scores' else transfer_checks()
            report['peak_torch_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        report['status'] = 'passed_components_only'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished'] = time.time()
        save()


if __name__ == '__main__':
    main()
