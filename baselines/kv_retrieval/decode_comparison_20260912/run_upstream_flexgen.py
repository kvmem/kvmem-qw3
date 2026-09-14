"""Measure the author's untouched FlexGen entry point; no qw3 dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'upstream/InfiniGen'


def setup_package(scheme):
    target = HERE / 'native_packages' / scheme / 'flexgen'
    target.mkdir(parents=True, exist_ok=True)
    root = SOURCE / 'speedup/flexgen'
    files = {p.name: p for p in (root / 'flexgen').glob('*.py')}
    for name in ['flex_opt.py', 'pytorch_backend.py']:
        files[name] = root / scheme / name
    for name, source in files.items():
        link = target / name
        if not link.is_symlink():
            link.symlink_to(source.resolve())
        assert link.resolve() == source.resolve()
    sys.path.insert(0, str(target.parent))
    sys.path.insert(0, str(SOURCE / 'speedup/infinigen'))
    return {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in files.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheme', choices=['original', 'infinigen'], required=True)
    parser.add_argument('--kv', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--overlap', choices=['false', 'true'], default='false')
    parser.add_argument('--prompt', type=int, default=1920)
    parser.add_argument('--gen', type=int, default=128)
    parser.add_argument('--decode-warmup', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--weights', default='/data/huggingface/kvmem_upstream_decode_20260912/opt_weights')
    args = parser.parse_args()
    assert args.prompt + args.gen <= 2048
    assert args.scheme != 'infinigen' or args.kv == 'cpu'
    assert args.scheme != 'infinigen' or args.overlap == 'false', 'Author InfiniGen prefetch is only in normal loop'
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'running', 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
              'files_sha256': setup_package(args.scheme), 'started': time.time(), 'qw3_backend': False,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(record, indent=2))
    import torch
    import transformers
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    import flexgen.flex_opt as upstream
    record['environment'] = {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
                             'gpu': torch.cuda.get_device_name(0), 'weight_dtype': 'FP16', 'kv_dtype': 'FP16'}
    p = argparse.ArgumentParser()
    upstream.add_parser_arguments(p)
    pg19 = SOURCE / 'speedup/flexgen/pg19_firstbook.txt'
    argv = ['--model', 'huggingface/opt-6.7b', '--path', args.weights,
            '--offload-dir', str(HERE / 'native_offload' / args.out.stem),
            '--percent', '100', '0', *(['0', '100'] if args.kv == 'cpu' else ['100', '0']), '100', '0',
            '--overlap', args.overlap, '--gpu-batch-size', str(args.batch), '--num-gpu-batches', '1',
            '--prompt-len', str(args.prompt), '--gen-len', str(args.gen),
            '--warmup-input-path', str(pg19), '--test-input-path', str(pg19), '--verbose', '0', '--no-log']
    if args.scheme == 'infinigen':
        argv += ['--alpha', '4', '--partial-weight-ratio', '0.2', '--max-num-kv', '400']
    record['upstream_argv'] = argv
    record['input_sha256'] = hashlib.sha256(pg19.read_bytes()).hexdigest()
    # Only observe the two whole generate calls (author calibration and test).
    original = upstream.OptLM.generate
    calls = []

    def observe_generate(self, *a, **kw):
        if calls and args.decode_warmup:
            from flexgen.timer import timers
            warm_kwargs = dict(kw, max_new_tokens=args.decode_warmup)
            torch.cuda.synchronize()
            warm_start = time.perf_counter()
            warm_result = original(self, *a, **warm_kwargs)
            torch.cuda.synchronize()
            calls.append({'role': 'decode_warmup', 'wall_s': time.perf_counter() - warm_start,
                          'output_shape': list(warm_result.shape), 'output_ids': warm_result.tolist()})
            timers('generate').reset()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(self, *a, **kw)
        torch.cuda.synchronize()
        calls.append({'wall_s': time.perf_counter() - start, 'warmup': bool(kw.get('warmup', False)),
                      'output_shape': list(result.shape), 'output_ids': result.tolist()})
        return result

    upstream.OptLM.generate = observe_generate
    try:
        upstream.run_flexgen(p.parse_args(argv))
        from flexgen.timer import timers
        costs = list(timers('generate').costs)
        assert len(costs) == args.gen and all(t > 0 for t in costs)
        record.update(status='complete', costs_s=costs, prefill_s=costs[0], decode_s=sum(costs[1:]),
                      decode_forwards=args.batch * (args.gen - 1),
                      decode_tokens_per_s=args.batch * (args.gen - 1) / sum(costs[1:]),
                      total_tokens_per_s=args.batch * args.gen / sum(costs),
                      torch_peak_gib=torch.cuda.max_memory_allocated() / 2**30)
    except BaseException as exc:
        record.update(status='error', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(calls=calls, ended=time.time())
        args.out.write_text(json.dumps(record, indent=2))
        print(json.dumps({k: record[k] for k in ['status', 'decode_tokens_per_s', 'prefill_s', 'decode_s'] if k in record}), flush=True)


if __name__ == '__main__':
    main()
