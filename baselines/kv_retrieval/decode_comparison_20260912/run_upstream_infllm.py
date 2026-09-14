"""Original InfLLM patch_hf / Mistral, with a bounded decode timing harness."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'upstream/InfLLM'
MODEL = Path('/data/huggingface/kvmem_upstream_decode_20260912/Mistral-7B-Instruct-v0.2.json')
SOURCES = {
    'tuning': HERE.parent / 'results/calibration_trace_termenv_dense_20260912/request_0061.json.gz',
    'evaluation': HERE.parent / 'results/infllm_small_deepswe4_s1000_20260912/superjson-error-stack-serialization/small/attempt_01/server/inputs/e9ee4b6ad3f146469f15637b30781592.json',
}


def save(path, report):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['infllm', 'dense'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lengths', nargs='+', type=int, default=[8192, 32768, 122880])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--thread-grid', nargs='+', type=int)
    parser.add_argument('--sync-global', action='store_true')
    parser.add_argument('--no-pin', action='store_true')
    parser.add_argument('--torch-attention', action='store_true')
    parser.add_argument('--source', choices=list(SOURCES), default='evaluation')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Exclusive GPU required')
    assert not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain', '--untracked-files=no'])
    sys.path.insert(0, str(SOURCE))
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from inf_llm import patch_hf
    from omegaconf import OmegaConf
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    snapshot = json.loads(MODEL.read_text())
    cfg = dict(OmegaConf.load(SOURCE / 'config/mistral-inf-llm.yaml').model)
    cfg.update(fattn=not args.torch_attention, async_global_stream=not args.sync_global, pin_memory=not args.no_pin)
    report = {'status': 'loading', 'started': time.time(), 'qw3_backend': False, 'results': [],
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'source_commit': subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_files_sha256': {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE / 'inf_llm').rglob('*.py')},
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'configuration': cfg, 'model': snapshot, 'weight_dtype': 'BF16', 'kv_dtype': 'BF16', 'batch_size': 1,
              'environment': {'torch': torch.__version__, 'transformers': transformers.__version__, 'python': sys.version,
                              'gpu': torch.cuda.get_device_name(0)},
              'timing': '128 fixed greedy outputs, ignore EOS; prefill produces first output; 127 decode forwards including selection, transfer, sampling and history append; prefill/setup excluded',
              'accuracy_claim': False}
    save(args.out, report)
    raw = SOURCES[args.source].read_bytes()
    request = json.loads(gzip.decompress(raw) if SOURCES[args.source].suffix == '.gz' else raw)
    # Retokenize real chronological conversation content with Mistral's tokenizer.
    # No Qwen token IDs, rendering library, padding or engine is used.
    text = json.dumps({k: request[k] for k in ['system', 'tools', 'messages']}, ensure_ascii=False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot['snapshot'], local_files_only=True)
    all_ids = tokenizer.encode(text, add_special_tokens=True)
    suffix = tokenizer.encode('\nExplain the next implementation and verification steps in detail.\n', add_special_tokens=False)
    report['input'] = {'path': str(SOURCES[args.source]), 'raw_sha256': hashlib.sha256(raw).hexdigest(),
                       'serialization': 'JSON system/tools/messages in original order, Mistral tokenization', 'available_tokens': len(all_ids)}
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(snapshot['snapshot'], torch_dtype=torch.bfloat16,
                                                device_map='cuda', local_files_only=True, attn_implementation='sdpa').eval()
    if args.method == 'infllm':
        model = patch_hf(model, cfg['type'], **cfg)
    report.update(load_s=time.perf_counter() - load_start, status='running')
    save(args.out, report)

    @torch.inference_mode()
    def measure(length, outputs):
        assert length > len(suffix) and len(all_ids) >= length
        ids = all_ids[:length-len(suffix)] + suffix
        input_ids = torch.tensor([ids], dtype=torch.long, device='cuda')
        attention_mask = torch.ones_like(input_ids)
        past = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        # The author's GreedySearch uses chunked prefill and a final one-token
        # forward. Keep that convention for both methods.
        for st in range(0, length - 1, 8192):
            ed = min(length - 1, st + 8192)
            out = model(input_ids=input_ids[:, st:ed], attention_mask=attention_mask[:, :ed],
                        past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            del out
        out = model(input_ids=input_ids[:, -1:], attention_mask=attention_mask, past_key_values=past,
                    use_cache=True, return_dict=True)
        past = out.past_key_values
        word = out.logits[:, -1, :].argmax(dim=-1)
        output_ids = [word.item()]
        del out
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start
        input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long, device='cuda')), dim=-1)
        load_before = [x.load_count for x in past] if args.method == 'infllm' else None
        started = time.perf_counter()
        steps = []
        for _ in range(outputs - 1):
            tick = time.perf_counter()
            out = model(input_ids=input_ids[:, -1:], attention_mask=attention_mask, past_key_values=past,
                        use_cache=True, return_dict=True)
            past = out.past_key_values
            word = out.logits[:, -1, :].argmax(dim=-1)
            output_ids.append(word.item())  # Same CPU-visible greedy token as original GreedySearch.
            del out
            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=torch.long, device='cuda')), dim=-1)
            steps.append(time.perf_counter() - tick)
        torch.cuda.synchronize()
        decode_s = time.perf_counter() - started
        result = {'prompt_tokens': length, 'generated_tokens': outputs, 'prefill_s': prefill_s,
                  'decode_s': decode_s, 'decode_forwards': outputs - 1, 'decode_tokens_per_s': (outputs - 1) / decode_s,
                  'step_s': steps, 'output_ids': output_ids, 'input_ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                  'torch_peak_gib': torch.cuda.max_memory_allocated() / 2**30, 'threads': torch.get_num_threads()}
        if args.method == 'infllm':
            result.update(cache_type=type(past[0]).__module__ + '.' + type(past[0]).__name__,
                          attention_type=past[0].Attn.__module__ + '.' + past[0].Attn.__name__,
                          lru_clock_before=load_before, lru_clock_after=[x.load_count for x in past],
                          history_blocks_per_layer=[len(x.global_blocks[0]) for x in past],
                          resident_history_blocks_per_layer=[sum(b.gpu_data is not None for b in x.global_blocks[0]) for x in past])
        del past, input_ids, attention_mask, word
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        return result

    try:
        report['warmup'] = measure(1024, 16)
        save(args.out, report)
        for threads in args.thread_grid or [args.threads]:
            torch.set_num_threads(threads)
            for length in args.lengths:
                # The first full-history run warms all retrieval shapes; kept
                # separately rather than mixed into the measured repetitions.
                warmup = measure(length, 16)
                report.setdefault('history_warmups', []).append(warmup)
                save(args.out, report)
                for rep in range(args.reps):
                    result = measure(length, args.tokens)
                    result['rep'] = rep
                    report['results'].append(result)
                    save(args.out, report)
                    print(json.dumps({k: result[k] for k in ['prompt_tokens', 'threads', 'rep', 'decode_tokens_per_s', 'prefill_s']}), flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='error', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['ended'] = time.time()
        save(args.out, report)


if __name__ == '__main__':
    main()
