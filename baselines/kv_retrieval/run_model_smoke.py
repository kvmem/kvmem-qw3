"""Load real local BF16 text weights once and validate the InfLLM boundary.

Shared-GPU results are correctness evidence only, never latency measurements.
No server, benchmark harness, shared environment, or existing model is modified.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding

from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import forward_text, install, make_cache
from kv_baselines.upstream import verify_source
from kv_baselines.precision import use_fp32_logits


def load_text_model(directory):
    raw = json.loads((directory / "config.json").read_text())
    if raw.get("quantization_config"):
        raise ValueError("this loader validates unquantized BF16 safetensors only")
    config = Qwen3_5TextConfig(**raw["text_config"])
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    expected = dict(model.named_parameters())
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    mapping = {}
    for src, shard in weight_map.items():
        dst = src.replace("model.language_model.", "model.", 1)
        if src.startswith("model.language_model.") or src == "lm_head.weight":
            if dst not in expected:
                raise ValueError(f"unexpected text weight: {src}")
            mapping[dst] = (src, shard)
    if mapping.keys() != expected.keys():
        raise ValueError(f"missing text weights: {expected.keys() - mapping.keys()}")
    count = 0
    for shard in sorted({v[1] for v in mapping.values()}):
        with safe_open(directory / shard, framework="pt", device="cpu") as reader:
            for dst, (src, filename) in mapping.items():
                if filename != shard:
                    continue
                tensor = reader.get_tensor(src)
                if tuple(tensor.shape) != tuple(expected[dst].shape) or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"unexpected shape/dtype: {src} {tensor.shape} {tensor.dtype}")
                parent, leaf = dst.rsplit(".", 1)
                model.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(
                    tensor.to("cuda"), requires_grad=False)
                count += 1
        print(json.dumps({"loaded_shard": shard, "loaded_parameters": count,
            "gpu_allocated_gib": torch.cuda.memory_allocated() / 2**30}), flush=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device="cuda")
    if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
        raise ValueError("uninitialized meta tensors remain")
    use_fp32_logits(model)
    return model.eval(), {"text_parameters": count, "model_path": str(directory),
        "config_sha256": hashlib.sha256((directory / "config.json").read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256((directory / "model.safetensors.index.json").read_bytes()).hexdigest(),
        "weight_dtype": "bfloat16", "logits_dtype": "float32", "vision_loaded": False, "mtp_loaded": False}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-gpu-gib", type=float, default=55)
    p.add_argument("--shared-gpu", action="store_true")
    p.add_argument("--long-tokens", type=int, default=0)
    args = p.parse_args()
    free, total = torch.cuda.mem_get_info()
    if free < (args.max_gpu_gib + 2) * 2**30:
        raise RuntimeError(f"insufficient free GPU memory for bounded smoke test: {free / 2**30:.1f} GiB")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_gib * 2**30 / total)
    torch.manual_seed(719)
    report = {"status": "loading", "correctness_only": True, "performance_claim": False,
        "shared_gpu": args.shared_gpu, "sources": {"InfLLM": verify_source("InfLLM")},
        "started": time.time(), "max_gpu_gib": args.max_gpu_gib, "torch": torch.__version__}
    save(args.out, report)
    try:
        with torch.inference_mode():
            model, manifest = load_text_model(args.model.resolve())
            report["model"] = manifest
            from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
            report["recurrence_backend"] = (
                modeling.chunk_gated_delta_rule.__module__ if modeling.chunk_gated_delta_rule is not None else "torch_fallback")
            report["status"] = "checking_dense_vs_infllm"
            save(args.out, report)
            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
            prompt = tokenizer.apply_chat_template([
                {"role": "user", "content": "Write a Python function that adds two integers. Explain it briefly."}
            ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            # Small shared-GPU calibration deliberately stays inside local KV.
            cfg = InfLLMConfig(n_local=512, n_init=16, block_size=32, topk=4,
                max_cached_block=8, exc_block_size=128, repr_topk=4,
                async_global_stream=True, pin_memory=True, attention_dtype="float16")
            dense_cache = DynamicCache(config=model.config)
            dense = model(input_ids=ids, past_key_values=dense_cache,
                          use_cache=True, logits_to_keep=1).logits.detach().float().cpu()
            del dense_cache
            gc.collect()
            torch.cuda.empty_cache()
            report["patched_layers"] = install(model)
            cache = make_cache(model.config, cfg)
            adapted = forward_text(model, ids, cache, logits_to_keep=1).logits.detach().float().cpu()
            report["config"] = asdict(cfg)
            report["prompt_tokens"] = ids.shape[-1]
            report["dense_argmax"] = dense.argmax(-1).tolist()
            report["infllm_argmax"] = adapted.argmax(-1).tolist()
            report["logits_max_abs_error"] = (dense - adapted).abs().max().item()
            report["logits_rmse"] = (dense - adapted).square().mean().sqrt().item()
            if not torch.equal(dense.argmax(-1), adapted.argmax(-1)):
                raise AssertionError("short-context next-token identity mismatch")
            token = adapted.argmax(-1).to("cuda")
            generated = []
            for _ in range(12):
                generated.append(token.item())
                result = forward_text(model, token, cache, logits_to_keep=1)
                token = result.logits[:, -1].argmax(-1, keepdim=True)
            report["generated_token_ids"] = generated
            report["generated_text"] = tokenizer.decode(generated)
            report["cache_seq_length"] = cache.get_seq_length()
            report["status"] = "passed_short_context_smoke"
            if args.long_tokens:
                if args.long_tokens < 8192:
                    raise ValueError("long smoke requires at least 8192 tokens to force default-budget retrieval")
                if report["recurrence_backend"] == "torch_fallback":
                    raise ValueError("use the isolated FLA environment for the long model check")
                del cache, result
                gc.collect()
                torch.cuda.empty_cache()
                cfg = InfLLMConfig()
                cache = make_cache(model.config, cfg)
                repo = Path(__file__).resolve().parents[2]
                paths = [repo / "src" / n for n in ["kvmem_gc.cpp", "qwen_config.cpp", "model_source.cpp", "tokenizer.cpp"]]
                content = "\n".join(p.read_text() for p in paths)
                all_ids = tokenizer(content, return_tensors="pt").input_ids
                if all_ids.shape[1] < args.long_tokens:
                    raise ValueError("not enough distinct source text; no repeated padding allowed")
                ids = all_ids[:, :args.long_tokens].to("cuda")
                report["status"] = "checking_long_history"
                report["long_config"] = asdict(cfg)
                report["long_source_sha256"] = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                report["long_chunks_completed"] = 0
                save(args.out, report)
                for start in range(0, args.long_tokens, 512):
                    output = forward_text(model, ids[:, start:start + 512], cache, logits_to_keep=1)
                    if not torch.isfinite(output.logits).all():
                        raise AssertionError("non-finite long-context model logits")
                    report["long_chunks_completed"] += 1
                    report["long_processed_tokens"] = cache.get_seq_length()
                    save(args.out, report)
                    print(json.dumps({"long_processed_tokens": cache.get_seq_length()}), flush=True)
                managers = [layer.manager for layer in cache.layers if hasattr(layer, "manager")]
                report["long_global_blocks_per_layer"] = [m.num_global_block for m in managers]
                report["long_selected_blocks_per_layer"] = [m.global_buffer_block_id_list for m in managers]
                if not all(m.num_global_block > cfg.topk for m in managers):
                    raise AssertionError("long smoke did not force top-k retrieval")
                report["status"] = "passed_short_and_long_model_smoke"
    except BaseException as error:
        import traceback
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished"] = time.time()
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        save(args.out, report)


if __name__ == "__main__":
    main()
