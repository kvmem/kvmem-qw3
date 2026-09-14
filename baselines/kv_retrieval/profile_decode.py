"""Held-out teacher-forced decode controls; profiler runs are diagnostic only.

HF dense is an implementation control (BF16 KV), not a main matched-precision
baseline. Original InfLLM uses FP16 attention KV. All reuse the same BF16 weights
and the same recorded input/continuation; no tool actions or task scores result.
"""
import argparse
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import MethodType

import torch
from transformers import DynamicCache

from kv_baselines.frontend import NativeBridge
from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import forward_text, install, make_cache, uninstall
from kv_baselines.memory import storage_inventory
from run_model_smoke import load_text_model, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--infllm-config", type=Path, required=True)
    p.add_argument("--request-index", type=int, default=8)
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--profile-tokens", type=int, default=16)
    p.add_argument("--shared-gpu", action="store_true")
    p.add_argument("--max-gpu-gib", type=float, default=58)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True)
    if owners.strip() and not a.shared_gpu:
        raise RuntimeError("exclusive GPU required without --shared-gpu")
    free, total = torch.cuda.mem_get_info()
    if free < (a.max_gpu_gib + 2) * 2**30:
        raise RuntimeError("insufficient spare memory")
    torch.cuda.set_per_process_memory_fraction(a.max_gpu_gib * 2**30 / total)
    manifest = json.loads((a.trace / "manifest.json").read_text())
    bridge = NativeBridge(a.model)
    inputs = []
    for row in manifest["requests"][a.request_index:a.request_index + 2]:
        raw = gzip.decompress((a.trace / row["path"]).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == row["body_sha256"]
        inputs.append(bridge.call("render", request=json.loads(raw))["input_ids"])
    bridge.close()
    prompt, following = inputs
    if following[:len(prompt)] != prompt:
        raise ValueError("chosen held-out continuation must be an exact token append")
    continuation = following[len(prompt):len(prompt) + a.decode_tokens + a.profile_tokens]
    if len(continuation) != a.decode_tokens + a.profile_tokens:
        raise ValueError("not enough recorded continuation; no synthetic token padding")
    report = {"status": "loading", "shared_gpu": a.shared_gpu, "performance_claim": False,
        "task_success_claim": False, "initial_gpu_owners": owners,
        "trace_manifest_sha256": hashlib.sha256((a.trace / "manifest.json").read_bytes()).hexdigest(),
        "prompt_tokens": len(prompt), "request_index": a.request_index,
        "continuation_ids": continuation, "methods": [],
        "dense_control_kv_dtype": "bfloat16", "InfLLM_kv_dtype": "float16",
        "profiler_measurements": "intrusive diagnostic; not end-to-end performance"}
    save(a.out / "result.json", report)
    try:
        model, provenance = load_text_model(a.model)
        report["model"] = provenance
        from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
        report["causal_convolution"] = "torch_fallback" if modeling.causal_conv1d_fn is None else modeling.causal_conv1d_fn.__module__
        report["recurrence"] = modeling.chunk_gated_delta_rule.__module__
        for name, config in [("hf_dense_control", None), ("infllm_author_defaults", InfLLMConfig()),
                             ("infllm_budget_candidate", InfLLMConfig(**json.loads(a.infllm_config.read_text())))]:
            if config:
                install(model)
            cache = make_cache(model.config, config) if config else DynamicCache(config=model.config)
            row = {"method": name, "config": asdict(config) if config else None}
            with torch.inference_mode():
                start = time.perf_counter()
                for pos in range(0, len(prompt), 512):
                    output = forward_text(model, torch.tensor([prompt[pos:pos + 512]], device="cuda"), cache, logits_to_keep=1)
                torch.cuda.synchronize()
                row["prefill_wall_s"] = time.perf_counter() - start
                row["next_token_argmax"] = output.logits[:, -1].argmax().item()
                steps = []
                for token in continuation[:a.decode_tokens]:
                    start = time.perf_counter()
                    output = forward_text(model, torch.tensor([[token]], device="cuda"), cache, logits_to_keep=1)
                    torch.cuda.synchronize()
                    steps.append(time.perf_counter() - start)
                row["decode_step_wall_s"] = steps
                row["state_storage"] = storage_inventory({"live_cache": cache})
                # Label whole components only for the separate profiler pass.
                patched = []
                for module in model.modules():
                    class_name = type(module).__name__
                    if class_name not in {"Qwen3_5GatedDeltaNet", "Qwen3_5Attention", "Qwen3_5MLP"}:
                        continue
                    original = module.forward
                    def labelled(self, *args, _forward=original, _label=class_name, **kwargs):
                        with torch.profiler.record_function("baseline/component/" + _label):
                            return _forward(*args, **kwargs)
                    module.forward = MethodType(labelled, module)
                    patched.append((module, original))
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for token in continuation[a.decode_tokens:]:
                            output = forward_text(model, torch.tensor([[token]], device="cuda"), cache, logits_to_keep=1)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(str(a.out / (name + ".chrome.json")))
                    operators = [{"name": event.key, "count": event.count,
                        "cpu_total_us": event.cpu_time_total, "self_cpu_us": event.self_cpu_time_total,
                        "device_total_us": event.device_time_total, "self_device_us": event.self_device_time_total}
                        for event in prof.key_averages()]
                    save(a.out / (name + ".operators.json"), operators)
                finally:
                    for module, original in patched:
                        module.forward = original
            row["process_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            report["methods"].append(row)
            report["status"] = "running"
            save(a.out / "result.json", report)
            del cache, output
            if config:
                uninstall(model)
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "completed_diagnostic_controls"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        save(a.out / "result.json", report)


if __name__ == "__main__":
    main()
