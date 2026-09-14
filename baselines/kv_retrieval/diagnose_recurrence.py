"""Factorial teacher-forcing check of optimized recurrence and gated RMSNorm.

No autonomous generation: every variant consumes the same native token trace.
Run only as correctness investigation; shared-GPU timing is not a benchmark.
"""
import argparse
import gc
import json
from pathlib import Path

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

from run_model_smoke import load_text_model, save
from kv_baselines.infllm_adapter import InfLLMConfig
from kv_baselines.infllm_hybrid import install, make_cache, forward_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    free, total = torch.cuda.mem_get_info()
    if free < 57 * 2**30:
        raise RuntimeError("need 57 GiB free for this bounded 55-GiB check")
    torch.cuda.set_per_process_memory_fraction(55 * 2**30 / total)
    events = [json.loads(s) for s in args.trace.read_text().splitlines()]
    prompt = next(e["tokens"] for e in events if e["event"] == "prompt")
    prefill = [e for e in events if e.get("phase") == "prefill"]
    native_steps = [prefill[-1]] + [e for e in events if e.get("phase") == "decode"]
    authoritative = [e["argmax_token"] for e in native_steps]
    report = {"correctness_only": True, "performance_claim": False, "shared_gpu": True,
        "status": "loading", "authoritative_native_tokens": authoritative, "variants": {}}
    save(args.out, report)
    with torch.inference_mode():
        model, report["model"] = load_text_model(args.model.resolve())
        install(model)
        linear = [m for m in model.modules() if isinstance(m, modeling.Qwen3_5GatedDeltaNet)]
        original_norms = [m.norm for m in linear]
        reference_norms = []
        for module in linear:
            norm = modeling.Qwen3_5RMSNormGated(module.head_v_dim, eps=module.layer_norm_epsilon)
            norm.weight = module.norm.weight
            reference_norms.append(norm.to("cuda"))
        report["status"] = "checking"
        for fast_core, fast_norm in [(False, False), (True, False), (False, True), (True, True)]:
            name = f"core_{'fla' if fast_core else 'torch'}_norm_{'fla' if fast_norm else 'torch'}"
            for module, fused_norm, reference_norm in zip(linear, original_norms, reference_norms):
                module.norm = fused_norm if fast_norm else reference_norm
                module.chunk_gated_delta_rule = (modeling.chunk_gated_delta_rule if fast_core
                    else modeling.torch_chunk_gated_delta_rule)
                module.recurrent_gated_delta_rule = (modeling.fused_recurrent_gated_delta_rule if fast_core
                    else modeling.torch_recurrent_gated_delta_rule)
            cache = make_cache(model.config, InfLLMConfig(n_local=512, n_init=16,
                block_size=32, topk=4, max_cached_block=8, exc_block_size=128))
            rows = []
            for step, expected in enumerate(native_steps):
                tokens = prompt if step == 0 else [authoritative[step - 1]]
                ids = torch.tensor([tokens], device="cuda", dtype=torch.long)
                logits = forward_text(model, ids, cache, logits_to_keep=1).logits[0, -1].float()
                scores, token_ids = logits.topk(5)
                wanted = [t["id"] for t in expected["top"]]
                rows.append({"step": step, "native_argmax": expected["argmax_token"],
                    "argmax": token_ids[0].item(), "top5_ids": token_ids.cpu().tolist(),
                    "top5_scores": scores.cpu().tolist(), "native_top_ids": wanted,
                    "scores_for_native_top_ids": logits[wanted].cpu().tolist(),
                    "native_top_scores": [t["logit"] for t in expected["top"]]})
            matches = sum(row["argmax"] == row["native_argmax"] for row in rows)
            report["variants"][name] = {"argmax_matches": matches, "positions": len(rows), "rows": rows}
            save(args.out, report)
            print(json.dumps({"variant": name, "matches": matches, "positions": len(rows)}), flush=True)
            del cache, logits
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "complete"
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        save(args.out, report)


if __name__ == "__main__":
    main()
