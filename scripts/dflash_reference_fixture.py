#!/usr/bin/env python3
"""Generate small full-network oracles using the pinned official DFlash code.

Use z-lab/dflash commit 07ebd93db9f472af339b644bb70221ad8428328a,
dflash/model.py. The fixture is independent of the native executor and includes
the real convolution/attention/selector topology, small FP32 weights, and both
short and window-overflow prefixes. It does not measure model quality or speed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight-dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()
    source_sha256 = hashlib.sha256(args.official_source.read_bytes()).hexdigest()
    if source_sha256 != "f55b7fe0a4c0b3073e0f9cdce547cce29f4b8e2168c4d2818760007c43b7651e":
        raise ValueError("official source does not match the pinned reference revision")
    torch.set_num_threads(4)
    torch.manual_seed(7841)
    spec = importlib.util.spec_from_file_location("dflash_official_fixture", args.official_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.Qwen3Config(
        hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=97, max_position_embeddings=256, num_target_layers=8,
        is_causal=False, attention_bias=False, attention_dropout=0,
        use_sliding_window=True, sliding_window=16, max_window_layers=2,
        layer_types=["sliding_attention"] * 2, tie_word_embeddings=False,
        rms_norm_eps=1e-6, hidden_act="silu",
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
        dflash_config={"block_size": 8, "mask_token_id": 96,
                       "conv_group_size": 8, "conv_kernel_size": 2,
                       "selector_rank": 16, "selector_top_k": 4,
                       "target_layer_ids": [1, 5]},
        architectures=["DFlash2DraftModel"],
    )
    model = module.DFlash2DraftModel(cfg).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1 + torch.randn_like(p) * 0.03)
            elif "base_kernel" in name:
                p.copy_(torch.randn_like(p) * 0.02)
                p[:, 0, :].add_(1)
            else:
                p.copy_(torch.randn_like(p) * 0.07)
        if args.weight_dtype == "bf16":
            for p in model.parameters():
                p.copy_(p.bfloat16().float())
            # Native BF16 GEMM rounds the activation before its FP32
            # accumulation/output. Preserve the official model topology and
            # emulate just that storage boundary in the CPU reference.
            for layer in model.modules():
                if isinstance(layer, torch.nn.Linear):
                    layer.register_forward_pre_hook(
                        lambda _layer, inputs: (inputs[0].bfloat16().float(),))
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(root)
    weights = {}
    for name, value in model.state_dict().items():
        if "codebook.weight" in name:
            name = name.removesuffix(".weight")
        weights[name] = (value.bfloat16() if args.weight_dtype == "bf16" else value).contiguous()
    save_file(weights, root / "model.safetensors")
    head = torch.nn.Linear(64, 97, bias=False).eval()
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight) * 0.1)

    def write(name, tensor):
        tensor.detach().float().contiguous().numpy().tofile(root / name)
        return name

    write("lm_head.f32", head.weight)
    cases = []
    with torch.no_grad():
        for i, (prefix, block, base) in enumerate([
            (0, 8, 0), (5, 8, 0), (16, 8, 0), (35, 8, 0),
            (35, 3, 0), (9, 1, 0), (5, 3, 64),
        ]):
            features = torch.randn(1, prefix, 128)
            noise = torch.randn(1, block, 64)
            output = model(
                target_hidden=features, noise_embedding=noise,
                position_ids=torch.arange(base, base + prefix + block)[None],
                use_cache=False,
            )
            anchor = (i * 13) % 97
            tokens = []
            if block > 1:
                tokens = model.propose(output[:, 1:], torch.tensor([anchor]), head, 0)[0][0].tolist()
            cases.append({
                "prefix": prefix, "block": block, "base": base, "anchor": anchor,
                "features": write(f"{i}_features.f32", features),
                "noise": write(f"{i}_noise.f32", noise),
                "output": write(f"{i}_output.f32", output), "greedy_tokens": tokens,
            })
    manifest = {
        "source_commit": "07ebd93db9f472af339b644bb70221ad8428328a",
        "source_sha256": source_sha256,
        "torch": torch.__version__, "seed": 7841, "cases": cases,
        "weight_dtype": args.weight_dtype,
        "activation_precision": "FP32 residuals; BF16 GEMM inputs" if args.weight_dtype == "bf16" else "FP32",
        "purpose": "complete draft topology and bounded-cache numerical oracle; not performance",
    }
    (root / "fixture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"fixture": str(root), "cases": len(cases), "source_sha256": manifest["source_sha256"]}))


if __name__ == "__main__":
    main()
