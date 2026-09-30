#!/usr/bin/env python3
"""Freeze target distributions from pinned, unmodified SGLang functions.

Extract function ASTs to avoid initializing a serving runtime. The DFlash
builder uses the actual installed FlashInfer renormalization operators on CUDA;
the ordinary sampler uses SGLang's independent PyTorch fallback.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
import flashinfer
from flashinfer.sampling import top_k_renorm_probs, top_p_renorm_probs

SOURCES = {
    "speculative/dflash_utils.py": "dbb3bb6d6a9939d888a7c4e1630080ec3c29a2a4277d1c59e6c5a0f75fd4b4d5",
    "layers/sampler.py": "6e23da69eb7273bff67e60fc7f3ba112ba1d480438ee17a888bd3852c00590c1",
}


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.reference_source / "python/sglang/srt"
    for name, digest in SOURCES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"unrecognized upstream source: {name}")
    namespace = dict(torch=torch, F=F, Any=Any, Optional=Optional,
        _dflash_top_k_renorm_prob=top_k_renorm_probs,
        _dflash_top_p_renorm_prob=top_p_renorm_probs)
    dflash = extract(source / "speculative/dflash_utils.py", "build_dflash_verify_target_probs", namespace)
    ordinary = extract(source / "layers/sampler.py", "top_k_top_p_min_p_sampling_from_probs_torch", namespace)
    generator = torch.Generator().manual_seed(12419)
    rows = [(torch.tensor([.4, .3, .2, .1]).log(), 1., 2, .55)]
    for n in (37, 257, 2049):
        row = torch.randn(n, generator=generator) * 3
        for temperature in (.6, 1., 2.):
            for k, p in ((20, .95), (1, 1.), (n, .55)):
                rows.append((row, temperature, k, p))
    rows.append((torch.randn(248320, generator=generator) * 3, 1., 20, .95))
    cases = []
    for row, temperature, k, p in rows:
        logits = row[None].cuda()
        top_ks = torch.tensor([k], dtype=torch.int32, device="cuda")
        top_ps = torch.tensor([p], device="cuda")
        info = SimpleNamespace(temperatures=torch.tensor([[temperature]], device="cuda"),
            top_ks=top_ks, top_ps=top_ps, need_top_k_sampling=k < row.numel(), need_top_p_sampling=p < 1)
        for order in ("joint", "top-k-first"):
            if order == "top-k-first":
                probabilities = dflash(next_token_logits=logits, sampling_info=info,
                    draft_token_num=1, bs=1, max_top_k=k, uniform_top_k_value=k).flatten()
            else:
                _, weights, ids, _ = ordinary(F.softmax(logits / temperature, dim=-1),
                    top_ks, top_ps, torch.zeros_like(top_ps), False, None,
                    torch.zeros(1, device="cuda", dtype=torch.int64), return_filtered_probs=True)
                probabilities = torch.zeros_like(weights).scatter_(1, ids.long(), weights).flatten()
                probabilities /= probabilities.sum()
            values = probabilities.cpu().tolist()
            cases.append(dict(order=order, temperature=temperature, top_k=k, top_p=p,
                logits=row.tolist(), nonzero=[[i, v] for i, v in enumerate(values) if v > 0]))
    # Nucleus cutoff ties are observable: FlashInfer keeps the entire tied
    # group. Keep top-k untruncated here because top-k tie ordering is specific
    # to the engine's selection kernel.
    for row in (torch.zeros(4), torch.tensor([0., 0., -100., -100.])):
        probabilities = top_p_renorm_probs(F.softmax(row[None].cuda(), dim=-1), .5).flatten().cpu().tolist()
        cases.append(dict(order="top-k-first", temperature=1., top_k=4, top_p=.5,
            logits=row.tolist(), nonzero=[[i, v] for i, v in enumerate(probabilities) if v > 0]))
    payload = dict(reference_revision="9f6fc553322e897a22f30320dae750f4a40a0b84",
        source_sha256=SOURCES, torch=torch.__version__, flashinfer=flashinfer.__version__,
        device=torch.cuda.get_device_name(), cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n")
    print(f"pinned_sglang_target_distributions cases={len(cases)} output={args.output}")


if __name__ == "__main__":
    main()
