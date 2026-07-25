#!/usr/bin/env python3
"""Export Qw3's SM120 GDN prefill kernel as a C-compatible AOT object."""

import argparse
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache import (
    _in_mem_compile_cache,
)
from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import (
    _FullyFusedDeltaRuleSm120,
)
from flashinfer.gdn_kernels.delta_rule_dsl.schedule import WorkDesc
from flashinfer.gdn_prefill import chunk_gated_delta_rule


def qw3_get_next_work(
    self,
    cu_seqlens: cute.Tensor,
    num_q_heads: cutlass.Int32,
    num_v_heads: cutlass.Int32,
    num_sab_heads: cutlass.Int32,
) -> WorkDesc:
    """Use Qw3's interleaved grouped-head convention."""
    bx, _, _ = cute.arch.block_idx()
    seq_idx = bx // num_sab_heads
    output_head = bx % num_sab_heads
    token_start = cutlass.Int32(cu_seqlens[seq_idx])
    token_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
    return WorkDesc(
        seq_idx=seq_idx,
        private_q_head_idx=output_head % num_q_heads,
        private_v_head_idx=output_head % num_v_heads,
        tok_offset=token_start,
        seq_len=token_end - token_start,
        tile_idx=cutlass.Int32(0),
    )


def patch_aot_types() -> None:
    _FullyFusedDeltaRuleSm120.get_next_work = qw3_get_next_work
    annotations = dict(_FullyFusedDeltaRuleSm120.__call__.__annotations__)
    annotations["grid_x"] = cutlass.Int32
    annotations["stream"] = cuda.CUstream
    _FullyFusedDeltaRuleSm120.__call__.__annotations__ = annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for gdn_sm120_0.h and gdn_sm120_0.o",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        raise RuntimeError(f"SM120 GPU required, found sm_{major}{minor}")

    patch_aot_types()
    _in_mem_compile_cache.clear()

    tokens, q_heads, v_heads, head_dim = 64, 16, 48, 128
    q = torch.randn(
        tokens, q_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn(
        tokens, v_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    decay = torch.rand(tokens, v_heads, device="cuda", dtype=torch.float32)
    beta = torch.rand_like(decay)
    state = torch.zeros(
        1, v_heads, head_dim, head_dim, device="cuda", dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int64)

    chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=decay,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_cp=False,
    )
    torch.cuda.synchronize()

    if len(_in_mem_compile_cache) != 1:
        raise RuntimeError(
            f"expected one compiled GDN object, got {len(_in_mem_compile_cache)}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiled = next(iter(_in_mem_compile_cache.values()))
    compiled.export_to_c(
        str(args.output_dir), "gdn_sm120_0", "qw3_gdn_sm120_0"
    )
    print(args.output_dir / "gdn_sm120_0.h")
    print(args.output_dir / "gdn_sm120_0.o")


if __name__ == "__main__":
    main()
