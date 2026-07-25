#!/usr/bin/env python3
"""Export FlashInfer's BF16 RMSNorm+NVFP4 quantizer as a C AOT object."""

import argparse
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from flashinfer.cute_dsl.rmsnorm_fp4quant import RMSNormFP4QuantKernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination for rms_fp4_sm120_0.h and rms_fp4_sm120_0.o",
    )
    parser.add_argument("--hidden-size", type=int, default=5120)
    return parser.parse_args()


def patch_stream_type() -> None:
    call = RMSNormFP4QuantKernel.__call__
    annotations = dict(call.__annotations__)
    annotations["stream"] = cuda.CUstream
    call.__annotations__ = annotations

    wrapped = call.__wrapped__
    wrapped_annotations = dict(wrapped.__annotations__)
    wrapped_annotations["stream"] = cuda.CUstream
    wrapped.__annotations__ = wrapped_annotations


def main() -> None:
    args = parse_args()
    hidden = args.hidden_size
    if hidden < 64 or hidden % 16 != 0:
        raise ValueError("hidden size must be >= 64 and divisible by 16")

    patch_stream_type()
    sym_m = cute.sym_int()
    sym_scales = cute.sym_int()
    kernel = RMSNormFP4QuantKernel(
        dtype=cutlass.BFloat16,
        H=hidden,
        block_size=16,
        output_swizzled=True,
        is_fp16=False,
        sm_version=120,
        scale_format="e4m3",
    )
    x = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_m, hidden),
        stride_order=(1, 0),
        assumed_align=128,
    )
    weight = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (hidden,),
        assumed_align=128,
    )
    packed = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, hidden // 2),
        stride_order=(1, 0),
        assumed_align=128,
    )
    scales = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_scales,),
        assumed_align=128,
    )
    global_scale = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (1,),
        assumed_align=4,
    )
    compiled = cute.compile(
        kernel,
        x,
        weight,
        packed,
        scales,
        global_scale,
        Int32(1),
        Float32(1e-6),
        False,
        cuda.CUstream(0),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(
        str(args.output_dir),
        "rms_fp4_sm120_0",
        "qw3_rms_fp4_sm120_0",
    )
    print(args.output_dir / "rms_fp4_sm120_0.h")
    print(args.output_dir / "rms_fp4_sm120_0.o")


if __name__ == "__main__":
    main()
