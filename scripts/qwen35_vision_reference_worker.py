#!/usr/bin/env python3
"""Diagnostic visual references with identical BF16 input pixels.

This worker does not change the production encoder. FP32 refers to arithmetic
on the same BF16 checkpoint values and input patches, not higher-precision
image preprocessing. Keep its output separate from earlier mixed-input audits.
"""
from pathlib import Path
import importlib.util
import os
import sys

SOURCE = Path(__file__).with_name("qw3_vision_cpu_worker.py")
spec = importlib.util.spec_from_file_location("production_vision_worker", SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
original_load = base._load_runtime


def load(*args, **kwargs):
    torch, Image, vision, processor = original_load(*args, **kwargs)
    if vision is None:
        return torch, Image, vision, processor
    mode = os.environ["QW3_VISION_REFERENCE_MODE"]
    if mode not in ("math_fp32", "math_bf16", "flash_bf16"):
        raise ValueError(mode)
    torch.backends.cuda.enable_flash_sdp(mode == "flash_bf16")
    torch.backends.cuda.enable_math_sdp(mode != "flash_bf16")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if mode == "math_fp32":
        vision.float()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    print(f"reference_mode={mode} input_rounding=bf16 arithmetic_dtype="
          f"{next(vision.parameters()).dtype}", file=sys.stderr)
    return torch, Image, vision, processor


def encode(torch, Image, vision, processor, request):
    # Reuse the production decoding and preprocessing exactly. Only promote
    # already-rounded patches when the arithmetic reference uses FP32.
    header, raw = base._preprocess(torch, Image, processor, request)
    parameter = next(vision.parameters())
    patches = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
    patches = patches.reshape(header["rows"], header["dim"])
    patches = patches.to(device=parameter.device, dtype=parameter.dtype)
    grid = torch.tensor(header["grids"], dtype=torch.long, device=parameter.device)
    with torch.inference_mode():
        output = vision(patches, grid_thw=grid).pooler_output
    output = output.float().contiguous().cpu()
    if output.shape[0] != sum(header["counts"]):
        raise RuntimeError("reference output rows do not match input grids")
    payload = output.numpy().tobytes(order="C")
    return dict(ok=True, rows=int(output.shape[0]), dim=int(output.shape[1]),
                grids=header["grids"], counts=header["counts"], dtype="f32",
                bytes=len(payload)), payload


base._load_runtime = load
base._encode = encode
if __name__ == "__main__":
    raise SystemExit(base.main())
