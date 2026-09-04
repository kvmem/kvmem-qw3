#!/usr/bin/env python3
"""Persistent Qwen3.5 vision worker for qw3 serve.

The parent process speaks a deliberately small framed protocol:

  request:  one UTF-8 JSON line
  response: one UTF-8 JSON line followed by ``bytes`` raw float32 bytes

In the CPU frontend, only the vision tower and ``model.visual.*`` tensors are
loaded. In native-CUDA preprocess-only mode, no learned weights are loaded and
the worker returns exact BF16 flattened patches; qw3 owns all visual-tower
weights and learned computation on the GPU. Keeping the worker persistent
avoids repeated PyTorch/image-processor startup.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path


def _load_runtime(
    model_dir: Path, threads: int, preprocess_only: bool = False, device: str = "cpu"
):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads // 4)))

    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    if preprocess_only:
        return torch, Image, None, processor

    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    vision = Qwen3_5VisionModel(config.vision_config)

    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"missing model.safetensors.index.json under {model_dir}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    visual_names = sorted(name for name in weight_map if name.startswith("model.visual."))
    if not visual_names:
        raise RuntimeError("checkpoint contains no model.visual.* tensors")

    by_shard: dict[str, list[str]] = {}
    for name in visual_names:
        by_shard.setdefault(weight_map[name], []).append(name)
    state = {}
    for shard, names in by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
            for name in names:
                state[name.removeprefix("model.visual.")] = handle.get_tensor(name)
    missing, unexpected = vision.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"vision checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    del state
    vision.eval().to(device=device, dtype=torch.bfloat16)
    return torch, Image, vision, processor


def _decode_image(Image, item: dict):
    encoded = item.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image.data must be non-empty base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError(f"invalid image base64: {exc}") from exc
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image.convert("RGB")


def _encode(torch, Image, vision, processor, request: dict):
    items = request.get("images")
    if not isinstance(items, list) or not items:
        raise ValueError("encode requires a non-empty images array")
    images = [_decode_image(Image, item) for item in items]
    processed = processor(images=images, return_tensors="pt")
    device = next(vision.parameters()).device
    pixel_values = processed["pixel_values"].to(dtype=torch.bfloat16, device=device)
    grid = processed["image_grid_thw"].to(dtype=torch.long, device=device)
    with torch.inference_mode():
        output = vision(pixel_values, grid_thw=grid).pooler_output
    embedding = output.float().contiguous().cpu()
    grids = [[int(v) for v in row] for row in grid.tolist()]
    counts = [(t * h * w) // (vision.config.spatial_merge_size**2) for t, h, w in grids]
    if sum(counts) != embedding.shape[0]:
        raise RuntimeError(
            f"vision row mismatch: grid rows={sum(counts)} output={embedding.shape[0]}"
        )
    raw = embedding.numpy().tobytes(order="C")
    return {
        "ok": True,
        "rows": int(embedding.shape[0]),
        "dim": int(embedding.shape[1]),
        "grids": grids,
        "counts": counts,
        "dtype": "f32",
        "bytes": len(raw),
    }, raw


def _preprocess(torch, Image, processor, request: dict):
    items = request.get("images")
    if not isinstance(items, list) or not items:
        raise ValueError("preprocess requires a non-empty images array")
    images = [_decode_image(Image, item) for item in items]
    processed = processor(images=images, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(dtype=torch.bfloat16, device="cpu").contiguous()
    grid = processed["image_grid_thw"].to(dtype=torch.long, device="cpu")
    if pixel_values.ndim != 2 or pixel_values.shape[1] != 1536:
        raise RuntimeError(f"unexpected flattened patch shape: {tuple(pixel_values.shape)}")
    grids = [[int(v) for v in row] for row in grid.tolist()]
    patch_counts = [t * h * w for t, h, w in grids]
    output_counts = [count // 4 for count in patch_counts]
    if sum(patch_counts) != pixel_values.shape[0]:
        raise RuntimeError(
            f"vision patch mismatch: grid patches={sum(patch_counts)} "
            f"output={pixel_values.shape[0]}"
        )
    # NumPy has no universally supported bfloat16 dtype. Reinterpret the
    # contiguous PyTorch storage as uint16 so the parent receives the exact
    # BF16 payload expected by the native CUDA frontend.
    raw = pixel_values.view(torch.uint16).numpy().tobytes(order="C")
    return {
        "ok": True,
        "rows": int(pixel_values.shape[0]),
        "dim": int(pixel_values.shape[1]),
        "grids": grids,
        "counts": output_counts,
        "patch_counts": patch_counts,
        "dtype": "bf16",
        "bytes": len(raw),
    }, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    threads = max(1, args.threads)

    # Transformers may emit import/config diagnostics on stdout. stdout is the
    # framed binary protocol, so all third-party chatter must stay on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        torch, Image, vision, processor = _load_runtime(
            args.model, threads, preprocess_only=args.preprocess_only,
            device=args.device)
    print(json.dumps({"ok": True, "event": "ready"}, separators=(",", ":")), flush=True)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    for line in stdin:
        if not line.strip():
            continue
        raw = b""
        try:
            request = json.loads(line)
            if request.get("op") == "shutdown":
                break
            if request.get("op") not in ("encode", "preprocess"):
                raise ValueError("unsupported operation")
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("op") == "preprocess":
                    if not args.preprocess_only:
                        raise ValueError("preprocess requires --preprocess-only worker")
                    response, raw = _preprocess(torch, Image, processor, request)
                else:
                    if vision is None:
                        raise ValueError("encode is unavailable in preprocess-only worker")
                    response, raw = _encode(
                        torch, Image, vision, processor, request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "bytes": 0}
            traceback.print_exc(file=sys.stderr)
        stdout.write(json.dumps(response, separators=(",", ":")).encode() + b"\n")
        if raw:
            stdout.write(raw)
        stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
