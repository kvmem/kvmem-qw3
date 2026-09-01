#!/usr/bin/env python3
"""Persistent CPU-only Qwen3.5 vision encoder for qw3 serve.

The parent process speaks a deliberately small framed protocol:

  request:  one UTF-8 JSON line
  response: one UTF-8 JSON line followed by ``bytes`` raw float32 bytes

Only the vision tower is instantiated and only ``model.visual.*`` tensors are
loaded.  The language model remains owned by qw3 on the GPU.  Keeping this as a
persistent worker avoids importing PyTorch and loading ~0.9 GiB of vision
weights for every image request.
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


def _load_runtime(model_dir: Path, threads: int):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    import torch
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(max(1, min(4, threads // 4)))

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
    vision.eval().to(device="cpu", dtype=torch.bfloat16)
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
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
    pixel_values = processed["pixel_values"].to(dtype=torch.bfloat16, device="cpu")
    grid = processed["image_grid_thw"].to(dtype=torch.long, device="cpu")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()
    threads = max(1, args.threads)

    # Transformers may emit import/config diagnostics on stdout. stdout is the
    # framed binary protocol, so all third-party chatter must stay on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        torch, Image, vision, processor = _load_runtime(args.model, threads)
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
            if request.get("op") != "encode":
                raise ValueError("unsupported operation")
            with contextlib.redirect_stdout(sys.stderr):
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
