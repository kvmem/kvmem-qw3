#!/usr/bin/env python3
"""Regression for BF16 weights with FP32 visual rotary buffers; no GPU required."""
import importlib.util
import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vision_worker", root / "scripts/qw3_vision_cpu_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    torch.manual_seed(23)
    config = Qwen3_5VisionConfig(
        depth=1, hidden_size=72, intermediate_size=80, num_heads=1,
        out_hidden_size=64, num_position_embeddings=16, patch_size=2,
        spatial_merge_size=2, temporal_patch_size=2,
    )
    original = Qwen3_5VisionModel(config).eval()
    state = {name: value.to(torch.bfloat16) for name, value in original.state_dict().items()}
    with tempfile.TemporaryDirectory(prefix="qw3-vision-loader-") as directory:
        path = Path(directory)
        Qwen3_5Config(vision_config=config.to_dict()).save_pretrained(path)
        names = {"model.visual." + name: value for name, value in state.items()}
        save_file(names, path / "model.safetensors")
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model.safetensors" for name in names},
        }))
        (path / "preprocessor_config.json").write_text(json.dumps({
            "image_processor_type": "Qwen2VLImageProcessorFast",
            "patch_size": 2, "temporal_patch_size": 2, "merge_size": 2,
        }))
        _, _, actual, _ = worker._load_runtime(path, threads=2)
        expected = Qwen3_5VisionModel.from_pretrained(
            None, config=config, state_dict=state, dtype=torch.bfloat16,
        ).eval()
        assert all(p.dtype == torch.bfloat16 for p in actual.parameters())
        assert actual.rotary_pos_emb.inv_freq.dtype == torch.float32
        torch.testing.assert_close(actual.rotary_pos_emb.inv_freq,
                                   expected.rotary_pos_emb.inv_freq, rtol=0, atol=0)
        # High patch coordinates expose frequency/angle rounding that small
        # uniform images can hide. Compare the actual worker's full forward.
        grid = torch.tensor([[1, 4, 120]])
        patches = torch.randn(480, 24, dtype=torch.bfloat16)
        with torch.inference_mode():
            a = actual(patches, grid_thw=grid).pooler_output
            b = expected(patches, grid_thw=grid).pooler_output
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    print("vision_worker_loading: PASS (BF16 weights, FP32 RoPE, exact reference forward)")


if __name__ == "__main__":
    main()
