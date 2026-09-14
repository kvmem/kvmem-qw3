#!/usr/bin/env python3
"""Collect controlled visual-reference comparisons; do not waive parity gates."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "benchmark/qwen35_small_20260914"
OUT = EXPERIMENT / "vision_reference_identical_inputs"
BINARY = ROOT / "build-qwen35-small/qw3-vision-gpu-parity"
WORKER = Path(__file__).with_name("qwen35_vision_reference_worker.py")
IMAGES = ROOT / "benchmark/osworld_v2/prefix_checkpoint_memory_20260910/validation_v4/vision_inputs"
# Freeze these before running. 030 reproduces the original failure; 006/095
# are held-out screenshot content, not selected by numerical outcome.
CASES = ("image_030.jpg", "image_006.png", "image_095.jpg")
MODES = ("math_fp32", "math_bf16", "flash_bf16")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(name, value):
    path = OUT / name
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def metrics(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    error = a - b
    rms = float(np.sqrt(np.mean(error * error)))
    return dict(cosine=float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))),
                rmse=rms, relative_rmse=rms / float(np.sqrt(np.mean(b * b))),
                mean_abs=float(np.mean(np.abs(error))),
                max_abs=float(np.max(np.abs(error))))


def main():
    OUT.mkdir(exist_ok=True)
    identity = {str(p): sha(p) for p in (
        Path(__file__), BINARY, WORKER, WORKER.with_name("qw3_vision_cpu_worker.py"),
        ROOT / ".venv/lib/python3.13/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
        ROOT / ".venv/lib/python3.13/site-packages/transformers/vision_utils.py",
        ROOT / ".venv/lib/python3.13/site-packages/torch/version.py",
        *(IMAGES / name for name in CASES))}
    # Hash all auxiliary tensors/configs used by the reference, not only the
    # main GGUF language weights (which this diagnostic does not load).
    models = {}
    for size in ("9B", "4B", "27B"):
        directory = ROOT / ("models/Qwen3.8-27B-BF16" if size == "27B"
                            else f"models/Qwen3.5-{size}-BF16")
        index = directory / "model.safetensors.index.json"
        weights = json.loads(index.read_text())["weight_map"]
        shards = sorted({v for k, v in weights.items() if k.startswith("model.visual.")})
        files = {directory / name for name in shards}
        files.update((directory / "config.json", directory / "preprocessor_config.json", index))
        identity.update({str(p): sha(p) for p in sorted(files)})
        models[size] = directory
    if (OUT / "identity.json").exists():
        assert json.loads((OUT / "identity.json").read_text()) == identity, "Cached identity changed"
    save("identity.json", identity)
    assert not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "GPU occupied"
    state = dict(status="running", pid=os.getpid(), started_at=time.time(), completed=0, total=27)
    results = {}
    save("status.json", state)
    try:
        for size, model in models.items():
            results[size] = {}
            for case in CASES:
                outputs = {}
                native_bits = None
                returns = {}
                for mode in MODES:
                    directory = OUT / size / case / mode
                    directory.mkdir(parents=True, exist_ok=True)
                    state.update(model=size, image=case, mode=mode)
                    save("status.json", state)
                    marker = directory / "run.json"
                    if not marker.exists():
                        env = {k: v for k, v in os.environ.items()
                               if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
                        env.update(QW3_VISION_REFERENCE_DEVICE="cuda",
                                   QW3_VISION_REFERENCE_MODE=mode,
                                   QW3_VISION_PARITY_DUMP_DIR=str(directory))
                        command = [str(BINARY), str(model), str(ROOT / ".venv/bin/python"),
                                   str(WORKER), str(IMAGES / case)]
                        with (directory / "console.log").open("w") as log:
                            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                                  stderr=log, timeout=900)
                        assert (directory / "native.bf16").exists()
                        assert (directory / "reference.f32").exists()
                        console = (directory / "console.log").read_text()
                        assert proc.returncode in (0, 1) and "vision parity failed:" not in console
                        assert "rows=" in console and "cosine=" in console
                        marker.write_text(json.dumps(dict(command=command, returncode=proc.returncode,
                            original_gate_passed=proc.returncode == 0), indent=2) + "\n")
                    returns[mode] = json.loads(marker.read_text())
                    bits = np.fromfile(directory / "native.bf16", np.uint16)
                    if native_bits is None:
                        native_bits = bits
                    else:
                        assert np.array_equal(bits, native_bits), "Native output changed across references"
                    value = np.fromfile(directory / "reference.f32", np.float32)
                    assert value.shape == bits.shape and np.isfinite(value).all()
                    outputs[mode] = value
                    state["completed"] += 1
                    save("status.json", state)
                outputs["native_bf16"] = (native_bits.astype(np.uint32) << 16).view(np.float32)
                assert np.isfinite(outputs["native_bf16"]).all()
                results[size][case] = dict(
                    against_fp32={name: metrics(value, outputs["math_fp32"])
                                  for name, value in outputs.items() if name != "math_fp32"},
                    torch_bf16_pair=metrics(outputs["flash_bf16"], outputs["math_bf16"]),
                    original_gates={name: run["original_gate_passed"] for name, run in returns.items()},
                    input_patches="Identical BF16 values for every path")
                save("comparison.json", results)
        state.update(status="complete_pending_interpretation", finished_at=time.time())
    except BaseException as error:
        state.update(status="needs_attention", error=repr(error), finished_at=time.time())
        raise
    finally:
        save("status.json", state)


if __name__ == "__main__":
    main()
