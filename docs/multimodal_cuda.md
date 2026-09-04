# Native CUDA vision frontend

QW3 can run the Qwen3.5/Qwen3.8 visual tower on the same CUDA device as the
language model while keeping the existing CPU frontend available.

```bash
./build/qw3 serve \
  --model models/Qwen3.8-27B-Q8_0.gguf \
  --vision-model models/Qwen3.8-27B-BF16 \
  --vision-device cuda
```
`--vision-device cpu` selects the original worker-based implementation. The
legacy `--vision-cpu-model DIR` option remains an alias for the CPU form.

The CUDA frontend uses the Hugging Face image processor only for image decode,
resize, normalization, and patchification. It loads the `model.visual.*` BF16
weights directly from safetensors and runs patch projection, learned position
interpolation, all 27 visual transformer blocks, non-causal FlashInfer
attention, and the merger on CUDA. The projected BF16 rows remain on the GPU
and are scattered into the language-model input tensor in one device launch;
there is no projected-embedding D2H/H2D round trip.

The visual weights consume about 879 MiB for Qwen3.8-27B. Final embeddings are
cached per image, so full-transcript clients only encode newly added or changed
images. Cached image tensors are assembled as a zero-copy segmented embedding
table in request order. The default GPU cache limit is 512 MiB and can be
changed with `QW3_VISION_GPU_CACHE_MIB` (use `0` to disable it). Serving logs
report both `vision_cache_hits` and `vision_cache_misses` per request. The CPU
vision frontend uses the same per-image policy under `QW3_VISION_CPU_CACHE_MIB`.

Visual token spans remain mandatory under KVMem, retain their three-axis
M-RoPE coordinates, and are preserved by local-cache and persistent-session
checkpoints. A request is rejected when its mandatory visual span alone exceeds
the configured KVMem active budget.

For numerical validation against the Transformers visual tower:

```bash
QW3_VISION_REFERENCE_DEVICE=cuda \
  ./build/qw3-vision-gpu-parity \
  models/Qwen3.8-27B-BF16 \
  .venv/bin/python \
  scripts/qw3_vision_cpu_worker.py \
  image.jpg
```
