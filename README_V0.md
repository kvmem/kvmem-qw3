# KVMem QW3

**Million-scale agent memory and high-efficiency on-device inference for
Qwen3.6 and Qwen3.8 27B.**

KVMem QW3 is a purpose-built inference and memory system for running Qwen3.6
and Qwen3.8 27B models on local NVIDIA GPUs.

At its core, **KVMem** turns the model's already-computed KV cache into reusable
agent memory. It keeps an active working set on the GPU, moves long-term memory
across host RAM and NVMe, and retrieves relevant KV blocks for each query. This
enables million-token-scale accumulated memory while reducing text compaction
and repeated prefill computation.

The QW3 runtime is optimized end to end for this workload, including CUDA
kernels, quantized model execution, paged KV management, hierarchical memory,
and OpenAI- and Anthropic-compatible APIs.

## Highlights

- KVMem GPU/host/NVMe tiering for million-token-scale accumulated memory.
- Query-conditioned KV-block retrieval with a bounded GPU working set.
- Direct reuse of contextualized KV memory with less compaction and repeated
  prefill.
- CUDA-native inference optimized for Qwen3.6/3.8 27B hybrid models.
- Q8_0 GGUF and 4-bit NVFP4 `compressed-tensors` weight paths.
- Optional MTP speculative decoding, paged KV, and continuous batching.
- OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible
  `/v1/messages` APIs.

## Requirements

- Linux (current development and validation use x86-64)
- CMake 3.16 or newer
- A C++17 compiler
- A single NVIDIA GPU and a compatible CUDA Toolkit
- CUDA 12.x or 13.x; an SM120 build requires CUDA 12.8 or newer
- Fast local NVMe storage for million-scale KVMem configurations
- A compatible Qwen3.6 or Qwen3.8 27B checkpoint

The model weights are not included in this repository. GPU and host-memory
requirements vary with the model format, KV dtype, context capacity, and KVMem
working-set budget. Reserve system memory in addition to `--kvmem-cpu-gb`,
which controls only the KVMem host tier and is not the total RAM requirement.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Di-Chai/qw3.git
cd qw3
```

### 2. Choose a build profile

KVMem QW3 provides two recommended single-GPU 27B deployment profiles.

| Profile | Weight format | Model input | GPU requirement | FlashInfer |
|---|---|---|---|---|
| 24 GiB NVFP4 | 4-bit NVFP4 | Hugging Face `compressed-tensors` directory | NVIDIA Blackwell SM120, built as `120a-real` | Required |
| 48 GiB Q8_0 | Q8_0 | GGUF file | Compute Capability 8.0+ with at least 48 GiB VRAM | Required for the recommended FP8-KV configuration |

The 24 GiB profile is an SM120 NVFP4 configuration. RTX 3090 (SM86) and RTX
4090 (SM89) use different CUDA architectures and cannot run this NVFP4 weight
path. The Q8_0 kernels target Compute Capability 8.0 and newer, with 48 GiB
VRAM recommended for the complete 27B deployment.

FlashInfer is a build-time source/header dependency. When installed as a
Python package, its package data supplies the headers and bundled CUTLASS tree;
the built `qw3` executable does not import FlashInfer at runtime. NVFP4 weights
and the recommended FP8-KV configurations below require
`QW3_ENABLE_FLASHINFER=ON`.

#### Profile A: 24 GiB GPU, 4-bit NVFP4

The native 4-bit path currently means **NVFP4**, not a generic `Q4_K` or
`IQ4_XS` GGUF. It requires an SM120 Blackwell GPU and FlashInfer headers with
the bundled CUTLASS tree.

Set `FLASHINFER_DATA` to the `flashinfer/data` directory from a compatible
FlashInfer installation. It must contain `include/` and the bundled
`cutlass/include/` tree:

```bash
export FLASHINFER_DATA=/path/to/site-packages/flashinfer/data
export QW3_BUILD_DIR=build-nvfp4

cmake -S . -B "${QW3_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR="${FLASHINFER_DATA}/include"

cmake --build "${QW3_BUILD_DIR}" -j
```

Use a compatible NVFP4 checkpoint directory:

```bash
export QW3_MODEL=/absolute/path/to/Qwen3.6-or-Qwen3.8-27B-NVFP4
```

Pre-converted model artifacts and their download locations will be published
separately. Replace the placeholder with the local checkpoint directory.

#### Profile B: 48 GiB GPU, Q8_0 GGUF

Set `CMAKE_CUDA_ARCHITECTURES` for the target GPU. Common examples include
`80` for A100, `86` for RTX 3090/A6000, `89` for Ada, `90` for Hopper, and
`120a-real` for SM120 Blackwell. Select the value that matches the deployment
GPU; this profile is sized for a card with at least 48 GiB VRAM.

The recommended build below enables FlashInfer for the FP8-KV KVMem path. Set
`FLASHINFER_DATA` as described in Profile A.

```bash
export FLASHINFER_DATA=/path/to/site-packages/flashinfer/data
export QW3_CUDA_ARCH=89
export QW3_BUILD_DIR=build-q8

cmake -S . -B "${QW3_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${QW3_CUDA_ARCH}" \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR="${FLASHINFER_DATA}/include"

cmake --build "${QW3_BUILD_DIR}" -j
```

Use a compatible Q8_0 GGUF file:

```bash
export QW3_MODEL=/absolute/path/to/Qwen3.6-or-Qwen3.8-27B-Q8_0.gguf
```

Pre-converted model artifacts and their download locations will be published
separately. Replace the placeholder with a compatible Q8_0 GGUF file. The
native 4-bit deployment uses the NVFP4 Profile A checkpoint instead of a Q4
GGUF file.

### 3. Inspect the model

Check that KVMem QW3 can read the model before loading it for inference:

```bash
"${QW3_BUILD_DIR}/qw3-inspect" "${QW3_MODEL}"
```

### 4. Prepare KVMem storage

Choose a directory on a fast local NVMe filesystem. The recommended profiles
reserve 128 GiB for KVMem and require at least that much free space.

```bash
export QW3_KVMEM_DIR=/path/on/local/nvme/qw3-kvmem
mkdir -p "${QW3_KVMEM_DIR}"
```

`--kvmem-cpu-gb` controls the pinned host-memory tier. The machine also needs
RAM for the operating system, model mapping, and runtime allocations.

### 5. Start KVMem QW3

Choose the command that matches the build and model selected in step 2.

#### 24 GiB recommended profile: NVFP4

This profile uses a 32K semantic working set and a 16K generation reserve. It
keeps the input embedding on the CPU, uses FP8 KV, and places the million-scale
retrieval index and long-term KV memory in host RAM and NVMe.

```bash
"${QW3_BUILD_DIR}/qw3" serve \
  --model "${QW3_MODEL}" \
  --host 127.0.0.1 --port 8080 \
  --ctx 1048576 \
  --cpu-embedding \
  --kv-dtype fp8 \
  --prefill-chunk 1024 \
  --mtp-chain 0 \
  --no-continuous-batching \
  --kvmem \
  --kvmem-block-tokens 128 \
  --kvmem-budget 32768 \
  --kvmem-gen-budget 16384 \
  --kvmem-sink-tokens 512 \
  --kvmem-recent-tokens 0 \
  --kvmem-method retrieval \
  --kvmem-retrieval-method key-direction-adaptive \
  --kvmem-adaptive-gain-1to2 0.10 \
  --kvmem-adaptive-gain-2to4 0.06 \
  --kvmem-query-conditioned \
  --kvmem-update-mode step \
  --kvmem-prefix-cache \
  --kvmem-query-replay \
  --kvmem-index-placement cpu \
  --kvmem-gpu-memory-ratio 0.50 \
  --kvmem-cpu-gb 16 \
  --kvmem-nvme-dir "${QW3_KVMEM_DIR}" \
  --kvmem-nvme-gb 128 \
  --kvmem-raw-k-nvme
```

#### 48 GiB recommended profile: Q8_0

This profile uses a 100K semantic working set, a 32K generation reserve, FP8
KV, and MTP chain 4. It is the recommended balanced configuration for
million-token Q8_0 workloads.

```bash
"${QW3_BUILD_DIR}/qw3" serve \
  --model "${QW3_MODEL}" \
  --host 127.0.0.1 --port 8080 \
  --ctx 1310720 \
  --kv-dtype fp8 \
  --prefill-chunk 2048 \
  --mtp-chain 4 \
  --no-continuous-batching \
  --kvmem \
  --kvmem-block-tokens 128 \
  --kvmem-budget 102400 \
  --kvmem-gen-budget 32768 \
  --kvmem-sink-tokens 512 \
  --kvmem-recent-tokens 0 \
  --kvmem-method retrieval \
  --kvmem-retrieval-method key-direction-adaptive \
  --kvmem-adaptive-gain-1to2 0.10 \
  --kvmem-adaptive-gain-2to4 0.06 \
  --kvmem-query-conditioned \
  --kvmem-update-mode step \
  --kvmem-prefix-cache \
  --kvmem-query-replay \
  --kvmem-index-placement cpu \
  --kvmem-gpu-memory-ratio 0.50 \
  --kvmem-cpu-gb 32 \
  --kvmem-nvme-dir "${QW3_KVMEM_DIR}" \
  --kvmem-nvme-gb 128 \
  --kvmem-raw-k-nvme
```

### 6. Test the server

Check the service:

```bash
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/models
```

Send an OpenAI-compatible chat request:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qw3",
    "messages": [
      {"role": "user", "content": "Explain virtual memory in one paragraph."}
    ],
    "temperature": 0,
    "max_tokens": 256
  }'
```

OpenAI clients should send the accumulated `messages` history on each turn.
With `--kvmem-prefix-cache`, KVMem QW3 reuses an exact shared
prompt-and-response prefix and prefills only the appended suffix. The current
user message drives adaptive KV-block retrieval, and query replay presents that
message to the selected memory before answer generation.

### 7. Optional: run a local one-shot prompt

```bash
"${QW3_BUILD_DIR}/qw3" \
  --model "${QW3_MODEL}" \
  -p "Explain virtual memory in one paragraph." \
  -n 256
```

Use `--temp 0 --top-k 1` when deterministic output is required.

## How KVMem Capacity Works

KVMem separates accumulated memory from the active context used for each
answer:

- `--ctx` is the maximum accumulated session capacity.
- `--kvmem-budget` is the semantic working set selected for each query.
- `--kvmem-gen-budget` reserves space for newly generated tokens.

A million-token history can therefore remain addressable without making every
decode step attend densely to all one million tokens. KVMem retrieves the
relevant working set, assembles it on the GPU, and keeps the rest of the memory
in the host and NVMe tiers.

The two profiles are tuned reference configurations. Exact headroom depends on
the checkpoint, GPU, CUDA build, prompt shape, and other processes using the
device. KVMem QW3 reports its effective configuration and validates tier
capacity at startup.

## Hardware and Model Support

- KVMem QW3 is designed and optimized for Qwen3.6 and Qwen3.8 27B models.
- The 24 GiB path uses NVFP4 `compressed-tensors` checkpoints on NVIDIA
  Blackwell SM120 and is built with `120a-real`.
- The 48 GiB path uses native Q8_0 GGUF execution. Its CUDA kernels target
  Compute Capability 8.0 and newer.
- Primary end-to-end development and validation use Linux x86-64 on NVIDIA
  Blackwell SM120.
- CUDA 12.x and 13.x builds are used in development; SM120 requires CUDA 12.8
  or newer.
- QW3 inference is CUDA-only. A host-only build supports model inspection and
  host unit tests, but does not provide a generation runtime.
- Million-scale memory uses GPU/host/NVMe tiering with a bounded retrieved
  working set. Retrieval quality is controlled by the working-set budget,
  block size, and retrieval method.

## Troubleshooting

- **`no kernel image is available for execution on the device`:** rebuild with
  `CMAKE_CUDA_ARCHITECTURES` set for the installed GPU.
- **`NVFP4 requires an SM120a FlashInfer/CUTLASS build`:** use an SM120 GPU,
  build with `120a-real`, and verify the FlashInfer and CUTLASS paths.
- **`fp8 KV decode requires FlashInfer`:** rebuild with
  `QW3_ENABLE_FLASHINFER=ON`, or use `--kv-dtype fp16` for a non-FP8 basic
  configuration.
- **CUDA out of memory:** stop other GPU workloads or reduce `--ctx`,
  `--kvmem-budget`, and `--kvmem-gen-budget`.

## Command Reference

```bash
"${QW3_BUILD_DIR}/qw3" --help
```

## License

KVMem QW3 is planned to be released under the Apache License 2.0. A top-level
`LICENSE` file will be added before the public release. Third-party components
retain their respective licenses under `LICENSES/`, and model weights may be
subject to separate license terms.
