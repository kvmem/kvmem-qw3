# QW3 + KVMem

CUDA-native inference and tiered KV memory for long-running Qwen
agents.

QW3 is a purpose-built inference runtime for Qwen `qwen35` hybrid models. It
owns model loading, tokenization, CUDA execution, and device memory instead of
delegating generation to another inference engine. Development currently
focuses on Qwen3.6 and Qwen3.8 27B text generation plus a native
Qwen3.8 image-input path.

Its main feature, **KVMem**, turns previously computed attention KV
state into reusable agent memory. KVMem keeps a bounded working set on the GPU,
stages colder blocks through host RAM and optional NVMe, and selects relevant
blocks for each new query. The goal is to reduce repeated full-history prefill
and text compaction in long-lived agent sessions.

## Why QW3?

- **Native CUDA runtime:** QW3 owns the loader, tokenizer, executor, CUDA
  kernels, KV cache, and memory lifecycle.
- **KV memory for agents:** retain contextualized KV blocks across GPU, host
  RAM, and NVMe, then retrieve a bounded working set for the current query.
- **Hybrid-model specialization:** kernels and execution paths are designed for
  Qwen's DeltaNet-plus-attention `qwen35` architecture.
- **Local API serving:** expose OpenAI-compatible completions and an
  Anthropic-compatible Messages subset from the same native runtime.
- **Optional native vision:** run the Qwen3.8 visual tower on CPU or CUDA and
  reuse unchanged per-image embeddings across full-transcript requests.
- **Optional performance paths:** MTP speculative decoding, paged KV,
  continuous batching, FP8 KV, and NVFP4 weights can be enabled independently.

## How KVMem works

KVMem treats an agent’s accumulated KV cache as virtual memory. When the workspace exceeds GPU capacity or the model’s context window, it stores completed KV blocks in host memory or NVMe instead of discarding or summarizing them. At each agent step, KVMem uses the current query to select relevant historical blocks and materializes them, in chronological order, into a bounded GPU-resident execution view. By reusing previously computed KV states and loading only the blocks needed for the current step, KVMem supports large persistent workspaces while keeping GPU memory usage bounded.

![Animated high-level KVMem overview: long history remains recoverable across
memory tiers, a new question ranks an internal Mean-K block-summary index, a
selected KV block replaces another block in a fixed-size GPU attention window,
and QW3 generates with the refreshed context.](docs/assets/kvmem-flow.svg)

The following three parameters control KVMem’s capacity:

- `--ctx` sets the maximum logical size of the accumulated request or agent
  workspace, including history stored outside GPU memory.
- `--kvmem-budget` limits the number of historical tokens selected and
  materialized into the active working set at each agent step.
- `--kvmem-gen-budget` reserves space in the active context for newly generated
  response tokens, preventing retrieved history from occupying the entire
  context window.

The active working set, current-step context, and generation reserve must
together fit within the model’s context window and the available GPU KV
capacity. When the accumulated history exceeds `--kvmem-budget`, KVMem uses
query-conditioned retrieval to select the most relevant historical blocks for
the current step. The retrieval method, block size, query construction, and
working-set budget provide flexible controls for balancing context coverage and
execution efficiency. KVMem is designed for long-running agents, with its
benefits becoming increasingly pronounced as the accumulated workspace grows
beyond practical GPU KV capacity.

## Requirements

- Linux; current development and validation use x86-64.
- An Ampere-or-newer NVIDIA GPU and a matching CUDA toolkit. A dedicated GPU
  with at least 48 GiB of memory is recommended for the 27B Q8_0 quick start.
- CMake 3.16 or newer for the host build, CMake 3.18 or newer for the CUDA
  quick start, and a C++17 compiler. SM120 builds require CUDA 12.8 or newer.
- A compatible language checkpoint for the Qwen `qwen35` architecture. Image
  input additionally requires the matching Hugging Face model directory with
  `model.visual.*` weights and the Python preprocessing dependencies.
- Python 3 and the Hugging Face CLI are needed only for the download commands;
  install the latter with `python3 -m pip install --upgrade huggingface_hub`.
- Additional host RAM and fast local NVMe for large KVMem configurations.

QW3 generation is CUDA-only. A host-only build supports model inspection and
unit tests, but it does not provide a CPU generation runtime. AMD/ROCm, Metal,
and other non-NVIDIA device backends are not implemented.

## Quick start: Q8_0 GGUF

This is the smallest QW3 runtime path: in-tree CUDA kernels, FP16 KV, no MTP,
and one request at a time. FlashInfer and Python packages are not required for
the runtime. If a compatible Q8_0 GGUF is already available, set `QW3_MODEL`
to it and skip the optional download block below.

The native GGUF loader currently supports GGUF v3 files whose model tensors are
Q8_0 with F32 metadata/scalar tensors. Generic Q4/IQ GGUF files are not
supported.

### 1. Clone and select the model

```bash
git clone https://github.com/Di-Chai/qw3.git
cd qw3

export QW3_MODEL=/absolute/path/to/Qwen3.6-or-Qwen3.8-27B-Q8_0.gguf
# Example for RTX PRO 6000 Blackwell; replace this for the target GPU.
export QW3_CUDA_ARCH=120a-real
```

For a reproducible public Qwen3.6 Q8 candidate, download the pinned revision
and verify its checksum:

```bash
export QW3_MODEL_ROOT="${PWD}/../qw3-models"
export QW3_Q8_REVISION=82d411acf4a06cfb8d9b073a5211bf410bfc29bf
export QW3_Q8_DIR="${QW3_MODEL_ROOT}/qwen36-q8"
mkdir -p "${QW3_Q8_DIR}"

hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-Q8_0.gguf \
  --revision "${QW3_Q8_REVISION}" \
  --local-dir "${QW3_Q8_DIR}"

export QW3_MODEL="${QW3_Q8_DIR}/Qwen3.6-27B-Q8_0.gguf"
export QW3_Q8_SHA256=f93f517f38e696d35a1a7df2c0e3155a64f4c4dcd662107a146ae263f7fb14ce
printf '%s  %s\n' "${QW3_Q8_SHA256}" "${QW3_MODEL}" | sha256sum -c -
```

Set `QW3_CUDA_ARCH` to the deployment GPU instead of copying `120a-real`
unchanged: use `80` for A100, `86` for A40/A6000/RTX 30, `89` for Ada, `90`
for Hopper, and `120a-real` for SM120 Blackwell.

### 2. Build the CUDA runtime

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${QW3_CUDA_ARCH}"

cmake --build build -j
```

If CMake cannot find `nvcc`, add the CUDA `bin` directory to `PATH` or set
`CUDACXX` before configuring.

### 3. Inspect the GGUF

```bash
./build/qw3-inspect "${QW3_MODEL}"
```

`qw3-inspect` is a GGUF inspection tool; it does not inspect Hugging Face model
directories.

### 4. Run one prompt

```bash
./build/qw3 \
  --model "${QW3_MODEL}" \
  --ctx 8192 \
  --kv-dtype fp16 \
  --prefill-chunk 1024 \
  --temp 0 --top-p 1 --top-k 1 \
  -p "Explain virtual memory in one paragraph." \
  -n 256
```

### 5. Start the HTTP server

```bash
./build/qw3 serve \
  --model "${QW3_MODEL}" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 32768 \
  --kv-dtype fp16 \
  --mtp-chain 0 \
  --no-continuous-batching
```

The server defaults are deliberately conservative: serialized requests, FP16
KV, and no MTP, paged KV, or continuous batching.

In another terminal:

```bash
curl -sS http://127.0.0.1:8080/health

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

The commands above have been checked with the pinned Qwen3.6 Q8_0 file. See
[the hardware compatibility notes](docs/release_baseline.md) for details about
other configurations.

## Optional Qwen3.8 image input

QW3 can accept base64 `data:` image URLs on the OpenAI-compatible chat endpoint
when a matching Qwen3.8 Hugging Face model directory supplies the visual
weights. First prepare the Python worker used for image decoding and
preprocessing:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  torch pillow safetensors 'transformers>=5.10,<6'

export QW3_VISION_MODEL=/absolute/path/to/Qwen3.8-27B-BF16
export QW3_VISION_CPU_PYTHON="${PWD}/.venv/bin/python"
```

The CPU visual frontend works with the basic Q8 build from the previous
section. Only the visual tower runs on CPU; the language model remains on GPU:

```bash
./build/qw3 serve \
  --model "${QW3_MODEL}" \
  --vision-model "${QW3_VISION_MODEL}" \
  --vision-device cpu \
  --host 127.0.0.1 --port 8080 \
  --ctx 32768 \
  --kv-dtype fp16 \
  --mtp-chain 0 \
  --no-continuous-batching
```

Native CUDA vision additionally requires a FlashInfer-enabled build; the basic
`build` binary above does not contain that path:

```bash
export FLASHINFER_DATA=/absolute/path/to/compatible/flashinfer/data

cmake -S . -B build-vision \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_BUILD_TESTS=OFF \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${QW3_CUDA_ARCH}" \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR="${FLASHINFER_DATA}/include" \
  -DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR="${FLASHINFER_DATA}/cutlass/include"

cmake --build build-vision -j

./build-vision/qw3 serve \
  --model "${QW3_MODEL}" \
  --vision-model "${QW3_VISION_MODEL}" \
  --vision-device cuda \
  --host 127.0.0.1 --port 8080 \
  --ctx 32768 \
  --kv-dtype fp16 \
  --mtp-chain 0 \
  --no-continuous-batching
```

Both servers accept the same OpenAI content-block request:

```bash
export IMAGE_FILE=/absolute/path/to/image.png
export IMAGE_BASE64="$(base64 -w0 "${IMAGE_FILE}")"

curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"qw3\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image.\"},
        {\"type\": \"image_url\", \"image_url\": {
          \"url\": \"data:image/png;base64,${IMAGE_BASE64}\"
        }}
      ]
    }],
    \"temperature\": 0,
    \"max_tokens\": 256
  }"
```

`--vision-device cuda` keeps the BF16 visual tower and projected embeddings on
the GPU; `--vision-device cpu` runs the visual model in the worker process and
copies its projected embeddings into the language model. Both paths cache
results per image, so an accumulated request such as `[A,B,C] -> [A,B,C,D]`
only reruns the visual tower for `D`. The default CPU and GPU cache limits are
512 MiB and can be changed with `QW3_VISION_CPU_CACHE_MIB` and
`QW3_VISION_GPU_CACHE_MIB`.

KVMem keeps each visual token span atomic and preserves its M-RoPE coordinates.
Current full-transcript serving does not yet reuse the language-model KV prefix
when the ordered image set changes; per-image caching currently removes visual
tower recomputation, not repeated LM prefill. Video input and multimodal
continuous batching are not implemented. See the
[native CUDA vision notes](docs/multimodal_cuda.md) for validation details.

## Try KVMem

Prepare a directory on a fast local NVMe filesystem:

```bash
export QW3_KVMEM_DIR=/absolute/path/on/local/nvme/qw3-kvmem
mkdir -p "${QW3_KVMEM_DIR}"
```

Start a serialized KVMem service with a 32K selected working set inside a 128K
logical capacity:

```bash
./build/qw3 serve \
  --model "${QW3_MODEL}" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx 131072 \
  --kv-dtype fp16 \
  --mtp-chain 0 \
  --no-continuous-batching \
  --kvmem \
  --kvmem-block-tokens 128 \
  --kvmem-budget 32768 \
  --kvmem-gen-budget 8192 \
  --kvmem-method retrieval \
  --kvmem-retrieval-method mean-k \
  --kvmem-query-conditioned \
  --kvmem-update-mode step \
  --kvmem-index-placement cpu \
  --kvmem-gpu-memory-ratio 0.90 \
  --kvmem-cpu-gb 16 \
  --kvmem-nvme-dir "${QW3_KVMEM_DIR}" \
  --kvmem-nvme-gb 64 \
  --kvmem-query-replay
```

OpenAI clients should send the accumulated `messages` history on each turn.
The command above deliberately leaves warm prefix caching off so that the
first smoke test validates the stateless full-transcript path without relying
on checkpoint reuse. The final user message drives query-conditioned retrieval.

`--kvmem-prefix-cache` is an optional serialized, single-trajectory
optimization that reuses an exact prompt-and-response prefix and prefills only
the appended suffix. Before adding it to a deployment, run
`scripts/kvmem_prefix_cache_above_budget_canary.py` against the exact build and
configuration; it has stricter prompt-continuation and checkpoint invariants
than the base KVMem path.

The public retrieval scorer values are:

- `mean-k`.
- `per-token` (GPU index only).
- `sub-block-mean-k`.
- `key-direction-fixed4`.
- `key-direction-adaptive`.

The older names `mean_attention` and `content_mean` are not accepted by the
current CLI. CPU index placement requires query-conditioned retrieval and does
not support `per-token`.

`--kvmem-gpu-memory-ratio` bounds total QW3 process memory on the GPU, including
model weights and scratch space; it is not a KV-only percentage. The `0.90`
template assumes a dedicated GPU. Lower it only if the resulting ceiling can
still contain the resident model, scratch reserve, selected KV window, and
generation reserve.

Enable `QW3_KVMEM_TRACE=1` and `QW3_KVMEM_TIER_TRACE=1` when validating a new
installation. An above-budget request should show query replay, a non-fallback
retrieval scorer, and tier movement. A successful short chat request proves
HTTP and generation health, but does not prove that sparse KVMem retrieval ran.

## Optional NVFP4 profile

NVFP4 is the memory-efficient weight path for NVIDIA SM120a. It loads a Hugging
Face `compressed-tensors` directory and uses FlashInfer/CUTLASS-backed NVFP4
kernels; it is not a generic Q4 GGUF path.

```bash
export QW3_MODEL_ROOT="${QW3_MODEL_ROOT:-${PWD}/../qw3-models}"
export QW3_NVFP4_REVISION=ccdaab7e68af2409599b8949a8f2685703c9bae5
export QW3_NVFP4_MODEL="${QW3_MODEL_ROOT}/qwen36-nvfp4"
mkdir -p "${QW3_NVFP4_MODEL}"

hf download unsloth/Qwen3.6-27B-NVFP4 \
  --revision "${QW3_NVFP4_REVISION}" \
  --local-dir "${QW3_NVFP4_MODEL}"
```

Set `FLASHINFER_DATA` to a FlashInfer package-data directory containing
`include/` and `cutlass/include/`:

```bash
export FLASHINFER_DATA=/absolute/path/to/compatible/flashinfer/data

cmake -S . -B build-nvfp4 \
  -DCMAKE_BUILD_TYPE=Release \
  -DQW3_BUILD_TESTS=OFF \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR="${FLASHINFER_DATA}/include" \
  -DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR="${FLASHINFER_DATA}/cutlass/include"

cmake --build build-nvfp4 -j

# Loader and tensor-binding smoke test.
./build-nvfp4/qw3 --model "${QW3_NVFP4_MODEL}" --native-plan

# Short generation smoke test; FP8 is the KV-cache dtype, not the weight dtype.
./build-nvfp4/qw3 \
  --model "${QW3_NVFP4_MODEL}" \
  --ctx 8192 \
  --kv-dtype fp8 \
  --prefill-chunk 1024 \
  --temp 0 --top-p 1 --top-k 1 \
  -p "Explain virtual memory in one paragraph." \
  -n 256
```

Do not run `qw3-inspect` on this Hugging Face directory; that tool accepts GGUF
files only.

## APIs

The serialized server exposes:

- `GET /health`.
- `GET /v1/models`.
- `POST /v1/completions`.
- `POST /v1/chat/completions`.
- `POST /v1/messages` and `POST /v1/messages/count_tokens` for the supported
  Anthropic-compatible subset.
- `POST /tokenize` and `POST /v1/tokenize`.

The built-in server does not provide authentication or TLS. Keep it bound to
`127.0.0.1`; if it must be reachable beyond localhost, place it behind an
authenticated TLS reverse proxy and apply normal network access controls.

## Model format notes

The language-model Hugging Face loader accepts `qwen3_5_text` directories using
the `compressed-tensors` format. A generic official FP8 or BF16 directory is
not a drop-in replacement for that language path. The optional visual frontend
separately reads matching `model.visual.*` weights from a Hugging Face model
directory; it currently accepts images but not video.

## Optional runtime paths

The following are intentionally outside the first quick start:

- **FlashInfer:** optional build-time headers for optimized prefill/decode,
  required by FP8 KV and NVFP4 paths used in current development.
- **NVFP4:** 4-bit `compressed-tensors` weights on SM120a, not generic Q4 GGUF.
- **MTP speculative decode:** enable with `--mtp-chain N`.
- **Image input:** enable with `--vision-model DIR --vision-device cpu|cuda`;
  image requests currently use serialized serving.
- **Continuous batching:** enable with `--continuous-batching`; it also enables
  the required paged-KV serving pool and body-batch executor by default.
- **Frozen archives:** `qw3 archive build|query|info` provides an immutable
  long-context workflow with stricter FP8 and serving constraints.

Run `./build/qw3 --help` for the complete current CLI.

## Documentation

- [Release status and support matrix](docs/release_baseline.md)
- [Architecture](docs/architecture.md)
- [Harness compatibility](docs/harness_compatibility.md)
- [Claude Code integration](docs/claude_code.md)
- [Incremental KVMem sessions](docs/kvmem_incremental_session_api.md)
- [Semantic-group retrieval](docs/kvmem_semantic_group_retrieval.md)
- [Native CUDA vision frontend](docs/multimodal_cuda.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## Host build and unit tests

The host build is useful for contributors working on parsing, policies,
protocol handling, and storage logic. It does not provide generation:

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

## Project layout

```text
include/qw3/   Public C++ headers and model/storage interfaces
src/           Runtime, server, model loading, and CUDA kernels
tests/         Host and CUDA component tests
scripts/       Regression, benchmark, and research tooling
docs/          Architecture, integration, and research documentation
third_party/   Vendored source headers covered by third-party notices
```

## License

QW3 source is licensed under the [Apache License 2.0](LICENSE). Third-party
components retain their respective licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model checkpoints are
distributed separately and may use different license terms.
