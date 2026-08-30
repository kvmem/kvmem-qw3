# QW3 + KVMem

CUDA-native inference and experimental tiered KV memory for long-running Qwen
agents.

QW3 is a purpose-built inference runtime for Qwen `qwen35` hybrid models. It
owns model loading, tokenization, CUDA execution, and device memory instead of
delegating generation to another inference engine. Development currently
focuses on the text path of Qwen3.6 and Qwen3.8 27B models.

Its main research feature, **KVMem**, turns previously computed attention KV
state into reusable agent memory. KVMem keeps a bounded working set on the GPU,
stages colder blocks through host RAM and optional NVMe, and selects relevant
blocks for each new query. The goal is to reduce repeated full-history prefill
and text compaction in long-lived agent sessions.

> [!IMPORTANT]
> **QW3 is currently a research preview.** The host build and host unit tests
> have been reproduced from a fresh build of the current release-preparation
> worktree, but the final release commit has not yet been retested and a
> release-pinned GPU/model gate has not been published. Native Q8 inference is
> currently **Expected**; KVMem, NVFP4, MTP, continuous batching, and frozen
> archives are **Experimental**.
>
> Model weights are not included. A public model revision, checksum, and
> conversion/download procedure still need to be pinned before the first
> release can describe any GPU configuration as release-tested.

## Why QW3?

- **Native CUDA runtime:** QW3 owns the loader, tokenizer, executor, CUDA
  kernels, KV cache, and memory lifecycle.
- **KV memory for agents:** retain contextualized KV blocks across GPU, host
  RAM, and NVMe, then retrieve a bounded working set for the current query.
- **Hybrid-model specialization:** kernels and execution paths are designed for
  Qwen's DeltaNet-plus-attention `qwen35` architecture.
- **Local API serving:** expose OpenAI-compatible completions and an
  Anthropic-compatible Messages subset from the same native runtime.
- **Optional performance paths:** MTP speculative decoding, paged KV,
  continuous batching, FP8 KV, and NVFP4 weights are available for research and
  tuning.

The external `llama-completion` executable is not linked into QW3 and is not a
runtime backend; it is used only by an optional development benchmark. A small
set of llama.cpp-derived CUDA source files remains compiled into QW3 under the
MIT license and is identified in
[the third-party notices](THIRD_PARTY_NOTICES.md).

## How KVMem works

Normal full-context inference keeps every attention KV entry in the active
device context. KVMem separates accumulated memory from the working set used
for the next answer:

```text
conversation history
        │
        ▼
  prefill into KV blocks ───► GPU working set
        │                         ▲
        ├────────────────────► host RAM
        │                         ▲
        └────────────────────► local NVMe
                                  │
new query ──► block scorer ──► selected blocks ──► compact attention ──► answer
```

The three important capacity controls are:

- `--ctx`: maximum accumulated request/session capacity.
- `--kvmem-budget`: history tokens selected into the active semantic working
  set.
- `--kvmem-gen-budget`: space reserved for the new response.

When the working-set budget is smaller than the accumulated history, retrieval
is intentionally lossy. Answer quality therefore depends on the retrieval
method, block size, query, and budget. KVMem is most useful for long-running
agents; it is unnecessary for ordinary short prompts that already fit in GPU
memory.

## Requirements

- Linux; current development and validation use x86-64.
- An NVIDIA GPU and compatible CUDA toolkit.
- CMake 3.16 or newer for the host build, CMake 3.18 or newer for the CUDA
  quick start, and a C++17 compiler. The recorded host baseline uses CMake
  3.22.1 and GCC 11.4. The final SM120 toolchain floor still needs a clean
  release gate.
- A compatible text checkpoint for the Qwen `qwen35` architecture.
- Additional host RAM and fast local NVMe for large KVMem configurations.

QW3 generation is CUDA-only. A host-only build supports model inspection and
unit tests, but it does not provide a CPU generation runtime. AMD/ROCm, Metal,
and other non-NVIDIA device backends are not implemented.

## Quick start: Q8_0 GGUF

This is the smallest QW3 runtime path: in-tree CUDA kernels, FP16 KV, no MTP,
and one request at a time. FlashInfer and Python packages are not required.
The commands assume that CUDA and a compatible Q8_0 GGUF are already
available; model acquisition or conversion time is not included.

The native GGUF loader currently supports GGUF v3 files whose model tensors are
Q8_0 with F32 metadata/scalar tensors. Generic Q4/IQ GGUF files are not
supported.

### 1. Clone and select the model

```bash
git clone https://github.com/Di-Chai/qw3.git
cd qw3

export QW3_MODEL=/absolute/path/to/Qwen3.6-or-Qwen3.8-27B-Q8_0.gguf
export QW3_CUDA_ARCH=120a-real
```

`120a-real` is the current Blackwell development target. Select the architecture
that matches the deployment GPU; intended Q8 build targets include `80`
(Ampere), `86` (RTX 30/A6000), `89` (Ada), and `90` (Hopper). Those targets
remain Expected until each has passed a public release gate. SM120 builds
require CUDA 12.8 or newer.

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

This is an Expected native CUDA path, not yet a release-tested hardware
profile. See [the release baseline](docs/release_baseline.md) for the current
evidence level.

## Try KVMem

> [!WARNING]
> KVMem is experimental. The following command is a configuration template,
> not a capacity, quality, or performance promise. Start with a short prompt and
> compare its output with KVMem disabled before moving to long agent histories.

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
  --kvmem-prefix-cache \
  --kvmem-query-replay
```

OpenAI clients should send the accumulated `messages` history on each turn.
For the serialized route above, `--kvmem-prefix-cache` reuses an exact shared
prompt-and-response prefix and prefills only the appended suffix. The final user
message drives query-conditioned retrieval.

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

KVMem cache/session endpoints and request extensions are experimental. See the
integration documents below before depending on their lifecycle or concurrency
semantics.

## Current support boundary

| Area | RC0 status | Boundary |
|---|---|---|
| Host build, GGUF inspection, and host unit tests | Tested | Recorded preparation snapshot; must be rerun for the release commit |
| Qwen3.6 27B Q8_0 on NVIDIA CUDA | Expected | Internally exercised on RTX PRO 6000 Blackwell; public model/GPU gate pending |
| Qwen3.8 27B Q8_0 on NVIDIA CUDA | Expected | Implementation path exists; public clean-run artifact pending |
| Other Qwen `qwen35`/legacy `qwen3` shapes | Experimental | Metadata may parse, but shape coverage is not release-validated |
| HF `compressed-tensors` NVFP4 on SM120 | Experimental | Requires FlashInfer/CUTLASS and an exact compatible checkpoint |
| KVMem GPU/CPU/NVMe retrieval | Experimental | Off by default; retrieval quality and resource needs depend on the workload |
| MTP, paged KV, and continuous batching | Experimental | Implemented research paths with combination-specific constraints |
| Frozen KVMem archives | Experimental | FP8-only serialized workflow; no continuous batching |
| Q4/IQ GGUF, CPU generation, or non-NVIDIA generation | Unsupported | No native runtime path |

The Hugging Face loader accepts `qwen3_5_text` model directories using the
`compressed-tensors` format. A generic official FP8 or BF16 directory is not a
drop-in replacement. QW3 currently provides language-model inference only; it
does not accept image or video inputs from multimodal Qwen checkpoints.

## Optional runtime paths

The following are intentionally outside the first quick start:

- **FlashInfer:** optional build-time headers for optimized prefill/decode,
  required by FP8 KV and NVFP4 paths used in current development.
- **NVFP4:** 4-bit `compressed-tensors` weights on SM120a, not generic Q4 GGUF.
- **MTP speculative decode:** enable with `--mtp-chain N`; currently
  Experimental.
- **Continuous batching:** enable with `--continuous-batching`; it also enables
  the required paged-KV serving pool and body-batch executor by default.
- **Frozen archives:** `qw3 archive build|query|info` provides an immutable
  long-context workflow with stricter FP8 and serving constraints.

Run `./build/qw3 --help` for the complete current CLI. Advanced flags are not a
stable compatibility contract during the research-preview phase.

## Documentation

- [Release status and support matrix](docs/release_baseline.md)
- [Architecture](docs/architecture.md)
- [Harness compatibility](docs/harness_compatibility.md)
- [Claude Code integration](docs/claude_code.md)
- [Incremental KVMem sessions](docs/kvmem_incremental_session_api.md)
- [Semantic-group retrieval](docs/kvmem_semantic_group_retrieval.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Historical optimization notes and benchmark records are development evidence,
not current support promises. They will be moved out of the stable user
documentation before RC0.

## Host build and unit tests

The host build is useful for contributors working on parsing, policies,
protocol handling, and storage logic. It does not provide generation:

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

A CUDA build without a physical GPU is not runtime evidence: some component
tests can skip when no device is available. Release runtime claims require a
pinned model, a supported physical GPU, and the GPU gate described in the
release baseline.

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
