# qw3

`qw3` is a native CUDA inference engine and OpenAI-compatible server for the
Qwen3.5, Qwen3.6, and Qwen3.8 hybrid text-model family (`qwen35` /
`qwen3_5_text`). It owns model loading, tokenization, device memory, attention,
DeltaNet recurrent state, speculative decoding, and serving. The production
path does not call llama.cpp.

The project is optimized for model-specific local inference rather than broad
framework compatibility. NVFP4 execution currently targets Blackwell SM120a;
other CUDA targets are primarily intended for GGUF Q8 models.

## Features

- Native Q8_0 GGUF and Hugging Face `compressed-tensors` model loading.
- Mixed NVFP4, FP8, and BF16 safetensors execution on Blackwell.
- FlashInfer prefill and decode attention, including FP8 KV cache support.
- Qwen MTP speculative decoding with fixed or adaptive chain depth.
- OpenAI-compatible chat, completions, streaming, thinking, and function tools.
- Schema-guided tool-call structure and incremental streaming for large
  string-valued tool arguments.
- Continuous batching with paged KV for multi-request serving.
- KVMem block retrieval with bounded GPU windows and optional CPU/NVMe tiers.
- Prefix reuse for both continuous-batching and single-request KVMem routes.

## Support Matrix

| Model or feature | Current support | Requirements |
|---|---|---|
| Q8_0 GGUF | Native load and execution | CUDA SM80 or newer; select the matching CMake architecture |
| NVFP4/FP8 safetensors | Native mixed-precision execution | FlashInfer/CUTLASS build targeting SM120a |
| Qwen3.5/3.6/3.8 text models | Supported when the checkpoint uses the `qwen3_5_text` architecture | Text model only; multimodal heads are not loaded |
| FP8 KV cache | Supported | FlashInfer-enabled build |
| MTP | Supported when the checkpoint contains a compatible draft head | Enable with `--mtp-chain N` |
| KVMem + MTP | Supported on the serialized single-request route | Do not enable continuous batching |
| Continuous batching + MTP | Supported without KVMem | Paged KV is enabled automatically |
| Continuous batching + KVMem + MTP | Not supported | The server rejects this combination |

## Requirements

- Linux or WSL2.
- CMake 3.16 or newer and a C++17 compiler.
- A CUDA toolkit and driver compatible with the selected GPU architecture.
- FlashInfer headers for the recommended attention path and FP8 KV cache.
- FlashInfer CUTLASS headers for NVFP4 on SM120a.
- Enough host memory for the model mapping, tokenizer, pinned transfer buffers,
  and any configured KVMem CPU tier.

For Q8 on another GPU, replace `120a-real` with the appropriate architecture,
for example `90` for Hopper or `89` for Ada. NVFP4 requires an SM120a build.

## Build

### CUDA build

This build is sufficient for GGUF/Q8 execution with the in-tree CUDA kernels:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real
cmake --build build -j
```

### FlashInfer and NVFP4 build

Point CMake at the headers shipped by a recent FlashInfer installation. The
example below uses the common Python-wheel data layout:

```sh
FI_DATA=/path/to/python/site-packages/flashinfer/data

cmake -S . -B build-flashinfer -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR="$FI_DATA/include" \
  -DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR="$FI_DATA/cutlass/include"
cmake --build build-flashinfer -j
```

Optional CuTe DSL AOT kernels for GDN prefill and fused RMSNorm/FP4
quantization are documented in
[`docs/kvmem_gpu_memory_optimization.md`](docs/kvmem_gpu_memory_optimization.md).

### Verify the build

```sh
ctest --test-dir build-flashinfer --output-on-failure
./build-flashinfer/qw3 --help
```

To inspect model bindings without generating:

```sh
./build-flashinfer/qw3 --model /path/to/model --native-plan
```

## Models

`--model` accepts either:

- a GGUF file, such as `Qwen3.6-27B-Q8_0.gguf`; or
- a Hugging Face model directory containing safetensors, configuration, and
  `compressed-tensors` quantization metadata.

The loader selects the source automatically from whether the path is a file or
directory. `--cpu-embedding` can keep an untied BF16 input embedding table on
CPU and transfer only the selected rows. It requires a separate LM head.

### MTP weight preparation

MTP weights are model weights, not KV-cache data. The server does not quantize
a BF16 MTP head to FP8 while loading. If a memory profile assumes an FP8 MTP
head, create that model directory offline and update its quantization metadata
accordingly.

In particular:

```text
--kv-dtype fp8
```

only selects FP8 storage for the KV cache. It does not change the precision of
the base model or MTP head.

## Quick Start

### Local generation

```sh
./build/qw3 \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  -p "Explain the Adam optimizer in one paragraph." \
  -n 256
```

### OpenAI-compatible server

The conservative server default is one request at a time, FP16 KV, no MTP,
and no KVMem:

```sh
./build-flashinfer/qw3 serve \
  --model /path/to/model \
  --host 127.0.0.1 \
  --port 18080
```

Check the server:

```sh
curl -sS http://127.0.0.1:18080/v1/models
```

Send a streaming chat request:

```sh
curl -N http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "local-model",
    "stream": true,
    "stream_options": {"include_usage": true},
    "messages": [{"role": "user", "content": "Explain prefix caching."}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 512
  }'
```

## Recommended 24 GiB Qwen3.8 KVMem Profile

This profile is intended for a nominal 24 GiB GPU. It provides a 256K logical
context, a 48K selected active window, and a 24K generation reserve.

> **The tested model is the Unsloth Qwen3.8-27B-NVFP4 text checkpoint with an
> MTP head that was separately quantized to FP8.** The stock NVFP4 directory
> must be converted before using this exact memory profile. Qw3 does not perform
> that conversion at startup.

```sh
QW3_KVMEM_SCRATCH_RESERVE_MIB=768 \
./build-flashinfer/qw3 serve \
  --model /path/to/Qwen3.8-27B-NVFP4-MTPFP8 \
  --host 127.0.0.1 \
  --port 18080 \
  --ctx 262144 \
  --kv-dtype fp8 \
  --prefill-chunk 512 \
  --cpu-embedding \
  --kvmem \
  --kvmem-budget 49152 \
  --kvmem-gen-budget 24576 \
  --kvmem-cpu-gb 10 \
  --kvmem-gpu-memory-ratio 0.99 \
  --kvmem-opt-stage-out off \
  --kvmem-update-mode step \
  --kvmem-prefix-cache \
  --kvmem-query-conditioned \
  --mtp-chain 3 \
  --native-mtp-speculate \
  --no-continuous-batching \
  --enable-thinking \
  --preserve-thinking \
  --thinking-budget 10240
```

`QW3_KVMEM_SCRATCH_RESERVE_MIB=768` is a GPU-pool sizing knob (default 3072).
It is not an allocated scratch buffer; it leaves room under
`--kvmem-gpu-memory-ratio 0.99` so the 48K+24K resident window still fits.
No SSD tier is configured. At this context size, approximately 8.5 GiB is
enough for complete immutable raw-K and spilled-V CPU backing; the 10 GiB tier
leaves working margin.

| Capacity item | Value |
|---|---:|
| Logical context | 262,144 tokens |
| Active selection budget | 49,152 tokens |
| Generation reserve and server output cap | 24,576 tokens |
| Always-kept prefix | auto (~1,024 tokens) |
| KVMem CPU tier | 10 GiB |
| NVMe tier | Disabled |
| 256K FP16 mean-K index at block size 128 | About 64 MiB on GPU |
| Observed process GPU peak | About 23.7 GiB on a 24.5 GiB device |

Driver behavior, CUDA allocator high-watermarks, display use, and model layout
can change the peak; keep the GPU free of other CUDA workloads and measure on
the target machine.

`--kvmem-cpu-gb` only accounts for the KVMem tier. Keep at least 16-20 GiB of
total host memory free for the tier, host model mapping, pinned buffers,
tokenizer, and OS. A 128-token block reduces index and metadata overhead but
uses coarser retrieval units than smaller block sizes.

## Serving

### Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List the loaded model |
| `POST /v1/completions` | OpenAI-style text completion |
| `POST /v1/chat/completions` | Chat, streaming, thinking, and function tools |

The API reports prompt and completion usage for both buffered and streaming
responses. A client disconnect cancels the in-flight generation; committed
KVMem prefix checkpoints remain available for the next request.

### Thinking and sampling

`--enable-thinking` sets the server default for Qwen thinking mode.
`--thinking-budget N` force-closes an overlong `<think>` section after `N`
tokens. `--preserve-thinking` keeps historical assistant reasoning when a
client returns `reasoning_content`, which is important for exact multi-turn
prompt reconstruction and KVMem prefix-cache hits.

Requests can override the server defaults with `enable_thinking`,
`preserve_thinking`, or `chat_template_kwargs.preserve_thinking`.

The CLI defaults are temperature `0.6`, top-p `0.95`, and top-k `20`. API
request values take precedence. Passing `-n N` to `serve` installs a
server-side output cap; otherwise each request may use its remaining context.

### Function tools

`/v1/chat/completions` accepts OpenAI-style `tools` schemas. When tools are
present, the server:

- constrains legal function and parameter names from the supplied schema;
- validates required fields and the completed call before returning it;
- streams argument deltas for tools whose properties are plain strings;
- buffers complex array/object schemas until the complete call validates; and
- returns a structured `tool_call_parse_error` when a malformed call cannot be
  repaired safely.

Incremental string arguments allow clients to observe progress during large
`write` or `edit` calls without buffering the complete payload. The current
constraint is tool-structure guidance, not a general-purpose implementation of
every JSON Schema keyword.

### Continuous batching

Enable the scheduler explicitly:

```sh
./build-flashinfer/qw3 serve \
  --model /path/to/model \
  --host 127.0.0.1 \
  --port 18080 \
  --continuous-batching \
  --max-active 4 \
  --kv-dtype fp8 \
  --mtp-chain 3
```

Continuous batching automatically enables the global paged-KV pool and body
batch executor. The default maximum active request count is 2. Increase it
only after sizing the KV pool and generation reservations for the target GPU.
Do not combine continuous batching, KVMem, and MTP in the same process.

### Prefix caching

There are two prefix-cache implementations for different serving routes:

| Flag | Route | Reuse behavior |
|---|---|---|
| `--prefix-cache` | Continuous batching | Reuses page-aligned main/MTP KV and recurrent state across shared prompt prefixes |
| `--kvmem-prefix-cache` | Serialized KVMem | Keeps the executor warm and resumes strict multi-turn prompt extensions from committed checkpoints |

For agent clients, use `--preserve-thinking` when the client returns historical
reasoning. Otherwise the reconstructed token prefix can differ and prevent a
cache hit.

## MTP Speculative Decoding

Models with a compatible MTP draft head can propose multiple tokens and verify
them against the target model in one batched step:

```sh
./build-flashinfer/qw3 serve \
  --model /path/to/model-with-mtp \
  --mtp-chain 3
```

`--mtp-chain 0` disables MTP. A fixed chain is the recommended production
policy; `--mtp-policy adaptive` is available for workloads that have been
benchmarked with dynamic depth. MTP consumes additional model, KV, and state
memory, so disable it first when fitting a strict VRAM limit.

Use `--native-plan` to confirm `mtp_supported: yes` and inspect the bound draft
tensors. Speculative verification follows the target path, although bit-exact
greedy comparisons can still differ at borderline ties when different
FlashInfer kernels introduce FP16 rounding differences.

## KVMem

KVMem separates the logical context length from the GPU-resident attention
window. It stores context in fixed token blocks, scores candidate blocks for a
query, keeps a selected working set on GPU, and stages cold data to CPU memory
or NVMe when configured:

```text
logical context -> selected GPU window -> CPU tier -> optional NVMe tier
```

KVMem is opt-in. Without `--kvmem`, the normal dense KV path is unchanged.

### Selection and update policy

The recommended query-conditioned path uses per-block mean-K representations:

```sh
--kvmem-method retrieval \
--kvmem-retrieval-method mean-k \
--kvmem-query-conditioned \
--kvmem-select-policy topk
```

`--kvmem-update-mode step` selects after prefill and keeps that selection
during decode. `interval` additionally reselects every `--kvmem-interval N`
generated tokens. Step mode avoids repeated scoring and I/O during a response;
interval mode can adapt during very long generations.

### Main KVMem options

| Option | Meaning |
|---|---|
| `--ctx N` | Complete logical prompt plus generation limit |
| `--kvmem-block-tokens N` | Physical retrieval granularity; default 128 |
| `--kvmem-budget N` | Maximum selected semantic/decode window |
| `--kvmem-prefill-budget N` | Pressure window while ingesting long history; defaults to the selection budget |
| `--kvmem-gen-budget N` | GPU reserve for newly generated tokens |
| `--kvmem-sink-tokens N` | Prefix tokens always retained |
| `--kvmem-recent-tokens N` | Recent suffix tokens always retained |
| `--kvmem-retrieval-method M` | `mean-k`, `per-token`, `sub-block-mean-k`, or key-direction variants |
| `--kvmem-index-placement gpu|cpu` | Keep the complete retrieval index on GPU or stream bounded tiles from CPU |
| `--kvmem-query-max-tokens N` | Split long retrieval queries into bounded groups; default 512 |
| `--kvmem-update-mode step|interval` | Selection cadence |
| `--kvmem-cpu-gb F` | CPU capacity for raw-K and offloaded KV records |
| `--kvmem-nvme-gb F` | Optional NVMe capacity; requires `--kvmem-nvme-dir` |
| `--kvmem-raw-k-nvme` | Put the complete immutable raw-K authority on NVMe instead of CPU |
| `--kvmem-prefix-cache` | Resume strict prompt extensions on the serialized KVMem server route |

Smaller blocks improve retrieval granularity but increase block metadata,
index size, and selection work. GPU index placement is simple and fast when
the index fits. CPU placement bounds index VRAM for much longer logical
contexts at the cost of H2D scoring traffic.

### CPU and NVMe tiers

For moderate contexts, provide enough CPU memory and omit all NVMe arguments.
For larger contexts, add a backing directory and capacity:

```sh
--kvmem-cpu-gb 32 \
--kvmem-nvme-gb 256 \
--kvmem-nvme-dir /fast/nvme/qw3-kvmem
```

By default, immutable raw-K remains fully resident in the CPU tier. Add
`--kvmem-raw-k-nvme` only when the CPU budget cannot hold it and the NVMe tier
has enough additional capacity. NVMe removes the CPU capacity requirement but
adds storage latency and write traffic.

### KVMem compatibility

- Single-request KVMem supports MTP and `--kvmem-prefix-cache`.
- KVMem with continuous batching is supported when MTP is disabled.
- KVMem with continuous batching and MTP is rejected.
- Local-position KVMem MTP requires the active budget plus generation reserve
  to fit inside the model's trained context limit.
- Frozen context archives use FP8 storage and a serialized query route; see
  [`docs/kvmem_context_archive_status.md`](docs/kvmem_context_archive_status.md).

## Common Configuration

Run `qw3 --help` for the complete and authoritative option list.

| Option | Default | Purpose |
|---|---:|---|
| `--ctx N` | 262144 | Maximum prompt plus generated tokens |
| `--prefill-chunk N` | 2048 for serving | Bounds prefill scratch; smaller chunks reduce peak VRAM |
| `--kv-dtype fp16|fp8|fp32|q8` | fp16 | KV-cache storage precision |
| `--cpu-embedding` | off | Keep an untied BF16 input embedding table on CPU |
| `-n N` | remaining context for serving | Server output cap |
| `--temp F` | 0.6 | Sampling temperature |
| `--top-p F` | 0.95 | Nucleus-sampling threshold |
| `--top-k N` | 20 | Top-k sampling threshold |
| `--enable-thinking` | off | Enable thinking mode by default |
| `--thinking-budget N` | 0 | Maximum tokens inside a thinking section; 0 disables the cap |
| `--preserve-thinking` | off | Preserve returned historical reasoning in later prompts |
| `--mtp-chain N` | 0 | MTP speculative depth; 0 disables MTP |
| `--continuous-batching` | off | Enable paged multi-request scheduling |
| `--max-active N` | 2 | Maximum active continuous requests |

## Diagnostics

Normal serving should not require development environment overrides. These
traces are useful when validating a deployment:

| Variable | Output |
|---|---|
| `QW3_KVMEM_TIER_TRACE=1` | GPU/CPU/NVMe block usage and transfer events |
| `QW3_KVMEM_TRACE=1` | Retrieval, selection, and assembly details |
| `QW3_KVMEM_PREFIX_CACHE_TRACE=1` | Serialized KVMem checkpoint hits and restores |
| `QW3_PREFIX_CACHE_TRACE=1` | Continuous-batching prefix-cache activity |

The server also logs per-request prompt tokens, prefill time, decode speed,
finish reason, tool-call status, and client cancellation.

## Tests

Build-time tests include tokenizer, chat-template, tool parser/constraint,
KVMem store, archive, and CUDA parity coverage when the corresponding backend
is enabled:

```sh
ctest --test-dir build-flashinfer --output-on-failure
```

Benchmark and evaluation harnesses remain under `scripts/`; they are not part
of the normal serving startup path.

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmark suite](docs/benchmark_suite.md)
- [KVMem implementation notes](docs/kvmem_implementation_notes.md)
- [KVMem GPU-memory optimization](docs/kvmem_gpu_memory_optimization.md)
- [KVMem tiered I/O design](docs/kvmem_tiered_io_design.md)
- [KVMem known issues](docs/kvmem_known_issues.md)
- [KVMem utility evaluation](docs/kvmem_utility_evaluation.md)
- [MTP and paged-KV optimization](docs/mtp_paged_kv_optimization.md)
- [Development history](docs/DEVELOPMENT_LOG.md)

Third-party notices are available in [`LICENSES/`](LICENSES/).
