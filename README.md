# qw3

A from-scratch CUDA inference framework for the **Qwen 3.5 / Qwen 3.6** family
(architecture `qwen35`) — a hybrid model with 16 standard attention layers
interleaved among 48 DeltaNet recurrent layers. The framework loads GGUF
weights directly, runs its own tokenizer, owns all device memory, and ships
hand-written CUDA kernels (Q8_0 DP4A matvec / matmul, fused flash-attention
decode, batched prefill).

llama.cpp is kept around only as a correctness baseline and benchmarking
counterpart — it never participates in actual generation.

## Status (Qwen 3.6 27B Q8_0, RTX Pro 6000 Blackwell, CUDA-enabled llama.cpp)

Both engines run greedy (qw3 is always argmax; llama.cpp is invoked with
`--temp 0`), so a correct implementation would emit identical tokens.
Numbers below come from `scripts/fi_sweep.py` (3 trials per cell,
alternating qw3-default ↔ qw3+FlashInfer ↔ llama.cpp to spread thermal
drift), median tok/s, peak HBM polled at 50 ms.

**Default config — FlashInfer prefill + decode + MMQ v8 matmul,
memory parity with llama.cpp.** Build runs `--prefill-chunk 2048`
and `QW3_MATMUL=auto`, which routes every prefill matmul (batch ≥ 8)
to the INT8-MMA path: **MMQ v8** (128×128 tile) at batch ≥ 128,
**MMQ v7** (64×64 tile) below. Prefill attention runs **FlashInfer**
(`SinglePrefillWithKVCacheDispatched<HEAD_DIM=256, kCausal, ...>`)
and decode attention runs **FlashInfer**
(`SingleDecodeWithKVCacheDispatched<HEAD_DIM=256, group_size=6, ...>`,
stream-K + MMA) when the build flag `-DQW3_ENABLE_FLASHINFER=ON` is
set — the configuration the table below measures. Override with
`QW3_PREFILL_ATTN=mma-gqa-v2` to restore the in-tree FA2 v2 prefill,
or `QW3_DECODE_ATTN=native` to restore the in-tree
`fattn_vec_decode_f16_splitk`. HGEMM-with-FP16-dequant is no longer
in the default path; Q8 weights stay 8-bit in HBM end-to-end.
`qw3_cli` sets `CUDA_MODULE_LOADING=EAGER` at process init so all
kernel modules resolve before the first launch (short prefill is no
longer launch-overhead-bound, and FI's modules load without affecting
later kernel-launch cost). Peak process HBM sits ~2.3 GiB above
llama.cpp at every T, flat in T (chunk=2048 batch scratch + cuBLAS
workspace + FI Q/O-pack + FI decode chunked merge tmp, not a
per-token leak).

| Prompt tokens | qw3 prefill | llama prefill | prefill % | qw3 decode | llama decode | decode % | qw3 peak | llama peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|    827 | 3445 tok/s | 3264 tok/s | **105.5%** | 45.06 tok/s | 45.87 tok/s | **98.2%**  | 30.8 GiB | 29.0 GiB |
|   2453 | 3737 tok/s | 3684 tok/s | **101.4%** | 45.73 tok/s | 45.35 tok/s | **100.8%** | 31.4 GiB | 29.0 GiB |
|   4621 | 3835 tok/s | 3721 tok/s | **103.1%** | 45.66 tok/s | 45.14 tok/s | **101.2%** | 31.4 GiB | 29.0 GiB |
|   8686 | 3962 tok/s | 3796 tok/s | **104.4%** | 45.65 tok/s | 44.44 tok/s | **102.7%** | 31.4 GiB | 29.0 GiB |
|  16816 | 3904 tok/s | 3716 tok/s | **105.1%** | 45.07 tok/s | 43.50 tok/s | **103.6%** | 31.4 GiB | 29.0 GiB |
|  33347 | 3741 tok/s | 3522 tok/s | **106.2%** | 43.52 tok/s | 42.41 tok/s | **102.6%** | 31.4 GiB | 29.0 GiB |
|  66138 | 3444 tok/s | 3057 tok/s | **112.7%** | 41.02 tok/s | 40.17 tok/s | **102.1%** | 33.2 GiB | 30.9 GiB |
| 131720 | 2940 tok/s | 2292 tok/s | **128.3%** | 36.79 tok/s | 35.83 tok/s | **102.7%** | 37.3 GiB | 34.8 GiB |

Throughput in absolute tokens/second, n_decode=512, ctx=36864 (T=66K
and T=131K bump ctx to fit). Memory columns are net process-peak
(peak `nvidia-smi memory.used` minus idle-GPU baseline) sampled at
50 ms while each engine runs — same instrument for both. The `%`
columns report `qw3 / llama.cpp`. Headlines:

- **Prefill beats llama.cpp at every T** — 101.4–106.2% from T=2K
  through T=33K, then widens to **+12.7% at T=66K and +28.3% at
  T=131K**. The FlashInfer kernel closes the Tensor-Core utilization
  gap that bottlenecked the in-tree FA2 v2 at long T (23% TC util →
  ~65%, NCU-confirmed).
- **Decode beats llama.cpp at every T ≥ 2K** — 100.8–103.6% from T=2K
  through T=131K. The FlashInfer decode kernel (stream-K + MMA at
  single-token) closes the long-T per-call attention gap that
  bottlenecked the in-tree `fattn_vec_decode` at T=131K (667 us/call
  → ~357 us/call, nsys-confirmed); decode at T=131K lifts from 91.0%
  → **102.7%** of llama.cpp.
- **Memory: +2.3 GiB at every T, flat.** Includes the ~17 MiB FI
  prefill Q-pack + O-pack (fp16) sharing `prefill_gqa_scratch_` plus
  the FI decode chunked-merge tmp buffer. Memory parity with
  llama.cpp is preserved end-to-end.

Reproduce with:

```sh
# qw3-default vs qw3+FlashInfer vs llama.cpp, 8 cells, 3 trials/cell:
python3 scripts/fi_sweep.py \
  --prompt-tokens "556 2182 4350 8415 16545 33076 65867 131073" \
  --trials 3 -n 512 --json /tmp/fi_sweep.json

# Or default-only (no FlashInfer column):
python3 scripts/long_prompt_sweep.py \
  --prompt-tokens "512 2048 4096 8192 16384 32768 65536" \
  --trials 3 -n 512 -c 70000 \
  --json /tmp/sweep.json
```

### Why MMQ at every batch size

Earlier auto policy was `batch ≤ 512 → MMQ v7, batch > 512 → HGEMM`
because v7's 64×64 tile lost ~10% to HGEMM at large batch. **MMQ v8**
(8-warp 128×128 tile + v7's split-plane Q8 weight + 144-B
`block_q8_1_mmq_t` activations + 2-stage cp.async + XOR-swizzled shmem +
m16n8k32 INT8 MMA) closes that gap and wins outright at every batch ≥
1K. The auto router now picks v8 for batch ≥ 128 and v7 for the tail —
both INT8 paths, identical memory profile, no FP16 weight scratch. See
`DEVELOPMENT_LOG.md` for the v8 derivation.

### Prefill chunk vs. throughput ceiling

The default `--prefill-chunk 2048` keeps per-chunk batch scratch
bounded (~33.5 GiB peak at T=64K vs ~67 GiB whole-prompt). For the
absolute throughput ceiling — at the cost of larger per-prompt scratch
— pass `--no-prefill-chunk` to batch the entire prompt in one call.
That's the right knob for short-T runs where extra throughput matters
more than memory headroom. The default chunk=2048 is the recommended
config for any prompt length where memory parity matters.

### FA2 v2 fallback — builds without FlashInfer

When `-DQW3_ENABLE_FLASHINFER=ON` is not set at build time, prefill
attention falls back to the in-tree **FA2 v2** kernel (BR=16 NCOLS2=2
+ K/V padded shmem + selective cp.async + s_S/s_P stride pad). To
A/B against the FI default at runtime in an FI build:
`QW3_PREFILL_ATTN=mma-gqa-v2`. FA2 v2 vs llama.cpp under the same
sweep:

| Prompt tokens | qw3 prefill (v2) | llama prefill | prefill % | qw3 decode (v2) | llama decode | decode % |
|---:|---:|---:|---:|---:|---:|---:|
|    827 | 3386 tok/s | 3213 tok/s | **105.4%** | 45.68 tok/s | 44.56 tok/s | **102.5%** |
|   2453 | 3699 tok/s | 3679 tok/s | **100.5%** | 45.49 tok/s | 44.68 tok/s | **101.8%** |
|   4621 | 3765 tok/s | 3720 tok/s | **101.2%** | 44.79 tok/s | 44.42 tok/s | **100.8%** |
|   8686 | 3836 tok/s | 3810 tok/s | **100.7%** | 44.60 tok/s | 42.50 tok/s | **104.9%** |
|  16816 | 3667 tok/s | 3723 tok/s | **98.5%**  | 42.79 tok/s | 42.65 tok/s | **100.3%** |
|  33347 | 3332 tok/s | 3519 tok/s | **94.7%**  | 42.24 tok/s | 41.36 tok/s | **102.1%** |
|  66138 | 2813 tok/s | 3060 tok/s | **91.9%**  | 38.12 tok/s | 38.99 tok/s | **97.8%**  |
| 131720 | 2138 tok/s | 2296 tok/s | **93.1%**  | 31.83 tok/s | 35.01 tok/s | **90.9%**  |

FA2 v2 is competitive at short-to-mid T (94–105% of llama) but
Tensor-Core-stalled at long T (T=66K = 91.9%, T=131K = 93.1%, NCU
confirms 23% TC util vs llama's 65%). Levers attempted on FA2 v2
(BR=64 NCOLS2=1, BR=32 NCOLS2=2, FP16-O accumulator) all hit
occupancy / spill / argmax walls in our codebase — the q-rows-per-CTA
direction is exhausted without a major rewrite. FlashInfer was ported
in as the escape valve.

### How the FlashInfer port is wired

**Prefill** — the adapter
(`src/flashinfer_prefill_adapter.{cu,hpp}`, ~240 LoC) dispatches
`flashinfer::SinglePrefillWithKVCacheDispatched<HEAD_DIM=256,
KV_LAYOUT=NHD, kCausal, ...>` from inside qw3's attention path. Q
is packed FP16 from the FP32 Q buffer, K/V are already FP16 in the
KV cache, O is written FP16 then converted back to FP32 for the
rest of the forward. Q-pack and O-pack share
`prefill_gqa_scratch_`, the same scratch buffer the in-tree FA2 v2
path uses — no extra allocation. The dispatch is gated on `batch
≥ 8` (the existing `QW3_PREFILL_ATTN_MIN_BATCH` knob), so it only
fires for prefill, not decode.

**Decode** — the adapter
(`src/flashinfer_decode_adapter.{cu,hpp}`, ~700 LoC) dispatches
`flashinfer::SingleDecodeWithKVCacheKernel<PosEncodingMode::kNone,
NumStages=2, ..., GroupSize=6>` (Qwen 3.6 GQA group=6 specialization,
stream-K with MMA at single-token). Q is the same FP32 Q buffer as
the in-tree decode path, K/V are FP16 in the KV cache, O is written
FP16 to a workspace then unpacked to FP32. At seq_len > 256 the
adapter chunks the KV using `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
and merges partial outputs via `flashinfer::MergeStates` — that's
the lever the in-tree `fattn_vec_decode` (fixed NSPLIT=64) was
missing at long T. Workspace memory is the chunked-merge tmp buffer
(O fp16 + LSE fp32, sized to (heads × HEAD_DIM × num_chunks)
elements). Override at runtime with `QW3_DECODE_ATTN=native` to
restore the in-tree path.

`CUDA_MODULE_LOADING=EAGER` (set by `qw3_cli` `main()` before any CUDA
call) is load-bearing: without it, FI's first call triggers a one-shot
`cuLibraryLoadData` (135 ms, nsys-confirmed) that grows the CUDA
driver's kernel registry process-wide and raises the floor cost of
every subsequent `cudaLaunchKernel` — including decode's ~700
launches/token, causing a flat 15% decode regression at every T. EAGER
pre-resolves all modules at `cuInit` and eliminates the regression.

Build with FlashInfer:

```sh
git clone --depth 1 https://github.com/flashinfer-ai/flashinfer.git /tmp/flashinfer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR=/tmp/flashinfer/include
cmake --build build -j
```

Reproduce the A/B with `scripts/fi_sweep.py` (3-way sweep:
qw3-default vs qw3+`QW3_PREFILL_ATTN=mma-gqa-v2` vs llama.cpp).

## Build

CUDA build (required for actual model execution):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CPU-only (inspection / mock backend tests):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### NVFP4 / FP8 safetensors

The native backend also accepts a Hugging Face model directory containing a
`compressed-tensors` Qwen3.5/Qwen3.6 checkpoint. Mixed checkpoints may combine
NVFP4 W4A4 group-16 MLP weights, FP8 W8A8 projections, and BF16 fallback
weights. NVFP4 uses FlashInfer's SM120 CUTLASS kernel and is enabled
automatically only for a `120a` CUDA build; Q8 and non-Blackwell FlashInfer
builds do not compile this adapter.

For memory-constrained GPUs, `--cpu-embedding` keeps an untied BF16 input
embedding table in the checkpoint's host mapping. Prefill gathers the selected
token rows through a double-buffered pinned staging area, and decode transfers
one row per token. The LM head and all compute weights remain on GPU. The flag
is opt-in and rejects checkpoints whose input embedding is also the LM head.

When MTP is disabled (`--mtp-chain 0`, the serving default), qwen-native also
skips uploading the unused MTP draft weights. Enabling MTP or its diagnostic
trace automatically restores those weights at model load.

#### Recommended 24 GiB NVFP4 profile

For the Unsloth Qwen3.6-27B-NVFP4 checkpoint on a 24 GiB GPU, prefer
single-request serving with CPU-resident input embeddings, FP8 KV cache, a
96K-token context, 1024-token prefill chunks, and MTP disabled:

```sh
./build-cu13/qw3 serve \
  --model /path/to/Qwen3.6-27B-NVFP4 \
  --host 127.0.0.1 --port 18080 \
  --ctx 98304 \
  --kv-dtype fp8 \
  --prefill-chunk 1024 \
  --cpu-embedding \
  --mtp-chain 0 \
  --no-continuous-batching
```

This profile reached a measured peak of approximately 22.63 GiB with a
96,490-token prompt, leaving about 1.37 GiB of a nominal 24 GiB device for
runtime variation. The `--ctx` limit includes both prompt and generated tokens,
so their combined length must remain at or below 98,304. Exact headroom varies
with the CUDA build, driver, and GPU.

For the FlashInfer wheel layout used by the `vllm` environment:

```sh
FI_DATA=/path/to/vllm/lib/python3.12/site-packages/flashinfer/data
cmake -S . -B build-flashinfer -DCMAKE_BUILD_TYPE=Release \
  -DQW3_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a-real \
  -DQW3_ENABLE_FLASHINFER=ON \
  -DQW3_FLASHINFER_INCLUDE_DIR="$FI_DATA/include" \
  -DQW3_FLASHINFER_CUTLASS_INCLUDE_DIR="$FI_DATA/cutlass/include"
cmake --build build-flashinfer -j

./build-flashinfer/qw3 serve \
  --model /path/to/Qwen3.6-27B-NVFP4 \
  --host 127.0.0.1 --port 18080 --mtp-chain 4
```

SM120 builds can optionally use FlashInfer's CuTe-DSL chunked GDN kernel for
long prefill. Export the Qw3-specific modulo-head variant, then point CMake at
the generated object and the CuTe DSL runtime:

```sh
PYTHONPATH=/path/to/flashinfer-main \
  /path/to/cuda13/python scripts/export_gdn_sm120_aot.py \
  --output-dir build-cu13/gdn-sm120-aot

cmake -S . -B build-cu13 \
  -DQW3_GDN_SM120_AOT_DIR="$PWD/build-cu13/gdn-sm120-aot" \
  -DQW3_CUTE_DSL_RUNTIME_DIR=/path/to/nvidia_cutlass_dsl/lib
cmake --build build-cu13 -j

./build-cu13/qw3 serve \
  --model /path/to/Qwen3.6-27B-NVFP4
```

The AOT route applies only to regular prefill batches of at least 64 tokens.
Decode, MTP checkpointing, and KVMem DeltaNet capture retain the existing
recurrent kernel. It is the default when the build includes the AOT object and
also accelerates compatible Q8 models. Set `QW3_RECURRENT_KERNEL=ported` to
restore the previous recurrent prefill path.

NVFP4 builds can additionally AOT-compile FlashInfer's fused BF16
RMSNorm-plus-FP4-quantization kernel. The exported kernel is specialized for
the model hidden size:

```sh
/path/to/cuda13/python scripts/export_rms_fp4_sm120_aot.py \
  --hidden-size 5120 \
  --output-dir build-cu13/rms-fp4-sm120-aot

cmake -S . -B build-cu13 \
  -DQW3_RMS_FP4_SM120_AOT_DIR="$PWD/build-cu13/rms-fp4-sm120-aot" \
  -DQW3_CUTE_DSL_RUNTIME_DIR=/path/to/nvidia_cutlass_dsl/lib
cmake --build build-cu13 -j
```

When present, this route is used for compatible BF16 NVFP4 prefill batches of
at least 256 tokens. Smaller batches and decode retain the existing kernel, and
an unsupported launch falls back automatically. Set
`QW3_NVFP4_FLASHINFER_RMS_QUANT=0` to disable it.

On RTX Pro 6000 Blackwell, the default single-request decode policy uses the
fastest measured hybrid route: one packed gate/up CUTLASS FP4 Tensor Core
GEMM, the tuned CUTLASS FP4 down-projection kernel, and cuBLASLt FP8 Tensor
Cores for the mixed checkpoint's FP8 projections. The residual stream and
final attention/FFN projection outputs use BF16 storage while normalization,
attention, and recurrent state calculations remain FP32. The equivalent
explicit settings are:

```sh
QW3_NVFP4_SMALL_SINGLE=0 \
QW3_NVFP4_FUSED_CUTLASS_PAIR=1 \
QW3_NVFP4_VLLM_TACTICS=1 \
QW3_NVFP4_BF16_MAIN=1 \
QW3_FP8_SMALL_CUBLASLT=1 \
./build-flashinfer/qw3 serve \
  --model /path/to/Qwen3.6-27B-NVFP4 \
  --host 127.0.0.1 --port 18080
```

Mixed FP8/BF16 recurrent fanouts share both activation conversions by default.
On the Qwen 3.6 `10240/6144/48/48 x 5120` projection shapes, shared FP8
scaling plus the fused small BF16 pair reduced the single-layer microbenchmark
from about `0.101 ms` to `0.050 ms`; set `QW3_FP8_MIXED_FANOUT=0` to restore
per-projection dispatch for comparison.

The GGUF/Q8 model path and kernels remain unchanged; the loader selects the
model source from whether `--model` names a file or a directory.

Tested with CUDA 12.x / 13.x. For non-Blackwell targets, swap
`CMAKE_CUDA_ARCHITECTURES` accordingly (e.g. `90` for Hopper, `89` for Ada).
The default of `120a-real` matters — JIT'd Ampere PTX leaves measurable
performance on the table on consumer Blackwell.

## Run

```sh
./build/qw3 \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  -p "Explain Adam optimizer in one paragraph." \
  -n 256
```

Key flags (see `qw3 --help` for the full list):

| Flag | Purpose |
|---|---|
| `--backend qwen-native`        | Native engine. This is now the default; use `--backend llama-cli` only for external llama.cpp comparison. |
| `--native-heavy`               | Compatibility flag. Native generation is enabled by default. |
| `--native-kernels cuda`        | Diagnostic override. CUDA is the default and currently required for native inference. |
| `--native-linear-backend auto` | Diagnostic override for linear backend A/B tests. Default is `auto`. |
| `-p "..."` / `--prompt-file`   | Prompt (chat-formatted by default; use `--raw` to skip Qwen chat template). |
| `--system "..."`               | System prompt (default: a generic assistant prompt). |
| `--think`                      | Don't inject the empty `<think>` block. |
| `-n N`                         | Max new tokens (default 256). |
| `-c N`                         | KV cache size (default 32768). |
| `--dump-logits PATH`           | Write per-step top-K logits as JSONL for parity diffs. |
| `--dump-tokens`                | Tokenize prompt and exit. |

Inspect a GGUF without running it:

```sh
./build/qw3-inspect /path/to/Qwen3.6-27B-Q8_0.gguf
```

Smoke test without weights:

```sh
./build/qw3 --backend mock -p hello
```

## MTP speculative decode

Qwen 3.6 ships a single Multi-Token-Prediction (MTP / NextN) draft head
(`nextn_predict_layers: 1`, bound at layer index 64). The Q8_0 GGUF already
carries the 15 MTP head tensors — check with `qw3-inspect` / `--native-plan`
(`mtp_supported: yes`). The draft head proposes a chain of speculative tokens;
the target verifies them in a single batched forward and accepts the longest
greedy-matching prefix, rolling back KV + DeltaNet recurrent state on the first
rejection.

Single-request speculative decode:

```sh
./build/qw3 \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  --mtp-chain 2 \
  -p "Explain Adam optimizer in one paragraph." \
  -n 256
```

Serving with MTP:

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --continuous-batching \
  --max-active 4 \
  --kv-dtype fp8 \
  --mtp-chain 4
```

| Flag | Purpose |
|---|---|
| `--mtp-chain N` | MTP speculative chain length. `0` disables MTP; `N>0` enables speculative decode. |
| `--continuous-batching` | Enables the service scheduler. When combined with `--mtp-chain N`, the server enables the continuous MTP path. |
| `--mtp-batched-draft` / `--no-mtp-batched-draft` | Force-enable or force-disable batched MTP draft projection/FFN/logits in serving. By default this is enabled when MTP + continuous batching are enabled. |
| `--mtp-paged-prefix` / `--no-mtp-paged-prefix` | Force-enable or force-disable the paged MTP prefix cache. By default this is enabled when MTP + paged KV are enabled. |
| `--native-mtp-chain N` | Compatibility alias for `--mtp-chain N`. Prefer `--mtp-chain`. |
| `--native-mtp-trace` | Diagnostic mode: run draft-head tracing without normal speculation. |

**Acceptance / lossiness.** Speculative decode is greedy-lossless *when the
draft-built and decode-built KV caches come from the same attention kernel*. The
in-tree `native` attention (`QW3_DECODE_ATTN=native QW3_PREFILL_ATTN=mma-gqa-v2`)
satisfies this: speculative output is byte-identical to plain greedy. The
**FlashInfer default** uses a batched prefill kernel to prime the MTP prefix and
a single-token decode kernel for plain steps; those agree only up to fp16
rounding, so a borderline-tied argmax can flip a token. This is a numerical
property of mixing two FI kernels, not an acceptance-logic error. Set
`QW3_DECODE_ATTN=native QW3_PREFILL_ATTN=mma-gqa-v2` if you need bit-exact
speculative == greedy.

The non-MTP greedy path is untouched and remains byte-identical regardless of
attention backend.

Scripts (`./build/qw3` is the default binary path):

```sh
# Acceptance sweep, compact table + optional JSON:
python3 scripts/mtp_acceptance_probe.py --mtp-chain 2 --mtp-speculate -n 64

# Head-to-head vs llama.cpp draft-MTP (needs llama-server with --spec-type draft-mtp):
python3 scripts/mtp_compare_with_llama_cpp.py --mtp-chains 2 --prompt-tokens "4096 8192"
```

### Adaptive MTP depth policy

The current serving recommendation is to choose a fixed chain with
`--mtp-chain N` and benchmark it for the workload. The older adaptive controller
still exists as a development diagnostic. If enabled with `QW3_MTP_POLICY`, it
promotes or demotes the draft depth from windowed acceptance statistics
(benefit = `full_accept_rate / avg_committed_tokens`; marginal cost from a
per-depth round-cost table). Add `QW3_MTP_POLICY_TRACE=1` to log per-batch
`depth / action / benefit / cost`. These knobs are not required for normal
serving:

| Env var | Default | Effect |
|---|---|---|
| `QW3_MTP_POLICY`              | `off`   | `adaptive` enables the depth controller. |
| `QW3_MTP_ADAPTIVE_MAX_CHAIN`  | chain   | Upper depth bound for promotion. |
| `QW3_MTP_ADAPTIVE_MIN_CHAIN`  | `4` when max depth >= 4, else `1` | Lower depth bound for demotion. |
| `QW3_MTP_ADAPTIVE_UPDATE_INTERVAL` | `16` | Batches per control window. |
| `QW3_MTP_ADAPTIVE_MIN_BATCHES`| `64`    | Warmup batches before the first promotion/demotion. |
| `QW3_MTP_ADAPTIVE_COOLDOWN`   | `8`     | Windows to wait after a depth change. |
| `QW3_MTP_ADAPTIVE_PROMOTE_MARGIN` / `_DEMOTE_MARGIN` | `0.005` | Benefit-vs-cost margins gating a change. |
| `QW3_MTP_ADAPTIVE_STARTUP_DEMOTE_BATCHES` | `0` | Optional early demotion probe at the initial max depth; disabled by default. |

### MTP correctness / verifier knobs

These are internal diagnostic knobs for verifier A/B tests. They are not part
of the normal serving startup command.

| Env var | Default | Effect |
|---|---|---|
| `QW3_MTP_VERIFY`              | `batched` | `sequential` verifies drafts one token at a time (slower; same acceptance). |
| `QW3_MTP_SAFE_MAX_CHAIN`      | guard   | Caps the effective chain to a correctness-safe maximum. |
| `QW3_MTP_PREFIX_MAX_PROMPT`   | guard   | Disables prefix priming above this prompt length (falls back gracefully). |
| `QW3_MTP_TRANSACTIONAL_REPLAY`| `1`     | Commit verifier tokens through the stable single-token state path. |

## Paged KV and continuous batching

The `continuous_batching` branch now has a service-oriented paged-KV and
continuous-batching path for OpenAI-compatible workloads. Service behavior is
controlled by explicit `qw3 serve` switches. Some lower-level backend toggles
still exist for development diagnostics, but normal serving should not require
setting environment variables.

### Current implementation status

Implemented:

- **Request-isolated logical page tables.** Each request owns a logical KV page
  table. Logical pages map to physical pages allocated by a backend-level page
  allocator.
- **Global physical KV pool for continuous batching.** When the server is
  started with `--continuous-batching`, standard attention K/V storage is
  allocated once at backend load time and shared by active request executors
  through per-request page tables.
- **Paged KV dtypes.** Continuous batching supports `fp16` and raw e4m3 `fp8`
  KV cache storage. FP8 paged decode paths dispatch through the FP8 FlashInfer
  launcher instead of accidentally reading FP8 cache as FP16.
- **OpenAI-compatible service routing.** `/v1/chat/completions` and
  `/v1/completions` can route greedy, streaming, tools-shaped, and OpenCode-like
  requests through the continuous scheduler. Usage accounting reports real
  prompt/completion token counts for both streaming and non-streaming responses.
- **Admission control.** The service tracks active requests, pending requests,
  and token reservations. Over-context prompts return HTTP 413; pool/token
  exhaustion returns clear admission errors instead of crashing the worker.
- **Batched decode executor.** The body-batch path can batch standard attention
  layers with per-row RoPE positions, ragged paged KV append, FlashInfer ragged
  paged decode attention, FFN, recurrent layers, lm-head, and argmax for greedy
  no-penalty requests. Unsupported sampling/penalty cases fall back to the
  delegated safe path.
- **Continuous MTP integration.** MTP speculative decode can run with paged KV
  and continuous batching. The verifier is batched; the MTP draft path can also
  batch the expensive projection/FFN/logits work across requests. This is the
  default when serving with `--continuous-batching --mtp-chain N`.

Not complete yet:

- **True fully batched MTP attention.** Current batched MTP draft uses batched
  projection/FFN/logits, but per-row paged MTP attention is still used for
  stability. A direct ragged paged attention call with MTP's gated Q layout
  exposed a crash, so the stable path reuses the existing paged MTP attention
  interface row-by-row rather than adding a new kernel.
- **True FlashInfer paged prefill everywhere.** Plain local prefill was restored
  to the contiguous high-throughput FlashInfer prefill path by default.
  External/global KV pool prefill still needs a true FlashInfer paged prefill
  integration to recover the same prefill throughput without giving up paged KV
  storage.
- **Sampling and penalties in the full batched body executor.** Greedy
  no-penalty requests use the optimized body-batch path. Sampling, presence
  penalty, and repetition penalty still use the safer fallback path where full
  logits are required.
- **Large-matrix benchmark coverage.** Current regression and smoke benchmarks
  cover correctness and representative throughput, but the full 4K/8K/16K/64K/
  128K by concurrency matrix should be rerun before declaring service-level
  performance final.

### Service command

`qw3 serve` defaults to the conservative baseline: one request at a time,
FP16 KV, no global paged-KV serving pool, no continuous batching, and no MTP.
Use explicit flags to opt into the newer serving features.

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Notes:

- `--ctx` already defaults to `262144`.
- Omit `-n` for OpenAI-compatible serving if the desired default is "use the
  remaining context window". Passing `-n N` installs an explicit service-side
  generation cap.
- Serving defaults to `--kv-dtype fp16`. Use `--kv-dtype fp8` for lower KV
  memory pressure after FP8 validation.
- Add `--continuous-batching` to enable the continuous scheduler. This also
  enables the required global paged-KV serving pool and body-batch decode
  executor by default.
- Add `--paged-kv` to enable the global paged-KV serving pool explicitly.
- Use `--mtp-chain N` to control MTP. `--mtp-chain 0` is the default and means
  MTP is off. `N>0` enables MTP speculation.
- Use `--max-active N` to raise or lower continuous-batching concurrency once
  `--continuous-batching` is enabled.
- Use `--kv-page-size N`, `--kv-pool-pages N`, and `--mtp-kv-pool-pages N` to
  tune the paged-KV pool. `0` pool pages means auto.
- Enable **prefix caching** (lossless page-aligned KV reuse across requests that
  share a prompt prefix) with `--prefix-cache`. It is off by default and
  currently wired on the plain continuous-batching path (MTP requests
  cold-prefill in v1). See the prefix-cache section below.
- The intended serving matmul path is MMQ and HGEMM is disabled internally for
  `qw3 serve`; HGEMM is not part of the continuous batching optimization plan.

Serving feature switches:

| Switch | Default | Effect |
|---|---:|---|
| `--continuous-batching` | off | Enables the continuous scheduler. Also enables paged KV and body batch unless explicitly disabled. |
| `--no-continuous-batching` | n/a | Compatibility/debug switch to force the baseline single-request service path. |
| `--paged-kv` | off | Enables the global paged-KV serving pool without enabling the continuous scheduler. |
| `--no-paged-kv` | n/a | Disables the global paged-KV pool. Cannot be combined with `--continuous-batching`. |
| `--body-batch` | on when continuous batching is on | Enables the optimized greedy decode body executor. |
| `--no-body-batch` | n/a | Forces the safer delegated per-request body path. |
| `--max-active N` | `2` | Maximum active continuous requests admitted to the decode loop. |
| `--max-pending N` | `128` | Maximum queued requests waiting for continuous admission. |
| `--max-total-tokens N` | `ctx` | Admission token reservation budget. `0` disables this budget. |
| `--kv-page-size N` | `16` | Logical and physical KV page size in tokens. |
| `--kv-pool-pages N` | auto | Number of physical pages in the global KV pool. `0` means auto. |
| `--mtp-kv-pool-pages N` | auto | Number of physical pages in the separate MTP prefix KV pool. `0` means auto. |
| `--kv-dtype fp16\|fp8\|fp32\|q8` | `fp16` | KV cache dtype. FP16 is the conservative default; FP8 reduces KV memory. |
| `--mtp-chain N` | `0` | MTP speculative chain length. `0` disables MTP; `N>0` enables it. |
| `--mtp-batched-draft` | auto | Forces batched MTP draft projection/FFN/logits. Auto-on for MTP + continuous batching. |
| `--no-mtp-batched-draft` | n/a | Disables batched MTP draft for debugging. |
| `--mtp-paged-prefix` | auto | Forces paged MTP prefix KV. Auto-on for MTP + paged KV. |
| `--no-mtp-paged-prefix` | n/a | Disables paged MTP prefix for debugging. |
| `--prefix-cache` | off | Lossless prefix KV caching: reuse a shared prompt prefix's KV across requests. Requires `--continuous-batching`; MTP requests cold-prefill in v1. |

Explicit examples:

Enable continuous batching:

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --continuous-batching
```

Enable continuous batching with FP8 KV and MTP:

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --continuous-batching \
  --kv-dtype fp8 \
  --mtp-chain 4
```

Increase active continuous requests:

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --continuous-batching \
  --max-active 4 \
  --max-pending 128
```

Tune paged-KV block and pool sizing:

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --continuous-batching \
  --kv-page-size 32 \
  --kv-pool-pages 8192 \
  --mtp-kv-pool-pages 8192
```

### Prefix caching (lossless KV reuse)

Prefix caching reuses the KV pages and recurrent/conv state of an already-served
prompt prefix, so a later request that shares a page-aligned prefix skips
re-prefilling it. Reuse is lossless: a cache-hit greedy completion is
byte-identical to the same prompt run cold (verified on the hybrid
DeltaNet+attention model). Enable it with the `--prefix-cache` switch; it is
**off by default**. The page budget (unlimited, bounded by the KV pool) is the
tuned value and is not exposed as a separate switch.

| Switch | Default | Effect |
|---|---:|---|
| `--prefix-cache` | off | Enables lossless prefix KV caching on the continuous-batching path. Requires `--continuous-batching`. |

Scope and limits (v1):

- Reuse fires when one prompt is a **strict token-prefix** of another — re-ask
  of the same prompt, or multi-turn append (identical history + a new question
  at the end). Measured end-to-end latency saved in the multi-turn-append case:
  ~92% at 16k, ~96% at 32k, ~98% at 64k shared prefix (prefill collapses from
  seconds to ~0.06s; the longer the shared context, the larger the win).
- The "shared system preamble + a *different* question" case does not reuse yet
  (each distinct question commits its own entry). That is the next milestone.
- **MTP** requests (`--mtp-chain N`) bypass the cache in v1 and cold-prefill.
- Enabling the cache currently forces the per-row prefill path (batch-prefill is
  disabled so commits land exactly on page boundaries).

Complete launch command with continuous batching + prefix caching:

```sh
./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --continuous-batching \
  --max-active 4 \
  --max-pending 128 \
  --kv-dtype fp16 \
  --prefix-cache
```

Verification scripts: `scripts/prefix_cache_canary.py` (lossless invariant),
`scripts/prefix_cache_eviction.py` (LRU eviction + cache-off regression),
`scripts/prefix_cache_latency_bench.py` (16k/32k/64k latency savings).

### OpenAI-compatible smoke tests

Model list:

```sh
curl -sS http://127.0.0.1:8080/v1/models
```

Non-streaming chat:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "你好，用一句话介绍你自己。"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

Streaming chat:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-27B-Q8_0.gguf",
    "stream": true,
    "stream_options": {"include_usage": true},
    "messages": [{"role": "user", "content": "写一个三句话的科幻故事。"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Simple concurrent probe:

```sh
python3 - <<'PY'
import concurrent.futures, json, urllib.request

url = "http://127.0.0.1:8080/v1/chat/completions"
prompts = [
    "请帮我写一个末日生存的小说开头。",
    "请帮我写一个末日生存的小说开头。",
    "请帮我写一个末日生存的小说开头。",
    "请帮我写一个末日生存的小说开头。",
]

def run(i, prompt):
    body = json.dumps({
        "model": "Qwen3.6-27B-Q8_0.gguf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
    }).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as resp:
        data = json.loads(resp.read().decode())
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return i, len(text), usage

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    for row in ex.map(lambda x: run(*x), enumerate(prompts)):
        print(row)
PY
```

### Regression commands

Paged KV regression:

```sh
python3 scripts/paged_kv_regression.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --page-sizes '16 32' \
  --alloc-modes 'identity reverse' \
  --prompts 'short chinese' \
  --max-tokens 8 \
  --ctx 1024 \
  --prefill-chunk 512 \
  --out-json /tmp/qw3_paged_kv_regression.json \
  --timeout 900
```

Continuous batching regression:

```sh
python3 scripts/continuous_batching_regression.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --prompts 'capital math cuda chinese' \
  --max-tokens 8 \
  --ctx 1024 \
  --prefill-chunk 512 \
  --out-json /tmp/qw3_continuous_batching_regression.json \
  --timeout 900 \
  --max-active 4 \
  --min-batch 2 \
  --enable-body-batch \
  --require-body-batch-mode \
  --require-ragged-metadata
```

Continuous batching with FP8 KV:

```sh
python3 scripts/continuous_batching_regression.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --prompts 'capital math cuda chinese' \
  --max-tokens 8 \
  --ctx 1024 \
  --prefill-chunk 512 \
  --out-json /tmp/qw3_continuous_batching_fp8_regression.json \
  --timeout 900 \
  --max-active 4 \
  --min-batch 2 \
  --enable-body-batch \
  --require-body-batch-mode \
  --require-ragged-metadata \
  --extra-arg=--kv-dtype \
  --extra-arg=fp8
```

MTP + continuous batching regression:

```sh
python3 scripts/mtp_continuous_regression.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 4096 \
  --max-tokens 16 \
  --chain 4 \
  --kv-dtype fp8 \
  --concurrent-continuous 2 \
  --timeout 900 \
  --out-json /tmp/qw3_mtp_continuous_default_fp8.json \
  --skip-text-compare
```

MTP + continuous batching with FP16 KV:

```sh
python3 scripts/mtp_continuous_regression.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 4096 \
  --max-tokens 16 \
  --chain 4 \
  --kv-dtype fp16 \
  --concurrent-continuous 2 \
  --timeout 900 \
  --out-json /tmp/qw3_mtp_continuous_fp16.json \
  --skip-text-compare
```

### Benchmark commands and current results

Continuous batching smoke benchmark:

```sh
python3 scripts/continuous_batching_benchmark.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --prompts 'capital math' \
  --max-tokens 32 \
  --ctx 2048 \
  --prefill-chunk 512 \
  --out-json /tmp/qw3_cb_benchmark_smoke.json \
  --timeout 900 \
  --max-active 2 \
  --variants 'plain continuous body recurrent'
```

Observed smoke result:

| Variant | Completion throughput |
|---|---:|
| plain | 43.52 tok/s |
| continuous | 43.91 tok/s |
| body | 47.17 tok/s |
| recurrent | 70.98 tok/s |

MTP continuous-batching throughput check:

```sh
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 4096 \
  --max-tokens 128 \
  --prompt-repeat 32 \
  --chain 4 \
  --kv-dtype fp8 \
  --concurrency-levels '1,4' \
  --modes 'continuous,continuous_mtp' \
  --timeout 900 \
  --out-json /tmp/qw3_mtp_batched_draft_compare_c1_c4_fp8.json
```

Short smoke variant used during README validation:

```sh
python3 scripts/mtp_throughput_benchmark.py \
  --qw3 ./build/qw3 \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 4096 \
  --max-tokens 32 \
  --prompt-repeat 8 \
  --chain 4 \
  --kv-dtype fp8 \
  --concurrency-levels '1,2' \
  --modes 'continuous,continuous_mtp' \
  --timeout 900 \
  --out-json /tmp/qw3_mtp_batched_draft_smoke_fp8.json
```

Observed short smoke result:

| Mode | Concurrency | Output throughput | Notes |
|---|---:|---:|---|
| continuous | 1 | 37.28 tok/s | no MTP |
| continuous | 2 | 61.75 tok/s | no MTP |
| continuous_mtp | 1 | 58.29 tok/s | 18 accepted / 38 drafted |
| continuous_mtp | 2 | 74.03 tok/s | 43 accepted / 76 drafted |

Observed result after `7081750 Batch MTP draft across continuous requests`:

| Mode | Concurrency | Output throughput | Notes |
|---|---:|---:|---|
| continuous | 1 | 39.46 tok/s | no MTP |
| continuous | 4 | 112.51 tok/s | no MTP |
| continuous_mtp | 1 | 97.97 tok/s | 95 accepted / 131 drafted |
| continuous_mtp | 4 | 117.85 tok/s | 365 accepted / 570 drafted, 146 verifier batches |

This is the first measured point where MTP plus continuous batching beats
ordinary continuous batching at four concurrent requests on this benchmark
(`117.85 tok/s` vs `112.51 tok/s`). Before the batched draft change, the same
4-way MTP benchmark was around `88 tok/s`; the draft compute was still mostly
per-request and dominated high-concurrency MTP.

### Development checkpoints

Recent local commits on the `continuous_batching` branch:

| Commit | Summary |
|---|---|
| `7081750` | Batch MTP draft across continuous requests. |
| `13ae691` | Stabilize paged MTP prefix cache with a separate global MTP KV pool and FlashInfer paged prefill for prefix priming. |
| `e2ad3c7` | Add paged MTP prefix cache path. |
| `0f70675` | Add continuous MTP batched verifier path. |

The latest validated baseline at the time of writing:

- `cmake --build build -j`: passed.
- `git diff --check`: passed.
- MTP continuous regression, default FP8 path: passed.
- MTP continuous regression, FP8 path: passed.
- MTP continuous regression, FP16 path: passed.

### Next optimization priorities

1. **Implement true FlashInfer paged prefill for global KV.** This is the
   highest-priority paged-KV performance item. Local/plain prefill can reach
   the contiguous FlashInfer prefill path, but continuous batching with global
   paged KV still needs the proper paged prefill interface.
2. **Make MTP attention truly batched.** Current batched MTP draft batches the
   heavy projection/FFN/logits work, but attention remains per-row for
   stability. The next step is to adapt FlashInfer paged decode/paged-ragged
   decode to MTP's gated Q layout without writing a custom attention kernel.
3. **Broaden FP8 KV coverage.** FP8 paged decode and MTP regression pass, but
   long-context and higher-concurrency FP8 sweeps should be repeated before
   treating FP8 KV as the production default.
4. **Finish the full throughput matrix.** Run 4K, 8K, 16K, 64K, and 128K
   prompt lengths with concurrency 1/2/4/8 and decode length 512 for
   prefill/decode throughput reporting.
5. **Sampling/penalty batching.** The optimized body-batch executor currently
   targets greedy no-penalty requests. Sampling and penalty paths need a batched
   logits/filtering design before they can use the same high-throughput body
   executor.
6. **OpenAI compatibility stress tests.** Continue OpenCode/tool-call/streaming
   tests with multiple concurrent agents, long conversations, cancellation, and
   client disconnects.

## Development tuning knobs

Most defaults are correct on Blackwell + Qwen 3.6. Normal serving should use
the explicit `qw3 serve` switches above. The environment knobs below are
developer diagnostics for A/B-ing kernel choices, disabling a specific
sub-path, or recovering from regressions:

| Env var                     | Default | Effect |
|---|---|---|
| `QW3_PREFILL_ATTN`          | `flashinfer` (FI build) / `mma-gqa-v2` (non-FI build) | Prefill FA kernel. `flashinfer` is the default when built with `-DQW3_ENABLE_FLASHINFER=ON` (FI port of `SinglePrefillWithKVCacheDispatched<HEAD_DIM=256,kCausal>`; beats llama at every T, +28% at T=131K). `mma-gqa-v2` is the default otherwise and the override for FI builds (in-tree FA2 v2). Other choices: `mma-gqa` (v1, 6-head loop), `mma-pipe`, `mma`, `vec`, `cublas`. |
| `QW3_DECODE_ATTN`           | `flashinfer` (FI build) / `native` (non-FI build) | Decode FA kernel. `flashinfer` is the default in FI builds (`SingleDecodeWithKVCacheKernel<HEAD_DIM=256, group_size=6>` — stream-K + MMA at single-token; +18% at T=131K vs in-tree, lifts decode 91.0%→102.7% of llama). `native` (= in-tree `fattn_vec_decode_f16_splitk`) is the only choice in non-FI builds and the override for FI builds. |
| `QW3_PREFILL_FA2_BR`        | `16`    | v2 q-rows-per-CTA: `8`, `16` (default), `32` (parity-correct, regresses 1.5%). |
| `QW3_PREFILL_FA2_BC`        | `32`    | v2 K/V tile width: `32` (default — 2 blocks/SM occupancy), `64`. |
| `QW3_PREFILL_FA2_KCPASYNC`  | `1`     | `0` reverts to sync K loads (dropped +5–7% at long T). |
| `QW3_PREFILL_FA2_VCPASYNC`  | `1`     | `0` reverts to sync V loads (dropped +14% at 65K). |
| `QW3_PREFILL_FA2_KPAD`      | `8`     | v2 K/V shmem row-pitch pad in halves (`0`/`8`). 8 breaks the 8-way LDS bank conflict on K/V reads (+10% at T=65K). `0` reverts; +1 KB shmem cost. |
| `QW3_PREFILL_FA2_SPAD`      | `4`     | v2 s_S/s_P score-tile row-pitch pad in fp32 elems (`0`/`4`). 4 shifts each adjacent score row off the bank anchor on the BC=32 row (+0.2-1.5% across T). `0` reverts; +1 KB shmem cost. |
| `QW3_FATTN_NSPLIT`          | adaptive | Decode-attn split-K: `{1,2,4,8,16,32,64}`. Default policy targets ≈128 KV/split. |
| `QW3_PREFILL_FA2_NSPLIT`    | adaptive | FA2 v2 prefill split-KV: `{1,2,4}`. Default heuristic picks NSPLIT=2 at chunk=512 (under-saturated grid), NSPLIT=1 otherwise. |
| `QW3_FUSE_SILU_MUL`         | `1`     | `0` reverts FFN gate+up+silu+mul to two matvecs + a separate silu_mul. |
| `QW3_FUSE_ADD`              | `1`     | `0` reverts attn_output / ffn_down to plain matvec + separate add. |
| `QW3_NVFP4_SMALL_MMVQ`      | per-route | Global override for both batch-1 NVFP4 routes. `0` selects FlashInfer/CUTLASS; `1` selects packed-FP4 DP4A. The more specific variables below take precedence. Batch 2+ always uses native FP4 MMA. |
| `QW3_NVFP4_SMALL_SINGLE`    | `0`     | Per-projection override. `0` routes batch-1 down projections through the FlashInfer/CUTLASS FP4 Tensor Core path; `1` restores the packed-FP4 DP4A path. |
| `QW3_NVFP4_SMALL_PAIR`      | `1`     | Fallback gate/up override used when a compatible packed CUTLASS pair is unavailable. `1` retains the fused packed-FP4 gate+up/SwiGLU DP4A path. |
| `QW3_NVFP4_FUSED_CUTLASS_PAIR` | `1` | Pack compatible NVFP4 gate/up weights contiguously and issue one native FP4 Tensor Core GEMM followed by SwiGLU. Set `0` to restore the DP4A pair route. |
| `QW3_NVFP4_VLLM_TACTICS`   | `1`     | Use the SM120 batch-1 CUTLASS tactics selected by FlashInfer/vLLM autotuning for Qwen3.6-27B shapes: swapped A/B, 128x32x256 without Stream-K for packed gate/up, and 128x32x128 with Stream-K for down projections. |
| `QW3_NVFP4_BF16_MAIN`      | `1`     | Store the single-request NVFP4 model's hidden/residual stream and final FP8/NVFP4 projection outputs in BF16. RMSNorm reductions, attention, DeltaNet state, and logits remain FP32. Set `0` for the former all-FP32 activation path. Continuous batching currently keeps the FP32 path. |
| `QW3_Q8_BF16_MAIN`         | `1`     | Store the single-request Q8 model's hidden/residual stream and final Q8 projection outputs in BF16 while keeping linear accumulation, RMSNorm reductions, attention, DeltaNet state, and logits in FP32. Set `0` for the former all-FP32 activation path. Continuous batching currently keeps the FP32 path. |
| `QW3_NVFP4_FLASHINFER_RMS_QUANT` | `1` when built | Use the optional SM120 FlashInfer CuTe-DSL AOT kernel for fused BF16 RMSNorm and NVFP4 activation quantization. Set `0` to restore the built-in kernel. |
| `QW3_NVFP4_FLASHINFER_RMS_MIN_BATCH` | `256` | Minimum prefill batch size for the optional fused FlashInfer RMSNorm/quantization route. |
| `QW3_NVFP4_CUTLASS_OUTPUT_BF16` | `0` | Experimental BF16 CUTLASS epilogue for packed gate/up. It is disabled because the additional conversion/SwiGLU path is slower than the FP32 epilogue on SM120. |
| `QW3_FUSED_QK_NORM_ROPE`    | `1`     | Fuse decode-time Q/K per-head RMSNorm and partial RoPE into one CUDA launch on full-attention layers. Set `0` to restore four separate launches. |
| `QW3_FUSED_RECURRENT_QK_NORM` | `1`   | Fuse recurrent decode Q/K L2 normalization into one CUDA launch per DeltaNet layer. Set `0` to restore separate Q and K launches. |
| `QW3_FP8_SMALL_CUBLASLT`    | `1`     | `1` routes FP8 batch 1-8 linears through native cuBLASLt FP8 Tensor Cores, caches per-shape plans, shares activation quantization across fanout projections, and fuses SwiGLU/output scaling/residual add. |
| `QW3_FP8_PREFILL_PACKED_SWIGLU` | `1` | Use one packed cuBLASLt FP8 GEMM plus fused output scaling/SwiGLU for compatible gate/up projections at prefill batch sizes above 8. Set `0` to restore separate gate/up projections. |
| `QW3_GDN_SM120_FUSED_IO` | `1` | Fuse recurrent Q/K L2 normalization into the SM120 BF16 Q/K/V pack and fuse the BF16 GDN output conversion with RMSNorm/gating. FP32 recurrent state and checkpoint paths are unchanged. Set `0` to restore separate normalization, pack, unpack, and gate kernels. |
| `QW3_FP8_LAZY_RECURRENT_PAIR` | `1` | Keep a compatible recurrent FP8 QKV/gate pair in packed GEMM storage and apply its token/channel scales inside the causal convolution and final gate consumer. Unsupported, checkpoint, and DeltaNet-capture paths materialize the original FP32 outputs. |
| `QW3_FP8_LAZY_ATTENTION_FANOUT` | `1` | With FP8 KV cache, defer full-attention Q/K/V output scaling: Q/K normalize directly from scaled raw GEMM views and V scales directly into the FP8 KV cache. Other KV dtypes retain materialized FP32 projections. |
| `QW3_FP8_MIXED_FANOUT`      | `1`     | Share FP8 activation quantization and BF16 input conversion inside mixed FP8/BF16 fanouts such as Qwen 3.6 recurrent projections. Set `0` for the legacy per-projection dispatch. |
| `QW3_FP8_FANOUT_SCALE_FUSION` | `1`   | Combine per-row output scaling for 2-4 FP8 projections sharing one activation into one CUDA kernel. Set `0` to restore one scaling launch per projection. |
| `QW3_FP8_CUTLASS_BF16` | `1` | Use the SM120 CUTLASS FP8 GEMM epilogue to apply activation/weight scales and write BF16 directly for compatible large prefill projections. Set `0` to restore cuBLASLt plus a separate scaling kernel. |
| `QW3_FP8_FUSED_RMS_QUANT` | `1` | Fuse BF16 RMSNorm with dynamic FP8 activation quantization for large attention and recurrent projection fanouts. Mixed recurrent fanouts also emit their shared BF16 activation in the same pass. Set `0` to restore separate RMSNorm, FP8 quantization, and FP32-to-BF16 kernels. |
| `QW3_FP8_FUSED_RMS_THREADS` | `512` | Thread count for the fused RMSNorm/FP8 quantization kernel: `256`, `512`, or `1024`. SM120 profiling selects 512 by default. |
| `QW3_FP8_PACKED_PAIR` | `1` | Pack compatible FP8 projection pairs (FFN gate/up and recurrent QKV/gate) contiguously at model load so one cuBLASLt GEMM can produce both outputs. This does not increase steady-state model memory; set `0` to keep separate allocations and GEMMs. |
| `QW3_FP8_DIRECT_VEC2` | `1` | Convert two adjacent E4M3 values per instruction in direct FP8-weight/F32-activation decode GEMVs. Set `0` to use scalar FP8 conversion. |
| `QW3_FP8_DIRECT_PAIR` | `1` | Fuse compatible small FP8 projection pairs such as attention K/V into one direct decode launch. Set `0` to launch each projection separately. |
| `QW3_FP8_DIRECT_THREADS` | `256` | Threads per block for direct vec2 FP8 decode GEMVs. Supported values: 128, 256, 512. |
| `QW3_FP8_DIRECT_PAIR_THREADS` | `256` | Threads per block for fused direct FP8 projection pairs. Supported values: 128, 256, 512. |
| `QW3_FP8_ATTN_Q_DIRECT` | `1` | Route the large Q projection through the direct FP8 decode GEMV when a three-way attention fanout also contains a prepared direct K/V pair. This shape-specific route does not affect recurrent QKV/gate fanouts. |
| `QW3_FP8_ATTN_Q_DIRECT_MAX_ROWS` | `16384` | Maximum Q-projection row count eligible for the shape-specific direct attention route. |
| `QW3_FP8_ATTN_Q_DIRECT_THREADS` | `128` | Threads per block for the direct attention-Q FP8 GEMV. Supported values: 128, 256, 512. |
| `QW3_FP8_FUSED_BF16_QUANT` | `1` | In batch-1 mixed FP8/BF16 fanouts, stage the BF16 activation while the FP8 dynamic-quantization kernel scans the shared FP32 input. Set `0` to restore a separate FP32-to-BF16 conversion launch. Batch 2+ keeps the separate conversion path, which is faster at those shapes. |
| `QW3_FP8_FANOUT_CUSTOM_MAX_ROWS` | `2048` | In batch-1 FP8 fanouts, use the direct FP8-weight/F32-activation GEMV for projections at or below this row count while larger projections stay on cuBLASLt. Set `0` to route every projection through cuBLASLt. |
| `QW3_FP8_MATVEC_ADD_CUSTOM_MAX_COLS` | `8192` | Use the fused direct FP8-weight/F32-activation matvec+residual kernel for batch-1 output projections at or below this input width. Set `0` to retain dynamic activation quantization plus cuBLASLt. |
| `QW3_BF16_SMALL_PAIR_MAX_ROWS` | `128` | Fuse a pair of same-shape batch-1 BF16 projections into one direct kernel at or below this row count, reusing the shared BF16 activation. Set `0` to retain two cuBLAS calls. |
| `QW3_FP8_CUBLASLT_OUTER_SCALE` | `0`  | Experimental cuBLASLt outer-vector scaling that applies activation and weight scales in the GEMM. It removes post-scale kernels but is slower for current batch-1 SM120 shapes. |
| `QW3_GRAPH`                 | `0`     | Experimental per-token CUDA graph capture/update for non-paged decode. Enable with `1`; it is not the persistent static graph used by vLLM and is slower on the current NVFP4 path. |
| `QW3_HGEMM_X_CACHE`         | `1`     | `0` disables FP16 input reuse across consecutive HGEMMs sharing an input. |
| `QW3_KV_DTYPE`              | `fp16` | KV cache dtype for env-driven paths. `fp16` is the baseline/default, `fp8` uses raw e4m3 KV cache storage, `fp32` is parity-only, and `q8` is experimental. CLI `--kv-dtype` overrides this for service/local runs that expose the flag. |
| `QW3_MATMUL`                | `auto` CLI / `mmq` serve | Per-call: batch ≥ 8 → MMQ (v8 at batch ≥ 128 for 128×128 tile, v7 below for 64×64 + occupancy). `hgemm` forces cuBLAS HGEMM with FP16 dequant scratch (uses ~3 GiB extra at chunk=2048); `mmq` forces MMQ unconditionally. |
| `QW3_MMQ_VERSION`           | `auto`  | MMQ kernel variant (`auto`/2/3/4/5/6/7/8). Auto picks v8 (rows ≥ 128 & batch ≥ 128) else v7. Both are split-plane Q8 + 144-B Q8_1_MMQ activations + XOR-swizzled shmem; v8 is 128×128 tile (1 block/SM), v7 is 64×64 (2 blocks/SM). |
| `QW3_PREFILL_CHUNK`         | `2048`  | Chunk size for prefill batches. `0` disables chunking entirely (peak throughput, peak scratch). The CLI flag `--prefill-chunk N` / `--no-prefill-chunk` overrides this when set. |
| `QW3_CONTINUOUS_BATCHING`   | `0` | Internal bridge for the OpenAI-compatible continuous batching scheduler and backend-owned global KV page pool. Use `qw3 serve --continuous-batching` instead. |
| `QW3_CONTINUOUS_BATCHING_BODY_BATCH` | `1` when continuous batching is enabled | Enable the optimized greedy body-batch executor for continuous batching. |
| `QW3_CONTINUOUS_BATCHING_RECURRENT_BATCH` | `1` when body-batch is enabled | Use the independent-state recurrent batch path inside the body-batch executor. Set `0` to fall back to per-request recurrent layers. |
| `QW3_CONTINUOUS_BATCHING_MAX_ACTIVE` | service default | Maximum active continuous requests admitted to the decode loop. |
| `QW3_CONTINUOUS_BATCHING_MAX_PENDING` | `128` | Maximum queued continuous requests before admission returns an error. |
| `QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS` | `ctx * max_active` | Token reservation budget. Set `0` to disable this budget. |
| `QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES` | `ceil(ctx / page_size)` | Number of physical pages in the backend-owned global KV pool. |
| `QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES` | same as KV pool | Number of physical pages in the separate global MTP prefix KV pool. |
| `QW3_PAGED_KV_PAGE_SIZE`    | `16`    | Logical/physical page size in tokens for paged KV. |
| `QW3_MTP_PAGED_PREFIX`      | `0` | Use paged MTP prefix KV append/attention paths. `qw3 serve --continuous-batching --mtp-chain N` enables this internally when paged KV is active. |
| `QW3_CONTINUOUS_MTP_BATCHED_DRAFT` | `0` | Batch MTP draft projection/FFN/logits across continuous requests. `qw3 serve --continuous-batching --mtp-chain N` enables this internally. Keeps per-row paged MTP attention for stability. |

## KVMem

KVMem treats the KV cache as a block-structured memory system. It can keep a
selected working set on GPU while staging cold blocks to CPU memory and, when
configured, to an NVMe backing file. The default path is unchanged unless
`--kvmem` is passed.

Recommended manual service test:

```sh
QW3_KVMEM_TIER_TRACE=1 ./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx 65536 \
  --temp 0 \
  --top-k 1 \
  --kvmem \
  --kvmem-block-tokens 128 \
  --kvmem-budget 8192 \
  --kvmem-sink-blocks 1 \
  --kvmem-recent-blocks 8 \
  --kvmem-update-mode step \
  --kvmem-method retrieval \
  --kvmem-retrieval-method mean_attention \
  --kvmem-gpu-memory-ratio 0.50 \
  --kvmem-cpu-gb 16
```

Add an NVMe tier when CPU memory is not enough:

```sh
mkdir -p /tmp/qw3_kvmem_nvme

QW3_KVMEM_TIER_TRACE=1 ./build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx 65536 \
  --temp 0 \
  --top-k 1 \
  --kvmem \
  --kvmem-block-tokens 128 \
  --kvmem-budget 8192 \
  --kvmem-sink-blocks 1 \
  --kvmem-recent-blocks 8 \
  --kvmem-update-mode step \
  --kvmem-method retrieval \
  --kvmem-retrieval-method mean_attention \
  --kvmem-gpu-memory-ratio 0.50 \
  --kvmem-cpu-gb 16 \
  --kvmem-nvme-dir /tmp/qw3_kvmem_nvme \
  --kvmem-nvme-gb 64
```

KVMem CLI parameters:

| Parameter | Default | Description |
|---|---:|---|
| `--kvmem` | off | Enable KVMem block-sparse KV attention. Without this flag the normal forward path is used. |
| `--kvmem-block-tokens N` | `128` | KV block granularity in tokens. Must be a positive multiple of the KV page size. |
| `--kvmem-budget N` | `131072` | Maximum selected working-set window per selection, in tokens. Approximate selected block count is `budget / block_tokens`. |
| `--kvmem-sink-blocks N` | `1` | Always keep the first N blocks for attention-sink behavior. |
| `--kvmem-recent-blocks N` | `0` | Always keep the most recent N blocks. `0` lets KVMem derive the recent allocation. |
| `--kvmem-method M` | `retrieval` | Selection signal: `retrieval`, `h2o`, or `recency`. |
| `--kvmem-retrieval-method M` | `mean_attention` | Retrieval scorer: `mean_attention` or `content_mean`. |
| `--kvmem-query-conditioned` | off | Score blocks by the multi-token mean of the final user message (the question) against each block's mean-k, instead of falling back to a recency window. Default-OFF, byte-identical when unset. Required for the LongMemEval benchmark below. |
| `--kvmem-select-policy M` | `topk` | Selection policy: `topk` or `quota`. |
| `--kvmem-retrieval-blocks N` | `0` | Quota-policy retrieval block count. `0` derives from remaining budget. |
| `--kvmem-profile-blocks N` | `0` | Quota-policy profile block count. `0` derives from remaining budget. |
| `--kvmem-update-mode M` | `interval` | Reselect cadence: `interval` or `step`. `step` selects after prefill and does not automatically reselect during decode. |
| `--kvmem-interval N` | `64` | Decode tokens between reselections when `--kvmem-update-mode interval` is used. |
| `--kvmem-gpu-memory-ratio F` | `0.50` | Approximate fraction of GPU memory that KVMem may use for its bounded GPU KV pool. |
| `--kvmem-gpu-high-watermark F` | `0.95` | GPU tier high-watermark knob reserved for tiering policy. Leave at default for normal tests. |
| `--kvmem-gpu-low-watermark F` | `0.85` | GPU tier low-watermark knob reserved for tiering policy. Leave at default for normal tests. |
| `--kvmem-cpu-gb F` | `0` | CPU tier budget in GiB. `0` disables CPU tier runtime page release. |
| `--kvmem-cpu-bytes N` | `0` | Legacy byte-level CPU tier budget. Prefer `--kvmem-cpu-gb` for manual runs. |
| `--kvmem-nvme-dir DIR` | unset | Directory for the NVMe backing file. Required when NVMe tier budget is nonzero. |
| `--kvmem-nvme-gb F` | `0` | NVMe tier budget in GiB. Requires `--kvmem-nvme-dir`. |
| `--kvmem-nvme-bytes N` | `0` | Legacy byte-level NVMe tier budget. Prefer `--kvmem-nvme-gb` for manual runs. |

`--kvmem-cpu-gb` and `--kvmem-nvme-gb` use GiB units
(`1 GiB = 1024^3 bytes`). The older `--kvmem-cpu-bytes` and
`--kvmem-nvme-bytes` flags remain available for scripts and exact-byte tests.

Useful KVMem environment variables:

| Variable | Description |
|---|---|
| `QW3_KVMEM_TIER_TRACE=1` | Print tier events such as `stage_out`, `stage_in`, `cpu_evict`, `stage_in_async_read`, and `bounded_gpu_pool`. Recommended when testing CPU/NVMe offload. |
| `QW3_KVMEM_TRACE=1` | Print selection/retrieval diagnostics. This can be verbose. |
| `QW3_KVMEM_ATTN_TRACE=/path/to/file.jsonl` | Dump KVMem attention-mass traces for analysis. This is expensive and should not be enabled for normal serving. |
| `QW3_KVMEM_ATTN_TRACE_INTERVAL=N` | Sampling interval for `QW3_KVMEM_ATTN_TRACE`; default is every token. |
| `QW3_KVMEM_QC_SOFTMAX=1` | Query-conditioned scorer: use softmax-over-pages (accumulated attention mass per block) instead of the default sum-of-ReLU. A/B knob; does not change which blocks survive the budget cut (see the retrieval finding below). |
| `QW3_KVMEM_QC_SINGLE_LAYER=1` | Score using only the first normal-attention layer's mean-k instead of all 16. Apples-to-apples comparison knob for the multi-layer scorer. |
| `QW3_KVMEM_QC_LAYERS=N` | Cap the number of normal-attention layers used by the query-conditioned scorer (debug). |
| `QW3_KVMEM_NO_REROPE=1` | Skip the position-collapse re-RoPE in window assembly; selected blocks keep their true-position rotation. Diagnostic only — run with MTP off (the MTP draft position site is not rewired). |

Compatibility notes:

- Single-request KVMem + MTP is supported.
- KVMem + continuous batching is supported without MTP.
- KVMem + continuous batching + MTP is guarded with a hard error in the
  current implementation, because the continuous batching MTP verifier is not
  fully KVMem-window-aware yet.
- For fast correctness checks, prefer short deterministic requests
  (`--temp 0 --top-k 1`) and compare KVMem lossless or short decode output
  against baseline before running full agent/SWE evaluations.

### LongMemEval-S utility benchmark

The external Python harness in `scripts/kvmem_eval/` measures KVMem retrieval
accuracy on a 102-sample LongMemEval-S subset (17 per question type × 6 types:
`single-session-{user,assistant,preference}`, `multi-session`,
`temporal-reasoning`, `knowledge-update`). It hits `qw3 serve` **only** through the
OpenAI HTTP API; grading uses a DeepSeek answer-equivalence judge
(`evaluate_qa_deepseek.py`). No eval logic lives in qw3 C++.

Data: build the subset from the public `xiaowu0162/longmemeval` dataset
(`scripts/kvmem_eval/dataset.py`), or point `--data` at your own JSONL with the same
schema. Histories are ~110K tokens, so the data file is large (~50 MB) and is not
committed to the repo.

1. Serve with KVMem query-conditioned selection:

```sh
build/qw3 serve \
  --model models/Qwen3.6-27B-Q8_0.gguf \
  --ctx 163840 --kv-dtype fp16 \
  --kvmem --kvmem-block-tokens 1024 --kvmem-budget 32768 \
  --kvmem-method retrieval --kvmem-retrieval-method content_mean \
  --kvmem-update-mode step --kvmem-query-conditioned \
  --kvmem-gpu-memory-ratio 0.5 \
  --kvmem-cpu-gb 64 --kvmem-nvme-gb 256 --kvmem-nvme-dir /data/qw3_kvmem_eval_nvme \
  --prefill-chunk 2048 --enable-thinking --temp 0.6 \
  --native-mtp-speculate --mtp-chain 4 --port 8080
```

2. Run the harness (`--use-all` runs the provided 102-subset in file order;
   `--no-judge` records model answers without inline grading; `--indices` runs a
   subset such as the multi-session bucket):

```sh
python3 scripts/kvmem_eval/run_eval.py --use-all --no-judge \
  --data selected_12_samples.jsonl --tag kvmem_run
# e.g. multi-session only: --indices 6,7,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71
```

Outputs land in `/data/chaidi/kvmem_eval/results/<tag>_eval_<ts>{,_hyp,_summary}.jsonl`.

3. Grade with DeepSeek (API key from the `DEEPSEEK_API_KEY` env var **only** — never
   hardcode or commit it):

```sh
DEEPSEEK_API_KEY=... python3 evaluate_qa_deepseek.py \
  --hyp-file /data/chaidi/kvmem_eval/results/<tag>_eval_<ts>_hyp.jsonl \
  --ref-file selected_12_samples.jsonl \
  --output   /data/chaidi/kvmem_eval/results/<tag>_eval_<ts>_graded.jsonl
```

Baselines for comparison:

- **Full Context** (accuracy upper bound): the same command with **no** `--kvmem*`
  flags (plain `--ctx 163840`).
- **budget=all diagnostic** (isolates retrieval vs post-retrieval): keep every flag
  but set `--kvmem-budget 163840 --kvmem-gpu-memory-ratio 0.9` so all ~110 history
  blocks stay GPU-resident (verify with `QW3_KVMEM_TRACE=1`: tier usage shows
  `gpu_used == gpu_cap`, cpu/nvme = 0).

> **Known result (2026-06-30): retrieval selection is the accuracy bottleneck, not
> the post-retrieval path.** On the worst (multi-session) bucket the default 32-block
> budget scores ~12%, but budget=all scores ~82% — matching Full Context. So
> assembly, re-RoPE, windowed attention, and generation are lossless; the
> `content_mean` scorer simply ranks the answer-bearing block too low to survive a
> tight budget. Changing the scoring function (`QW3_KVMEM_QC_SOFTMAX`) or position
> handling (`QW3_KVMEM_NO_REROPE`) does **not** help, because neither changes which
> blocks are kept. The open lever is the ranking signal and/or the budget. See
> `docs/kvmem_utility_eval_plan.md` for the full A/B table.

## Backends

| Backend       | When to use it |
|---|---|
| `qwen-native` | Real inference. Owns GGUF loading, tokenizer, CUDA execution. |
| `mock`        | CI / build sanity. No weights needed. |
| `llama-cli`   | Forward to an external `llama-completion`. Reference only; not the optimization target. |

## Benchmark against llama.cpp

For stable numbers, use **`scripts/long_prompt_sweep.py`** — the comparison
tool the Status section above is built from. It alternates qw3 ↔ llama.cpp
trial-by-trial to spread thermal drift, sweeps over a configurable list of
prompt lengths, and reports per-cell median tok/s with prefill and decode
ratios:

```sh
python3 scripts/long_prompt_sweep.py \
    --prompt-tokens "512 1024 2048 4096" --trials 3 -n 64 \
    --json /tmp/sweep.json
```

For ad-hoc single-prompt comparisons, `scripts/compare_with_llama_cpp.py`
(driven by `scripts/run_compare.sh`) runs a fixed prompt set through both
engines and reports prefill/decode tok/s, common prefix, and first-char
match. Both engines are greedy, so a correct implementation produces
identical token streams.

```sh
# Default prompt set (4 short prompts):
bash scripts/run_compare.sh -n 64

# Add the 1322-token prompt to the default set:
bash scripts/run_compare.sh --long -n 128

# Run only the long prompt:
bash scripts/run_compare.sh --long-only -n 128 --token-diff
```

`--token-diff` re-tokenizes both engines' outputs via qw3 and reports the
common token-level prefix length and whether the full token sequences are
identical.

llama.cpp must be built with `-DGGML_CUDA=ON` for a meaningful comparison;
the script invokes `llama-completion` (not `llama-cli`) for deterministic,
non-interactive execution.

## Logit-level parity diffs

To compare a single prompt token-by-token:

```sh
./build/qw3 \
  --model /path/to/model.gguf \
  -p "Hello" -n 8 \
  --dump-logits qw3.jsonl --dump-logits-top-k 16 --dump-tokens
```

## Layout

```
include/qw3/
  device_backend.hpp  -- CUDA-agnostic tensor / weight / op interface
  qwen_config.hpp     -- Qwen3.5/3.6 hyperparams parsed from GGUF
  tokenizer.hpp       -- byte-level BPE tokenizer
  gguf.hpp / qw3.hpp  -- GGUF reader + engine surface

src/
  kernels_cuda.cu     -- CUDA backend (matvec/HGEMM dispatch, KV/RoPE/RMS, executor glue)
  mmvq_q8.cu          -- Q8_0 × Q8_1 DP4A matvec (decode default)
  mmq_q8.cu           -- Q8_0 INT8-MMA matmul (opt-in via QW3_MATMUL=mmq)
  fattn_vec_decode.cu -- Flash-attention decode + FA2 prefill kernels (v1, v2)
  gated_delta_net.cu  -- DeltaNet recurrent prefill kernel
  qwen_executor.cpp   -- forward_one_token (decode), forward_n_tokens (prefill)
  qwen_weights.cpp    -- device-resident weight uploads, kept across calls
  qwen_native_backend.cpp -- prompt formatting, generate(), perf logging
  qwen_config.cpp     -- GGUF -> QwenConfig
  tokenizer.cpp       -- BPE / pre-tokenization / special tokens
  qw3_cli.cpp         -- CLI entry point
```

The replacement points for further optimization are
`include/qw3/device_backend.hpp` (kernels) and `src/qwen_executor.cpp`
(layer logic / scratch). See `docs/architecture.md` for the long-form
description.

## Development history & roadmap

For the optimization journey, profiles, abandoned attacks, and the
priority-ordered list of remaining gaps to llama.cpp, see
[`DEVELOPMENT_LOG.md`](DEVELOPMENT_LOG.md).
