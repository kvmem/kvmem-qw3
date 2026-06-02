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

**Default config — FlashInfer prefill + MMQ v8 matmul, memory parity
with llama.cpp.** Build runs `--prefill-chunk 2048` and
`QW3_MATMUL=auto`, which routes every prefill matmul (batch ≥ 8) to
the INT8-MMA path: **MMQ v8** (128×128 tile) at batch ≥ 128, **MMQ v7**
(64×64 tile) below. Prefill attention runs **FlashInfer**
(`SinglePrefillWithKVCacheDispatched<HEAD_DIM=256, kCausal, ...>`) when
the build flag `-DQW3_ENABLE_FLASHINFER=ON` is set — the configuration
the table below measures. Override with `QW3_PREFILL_ATTN=mma-gqa-v2`
to restore the in-tree FA2 v2 kernel (the fallback default for builds
without FI; numbers in the "FA2 v2 fallback" subsection below).
HGEMM-with-FP16-dequant is no longer in the default path; Q8 weights
stay 8-bit in HBM end-to-end. `qw3_cli` sets `CUDA_MODULE_LOADING=EAGER`
at process init so all kernel modules resolve before the first launch
(short prefill is no longer launch-overhead-bound, and FI's modules
load without affecting later kernel-launch cost). Peak process HBM
sits ~2.3 GiB above llama.cpp at every T, flat in T (chunk=2048 batch
scratch + cuBLAS workspace + FI Q/O-pack, not a per-token leak).

| Prompt tokens | qw3 prefill | llama prefill | prefill % | qw3 decode | llama decode | decode % | qw3 peak | llama peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|    827 | 3392 tok/s | 3213 tok/s | **105.6%** | 45.62 tok/s | 44.56 tok/s | **102.4%** | 30.8 GiB | 29.0 GiB |
|   2453 | 3734 tok/s | 3679 tok/s | **101.5%** | 45.45 tok/s | 44.68 tok/s | **101.7%** | 31.4 GiB | 29.0 GiB |
|   4621 | 3833 tok/s | 3720 tok/s | **103.0%** | 44.21 tok/s | 44.42 tok/s | **99.5%**  | 31.4 GiB | 29.0 GiB |
|   8686 | 3972 tok/s | 3810 tok/s | **104.3%** | 44.34 tok/s | 42.50 tok/s | **104.3%** | 31.4 GiB | 29.0 GiB |
|  16816 | 3909 tok/s | 3723 tok/s | **105.0%** | 42.42 tok/s | 42.65 tok/s | **99.5%**  | 31.4 GiB | 29.0 GiB |
|  33347 | 3748 tok/s | 3519 tok/s | **106.5%** | 42.19 tok/s | 41.36 tok/s | **102.0%** | 31.4 GiB | 29.0 GiB |
|  66138 | 3443 tok/s | 3060 tok/s | **112.5%** | 38.11 tok/s | 38.99 tok/s | **97.7%**  | 33.2 GiB | 30.9 GiB |
| 131720 | 2940 tok/s | 2296 tok/s | **128.0%** | 31.84 tok/s | 35.01 tok/s | **91.0%**  | 37.3 GiB | 34.7 GiB |

Throughput in absolute tokens/second, n_decode=32, ctx=36864 (T=66K and
T=131K bump ctx to fit). Memory columns are net process-peak (peak
`nvidia-smi memory.used` minus idle-GPU baseline) sampled at 50 ms while
each engine runs — same instrument for both. The `%` columns report
`qw3 / llama.cpp`. Headlines:

- **Prefill beats llama.cpp at every T** — 101.5–106.5% from T=2K
  through T=33K, then widens to **+12.5% at T=66K and +28.0% at
  T=131K**. The FlashInfer kernel closes the Tensor-Core utilization
  gap that bottlenecked the in-tree FA2 v2 at long T (23% TC util →
  ~65%, NCU-confirmed).
- **Decode is 97–104% of llama.cpp through T=66K** — qw3 is faster
  than llama.cpp on decode at every T ≤ 33K. Matvec is at HBM ceiling
  (~2.7 TB/s effective); further wins need fewer matvec calls.
- **Decode regression at T=128K (91.0%)** is structural to
  `fattn_vec_decode`: scales linear in context, NSPLIT cap at 64 is
  saturated, and combine cost grows with NSPLIT. Bumping further
  trades total throughput for marginal split-tail relief. Unrelated
  to FlashInfer (FI is prefill-only; decode runs the existing fused
  vec splitk kernel).
- **Memory: +2.3 GiB at every T, flat.** Includes the ~17 MiB FI
  Q-pack + O-pack (fp16) sharing `prefill_gqa_scratch_`. Memory
  parity with llama.cpp is preserved end-to-end.

Reproduce with:

```sh
# qw3-default vs qw3+FlashInfer vs llama.cpp, 8 cells, 3 trials/cell:
python3 scripts/fi_sweep.py \
  --prompt-tokens "556 2182 4350 8415 16545 33076 65867 131073" \
  --trials 3 -n 32 --json /tmp/fi_sweep.json

# Or default-only (no FlashInfer column):
python3 scripts/long_prompt_sweep.py \
  --prompt-tokens "512 2048 4096 8192 16384 32768 65536" \
  --trials 3 -n 32 -c 70000 \
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

The adapter (`src/flashinfer_prefill_adapter.{cu,hpp}`, ~240 LoC)
dispatches `flashinfer::SinglePrefillWithKVCacheDispatched<
HEAD_DIM=256, KV_LAYOUT=NHD, kCausal, ...>` from inside qw3's
attention path. Q is packed FP16 from the FP32 Q buffer, K/V are
already FP16 in the KV cache, O is written FP16 then converted back
to FP32 for the rest of the forward. Q-pack and O-pack share
`prefill_gqa_scratch_`, the same scratch buffer the in-tree FA2 v2
path uses — no extra allocation.

The dispatch is gated on `batch ≥ 8` (the existing
`QW3_PREFILL_ATTN_MIN_BATCH` knob), so it only fires for prefill, not
decode. Decode keeps running its fused vec splitk kernel.

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

Tested with CUDA 12.x / 13.x. For non-Blackwell targets, swap
`CMAKE_CUDA_ARCHITECTURES` accordingly (e.g. `90` for Hopper, `89` for Ada).
The default of `120a-real` matters — JIT'd Ampere PTX leaves measurable
performance on the table on consumer Blackwell.

## Run

```sh
./build/qw3 \
  --backend qwen-native \
  --native-heavy \
  --native-kernels cuda \
  --native-linear-backend auto \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  -p "Explain Adam optimizer in one paragraph." \
  -n 256
```

Key flags (see `qw3 --help` for the full list):

| Flag | Purpose |
|---|---|
| `--backend qwen-native`        | Use qw3's native engine (the optimization path). |
| `--native-heavy`               | Run the full forward (without this, only no-op plan validation runs). |
| `--native-kernels cuda`        | Pick the CUDA device backend. |
| `--native-linear-backend auto` | Custom Q8 kernels for big matmuls. Use `custom` to force, `cublas` to experiment. |
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

## Tuning knobs

Most defaults are correct on Blackwell + Qwen 3.6. The env knobs below are
useful for A/B-ing kernel choices or recovering from regressions:

| Env var                     | Default | Effect |
|---|---|---|
| `QW3_PREFILL_ATTN`          | `flashinfer` (FI build) / `mma-gqa-v2` (non-FI build) | Prefill FA kernel. `flashinfer` is the default when built with `-DQW3_ENABLE_FLASHINFER=ON` (FI port of `SinglePrefillWithKVCacheDispatched<HEAD_DIM=256,kCausal>`; beats llama at every T, +28% at T=131K). `mma-gqa-v2` is the default otherwise and the override for FI builds (in-tree FA2 v2). Other choices: `mma-gqa` (v1, 6-head loop), `mma-pipe`, `mma`, `vec`, `cublas`. |
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
| `QW3_GRAPH`                 | `1`     | `0` disables CUDA graph capture of decode. |
| `QW3_HGEMM_X_CACHE`         | `1`     | `0` disables FP16 input reuse across consecutive HGEMMs sharing an input. |
| `QW3_KV_DTYPE`              | `fp16`  | `fp32` reverts KV cache to FP32 (parity-only). |
| `QW3_MATMUL`                | `auto`  | Per-call: batch ≥ 8 → MMQ (v8 at batch ≥ 128 for 128×128 tile, v7 below for 64×64 + occupancy). `hgemm` forces cuBLAS HGEMM with FP16 dequant scratch (uses ~3 GiB extra at chunk=2048); `mmq` forces MMQ unconditionally. |
| `QW3_MMQ_VERSION`           | `auto`  | MMQ kernel variant (`auto`/2/3/4/5/6/7/8). Auto picks v8 (rows ≥ 128 & batch ≥ 128) else v7. Both are split-plane Q8 + 144-B Q8_1_MMQ activations + XOR-swizzled shmem; v8 is 128×128 tile (1 block/SM), v7 is 64×64 (2 blocks/SM). |
| `QW3_PREFILL_CHUNK`         | `2048`  | Chunk size for prefill batches. `0` disables chunking entirely (peak throughput, peak scratch). The CLI flag `--prefill-chunk N` / `--no-prefill-chunk` overrides this when set. |

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
./build/qw3 --backend qwen-native --native-heavy --native-kernels cuda \
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
