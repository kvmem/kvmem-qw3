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
Short prompts (< ~100 tokens) make prefill numbers dominated by per-process
startup overhead — both qw3 and llama.cpp pay fixed costs (model load, CUDA
context, kernel cache warmup, lazy init), and qw3 even charges the first
prefill with one-shot kernel selection. Comparisons below run a sweep across
{512, 1024, 2048, 4096}-token prompts with 3 trials per cell, alternating
qw3 ↔ llama.cpp to spread thermal drift, and report the median.

Reproduce with:

```sh
python3 scripts/long_prompt_sweep.py --prompt-tokens "512 1024 2048 4096" \
    --trials 3 -n 64 --json /tmp/sweep.json
```

| Prompt tokens | qw3 prefill | llama prefill | prefill Δ | qw3 decode | llama decode | decode Δ |
|---:|---:|---:|---:|---:|---:|---:|
|  556 | 1885.3 tok/s | 2772.5 tok/s | **−887.2 tok/s** | 39.3 tok/s | 45.6 tok/s | **−6.3 tok/s** |
| 1098 | 2522.5 tok/s | 3324.8 tok/s | **−802.3 tok/s** | 40.2 tok/s | 45.8 tok/s | **−5.5 tok/s** |
| 2182 | 2883.7 tok/s | 3601.1 tok/s | **−717.3 tok/s** | 40.5 tok/s | 45.4 tok/s | **−4.9 tok/s** |

All numbers are absolute tokens/second. The Δ column reports `qw3 − llama.cpp`
(negative = llama.cpp is faster). Two trends jump out:

- **Decode is flat across context length** (~39–41 tok/s from 550 → 2200
  tokens) and trails llama.cpp by a constant ~5 tok/s. Earlier the
  decode rate fell from 37 → 22 tok/s as KV grew, because per-token
  attention bandwidth scaled with context. Two changes flipped that: F16
  KV cache (halves K/V bytes) and **split-K attention**
  (`fattn_vec_decode_kernel` now parallelizes over KV slices via
  `NSPLIT ∈ {1,2,4,8}` plus an online-softmax combine kernel — the
  original kernel only used `n_heads × 4` warps, leaving most of
  Blackwell's SMs idle). CUDA graph capture took launch overhead off
  the decode hot path; FFN gate+up matvec fusion shaved another ~1%.
  The remaining gap is pure Q8 matvec wall time; see roadmap.
- **Prefill trails llama.cpp by ~700–900 tok/s** at 1K–2K. The dominant
  cost on llama.cpp's prefill is `mul_mat_q` (INT8 MMA on raw Q8_0)
  at 64% of GPU time — they bypass dequant entirely. qw3 currently runs
  prefill matmul through cuBLAS HGEMM with on-the-fly Q8 → FP16 dequant
  (~12% of prefill GPU time goes to dequant + FP32→FP16 staging). A
  4-warp cooperative MMQ port that beats HGEMM is the next prefill
  attack — see [`feedback_mmq_int8_mma_q8_prefill.md`](./feedback)
  for the prior single-warp attempt that was 3× slower and the analysis
  of why.

How we got here (prefill on a 1322-token prompt, earlier baseline measurements):

| Stage | Prefill (tok/s) | Δ vs llama.cpp (tok/s) |
|---|---:|---:|
| Baseline (dp4a matvec, batched over T) |   100.6 | −3300 |
| + cuBLAS HGEMM (FP16 dequant cache + tensor cores) |  602.0 | −2800 |
| + Fused batched recurrent (1 launch / layer / sub-op) | 1031.6 | −2270 |
| + Ported MMVQ (Q8_1 activation) + flash-attention-vec decode | 1539 | −1755 |
| + F16 KV cache + split-K decode attention | 1590 | −1700 |
| + Eager Q8 → FP16 weight prewarm at upload time (1098-tok prompt) | 2875 | **−445** |

(llama.cpp on the 1098-token prompt is ~3320 tok/s on this GPU.)

The split-K decode change moved the needle on **decode throughput** at long
context (4350 tokens: 22 → 37 tok/s, +68%) while leaving prefill ~unchanged
— prefill goes through HGEMM, not the per-token fattn-vec path. The eager
prewarm pulled the one-shot Q8 → FP16 dequant out of the prefill timing
window: same total work, just paid during model load (3.6 → 4.1 s) where
it folds into existing H2D-copy time, instead of biting the first prefill.

## Bottlenecks and roadmap

`nsys` on the 2305-token prompt + 64-token decode (1.8 s wall) gives the
new picture:

| Kernel | % wall | Total time | Calls |
|---|---:|---:|---:|
| `mul_mat_vec_q8_0` (decode matvec, all linears) | **45.8%** | 1.18 s | 31312 |
| `q8_dequant_f16` (per-call weight unpack into shared scratch) | 13.4% | 0.35 s | 498 |
| CUTLASS HGEMM 256×128 (prefill linears) | 12.2% | 0.32 s | 400 |
| `fattn_vec_decode_kernel` NSPLIT=4 (decode attention) | 6.8% | 0.18 s | 1008 |
| `fattn_vec_decode_kernel` NSPLIT=1 (prefill attention) | 5.4% | 0.14 s | 16 |
| RMSNorm | 4.2% | 0.11 s | 8256 |
| `gated_delta_net` (DeltaNet recurrent) | 3.4% | 0.09 s | 48 |

The decode-attention shift was dramatic: it used to be **47.5%** of wall
(the #1 bottleneck) and is now **6.8%**. Per-call attention dropped from
1.38 ms to 0.18 ms because NSPLIT=4 multiplies grid occupancy by 4× — 24
heads × 4 splits = 96 blocks of 4 warps = 384 warps, well above what
Blackwell's 128 SMs need to fill the pipeline. The combine kernel is
1.5 ms total across the whole run (negligible). `q8_dequant_f16` runs
once per HGEMM call into a shared FP16 scratch (sized to the largest
single weight, ~150 MB) — earlier versions kept a persistent per-weight
FP16 mirror but that ~doubled live model memory (54 GB on Qwen 3.6 27B)
and defeated the point of Q8. The current shape pipelines the dequant
on a side stream against the cuBLAS HGEMM via two ping-pong buffers so
short prompts pay only the dequant overhead the side stream cannot hide
(~20% at 401 tokens, near zero by 2K).

The new bottleneck is **decode matvec**: 31312 calls (≈489/token across
48 layers × ~10 linears) at 38 µs each = 18.5 ms/token, ~70% of decode
wall time. Closing the remaining ~7 tok/s decode gap to llama.cpp now
means attacking matvec, not attention. So the next gains will come from:

1. **Tiled flash-attention prefill kernel.** The 4K-prompt deficit
   (−1000 tok/s) is dominated by `fattn_vec_decode_kernel` running with
   batch = T queries; each query's block reads K/V independently from
   HBM with no reuse, so the kernel scales O(T²) at ~580 GB/s effective
   bandwidth. A flash-attention-2 style kernel that tiles BR × BC and
   shares K/V loads across queries should cut prefill attention from
   ~534 ms to ~100 ms on the 4K case (estimated +500 tok/s prefill at
   long context).

2. **MMQ-style prefill (INT8 MMA on raw Q8).** Drops the on-the-fly
   Q8 → FP16 dequant scratch and the FP32 → FP16 activation conversion,
   and uses tensor-core integer MMA (`m16n8k32.s8.s8.s32`) the way
   llama.cpp's `mmq.cuh` does. v1 is in tree (`src/mmq_q8.cu`,
   selectable via `QW3_MATMUL=mmq`) and is parity-correct, but its
   16×32 / single-warp tile runs ~3× slower than cuBLAS HGEMM. To
   beat HGEMM the kernel needs to grow to a 4-warp CTA / 64×64
   output tile with cooperative shmem-staged weight loads and
   `ldmatrix.x4` for the A fragments — same shape as upstream
   `mmq.cuh`. Once that lands, the dequant scratch and ping-pong
   pipeline added to keep prefill fast without a persistent FP16
   weight mirror go away entirely, and the short-prompt prefill
   gap (−20% at 401 tokens vs the old mirror path) closes too.

3. **Persistent activation Q8_1 buffer reuse across Q/K/V/gate/up.** *(Done — for reference)*
   Decode used to re-quantize the input to Q8_1 once per matvec; the
   `q8_0_matvec_fanout` path now hoists the quantization across the
   3+ matvecs that share an input (recurrent QKVA/B, attention QKV,
   FFN gate+up) and reuses the staging buffer. Decode +2.5%; prefill
   path unchanged.

4. **CUDA graph capture of the decode loop.** *(Done)* Every
   `forward_one_token` after a one-token warmup is captured into a
   `cudaGraph_t` and replayed via `cudaGraphLaunch`. The position
   counter that drops into `kv_append` / RoPE / `attention_decode`
   is handled by re-recording each token and patching the exec via
   `cudaGraphExecUpdate` (re-instantiating on
   `cudaErrorGraphExecUpdateFailure`), the same shape llama.cpp
   uses. The argmax tail is split into `argmax_launch` (kernel +
   async D2H, recorded into the graph) and `argmax_collect` (sync +
   read pinned mirror, after replay). Yield: +3.3% decode at
   2453-token prompt (38.34 → 39.59 tok/s, n_decode=256). Profile
   afterwards shows decode is no longer host-launch-bound:
   `mul_mat_vec_q8_0_kernel<1>` is now **76.9% of decode GPU time**
   (38 µs median × 13 420 calls), so the next decode lever is the
   matvec kernel itself, not launch overhead. Gated by `QW3_GRAPH=off`
   for A/B.

5. **DeltaNet recurrent fusion.** 3.4% on long prompts; the four kernels
   per layer (conv, l2, deltanet, norm-gate) are each light but sequential.
   A single fused launch per layer is mostly a launch-latency win (small,
   but easy and stacks with #4).

6. **(Done — for reference)** F16 KV cache and split-K attention. Both
   shipped above; together they took decode at 4350 tokens from 22 → 37
   tok/s. Originals kept selectable for comparison.

The originals of every ported kernel are kept in tree (selectable via
`QW3_MATVEC=qw3` / `QW3_ATTN=qw3`) so we can keep iterating on our own
designs alongside the llama.cpp ports.

## What's implemented

**Model loading + tokenizer**
- Native GGUF reader (`src/gguf.cpp`).
- `QwenConfig` parsed entirely from GGUF metadata (no per-model hard-codes).
- Full GPT-2 byte-level BPE tokenizer with merges, special tokens, Qwen
  pre-tokenization rules, and `add_bos` handling.

**CUDA backend** (`include/qw3/device_backend.hpp`, `src/kernels_cuda.cu`)
- Q8_0 weights uploaded once at load time; never re-quantized or re-uploaded
  per step.
- **DP4A Q8_0 matvec** with shmem-staged input pre-quantization, multi-row /
  warp-shuffle reductions, and adaptive `WARPS_PER_BLOCK` (8/16/32 by row
  count). Used for small-batch prefill (batch 2..7) where it beats both
  HGEMM and the ported MMVQ.
- **MMVQ Q8_0 × Q8_1 matvec** (`src/mmvq_q8.cu`, ported from llama.cpp,
  see `LICENSES/llama.cpp.txt`). Decode (batch == 1) goes through this
  path: input → Q8_1 quantize once into a reusable scratch, then DP4A
  against the raw 34-byte Q8_0 weight blocks. The original qw3 matvec
  is kept and selectable with `QW3_MATVEC=qw3`.
- **Q8_0 matmul via cuBLAS HGEMM** for prefill (batch >= 8). Each Q8 weight
  is dequantized to FP16 once (cached on the weight for the lifetime of
  the process) and the prefill activations are converted FP32 → FP16 per
  call; cuBLAS picks a tensor-core algorithm automatically.
- **Flash-attention vec decode** (`src/fattn_vec_decode.cu`, ported from
  llama.cpp). One block per (head, batch, kv-slice); 4 warps × 32 lanes;
  per-warp KV slot, per-lane Q in registers, online softmax, V accumulator
  combined across warps via shmem. **Split-K extension** (qw3-side) adds an
  `NSPLIT ∈ {1,2,4,8}` template parameter that parallelizes over KV
  segments and writes per-split (max, sum, partial) tuples to scratch; a
  combine kernel merges them with online-softmax recurrence. Default for
  `head_dim ∈ {128, 256}`. The original qw3 fused-tile kernel is kept and
  selectable with `QW3_ATTN=qw3`.
- **F16 KV cache** by default. Each `kv_append_*` kernel writes FP16, and
  the fattn-vec decode kernel is templated on `KVT ∈ {float, __half}` with
  on-the-fly `__half2float` reads. Halves per-token K/V bandwidth at
  decode. Set `QW3_KV_DTYPE=fp32` to revert to FP32 KV (kept for parity
  diffs and exploration).
- **Batched prefill attention** with causal seq_len = `base + b + 1` per
  batch row, also in a single kernel launch per layer.
- DeltaNet recurrent step: per-layer state + conv1d ring buffer. Decode uses
  the per-token kernels; prefill uses **time-batched kernels** that process
  all T tokens of a layer in 4 kernel launches (conv, l2-norm, deltanet,
  norm+gate) instead of `5 * T` launches.
- Other fused kernels: `silu_mul`, batched RMSNorm / RoPE / KV-append /
  attention-gate.

**Executor** (`src/qwen_executor.{hpp,cpp}`)
- `forward_one_token` — decode path. Single-token forward, ~37 tok/s on
  Qwen 3.6 27B and **flat across context length** (37 tok/s at 550 tokens,
  37 tok/s at 4350 tokens — F16 KV + split-K removed the long-context
  cliff); uses the ported MMVQ for Q8 weights and per-token recurrent
  kernels.
- `forward_n_tokens` — batched prefill. All Q8 matmuls go through HGEMM /
  tensor cores; recurrent layers run the time-batched kernels above; the
  per-batch attention loop is still O(N²) but vectorized in a single launch.
  ~10× faster than the original per-token prefill (100 → 1539 tok/s on the
  1098-token prompt).
- Pre-allocated scratch (`h_`, `norm_`, `q/k/v_`, `mid_`, `ffn_*`, recurrent
  state) + batched versions allocated on demand by `ensure_batch_scratch`.

**Tooling**
- `--dump-logits` + `--dump-tokens` for parity diffs against llama.cpp.
- `scripts/compare_with_llama_cpp.py` runs both engines on the same prompts
  and reports tok/s, common prefix, and first-char match.

## Build

CPU-only (for inspection / mock backend tests):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA (required for actual model execution):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQW3_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Tested with CUDA 12.x / 13.x and SM_90 (Blackwell). Earlier compute
capabilities should work; pass `-DCMAKE_CUDA_ARCHITECTURES=<sm>` if needed.

## Run

End-to-end generation on a Qwen 3.6 GGUF:

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
| `--backend qwen-native`       | Use qw3's native engine (the optimization path). |
| `--native-heavy`              | Run the full forward (without this, only the no-op plan validation runs). |
| `--native-kernels cuda`       | Pick the CUDA device backend. |
| `--native-linear-backend auto`| Custom Q8 kernels for big matmuls; cuBLAS path is disabled in `auto` because the F32 dequant cache costs more bandwidth than it saves on a properly tuned Q8 kernel. Use `custom` to force, `cublas` to experiment. |
| `-p "..."` / `--prompt-file`  | Prompt (chat-formatted by default; use `--raw` to skip Qwen chat template). |
| `--system "..."`              | System prompt (default: a generic assistant prompt). |
| `--think`                     | Don't inject the empty `<think>` block. |
| `-n N`                        | Max new tokens (default 256). |
| `-c N`                        | KV cache size (default 32768). |
| `--dump-logits PATH`          | Write per-step top-K logits as JSONL for parity diffs. Forces the per-token prefill path. |
| `--dump-tokens`               | Tokenize prompt and exit. |

Inspect a GGUF without running it:

```sh
./build/qw3-inspect /path/to/Qwen3.6-27B-Q8_0.gguf
```

Smoke test without weights:

```sh
./build/qw3 --backend mock -p hello
```

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
match. Both engines are greedy (qw3 is always argmax; llama.cpp uses
`--temp 0`), so a correct implementation produces identical token streams.

Short prompts (< ~100 tokens) make prefill numbers noisy because per-process
warmup dominates. For prefill numbers worth reporting, prefer the long
sweep above; the legacy `--long`/`--long-only` modes use a single 1.3K
prompt and are kept for parity-style spot checks:

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

The Python entry point is runnable directly with the same flags:

```sh
python3 scripts/compare_with_llama_cpp.py \
  --qw3   ./build/qw3 \
  --llama /path/to/llama.cpp/build/bin/llama-completion \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  --long-only -n 128 --token-diff
```

llama.cpp must be built with `-DGGML_CUDA=ON` for a meaningful comparison
on this GPU — the CPU build hits ~2 tok/s decode and ~50 tok/s prefill,
which would make qw3 look fast for the wrong reason. The script invokes
llama.cpp through `llama-completion` (not `llama-cli`) for deterministic,
non-interactive execution.

## Logit-level parity diffs

To compare a single prompt token-by-token:

```sh
./build/qw3 --backend qwen-native --native-heavy --native-kernels cuda \
  --model /path/to/model.gguf \
  -p "Hello" -n 8 \
  --dump-logits qw3.jsonl --dump-logits-top-k 16 --dump-tokens

# Same prompt with llama.cpp's own logit dump (or any other reference) and
# diff JSONL records.
```

## Backends

| Backend         | When to use it |
|---|---|
| `qwen-native`   | Real inference. Owns GGUF loading, tokenizer, CUDA execution. |
| `mock`          | CI / build sanity. No weights needed. |
| `llama-cli`     | Forward to an external `llama-completion`. Reference only; not the optimization target. |

## Layout

```
include/qw3/
  device_backend.hpp  -- CUDA-agnostic tensor / weight / op interface
  qwen_config.hpp     -- Qwen3.5/3.6 hyperparams parsed from GGUF
  tokenizer.hpp       -- byte-level BPE tokenizer
  gguf.hpp / qw3.hpp  -- GGUF reader + engine surface

src/
  kernels_cuda.cu     -- CUDA backend (matvec/HGEMM dispatch, KV/RoPE/RMS, executor glue)
  mmvq_q8.cu          -- Ported Q8_0 × Q8_1 DP4A matvec (decode default)
  fattn_vec_decode.cu -- Ported flash-attention-vec decode kernel
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

## Known remaining gaps vs llama.cpp

See **Bottlenecks and roadmap** above for the prioritized list backed by
`nsys` profiling. The short version:

- **Decode plateau (~5 tok/s behind llama.cpp, flat across context).**
  Post-fusion, decode is 85% Q8 matvec (`mul_mat_vec_q8_0_kernel<1>`
  + the new `mul_mat_vec_q8_0_two_kernel<1>` for fused gate+up). Each
  matvec runs at ~2.5 TB/s effective on a 96 MB weight (above HBM
  peak — partial L2 reuse), so per-call latency is at the HBM
  bandwidth ceiling. Further wins must come from fewer matvec calls
  (Q+K+V fusion for standard attention layers, recurrent QKV+gate
  fusion across heterogeneous shapes) or from non-matvec kernels
  (DeltaNet recurrent fusion is the largest at ~5% of decode time).
- **Prefill: −500 tok/s at 1K–2K, widening to −1000 tok/s at 4K.** Two
  separate gaps: HGEMM via FP16 mirror loses to direct INT8 MMA on
  every prompt length (closing this lifts the whole curve), and at
  ≥ 4K the per-query fattn-vec attention scales O(T²) without K/V
  reuse — a tiled flash-attention prefill kernel is the right fix.
- **Numerical drift on greedy.** Greedy on both engines should yield
  identical tokens; today qw3 matches for ~5–22 tokens then drifts. Sources
  are the per-block Q8 → FP16 dequant rounding, FP16 accumulator in HGEMM
  (`CUBLAS_COMPUTE_32F_FAST_16F`), and the manual reductions in the
  recurrent stack. None block correct *generation*; a tight bit-exact match
  against llama.cpp is future work.
