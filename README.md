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

| Prompt tokens | qw3 prefill | llama prefill | pref ratio | qw3 decode | llama decode | dec ratio |
|---:|---:|---:|---:|---:|---:|---:|
|  556 | 996.7  | 2814.4 | **35.4%** | 37.0 | 45.5 | **81.4%** |
| 1098 | 1539.0 | 3313.0 | **46.5%** | 33.6 | 45.5 | **74.0%** |
| 2182 | 1961.4 | 3592.8 | **54.6%** | 28.5 | 45.3 | **62.8%** |
| 4350 | 1916.3 | 3780.3 | **50.7%** | 21.8 | 44.8 | **48.5%** |

Two trends jump out:

- **Prefill plateaus around 1960 tok/s** (~50–55% of llama.cpp). The
  HGEMM tensor-core path is the right idea but the Q8 → FP16 dequant +
  FP32 → FP16 activation conversion eat ~7% of total time and bound peak
  throughput. llama.cpp's MMQ runs INT8 MMA directly on repacked Q8 with
  no conversions, and keeps scaling with prompt length.
- **Decode degrades steeply with KV cache size** (37 → 22 tok/s as the
  prompt grows from ~550 to ~4.4K tokens) while llama.cpp stays flat at
  ~45 tok/s. qw3 stores the KV cache in FP32 and llama.cpp uses F16, so
  per-token attention reads 2× the bytes — the 2× decode gap at long
  context tracks that exactly.

How we got here (prefill on a 1322-token prompt, earlier baseline measurements):

| Stage | Prefill tok/s | vs llama.cpp |
|---|---|---|
| Baseline (dp4a matvec, batched over T) |   100.6 |   2.9% |
| + cuBLAS HGEMM (FP16 dequant cache + tensor cores) | 602.0 | 17.6% |
| + Fused batched recurrent (1 launch / layer / sub-op) | 1031.6 | 30.1% |
| + Ported MMVQ (Q8_1 activation) + flash-attention-vec decode | 1539 | 46.5% |

With greedy sampling the output streams are identical for the first
generated chunk (the `<think></think>` chat-template prefix, ~20 tokens),
then floating-point drift in the Q8 dequant + softmax + recurrent stack
causes one token to flip and the streams to diverge. This is a numerical
gap to bit-exactness with llama.cpp, not a logic bug.

## Bottlenecks and roadmap

`nsys` on the 4350-token prompt + 64-token decode (5.0 s wall) gives a
clear picture of where time goes:

| Kernel | % wall | Total time | Calls |
|---|---:|---:|---:|
| `fattn_vec_decode` (decode attention) | **47.5%** | 2.42 s | 1024 |
| `mul_mat_vec_q8_0` (decode + batch=1 prefill matvec) | **23.3%** | 1.19 s | 31312 |
| CUTLASS HGEMM 256×128 (prefill linears) | 10.1% | 0.51 s | 320 |
| `q8_dequant_f16` (prefill weight unpack) | 5.8% | 0.30 s | 496 |
| `gated_delta_net` (DeltaNet recurrent) | 3.2% | 0.16 s | 48 |
| RMSNorm / SiLU·mul / KV-append / RoPE | ~5% | 0.25 s | many |

Per-token decode budget at 4350-token KV: 46.9 ms. Of that, decode-attention
is 37.7 ms (16 layers × 2.36 ms median per layer) — the rest of the network
fits in ~9 ms. So the next gains will come from:

1. **F16 KV cache.** Halve per-token attention bandwidth. The fattn-vec
   kernel is bandwidth-bound on the K/V reads (each token does
   `seq_len × n_kv_heads × head_dim × 4 bytes`, twice — once for K, once
   for V). Switching to FP16 cuts that by 2× and lines up with llama.cpp's
   layout. Rough projection: 22 → 38 tok/s at 4K context, 28 → 45 tok/s at
   2K context. **Biggest expected win for long-context decode.**

2. **MMQ-style prefill (INT8 MMA on repacked Q8).** Drops the Q8 → FP16
   dequant cache and the FP32 → FP16 activation conversion (currently 7%
   of wall on long prompts), and uses tensor-core integer MMA the same
   way llama.cpp does. Should lift prefill above ~3000 tok/s and remove
   the plateau at 1960 tok/s.

3. **Persistent activation Q8_1 buffer reuse across Q/K/V/gate/up.** The
   current `mul_mat_vec_q8_0` path re-quantizes the input to Q8_1 once per
   matvec; in attention and FFN, the same input feeds 3+ matvecs. Hoisting
   the quantization across them cuts the `quantize_q8_1` cost (~0.8% on
   long, more at shorter context) and saves a pass over input memory.

4. **CUDA graph capture of the decode loop.** Decode is 64+ kernel
   launches per token (RMS, mmvq×K, RoPE, KV append, fattn, RMS, mmvq×Q,
   SiLU·mul, mmvq, add) replayed each token. A single captured graph
   removes per-launch overhead — small absolute win, but a free one once
   the steady-state shape is fixed.

5. **DeltaNet recurrent fusion.** 3.2% on long prompts; the four kernels
   per layer (conv, l2, deltanet, norm-gate) are each light but sequential.
   A single fused launch per layer is mostly a launch-latency win (small,
   but easy).

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
  llama.cpp). One block per (head, batch); 4 warps × 32 lanes; per-warp KV
  slot, per-lane Q in registers, online softmax, V accumulator combined
  across warps via shmem. Default for `head_dim ∈ {128, 256}`. The original
  qw3 fused-tile kernel is kept and selectable with `QW3_ATTN=qw3`.
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
  Qwen 3.6 27B at small KV (drops to ~22 tok/s at 4K KV — see roadmap);
  uses the ported MMVQ for Q8 weights and per-token recurrent kernels.
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

- **Decode at long context (~50% of llama.cpp at 4K).** FP32 KV cache
  vs llama.cpp's F16 — 2× the attention-read bandwidth.
- **Prefill plateau (~50% of llama.cpp from 2K onward).** No INT8 MMA
  path; the HGEMM route pays Q8 → FP16 dequant + FP32 → FP16 activation
  conversion every step.
- **Numerical drift on greedy.** Greedy on both engines should yield
  identical tokens; today qw3 matches for ~5–22 tokens then drifts. Sources
  are the per-block Q8 → FP16 dequant rounding, FP16 accumulator in HGEMM
  (`CUBLAS_COMPUTE_32F_FAST_16F`), and the manual reductions in the
  recurrent stack. None block correct *generation*; a tight bit-exact match
  against llama.cpp is future work.
