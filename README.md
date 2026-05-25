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
`--temp 0`), so a correct implementation would emit identical tokens. The
benchmark uses a ~1.3K-token prompt so warmup overhead is amortized.

| | qw3 | llama.cpp | qw3 / llama.cpp |
|---|---|---|---|
| Prefill (1322-tok prompt) | 1031.6 tok/s | 3428.7 tok/s | **30.1%** |
| Decode  (1322-tok prompt) |   22.6 tok/s |   45.3 tok/s | **49.8%** |
| First-char match          | ✓ on every prompt | — | — |

How we got here (prefill on the same long prompt):

| Stage | Prefill tok/s | vs llama.cpp |
|---|---|---|
| Baseline (dp4a matvec, batched over T) |   100.6 |   2.9% |
| + cuBLAS HGEMM (FP16 dequant cache + tensor cores) | 602.0 | 17.6% |
| + Fused batched recurrent (1 launch / layer / sub-op) | 1031.6 | 30.1% |

With greedy sampling the output streams are identical for the first
generated chunk (the `<think></think>` chat-template prefix, ~20 tokens),
then floating-point drift in the Q8 dequant + softmax + recurrent stack
causes one token to flip and the streams to diverge. This is a numerical
gap to bit-exactness with llama.cpp, not a logic bug.

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
  count). Used for decode (batch = 1) where dp4a beats tensor cores.
- **Q8_0 matmul via cuBLAS HGEMM** for prefill (batch >= 8). Each Q8 weight
  is dequantized to FP16 once (cached on the weight for the lifetime of
  the process) and the prefill activations are converted FP32 → FP16 per
  call; cuBLAS picks a tensor-core algorithm automatically.
- **Fused flash-attention decode**, online softmax, no scores materialization;
  instantiated for both `head_dim=128` and `head_dim=256` (Qwen 3.6 uses 256).
- **Batched prefill attention** with causal seq_len = `base + b + 1` per
  batch row, also in a single kernel launch per layer.
- DeltaNet recurrent step: per-layer state + conv1d ring buffer. Decode uses
  the per-token kernels; prefill uses **time-batched kernels** that process
  all T tokens of a layer in 4 kernel launches (conv, l2-norm, deltanet,
  norm+gate) instead of `5 * T` launches.
- Other fused kernels: `silu_mul`, batched RMSNorm / RoPE / KV-append /
  attention-gate.

**Executor** (`src/qwen_executor.{hpp,cpp}`)
- `forward_one_token` — decode path. Single-token forward, ~36 tok/s on
  Qwen 3.6 27B; uses the dp4a matvec for Q8 weights and per-token recurrent
  kernels.
- `forward_n_tokens` — batched prefill. All Q8 matmuls go through HGEMM /
  tensor cores; recurrent layers run the time-batched kernels above; the
  per-batch attention loop is still O(N²) but vectorized in a single launch.
  ~10× faster than the original per-token prefill (100 → 1031 tok/s on the
  1322-token prompt).
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
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DQW3_ENABLE_CUDA=ON
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

Tested with CUDA 12.x / 13.x and SM_90 (Blackwell). Earlier compute
capabilities should work; pass `-DCMAKE_CUDA_ARCHITECTURES=<sm>` if needed.

## Run

End-to-end generation on a Qwen 3.6 GGUF:

```sh
./build-cuda/qw3 \
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
./build-cuda/qw3-inspect /path/to/Qwen3.6-27B-Q8_0.gguf
```

Smoke test without weights:

```sh
./build-cuda/qw3 --backend mock -p hello
```

## Benchmark against llama.cpp

`scripts/compare_with_llama_cpp.py` (driven by `scripts/run_compare.sh`) runs
a fixed set of prompts through both engines and reports prefill tok/s,
decode tok/s, and common-prefix length. Both engines are greedy (qw3 is
always argmax; llama.cpp is invoked with `--temp 0`), so a correct
implementation would produce identical output sequences.

Short prompts (< 32 tokens) make prefill numbers noisy because per-process
warmup dominates. For stable measurements use the **long-prompt** mode,
which adds (or substitutes) a single ~1.3K-token prompt:

```sh
# Default prompt set (4 short prompts):
bash scripts/run_compare.sh -n 64

# Add the 1322-token prompt to the default set:
bash scripts/run_compare.sh --long -n 128

# Run only the long prompt (best for prefill benchmarking):
bash scripts/run_compare.sh --long-only -n 128 --token-diff
```

`--token-diff` re-tokenizes both engines' outputs via qw3 and reports the
common token-level prefix length and whether the full token sequences are
identical.

The Python entry point is runnable directly with the same flags:

```sh
python3 scripts/compare_with_llama_cpp.py \
  --qw3   ./build-cuda/qw3 \
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
./build-cuda/qw3 --backend qwen-native --native-heavy --native-kernels cuda \
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
  kernels_cuda.cu     -- all CUDA kernels + CudaDeviceBackend
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

- **Prefill (~30% of llama.cpp on 1322-token prompt, ~3.3× off).** The
  HGEMM tensor-core path dequantizes Q8 → FP16 in a persistent cache and
  also converts the FP32 activations to FP16 per call; llama.cpp's MMQ
  kernels run INT8 mma directly on a repacked Q8 layout, which avoids both
  conversions. Closing this gap needs an INT8 mma kernel (or
  `cublasLtMatmul` with INT8 inputs + per-block scales).
- **Decode (~50% of llama.cpp).** Still on the dp4a matvec path; the same
  MMA approach (now with tensor-core matvec or a 4-bit / 8-bit MMQ style
  kernel that handles batch == 1 well) would close it.
- **Numerical drift on greedy.** Greedy on both engines should yield
  identical tokens; today qw3 matches for ~5–22 tokens then drifts. Sources
  are the per-block Q8 → FP16 dequant rounding, FP16 accumulator in HGEMM
  (`CUBLAS_COMPUTE_32F_FAST_16F`), and the manual reductions in the
  recurrent stack. None block correct *generation*; a tight bit-exact match
  against llama.cpp is future work.
- **F16 KV cache, repacked Q8 weight layout, CUDA graph capture of the
  decode loop** — deferred.
