# qw3

A from-scratch CUDA inference framework for the **Qwen 3.5 / Qwen 3.6** family
(architecture `qwen35`) — a hybrid model with 16 standard attention layers
interleaved among 48 DeltaNet recurrent layers. The framework loads GGUF
weights directly, runs its own tokenizer, owns all device memory, and ships
hand-written CUDA kernels (Q8_0 DP4A matvec / matmul, fused flash-attention
decode, batched prefill).

llama.cpp is kept around only as a correctness baseline and benchmarking
counterpart — it never participates in actual generation.

## Status (Qwen 3.6 27B Q8_0, RTX Pro 6000 Blackwell)

| | qw3 | llama.cpp | qw3 / llama.cpp |
|---|---|---|---|
| Prefill (4-prompt avg)   |  70.27 tok/s | 142.6 tok/s |  **49.3%** |
| Prefill (long prompt N=68)| 93.86 tok/s | 142.6 tok/s |  **65.8%** |
| Decode (4-prompt avg)    |  35.75 tok/s |  45.3 tok/s |  **78.9%** |
| First-char match         |  ✓ on every prompt |   —   |  —  |

Output is semantically aligned with llama.cpp (every first generated token
matches; common-prefix typically 40–80% of the qw3 output). Sub-sentence
divergence is floating-point drift, not a logic bug.

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
  count).
- **Q8_0 matmul** — same kernel batched over `gridDim.y`; the FFN weight is
  read once and reused across the prompt's N tokens during prefill.
- **Fused flash-attention decode**, online softmax, no scores materialization;
  instantiated for both `head_dim=128` and `head_dim=256` (Qwen 3.6 uses 256).
- **Batched prefill attention** with causal seq_len = `base + b + 1` per
  batch row, also in a single kernel launch per layer.
- DeltaNet recurrent step: per-layer state + conv1d ring buffer, with an
  offset-aware variant for the batched prefill path.
- Other fused kernels: `silu_mul`, batched RMSNorm / RoPE / KV-append /
  attention-gate.

**Executor** (`src/qwen_executor.{hpp,cpp}`)
- `forward_one_token` — decode path. Single-token forward, ~36 tok/s on
  Qwen 3.6 27B.
- `forward_n_tokens` — batched prefill. Single forward pass with batched
  matmuls + per-token attention/recurrent loops; ~2× faster than the old
  per-token prefill loop.
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
a fixed set of prompts through both engines on the remote machine, comparing
prefill tok/s, decode tok/s, and output common-prefix length.

```sh
# By default both binaries live on the remote Pro6000 host the script
# was wired for; edit the env vars in run_compare.sh for your setup.
bash scripts/run_compare.sh
```

The Python entry point is also runnable directly:

```sh
python3 scripts/compare_with_llama_cpp.py \
  --qw3   ./build-cuda/qw3 \
  --llama /path/to/llama.cpp/build/bin/llama-completion \
  --model /path/to/Qwen3.6-27B-Q8_0.gguf \
  --tokens 32
```

llama.cpp is invoked through `llama-completion` (not `llama-cli`) for
deterministic, non-interactive execution. The script tolerates `inf` rates
when prefill is too fast to time.

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

- **Prefill (~50% of llama.cpp).** Limited by Q8 weight read bandwidth.
  Closing this needs MMA / tensor-core kernels on a repacked Q8 layout
  (similar to llama.cpp's MMQ kernels).
- **Decode (~79% of llama.cpp).** Memory-bound at this point with a near-
  optimal scalar Q8 kernel. Same MMA path would close the rest.
- **Recurrent layer per-token launches** — 5 small kernels per token ×
  48 recurrent layers per prefill is ~80ms of launch overhead. Fusing the
  conv1d + L2 norm + DeltaNet + norm_gate sub-kernels (and/or capturing
  the loop into a CUDA Graph) would close another chunk.
- F16 KV cache, repacked Q8 weight layout — deferred.
