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
Numbers below come from `scripts/long_prompt_sweep.py` (3 trials per cell,
alternating qw3 ↔ llama.cpp to spread thermal drift), median tok/s.

| Prompt tokens | qw3 prefill | llama prefill | prefill % | qw3 decode | llama decode | decode % |
|---:|---:|---:|---:|---:|---:|---:|
|    556 | 1989 tok/s | 2825 tok/s | **70.4%** | 45.89 tok/s | 45.51 tok/s | **100.8%** |
|   1098 | 2778 tok/s | 3321 tok/s | **83.7%** | 45.06 tok/s | 45.50 tok/s | **99.0%**  |
|   2182 | 3492 tok/s | 3592 tok/s | **97.2%** | 45.12 tok/s | 45.39 tok/s | **99.4%**  |
|   4350 | 3897 tok/s | 3767 tok/s | **103.5%** | 43.79 tok/s | 44.84 tok/s | **97.7%** |
|   8415 | 4000 tok/s | 3782 tok/s | **105.8%** | 43.75 tok/s | 43.95 tok/s | **99.6%** |
|  16545 | 3908 tok/s | 3706 tok/s | **105.4%** | 41.62 tok/s | 43.67 tok/s | **95.3%** |
|  33076 | 3502 tok/s | 3528 tok/s | **99.3%**  | 41.02 tok/s | 42.26 tok/s | **97.1%** |
|  65867 | 2840 tok/s | 3075 tok/s | **92.4%**  | 37.17 tok/s | 39.80 tok/s | **93.4%** |
| 131720 | 2018 tok/s | 2299 tok/s | **87.8%**  | 31.07 tok/s | 35.64 tok/s | **87.2%** |

All numbers absolute tokens/second, n_decode=64, ctx=140K (140000 for the
T=131K row; 70K for shorter prompts). The `%` columns report `qw3 /
llama.cpp`. Headlines:

- **Prefill ≥ 97% of llama.cpp at all T ≥ 2K**, with **103–106% from 4K–16K**.
  Long-context: 92% at T=65K, 88% at T=128K.
- **Decode is 87–101% of llama.cpp across the full 556–128K range**. Matvec
  is at HBM ceiling (~2.7 TB/s effective); further wins need fewer matvec
  calls, not faster kernels.
- **Short-prompt gap (T<2K) is launch-overhead-bound**, not compute. nsys
  attributes ~64% of T=556 prefill to the first-call HGEMM autotuner ("Kernel2",
  ~111 ms over 32 attention layers × ~14 calls/layer) plus q8 weight dequant
  staging (~48 ms). Long-T amortizes both. See development log.

Reproduce with:

```sh
python3 scripts/long_prompt_sweep.py \
  --prompt-tokens "512 1024 2048 4096 8192 16384 32768 65536 131072" \
  --trials 3 -n 64 -c 140000 \
  --json /tmp/sweep.json
```

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
| `QW3_PREFILL_ATTN`          | `mma-gqa-v2` | Prefill FA kernel: `mma-gqa-v2` (current default), `mma-gqa` (v1, 6-head loop), `mma-pipe`, `mma`, `vec`, `cublas`. |
| `QW3_PREFILL_FA2_BR`        | `16`    | v2 q-rows-per-CTA: `8`, `16` (default), `32` (parity-correct, regresses 1.5%). |
| `QW3_PREFILL_FA2_BC`        | `32`    | v2 K/V tile width: `32` (default — 2 blocks/SM occupancy), `64`. |
| `QW3_PREFILL_FA2_KCPASYNC`  | `1`     | `0` reverts to sync K loads (dropped +5–7% at long T). |
| `QW3_PREFILL_FA2_VCPASYNC`  | `1`     | `0` reverts to sync V loads (dropped +14% at 65K). |
| `QW3_FATTN_NSPLIT`          | adaptive | Decode-attn split-K: `{1,2,4,8,16,32,64}`. Default policy targets ≈128 KV/split. |
| `QW3_FUSE_SILU_MUL`         | `1`     | `0` reverts FFN gate+up+silu+mul to two matvecs + a separate silu_mul. |
| `QW3_FUSE_ADD`              | `1`     | `0` reverts attn_output / ffn_down to plain matvec + separate add. |
| `QW3_GRAPH`                 | `1`     | `0` disables CUDA graph capture of decode. |
| `QW3_HGEMM_X_CACHE`         | `1`     | `0` disables FP16 input reuse across consecutive HGEMMs sharing an input. |
| `QW3_KV_DTYPE`              | `fp16`  | `fp32` reverts KV cache to FP32 (parity-only). |
| `QW3_MATMUL`                | `hgemm` | `mmq` enables in-tree MMQ kernel (still trails HGEMM at long T). |

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
