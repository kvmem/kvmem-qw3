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

| Prompt tokens | qw3 prefill | llama prefill | prefill % | qw3 decode | llama decode | decode % |
|---:|---:|---:|---:|---:|---:|---:|
|  556 | 1903.9 tok/s | 2812.6 tok/s | **67.7%** | 45.41 tok/s | 45.26 tok/s | **100.3%** |
| 1098 | 2568.0 tok/s | 3317.3 tok/s | **77.4%** | 45.05 tok/s | 45.45 tok/s | **99.1%** |
| 2182 | 3022.8 tok/s | 3599.4 tok/s | **84.0%** | 45.12 tok/s | 45.35 tok/s | **99.5%** |
| 4350 | 2969.7 tok/s | 3779.3 tok/s | **78.6%** | 43.77 tok/s | 44.80 tok/s | **97.7%** |

All numbers are absolute tokens/second. The `%` columns report `qw3 / llama.cpp`.
Two trends jump out:

- **Decode is at parity with llama.cpp** (98–100% across all context lengths,
  even modestly *beating* it at 556 tokens). It used to be 88–92%. Four
  changes in this round closed the gap, in order of impact:

  1. **Adaptive decode-attention split-K** (`src/fattn_vec_decode.cu`).
     The old `pick_nsplit` returned `NSPLIT=1` for `seq_len <= 1024`, so
     decode ran with 24 heads × 4 warps = 96 warps total — under 5% of
     Blackwell's 128 SMs × 16 active warps capacity. Each block then
     walked the full KV tail sequentially in ~315 µs (vs the ~2 µs the
     read is actually worth); the kernel was sync/latency-bound, not
     bandwidth-bound. The new policy targets per_split ≈ 128 KV tokens:
     `{1, 2, 4, 4, 8, 16}` for `{≤64, ≤128, ≤512, ≤2048, …}`. At 1K
     context this drops per-call attention from 315 → 39 µs (8× faster);
     at 4K NSPLIT=16 instantiation gives another 50% over the previous
     NSPLIT=8 max. Override with `QW3_FATTN_NSPLIT={1,2,4,8,16}`.
  2. **Fused SwiGLU matvec** (`mul_mat_vec_q8_0_silu_mul_kernel` in
     `src/mmvq_q8.cu`). FFN previously ran as: two-weight matvec writes
     `ffn_gate` + `ffn_up` (n_ff = 17408 each), `silu_mul_kernel` reads
     both back and writes `ffn_mid`. The fused kernel does the same
     two matvecs with the activation-shared MMVQ path, then writes
     `silu(g) * u` directly — eliminating one full kernel launch per
     layer (64/token) plus the n_ff round-trip of both intermediates.
     Disable with `QW3_FUSE_SILU_MUL=off`.
  3. **Fused matvec + residual add** (`mul_mat_vec_q8_0_add_kernel`).
     The attn_output and ffn_down matvecs are each followed by
     `h = h + Wx`. The new kernel reads its existing `dst` and writes
     `dst += Wx` directly, eliminating the n_embd-wide `attn_out` /
     `ffn_out` scratch buffers and the `add_kernel` launches (128/token
     across all 64 layers). Disable with `QW3_FUSE_ADD=off`.
  4. **`NSPLIT=16` instantiation.** Adds a fifth template variant to
     the split-K decode kernel so the longest contexts (≥2K) can keep
     filling SMs as KV grows. The combine kernel scales naturally with
     `NSPLIT` (one block per (head, batch), `HEAD_DIM` threads each).

  After these changes the post-fusion decode profile (1029-token prompt,
  CUDA graphs off, 64 decode tokens):

  | Kernel | per-token (ms) | share |
  |---|---:|---:|
  | `mul_mat_vec_q8_0_silu_mul` (FFN gate+up+silu+mul fused)  | 8.05 | 37% |
  | `mul_mat_vec_q8_0_add` (attn_output / ffn_down + residual) | 5.55 | 26% |
  | `mul_mat_vec_q8_0` (Q/K+V/recurrent/LM-head matvec)       | 4.49 | 21% |
  | `fattn_vec_decode<NSPLIT=8>` (decode attention)            | 0.61 | 3%  |
  | DeltaNet recurrent stack (conv + l2 + deltanet + norm_gate)| 0.70 | 3%  |
  | RMSNorm (post-attn + post-FFN)                            | 0.38 | 2%  |
  | other (RoPE, KV append, argmax, …)                        | 0.4  | 2%  |
  | **total**                                                 | **20.2** | **~85% of measured 22 ms graph-off** |

  Matvec is **84% of decode** and runs at HBM ceiling (~2.7 TB/s effective
  on the FFN weights with partial L2 reuse). Decode attention dropped
  from 19% → 3% of decode time — that's the change that mattered most.
  The remaining ~1 tok/s gap at 4K is launch dispatch + DeltaNet recurrent
  fusion potential (~5% of decode); both are small wins from here.

- **Prefill sits at 67–84% of llama.cpp**, with the gap narrowing as
  context grows past 1K. The dominant cost on llama.cpp's prefill is
  `mul_mat_q` (INT8 MMA on raw Q8_0) at 64% of GPU time — they bypass
  dequant entirely. qw3 currently runs prefill matmul through cuBLAS
  HGEMM with on-the-fly Q8 → FP16 dequant (~12% of prefill GPU time goes
  to dequant + FP32→FP16 staging). The 4K case used to fall off
  to 71% because the prefill attention kernel re-used the per-query
  decode kernel (O(T²) HBM traffic, no K/V reuse); the new tiled
  FA2-style prefill kernel (BR=16, BC=32, vectorized int4 K/V loads)
  shares K/V across 16 queries and cuts prefill attention by ~28% at
  4K, lifting prefill from 71% → 79%. The in-tree MMQ kernel
  (`src/mmq_q8.cu`, opt-in via `QW3_MATMUL=mmq`) hits 74% of HGEMM at
  2K — still trailing — and v3 ldmatrix is parity-correct but slightly
  slower. Closing the rest needs cp.async + repacked weights; see roadmap.

How we got here, decode at 4350-token context (median of 3 trials, n=64,
llama.cpp baseline 44.77 tok/s on this GPU):

| Stage | Decode (tok/s) | % of llama.cpp |
|---|---:|---:|
| Baseline (per-token decode, F32 KV)                   | ~22.0 | 49% |
| + F16 KV cache                                        | ~28   | 63% |
| + Ported MMVQ (DP4A Q8_0 × Q8_1)                      | ~33   | 74% |
| + Split-K decode attention (NSPLIT ∈ {1,2,4,8})       | 37.3  | 83% |
| + CUDA graph capture of decode loop                   | 38.5  | 86% |
| + FFN gate+up matvec fusion (two-weight kernel)       | 41.0  | 92% |
| **+ Adaptive NSPLIT + NSPLIT=16 + SwiGLU/add fusion** | **43.8** | **98%** |

The biggest single jump was the **NSPLIT policy rewrite** — at 1K context
the old code ran decode attention with 24 heads × 4 warps total, 4.7% of
Blackwell's warp capacity, sequentially walking the full KV tail. Bumping
to per_split ≈ 128 KV tokens (target NSPLIT=8 by 1K, NSPLIT=16 by 4K)
fills the SMs and drops per-call attention 8×. Decode at short context
(556 tokens) was the most under-occupied and gained the most: 40 → 45 tok/s,
i.e. now matches llama.cpp exactly.

How we got here on prefill (1322-token prompt, earlier baseline measurements):

| Stage | Prefill (tok/s) | Δ vs llama.cpp (tok/s) |
|---|---:|---:|
| Baseline (dp4a matvec, batched over T) |   100.6 | −3300 |
| + cuBLAS HGEMM (FP16 dequant cache + tensor cores) |  602.0 | −2800 |
| + Fused batched recurrent (1 launch / layer / sub-op) | 1031.6 | −2270 |
| + Ported MMVQ (Q8_1 activation) + flash-attention-vec decode | 1539 | −1755 |
| + F16 KV cache + split-K decode attention | 1590 | −1700 |
| + Eager Q8 → FP16 weight prewarm at upload time (1098-tok prompt) | 2875 | **−445** |

(llama.cpp on the 1098-token prompt is ~3305 tok/s on this GPU.)

## Bottlenecks and roadmap

`nsys` on the 965-token prompt + 64-token decode (graphs OFF, 1.6 s wall)
gives the post-fusion picture:

| Kernel | % wall | Total time | Calls |
|---|---:|---:|---:|
| `mul_mat_vec_q8_0_silu_mul` (fused FFN gate+up+silu+mul) | **31.7%** | 0.52 s | 4032 |
| `mul_mat_vec_q8_0_add` (matvec + residual add)          | 21.9% | 0.36 s | 8064 |
| `mul_mat_vec_q8_0` (Q/K+V/recurrent/LM-head decode)     | 17.7% | 0.29 s | 7120 |
| CUTLASS HGEMM 256×128 (prefill linears)                 | 7.5%  | 0.12 s | 320  |
| `q8_dequant_f16` (per-call HGEMM weight unpack)         | 2.8%  | 0.05 s | 496  |
| `fattn_vec_decode<NSPLIT=8>` (decode attention)         | 2.4%  | 0.04 s | 1008 |
| `gated_delta_net` (DeltaNet recurrent, prefill)         | 2.3%  | 0.04 s | 48   |
| DeltaNet decode kernels (conv/l2/deltanet/norm_gate)    | 3.6%  | 0.06 s | ~12K |
| `fattn_prefill_kernel` (prefill attention)              | 1.2%  | 0.02 s | 16   |
| `mul_mat_vec_q8_0_two` (legacy gate+up, recurrent fanout) | 1.1% | 0.02 s | 4032 |

The decode-attention shift was dramatic: it used to be 47.5% of wall in
an earlier baseline, then 6.8% after the first split-K instantiation,
and is **2.4% now** with NSPLIT=8 at this context length. Per-call
attention dropped from 1.38 ms → 315 µs → 39 µs. The combine kernel is
<1 ms total across the whole run (negligible).

Decode is now overwhelmingly matvec-bound: 71% of GPU time is the three
Q8_0 matvec variants (`silu_mul`, `add`, plain), running at the HBM
bandwidth ceiling (~2.7 TB/s effective with partial L2 reuse on the FFN
weights). Closing the remaining ~1 tok/s decode gap to llama.cpp now
means either fewer matvec calls or a structurally different layout
(MMQ-style INT8 MMA against repacked weights). So the next gains will
come from:

1. **Adaptive decode-attention split-K.** *(Done — default.)* The old
   policy returned `NSPLIT=1` for `seq_len ≤ 1024`, which on Blackwell
   left 96 warps total (24 heads × 4 warps) — under 5% of the 2048-warp
   capacity. Each block then walked the full KV tail sequentially in
   ~315 µs (vs the ~2 µs the actual HBM read is worth), so the kernel
   was latency- / sync-bound, not bandwidth-bound. The new
   `pick_nsplit` targets per_split ≈ 128 KV tokens:
   `{≤64: 1, ≤128: 2, ≤512: 4, ≤2048: 8, else 16}`. At 1K context
   per-call attention dropped 315 → 39 µs; at 4K with `NSPLIT=16`
   another 2× over NSPLIT=8. Decode at 1K context went from
   40 → 45 tok/s (matches llama.cpp). Override with
   `QW3_FATTN_NSPLIT={1,2,4,8,16}` for parity diffs.

2. **Fused FFN SwiGLU matvec.** *(Done — default.)*
   `mul_mat_vec_q8_0_silu_mul_kernel` reads gate and up weights with
   shared activations (the existing two-weight DP4A pattern), then
   writes `silu(g) * u` directly to `ffn_mid`. Eliminates one full
   kernel launch per layer (`silu_mul_kernel`) and the two n_ff-wide
   intermediates' round-trip through DRAM. Disable with
   `QW3_FUSE_SILU_MUL=off`.

3. **Fused matvec + residual add.** *(Done — default.)*
   `mul_mat_vec_q8_0_add_kernel` writes `dst[row] += W·x` with
   read-modify-write at the output store. Used for attn_output and
   ffn_down (the two layer-ending matvecs that immediately feed
   the residual stream). Saves 128 `add_kernel` launches per token
   and removes the n_embd-wide `attn_out` / `ffn_out` scratch
   round-trip. Disable with `QW3_FUSE_ADD=off`.

4. **Tiled flash-attention prefill kernel.** *(Done — default.)*
   The 4K-prompt deficit used to be dominated by `fattn_vec_decode_kernel`
   running with batch = T queries; each query's block read K/V
   independently from HBM with no reuse, so the kernel scaled O(T²) at
   ~580 GB/s effective bandwidth (~534 ms on the 4K case). The new
   `fattn_prefill_kernel` (in `src/fattn_vec_decode.cu`) is FA2-style:
   BR=16 queries per block share a BC=32 token K/V tile loaded
   cooperatively into shmem with vectorized 16-byte (`int4`) HBM loads,
   then each warp runs an online softmax against its own query. K/V is
   read from HBM once per 16 queries instead of once per query. Per-call
   prefill attention dropped 401 → 355 ms at 4K, total prefill attention
   from 495 → 355 ms (-28%). Override via `QW3_PREFILL_ATTN=vec`
   (legacy split-K) or `QW3_PREFILL_ATTN_BR={4,8,16,32}`.

## Prefill roadmap — closing the gap to llama.cpp

Current state: 67–84% of llama.cpp prefill. The default matmul backend is
HGEMM (cuBLAS HGEMM on Q8 → FP16 weights). MMQ exists as opt-in
(`QW3_MATMUL=mmq`) but is not yet faster than HGEMM at our shapes.

GPU-time breakdown on a 601-token prefill (HGEMM default, CUDA graphs off):

| Kernel                      | Total ms | % of GPU | Per-call median |
|---|---:|---:|---:|
| CUTLASS HGEMM (Q8 → FP16 + MMA) | 96.2 | **44.3%** | 261 µs |
| `q8_dequant_f16` (per-call weight unpack) | 44.1 | **20.3%** | 89 µs |
| `gated_delta_net_kernel`        | 24.6 | 11.3% | 512 µs |
| `recurrent_conv_batch_kernel`   | 19.7 |  9.1% | 410 µs |
| `fattn_prefill_kernel`          |  8.9 |  4.1% | 555 µs |
| `fp32_to_fp16_kernel` (act staging) | 8.2 |  3.8% | 16 µs |
| `rms_norm_kernel`               |  4.1 |  1.9% | 32 µs |
| `silu_mul_kernel`               |  3.9 |  1.8% | 60 µs |
| balance (add, fanout, …)        |  ≈10 |  ≈5% | — |
| **Matmul-related** (HGEMM + dequant + fp16 stage) | **148.5** | **68.4%** | — |

llama.cpp on the same prompt shape spends **111 ms** on matmul (mmq<128>
85 ms + mmq<96> 26 ms + fixup 3.8 ms) — i.e. ~37 ms (17% of our prefill
time) is **literally the dequant + activation-staging tax** the HGEMM path
pays that llama.cpp's `mul_mat_q` avoids by reading raw Q8_0 directly.

Two non-exclusive paths to close that 37 ms (and the 5–10 ms beyond it):

### Path A — Eliminate dequant from the HGEMM path *(highest leverage, biggest unknown)*

Fuse Q8_0 → FP16 dequant **into** the matmul kernel. Either:

- **Custom CUTLASS GEMM with a Q8_0 loader-stage** (cute / 3.x layout API).
  CUTLASS `tn_align8` SM80 kernel ships an FP16 → FP16 path; replacing
  the global-load fragment with a fused Q8_0 → FP16 dequant in the
  shared-memory staging step gives "HGEMM speed, no dequant launch."
  Pros: keeps the proven HGEMM tile-tuning + tensor-core schedule.
  Cons: medium-large effort (CUTLASS kernel authoring), risk of perf
  regression on small batches.
- **Custom FP16-MMA kernel** in `src/mmq_q8.cu` analogous to v5 but
  with `m16n8k16.f16.f16.f32` instead of `m16n8k32.s8.s8.s32` and an
  inline Q8_0 → FP16 unpack at the shmem-fill step. Pros: full control,
  reuses our v5 plumbing; Cons: no tensor-core auto-tuning, must
  manually find the best `mmq_x`/`mmq_y`/`nwarps` for each shape.

Expected gain: **+15–20% prefill** (close the 37 ms gap) → ~3500 tok/s
at 4K, ≈ **93% of llama.cpp**. This is the **biggest single lever**.

### Path B — Make MMQ faster than HGEMM *(matches llama.cpp's exact approach)*

v5 is spill-free at `mmq_x=64` but still ~50% slower than HGEMM
(1542 vs 3000 tok/s at 4K). To turn MMQ into the default we'd need:

B1. **Stream-K work partitioning** (`mmq.cuh:3528–3825`).
    Today: 200 CTAs × 1 col-tile-wide on 188 Blackwell SMs → 1.06 waves +
    a 12-CTA tail wave that uses 12/188 = 6% of SMs. With stream-K:
    one persistent CTA per SM partitions the K dim and combines via
    a fixup buffer + epilog kernel. Bumps SM utilization to ~100% and
    cuts the partial-wave tax to ~0.
    *Expected: +25–35% MMQ throughput → 1900–2050 tok/s. Medium
    complexity (tmp_fixup allocator + 2nd kernel for combine).*

B2. **Per-shape `mmq_x` ladder** (`mmq.cuh:4055–4138`).
    Add template instantiations for `mmq_x ∈ {8, 16, 24, …, 128}` and a
    runtime dispatch (8-step ladder) that picks the value minimising
    `ntiles_x = ceil(batch / mmq_x)`. At batch=601, mmq_x=128 gives 5
    col tiles; mmq_x=96 gives 7 (worse); mmq_x=112 gives 6 (worse).
    But at batch=129 the optimum drops to mmq_x=72 (2 tiles instead of
    1×128+1×1 = 2 tiles, but lower compute waste in the last tile).
    *Expected: +5–10% MMQ throughput at non-128-multiple batches.
    Low complexity, mostly boilerplate.*

B3. **MMQ inner-loop micro-tuning.**
    Even spill-free, v5<64> takes 720 µs/CTA vs llama.cpp mmq<128>'s
    215 µs/CTA — a 3.3× per-MAC gap that stream-K alone can't close.
    Most likely culprits, in order:
    - **Per-MMA scalar accumulator chain** (`sum[…] += C.x[l] * dA *
      dB`) — 4 FP32 MADs per MMA, sequential on the accumulator slot.
      Worth checking whether splitting `sum[]` into two ping-pong
      halves or using `fma.rn.f32` explicit instructions helps.
    - **`ldmatrix.x4` bank conflict pattern.** Padding
      `MMQ_MMA_TILE_X_K_Q8_0 = 76 ints/row` gives `% 8 == 4` (per
      llama.cpp's static_assert), 2-way conflicts at our access pattern.
      Worth verifying with `ncu`.
    - **`cp.async` for the activation tile.** `tile_y` is contiguous
      `int[]` packed Q8_1 — naturally 16-byte aligned. Loading it via
      `cp.async.cg.shared.global.L2::128B` with double-buffering across
      the K-superblock boundary would overlap the next outer iter's Y
      read with the current MMA. **Crucially this does not require
      repacking weights** — only the *activation* tile uses cp.async,
      and activations are scratch (zero added memory). Llama.cpp's mmq
      doesn't do this but its kernel runs ~3× faster per CTA than ours
      anyway; for us, this is the lowest-risk path to compute/load
      overlap.
    *Expected (combined): another +20–30% on top of stream-K → 2300–
    2700 tok/s. Medium-high complexity.*

If A1 lands, B is no longer on the critical path. If A1 turns out to
be infeasible (CUTLASS authoring blockers), B1+B2+B3 combined still
gets MMQ above HGEMM and closes most of the gap.

### Path C — Non-matmul wins *(small but additive, no risk to backend stability)*

C1. **CUDA graph capture for prefill** *(low effort, moderate impact)*.
    Decode is graph-captured; prefill isn't. With ~200 kernel launches
    per prefill at large batch and ~3.5 µs launch latency apiece, that's
    ~700 µs/forward of pure launch overhead — visible as ~5% of prefill
    time at small batches. Same shape as the decode capture: one graph
    per (batch, layer_count) tuple, re-instantiated on shape change.
    *Expected: +2–5% prefill at large batch, +5–10% at small batch.*

C2. **DeltaNet recurrent kernel audit** *(11% of GPU time, 24 ms
    median per call)*. Each call processes T tokens × 48 layers of the
    `gated_delta_net` recurrence. Cost-density is `~512 µs ÷ 4350
    tokens / 16 attn-replaced-by-recurrent layers ≈ 7 µs/token/layer`.
    Compare to llama.cpp's `gated_delta_net_cuda<128>` at 169 µs/call
    × 48 calls (3× faster per call at the same shape). Audit shmem
    usage + warp distribution. *Expected: +3–5% prefill.*

C3. **`recurrent_conv_batch_kernel` audit** *(9% of GPU time)*. 410 µs
    median per call. Similar audit-vs-llama.cpp opportunity. *Expected:
    +2–4% prefill.*

C4. **`fp32_to_fp16_kernel` elimination** *(3.8% of GPU time, 8.2 ms
    total)*. Prefill activations are currently FP32 → converted to FP16
    once per matmul. If we kept activations in BF16 throughout the
    pipeline (matches Qwen native dtype), this kernel drops entirely.
    Risk: changes precision of every prefill op. *Expected: +2–3%
    prefill, but invasive.*

### Recommended ordering

1. **B2 (per-shape mmq_x ladder)** — 1 day, low risk, lets MMQ keep up
   at small batches and gives clean A/B against llama.cpp on the same
   tile shape.
2. **B1 (stream-K)** — 2–3 days. Closes most of the MMQ gap and is
   well-understood from llama.cpp's reference.
3. **C1 (CUDA graph prefill)** — 1 day. Helps both HGEMM and MMQ paths.
4. **A1 (CUTLASS Q8 → FP16 dequant fusion)** — 3–5 days. Biggest
   single prefill win available given the user's strict no-extra-memory
   constraint. Highest payoff but highest unknown.
5. **B3 (MMQ inner-loop tuning)** — only if A1 is blocked or after B1
   leaves MMQ within striking distance of HGEMM.
6. **C2/C3 (non-matmul audits)** — opportunistic; do during build/test
   cycles, not as a focus push.

Estimated trajectory if A1 + B1 + B2 + C1 land in that order:

| Step | Expected prefill at 4K | % of llama.cpp |
|---|---:|---:|
| Today (HGEMM)                 | 2970 | 78.6% |
| + B2 mmq_x ladder (if MMQ default) | 1700 (MMQ) | 45% (still slower than HGEMM) |
| + B1 stream-K                 | 2300 (MMQ)  | 61% (still slower than HGEMM) |
| + C1 prefill graph (HGEMM)    | ~3100 (HGEMM) | 82% |
| + A1 CUTLASS Q8 dequant fuse  | **~3500** (custom HGEMM-equiv) | **~93%** |
| + further inner-loop tuning   | ~3600+      | ~95–98% |

The constraint "do not increase GPU memory" is fully respected by all of
A1, B1, B2, B3, C1, C2, C3 — none of them maintain a persistent FP16
weight mirror or any weight duplication.

---

5. **MMQ-style prefill (INT8 MMA on raw Q8).** *(In progress.)*
   The prefill gap to llama.cpp (67–84%) is structural: their
   `mul_mat_q` runs INT8 MMA (`m16n8k32.s8.s8.s32`) directly against
   raw Q8_0 weights, bypassing the FP16 dequant + FP32→FP16 activation
   staging that qw3's HGEMM path needs. Selectable via `QW3_MATMUL=mmq`,
   parity-correct top-0 logits. Five tile generations live in
   `src/mmq_q8.cu`:
   - **v1** (16×32 / 1 warp / CTA): 3× slower than HGEMM. No shmem reuse
     across warps, leaves SMs starved.
   - **v2** (64×128 / 4 warps / CTA, per-lane shmem reads + DP4A,
     **default** when MMQ is enabled): hits **2080 tok/s at 4K prefill**
     ≈ 69% of HGEMM. Covers most of the v1 gap.
   - **v3** (same shape, `ldmatrix.x4`/`x2` for fragment loads, opt-in
     via `QW3_MMQ_VERSION=3`): parity-correct but 8–11% slower than v2.
   - **v4** (`QW3_MMQ_VERSION=4`, port of llama.cpp's `mul_mat_q`
     geometry — 8 warps / 128×128 tile, `m16n8k32.s8.s8.s32` MMA,
     `ldmatrix.x4` A frags, pre-staged across 8 K-sub-iters): the
     pre-stage allocates 32 unsigned A + 16 float dA + 64-float `sum[]`
     accumulator per lane. On sm_120a (Blackwell) Q8 with the 255
     register/thread cap this hit the ceiling and spilled 160–208 B per
     thread to local memory — measured **1473 tok/s at 4K, slower than
     v2**.
   - **v5** (`QW3_MMQ_VERSION=5`): the lessons-learned tile.
     - **mmq_x=64** (vs v4's 128) halves the per-lane `sum[]` footprint
       from 64 → 32 floats, dropping spills to 0 B (no-check) / 28 B
       (check) — matches llama.cpp's own q8_0 instantiation profile
       (8–32 B spill at mmq_x=128).
     - Tile fragments are wrapped in `tile<>` structs (port of
       `mma.cuh` layout) so ptxas keeps `A.x[0..3]` as a 4-register
       tuple rather than scattering across four 2D arrays.
     - `mma` and `ldmatrix` use **non-`volatile`** inline asm (matches
       llama.cpp/mma.cuh:946); the `volatile` v4 used blocked ptxas
       from reordering MMAs for ILP across the inner k01 loop.
     - Result: **1542–1667 tok/s at 4K**, slightly faster than v4 but
       still trailing v2. The remaining gap to llama.cpp's 3.6 K tok/s
       (3.3× per CTA) is **stream-K work partitioning** and per-shape
       `mmq_x` tuning, both pending.

   Important diagnostic finding: the project was building for
   `CMAKE_CUDA_ARCHITECTURES=80` (sm_80, Ampere) while running on
   sm_120a Blackwell — pure JIT, no Blackwell instructions, no Blackwell
   shmem bank model. Fixed by setting `-DCMAKE_CUDA_ARCHITECTURES=120a-real`.
   HGEMM (cuBLAS) was unaffected (it picks the arch internally) but our
   in-tree kernels gain a non-trivial speedup just from compiling for
   the right arch.

   The original v5 roadmap (cp.async + repacked SoA weights) was based
   on a misread of llama.cpp's mmq path. Their `mmq.cuh` uses **only
   synchronous shmem fills** via `get_int_b2` (handles the 2-byte FP16
   d-prefix offset with two `.b16` loads), then `ldmatrix.x4` /
   `load_generic` for fragment reads, then `m16n8k32.s8.s8.s32` MMA.
   cp.async appears only in `fattn-mma-f16.cuh` (flash-attention),
   never in matmul. Weights are **never repacked** — they read raw
   34-byte `block_q8_0` blocks. So the next prefill push is:
   - **Stream-K work partitioning** (the actual llama.cpp trick) —
     one CTA per SM, partition along K with an atomic / fixup-buffer
     combine. Cuts the partial tail wave waste that costs ~5–10% at
     our current shapes.
   - **Per-shape `mmq_x` tuning** — llama.cpp picks
     `mmq_x ∈ {8, 16, …, 128}` to minimise the column-tile count for
     the actual batch, e.g. 5 tiles at batch=601 with mmq_x=128 vs 7
     at mmq_x=96. Cheap to add (template-switch by 8-step ladder).

6. **Persistent activation Q8_1 buffer reuse across Q/K/V/gate/up.** *(Done — for reference.)*
   Decode used to re-quantize the input to Q8_1 once per matvec; the
   `q8_0_matvec_fanout` path now hoists the quantization across the
   3+ matvecs that share an input (recurrent QKVA/B, attention QKV,
   FFN gate+up) and reuses the staging buffer.

7. **CUDA graph capture of the decode loop.** *(Done — default.)*
   Every `forward_one_token` after a one-token warmup is captured into
   a `cudaGraph_t` and replayed via `cudaGraphLaunch`. The position
   counter that drops into `kv_append` / RoPE / `attention_decode`
   is handled by re-recording each token and patching the exec via
   `cudaGraphExecUpdate` (re-instantiating on
   `cudaErrorGraphExecUpdateFailure`), the same shape llama.cpp
   uses. The argmax tail is split into `argmax_launch` (kernel +
   async D2H, recorded into the graph) and `argmax_collect` (sync +
   read pinned mirror, after replay). Gated by `QW3_GRAPH=off` for A/B.

8. **DeltaNet recurrent fusion.** *(Open — minor.)* The four kernels
   per recurrent layer (conv, l2, deltanet, norm-gate) total ~3% of
   decode time. With CUDA graphs already on, fusing them into a single
   per-layer launch is mostly a launch-latency micro-win (~0.3 tok/s
   estimated). Easy to do, but small.

9. **(Done — for reference)** F16 KV cache and split-K attention. Both
   shipped above; together with the adaptive NSPLIT they took decode
   at 4350 tokens from 22 → 44 tok/s. Originals kept selectable for
   comparison.

The originals of every ported kernel are kept in tree (selectable via
`QW3_MATVEC=qw3` / `QW3_ATTN=qw3` / `QW3_FUSE_SILU_MUL=off` /
`QW3_FUSE_ADD=off`) so we can keep iterating on our own designs
alongside the llama.cpp ports and A/B the fusions.

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
- **MMVQ Q8_0 × Q8_1 matvec family** (`src/mmvq_q8.cu`, base kernels
  ported from llama.cpp, see `LICENSES/llama.cpp.txt`). Decode
  (batch == 1) quantizes the input to Q8_1 once into a reusable scratch,
  then DP4A against the raw 34-byte Q8_0 weight blocks. Three fused
  variants:
  - `mul_mat_vec_q8_0_kernel`: plain matvec.
  - `mul_mat_vec_q8_0_two_kernel`: two weights × one shared activation,
    writes two outputs. Used for the recurrent fanout's alpha+beta pair
    and the standard-attention K+V pair (same row count).
  - `mul_mat_vec_q8_0_silu_mul_kernel` *(qw3-side fusion)*: gate + up
    matvec → `silu(g) * u` in one launch; FFN SwiGLU writes only the
    n_ff-wide mid buffer. Disable with `QW3_FUSE_SILU_MUL=off`.
  - `mul_mat_vec_q8_0_add_kernel` *(qw3-side fusion)*: matvec with
    `dst += W·x` read-modify-write. Used for `attn_output` and
    `ffn_down` which feed the residual stream. Eliminates a separate
    `add_kernel` launch per layer. Disable with `QW3_FUSE_ADD=off`.

  The original qw3 matvec is kept and selectable with `QW3_MATVEC=qw3`.
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

- **Decode plateau (~89–91% of llama.cpp, flat across context).**
  Post-fusion, decode is 85% Q8 matvec (`mul_mat_vec_q8_0_kernel<1>`
  + the new `mul_mat_vec_q8_0_two_kernel<1>` for fused gate+up). Each
  matvec runs at ~2.5 TB/s effective on a 96 MB weight (above HBM
  peak — partial L2 reuse), so per-call latency is at the HBM
  bandwidth ceiling. Further wins must come from fewer matvec calls
  (Q+K+V fusion for standard attention layers, recurrent QKV+gate
  fusion across heterogeneous shapes) or from non-matvec kernels
  (DeltaNet recurrent fusion is the largest at ~5% of decode time).
- **Prefill: 67–80% of llama.cpp**, peaking at 2K (80%) and dropping
  to 71% at 4K. Two separate gaps: HGEMM via on-the-fly dequant loses
  to direct INT8 MMA on every prompt length (closing this lifts the
  whole curve — v4 weight repack + cp.async is the path), and at ≥ 4K
  the per-query fattn-vec attention scales O(T²) without K/V reuse —
  a tiled flash-attention prefill kernel is the right fix.
- **Numerical drift on greedy.** Greedy on both engines should yield
  identical tokens; today qw3 matches for ~5–22 tokens then drifts. Sources
  are the per-block Q8 → FP16 dequant rounding, FP16 accumulator in HGEMM
  (`CUBLAS_COMPUTE_32F_FAST_16F`), and the manual reductions in the
  recurrent stack. None block correct *generation*; a tight bit-exact match
  against llama.cpp is future work.
