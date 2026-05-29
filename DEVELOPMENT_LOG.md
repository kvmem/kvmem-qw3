# qw3 — Development log & roadmap

This file captures the optimization journey, profiles, abandoned attacks,
and remaining work. The user-facing surface and current numbers live in
[`README.md`](README.md).

---

## Headline trajectory

Both engines run greedy on the same Qwen 3.6 27B Q8_0 model on RTX Pro
6000 Blackwell (sm_120a). Numbers are tok/s; reproduce with
`scripts/long_prompt_sweep.py`.

### Prefill journey (4350-token context; T=4K)

| Stage | Prefill (tok/s) | % of llama.cpp |
|---|---:|---:|
| Baseline (DP4A matvec, batched over T)                              |  100  |   3% |
| + cuBLAS HGEMM (FP16 dequant cache + tensor cores)                  |  602  |  16% |
| + Fused batched recurrent (1 launch / layer / sub-op)               | 1032  |  28% |
| + Ported MMVQ + flash-attention-vec decode                          | 1539  |  41% |
| + F16 KV cache + split-K decode attention                           | 1590  |  42% |
| + Eager Q8 → FP16 weight prewarm at upload time                     | 2875  |  76% |
| + Tiled FA2 prefill (BR=16, vec int4 K/V)                           | 3170  |  84% |
| + m16n8k16.f16 tensor-core FA2 prefill                              | 3371  |  89% |
| + GQA q-head fusion (1 CTA per kv_head, 6 q-heads share K/V)        | 3629  |  96% |
| + K cp.async (post-QK issue, single-buffered)                       | 3680  |  98% |
| + V cp.async + row-major shmem + ldmatrix.x2.trans                  | 3744  |  99% |
| + FA2 v2 (BR=8 ncols2=2 q-head pack, Q-in-regs, BC=64)              | 3800  | 101% |
| + FA2 v2 BC=32 (2 blocks/SM occupancy)                              | 3830  | 102% |
| **+ FA2 v2 BR=16 ncols2=2 (M_TILES=2, default 2026-05-29)**         | **3910** | **104%** |

### Long-context cliff progression (T=65K)

| Stage | Prefill (tok/s) | % of llama.cpp |
|---|---:|---:|
| FA2 v1 + GQA fusion                                       | 1823 | 59% |
| + V cp.async                                              | 2085 | 68% |
| + Selective cp.async wait (V overlaps QK)                 | 2232 | 73% |
| + FA2 v2 (BR=8)                                           | 2367 | 77% |
| + FA2 v2 BC=32 (2 blocks/SM)                              | 2367 | 77% |
| **+ FA2 v2 BR=16 ncols2=2 (default 2026-05-29)**          | **2840** | **92%** |

The BR=16 step closed half of the long-T gap in a single change. The
remaining 7.7% at T=65K is compute-bound (TC util 13% qw3 vs 48% llama)
and blocked behind a register-pressure wall — see "What hit a wall" below.

### Decode journey (4350-token context; llama.cpp baseline 44.77 tok/s)

| Stage | Decode (tok/s) | % of llama.cpp |
|---|---:|---:|
| Baseline (per-token decode, F32 KV)                   | ~22.0 | 49% |
| + F16 KV cache                                        | ~28   | 63% |
| + Ported MMVQ (DP4A Q8_0 × Q8_1)                      | ~33   | 74% |
| + Split-K decode attention (NSPLIT ∈ {1,2,4,8})       | 37.3  | 83% |
| + CUDA graph capture of decode loop                   | 38.5  | 86% |
| + FFN gate+up matvec fusion (two-weight kernel)       | 41.0  | 92% |
| + Adaptive NSPLIT + NSPLIT=16 + SwiGLU/add fusion     | 43.8  | 98% |

The biggest single jump was the **NSPLIT policy rewrite** — at 1K
context the old code ran decode attention with 24 heads × 4 warps total,
4.7% of Blackwell's warp capacity, sequentially walking the full KV
tail. Bumping to per_split ≈ 128 KV tokens (target NSPLIT=8 by 1K,
NSPLIT=16 by 4K) fills the SMs and drops per-call attention 8×.

---

## Decode profile (post-fusion, 1029-token prompt, 64 decode tokens, graphs off)

| Kernel | per-token (ms) | share |
|---|---:|---:|
| `mul_mat_vec_q8_0_silu_mul` (FFN gate+up+silu+mul fused)   | 8.05 | 37% |
| `mul_mat_vec_q8_0_add` (attn_output / ffn_down + residual) | 5.55 | 26% |
| `mul_mat_vec_q8_0` (Q/K+V/recurrent/LM-head matvec)        | 4.49 | 21% |
| `fattn_vec_decode<NSPLIT=8>` (decode attention)            | 0.61 | 3%  |
| DeltaNet recurrent stack (conv + l2 + deltanet + norm_gate)| 0.70 | 3%  |
| RMSNorm (post-attn + post-FFN)                             | 0.38 | 2%  |
| other (RoPE, KV append, argmax, …)                         | 0.4  | 2%  |
| **total**                                                  | **20.2** | **~85% of measured 22 ms graph-off** |

Matvec is **84% of decode** and runs at HBM ceiling (~2.7 TB/s effective
on the FFN weights with partial L2 reuse). Decode attention dropped
from 19% → 3% of decode time — that's the change that mattered most.

## Long-context profile (T=64K / 96K, pre-v2)

`nsys` per-component breakdown:

| Component | qw3 64K | llama 64K | qw3/llama | qw3 96K | llama 96K | qw3/llama |
|---|---:|---:|---:|---:|---:|---:|
| **FA2 attention**   | **17.65 s** | **4.71 s** | **375%** | **39.45 s** | **12.23 s** | **323%** |
| HGEMM matmul        |  7.85 s | 11.29 s |  70% | 11.97 s | 16.83 s |  71% |
| DeltaNet            |  2.55 s |  2.48 s | 103% |  3.81 s |  3.73 s | 102% |
| Recurrent conv      |  1.36 s |  0.35 s | 389% |  —      |  —      |  —   |

Two findings worth pinning:

1. **The long-T gap is FA2 attention only** — it accounts for 100% of
   the +10.6 s gap at 64K and 100%+ of the +23.0 s gap at 96K. Matmul
   actually runs ahead of llama by 3–5 s.
2. **Our HGEMM beats llama's stream-K MMQ at long T** (1.30× faster at
   64K, 1.41× faster at 96K). The matmul side is firmly settled — the
   "MMQ revisit" thesis no longer applies for this length regime.

The mechanism behind the FA2 gap (post-v2 BR=16 reduction) is now
**compute-bound**, not bandwidth-bound. NCU was unavailable on this
host (`RmProfilingAdminOnly=1`); back-of-envelope from wall-time +
FLOPs:

| target   | wall    | TC util |
|----------|--------:|--------:|
| MMA-only |  2.23 s | 100%    |
| qw3 v2   | 17.06 s | 13%     |
| llama    |  4.66 s | 48%     |

llama runs the same FLOPs in 27% of qw3's wall by feeding the tensor
cores 3.66× as densely. The 73% of qw3's "non-MMA" wall (serialized
softmax + ldmatrix + sync waits + address arithmetic) is mostly *not
present* in llama — they hide it behind MMA via pipelining. The
q-rows-per-CTA lever is the most direct way to amortize sync /
ldmatrix / softmax over more compute.

---

## FA2 prefill kernel — full evolution

### v1 (`fattn_prefill_kernel`)

Tiled FA2 with BR=16 queries × BC=32 K/V tile, FP32 inner loop, vec int4
K/V loads. Cut per-call attn 33→22 ms at 4K (+10%) over the baseline
per-query fattn-vec scaling.

### v1 + tensor cores (`fattn_prefill_mma_kernel`, `fattn_prefill_mma_pipe_kernel`)

Promotes the FP32 inner loop to `m16n8k16.f16.f16.f32` MMA — same BR=16
× BC=32 tile, but QK^T and PV both run on tensor cores with FP32
accumulators. Per-call attn 22.8 → 14.8 ms at 4K (+11% prefill end-to-end).
The `mma-pipe` variant adds K-buffer ping-pong via `cp.async`.

### v1 + GQA fusion (`fattn_prefill_mma_gqa_kernel`, default for q_per_kv=6)

The `(q_head, q_block_row)` grid was launching one CTA per q-head; with
GQA group=6 each kv_head's K/V was re-read 6× (~100 MB extra HBM/layer
at 32K). Fused kernel launches one CTA per `(kv_head, q_block_row)` and
processes all 6 q-heads per block. K and V loads hoisted out of the
per-head loop. Prefill at 4350 went 3371→3629 (+8%); at 16545 went
2455→3199 (+30%); at 33076 went 1700→2520 (+48%). The single biggest
long-context lift to date.

### v1 + K and V cp.async with row-major V + ldmatrix.x2.trans

V was originally stored transposed in shmem (`s_VT[d, t]`), forcing strided
writes incompatible with cp.async. Two changes: (1) **K cp.async** issues
the next tile's K immediately after QK consumes the current tile (single-
buffered, `wait_group<0>` at the top of the next iter). (2) **V cp.async**
flips shmem layout to `s_V[t, d]` (row-major), V load uses
`cp_async_cg_16` like K, and the PV-side MMA uses
`ldmatrix.sync.aligned.m8n8.x2.trans.b16` to transpose 8x8 fp16 tiles
during the load. V[t+1] is issued post-PV (when s_V is dead); the overlap
window is the next iter's K wait + Q init + softmax + QK MMA — much wider
than K's overlap (just the current tile's softmax + PV). Standalone V
cp.async wins, K already on: +1.9% / +6.0% / +9.8% / +14.4% at
4K / 16K / 32K / 65K. Combined with K cp.async, 65K prefill went
~1823→2085 tok/s.

### v1 + selective cp.async wait

Top-of-tile `wait_group<1>` drains K only; V drains pre-PV. +4.7% at
T=65K. Argmax-perfect parity, shmem-neutral. First positive intra-CTA
win since split-KV came back negative.

### v2 (`fattn_prefill_mma_gqa_kernel_v2_t`, default for q_per_kv=6)

Mirrors llama.cpp's `flash_attn_ext_f16<DKQ=256, DV=256, ncols1=8,
ncols2=8>` — packs q-heads into the MMA m-axis. **ncols2=2** packs two
q-heads into the m=16 fragment (8 q-tokens × 2 q-heads = 16 fully-utilized
rows; zero waste against Qwen's q_per_kv=6 → 3 CTAs per kv_head). Q
lives in registers (saves 48 KB shmem). The packing layout
`m_in_frag = c * Q_ROWS_PER_TILE + r` is essential — c=0 fills m=0..7
(a-pair), c=1 fills m=8..15 (b-pair), so each m16n8k16 MMA produces
both packed q-heads in a single accumulator. Sync count drops 12→3 per
tile; at T=65K that's 3K syncs/CTA vs v1's 24K (8× drop). v2 is hard-
gated to `q_per_kv == 6`; other GQA shapes fall back to v1.

### v2 BC=32 (2 blocks/SM occupancy, default)

Halving BC from 64 to 32 drops shmem 70→35 KB. With the 96 KB optin cap
and `__launch_bounds__(128, 1)`, this lets 2 blocks reside per SM
instead of 1. HBM stalls in one block are hidden by MMA work in the
other. +6.0% at T=65K, +4.1% at 33K, +2.4% at 16K. Matches llama.cpp's
own `nbatch_fa=32` choice for the same shape.

### v2 BR=16 ncols2=2 (M_TILES=2, default 2026-05-29)

Generalizes v2 to BR ∈ {8, 16}. At BR=8 each K/V ldmatrix B-frag drove
exactly 1 MMA along m. BR=16 (still NCOLS2=2) packs M_TOTAL=32 q-rows
into M_TILES=2 m=16 fragments — same K/V load now consumed by **2 MMAs**
in the inner m-tile loop. MMAs/ldmatrix doubles, MMAs/sync doubles,
MMAs/softmax-call doubles.

Measured win vs BR=8 v2:

| T      | BR=8  | BR=16 | Δ       | BR=16 % of llama |
|-------:|------:|------:|--------:|-----------------:|
|   556  | 2005  | 1986  | -0.9%   | 71.1%  |
|  1098  | 2782  | 2775  | -0.3%   | 84.0%  |
|  2182  | 3477  | 3476  | ~0%     | 96.6%  |
|  4350  | 3830  | 3910  | +2.1%   | **103.6%** |
|  8415  | 3874  | 4010  | +3.5%   | **105.8%** |
| 16545  | 3662  | 3910  | +6.8%   | **105.2%** |
| 33076  | 3119  | 3505  | +12.4%  | 99.1%  |
| 65867  | 2367  | 2840  | +20.0%  | 92.3%  |

ptxas spills 528 B at BR=16 (vs 0 at BR=8) — confirmed off-hot-path
by walking the SASS. Argmax-perfect parity at -n 64.

---

## What hit a wall (negative results)

### FA2 v2 BR=32 NCOLS2=2 (M_TILES=4, M_TOTAL=64)

Tried the same lever again — m=64 q-rows/CTA, 4 MMAs per K/V ldmatrix,
matching llama's exact MMA density. Compiles parity-correct but
**regresses 1–1.5% across all T**. ptxas reports 4 KB spill (vs 528 B at
BR=16) at the 255-reg cap. Q_reg + c_h + O_acc + softmax temporaries
exceed the register envelope simultaneously, so accumulator fragments
spill to local memory inside the QK/PV inner loops — every MMA round
now pays an L1 round-trip. Kept env-reachable via `QW3_PREFILL_FA2_BR=32`
for diagnostic only.

This blocks the obvious follow-on (BR=64 NCOLS2=1) which has the same
M_TOTAL=64 register footprint. Closing the remaining 7.7% gap at T=65K
needs a **structural** change first: O_acc → shmem, or FP16 O accumulator,
or Q-from-shmem-on-demand.

### FA2 v2 grid-axis swap for L2 reuse (neutral)

`(q_blocks, kvg)` vs `(kvg, q_blocks)` within 0.4% at all T. K/V is
read once per tile per call → no inter-CTA reuse to exploit. Locality
reorderings won't move long-T.

### FA2 ldmatrix.x2 on K (neutral)

Within 0.3% at all T. K B-frag load is a tiny fraction of per-tile
time; consolidating 2× pack2h into 1 PTX issue exposed no extra
parallelism. Reverted.

### FA2 prefill split-KV (neutral, infra kept opt-in)

Decode trick didn't transfer: prefill grid is already 16336 CTAs at
T=64K (far past SM saturation). Long-T gap is intra-CTA (sync count,
single-stage cp.async, ldmatrix.x4 on K), not parallelism deficit.

### cuBLAS strided-batched attention prefill (-9% at 4K)

Two `cublasGemmStridedBatchedEx` calls (Q·Kᵀ then P·V, FP16 inputs,
FP32 accumulator) plus an in-place causal-softmax kernel between them.
Result: 2702 vs 2966 tok/s prefill at 4K. The loss is structural — the
full `[T, T_kv]` score matrix costs ~800 MB FP16 round-trip/layer at
T=4K, exceeding what tile-fused FA2 touches by an order of magnitude.
Tile-fused FA2 keeps each Q row's `[T_kv]` scores in registers/shmem
during the row-fused softmax + PV multiply. Blackwell's 1.7 TB/s HBM
is binding, not its tensor cores. Kept selectable for shape regimes
where the tiled kernel under-occupies SMs.

### Internal prefill chunking + graph capture (negative)

Chunked 512 regresses 4K throughput 28% before capture is even
attempted. Multi-stream HGEMM ping-pong wins ~10%, chunking kills it.
Infra kept opt-in.

### MMQ v3/v4/v5 (parity-correct, slower than v2)

- v3 (ldmatrix.x4/x2 fragment loads): 8–11% slower than v2.
- v4 (port of llama's 8-warp 128×128 geometry): 160–208 B spill at 255-
  reg cap → 1473 tok/s at 4K, slower than v2.
- v5 (mmq_x=64, tile<> structs, non-volatile asm): spill-free but still
  ~50% slower than HGEMM. Closing the rest needs stream-K + per-shape
  mmq_x tuning + activation cp.async (none of which were attempted).

The "MMQ revival" thesis was for short-T prefill before the FA2 fixes
landed. At long T, qw3's HGEMM is **30–40% faster than llama's
stream-K MMQ**, so MMQ is no longer on the critical path.

### Activation cache for MMQ (0% benefit)

Theoretical max 0.12% — call-scoped cache showed 0% measured benefit.

### Layout reorderings (LdsB persistent CTA, etc.)

Confirmed neutral by both grid-axis swap and split-KV — L2 isn't the
constraint.

---

## Why not just plug in flash-attention?

This was the obvious first thought. We checked. None of the established
FA2/FA3 kernels are drop-in for Qwen 3.6 27B's shape on RTX Pro 6000:

| Kernel                          | HEAD_DIM=256? | sm_120a? | Q8_0/FP16 cache?                  |
|---|:---:|:---:|---|
| Dao-AILab `flash-attn` v2       | yes (sm_80–sm_90) | **no** — sm_90+ only for HEAD_DIM≥192 | FP16/BF16 cache only |
| Dao-AILab `flash-attn` v3       | sm_90+ only       | no       | FP16/BF16 cache only             |
| cuDNN `cudnnFlashAttention`     | sm_90 only at 256 | partial — 64/128 only on sm_120 | FP16/BF16 cache only |
| CUTLASS Examples 41 (FMHA)      | yes               | no — Hopper WGMMA only            | FP16/BF16 cache only |
| xformers `memory_efficient`     | wraps the above   | inherits the same gates          | FP16/BF16 cache only |

The hard constraints in our setup that knock these out:

- **HEAD_DIM = 256.** Qwen 3.6 standard-attn layers use the larger 256
  head dim. FA2/FA3 gate this on Hopper because of shmem pressure;
  consumer Blackwell (sm_120a) is not in their support matrix for
  HEAD_DIM≥192 either.
- **GQA group = 6** (24 q-heads / 4 kv-heads). Off-the-shelf kernels
  expose `num_kv_heads` but don't fuse the q-head loop into one CTA
  — the +30–48% wins from `fattn_prefill_mma_gqa_kernel` came
  specifically from sharing K/V across all 6 q-heads in one block,
  which is not a tunable in the public kernels.
- **GGUF Q8_0 weights and the existing FP16 K/V cache.** Plugging in
  flash-attention would mean either copying the cache through a
  separate layout per call (a write the size of all KV) or maintaining
  a parallel cache, which conflicts with the "do not increase GPU
  memory" constraint.
- **llama.cpp's own FA2 (`fattn-mma-f16.cuh`) is bespoke for the same
  reasons** — they don't link Dao's flash-attention either, they
  hand-roll a split-KV kernel that respects ggml's quant-cache layout.

Conclusion: there is no general-purpose FA2 kernel that fits the
(HEAD_DIM=256, GQA=6, sm_120a) cell. The hand-rolled trajectory is the
only realistic path.

---

## Decode optimizations (default, all on)

### Adaptive decode-attention split-K

`pick_nsplit` returns `NSPLIT=1` for `seq_len ≤ 64`, then targets
per_split ≈ 128 KV tokens: `{≤64: 1, ≤128: 2, ≤512: 4, ≤2048: 8, ≤8192:
16, ≤32768: 32, else 64}`. At 1K context per-call attention dropped
315 → 39 µs; at 4K with `NSPLIT=16` another 2× over NSPLIT=8. Decode at
1K context went from 40 → 45 tok/s. At 128K, widening NSPLIT to 32/64
lifted decode from 46% → 85% of llama.cpp. Override with
`QW3_FATTN_NSPLIT=…` for parity diffs.

### Fused FFN SwiGLU matvec

`mul_mat_vec_q8_0_silu_mul_kernel` reads gate and up weights with shared
activations, then writes `silu(g) * u` directly to `ffn_mid`. Eliminates
one full kernel launch per layer (`silu_mul_kernel`) and the two n_ff-
wide intermediates' round-trip through DRAM. Disable with
`QW3_FUSE_SILU_MUL=off`.

### Fused matvec + residual add

`mul_mat_vec_q8_0_add_kernel` writes `dst[row] += W·x` with
read-modify-write at the output store. Used for attn_output and
ffn_down. Saves 128 `add_kernel` launches per token and removes the
n_embd-wide `attn_out` / `ffn_out` scratch round-trip. Disable with
`QW3_FUSE_ADD=off`.

### CUDA graph capture of decode loop

Every `forward_one_token` after a one-token warmup is captured into a
`cudaGraph_t` and replayed via `cudaGraphLaunch`. Position counter is
patched per token via `cudaGraphExecUpdate`. Argmax tail is split into
`argmax_launch` (kernel + async D2H, recorded into the graph) and
`argmax_collect` (sync + read pinned mirror, after replay). Gated by
`QW3_GRAPH=off` for A/B.

### Persistent activation Q8_1 buffer reuse

Decode used to re-quantize the input to Q8_1 once per matvec; the
`q8_0_matvec_fanout` path now hoists the quantization across the 3+
matvecs that share an input (recurrent QKVA/B, attention QKV, FFN
gate+up) and reuses the staging buffer.

### F16 KV cache + `KVT` template

Each `kv_append_*` kernel writes FP16, and the fattn-vec decode kernel
is templated on `KVT ∈ {float, __half}` with on-the-fly `__half2float`
reads. Halves per-token K/V bandwidth at decode. Set `QW3_KV_DTYPE=fp32`
to revert.

### X-FP16 cache across consecutive HGEMMs

`hgemm_q8` runs `fp32_to_fp16_kernel` on its FP32 input X every call.
The transformer's QKV pattern (and FFN gate/up pair) calls `hgemm_q8`
3× and 2× respectively in immediate succession with the *same* `x_ptr`
— the FP32→FP16 conversion of the layer input was redundant. Added a
1-slot cache keyed on `(x_ptr, x_elems)` that skips the conversion
launch on hit. At 4K context, prefill 2970 → 3024 tok/s (+1.8%). Disable
with `QW3_HGEMM_X_CACHE=0`.

---

## Baseline profile (601-token prefill, HGEMM default, graphs off)

| Kernel                          | Total ms | % of GPU | Per-call median |
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
85 ms + mmq<96> 26 ms + fixup 3.8 ms) — i.e. ~37 ms (17% of qw3's prefill
time) is **literally the dequant + activation-staging tax** the HGEMM
path pays that llama.cpp's `mul_mat_q` avoids by reading raw Q8_0
directly.

---

## Roadmap — open work

Priority is set by impact on the worst-T (T=65K, currently 92.3% of
llama.cpp). Items without an estimated impact are exploratory.

### High impact

**1. FA2 v2 BR=64 NCOLS2=1 with O_acc → shmem (or FP16)**
   *(blocked on accumulator-relocation prerequisite)*
   The q-rows-per-CTA lever still has 2× left but hits the 255-reg
   cap with the current accumulator layout. Moving O_acc to shmem
   (16 KB at M_TOTAL=64, HD_PER_WARP=64) frees ~512 B/thread of
   register pressure. Alternative: FP16 O_acc accumulator (matches
   llama at long context). Expected: another +5–8% at T=65K, closing
   most of the remaining gap.

**2. CUTLASS Q8 → FP16 dequant fusion** *(biggest single prefill win)*
   Replace the global-load fragment of CUTLASS's `tn_align8` SM80
   GEMM with a fused Q8_0 → FP16 dequant in the shared-memory staging
   step. Eliminates the 17% of prefill time spent on `q8_dequant_f16`
   + `fp32_to_fp16_kernel`. Expected: +15–20% prefill at 4K, ≈108%
   of llama.cpp. Medium-large effort; constraint: must respect
   "do not increase GPU memory" (no FP16 weight mirror).

### Medium impact

**3. CUDA graph capture for prefill (whole-prompt, not chunked)**
   Decode is captured; prefill isn't. ~200 kernel launches at large
   batch × ~3.5 µs launch latency = ~700 µs/forward of pure overhead.
   Same shape as the decode capture: one graph per (batch, layer_count)
   tuple, re-instantiated on shape change. Internal chunking + capture
   was already tried and is hostile (-28% at 4K — see negatives).
   Whole-prompt capture is the right path. Expected: +2–5% prefill at
   large batch.

**4. Stream-K work partitioning for MMQ**
   Today MMQ's 200 CTAs × 1 col-tile-wide on 188 SMs gives 1.06 waves
   + a 12-CTA tail wave (6% SM utilization). Stream-K (one persistent
   CTA per SM, partition along K with fixup) bumps utilization to
   ~100%. Expected: +25–35% MMQ throughput → still slower than HGEMM
   at long T, so this only matters if MMQ becomes faster than HGEMM
   for some shape regime.

**5. Per-shape `mmq_x` ladder**
   Template instantiations for `mmq_x ∈ {8, 16, 24, …, 128}` with a
   runtime dispatch picking the value that minimises tile count for
   the actual batch. Cheap; only matters for MMQ.

### Low impact

**6. DeltaNet recurrent fusion**
   Four kernels per recurrent layer (conv, l2, deltanet, norm-gate)
   total ~3% of decode time. Fusing into a single per-layer launch is
   a launch-latency micro-win. Expected: +0.3 tok/s decode.

**7. Layer-pipeline fusion (RMSNorm + residual + KV append)**
   Combine the post-attn RMSNorm with the residual add and KV append
   into a single launch. ~2% of decode time, mostly launch overhead.

**8. Keep activations FP16 end-to-end during prefill**
   `fp32_to_fp16_kernel` is 3.8% of GPU time (8.2 ms / 601-tok prefill).
   If activations stay FP16 through the pipeline (matches Qwen native
   dtype), this kernel drops entirely. Risk: changes precision of every
   prefill op. Expected: +2–3% prefill, but invasive.

### Bug/correctness

**9. Fix qw3 failure at T=128K**
   `q8_0_get_rows_batch invalid arg` at very long contexts. Affects
   parity testing and the upper end of the long sweep.

**10. Numerical drift on greedy.**
   Greedy on both engines should yield identical tokens; today qw3
   matches for ~5–22 tokens then drifts. Sources are the per-block
   Q8 → FP16 dequant rounding, FP16 accumulator in HGEMM
   (`CUBLAS_COMPUTE_32F_FAST_16F`), and the manual reductions in the
   recurrent stack. None block correct *generation*; a tight bit-exact
   match against llama.cpp is future work.

---

## Constraints

These shape every optimization decision. They override impact arguments
when they conflict:

- **"Do not increase GPU memory"** — no persistent FP16 weight mirror
  (would be 54 GB on Qwen 3.6 27B, defeats Q8). Activation scratch is
  fine; weight duplication isn't. Knocks out CUTLASS dequant variants
  that maintain a parallel weight buffer.
- **"Q8 weight storage must stay 8-bit"** — corollary of above. Dequant
  must be on-the-fly into a ping-pong scratch.
- **"Don't port llama.cpp kernels"** — kernels can't be ported
  independently of ggml infra (mma.cuh, cp_async helpers, fattn-common,
  ggml_type dispatch). Ports inhibit Blackwell/quant-format extension.
  Use as reference; rewrite in qw3 primitives.
- **"Match llama.cpp at any inputs at memory parity"** — the win
  condition is the full sweep (556–65K), not the headline number.
  Short-T regressions are real regressions even if long-T wins are
  big.
