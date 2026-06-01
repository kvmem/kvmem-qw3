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
| + FA2 v2 BR=16 ncols2=2 (M_TILES=2)                                 | 3910  | 104% |
| + chunk=2048 default (memory parity), HGEMM autotuner restart tax   | 3399  |  90% |
| + MMQ v7 swizzled at small batch (auto: batch ≤ 512 → MMQ)          | 3520  |  94% |
| **+ MMQ v8 at large batch (auto: batch ≥ 128 → v8, replaces HGEMM)** | **3711** | **99%** |

### Long-context cliff progression (T=65K)

| Stage | Prefill (tok/s) | % of llama.cpp |
|---|---:|---:|
| FA2 v1 + GQA fusion                                       | 1823 | 59% |
| + V cp.async                                              | 2085 | 68% |
| + Selective cp.async wait (V overlaps QK)                 | 2232 | 73% |
| + FA2 v2 (BR=8)                                           | 2367 | 77% |
| + FA2 v2 BC=32 (2 blocks/SM)                              | 2367 | 77% |
| + FA2 v2 BR=16 ncols2=2                                   | 2840 | 92% |
| + chunk=2048 default (HGEMM autotuner tax)                | 2766 | 90% |
| **+ MMQ v8 default (replaces HGEMM, frees ~3 GiB scratch)** | **2772** | **90%** |

The BR=16 step closed half of the long-T gap in a single change, and
the v8 default removes the FP16 batch scratch without throughput cost
at this length. The remaining 9.6% at T=65K is FA2 compute-bound
(NCU-measured Tensor pipe util: 23.3% qw3 vs 65.1% llama, both at
identical 16.7% theoretical occupancy, DRAM at 2.07% of peak — *not*
HBM-bound). See "v2 long-T residual gap, NCU-confirmed" below.

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

## Short-prompt profile (T=556, qw3 70% of llama)

The prefill table shows qw3 at 70.4% of llama at T=556 and 83.7% at
T=1098, then ≥97% from T=2K onward. The gap is launch-cost-driven, not
compute-bound. `nsys` cuda_gpu_kern_sum at T=726 (-n 4, no warmup) over
~564 ms wall clock:

| Kernel                           | Total (ms) | Calls | Avg (µs) | Share |
|---|---:|---:|---:|---:|
| `Kernel2` (cuBLAS HGEMM autotuner)        | 111.5 | 496 | 224.8 | 44.4% |
| `q8_dequant_f16_kernel`                   |  48.3 | 496 |  97.3 | 19.2% |
| `gated_delta_net_kernel`                  |  27.8 |  48 | 578.8 | 11.1% |
| `recurrent_conv_batch_kernel`             |  22.6 |  48 | 469.9 |  9.0% |
| `mul_mat_vec_q8_0_silu_mul_kernel` (decode) |  8.2 | 64 | 128.3 | 3.3% |
| `fattn_prefill_mma_gqa_kernel_v2_t`       |   2.4 |  16 | 148.5 |  0.9% |
| _(other)_                                 |  29.9 |  —  |   —   | 12.1% |

Key observations:

1. **Two-thirds of T=556 prefill time is HGEMM dispatch + dequant**:
   `Kernel2 + q8_dequant_f16_kernel` together = 159.8 ms (63.6%). At
   T=4K both kernels run ~3× longer per call (matmul m grows 5×, dequant
   identical) but the *total* wall time only doubles, so their fraction
   of prefill drops to ~30%. By T=16K dequant drops below the 1% noise
   floor and HGEMM is amortized across actual compute.
2. **DeltaNet recurrent kernels are per-T fixed**: 27.8 ms / 48 layers
   = 0.58 ms/layer at T=556, vs 35 ms / 48 = 0.73 ms/layer at T=4K (the
   prefill kernel is sequential along T, so wall time scales linearly).
   At T=556 they account for 11.1% (28 ms / 252 ms prefill); at T=4K
   they account for ~3% (35 ms / 1200 ms prefill).
3. **FA2 itself is fast (0.9% at T=556)**: 16 calls × 148 µs = 2.4 ms
   total. The per-launch overhead floor (~3.5 µs) dominates over the
   compute at this length.

llama.cpp avoids HGEMM entirely (it uses `mul_mat_q` MMQ stream-K) so
it skips the cuBLAS heuristic-search Kernel2 cost. It also captures
the full prefill in a CUDA graph and amortizes per-launch cost across
the whole prompt — qw3's per-prompt graph capture is the path to
closing this gap (task #35 in the roadmap).

The compute side has no slack to cut at T=556: HGEMM is already at
its tile-undersaturation floor (one CTA per matmul fits one SM
column). Faster kernels won't help here. Two attacks remain:
- **Replace HGEMM with MMQ across the board** (kills both Kernel2
  cost and dequant). Earlier MMQ work (#33-#37) showed v2 is ~74% of
  HGEMM at long T but ~110% at short T; promoting MMQ at small batch
  is conditionally a net win at T<2K.
- **Whole-prompt CUDA graph capture** (#35). llama re-uses captured
  graphs across replay, paying launch overhead once per ~512-token
  chunk. qw3's HGEMM ping-pong dependency makes capture infeasible
  without first dropping HGEMM (so #35 is blocked on the MMQ win).

## T=128K crash (Resolved 2026-05-29)

T=131K was failing with `cuda q8_0_get_rows_batch: invalid argument`.
Root cause was **silent OOM in `CudaTensor`'s constructor**: the
batch-scratch tensors at T=131K total ~85 GiB (h, ffn_gate/up/mid,
attn buffers, recurrent state buffers all allocate at full batch size),
exceeding the ~95 GiB free after weights+KV cache. `cudaMalloc` returned
`cudaErrorMemoryAllocation`, the constructor stored a null pointer
without checking, and the next kernel launch surfaced the error from
`cudaGetLastError()` at the wrong call site.

Two fixes (`src/kernels_cuda.cu`, `src/qwen_executor.cpp`):

1. `CudaTensor` now throws `std::runtime_error` with a message naming
   the failed tensor and requested size when `cudaMalloc` fails. No
   more silent null pointers.
2. `forward_n_tokens` auto-picks a chunk size when the full prompt
   wouldn't fit. The executor queries `backend_.free_device_bytes()`,
   computes per-token batch-scratch bytes, and caps the chunk at 80%
   of free memory. At T<128K this is a no-op (chunk = total); at
   T=131K it falls back to chunk≈8K and pays the ~10% chunking
   overhead documented in [[feedback-prefill-chunking-negative]] in
   exchange for not OOMing. Manual override via `QW3_PREFILL_CHUNK=N`
   still wins.

Result: T=131720 sweep now runs at 2018 tok/s (87.8% of llama 2299).
The remaining 12% gap is a mix of the chunking overhead, FA2 v2 long-T
compute-bound regime (same as T=65K), and KV-cache HBM at this length.

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

### v2 long-T residual gap, NCU-confirmed (2026-06-01)

Earlier sections estimated the long-T gap as "TC util ~13% qw3 vs 48%
llama" — derived from peak-FP16 throughput assumptions, with a ~1.3×
error band on the absolute number. Replaced that with direct ncu
measurement at T=65K, capture point kv≈26K, kernel role matched
(qw3 v2 vs llama `flash_attn_ext_f16<256, 256, 8, 8>`), 4 invocations
each, identical 16.7% theoretical occupancy on both sides.

| Section: GPU Speed Of Light  | qw3 v2 | llama  | reading                            |
|------------------------------|-------:|-------:|------------------------------------|
| Tensor pipe utilization      | 23.3%  | 65.1%  | qw3 stalls 2.79× more often        |
| Compute (SM) Throughput      | 50.8%  | 65.1%  | qw3 below NCU's 60% latency floor  |
| Memory Throughput (L2-dom.)  | 57.5%  | 82.6%  | qw3 leaves L2 idle                 |
| L2 Cache Throughput          | 57.5%  | 82.6%  | same — L2 is the SoL component     |
| L1/TEX Cache Throughput      | 57.6%  | 52.0%  | similar; not the bottleneck        |
| **DRAM Throughput**          | **2.07%** | **5.97%** | **decisively NOT HBM-bound**  |
| Theoretical occupancy        | 16.7%  | 16.7%  | same reg + shmem budget regime     |
| L1 uncoalesced shmem waves   | 45%    | 27%    | qw3 has shmem-pattern slack        |
| Per-block elapsed cycles     | 5.7M   | 723K   | 7.9× — only 2× from q-pack         |
| NCU rule fired               | "below 60% indicates latency issues" | "Tensor well-utilized" | — |

Two settled questions and one open lever fall out of this:

**1. HBM/TMA attacks are off the lever list.** DRAM throughput at 2.1%
of peak (and llama at 5.97%) means the long-T gap is *not* a memory-
bandwidth shortfall. Plans like multi-stage cp.async (nstages=2),
flashinfer-style TMA bulk loads, or FA3-style producer/consumer warp
specialization would each cost shmem (halving block-per-SM occupancy)
in exchange for hiding latency that isn't on the critical path.
Permanently dropped from the candidate list — see also
`feedback_fa2_multi_stage_cpasync_deferred.md`.

**2. Occupancy-budget reductions are off the lever list.** Both kernels
hit 16.7% theoretical occupancy (= 2 warps/scheduler), capped by the
same reg + shmem footprint. Reducing qw3's footprint cannot unlock more
parallelism unless we cross the next occupancy step (= 3 warps/sched =
25%), and our two attempts to do so (FP16-O accumulator standalone, and
BR=64 NCOLS2=1 with O_acc → shmem) both failed with regressions or
parity loss. See `feedback_fa2_v2_br32_spill_negative.md`,
`feedback_fa2_v3_br64_ncols2_1_negative.md`,
`feedback_fa2_fp16_o_standalone_negative.md`.

**3. The remaining structural lever is what llama actually does:**
NCOLS2=8 with q-head zero-padding. Qwen 3.6's gqa_ratio = 6, so
NCOLS2=8 wastes 2/8 = 25% of m-rows on zero-padded heads. In exchange,
M_TOTAL = BR × NCOLS2 doubles from qw3's 32 to llama's 64 q-rows/CTA,
which directly halves the softmax-phase amortization cost per K/V load
— the lever NCU-confirmed at 23.3% → 65.1% Tensor utilization. Llama's
65.1% is the realistic cap if we replicate that geometry exactly.

The smaller, separable lever is the 45% → 27% uncoalesced-shmem-
wavefronts gap — at the moment K/V are pad-8 swizzled (KPAD=8, see
`feedback_fa2_v2_kv_pad_8.md`) but s_S/s_P softmax/probability tiles
are not. XOR-swizzling those should consume some of the L2 + tensor
slack without changing geometry.

NCU run files preserved for replay:
- `/tmp/ncu_run.sh` — qw3 v2 capture (launch-skip 200, count 4)
- `/tmp/ncu_run_llama_only.sh` — llama with `--no-warmup` + broad regex
  (`flash_attn_ext_f16` matches function base name only; ncu does not
  see template args), launch-skip 200 to land kv≈26K.

---

## MMQ matmul evolution (chunk=2048 era)

Before chunk=2048 became the default, prefill matmul was cuBLAS HGEMM
with a per-call Q8_0 → FP16 dequant ping-pong scratch (constraint:
"Q8 weight storage must stay 8-bit"). HGEMM beat MMQ at every batch
size we'd implemented because Cutlass's wider tiles outpaced our
INT8 path.

The chunk=2048 default (memory parity with llama.cpp at long context)
created a new pressure: the matmul side runs at modest batch (2048
× hidden), where HGEMM's autotuner picks small tiles that under-utilize
SMs, and the per-call dequant tax stays roughly constant. MMQ — if we
could make it competitive — would also remove the FP16 batch scratch
allocation entirely, a ~3 GiB win at chunk=2048.

Two kernels closed this:

### MMQ v7 (small-batch optimum — 64×64 tile, 2 blocks/SM)

Built on three pieces of plumbing:

- **Split-plane Q8 weight layout.** Per row, the FP16 scales for all
  K/32 blocks live in one contiguous span, the INT8 quantized values
  in another. Lets the inner loop do contiguous cp.async loads of
  the INT8 quants without hop-and-skip across 34-byte interleaved
  blocks.
- **144-byte `block_q8_1_mmq_t` activation layout.** Activations
  staged as super-blocks of 4×32 INT8 values + 4 FP32 sub-block
  scales (16 B header + 128 B quants = 144 B). This matches what
  llama.cpp's MMQ consumes; aligns with the 128-element K-stride of
  the 8-warp tile.
- **2-stage cp.async on W and Y, m16n8k32 INT8 MMA.** Inner loop
  pipelines the next K-stride load behind the current MMA, then
  rescales accumulators with the per-sub-block FP32 scales.

The crucial unlock was **XOR-swizzled shmem** (`byte_off ^=
(row & 7) << 4` on both cp.async fill and inner-loop reads): without
it, all 256-element K rows anchor at bank 0, causing 8-way LDS bank
conflicts that LDSM has to pay through serialized ports. The swizzle
gives 8 group_id lanes 8 distinct banks. With it, v7 ≥ HGEMM at every
T at chunk=512.

v7's tile geometry is 64×64 outputs per CTA (4 warps, 128 threads,
36 KB shmem, 2 blocks/SM resident). Wins at small batch by saturating
the grid early — at T=556 with M=2304, the v7 grid is 36 × 9 = 324
CTAs across 2 blocks/SM = 162 SM-waves, fully utilized.

### MMQ v8 (large-batch optimum — 128×128 tile, 1 block/SM)

v7's plumbing on a 2× wider tile geometry. 8 warps split as 4 stripes
(32-row × 2-band 64-col), NTX=2 NTJ=8 → each warp does 16 m16n8k32
MMAs per sub-block × 4 sub-blocks = 64 MMAs per super-block. 250-255
regs (capped, 0 spills) and 72 KB dynamic shmem, so 1 block/SM.

Why the wider tile helps at large batch: per-CTA arithmetic intensity
scales with `(M·N)/(M+N)`. v7's 64×64 produces 32 outputs per byte
loaded along K; v8's 128×128 produces 64. At T=1K v8 is +33% over
HGEMM where v7 is roughly even. At T=65K both v7 and v8 approach
HBM ceiling, so v8's ~0.4% lead over HGEMM is small but the +0.2 GiB
memory win (no FP16 scratch) is the headline.

Critical wiring fix during integration: `mmq_uses_mmq_y_layout()` in
`kernels_cuda.cu` switches the activation scratch format based on the
selected version (36-int for v2/3/6; 144-B for v4/5/7/8/auto). When
v8 was added but the layout filter wasn't updated, qw3 staged 36-int
activations but v8 read them as 144-B blocks → silent argmax garbage.
The unit test (`tests/mmq_parity.cu`) couldn't catch it because it
stages activations directly via `launch_quantize_mmq_q8_1`. Catching
this required a real-prompt argmax check at Qwen shapes. Lesson:
parity tests need to exercise the dispatch path, not just the kernel
in isolation.

### Auto routing (`QW3_MATMUL=auto`, default)

Per-call decision tree:

```
batch ≥ 8 ? ──→ MMQ
              ├── batch ≥ 128 & rows ≥ 128 ? ──→ v8 (128×128 tile)
              └── else                       ──→ v7 (64×64 tile)
batch < 8  ? ──→ dp4a matvec (decode path)
```

Both paths use identical activation format (144-B `block_q8_1_mmq_t`)
and identical Q8_0 weight layout (split-plane). The choice between
them is purely tile geometry — v8 wins above 128 because of higher
arithmetic intensity; v7 wins below because of grid saturation +
occupancy + tile-padding (v8's 128 step wastes 66% of the last column
at T=556 vs v7's 64 step wasting 31%).

HGEMM-with-dequant is no longer in the default path. It's reachable
only via explicit `QW3_MATMUL=hgemm`; that costs +~3 GiB FP16 batch
scratch and is now slower at every batch ≥ 1K.

### A/B vs HGEMM at chunk=2048 (auto path):

| T (tokens) | qw3 prefill (auto MMQ) | qw3 prefill (HGEMM) | Δ vs HGEMM |
|-----------:|-----------------------:|--------------------:|-----------:|
|   1024     |  ~3300 tok/s | ~2490 tok/s | **+32.6%** |
|   2048     |  ~3397 tok/s | ~2891 tok/s | **+17.5%** |
|   4096     |  ~3713 tok/s | ~3399 tok/s | **+9.2%**  |
|   8192     |  ~3753 tok/s | ~3592 tok/s | **+4.5%**  |
|  16384     |  ~3616 tok/s | ~3552 tok/s | **+1.8%**  |
|  32768     |  ~3298 tok/s | ~3275 tok/s | **+0.7%**  |
|  65536     |  ~2777 tok/s | ~2766 tok/s | **+0.4%**  |

Memory at T=65K: 20.7 GiB vs HGEMM 20.9 GiB.

### End-to-end vs llama.cpp (auto path, default config):

| T (tokens) | qw3 prefill | llama prefill | qw3/llama |
|-----------:|------------:|--------------:|----------:|
|       556  |   2412 tok/s |   2748 tok/s |  87.8%  |
|      2182  |   3392 tok/s |   3584 tok/s |  94.6%  |
|      4350  |   3711 tok/s |   3760 tok/s |  98.7%  |
|      8415  |   3753 tok/s |   3787 tok/s |  99.1%  |
|     16545  |   3617 tok/s |   3711 tok/s |  97.5%  |
|     33076  |   3296 tok/s |   3526 tok/s |  93.5%  |
|     65867  |   2772 tok/s |   3066 tok/s |  90.4%  |

### Why no single MMQ kernel wins everywhere

The tile-vs-occupancy curve has two distinct optima on Blackwell:

- **Large batch** wants a wide tile to amortize K-axis loads (high
  per-CTA arithmetic intensity). That forces 1 block/SM (shmem +
  register budget binding on sm_120a).
- **Small batch** wants many small CTAs to saturate the grid +
  expose occupancy hiding for HBM stalls. That forces a small tile.

Trying to do both at once runs into hard limits: 128×128 at 2 blocks/SM
would need ≤50 KB shmem (kills 2-stage cp.async or shrinks K-stride,
both nuke arithmetic intensity); 128×128 at ≤128 regs spills O_acc
(4 KB stack traffic in the inner loop). cuBLAS HGEMM and llama.cpp's
MMQ both ship multiple tile sizes and dispatch by shape — same reason.

A single kernel could in principle replace both via **stream-K
decomposition** (each CTA owns a slice of the K dimension across
multiple output tiles, decoupling tile size from grid size). Task #73
explored this for v7 and was set aside; could revisit with v8 as the
base. Today's 8% gap at T=65K lives in FA2, not MMQ, so the lever has
moved.

---

## What hit a wall (negative results)

### Current state (2026-06-01) — q-rows-per-CTA lever exhausted in qw3 architecture

The FA2 long-T gap (NCU-measured 23.3% Tensor pipe util at T=65K vs
llama's 65.1%, identical 16.7% theoretical occupancy, DRAM at 2% of
peak) points at exactly one structural lever: increase q-rows-per-CTA
so each K/V load amortizes more MMA work. Llama uses M_TOTAL=64 (BR=8
× NCOLS2=8 with q-head zero-pad); qw3's v2 default is M_TOTAL=32
(BR=16 × NCOLS2=2 with no padding).

Every approach to push M_TOTAL=64 in qw3's current shape failed:

| Variant | Geometry | Outcome |
|---|---|---|
| FA2 v2 BR=32 NCOLS2=2 | M_TOTAL=64, 4 MMAs/ks | -1 to -1.5% (4 KB spill at 255-reg cap) |
| FA2 v2 BR=32 NCOLS2=2 + FP16-O | M_TOTAL=64 with FP16 acc | -5/-9/-20% at T=4K/16K/65K (FP16 mantissa loss) |
| FA2 v3 BR=64 NCOLS2=1 + Q-in-shmem | M_TOTAL=64 via Q-shmem | -6 to -37% across T (1 block/SM + Q-shmem traffic) |
| FA2 v4 NWARPS=8 + warp-pair-owns-mtile | M_TOTAL=64 via 8 warps × 4 m-tiles | -0.5/-3/-5/-8.4% at T=4K/16K/33K/65K (1 block/SM, lost SM-level latency hiding) |

Common cause: at HD=256, 256-thread CTA, Blackwell's 65536-reg/SM cap
puts the M_TOTAL=64 register footprint right at 1 block/SM. v2's
M_TOTAL=32 fits 2 blocks/SM, providing block-level latency hiding that
none of the larger geometries can recover.

What still hasn't been tried in this kernel family:
- O_acc → shmem (constraint: shmem 96 KB cap on sm_120a; M_TOTAL=64 ×
  HD=256 × 4 B = 64 KB additional, leaves no room for K+V+S+P)
- Asymmetric warp specialization (producer warps for cp.async only,
  consumer warps for MMA), Hopper-style — sm_120a doesn't expose
  `wgmma`-class accumulator-in-shmem MMAs that make the asymmetry pay
- A completely different kernel family (FlashInfer / FlashAttention3-
  style with TMA + warp-specialized softmax)

The remaining gap at memory parity is structural, not a tuning
oversight. Closing it needs a different kernel skeleton than the
ldmatrix + m16n8k16 chassis we've been iterating on.

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
   large batch, biggest impact at T=556 (where launch overhead is the
   binding constraint, not the HGEMM autotuner anymore).

**4. Stream-K work partitioning for MMQ (single-kernel replacement for v7+v8)**
   Today's auto router dispatches v7 (small batch, 64×64 tile, 2 blocks/SM)
   and v8 (large batch, 128×128 tile, 1 block/SM) — two kernels because
   the tile-vs-occupancy curve has two distinct optima on Blackwell.
   Stream-K decomposition (each CTA owns a slice of the K dimension across
   multiple output tiles, fixup kernel for cross-CTA reduction) decouples
   tile size from grid size, letting one kernel keep v8's arithmetic
   intensity while saturating the grid at small batch. Substantial
   rewrite (cross-CTA reduction via gmem or DSMEM, separate fixup pass,
   scratch memory for partial sums). llama.cpp's MMQ uses this. Currently
   the auto router gives ≥99% of llama at T=8K with no rewrite, so this
   is exploratory — only valuable if v8 can also win at T<2K under stream-K.

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
