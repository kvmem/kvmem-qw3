// SPDX-License-Identifier: MIT
//
// kernel_legacy.cu — TESTED-NEGATIVE prefill attention variants, kept for the
// record but NOT in any default code path.
//
// These FA2 prefill kernels (v3/v4/v5) were measured on RTX Pro 6000
// (sm_120a, Qwen 3.6 27B). Each pushed the q-rows-per-CTA lever past the v2
// default (BR=16 NCOLS2=2, M_TOTAL=32) and each regressed throughput:
//
//   v3  BR=64 NCOLS2=1, Q-in-shmem            -6% / -18% / -37% @ 4K/16K/65K
//   v4  NWARPS=8, warp-pair-owns-mtile,M=64   -0.5% .. -8.4% across T
//   v5  BR=32 NCOLS2=2, O_acc -> s_O fp16      -2% .. -31% across T
//
// They remain reachable only via QW3_PREFILL_ATTN=mma-gqa-v3|v4|v5 for A/B
// diffs; the live path (v1/v2 + the shared NSPLIT combine kernel) stays in
// src/fattn_vec_decode.cu. See feedback_fa2_v3_*.md / _v4_* / _v5_* for the
// full per-T tables. Do not delete: these encode the dead-ends of the long-T
// FA2 search so the lever isn't re-tried blindly.
//
// Linkage note: the mma_detail inline helpers and the NSPLIT combine kernel
// are DUPLICATED here from fattn_vec_decode.cu. The inline helpers are
// internal-linkage (__forceinline__) so duplication is ODR-safe; the combine
// kernel is wrapped in an anonymous namespace so its <HD,2> instantiation does
// not collide with the live v2 instantiation at link time. Only the external-
// linkage launch wrappers + scratch helpers (declared extern in
// kernels_cuda.cu) are the real cross-TU interface.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <type_traits>

#include "cuda_helpers.cuh"

namespace qw3 {
namespace ported {

// ---- duplicated shared device helpers (internal linkage) ----------------
namespace mma_detail {

// PTX wrapper: m16n8k16.row.col.f32.f16.f16.f32. Non-volatile so ptxas can
// schedule MMAs around loads for ILP. Same calling convention as the int8
// MMA in mmq_q8.cu.
__device__ __forceinline__ void mma_m16n8k16_f16f16f32(
        float &c0, float &c1, float &c2, float &c3,
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
#if QW3_MMA_AVAILABLE
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
#else
    (void)c0; (void)c1; (void)c2; (void)c3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

// Pack two consecutive halves at p[0], p[1] into a single .b32 register.
__device__ __forceinline__ unsigned pack2h(const __half *p) {
    __half2 v = *reinterpret_cast<const __half2 *>(p);
    return *reinterpret_cast<unsigned *>(&v);
}

__device__ __forceinline__ unsigned pack2h(__half a, __half b) {
    __half2 v = __halves2half2(a, b);
    return *reinterpret_cast<unsigned *>(&v);
}

// ldmatrix.x4 .b16 — loads a 16x16 fp16 tile from shmem into 4 .b32 regs per
// lane, distributing the result so that each lane holds the A fragment of a
// m16n8k16.row.col MMA (i.e. lane t -> rows {t/4, t/4+8}, cols
// {t%4*2..+1, t%4*2+8..+9}).
//
// Lane addressing: lane provides the shmem address of (row=lane%16,
// col_halves=(lane/16)*8). Stride is in halves between rows. The shmem region
// must be a contiguous 16x16 fp16 tile starting at `base` (i.e.
// base[row*stride_h + col]).
__device__ __forceinline__ void ldmatrix_x4_a(
        unsigned &a0, unsigned &a1, unsigned &a2, unsigned &a3,
        const __half *base, uint32_t stride_h) {
#if QW3_MMA_AVAILABLE
    const uint32_t lane = threadIdx.x % 32;
    const __half *p = base + (lane & 15) * stride_h + ((lane >> 4) * 8);
    unsigned smem = __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)base; (void)stride_h;
#endif
}

// ldmatrix.x2.trans .b16 — loads two 8x8 fp16 tiles from row-major shmem and
// transposes each 8x8 during the load. Result distribution maps directly to
// the B fragment of m16n8k16.row.col (col-major B):
//
//   lane t in [0, 32) holds:
//     b0: (s[k0+2*(t%4)+0, n0+t/4], s[k0+2*(t%4)+1, n0+t/4])  — tile 0
//     b1: (s[k0+2*(t%4)+8, n0+t/4], s[k0+2*(t%4)+9, n0+t/4])  — tile 1
//
// For a row-major source s[t, d] with stride_h halves between K rows, base
// must point to s[k0, n0]. Lanes 0..7 supply tile-0 row addresses (rows
// k0..k0+7 of the column n0..n0+7); lanes 8..15 supply tile-1 row addresses
// (rows k0+8..k0+15). Lanes 16..31 still receive the broadcast result.
__device__ __forceinline__ void ldmatrix_x2_b_trans(
        unsigned &b0, unsigned &b1,
        const __half *base, uint32_t stride_h) {
#if QW3_MMA_AVAILABLE
    const uint32_t lane = threadIdx.x % 32;
    const __half *p = base + (lane & 15) * stride_h;
    unsigned smem = __cvta_generic_to_shared(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0,%1}, [%2];"
        : "=r"(b0), "=r"(b1)
        : "r"(smem));
#else
    (void)b0; (void)b1; (void)base; (void)stride_h;
#endif
}

// cp.async.cg.shared.global of 16 bytes from gmem -> smem. Non-blocking;
// wait_group(0) drains all in-flight commits in this thread. Falls back to a
// vectorized synchronous load on pre-Ampere targets.
__device__ __forceinline__ void cp_async_cg_16(
        __half *smem_dst, const __half *gmem_src) {
#if QW3_MMA_AVAILABLE
    unsigned dst = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(gmem_src));
#else
    *reinterpret_cast<int4 *>(smem_dst) = *reinterpret_cast<const int4 *>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_async_zero_16(__half *smem_dst) {
#if QW3_MMA_AVAILABLE
    int4 z; z.x = z.y = z.z = z.w = 0;
    *reinterpret_cast<int4 *>(smem_dst) = z;
#else
    *reinterpret_cast<int4 *>(smem_dst) = make_int4(0, 0, 0, 0);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if QW3_MMA_AVAILABLE
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if QW3_MMA_AVAILABLE
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

} // namespace mma_detail

namespace {

// Duplicate of the NSPLIT combine kernel (anonymous namespace = TU-local,
// no link collision with the live v2 instantiation in fattn_vec_decode.cu).
// Combine kernel: merges NSPLIT partial (m, l, O_un_normalized) tuples per
// (head, q_token) via online softmax, divides by the merged denominator, and
// writes the final output.
//
// Layout:
//   partials [n_heads, batch, NSPLIT, HEAD_DIM]
//   ms       [n_heads, batch, NSPLIT, 2]   (col 0: m, col 1: l)
//   out      [batch, n_heads, HEAD_DIM]    (out_batch_stride between batches)
//
// Grid: (n_heads, batch). Block: HEAD_DIM threads — one thread per output dim.
template <uint32_t HEAD_DIM, uint32_t NSPLIT>
__global__ void
__launch_bounds__(HEAD_DIM, 2)
fattn_prefill_mma_gqa_combine_kernel(
        float       * __restrict__ out,
        const float * __restrict__ partials,
        const float * __restrict__ ms,
        uint32_t     batch,
        uint32_t     out_batch_stride) {
    const uint32_t head = blockIdx.x;
    const uint32_t b    = blockIdx.y;
    const uint32_t d    = threadIdx.x;
    if (d >= HEAD_DIM) return;

    const uint64_t mb_base = ((uint64_t)head * batch + b) * NSPLIT;

    __shared__ float s_max[NSPLIT];
    __shared__ float s_sum[NSPLIT];
    if (d < NSPLIT) {
        s_max[d] = ms[(mb_base + d) * 2 + 0];
        s_sum[d] = ms[(mb_base + d) * 2 + 1];
    }
    __syncthreads();

    // Global max across non-empty splits.
    float gmax = -INFINITY;
    #pragma unroll
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] > 0.0f && s_max[s] > gmax) gmax = s_max[s];
    }

    // Combine numerator and denominator.
    float gsum = 0.0f;
    float gnum = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < NSPLIT; ++s) {
        if (s_sum[s] <= 0.0f) continue;
        const float w = __expf(s_max[s] - gmax);
        gsum += s_sum[s] * w;
        const uint64_t pbase = (mb_base + s) * HEAD_DIM;
        gnum += partials[pbase + d] * w;
    }

    float * out_ptr = out + static_cast<uint64_t>(b) * out_batch_stride;
    out_ptr[head * HEAD_DIM + d] = (gsum > 0.0f) ? (gnum / gsum) : 0.0f;
}

} // anonymous namespace

// =====================================================================
// fattn_prefill_mma_gqa_kernel_v3_t — BR=64 + NCOLS2=1 + Q-in-shmem
// =====================================================================
//
// NEGATIVE RESULT (2026-05-30): kept opt-in via QW3_PREFILL_ATTN=mma-gqa-v3.
// Parity-correct (top-1 disagreements within v2's own run-to-run drift), but
// throughput regressed -6%/-18%/-37% at T=4K/16K/65K vs v2 default at
// chunk=512. Two compounding losses outweigh the q-rows-per-CTA win:
//   1. 76.75 KB shmem → 1 block/SM, vs v2's 38.4 KB → 2 blocks/SM. Halved
//      occupancy means K cp.async waits stall the whole SM.
//   2. Q-from-shmem: 64 ldmatrix.x4 per tile × tiles at long T, vs v2's
//      once-per-CTA Q load into registers (zero shmem traffic for Q).
//
// Long-T attack: q-rows-per-CTA lever. v2 default is BR=16 NCOLS2=2 →
// M_TOTAL=32 q-rows/CTA, M_TILES=2 MMAs per K/V ldmatrix. llama.cpp
// flash_attn_ext_f16<256,256,8,8> packs 64 q-rows/CTA. v3 mirrors that:
//
//   BR=64, NCOLS2=1, M_TILES=4 → 64 q-rows/CTA, 4 MMAs per K/V load.
//
// Why NCOLS2=1: with BR=64 NCOLS2=2, M_TOTAL=128 doesn't fit shmem at
// HD=256 (s_S alone = 128*BC*4 ≥ 16 KB before P/Q). q_per_kv=6 isn't a
// multiple of NCOLS2=4 either. v3 keeps each m-tile single-headed; the
// grid covers all 6 q-heads via Q_GROUPS_PER_KV=6 CTAs per kv-head.
//
// Why Q-in-shmem (not regs): BR=32 NCOLS2=2 in v2 already spilled 4 KB
// at the 255-reg cap. BR=64 needs more accumulator state (O_acc=128
// fp32) so the Q_reg budget can't survive. Move Q to shmem; reload via
// ldmatrix.x4 per (mt, ks) inside the QK loop. Shmem cost: 32 KB (HD=256).
//
// Shmem (HD=256, BC=32, BR=64, NCOLS2=1):
//   s_K: 16 KB ; s_V: 16 KB ; s_Q: 32 KB ; s_S: 8 KB ; s_P: 4 KB ;
//   m/l/alpha: ~0.75 KB. Total ≈ 76.75 KB ≤ 98304-byte opt-in cap.
//   1 block/SM.
//
// Lane mapping (NCOLS2=1):
//   m_in_frag = c * BR + r:
//     m=0..7  → c=0, r=0..7 (a-pair)
//     m=8..15 → c=1, r=0..7 (b-pair)
//   For NCOLS2=1, BOTH a and b pairs target the same q-head, but
//   different q-row groups within the m-tile:
//     a-pair → q-row mt*16 + group_id           (rows 0..7 of m-tile)
//     b-pair → q-row mt*16 + group_id + 8       (rows 8..15 of m-tile)
//   One MMA m16n8k16 covers 16 q-rows in the m-axis. M_TILES=4 → 64.
//
// Q layout in shmem: row-major s_Q[BR, HD] (64 rows × 256 halves).
// ldmatrix.x4 at base=&s_Q[mt*16, ks*16] with stride=HD loads the
// canonical (m=16, k=16) a-fragment for m-tile mt step ks.
template <uint32_t HEAD_DIM, uint32_t Q_PER_KV, uint32_t BR, uint32_t BC,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_v3_t(
        float       * __restrict__ out,
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    constexpr uint32_t NCOLS2    = 1;

    static_assert(BR == 32 || BR == 64,    "v3 supports BR ∈ {32, 64}");
    static_assert(BC == 32 || BC == 64,    "v3 supports BC ∈ {32, 64}");
    static_assert(BR % MMA_M == 0,         "BR must be multiple of MMA_M");
    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v3 supports HD 128/256");
    static_assert(HEAD_DIM % MMA_K == 0,   "HEAD_DIM multiple of MMA_K");
    static_assert(BC % MMA_N == 0,         "BC multiple of MMA_N");
    static_assert(BC % NWARPS == 0,        "BC divisible by NWARPS");

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t QK_N_TILES = (BC / NWARPS) / MMA_N;     // 2@BC=32
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;                // 2@BC=32
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;        // 64
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;      // 8
    constexpr uint32_t M_TILES     = BR / MMA_M;               // 2@BR=32, 4@BR=64
    constexpr uint32_t M_TOTAL     = BR;                       // NCOLS2=1
    static_assert(M_TOTAL % NWARPS == 0,   "M_TOTAL divisible by NWARPS");

    // Grid: blockIdx.x covers (n_kv_heads * Q_PER_KV) since each q-head is
    // its own CTA (NCOLS2=1, no head-pack into m-axis).
    const uint32_t hg_idx   = blockIdx.x;
    const uint32_t kv_head  = hg_idx / Q_PER_KV;
    const uint32_t q_in_kv  = hg_idx % Q_PER_KV;
    const uint32_t block_q  = blockIdx.y;
    const uint32_t tid      = threadIdx.x;
    const uint32_t warp     = tid / WARP_SIZE;
    const uint32_t lane     = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;
    const uint32_t head     = kv_head * Q_PER_KV + q_in_kv;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // Shmem layout.
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K = reinterpret_cast<__half *>(p); p += BC       * HEAD_DIM * sizeof(__half);
    __half *s_V = reinterpret_cast<__half *>(p); p += HEAD_DIM * BC       * sizeof(__half);
    __half *s_Q = reinterpret_cast<__half *>(p); p += BR       * HEAD_DIM * sizeof(__half);
    float  *s_S = reinterpret_cast<float  *>(p); p += M_TOTAL  * BC       * sizeof(float);
    __half *s_P = reinterpret_cast<__half *>(p); p += M_TOTAL  * BC       * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);

    // ---- Load Q into shmem (row-major [BR, HEAD_DIM]) -------------------
    {
        // 128 threads, BR*HEAD_DIM/2 int4-equivalent halves to fill.
        // Each thread fills 8 halves (one int4) per pass.
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;          // 32@HD=256
        constexpr uint32_t ROWS_PER_PASS = 128 / LANES_PER_ROW;               // 4@HD=256
        constexpr uint32_t N_PASS        = BR / ROWS_PER_PASS;                // 16@BR=64
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t r       = pass * ROWS_PER_PASS + row_in_pass;
            const uint32_t q_idx   = block_q * BR + r;
            const uint32_t base_d  = col_lane * HALVES_PER_LD;
            const bool active      = (q_idx < batch);
            __half *dst = &s_Q[r * HEAD_DIM + base_d];
            if (active) {
                const float *qp = q + (uint64_t)q_idx * q_batch_stride
                                + head * q_stride + base_d;
                __half h8[HALVES_PER_LD];
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    h8[i] = __float2half(qp[i]);
                }
                *reinterpret_cast<int4 *>(dst) = *reinterpret_cast<const int4 *>(h8);
            } else {
                int4 z; z.x = z.y = z.z = z.w = 0;
                *reinterpret_cast<int4 *>(dst) = z;
            }
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    float O_acc[M_TILES][PV_N_TILES][4];
    #pragma unroll
    for (uint32_t mt = 0; mt < M_TILES; ++mt) {
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) O_acc[mt][n][i] = 0.0f;
        }
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_DIM + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_V) {
                __half *dst = &s_V[shm_t * HEAD_DIM + base_d];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_V[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }
    };

    if (USE_CP_ASYNC_K) { issue_K(0); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(0); mma_detail::cp_async_commit(); }
    const uint32_t kv_hi = block_max_kv;

    // ---- Tile loop -------------------------------------------------------
    for (uint32_t t0 = 0; t0 < kv_hi; t0 += BC) {

        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // QK^T phase. Each (ks, nq) loads one K B-frag and runs M_TILES MMAs
        // reusing it (the q-rows-per-CTA lever).
        // For NCOLS2=1, both a-pair and b-pair within an m-tile target the
        // same q-head, different q-row groups. Q is loaded via ldmatrix.x4
        // from s_Q[mt*16 .. mt*16+15, ks*16 .. ks*16+15].
        float c_h[M_TILES][QK_N_TILES][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                c_h[mt][nq][0]=0.f; c_h[mt][nq][1]=0.f;
                c_h[mt][nq][2]=0.f; c_h[mt][nq][3]=0.f;
            }
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            // Load K B-frags for all QK_N_TILES.
            unsigned bb[QK_N_TILES][2];
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t kv_col = warp * (BC / NWARPS) + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_DIM + k0 + in_group * 2];
                bb[nq][0] = mma_detail::pack2h(krow0);
                bb[nq][1] = mma_detail::pack2h(krow0 + 8);
            }
            // Per m-tile: ldmatrix.x4 the Q a-frag, then run MMAs for all nq.
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_Q[mt * MMA_M * HEAD_DIM + k0],
                                          HEAD_DIM);
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    mma_detail::mma_m16n8k16_f16f16f32(
                        c_h[mt][nq][0], c_h[mt][nq][1], c_h[mt][nq][2], c_h[mt][nq][3],
                        a0, a1, a2, a3,
                        bb[nq][0], bb[nq][1]);
                }
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // Write S to shmem with causal mask. c_h[mt][nq] slots:
        //   .0/.1 → m-row m_in_frag=group_id     (q-row mt*16+group_id)
        //   .2/.3 → m-row m_in_frag=group_id+8   (q-row mt*16+group_id+8)
        // s_S layout: M_TOTAL rows × BC cols. m-tile mt occupies rows
        // [mt*MMA_M, mt*MMA_M+16).
        {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * MMA_M + group_id;
                const uint32_t q_idx_b = block_q * BR + mt * MMA_M + group_id + 8;
                const bool a_active = (q_idx_a < batch);
                const bool b_active = (q_idx_b < batch);
                const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
                const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
                const uint32_t m_a = mt * MMA_M + group_id;
                const uint32_t m_b = mt * MMA_M + group_id + 8;
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    const uint32_t base_col = warp * (BC / NWARPS) + nq * MMA_N + in_group * 2;
                    const uint32_t t_a0 = t0 + base_col;
                    const uint32_t t_a1 = t0 + base_col + 1;
                    float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[mt][nq][0] * scale : -INFINITY;
                    float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[mt][nq][1] * scale : -INFINITY;
                    float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[mt][nq][2] * scale : -INFINITY;
                    float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[mt][nq][3] * scale : -INFINITY;
                    s_S[m_a * BC + base_col + 0] = v00;
                    s_S[m_a * BC + base_col + 1] = v01;
                    s_S[m_b * BC + base_col + 0] = v02;
                    s_S[m_b * BC + base_col + 1] = v03;
                }
            }
        }

        __syncthreads();

        // Online softmax over M_TOTAL rows × BC cols.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;     // 16@BR=64
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;       // 1@BC=32, 2@BC=64
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * BC + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * BC + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // Rescale O_acc by per-row alpha. For each m-tile mt, lane group_id
        // holds m=mt*16+group_id (a-side, q-row mt*16+group_id) and
        // m=mt*16+group_id+8 (b-side, q-row mt*16+group_id+8).
        float alpha_a[M_TILES], alpha_b[M_TILES];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            alpha_a[mt] = s_alpha[mt * MMA_M + group_id];
            alpha_b[mt] = s_alpha[mt * MMA_M + group_id + 8];
        }
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                O_acc[mt][n][0] *= alpha_a[mt];
                O_acc[mt][n][1] *= alpha_a[mt];
                O_acc[mt][n][2] *= alpha_b[mt];
                O_acc[mt][n][3] *= alpha_b[mt];
            }
        }

        // Phase B: PV. Each (n, ks) loads one V B-frag and runs M_TILES MMAs
        // reusing it (the q-rows-per-CTA lever).
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    const __half *vbase = &s_V[k0 * HEAD_DIM + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_DIM);
                } else {
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                #pragma unroll
                for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                    unsigned a0, a1, a2, a3;
                    mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                              &s_P[mt * MMA_M * BC + k0], BC);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        O_acc[mt][n][0], O_acc[mt][n][1], O_acc[mt][n][2], O_acc[mt][n][3],
                        a0, a1, a2, a3, b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Final write: divide by l and store [batch, n_heads, HEAD_DIM] --
    {
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * MMA_M + group_id;
            const uint32_t q_idx_b = block_q * BR + mt * MMA_M + group_id + 8;
            const float l_a = s_l[mt * MMA_M + group_id];
            const float l_b = s_l[mt * MMA_M + group_id + 8];
            const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
            const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
            float * const out_a = (q_idx_a < batch)
                ? (out + (uint64_t)q_idx_a * out_batch_stride + head * HEAD_DIM) : nullptr;
            float * const out_b = (q_idx_b < batch)
                ? (out + (uint64_t)q_idx_b * out_batch_stride + head * HEAD_DIM) : nullptr;
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                const uint32_t col = n0 + in_group * 2;
                if (out_a) {
                    out_a[col + 0] = O_acc[mt][n][0] * inv_la;
                    out_a[col + 1] = O_acc[mt][n][1] * inv_la;
                }
                if (out_b) {
                    out_b[col + 0] = O_acc[mt][n][2] * inv_lb;
                    out_b[col + 1] = O_acc[mt][n][3] * inv_lb;
                }
            }
        }
    }
}

static size_t fattn_mma_gqa_v3_smem_bytes(uint32_t head_dim, uint32_t bc, uint32_t br) {
    const uint32_t M_TOTAL = br;                     // NCOLS2=1
    size_t s = 0;
    s += bc       * head_dim * sizeof(__half);   // K
    s += head_dim * bc       * sizeof(__half);   // V
    s += br       * head_dim * sizeof(__half);   // Q (in shmem)
    s += M_TOTAL  * bc       * sizeof(float);    // S
    s += M_TOTAL  * bc       * sizeof(__half);   // P
    s += M_TOTAL             * sizeof(float) * 3;// m, l, alpha
    return s;
}


// =====================================================================
// fattn_prefill_mma_gqa_kernel_v4_t — NWARPS=8 + warp-pair-owns-mtile
// =====================================================================
//
// Long-T attack #2 (after v3 negative). Mirrors llama.cpp's M_TOTAL=64
// q-rows-per-CTA exactly while keeping per-thread Q in registers.
//
// Geometry:
//   NWARPS = 8 (vs v2's 4), 256 threads/CTA
//   BR = 8, BC = 32, NCOLS2 = 2, M_TILES = 4 → M_TOTAL = 64
//   warp_pair = warp / 2 ∈ {0..3} → owns its own m-tile (8 q-tokens × 2 q-heads)
//   warp_in_pair = warp % 2 ∈ {0,1} → owns its BC half (16 cols × 2 MMA_N tiles)
//
// q-head pad: gqa=6 padded to 8 in m-axis. Heads 6,7 produce -INF scores
// (masked out before softmax) → zero contribution to O. M_TOTAL=64 active
// rows of which 8/64 = 12.5% are zero-padded compute (still cheaper than
// re-reading K/V to avoid the pad).
//
// Why warp-pair-owns-mtile (vs v2's all-warps-share-mtiles):
//   v2 NWARPS=4 with M_TILES=4 (BR=32 NCOLS2=2) hits the 255-reg cap
//   because every warp holds Q_reg for ALL 4 m-tiles (256 fp32/thread for
//   Q alone). v3 (BR=64 NCOLS2=1, NWARPS=4) moved Q to shmem to escape;
//   the 32 KB s_Q + ldmatrix.x4 per (mt, ks) cost regressed throughput.
//   v4 keeps Q in registers but partitions m-tiles ACROSS warp pairs:
//   each warp holds Q only for its own m-tile (64 fp32/thread, well
//   under cap).
//
// Why NWARPS=8 (vs v2's 4):
//   With 4 m-tiles × 2 warps each = 8 warps. Each warp pair splits the
//   BC axis (16 cols/warp = 2 MMA_N tiles). NWARPS=4 with warp-pair
//   structure would need M_TILES=2 (just BR=16 NCOLS2=2 = v2 default,
//   no win), so the 8-warp count is forced by the M_TOTAL=64 target.
//
// Per-thread register footprint (vs v2 default BR=16 NCOLS2=2):
//   Q_reg:  v2  M_TILES=2 × KSTEPS=16 × 4   = 128 fp32
//           v4  M_TILES_PER_WARP=1 × 16 × 4 = 64 fp32  (-50%)
//   c_h:    v2  2 × QK_N_TILES=2 × 4        = 16 fp32  (per warp)
//           v4  1 × 2 × 4                    = 8 fp32   (per warp)
//   O_acc:  v2  2 × PV_N_TILES=8 × 4        = 64 fp32
//           v4  1 × PV_N_TILES=16 × 4       = 64 fp32  (HD/2 cols/warp)
//   Net Q+c_h+O ≈ 136 fp32/thread (v2 def: 208). Predicted ~150-180 reg32
//   per thread, 0 spills.
//
// Shmem (HD=256, BC=32, BR=8, NCOLS2=2, M_TILES=4, KV_PAD=8, S_PAD=4):
//   s_K: 32×264×2 = 16.5 KB ; s_V: 16.5 KB
//   s_S: 64×36×4 = 9 KB     ; s_P: 64×40×2 = 5 KB
//   s_m, s_l, s_alpha: 64×4×3 = 0.75 KB
//   Total ≈ 47.7 KB — 2 blocks/SM at sm_120a 100 KB opt-in cap.
//
// Per-tile sync count (BC=32):
//   1× post-K-load (post-cp.async wait)
//   1× post-QK-write s_S (before softmax read)
//   1× post-softmax write s_P (before PV ldmatrix)
//   1× pre-V-load wait_group<0>
//   = 4 syncs/tile (v2 default = 3 syncs/tile; v4's 1 extra is the cost
//   of warp-pair partial-S merge implicit in s_S).
//
// Grid:
//   (n_kv_heads, n_blocks_q, NSPLIT)
//   block.x = NWARPS * 32 = 256
template <uint32_t HEAD_DIM,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V, uint32_t NSPLIT,
          bool USE_FP16_O = false>
__global__ void
__launch_bounds__(256, 1)
fattn_prefill_mma_gqa_kernel_v4_t(
        float       * __restrict__ out,
        float       * __restrict__ partials,    // NSPLIT>1 only
        float       * __restrict__ ms,          // NSPLIT>1 only: [n_heads*batch*NSPLIT*2]
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 8;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;
    constexpr uint32_t Q_PER_KV  = 6;     // active heads (gqa=6)
    constexpr uint32_t Q_PADDED  = 8;     // padded to 8 = NCOLS2 * M_TILES
    constexpr uint32_t BR        = 8;
    constexpr uint32_t BC        = 32;
    constexpr uint32_t NCOLS2    = 2;
    constexpr uint32_t M_TILES   = 4;     // 4 warp pairs × 1 m-tile/pair
    constexpr uint32_t M_TOTAL   = M_TILES * MMA_M;       // 64
    constexpr uint32_t KV_PAD    = 8;
    constexpr uint32_t S_PAD     = 4;     // fp32 elems
    constexpr uint32_t S_STRIDE  = BC + S_PAD;            // 36 fp32
    constexpr uint32_t P_STRIDE  = BC + S_PAD * 2;        // 40 half (=160 B = same byte width as S_STRIDE fp32)
    constexpr uint32_t HEAD_STRIDE = HEAD_DIM + KV_PAD;   // 264

    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v4 supports HD 128/256");
    static_assert(M_TOTAL == NCOLS2 * M_TILES * MMA_M / NCOLS2, "M_TOTAL consistency");
    static_assert(NWARPS == 2 * M_TILES, "v4: 2 warps per m-tile");

    constexpr uint32_t WARPS_PER_PAIR     = 2;
    constexpr uint32_t COLS_PER_WARP      = BC / WARPS_PER_PAIR;       // 16
    constexpr uint32_t QK_N_TILES_PER_WARP = COLS_PER_WARP / MMA_N;    // 2
    constexpr uint32_t QK_KSTEPS          = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t HD_PER_WARP_IN_PAIR = HEAD_DIM / WARPS_PER_PAIR;// 128
    constexpr uint32_t PV_N_TILES         = HD_PER_WARP_IN_PAIR / MMA_N; // 16
    constexpr uint32_t PV_KSTEPS          = BC / MMA_K;                // 2
    constexpr uint32_t Q_ROWS_PER_TILE    = MMA_M / NCOLS2;            // 8 = BR

    // Grid: each kv-head is one CTA along blockIdx.x. The CTA covers all
    // Q_PADDED=8 q-heads of that kv-head (6 active + 2 zero-padded) packed
    // into M_TILES=4 m-tiles. So Q_GROUPS_PER_KV = 1 here (vs v2's 3).
    const uint32_t kv_head = blockIdx.x;
    const uint32_t block_q = blockIdx.y;
    const uint32_t k_split = blockIdx.z;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp    = tid / WARP_SIZE;
    const uint32_t lane    = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;

    const uint32_t warp_pair    = warp / WARPS_PER_PAIR;   // 0..3 = mt
    const uint32_t warp_in_pair = warp % WARPS_PER_PAIR;   // 0,1
    const uint32_t mt           = warp_pair;
    const uint32_t my_bc_lo     = warp_in_pair * COLS_PER_WARP;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    constexpr uint32_t C_A = 0;
    constexpr uint32_t C_B = 1;
    const uint32_t r_a = group_id;       // q-row within tile, a-pair
    const uint32_t r_b = group_id;       // q-row within tile, b-pair

    // Active heads for this CTA: head_base + 0..5 active, head_base+6..7 padded.
    const uint32_t head_base = kv_head * Q_PER_KV;
    // For m-tile mt, this warp-pair owns 2 q-heads (NCOLS2=2):
    //   head_a_idx = mt * 2 + 0 (in [0..7] padded space)
    //   head_b_idx = mt * 2 + 1
    const uint32_t head_a_idx_in_kv = mt * NCOLS2 + C_A;
    const uint32_t head_b_idx_in_kv = mt * NCOLS2 + C_B;
    const bool head_a_active = (head_a_idx_in_kv < Q_PER_KV);
    const bool head_b_active = (head_b_idx_in_kv < Q_PER_KV);
    const uint32_t head_a = head_base + head_a_idx_in_kv;
    const uint32_t head_b = head_base + head_b_idx_in_kv;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    // Split-KV.
    uint32_t kv_lo, kv_hi;
    if (NSPLIT == 1) {
        kv_lo = 0;
        kv_hi = block_max_kv;
    } else {
        const uint32_t kv_total = base_seq_len + batch;
        const uint32_t per_split = ((kv_total + NSPLIT - 1) / NSPLIT + BC - 1) & ~uint32_t(BC - 1);
        kv_lo = k_split * per_split;
        kv_hi = kv_lo + per_split;
        if (kv_hi > block_max_kv) kv_hi = block_max_kv;
        if (kv_lo >= kv_hi) {
            // Dead partial.
            const uint32_t q_idx_a = block_q * BR + r_a;
            const uint32_t q_idx_b = block_q * BR + r_b;
            if (warp_in_pair == 0 && in_group == 0) {
                if (head_a_active && q_idx_a < batch) {
                    const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = -INFINITY;
                    ms[mb * 2 + 1] = 0.0f;
                }
                if (head_b_active && q_idx_b < batch) {
                    const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                    ms[mb * 2 + 0] = -INFINITY;
                    ms[mb * 2 + 1] = 0.0f;
                }
            }
            #pragma unroll
            for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
                const uint32_t col = n0 + in_group * 2;
                const uint64_t pa = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                const uint64_t pb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                if (head_a_active && q_idx_a < batch) {
                    partials[pa * HEAD_DIM + col + 0] = 0.0f;
                    partials[pa * HEAD_DIM + col + 1] = 0.0f;
                }
                if (head_b_active && q_idx_b < batch) {
                    partials[pb * HEAD_DIM + col + 0] = 0.0f;
                    partials[pb * HEAD_DIM + col + 1] = 0.0f;
                }
            }
            return;
        }
    }

    // Shmem.
    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    float  *s_S     = reinterpret_cast<float  *>(p); p += M_TOTAL * S_STRIDE * sizeof(float);
    __half *s_P     = reinterpret_cast<__half *>(p); p += M_TOTAL * P_STRIDE * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    (void)s_alpha;

    // ---- Q registers (one m-tile per warp pair, replicated across the pair).
    // Layout per thread: Q_reg[QK_KSTEPS][4]; the 4 slots are
    // (a0_lo, b0_lo, a0_hi, b0_hi) packed-half pairs that match the
    // ldmatrix.x4 a-fragment layout used in mma_m16n8k16_f16f16f32.
    unsigned Q_reg[QK_KSTEPS][4];
    {
        const uint32_t q_idx_a = block_q * BR + r_a;
        const uint32_t q_idx_b = block_q * BR + r_b;
        const bool a_active = head_a_active && (q_idx_a < batch);
        const bool b_active = head_b_active && (q_idx_b < batch);
        const float *qa = a_active
            ? (q + (uint64_t)q_idx_a * q_batch_stride + head_a * q_stride) : nullptr;
        const float *qb = b_active
            ? (q + (uint64_t)q_idx_b * q_batch_stride + head_b * q_stride) : nullptr;
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            const uint32_t col0 = k0 + in_group * 2;
            const __half qa0 = qa ? __float2half(qa[col0 + 0]) : (__half)0;
            const __half qa1 = qa ? __float2half(qa[col0 + 1]) : (__half)0;
            const __half qb0 = qb ? __float2half(qb[col0 + 0]) : (__half)0;
            const __half qb1 = qb ? __float2half(qb[col0 + 1]) : (__half)0;
            const __half qa8 = qa ? __float2half(qa[col0 + 8]) : (__half)0;
            const __half qa9 = qa ? __float2half(qa[col0 + 9]) : (__half)0;
            const __half qb8 = qb ? __float2half(qb[col0 + 8]) : (__half)0;
            const __half qb9 = qb ? __float2half(qb[col0 + 9]) : (__half)0;
            Q_reg[ks][0] = mma_detail::pack2h(qa0, qa1);
            Q_reg[ks][1] = mma_detail::pack2h(qb0, qb1);
            Q_reg[ks][2] = mma_detail::pack2h(qa8, qa9);
            Q_reg[ks][3] = mma_detail::pack2h(qb8, qb9);
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }

    using o_acc_t = typename std::conditional<USE_FP16_O, __half, float>::type;
    o_acc_t O_acc[PV_N_TILES][4];
    #pragma unroll
    for (uint32_t n = 0; n < PV_N_TILES; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) O_acc[n][i] = (o_acc_t)0;
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = (NWARPS * WARP_SIZE) / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_STRIDE + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = (NWARPS * WARP_SIZE) / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            __half *dst = &s_V_buf[shm_t * HEAD_STRIDE + base_d];
            if (USE_CP_ASYNC_V) {
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                *dst4 = *reinterpret_cast<int4 *>(v8);
            }
        }
    };

    // Prime: K[kv_lo].
    if (USE_CP_ASYNC_K) { issue_K(kv_lo); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(kv_lo); mma_detail::cp_async_commit(); }

    // Skip-out for entirely zero-padded m-tile (heads 6,7). With M_TILES=4
    // and active heads 0..5, mt=3's warp pair handles head_a=6 (active=false)
    // and head_b=7 (active=false). They still cooperate on K/V loads (all
    // warps must issue), but they skip the QK / softmax / PV work and write
    // zeros at writeback time. We collapse this by simply NOT executing
    // the QK accumulator update when both heads inactive — c_h stays zero,
    // and -INFINITY masking in score-write keeps softmax produce 0 P-values.
    // O_acc stays zero throughout.

    for (uint32_t t0 = kv_lo; t0 < kv_hi; t0 += BC) {
        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // ---- QK^T phase. This warp does its m-tile × its BC half (2 nq tiles).
        float c_h[QK_N_TILES_PER_WARP][4];
        #pragma unroll
        for (uint32_t nq = 0; nq < QK_N_TILES_PER_WARP; ++nq) {
            c_h[nq][0]=0.f; c_h[nq][1]=0.f; c_h[nq][2]=0.f; c_h[nq][3]=0.f;
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES_PER_WARP; ++nq) {
                const uint32_t kv_col = my_bc_lo + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_STRIDE + k0 + in_group * 2];
                const unsigned b0 = mma_detail::pack2h(krow0);
                const unsigned b1 = mma_detail::pack2h(krow0 + 8);
                mma_detail::mma_m16n8k16_f16f16f32(
                    c_h[nq][0], c_h[nq][1], c_h[nq][2], c_h[nq][3],
                    Q_reg[ks][0], Q_reg[ks][1], Q_reg[ks][2], Q_reg[ks][3],
                    b0, b1);
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // ---- Write S to shmem with causal mask + head-padding mask.
        {
            const uint32_t q_idx_a = block_q * BR + r_a;
            const uint32_t q_idx_b = block_q * BR + r_b;
            const bool a_active = head_a_active && (q_idx_a < batch);
            const bool b_active = head_b_active && (q_idx_b < batch);
            const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
            const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
            const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
            const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES_PER_WARP; ++nq) {
                const uint32_t base_col = my_bc_lo + nq * MMA_N + in_group * 2;
                const uint32_t t_a0 = t0 + base_col;
                const uint32_t t_a1 = t0 + base_col + 1;
                float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[nq][0] * scale : -INFINITY;
                float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[nq][1] * scale : -INFINITY;
                float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[nq][2] * scale : -INFINITY;
                float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[nq][3] * scale : -INFINITY;
                s_S[m_a * S_STRIDE + base_col + 0] = v00;
                s_S[m_a * S_STRIDE + base_col + 1] = v01;
                s_S[m_b * S_STRIDE + base_col + 0] = v02;
                s_S[m_b * S_STRIDE + base_col + 1] = v03;
            }
        }
        __syncthreads();

        // ---- Online softmax over M_TOTAL=64 rows × BC=32 cols.
        // 8 warps × ROWS_PER_WARP=8 rows each. Each lane covers
        // BC/WARP_SIZE = 1 col.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;       // 8
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;         // 1
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * S_STRIDE + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * P_STRIDE + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // ---- Rescale O_acc by per-row alpha (this warp's m-tile only).
        const float alpha_a = s_alpha[mt * MMA_M + C_A * Q_ROWS_PER_TILE + group_id];
        const float alpha_b = s_alpha[mt * MMA_M + C_B * Q_ROWS_PER_TILE + group_id];
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const float r0 = static_cast<float>(O_acc[n][0]) * alpha_a;
            const float r1 = static_cast<float>(O_acc[n][1]) * alpha_a;
            const float r2 = static_cast<float>(O_acc[n][2]) * alpha_b;
            const float r3 = static_cast<float>(O_acc[n][3]) * alpha_b;
            O_acc[n][0] = static_cast<o_acc_t>(r0);
            O_acc[n][1] = static_cast<o_acc_t>(r1);
            O_acc[n][2] = static_cast<o_acc_t>(r2);
            O_acc[n][3] = static_cast<o_acc_t>(r3);
        }

        // ---- PV phase. Each warp does ITS m-tile × ITS HD half.
        // P A-frag from s_P[mt*MMA_M..(mt+1)*MMA_M, k0..k0+16].
        // V B-frag from s_V_buf row-major; ldmatrix.x2.trans handles the n-axis.
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
            #pragma unroll
            for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                unsigned b0, b1;
                if (USE_CP_ASYNC_V) {
                    const __half *vbase = &s_V_buf[k0 * HEAD_STRIDE + n0];
                    mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_STRIDE);
                } else {
                    // Sync V load fallback (uses transposed layout).
                    const uint32_t out_col = n0 + group_id;
                    const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                    b0 = mma_detail::pack2h(vrow0);
                    b1 = mma_detail::pack2h(vrow0 + 8);
                }
                unsigned a0, a1, a2, a3;
                mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                          &s_P[mt * MMA_M * P_STRIDE + k0], P_STRIDE);
                if constexpr (USE_FP16_O) {
                    float d0 = static_cast<float>(O_acc[n][0]);
                    float d1 = static_cast<float>(O_acc[n][1]);
                    float d2 = static_cast<float>(O_acc[n][2]);
                    float d3 = static_cast<float>(O_acc[n][3]);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        d0, d1, d2, d3,
                        a0, a1, a2, a3, b0, b1);
                    O_acc[n][0] = static_cast<o_acc_t>(d0);
                    O_acc[n][1] = static_cast<o_acc_t>(d1);
                    O_acc[n][2] = static_cast<o_acc_t>(d2);
                    O_acc[n][3] = static_cast<o_acc_t>(d3);
                } else {
                    float &o0 = reinterpret_cast<float &>(O_acc[n][0]);
                    float &o1 = reinterpret_cast<float &>(O_acc[n][1]);
                    float &o2 = reinterpret_cast<float &>(O_acc[n][2]);
                    float &o3 = reinterpret_cast<float &>(O_acc[n][3]);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        o0, o1, o2, o3,
                        a0, a1, a2, a3, b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Writeback / partial dump.
    const uint32_t q_idx_a = block_q * BR + r_a;
    const uint32_t q_idx_b = block_q * BR + r_b;
    const bool a_active = head_a_active && (q_idx_a < batch);
    const bool b_active = head_b_active && (q_idx_b < batch);
    const float l_a = s_l[mt * MMA_M + C_A * Q_ROWS_PER_TILE + group_id];
    const float l_b = s_l[mt * MMA_M + C_B * Q_ROWS_PER_TILE + group_id];
    const float m_a_val = s_m[mt * MMA_M + C_A * Q_ROWS_PER_TILE + group_id];
    const float m_b_val = s_m[mt * MMA_M + C_B * Q_ROWS_PER_TILE + group_id];

    if (NSPLIT == 1) {
        const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
        const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
        float * const out_a = a_active
            ? (out + (uint64_t)q_idx_a * out_batch_stride + head_a * HEAD_DIM) : nullptr;
        float * const out_b = b_active
            ? (out + (uint64_t)q_idx_b * out_batch_stride + head_b * HEAD_DIM) : nullptr;
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
            const uint32_t col = n0 + in_group * 2;
            if (out_a) {
                out_a[col + 0] = static_cast<float>(O_acc[n][0]) * inv_la;
                out_a[col + 1] = static_cast<float>(O_acc[n][1]) * inv_la;
            }
            if (out_b) {
                out_b[col + 0] = static_cast<float>(O_acc[n][2]) * inv_lb;
                out_b[col + 1] = static_cast<float>(O_acc[n][3]) * inv_lb;
            }
        }
    } else {
        // NSPLIT > 1: write un-normalized partials + (m, l) per (head, q_idx, k_split).
        const uint64_t pa_base = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
        const uint64_t pb_base = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
        if (warp_in_pair == 0 && in_group == 0) {
            if (a_active) {
                ms[pa_base * 2 + 0] = m_a_val;
                ms[pa_base * 2 + 1] = l_a;
            }
            if (b_active) {
                ms[pb_base * 2 + 0] = m_b_val;
                ms[pb_base * 2 + 1] = l_b;
            }
        }
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0 = warp_in_pair * HD_PER_WARP_IN_PAIR + n * MMA_N;
            const uint32_t col = n0 + in_group * 2;
            if (a_active) {
                partials[pa_base * HEAD_DIM + col + 0] = static_cast<float>(O_acc[n][0]);
                partials[pa_base * HEAD_DIM + col + 1] = static_cast<float>(O_acc[n][1]);
            }
            if (b_active) {
                partials[pb_base * HEAD_DIM + col + 0] = static_cast<float>(O_acc[n][2]);
                partials[pb_base * HEAD_DIM + col + 1] = static_cast<float>(O_acc[n][3]);
            }
        }
    }
}

static size_t fattn_mma_gqa_v4_smem_bytes(uint32_t head_dim) {
    constexpr uint32_t BR       = 8;
    constexpr uint32_t BC       = 32;
    constexpr uint32_t M_TOTAL  = 64;
    constexpr uint32_t KV_PAD   = 8;
    constexpr uint32_t S_PAD    = 4;
    const uint32_t hd_stride = head_dim + KV_PAD;
    const uint32_t s_stride  = BC + S_PAD;
    const uint32_t p_stride  = BC + S_PAD * 2;
    (void)BR;
    size_t s = 0;
    s += BC      * hd_stride * sizeof(__half);     // K
    s += BC      * hd_stride * sizeof(__half);     // V
    s += M_TOTAL * s_stride  * sizeof(float);      // S
    s += M_TOTAL * p_stride  * sizeof(__half);     // P
    s += M_TOTAL             * sizeof(float) * 3;  // m, l, alpha
    return s;
}

// =====================================================================
// fattn_prefill_mma_gqa_kernel_v5_t — BR=32 NCOLS2=2 + O_acc → s_O fp16
// =====================================================================
//
// q-rows-per-CTA = M_TOTAL = 64 (same as v3/v4) but with v2's lane mapping
// preserved (NCOLS2=2 q-head pack, M_TILES=4 stacked along m-axis). The
// register-pressure relief lever vs v2 BR=32: O accumulator moves from
// register array O_acc[M_TILES=4][PV_N_TILES=8][4] (= 128 fp32 = 128 regs/lane)
// to shared memory s_O[M_TOTAL=64][HD+OPAD=264] (= 33 KB, fp16). During the
// PV inner loop, only the per-(n) accumulator strip lives in regs:
// o_regs[M_TILES][4] = 16 fp32 = 16 regs/lane. Register pressure drops
// 128 → 16 for O, leaving budget for Q_reg[M_TILES=4][QK_KSTEPS=16][4]
// without the 4-KB BR=32 spill that v2 hits at the 255-reg cap.
//
// Tradeoffs:
//   - Shmem: 80.75 KB (vs v2 BR=16: 35 KB). 1 block/SM (vs v2: 2 blocks/SM).
//   - PV per (n) iter: 4 fp16 reads + 4 fp16 writes per (mt, lane) of s_O.
//     Folded with the rescale: read O_old fp16, multiply by alpha_a/alpha_b
//     into o_regs (eliminates v2's separate rescale phase), accumulate
//     PV_KSTEPS=2 MMAs in regs, write back fp16. Net round-trip density:
//     1 fp16↔fp32 round-trip per output element per tile (vs v2's 0;
//     vs v2 USE_FP16_O's PV_KSTEPS=2 round-trips per output per tile).
//   - Bank conflicts: O_PAD=8 halves on row stride breaks 8-way LDS
//     conflict on group_id read pattern (same insight as KV_PAD=8 / S_PAD=4).
//
// FP16 round-trip parity risk: lower than v2 USE_FP16_O standalone because
// the round-trip happens *outside* the ks loop (at the (mt, n) granularity),
// halving the demote/promote density per output. Tested: see
// feedback_fa2_v5_*.md.
//
// Shmem budget at BR=32, BC=32, KV_PAD=8, S_PAD=4, O_PAD=8:
//   K        : 32 × 264 × 2 = 16.5 KB
//   V        : 32 × 264 × 2 = 16.5 KB
//   S (fp32) : 64 × 36  × 4 =  9.0 KB
//   P (fp16) : 64 × 40  × 2 =  5.0 KB
//   O (fp16) : 64 × 264 × 2 = 33.0 KB
//   m,l,a    : 64 × 12      =  0.75 KB
//   total    :              ~ 80.75 KB  (1 block/SM @ 99 KB optin)
template <uint32_t HEAD_DIM, uint32_t Q_PER_KV, uint32_t BR, uint32_t BC,
          uint32_t NCOLS2, uint32_t KV_PAD, uint32_t S_PAD, uint32_t O_PAD,
          bool USE_CP_ASYNC_K, bool USE_CP_ASYNC_V, uint32_t NSPLIT>
__global__ void
__launch_bounds__(128, 1)
fattn_prefill_mma_gqa_kernel_v5_t(
        float       * __restrict__ out,
        float       * __restrict__ partials,
        float       * __restrict__ ms,
        const float * __restrict__ q,
        uint32_t     q_stride,
        const __half* __restrict__ k_cache,
        const __half* __restrict__ v_cache,
        uint32_t     n_heads,
        uint32_t     n_kv_heads,
        uint32_t     base_seq_len,
        uint32_t     batch,
        uint32_t     q_batch_stride,
        uint32_t     out_batch_stride,
        float        scale) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS    = 4;
    constexpr uint32_t MMA_M     = 16;
    constexpr uint32_t MMA_N     = 8;
    constexpr uint32_t MMA_K     = 16;

    static_assert(BR == 16 || BR == 32, "v5 supports BR ∈ {16, 32}");
    static_assert(BC == 32,              "v5 supports BC=32");
    static_assert(NCOLS2 == 2,           "v5 supports NCOLS2=2");
    static_assert(KV_PAD == 8,           "v5 expects KV_PAD=8");
    static_assert(S_PAD  == 4,           "v5 expects S_PAD=4 fp32 elems");
    static_assert(O_PAD  == 8,           "v5 expects O_PAD=8 half elems");
    static_assert(BR % 8 == 0,           "BR must be multiple of 8");
    static_assert(Q_PER_KV % NCOLS2 == 0,"Q_PER_KV must divide by NCOLS2");
    static_assert(HEAD_DIM == 128 || HEAD_DIM == 256, "v5 supports HD 128/256");
    static_assert(HEAD_DIM % MMA_K == 0,"HEAD_DIM multiple of MMA_K");
    static_assert(BC % MMA_N == 0,      "BC multiple of MMA_N");
    static_assert(BC % NWARPS == 0,     "BC divisible by NWARPS");

    constexpr uint32_t S_STRIDE = BC + S_PAD;          // fp32 elements
    constexpr uint32_t P_STRIDE = BC + S_PAD * 2;      // half elements
    constexpr uint32_t O_STRIDE = HEAD_DIM + O_PAD;    // half elements

    constexpr uint32_t QK_KSTEPS  = HEAD_DIM / MMA_K;          // 16
    constexpr uint32_t QK_N_TILES = (BC / NWARPS) / MMA_N;     // 2
    constexpr uint32_t PV_KSTEPS  = BC / MMA_K;                // 2
    constexpr uint32_t HD_PER_WARP = HEAD_DIM / NWARPS;        // 64
    constexpr uint32_t PV_N_TILES  = HD_PER_WARP / MMA_N;      // 8
    constexpr uint32_t Q_GROUPS_PER_KV = Q_PER_KV / NCOLS2;    // 3
    constexpr uint32_t Q_ROWS_PER_TILE = MMA_M / NCOLS2;       // 8
    constexpr uint32_t M_TILES         = BR / Q_ROWS_PER_TILE; // 2@BR=16, 4@BR=32
    constexpr uint32_t M_TOTAL         = BR * NCOLS2;          // 32@BR=16, 64@BR=32
    static_assert(M_TILES * MMA_M == M_TOTAL, "M_TILES*MMA_M==M_TOTAL");
    static_assert(M_TOTAL % NWARPS == 0,      "M_TOTAL divisible by NWARPS");

    constexpr uint32_t HEAD_STRIDE = HEAD_DIM + KV_PAD;

    const uint32_t kvg_idx  = blockIdx.x;
    const uint32_t kv_head  = kvg_idx / Q_GROUPS_PER_KV;
    const uint32_t q_group  = kvg_idx % Q_GROUPS_PER_KV;
    const uint32_t block_q  = blockIdx.y;
    const uint32_t k_split  = blockIdx.z;
    const uint32_t tid      = threadIdx.x;
    const uint32_t warp     = tid / WARP_SIZE;
    const uint32_t lane     = tid % WARP_SIZE;
    if (kv_head >= n_kv_heads) return;
    const uint32_t head_base = kv_head * Q_PER_KV + q_group * NCOLS2;

    const uint32_t group_id = lane / 4;
    const uint32_t in_group = lane % 4;

    constexpr uint32_t C_A = 0;
    constexpr uint32_t C_B = 1;
    const uint32_t r_a = group_id;
    const uint32_t r_b = group_id;

    uint32_t block_max_q = block_q * BR + BR;
    if (block_max_q > batch) block_max_q = batch;
    const uint32_t block_max_kv = base_seq_len + block_max_q;

    uint32_t kv_lo, kv_hi;
    if (NSPLIT == 1) {
        kv_lo = 0;
        kv_hi = block_max_kv;
    } else {
        const uint32_t kv_total = base_seq_len + batch;
        const uint32_t per_split = ((kv_total + NSPLIT - 1) / NSPLIT + BC - 1) & ~uint32_t(BC - 1);
        kv_lo = k_split * per_split;
        kv_hi = kv_lo + per_split;
        if (kv_hi > block_max_kv) kv_hi = block_max_kv;
        if (kv_lo >= kv_hi) {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
                const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
                const uint32_t head_a  = head_base + C_A;
                const uint32_t head_b  = head_base + C_B;
                if (warp == 0 && in_group == 0) {
                    if (q_idx_a < batch) {
                        const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = -INFINITY;
                        ms[mb * 2 + 1] = 0.0f;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = -INFINITY;
                        ms[mb * 2 + 1] = 0.0f;
                    }
                }
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0 = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (q_idx_a < batch) {
                        const uint64_t pbase = (((uint64_t)head_a * batch + q_idx_a)
                                                * NSPLIT + k_split) * HEAD_DIM;
                        partials[pbase + col + 0] = 0.0f;
                        partials[pbase + col + 1] = 0.0f;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t pbase = (((uint64_t)head_b * batch + q_idx_b)
                                                * NSPLIT + k_split) * HEAD_DIM;
                        partials[pbase + col + 0] = 0.0f;
                        partials[pbase + col + 1] = 0.0f;
                    }
                }
            }
            return;
        }
    }

    extern __shared__ char smem_raw[];
    char *p = smem_raw;
    __half *s_K     = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    __half *s_V_buf = reinterpret_cast<__half *>(p); p += BC * HEAD_STRIDE * sizeof(__half);
    float  *s_S     = reinterpret_cast<float  *>(p); p += M_TOTAL * S_STRIDE * sizeof(float);
    __half *s_P     = reinterpret_cast<__half *>(p); p += M_TOTAL * P_STRIDE * sizeof(__half);
    __half *s_O     = reinterpret_cast<__half *>(p); p += M_TOTAL * O_STRIDE * sizeof(__half);
    float  *s_m     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_l     = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);
    float  *s_alpha = reinterpret_cast<float  *>(p); p += M_TOTAL          * sizeof(float);

    // ---- Q in registers (same packing as v2) -----------------------------
    unsigned Q_reg[M_TILES][QK_KSTEPS][4];
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
            const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
            const bool a_active = (q_idx_a < batch);
            const bool b_active = (q_idx_b < batch);
            const float *qa = a_active
                ? (q + (uint64_t)q_idx_a * q_batch_stride + head_a * q_stride) : nullptr;
            const float *qb = b_active
                ? (q + (uint64_t)q_idx_b * q_batch_stride + head_b * q_stride) : nullptr;
            #pragma unroll
            for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
                const uint32_t k0 = ks * MMA_K;
                const uint32_t col0 = k0 + in_group * 2;
                const __half qa0 = qa ? __float2half(qa[col0 + 0]) : (__half)0;
                const __half qa1 = qa ? __float2half(qa[col0 + 1]) : (__half)0;
                const __half qb0 = qb ? __float2half(qb[col0 + 0]) : (__half)0;
                const __half qb1 = qb ? __float2half(qb[col0 + 1]) : (__half)0;
                const __half qa8 = qa ? __float2half(qa[col0 + 8]) : (__half)0;
                const __half qa9 = qa ? __float2half(qa[col0 + 9]) : (__half)0;
                const __half qb8 = qb ? __float2half(qb[col0 + 8]) : (__half)0;
                const __half qb9 = qb ? __float2half(qb[col0 + 9]) : (__half)0;
                Q_reg[mt][ks][0] = mma_detail::pack2h(qa0, qa1);
                Q_reg[mt][ks][1] = mma_detail::pack2h(qb0, qb1);
                Q_reg[mt][ks][2] = mma_detail::pack2h(qa8, qa9);
                Q_reg[mt][ks][3] = mma_detail::pack2h(qb8, qb9);
            }
        }
    }

    if (tid < M_TOTAL) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    // Initialize s_O = 0 so the first-tile rescale (alpha=0) reads zeros
    // instead of uninitialized halves (NaN safety: 0 * NaN = NaN).
    {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t TOTAL_HALVES  = M_TOTAL * O_STRIDE;
        constexpr uint32_t STEPS_PER_THR = (TOTAL_HALVES + 128 * HALVES_PER_LD - 1)
                                         / (128 * HALVES_PER_LD);
        const int4 zero4 = make_int4(0, 0, 0, 0);
        #pragma unroll
        for (uint32_t step = 0; step < STEPS_PER_THR; ++step) {
            const uint32_t off = (step * 128 + tid) * HALVES_PER_LD;
            if (off + HALVES_PER_LD <= TOTAL_HALVES) {
                *reinterpret_cast<int4 *>(&s_O[off]) = zero4;
            } else {
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    if (off + i < TOTAL_HALVES) s_O[off + i] = (__half)0;
                }
            }
        }
    }
    __syncthreads();

    auto issue_K = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            __half *dst = &s_K[shm_t * HEAD_STRIDE + base_d];
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_K) {
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, k_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                int4 *dst4 = reinterpret_cast<int4 *>(dst);
                if (in_range) {
                    const __half *k_src = k_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *dst4 = *reinterpret_cast<const int4 *>(k_src);
                } else {
                    int4 z; z.x = z.y = z.z = z.w = 0;
                    *dst4 = z;
                }
            }
        }
    };

    auto issue_V = [&](uint32_t t0_target) {
        constexpr uint32_t HALVES_PER_LD = 8;
        constexpr uint32_t LANES_PER_ROW = HEAD_DIM / HALVES_PER_LD;
        constexpr uint32_t TOKENS_PER_PASS = 128 / LANES_PER_ROW;
        constexpr uint32_t N_PASS = BC / TOKENS_PER_PASS;
        const uint32_t row_in_pass = tid / LANES_PER_ROW;
        const uint32_t col_lane    = tid % LANES_PER_ROW;
        #pragma unroll
        for (uint32_t pass = 0; pass < N_PASS; ++pass) {
            const uint32_t shm_t = pass * TOKENS_PER_PASS + row_in_pass;
            const uint32_t t     = t0_target + shm_t;
            const uint32_t base_d = col_lane * HALVES_PER_LD;
            const bool in_range = (t < block_max_kv);
            if (USE_CP_ASYNC_V) {
                __half *dst = &s_V_buf[shm_t * HEAD_STRIDE + base_d];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    mma_detail::cp_async_cg_16(dst, v_src);
                } else {
                    mma_detail::cp_async_zero_16(dst);
                }
            } else {
                __half v8[HALVES_PER_LD];
                if (in_range) {
                    const __half *v_src = v_cache
                        + (uint64_t)t * n_kv_heads * HEAD_DIM
                        + kv_head * HEAD_DIM + base_d;
                    *reinterpret_cast<int4 *>(v8) = *reinterpret_cast<const int4 *>(v_src);
                } else {
                    *reinterpret_cast<int4 *>(v8) = make_int4(0,0,0,0);
                }
                #pragma unroll
                for (uint32_t i = 0; i < HALVES_PER_LD; ++i) {
                    s_V_buf[(base_d + i) * BC + shm_t] = v8[i];
                }
            }
        }
    };

    if (USE_CP_ASYNC_K) { issue_K(kv_lo); mma_detail::cp_async_commit(); }
    if (USE_CP_ASYNC_V) { issue_V(kv_lo); mma_detail::cp_async_commit(); }

    // ---- Tile loop -------------------------------------------------------
    for (uint32_t t0 = kv_lo; t0 < kv_hi; t0 += BC) {

        if (USE_CP_ASYNC_K && USE_CP_ASYNC_V) {
            mma_detail::cp_async_wait_group<1>();
        } else if (USE_CP_ASYNC_K) {
            mma_detail::cp_async_wait_group<0>();
        }
        if (!USE_CP_ASYNC_K) issue_K(t0);
        if (!USE_CP_ASYNC_V) issue_V(t0);
        __syncthreads();

        // QK^T phase. Same as v2.
        float c_h[M_TILES][QK_N_TILES][4];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                c_h[mt][nq][0]=0.f; c_h[mt][nq][1]=0.f;
                c_h[mt][nq][2]=0.f; c_h[mt][nq][3]=0.f;
            }
        }
        #pragma unroll
        for (uint32_t ks = 0; ks < QK_KSTEPS; ++ks) {
            const uint32_t k0 = ks * MMA_K;
            #pragma unroll
            for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                const uint32_t kv_col = warp * (BC / NWARPS) + nq * MMA_N + group_id;
                const __half *krow0 = &s_K[kv_col * HEAD_STRIDE + k0 + in_group * 2];
                const unsigned b0 = mma_detail::pack2h(krow0);
                const unsigned b1 = mma_detail::pack2h(krow0 + 8);
                #pragma unroll
                for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                    mma_detail::mma_m16n8k16_f16f16f32(
                        c_h[mt][nq][0], c_h[mt][nq][1], c_h[mt][nq][2], c_h[mt][nq][3],
                        Q_reg[mt][ks][0], Q_reg[mt][ks][1], Q_reg[mt][ks][2], Q_reg[mt][ks][3],
                        b0, b1);
                }
            }
        }

        if (USE_CP_ASYNC_K) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_K(t0_next); mma_detail::cp_async_commit(); }
        }

        // Write S to shmem with causal mask. Same lane mapping as v2.
        {
            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
                const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
                const bool a_active = (q_idx_a < batch);
                const bool b_active = (q_idx_b < batch);
                const uint32_t my_max_kv_a = base_seq_len + q_idx_a + 1;
                const uint32_t my_max_kv_b = base_seq_len + q_idx_b + 1;
                const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
                const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;
                #pragma unroll
                for (uint32_t nq = 0; nq < QK_N_TILES; ++nq) {
                    const uint32_t base_col = warp * (BC / NWARPS) + nq * MMA_N + in_group * 2;
                    const uint32_t t_a0 = t0 + base_col;
                    const uint32_t t_a1 = t0 + base_col + 1;
                    float v00 = (a_active && t_a0 < my_max_kv_a) ? c_h[mt][nq][0] * scale : -INFINITY;
                    float v01 = (a_active && t_a1 < my_max_kv_a) ? c_h[mt][nq][1] * scale : -INFINITY;
                    float v02 = (b_active && t_a0 < my_max_kv_b) ? c_h[mt][nq][2] * scale : -INFINITY;
                    float v03 = (b_active && t_a1 < my_max_kv_b) ? c_h[mt][nq][3] * scale : -INFINITY;
                    s_S[m_a * S_STRIDE + base_col + 0] = v00;
                    s_S[m_a * S_STRIDE + base_col + 1] = v01;
                    s_S[m_b * S_STRIDE + base_col + 0] = v02;
                    s_S[m_b * S_STRIDE + base_col + 1] = v03;
                }
            }
        }

        __syncthreads();

        // Online softmax. Same as v2.
        constexpr uint32_t ROWS_PER_WARP = M_TOTAL / NWARPS;
        constexpr uint32_t COLS_PER_LANE = BC / WARP_SIZE;
        #pragma unroll
        for (uint32_t r = 0; r < ROWS_PER_WARP; ++r) {
            const uint32_t m_row = warp * ROWS_PER_WARP + r;
            float local_max = -INFINITY;
            float s_vals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_vals[cc] = s_S[m_row * S_STRIDE + lane * COLS_PER_LANE + cc];
                if (s_vals[cc] > local_max) local_max = s_vals[cc];
            }
            const float row_max = cuda_helpers::warp_reduce_max<32>(local_max);
            const float prev_m = s_m[m_row];
            const float new_m  = fmaxf(prev_m, row_max);
            float local_sum = 0.0f;
            __half pvals[COLS_PER_LANE];
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                const float p_val = (s_vals[cc] == -INFINITY) ? 0.0f
                                                              : __expf(s_vals[cc] - new_m);
                local_sum += p_val;
                pvals[cc] = __float2half(p_val);
            }
            const float row_sum = cuda_helpers::warp_reduce_sum<32>(local_sum);
            const float prev_l  = s_l[m_row];
            const float alpha   = (prev_m == -INFINITY) ? 0.0f : __expf(prev_m - new_m);
            const float new_l   = prev_l * alpha + row_sum;
            if (lane == 0) {
                s_m[m_row]     = new_m;
                s_l[m_row]     = new_l;
                s_alpha[m_row] = alpha;
            }
            #pragma unroll
            for (uint32_t cc = 0; cc < COLS_PER_LANE; ++cc) {
                s_P[m_row * P_STRIDE + lane * COLS_PER_LANE + cc] = pvals[cc];
            }
        }

        if (USE_CP_ASYNC_V) {
            const bool k_next_pending = USE_CP_ASYNC_K && (t0 + BC < kv_hi);
            if (k_next_pending) mma_detail::cp_async_wait_group<1>();
            else                mma_detail::cp_async_wait_group<0>();
        }
        __syncthreads();

        // Per-row alpha for rescale fold (same packing as v2).
        float alpha_a[M_TILES], alpha_b[M_TILES];
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            alpha_a[mt] = s_alpha[mt * MMA_M + group_id];
            alpha_b[mt] = s_alpha[mt * MMA_M + group_id + 8];
        }

        // ---- PV with O in shmem ------------------------------------------
        // Loop nest: for each n, for each mt, hold a 4-fp32 accumulator strip
        // in regs across PV_KSTEPS MMAs. At strip start, read fp16 O_old from
        // s_O and multiply by alpha (folds rescale into PV). At strip end,
        // demote to fp16 and write back to s_O.
        //
        // V is loaded M_TILES times more than v2 (mt is now inside n, ks).
        // PV_KSTEPS=2 here so V issue count = PV_N_TILES * PV_KSTEPS * M_TILES
        // = 8*2*4 = 64 ldmatrix.x2 per warp per tile (vs v2: 16). Cheap; the
        // win is the register-pressure relief (O_acc 128 → 16 regs/lane).
        #pragma unroll
        for (uint32_t n = 0; n < PV_N_TILES; ++n) {
            const uint32_t n0  = warp * HD_PER_WARP + n * MMA_N;
            const uint32_t col = n0 + in_group * 2;

            #pragma unroll
            for (uint32_t mt = 0; mt < M_TILES; ++mt) {
                const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
                const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;

                // Read O_old (fp16) + rescale by alpha → 4 fp32 regs.
                const __half h0 = s_O[m_a * O_STRIDE + col + 0];
                const __half h1 = s_O[m_a * O_STRIDE + col + 1];
                const __half h2 = s_O[m_b * O_STRIDE + col + 0];
                const __half h3 = s_O[m_b * O_STRIDE + col + 1];
                float o0 = __half2float(h0) * alpha_a[mt];
                float o1 = __half2float(h1) * alpha_a[mt];
                float o2 = __half2float(h2) * alpha_b[mt];
                float o3 = __half2float(h3) * alpha_b[mt];

                #pragma unroll
                for (uint32_t ks = 0; ks < PV_KSTEPS; ++ks) {
                    const uint32_t k0 = ks * MMA_K;
                    unsigned b0, b1;
                    if (USE_CP_ASYNC_V) {
                        const __half *vbase = &s_V_buf[k0 * HEAD_STRIDE + n0];
                        mma_detail::ldmatrix_x2_b_trans(b0, b1, vbase, HEAD_STRIDE);
                    } else {
                        const uint32_t out_col = n0 + group_id;
                        const __half *vrow0 = &s_V_buf[out_col * BC + k0 + in_group * 2];
                        b0 = mma_detail::pack2h(vrow0);
                        b1 = mma_detail::pack2h(vrow0 + 8);
                    }
                    unsigned a0, a1, a2, a3;
                    mma_detail::ldmatrix_x4_a(a0, a1, a2, a3,
                                              &s_P[mt * MMA_M * P_STRIDE + k0], P_STRIDE);
                    mma_detail::mma_m16n8k16_f16f16f32(
                        o0, o1, o2, o3, a0, a1, a2, a3, b0, b1);
                }

                // Demote to fp16 and write back to s_O.
                s_O[m_a * O_STRIDE + col + 0] = __float2half(o0);
                s_O[m_a * O_STRIDE + col + 1] = __float2half(o1);
                s_O[m_b * O_STRIDE + col + 0] = __float2half(o2);
                s_O[m_b * O_STRIDE + col + 1] = __float2half(o3);
            }
        }

        if (USE_CP_ASYNC_V) {
            const uint32_t t0_next = t0 + BC;
            if (t0_next < kv_hi) { issue_V(t0_next); mma_detail::cp_async_commit(); }
        }
    }

    // ---- Final write -----------------------------------------------------
    // Each warp owns its own [n0, n0+HD_PER_WARP) cols of s_O, and each lane
    // wrote/reads exactly its own (m_a, m_b, col) cells, so no inter-thread
    // sync is needed before reading s_O for global write.
    {
        const uint32_t head_a = head_base + C_A;
        const uint32_t head_b = head_base + C_B;
        #pragma unroll
        for (uint32_t mt = 0; mt < M_TILES; ++mt) {
            const uint32_t q_idx_a = block_q * BR + mt * Q_ROWS_PER_TILE + r_a;
            const uint32_t q_idx_b = block_q * BR + mt * Q_ROWS_PER_TILE + r_b;
            const uint32_t m_a = mt * MMA_M + C_A * Q_ROWS_PER_TILE + r_a;
            const uint32_t m_b = mt * MMA_M + C_B * Q_ROWS_PER_TILE + r_b;
            const float l_a = s_l[m_a];
            const float l_b = s_l[m_b];
            if (NSPLIT == 1) {
                const float inv_la = (l_a > 0.0f) ? (1.0f / l_a) : 0.0f;
                const float inv_lb = (l_b > 0.0f) ? (1.0f / l_b) : 0.0f;
                float * const out_a = (q_idx_a < batch)
                    ? (out + (uint64_t)q_idx_a * out_batch_stride + head_a * HEAD_DIM) : nullptr;
                float * const out_b = (q_idx_b < batch)
                    ? (out + (uint64_t)q_idx_b * out_batch_stride + head_b * HEAD_DIM) : nullptr;
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0  = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (out_a) {
                        out_a[col + 0] = __half2float(s_O[m_a * O_STRIDE + col + 0]) * inv_la;
                        out_a[col + 1] = __half2float(s_O[m_a * O_STRIDE + col + 1]) * inv_la;
                    }
                    if (out_b) {
                        out_b[col + 0] = __half2float(s_O[m_b * O_STRIDE + col + 0]) * inv_lb;
                        out_b[col + 1] = __half2float(s_O[m_b * O_STRIDE + col + 1]) * inv_lb;
                    }
                }
            } else {
                float * const part_a = (q_idx_a < batch)
                    ? (partials + (((uint64_t)head_a * batch + q_idx_a)
                                   * NSPLIT + k_split) * HEAD_DIM) : nullptr;
                float * const part_b = (q_idx_b < batch)
                    ? (partials + (((uint64_t)head_b * batch + q_idx_b)
                                   * NSPLIT + k_split) * HEAD_DIM) : nullptr;
                #pragma unroll
                for (uint32_t n = 0; n < PV_N_TILES; ++n) {
                    const uint32_t n0  = warp * HD_PER_WARP + n * MMA_N;
                    const uint32_t col = n0 + in_group * 2;
                    if (part_a) {
                        part_a[col + 0] = __half2float(s_O[m_a * O_STRIDE + col + 0]);
                        part_a[col + 1] = __half2float(s_O[m_a * O_STRIDE + col + 1]);
                    }
                    if (part_b) {
                        part_b[col + 0] = __half2float(s_O[m_b * O_STRIDE + col + 0]);
                        part_b[col + 1] = __half2float(s_O[m_b * O_STRIDE + col + 1]);
                    }
                }
                if (warp == 0 && in_group == 0) {
                    const float m_a_v = s_m[m_a];
                    const float m_b_v = s_m[m_b];
                    if (q_idx_a < batch) {
                        const uint64_t mb = ((uint64_t)head_a * batch + q_idx_a) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = m_a_v;
                        ms[mb * 2 + 1] = l_a;
                    }
                    if (q_idx_b < batch) {
                        const uint64_t mb = ((uint64_t)head_b * batch + q_idx_b) * NSPLIT + k_split;
                        ms[mb * 2 + 0] = m_b_v;
                        ms[mb * 2 + 1] = l_b;
                    }
                }
            }
        }
    }
}

static size_t fattn_mma_gqa_v5_smem_bytes(uint32_t head_dim, uint32_t bc, uint32_t br) {
    constexpr uint32_t NCOLS2 = 2;
    constexpr uint32_t KV_PAD = 8;
    constexpr uint32_t S_PAD  = 4;
    constexpr uint32_t O_PAD  = 8;
    const uint32_t M_TOTAL    = br * NCOLS2;
    const uint32_t hd_stride  = head_dim + KV_PAD;
    const uint32_t s_stride   = bc + S_PAD;
    const uint32_t p_stride   = bc + S_PAD * 2;
    const uint32_t o_stride   = head_dim + O_PAD;
    size_t s = 0;
    s += bc      * hd_stride * sizeof(__half);   // K
    s += bc      * hd_stride * sizeof(__half);   // V
    s += M_TOTAL * s_stride  * sizeof(float);    // S
    s += M_TOTAL * p_stride  * sizeof(__half);   // P
    s += M_TOTAL * o_stride  * sizeof(__half);   // O (fp16, in shmem)
    s += M_TOTAL             * sizeof(float) * 3;// m, l, alpha
    return s;
}

// v3 launcher: BR=64 + NCOLS2=1 + Q-in-shmem. The q-rows-per-CTA lever
// applied: 64 q-rows/CTA, 4 MMAs per K/V load. Long-T attack.
// Currently gated to Qwen 3.6 27B's q_per_kv=6.
bool launch_fattn_prefill_mma_gqa_v3_f16(
        float *       out,
        const float * q,
        uint32_t      q_stride,
        const void  * k_cache,
        const void  * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      batch,
        uint32_t      base_seq_len,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (!(q_per_kv == 6)) return false;

    static const uint32_t bc_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BC");
        if (!e) return 32;
        if (std::strcmp(e, "32") == 0) return 32;
        if (std::strcmp(e, "64") == 0) return 64;
        return 32;
    }();
    static const uint32_t br_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_BR");
        if (!e) return 64;
        if (std::strcmp(e, "32") == 0) return 32;
        if (std::strcmp(e, "64") == 0) return 64;
        return 64;
    }();

    const uint32_t n_blocks_q = (batch + br_choice - 1) / br_choice;
    // Grid: each q-head is its own CTA (NCOLS2=1).
    const dim3 grid(n_kv_heads * q_per_kv, n_blocks_q, 1);
    const dim3 block(128);
    const size_t smem = fattn_mma_gqa_v3_smem_bytes(head_dim, bc_choice, br_choice);

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();

    auto launch = [&](auto HD_v, auto BR_v, auto BC_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t BR = decltype(BR_v)::value;
        constexpr uint32_t BC = decltype(BC_v)::value;
        auto dispatch = [&](auto K_v, auto V_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v3_t<HD, 6, BR, BC, K, V>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v3_t<HD, 6, BR, BC, K, V>
                <<<grid, block, smem, stream>>>(
                    out, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch(std::false_type{}, std::true_type{});
        } else {
            dispatch(std::false_type{}, std::false_type{});
        }
    };

    auto launch_with_bc = [&](auto HD_v, auto BR_v) {
        if (bc_choice == 32) launch(HD_v, BR_v, std::integral_constant<uint32_t, 32>{});
        else                 launch(HD_v, BR_v, std::integral_constant<uint32_t, 64>{});
    };
    auto launch_with_br = [&](auto HD_v) {
        if (br_choice == 64) launch_with_bc(HD_v, std::integral_constant<uint32_t, 64>{});
        else                 launch_with_bc(HD_v, std::integral_constant<uint32_t, 32>{});
    };
    if (head_dim == 256) launch_with_br(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_br(std::integral_constant<uint32_t, 128>{});
    return true;
}

// =====================================================================
// v4 launcher: NWARPS=8 + warp-pair-owns-mtile, M_TOTAL=64.
// =====================================================================
// Currently gated to Qwen 3.6 27B's q_per_kv=6 (kernel hard-codes Q_PER_KV=6
// with q-head zero-pad to 8).
size_t fattn_prefill_mma_gqa_v4_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                              uint32_t n_kv_heads,
                                              uint32_t head_dim) {
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return 0;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return 0;
    if (!(head_dim == 128 || head_dim == 256)) return 0;
    // v4 reuses the same NSPLIT scratch shape as v2: per (head, q_tok, split)
    // we stash HD floats of un-normalized O plus 2 floats of (m, l).
    constexpr uint32_t MAX_NSPLIT = 2;
    return (size_t)n_heads * batch * MAX_NSPLIT * head_dim * sizeof(float)
         + (size_t)n_heads * batch * MAX_NSPLIT * 2        * sizeof(float);
}

bool launch_fattn_prefill_mma_gqa_v4_f16(
        float *       out,
        void  *       scratch,
        size_t        scratch_bytes,
        const float * q,
        uint32_t      q_stride,
        const void  * k_cache,
        const void  * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      batch,
        uint32_t      base_seq_len,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return false;

    constexpr uint32_t BR = 8;

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_fp16_o = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_FP16_O");
        if (!e) return false;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();

    const uint32_t n_blocks_q = (batch + BR - 1) / BR;
    // NSPLIT heuristic: same shape as v2's. Kept conservative for first
    // landing — default NSPLIT=1 since the v4 grid is already smaller
    // (n_kv_heads × n_blocks_q vs v2's n_kv_heads*3 × n_blocks_q*2).
    static const uint32_t v4_nsplit_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_NSPLIT");
        if (!e) return 1;
        if (std::strcmp(e, "1") == 0) return 1;
        if (std::strcmp(e, "2") == 0) return 2;
        return 1;
    }();
    const uint32_t nsplit = v4_nsplit_choice;
    float *partials = nullptr, *ms_buf = nullptr;
    if (nsplit > 1) {
        const size_t need = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float)
                          + (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
        if (!scratch || scratch_bytes < need) return false;
        partials = static_cast<float *>(scratch);
        ms_buf   = partials + (size_t)n_heads * batch * nsplit * head_dim;
    }
    const dim3 grid(n_kv_heads, n_blocks_q, nsplit);
    const dim3 block(256);
    const size_t smem = fattn_mma_gqa_v4_smem_bytes(head_dim);

    auto launch = [&](auto HD_v, auto NS_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t NS = decltype(NS_v)::value;
        auto dispatch = [&](auto K_v, auto V_v, auto FP16O_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            constexpr bool FP16O = decltype(FP16O_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v4_t<HD, K, V, NS, FP16O>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v4_t<HD, K, V, NS, FP16O>
                <<<grid, block, smem, stream>>>(
                    out, partials, ms_buf, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        auto dispatch_fp16o = [&](auto K_v, auto V_v) {
            if (use_fp16_o) dispatch(K_v, V_v, std::true_type{});
            else            dispatch(K_v, V_v, std::false_type{});
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch_fp16o(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch_fp16o(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch_fp16o(std::false_type{}, std::true_type{});
        } else {
            dispatch_fp16o(std::false_type{}, std::false_type{});
        }
        if (NS > 1) {
            const dim3 cgrid(n_heads, batch, 1);
            const dim3 cblock(HD);
            fattn_prefill_mma_gqa_combine_kernel<HD, NS>
                <<<cgrid, cblock, 0, stream>>>(
                    out, partials, ms_buf, batch, out_batch_stride);
        }
    };

    auto launch_with_ns = [&](auto HD_v) {
        if (nsplit == 2) launch(HD_v, std::integral_constant<uint32_t, 2>{});
        else             launch(HD_v, std::integral_constant<uint32_t, 1>{});
    };
    if (head_dim == 256) launch_with_ns(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_ns(std::integral_constant<uint32_t, 128>{});
    return true;
}

size_t fattn_prefill_mma_gqa_v5_scratch_bytes(uint32_t batch, uint32_t n_heads,
                                              uint32_t n_kv_heads,
                                              uint32_t head_dim) {
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return 0;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return 0;
    if (!(head_dim == 128 || head_dim == 256)) return 0;
    constexpr uint32_t MAX_NSPLIT = 2;
    return (size_t)n_heads * batch * MAX_NSPLIT * head_dim * sizeof(float)
         + (size_t)n_heads * batch * MAX_NSPLIT * 2        * sizeof(float);
}

bool launch_fattn_prefill_mma_gqa_v5_f16(
        float *       out,
        void  *       scratch,
        size_t        scratch_bytes,
        const float * q,
        uint32_t      q_stride,
        const void  * k_cache,
        const void  * v_cache,
        uint32_t      n_heads,
        uint32_t      n_kv_heads,
        uint32_t      head_dim,
        uint32_t      batch,
        uint32_t      base_seq_len,
        uint32_t      q_batch_stride,
        uint32_t      out_batch_stride,
        float         scale,
        cudaStream_t  stream) {
    if (!(head_dim == 128 || head_dim == 256)) return false;
    if (batch == 0) return true;
    if (n_kv_heads == 0 || n_heads % n_kv_heads != 0) return false;
    const uint32_t q_per_kv = n_heads / n_kv_heads;
    if (q_per_kv != 6) return false;

    constexpr uint32_t NCOLS2  = 2;
    constexpr uint32_t BC      = 32;
    constexpr uint32_t KV_PAD  = 8;
    constexpr uint32_t S_PAD   = 4;
    constexpr uint32_t O_PAD   = 8;

    static const bool use_cp_async_k = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_KCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    static const bool use_cp_async_v = []() {
        const char *e = std::getenv("QW3_PREFILL_FA2_VCPASYNC");
        if (!e) return true;
        return !(std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
    }();
    // BR knob: default 32 (the lever). BR=16 is supported as an A/B knob to
    // isolate the O→shmem cost from the q-rows-per-CTA win.
    static const uint32_t br_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_V5_BR");
        if (!e) return 32;
        if (std::strcmp(e, "16") == 0) return 16;
        if (std::strcmp(e, "32") == 0) return 32;
        return 32;
    }();

    const uint32_t n_blocks_q = (batch + br_choice - 1) / br_choice;

    static const uint32_t v5_nsplit_choice = []() -> uint32_t {
        const char *e = std::getenv("QW3_PREFILL_FA2_NSPLIT");
        if (!e) return 1;
        if (std::strcmp(e, "1") == 0) return 1;
        if (std::strcmp(e, "2") == 0) return 2;
        return 1;
    }();
    const uint32_t nsplit = v5_nsplit_choice;
    float *partials = nullptr, *ms_buf = nullptr;
    if (nsplit > 1) {
        const size_t need = (size_t)n_heads * batch * nsplit * head_dim * sizeof(float)
                          + (size_t)n_heads * batch * nsplit * 2        * sizeof(float);
        if (!scratch || scratch_bytes < need) return false;
        partials = static_cast<float *>(scratch);
        ms_buf   = partials + (size_t)n_heads * batch * nsplit * head_dim;
    }
    const dim3 grid(n_kv_heads * (q_per_kv / NCOLS2), n_blocks_q, nsplit);
    const dim3 block(128);
    const size_t smem = fattn_mma_gqa_v5_smem_bytes(head_dim, BC, br_choice);

    auto launch = [&](auto HD_v, auto BR_v, auto NS_v) {
        constexpr uint32_t HD = decltype(HD_v)::value;
        constexpr uint32_t BR = decltype(BR_v)::value;
        constexpr uint32_t NS = decltype(NS_v)::value;
        auto dispatch = [&](auto K_v, auto V_v) {
            constexpr bool K = decltype(K_v)::value;
            constexpr bool V = decltype(V_v)::value;
            cudaFuncSetAttribute(
                fattn_prefill_mma_gqa_kernel_v5_t<HD, 6, BR, BC, NCOLS2, KV_PAD, S_PAD, O_PAD, K, V, NS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
            fattn_prefill_mma_gqa_kernel_v5_t<HD, 6, BR, BC, NCOLS2, KV_PAD, S_PAD, O_PAD, K, V, NS>
                <<<grid, block, smem, stream>>>(
                    out, partials, ms_buf, q, q_stride,
                    static_cast<const __half *>(k_cache),
                    static_cast<const __half *>(v_cache),
                    n_heads, n_kv_heads, base_seq_len, batch,
                    q_batch_stride, out_batch_stride, scale);
        };
        if (use_cp_async_k && use_cp_async_v) {
            dispatch(std::true_type{},  std::true_type{});
        } else if (use_cp_async_k) {
            dispatch(std::true_type{},  std::false_type{});
        } else if (use_cp_async_v) {
            dispatch(std::false_type{}, std::true_type{});
        } else {
            dispatch(std::false_type{}, std::false_type{});
        }
        if (NS > 1) {
            const dim3 cgrid(n_heads, batch, 1);
            const dim3 cblock(HD);
            fattn_prefill_mma_gqa_combine_kernel<HD, NS>
                <<<cgrid, cblock, 0, stream>>>(
                    out, partials, ms_buf, batch, out_batch_stride);
        }
    };

    auto launch_with_ns_br = [&](auto HD_v, auto BR_v) {
        if (nsplit == 2) launch(HD_v, BR_v, std::integral_constant<uint32_t, 2>{});
        else             launch(HD_v, BR_v, std::integral_constant<uint32_t, 1>{});
    };
    auto launch_with_br = [&](auto HD_v) {
        if (br_choice == 16) launch_with_ns_br(HD_v, std::integral_constant<uint32_t, 16>{});
        else                 launch_with_ns_br(HD_v, std::integral_constant<uint32_t, 32>{});
    };
    if (head_dim == 256) launch_with_br(std::integral_constant<uint32_t, 256>{});
    else                 launch_with_br(std::integral_constant<uint32_t, 128>{});
    return true;
}

} // namespace ported
} // namespace qw3
