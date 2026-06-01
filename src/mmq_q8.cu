// SPDX-License-Identifier: MIT
//
// Q8_0 weight x Q8_1 activation matmul via INT8 MMA tensor cores.
//
// v2: 4-warp cooperative CTA, 64x128 output tile (rows x batch) per CTA,
// weight + activation staged in shared memory once per K block. Approach
// mirrors llama.cpp's mmq.cuh shape (mmq_x=128, mmq_y=128, 8 warps,
// stream-K) but stripped to the essentials: vanilla 2D tile decomposition,
// single-K-block staging, direct lane-indexed shmem reads instead of
// ldmatrix. The single-warp v1 (16x32 per CTA, 1 MMA per K block) ran ~3x
// slower than cuBLAS HGEMM because each K-block's weight read goes
// straight to gmem with no reuse; v2 amortizes that read across the 4
// warps that share the row tile.
//
// Per CTA:
//   threads: 4 warps x 32 lanes = 128
//   tile:    64 rows x 128 batch cols x 32 K (per K block)
//   shmem:   ~7 KB (weight + activation int8 + per-tile FP32 scales, single buffer)
//
// Each warp owns a 16-row stripe (rows [16w, 16w+16)). Within that stripe
// it computes 16 m16n8k32 outputs spanning the 128 batch cols. Per K block
// each warp issues 1 A load (4 .b32) + 16 (B-load + MMA + scale) sequences.
//
// Status: parity-correct (greedy decode matches HGEMM byte-for-byte through
// 32 tokens on Qwen 3.6 27B). On RTX Pro 6000 Blackwell:
//   - 2K prefill: 2125 tok/s vs HGEMM 2870 (74% — still trailing).
//   - 285-token prefill: 1278 tok/s vs HGEMM 1190 (+7%; MMQ wins where
//     HGEMM's per-call dequant overhead bites hardest).
//
// The remaining gap to HGEMM (and to llama.cpp's mmq.cuh) is fragment-
// load efficiency: llama.cpp uses `ldmatrix.x4` PTX to fill the
// m16n8k32 A fragment from shmem in a single instruction per warp; we
// still do per-lane lane-indexed shmem reads. v3 should add ldmatrix
// for both A and B fragments + cp.async for shmem fills (overlap gmem
// reads with the next K block's MMA).
//
// Boundary handling: a `need_check` template flag enables row/batch range
// checks for the trailing tile when rows or batch are not multiples of
// {64, 128}. The non-check specialization is used when shapes align (the
// common case for Qwen 3.6 27B prefill, where rows in {3072, 4096, 5120,
// 6144, 14336, 17408, 27648, 28672} are all multiples of 64 and batch is
// the prompt length, padded only at the end).
//
// v1 kept around for shapes smaller than the v2 tile (only the lm_head
// embedding call hits that, and even that has rows >= 151936, so v2 still
// applies — v1 fallback is dormant in practice).
//
// Activation Q8_1 staging is produced by `quantize_q8_1` in mmvq_q8.cu and
// shared via the executor's q8_1_scratch buffer. Block layout is
// (batch, n_blocks_per_row) where each block is (half d, half s, int8 qs[32]).
//
// Compilation requirement: -arch=sm_80 or higher. m16n8k32.s8.s8.s32 is
// Ampere-and-up only.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

#include "cuda_helpers.cuh"

namespace qw3 {
namespace ported {

namespace {

static constexpr int MMQ_QK = 32;

// v2 multi-warp tile geometry.
static constexpr int V2_NWARPS    = 4;
static constexpr int V2_WARP      = 32;
static constexpr int V2_M_PER_CTA = 64;             // rows per CTA
static constexpr int V2_N_PER_CTA = 128;            // batch cols per CTA
static constexpr int V2_NW        = V2_N_PER_CTA / 8;   // N-tiles per warp = 16
static constexpr int V2_K_PAD     = 4;              // bank-conflict mitigation
static constexpr int V2_K_STRIDE  = MMQ_QK + V2_K_PAD;  // 36 bytes per row
static constexpr int V2_THREADS   = V2_NWARPS * V2_WARP;  // 128

// v7 tile geometry: 64×64 (vs v2's 64×128) so two stages of shmem fit under
// the 100 KB/SM cap for 2 blocks/SM. See v7 kernel doc for the breakdown.
static constexpr int V7M_M_PER_CTA = 64;
static constexpr int V7M_N_PER_CTA = 64;
static constexpr int V7M_NW        = V7M_N_PER_CTA / 8;  // 8

// v1 single-warp tile geometry (kept for shapes < 64 rows or < 64 batch).
static constexpr int V1_M_TILE     = 16;
static constexpr int V1_N_TILE     = 8;
static constexpr int V1_NW         = 4;
static constexpr int V1_N_PER_CTA  = V1_N_TILE * V1_NW;  // 32

// Q8_1 block layout (mirror of mmvq_q8.cu — kept anonymous to avoid ODR collision).
struct alignas(4) block_q8_1 {
    half2  ds;
    int8_t qs[MMQ_QK];
};
static_assert(sizeof(block_q8_1) == 36, "unexpected block_q8_1 size");

// PTX wrapper: m16n8k32.row.col.s32.s8.s8.s32. Non-volatile so ptxas can
// schedule MMAs and reorder around loads for ILP (matches llama.cpp).
__device__ __forceinline__ void mma_m16n8k32_s8s8s32(
        int &c0, int &c1, int &c2, int &c3,
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// ldmatrix.x4: load 16x8 .b16 tile from shmem into 4 .b32 regs per lane.
// For our int8 use this is 16 rows x 8 ints (= 32 int8 K-elements), the A
// operand of m16n8k32. ptr_int is in units of int (4 bytes); stride_int is
// the row stride in ints. Lane addressing matches the PTX spec:
//   lane 0..15 -> row=lane,    col=0
//   lane 16..31 -> row=lane-16, col=4
__device__ __forceinline__ void ldmatrix_x4(
        unsigned &a0, unsigned &a1, unsigned &a2, unsigned &a3,
        const int *ptr_int, int stride_int) {
    const int lane = threadIdx.x;
    const int *xs = ptr_int + (lane & 15) * stride_int + (lane >> 4) * 4;
    unsigned smem = __cvta_generic_to_shared(xs);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem));
}

// ldmatrix.x2: load 8x8 .b16 tile from shmem into 2 .b32 regs per lane.
// For our int8 use this is 8 rows x 8 ints (= 32 int8 K-elements x 8 cols),
// the B operand of m16n8k32 (col-major from the K dimension's perspective).
// Lane addressing per PTX:
//   lanes 0..7  -> row=lane,    col=0
//   lanes 8..15 -> row=lane-8,  col=4
//   lanes 16..23 -> row=lane-16, col=0   (duplicates 0..7)
//   lanes 24..31 -> row=lane-24, col=4   (duplicates 8..15)
__device__ __forceinline__ void ldmatrix_x2(
        unsigned &b0, unsigned &b1,
        const int *ptr_int, int stride_int) {
    const int lane = threadIdx.x;
    const int *xs = ptr_int + (lane & 7) * stride_int + ((lane >> 3) & 1) * 4;
    unsigned smem = __cvta_generic_to_shared(xs);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
        : "=r"(b0), "=r"(b1)
        : "r"(smem));
}

// Load one int (4 int8s) from a 2-byte-aligned byte stream. Q8_0 weight
// blocks place qs at byte 2 (after the FP16 scale), so qs ints are only
// 2-byte aligned to the start of the block.
__device__ __forceinline__ unsigned load_int_align2(const uint8_t *p, int byte_off) {
    const uint16_t *p16 = reinterpret_cast<const uint16_t *>(p + byte_off);
    return (static_cast<unsigned>(p16[0]) <<  0) |
           (static_cast<unsigned>(p16[1]) << 16);
}

// cp.async.cg.shared.global of 16 bytes from gmem -> smem. Used by v7 to
// pipeline the next K-block's W+Y read against the current K-block's MMA.
// Plane Q is 16-byte aligned (n_blocks % 16 == 0 for every Qwen3.6 col),
// so the int8 quants stream is now legal to cp.async with 16-B chunks.
__device__ __forceinline__ void cp_async_cg_16(void *smem_dst, const void *gmem_src) {
#if __CUDA_ARCH__ >= 800
    unsigned dst = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(gmem_src));
#else
    *reinterpret_cast<int4 *>(smem_dst) = *reinterpret_cast<const int4 *>(gmem_src);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

template <bool need_check>
__launch_bounds__(V2_THREADS, 2)
__global__ void mmq_q8_0_v2_kernel(
        const uint8_t   * __restrict__ wq,
        const block_q8_1 * __restrict__ ya,
        float           * __restrict__ dst,
        uint32_t          rows,
        uint32_t          cols,
        uint32_t          batch,
        uint32_t          stride_y_row,
        uint32_t          stride_dst_row)
{
    // Shared-memory tile (single-buffered, ~5 KB).
    __shared__ int8_t W_qs[V2_M_PER_CTA][V2_K_STRIDE];
    __shared__ float  W_d [V2_M_PER_CTA];
    __shared__ int8_t Y_qs[V2_N_PER_CTA][V2_K_STRIDE];
    __shared__ float  Y_d [V2_N_PER_CTA];

    const int n_blocks = static_cast<int>(cols / MMQ_QK);
    const int row0  = static_cast<int>(blockIdx.x) * V2_M_PER_CTA;
    const int col00 = static_cast<int>(blockIdx.y) * V2_N_PER_CTA;

    const int warp_id = static_cast<int>(threadIdx.y);    // 0..3
    const int lane    = static_cast<int>(threadIdx.x);    // 0..31
    const int tid     = warp_id * V2_WARP + lane;          // 0..127
    const int gid     = lane >> 2;                         // 0..7
    const int tid4    = lane & 3;                          // 0..3

    // Per-warp accumulators: 8 N-tiles x 4 outputs each.
    float acc[V2_NW][4];
    #pragma unroll
    for (int n = 0; n < V2_NW; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f;
        acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }

    for (int kb = 0; kb < n_blocks; ++kb) {
        // -------- Cooperative load of W tile (64 rows x 32 K) --------
        // 512 .b32 reads spread over 128 threads -> 4 passes.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int rl       = p * 16 + tid / 8;          // 0..63
            const int col_byte = 4 * (tid % 8);             // 0,4,...,28
            const int row      = row0 + rl;
            unsigned v = 0;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int8_t  *qs_plane = cuda_helpers::q8_qs_plane(row_base, n_blocks);
                v = load_int_align2(reinterpret_cast<const uint8_t *>(qs_plane + kb * 32), col_byte);
            }
            *reinterpret_cast<unsigned *>(&W_qs[rl][col_byte]) = v;
        }
        // W_d: 64 floats. First 64 threads each load one.
        if (tid < V2_M_PER_CTA) {
            const int row = row0 + tid;
            float d = 0.0f;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                d = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            }
            W_d[tid] = d;
        }

        // -------- Cooperative load of Y tile (128 batch x 32 K) --------
        // 1024 .b32 reads spread over 128 threads -> 8 passes.
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int bl       = p * 16 + tid / 8;
            const int col_byte = 4 * (tid % 8);
            const int batch_idx = col00 + bl;
            unsigned v = 0;
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(batch_idx) * stride_y_row + kb;
                v = *reinterpret_cast<const unsigned *>(&yblk->qs[col_byte]);
            }
            *reinterpret_cast<unsigned *>(&Y_qs[bl][col_byte]) = v;
        }
        // Y_d: 128 floats, one per thread.
        {
            const int batch_idx = col00 + tid;
            float d = 0.0f;
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(batch_idx) * stride_y_row + kb;
                d = __low2float(yblk->ds);
            }
            Y_d[tid] = d;
        }

        __syncthreads();

        // -------- Compute --------
        // A fragment (16 rows x 32 K). Per PTX:
        //   a0 row=gid     cols=4*tid4..+3
        //   a1 row=gid+8   cols=4*tid4..+3
        //   a2 row=gid     cols=4*tid4+16..+19
        //   a3 row=gid+8   cols=4*tid4+16..+19
        const int rl_top = warp_id * V1_M_TILE + gid;        // 0..63 (within tile)
        const int rl_bot = rl_top + 8;
        const unsigned a0 = *reinterpret_cast<unsigned *>(&W_qs[rl_top][4 * tid4]);
        const unsigned a1 = *reinterpret_cast<unsigned *>(&W_qs[rl_bot][4 * tid4]);
        const unsigned a2 = *reinterpret_cast<unsigned *>(&W_qs[rl_top][4 * tid4 + 16]);
        const unsigned a3 = *reinterpret_cast<unsigned *>(&W_qs[rl_bot][4 * tid4 + 16]);
        const float d_w_top = W_d[rl_top];
        const float d_w_bot = W_d[rl_bot];

        #pragma unroll
        for (int n = 0; n < V2_NW; ++n) {
            const int N0 = n * 8;
            // B fragment (32 K x 8 cols):
            //   b0 rows=4*tid4..+3      col=gid
            //   b1 rows=4*tid4+16..+19  col=gid
            const unsigned b0 = *reinterpret_cast<unsigned *>(&Y_qs[N0 + gid][4 * tid4]);
            const unsigned b1 = *reinterpret_cast<unsigned *>(&Y_qs[N0 + gid][4 * tid4 + 16]);
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma_m16n8k32_s8s8s32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

            // d_y for batch col (N0+2*tid4) lives in lane (gid=2*tid4, *) -> src = 8*tid4.
            const float d_y_my    = Y_d[N0 + gid];
            const float d_y_left  = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4,     V2_WARP);
            const float d_y_right = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4 + 4, V2_WARP);

            acc[n][0] += d_w_top * d_y_left  * static_cast<float>(c0);
            acc[n][1] += d_w_top * d_y_right * static_cast<float>(c1);
            acc[n][2] += d_w_bot * d_y_left  * static_cast<float>(c2);
            acc[n][3] += d_w_bot * d_y_right * static_cast<float>(c3);
        }

        __syncthreads();
    }

    // -------- Write back --------
    const int row_top = row0 + warp_id * V1_M_TILE + gid;
    const int row_bot = row_top + 8;
    const bool top_in = need_check ? (row_top < static_cast<int>(rows)) : true;
    const bool bot_in = need_check ? (row_bot < static_cast<int>(rows)) : true;
    #pragma unroll
    for (int n = 0; n < V2_NW; ++n) {
        const int col_left  = col00 + n * 8 + 2 * tid4;
        const int col_right = col_left + 1;
        const bool left_in  = need_check ? (col_left  < static_cast<int>(batch)) : true;
        const bool right_in = need_check ? (col_right < static_cast<int>(batch)) : true;
        if (left_in) {
            if (top_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_top] = acc[n][0];
            if (bot_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_bot] = acc[n][2];
        }
        if (right_in) {
            if (top_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_top] = acc[n][1];
            if (bot_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_bot] = acc[n][3];
        }
    }
}

// v3: same shape as v2 (4-warp 64x128 tile) but:
//   - Replaces per-lane shmem reads with ldmatrix.x4 (A) + ldmatrix.x2 (B).
//   - Uses a 16-byte-aligned shmem stride (12 ints/row = 48 bytes) so each
//     lane's ldmatrix address is 16-byte aligned. The 4 pad ints/row also
//     stagger banks to reduce conflicts on the K=4 ints offset access.
//
// Result on RTX Pro 6000 Blackwell, Qwen3.6-27B prefill (parity-correct):
//   1K: v2 2132 / v3 1894 tok/s   (v3 -11%)
//   2K: v2 2026 / v3 1800 tok/s   (v3 -11%)
//   4K: v2 1678 / v3 1544 tok/s   (v3  -8%)
// vs HGEMM 2828/2755/2199 — both still well behind cuBLAS.
//
// Why v3 regresses: ldmatrix replaces per-lane shmem reads, but those reads
// were never the bottleneck. The real cost is the gmem->shmem weight fill
// per K block, which both v2 and v3 perform with synchronous global loads.
// v3 loses on two fronts:
//   - 48-byte row stride vs v2's 36-byte stride inflates shmem footprint.
//   - ldmatrix.x2 only uses 16 of 32 lanes as address providers (B fragment
//     is 8x8, addresses duplicated across the upper half-warp).
//
// v4 path: cp.async double-buffered shmem fills to overlap the next K
// block's gmem read with the current MMA. Requires repacking Q8_0 weights
// to a 16-byte-aligned SoA layout (drop the 2-byte FP16 d-prefix, store
// qs and d in separate tables) so cp.async's 16-byte transfer is usable.
// That's a multi-day effort; v3 is kept as opt-in (QW3_MMQ_VERSION=3) but
// default remains v2.
// Note: cp.async double-buffering not viable here because Q8_0 weight rows
// aren't 16-byte aligned (FP16 d-prefix offsets the int8 qs to byte 2).
// Repacking weights to drop that prefix is the v4 path.
static constexpr int V3_KSTRIDE = 12;  // ints per row in shmem (8 data + 4 pad)
static constexpr int V3_NDATA   = 8;   // ints of actual data per row

template <bool need_check>
__launch_bounds__(V2_THREADS, 2)
__global__ void mmq_q8_0_v3_kernel(
        const uint8_t   * __restrict__ wq,
        const block_q8_1 * __restrict__ ya,
        float           * __restrict__ dst,
        uint32_t          rows,
        uint32_t          cols,
        uint32_t          batch,
        uint32_t          stride_y_row,
        uint32_t          stride_dst_row)
{
    __shared__ __align__(16) int   W_qs[V2_M_PER_CTA * V3_KSTRIDE];
    __shared__              float  W_d [V2_M_PER_CTA];
    __shared__ __align__(16) int   Y_qs[V2_N_PER_CTA * V3_KSTRIDE];
    __shared__              float  Y_d [V2_N_PER_CTA];

    const int n_blocks = static_cast<int>(cols / MMQ_QK);
    const int row0  = static_cast<int>(blockIdx.x) * V2_M_PER_CTA;
    const int col00 = static_cast<int>(blockIdx.y) * V2_N_PER_CTA;

    const int warp_id = static_cast<int>(threadIdx.y);    // 0..3
    const int lane    = static_cast<int>(threadIdx.x);    // 0..31
    const int tid     = warp_id * V2_WARP + lane;          // 0..127

    // Per-warp accumulators: 16 N-tiles x 4 outputs each.
    float acc[V2_NW][4];
    #pragma unroll
    for (int n = 0; n < V2_NW; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f;
        acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }

    for (int kb = 0; kb < n_blocks; ++kb) {
        // -------- Cooperative load of W tile (64 rows x 8 ints) --------
        // 512 ints across 128 threads -> 4 passes, 1 int/thread/pass.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int flat = p * V2_THREADS + tid;          // 0..511
            const int rl   = flat >> 3;                     // 0..63 (which row)
            const int kqsx = flat & 7;                      // 0..7  (which int in row)
            const int row  = row0 + rl;
            unsigned v = 0;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int8_t  *qs_plane = cuda_helpers::q8_qs_plane(row_base, n_blocks);
                v = load_int_align2(reinterpret_cast<const uint8_t *>(qs_plane + kb * 32), 4 * kqsx);
            }
            W_qs[rl * V3_KSTRIDE + kqsx] = static_cast<int>(v);
        }
        // W_d: 64 floats; first 64 threads each load one.
        if (tid < V2_M_PER_CTA) {
            const int row = row0 + tid;
            float d = 0.0f;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                d = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            }
            W_d[tid] = d;
        }

        // -------- Cooperative load of Y tile (128 rows x 8 ints) --------
        // 1024 ints across 128 threads -> 8 passes.
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int flat = p * V2_THREADS + tid;
            const int bl   = flat >> 3;                     // 0..127
            const int kqsx = flat & 7;                      // 0..7
            const int batch_idx = col00 + bl;
            unsigned v = 0;
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(batch_idx) * stride_y_row + kb;
                v = *reinterpret_cast<const unsigned *>(&yblk->qs[4 * kqsx]);
            }
            Y_qs[bl * V3_KSTRIDE + kqsx] = static_cast<int>(v);
        }
        // Y_d: 128 floats, one per thread.
        {
            const int batch_idx = col00 + tid;
            float d = 0.0f;
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(batch_idx) * stride_y_row + kb;
                d = __low2float(yblk->ds);
            }
            Y_d[tid] = d;
        }

        __syncthreads();

        // -------- Compute --------
        // A fragment (16 rows x 32 K = 16 rows x 8 ints): 1 ldmatrix.x4 / warp.
        //   xs = &W_qs[16*warp_id + (lane%16)][(lane/16)*4]
        unsigned a0, a1, a2, a3;
        {
            const int *base = &W_qs[(warp_id * V1_M_TILE) * V3_KSTRIDE];
            ldmatrix_x4(a0, a1, a2, a3, base, V3_KSTRIDE);
        }

        // d_w_top/d_w_bot: per-lane-row scales.
        const int gid    = lane >> 2;     // 0..7 (row within 8)
        const int tid4   = lane & 3;      // 0..3 (col within 4)
        const float d_w_top = W_d[warp_id * V1_M_TILE + gid];
        const float d_w_bot = W_d[warp_id * V1_M_TILE + gid + 8];

        #pragma unroll
        for (int n = 0; n < V2_NW; ++n) {
            const int N0 = n * 8;
            // B fragment (32 K x 8 cols = 8 rows x 8 ints): 1 ldmatrix.x2.
            //   xs = &Y_qs[N0 + (lane%8)][((lane/8) & 1) * 4]
            unsigned b0, b1;
            {
                const int *base = &Y_qs[N0 * V3_KSTRIDE];
                ldmatrix_x2(b0, b1, base, V3_KSTRIDE);
            }
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma_m16n8k32_s8s8s32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

            // d_y for batch col (N0+2*tid4) lives in lane (gid=2*tid4, *) -> src = 8*tid4.
            const float d_y_my    = Y_d[N0 + gid];
            const float d_y_left  = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4,     V2_WARP);
            const float d_y_right = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4 + 4, V2_WARP);

            acc[n][0] += d_w_top * d_y_left  * static_cast<float>(c0);
            acc[n][1] += d_w_top * d_y_right * static_cast<float>(c1);
            acc[n][2] += d_w_bot * d_y_left  * static_cast<float>(c2);
            acc[n][3] += d_w_bot * d_y_right * static_cast<float>(c3);
        }

        __syncthreads();
    }

    // -------- Write back (same as v2) --------
    const int gid_o  = lane >> 2;
    const int tid4_o = lane & 3;
    const int row_top = row0 + warp_id * V1_M_TILE + gid_o;
    const int row_bot = row_top + 8;
    const bool top_in = need_check ? (row_top < static_cast<int>(rows)) : true;
    const bool bot_in = need_check ? (row_bot < static_cast<int>(rows)) : true;
    #pragma unroll
    for (int n = 0; n < V2_NW; ++n) {
        const int col_left  = col00 + n * 8 + 2 * tid4_o;
        const int col_right = col_left + 1;
        const bool left_in  = need_check ? (col_left  < static_cast<int>(batch)) : true;
        const bool right_in = need_check ? (col_right < static_cast<int>(batch)) : true;
        if (left_in) {
            if (top_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_top] = acc[n][0];
            if (bot_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_bot] = acc[n][2];
        }
        if (right_in) {
            if (top_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_top] = acc[n][1];
            if (bot_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_bot] = acc[n][3];
        }
    }
}

// ===========================================================================
// v4: llama.cpp-style mul_mat_q port.
//
// Geometry mirrors llama.cpp/ggml/src/ggml-cuda/mmq.cuh (mul_mat_q for
// Q8_0):
//   mmq_y    = 128       (rows per CTA, fixed)
//   mmq_x    = 128       (batch cols per CTA, this v4 supports x=128 only)
//   nwarps   = 8         (256 threads/CTA)
//   ITER_K   = 256       (8 Q8_0 blocks per outer K iter)
//   tile_x   = MMQ_MMA_TILE_X_K_Q8_0 = 76 ints/row = 304 B
//                (64 ints quants + 8 ints d-scales + 4 ints pad)
//   tile_y   = MMQ_TILE_Y_K           = 36 ints/row
//                (4 ints d4 scales | 32 ints quants — block_q8_1_mmq laid
//                 out as [d4 | qs])
//
// Compared with v2 (4-warp 64x128):
//   - mmq_y=128 vs 64 -> 2x more weight reuse per shmem load (the dominant
//     cost; v2 shmem-fills weight 1.5-2x more bytes per K block)
//   - 8 warps vs 4    -> 2x parallel MMAs per CTA
//   - ITER_K=256 vs 32 -> tile_x loaded once for 8 K-blocks
//   - ldmatrix.x4/x2 fragment loads (replaces v2's per-lane shmem reads)
//   - Y format: 144-byte block_q8_1_mmq (4 floats d4 | 128 int8 qs),
//     super-block-major × j-minor. Lets us read mmq_x*36 contiguous ints
//     per K super-iter into shmem, no strided gather.
//
// Per CTA shmem:
//   x_qs : 128 * 76 ints = 38912 ints = ~38 KB
//   y    : 128 * 36 ints = 4608  ints = ~18 KB
//                                       ~56 KB total (well under 100 KB).
//
// Stream-K is NOT used by v4 (kept for v5 if needed). Standard 2D tile
// decomposition: grid (nty, ntx) where nty = ceil(rows/128), ntx =
// ceil(batch/128). For Qwen 3.6 27B prefill at 1K-4K, the wave count is
// already 8-10, and the stream-K tail-absorption gain is bounded by the
// last partial wave (~2-5%). Defer that complexity.
//
// Result targets vs HGEMM on RTX Pro 6000 Blackwell:
//   1K: HGEMM 2828 -> v4 ≥ HGEMM (ldmatrix + 2x reuse should win)
//   2K: HGEMM 2755 -> v4 ≥ HGEMM
//   4K: HGEMM 2199 -> v4 ≥ HGEMM
// If v4 hits these we'll flip QW3_MMQ_VERSION default to 4.

static constexpr int MMQ_NWARPS                 = 8;
static constexpr int MMQ_THREADS                = MMQ_NWARPS * V2_WARP;       // 256
static constexpr int MMQ_X                      = 128;
static constexpr int MMQ_Y                      = 128;
static constexpr int MMQ_TILE_NE_K              = 32;
static constexpr int MMQ_ITER_K                 = 256;
static constexpr int QI8_0                      = 8;
static constexpr int QI8_1                      = 8;
static constexpr int MMQ_MMA_TILE_X_K_Q8_0      = 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;  // 76
static constexpr int MMQ_TILE_Y_K               = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;             // 36
static_assert(MMQ_MMA_TILE_X_K_Q8_0 % 8 == 4, "Wrong padding.");

// Q8_1_MMQ block — 144 B, 4 fp32 d-scales + 128 int8 qs. Defined here in
// the anonymous namespace; matches block_q8_1_mmq in mmvq_q8.cu.
struct alignas(16) block_q8_1_mmq_t {
    float  d4[4];
    int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq_t) == 144, "unexpected block_q8_1_mmq_t size");

// ldmatrix.x4 from explicit base. Used by v4 vec_dot to load A fragments.
__device__ __forceinline__ void v4_ldmatrix_x4(
        unsigned &a0, unsigned &a1, unsigned &a2, unsigned &a3,
        const int *xs_base) {
    unsigned smem = __cvta_generic_to_shared(xs_base);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(smem));
}

// Load tile_x for the v4 kernel. Mirrors llama.cpp's load_tiles_q8_0 with
// nwarps=8, threads_per_row=32, nrows=1.
//
//   wq        : raw Q8_0 weight (rows * n_blocks * 34 bytes), row-major
//   x_qs      : shmem int buffer [mmq_y * MMQ_MMA_TILE_X_K_Q8_0]
//   x_df      : shmem float buffer (alias of x_qs + 2*MMQ_TILE_NE_K)
//   kbx0      : starting Q8_0 block index along K
//   row0      : first row of the tile in global memory
//   stride_in_blocks : Q8_0 blocks per weight row (= cols / 32)
template <bool need_check>
__device__ __forceinline__ void load_tiles_q8_0_v4(
        const uint8_t * __restrict__ wq,
        int           * __restrict__ x_tile,
        int             kbx0,
        int             row0,
        int             rows,
        int             stride_in_blocks)
{
    int   * x_qs = x_tile;
    float * x_df = reinterpret_cast<float *>(x_qs + 2 * MMQ_TILE_NE_K);

    constexpr int threads_per_row = V2_WARP;            // 32
    constexpr int nrows            = V2_WARP / threads_per_row;  // 1
    const int txi  = static_cast<int>(threadIdx.x);
    const int kbx  = txi / QI8_0;                       // 0..3 (first half-tile block index)
    const int kqsx = txi % QI8_0;                       // 0..7 (which int within the block's qs)

    // Outer loop over rows: 8 warps each handle one row per iter, 16 iters total.
    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += nrows * MMQ_NWARPS) {
        int i = i0 + (nrows == 1 ? static_cast<int>(threadIdx.y)
                                 : static_cast<int>(threadIdx.y) * nrows + txi / threads_per_row);
        const int row = row0 + i;
        const bool in_range = need_check ? (row < rows) : true;
        const int row_clamped = in_range ? row : (rows - 1);  // safe sentinel
        const uint8_t *row_base = wq + static_cast<size_t>(row_clamped) * stride_in_blocks * 34;
        const int8_t  *qs_plane = cuda_helpers::q8_qs_plane(row_base, stride_in_blocks);
        const int8_t  *qs_blk0  = qs_plane + (kbx0 + kbx) * 32;
        const int8_t  *qs_blk1  = qs_plane + (kbx0 + kbx + MMQ_TILE_NE_K / QI8_0) * 32;

        const unsigned v0 = in_range ? load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk0), 4 * kqsx) : 0u;
        const unsigned v1 = in_range ? load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk1), 4 * kqsx) : 0u;

        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + 0               + txi] = static_cast<int>(v0);
        x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + MMQ_TILE_NE_K   + txi] = static_cast<int>(v1);
    }

    // Load d-scales. blocks_per_tile_x_row = 8, rows_per_warp = 32/8 = 4.
    constexpr int blocks_per_tile_x_row = 2 * MMQ_TILE_NE_K / QI8_0;  // 8
    constexpr int rows_per_warp         = V2_WARP / blocks_per_tile_x_row;  // 4
    const int kbxd = txi % blocks_per_tile_x_row;       // 0..7 (block within tile)

    #pragma unroll
    for (int i0 = 0; i0 < MMQ_Y; i0 += MMQ_NWARPS * rows_per_warp) {
        const int i = i0 + static_cast<int>(threadIdx.y) * rows_per_warp + txi / blocks_per_tile_x_row;
        const int row = row0 + i;
        const bool in_range = need_check ? (row < rows) : true;
        const int row_clamped = in_range ? row : (rows - 1);
        const uint8_t *row_base = wq + static_cast<size_t>(row_clamped) * stride_in_blocks * 34;
        const float d = in_range
            ? __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kbx0 + kbxd]))
            : 0.0f;
        x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = d;
    }
}

// vec_dot for the v4 MMQ kernel. Mirrors llama.cpp's vec_dot_q8_0_q8_1_mma
// (NVIDIA non-AMD branch in mmq.cuh:1218-1293).
//
// Per warp: with mmq_x>=48 -> granularity=16, rows_per_warp=32, ntx=2.
// 8 warps total; warp pairs (warp_id/2 == warp_id'/2) share a 32-row stripe
// (i0=(warp_id/2)*32). Within a pair the two warps split the j dimension:
// the y pointer for warp w is offset by (w%2)*8*MMQ_TILE_Y_K ints, so warp 0
// reads y[j0..j0+8) and warp 1 reads y[j0+8..j0+16).
//
// Per call this consumes MMQ_TILE_NE_K=32 ints of K (4 K-subblocks of QI8_0
// ints each; one Q8_0 block = QI8_0=8 ints = 32 K-elements). Outer caller
// passes k00=0 then k00=32 to consume the full 8-block superblock loaded
// into tile_x.
template <int mmq_x>
__device__ __forceinline__ void vec_dot_q8_0_q8_1_mma_v4(
        const int * __restrict__ x_tile,
        const int * __restrict__ y_tile_in,
        float     * __restrict__ sum,
        int                       k00)
{
    constexpr int granularity   = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;       // 32 (or 16)
    constexpr int tile_C_I      = 16;
    constexpr int tile_C_J      = 8;
    constexpr int tile_C_ne     = 4;                     // 16*8/32
    constexpr int ntx           = rows_per_warp / tile_C_I;  // 2 (or 1)
    constexpr int K_steps       = MMQ_TILE_NE_K / QI8_0; // 4

    const int lane    = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(threadIdx.y);
    const int i0      = (warp_id / ntx) * rows_per_warp;
    const int * y_tile = y_tile_in + (warp_id % ntx) * (tile_C_J * MMQ_TILE_Y_K);

    const int   * x_qs = x_tile;
    const float * x_df = reinterpret_cast<const float *>(x_qs + 2 * MMQ_TILE_NE_K);
    const int   * y_qs = y_tile + 4;                          // skip 4 d-scales
    const float * y_df = reinterpret_cast<const float *>(y_tile);

    // Pre-load all A fragments and dA scales (ntx × K_steps).
    unsigned A0[ntx][K_steps];
    unsigned A1[ntx][K_steps];
    unsigned A2[ntx][K_steps];
    unsigned A3[ntx][K_steps];
    float    dA[ntx][tile_C_ne / 2][K_steps];

    #pragma unroll
    for (int n = 0; n < ntx; ++n) {
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;
            // ldmatrix.x4 base addr: row=lane%16, col=(lane/16)*4 within
            // the 16x8 fragment. Base of fragment = x_qs[(i0+n*16)*76 + k0].
            const int *xs_base = x_qs + (i0 + n * tile_C_I + (lane & 15)) * MMQ_MMA_TILE_X_K_Q8_0
                                       + k0 + (lane >> 4) * 4;
            v4_ldmatrix_x4(A0[n][k01 / QI8_0],
                           A1[n][k01 / QI8_0],
                           A2[n][k01 / QI8_0],
                           A3[n][k01 / QI8_0],
                           xs_base);
        }
        #pragma unroll
        for (int l = 0; l < tile_C_ne / 2; ++l) {
            // tile_C::get_i(2*l) = ((2l/2)%2)*8 + lane/4 = (l%2)*8 + lane/4
            // (For ne/2=2 with mma::tile<16,8>, l ranges 0..1.)
            const int i_local = i0 + n * tile_C_I + l * 8 + (lane >> 2);
            #pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;
                dA[n][l][k01 / QI8_0] = x_df[i_local * MMQ_MMA_TILE_X_K_Q8_0 + k0 / QI8_0];
            }
        }
    }

    // Iterate over j0 in steps of ntx*tile_C::J = 16. Each warp's j range is
    // (warp_id%ntx)*8 + [0..8), so the warp pair together covers [j0, j0+16).
    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C_J) {
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            // Load tile_B (8 rows K × 8 cols J) via load_generic.
            // tile_B::get_i(0..1) = lane/4
            // tile_B::get_j(0)    = lane%4
            // tile_B::get_j(1)    = 4 + lane%4
            const int b_row = lane >> 2;
            const int b_c0  = lane & 3;
            const int b_c1  = 4 + (lane & 3);
            const int *yb = y_qs + j0 * MMQ_TILE_Y_K + k01;
            const unsigned B0 = static_cast<unsigned>(yb[b_row * MMQ_TILE_Y_K + b_c0]);
            const unsigned B1 = static_cast<unsigned>(yb[b_row * MMQ_TILE_Y_K + b_c1]);

            // dB[l] for l in 0..1: j_local = j0 + (lane%4)*2 + (l%2)
            float dB[tile_C_ne / 2];
            #pragma unroll
            for (int l = 0; l < tile_C_ne / 2; ++l) {
                const int j_local = j0 + (lane & 3) * 2 + l;
                dB[l] = y_df[j_local * MMQ_TILE_Y_K + k01 / QI8_1];
            }

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma_m16n8k32_s8s8s32(
                        c0, c1, c2, c3,
                        A0[n][k01 / QI8_0], A1[n][k01 / QI8_0],
                        A2[n][k01 / QI8_0], A3[n][k01 / QI8_0],
                        B0, B1);
                const float c[4] = { static_cast<float>(c0),
                                     static_cast<float>(c1),
                                     static_cast<float>(c2),
                                     static_cast<float>(c3) };
                #pragma unroll
                for (int l = 0; l < tile_C_ne; ++l) {
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l]
                        += c[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
                }
            }
        }
    }
}

// MMQ v5 — port of llama.cpp's mul_mat_q for Q8_0 (mmq.cuh:1217-1293,
// NVIDIA non-AMD branch). We adopt llama.cpp's struct-based tile<> and
// load_ldmatrix/mma helpers so ptxas treats fragments as register tuples,
// keeping the 32 A registers + 16 dA + 64 sum within the 255-reg/thread
// budget. Their q8_0 mmq_x=128 instantiation hits only 8-24 byte spill;
// our v4 hit 160-208. The struct-of-arrays storage is the missing piece.

namespace v5_tile {

template <int I_, int J_> struct tile_int4 {
    static constexpr int I = I_;
    static constexpr int J = J_;
    int x[4];
};
template <int I_, int J_> struct tile_int2 {
    static constexpr int I = I_;
    static constexpr int J = J_;
    int x[2];
};

using tile_A = tile_int4<16, 8>;       // ne = I*J/32 = 4
using tile_B = tile_int2< 8, 8>;       // ne = I*J/32 = 2
using tile_C = tile_int4<16, 8>;       // ne = I*J/32 = 4

// tile_A 16x8 ldmatrix.x4 — pass the raw shared pointer through "l" and let
// the assembler emit the implicit cvta. Mirrors mma.cuh:830-838.
__device__ __forceinline__ void load_ldmatrix_A(tile_A &t, const int *xs0, int stride) {
    int *xi = t.x;
    const int *xs = xs0 + (threadIdx.x % 16) * stride + (threadIdx.x / 16) * 4;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
        : "l"(xs));
}

// tile_B 8x8 load_generic — direct shmem reads of the 2 ints per lane.
// Comment in llama.cpp mmq.cuh:1268 notes load_generic outperforms
// load_ldmatrix for B at this geometry.
__device__ __forceinline__ void load_generic_B(tile_B &t, const int *xs0, int stride) {
    const int lane = static_cast<int>(threadIdx.x);
    const int i_ = lane / 4;
    const int j0_ = (lane % 4);    // get_j(0) = (lane%4)*2 + 0 -> NO, see below
    // tile<8,8,int>::get_j(l) = (l*4) + (lane%4)
    t.x[0] = xs0[i_ * stride + 0 * 4 + j0_];
    t.x[1] = xs0[i_ * stride + 1 * 4 + j0_];
}

// MMA m16n8k32.s8.s8.s32: C += A * B. Non-volatile asm so ptxas can
// schedule across iterations (matches llama.cpp/mma.cuh:946).
__device__ __forceinline__ void mma(tile_C &C, const tile_A &A, const tile_B &B) {
    int *c = C.x;
    asm(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]),
          "r"(B.x[0]), "r"(B.x[1]));
}

}  // namespace v5_tile

template <int mmq_x>
__device__ __forceinline__ void vec_dot_q8_0_q8_1_mma_v5(
        const int * __restrict__ x_tile,
        const int * __restrict__ y_tile_in,
        float     * __restrict__ sum,
        int                       k00)
{
    constexpr int granularity   = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int tile_C_I      = v5_tile::tile_C::I;   // 16
    constexpr int tile_C_J      = v5_tile::tile_C::J;   // 8
    constexpr int tile_C_ne     = 4;                    // tile_C ne
    constexpr int ntx           = rows_per_warp / tile_C_I;
    constexpr int K_steps       = MMQ_TILE_NE_K / QI8_0;

    const int lane    = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(threadIdx.y);
    const int i0      = (warp_id / ntx) * rows_per_warp;
    const int * y_tile = y_tile_in + (warp_id % ntx) * (tile_C_J * MMQ_TILE_Y_K);

    const int   * x_qs = x_tile;
    const float * x_df = reinterpret_cast<const float *>(x_qs + 2 * MMQ_TILE_NE_K);
    const int   * y_qs = y_tile + 4;
    const float * y_df = reinterpret_cast<const float *>(y_tile);

    // Pre-load A fragments (struct-typed) and dA scales. Lifetime spans the
    // full j0 loop; the struct storage helps ptxas register-allocate them.
    v5_tile::tile_A A[ntx][K_steps];
    float           dA[ntx][tile_C_ne / 2][K_steps];

    #pragma unroll
    for (int n = 0; n < ntx; ++n) {
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;
            const int *xs_base = x_qs + (i0 + n * tile_C_I) * MMQ_MMA_TILE_X_K_Q8_0 + k0;
            v5_tile::load_ldmatrix_A(A[n][k01 / QI8_0], xs_base, MMQ_MMA_TILE_X_K_Q8_0);
        }
    }
    #pragma unroll
    for (int n = 0; n < ntx; ++n) {
        #pragma unroll
        for (int l = 0; l < tile_C_ne / 2; ++l) {
            const int i_local = i0 + n * tile_C_I + l * 8 + (lane >> 2);
            #pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;
                dA[n][l][k01 / QI8_0] = x_df[i_local * MMQ_MMA_TILE_X_K_Q8_0 + k0 / QI8_0];
            }
        }
    }

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C_J) {
        #pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            v5_tile::tile_B B;
            v5_tile::load_generic_B(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            float dB[tile_C_ne / 2];
            #pragma unroll
            for (int l = 0; l < tile_C_ne / 2; ++l) {
                const int j_local = j0 + (lane & 3) * 2 + l;
                dB[l] = y_df[j_local * MMQ_TILE_Y_K + k01 / QI8_1];
            }

            #pragma unroll
            for (int n = 0; n < ntx; ++n) {
                v5_tile::tile_C C;
                C.x[0] = C.x[1] = C.x[2] = C.x[3] = 0;
                v5_tile::mma(C, A[n][k01 / QI8_0], B);
                #pragma unroll
                for (int l = 0; l < tile_C_ne; ++l) {
                    sum[(j0 / tile_C_J + n) * tile_C_ne + l]
                        += static_cast<float>(C.x[l]) * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
                }
            }
        }
    }
}

// v5 outer kernel. Same outer structure as v4 but uses tile<>-based
// fragments and helpers ported from llama.cpp/mmq.cuh.
template <int mmq_x, bool need_check>
__launch_bounds__(MMQ_THREADS, 1)
__global__ void mmq_q8_0_v5_kernel(
        const uint8_t       * __restrict__ wq,
        const int           * __restrict__ y,
        float               * __restrict__ dst,
        uint32_t              rows,
        uint32_t              cols,
        uint32_t              batch,
        uint32_t              ncols_y,
        uint32_t              stride_dst_row)
{
    extern __shared__ int v5_smem[];
    int * tile_x = v5_smem;
    int * tile_y = tile_x + MMQ_Y * MMQ_MMA_TILE_X_K_Q8_0;

    const int row0  = static_cast<int>(blockIdx.x) * MMQ_Y;
    const int col00 = static_cast<int>(blockIdx.y) * mmq_x;
    const int n_blocks = static_cast<int>(cols) / MMQ_QK;
    const int stride_in_blocks = n_blocks;

    constexpr int sum_size = mmq_x * MMQ_Y / (MMQ_NWARPS * V2_WARP);
    float sum[sum_size];
    #pragma unroll
    for (int s = 0; s < sum_size; ++s) sum[s] = 0.0f;

    constexpr int blocks_per_iter = MMQ_ITER_K / MMQ_QK;       // 8
    constexpr int sz_y_block_int  = sizeof(block_q8_1_mmq_t) / sizeof(int);  // 36

    auto load_y_half = [&](int sb_index) {
        const int total = mmq_x * MMQ_TILE_Y_K;
        const int tid = static_cast<int>(threadIdx.y) * V2_WARP + static_cast<int>(threadIdx.x);
        #pragma unroll
        for (int l0 = 0; l0 < total; l0 += MMQ_THREADS) {
            const int l = l0 + tid;
            if (l >= total) break;
            const int j_in_tile = l / MMQ_TILE_Y_K;
            const int k_in_block = l % MMQ_TILE_Y_K;
            const int j_global = col00 + j_in_tile;
            int v = 0;
            const bool in_range = need_check ? (j_global < static_cast<int>(batch)) : true;
            if (in_range) {
                const int64_t int_idx =
                    (static_cast<int64_t>(sb_index) * ncols_y + j_global) * sz_y_block_int
                    + k_in_block;
                v = y[int_idx];
            }
            tile_y[l] = v;
        }
    };

    for (int kb0 = 0; kb0 < n_blocks; kb0 += blocks_per_iter) {
        load_tiles_q8_0_v4<need_check>(
                wq, tile_x, kb0, row0, static_cast<int>(rows), stride_in_blocks);

        const int sb0 = kb0 / 4;

        load_y_half(sb0);
        __syncthreads();
        vec_dot_q8_0_q8_1_mma_v5<mmq_x>(tile_x, tile_y, sum, /*k00=*/0);
        __syncthreads();

        load_y_half(sb0 + 1);
        __syncthreads();
        vec_dot_q8_0_q8_1_mma_v5<mmq_x>(tile_x, tile_y, sum, /*k00=*/MMQ_TILE_NE_K);
        __syncthreads();
    }

    // Writeback identical to v4.
    constexpr int granularity   = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int tile_C_I      = 16;
    constexpr int tile_C_J      = 8;
    constexpr int tile_C_ne     = 4;
    constexpr int ntx           = rows_per_warp / tile_C_I;

    const int lane    = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(threadIdx.y);
    const int i0_o    = (warp_id / ntx) * (ntx * tile_C_I);

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C_J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C_ne; ++l) {
                const int j_local = j0 + (warp_id % ntx) * tile_C_J + (lane & 3) * 2 + (l % 2);
                const int j_global = col00 + j_local;
                if (need_check && j_global >= static_cast<int>(batch)) continue;
                if (j_local >= mmq_x) continue;
                const int i_local  = i0_o + n * tile_C_I + (l / 2) * 8 + (lane >> 2);
                const int i_global = row0 + i_local;
                if (need_check && i_global >= static_cast<int>(rows)) continue;
                if (i_local >= MMQ_Y) continue;
                dst[static_cast<size_t>(j_global) * stride_dst_row + i_global]
                    = sum[(j0 / tile_C_J + n) * tile_C_ne + l];
            }
        }
    }
}

// v4 outer kernel. Grid: (rows/MMQ_Y, batch/MMQ_X). Block: (32, 8) = 256.
template <int mmq_x, bool need_check>
__launch_bounds__(MMQ_THREADS, 1)
__global__ void mmq_q8_0_v4_kernel(
        const uint8_t       * __restrict__ wq,
        const int           * __restrict__ y,
        float               * __restrict__ dst,
        uint32_t              rows,
        uint32_t              cols,
        uint32_t              batch,
        uint32_t              ncols_y,        // == batch (rounded if padded)
        uint32_t              stride_dst_row)
{
    extern __shared__ int v4_smem[];
    int * tile_x = v4_smem;
    int * tile_y = tile_x + MMQ_Y * MMQ_MMA_TILE_X_K_Q8_0;

    const int row0  = static_cast<int>(blockIdx.x) * MMQ_Y;
    const int col00 = static_cast<int>(blockIdx.y) * mmq_x;
    const int n_blocks = static_cast<int>(cols) / MMQ_QK;
    const int stride_in_blocks = n_blocks;

    // Per-warp accumulators. Layout matches llama:
    //   sum[(j0/tile_C::J + n) * tile_C::ne + l]   for (j0,n,l) ∈ ...
    // Total per warp = mmq_x * MMQ_Y / (MMQ_NWARPS * 32) = mmq_x * 16 / 32.
    constexpr int sum_size = mmq_x * MMQ_Y / (MMQ_NWARPS * V2_WARP);
    float sum[sum_size];
    #pragma unroll
    for (int s = 0; s < sum_size; ++s) sum[s] = 0.0f;

    constexpr int blocks_per_iter = MMQ_ITER_K / MMQ_QK;       // 8
    constexpr int sz_y_block_int  = sizeof(block_q8_1_mmq_t) / sizeof(int);  // 36

    auto load_y_half = [&](int sb_index) {
        const int total = mmq_x * MMQ_TILE_Y_K;
        const int tid = static_cast<int>(threadIdx.y) * V2_WARP + static_cast<int>(threadIdx.x);
        #pragma unroll
        for (int l0 = 0; l0 < total; l0 += MMQ_THREADS) {
            const int l = l0 + tid;
            if (l >= total) break;
            const int j_in_tile = l / MMQ_TILE_Y_K;
            const int k_in_block = l % MMQ_TILE_Y_K;
            const int j_global = col00 + j_in_tile;
            int v = 0;
            const bool in_range = need_check ? (j_global < static_cast<int>(batch)) : true;
            if (in_range) {
                const int64_t int_idx =
                    (static_cast<int64_t>(sb_index) * ncols_y + j_global) * sz_y_block_int
                    + k_in_block;
                v = y[int_idx];
            }
            tile_y[l] = v;
        }
    };

    for (int kb0 = 0; kb0 < n_blocks; kb0 += blocks_per_iter) {
        // Load 8 Q8_0 blocks (256 K elements) of tile_x.
        load_tiles_q8_0_v4<need_check>(
                wq, tile_x, kb0, row0, static_cast<int>(rows), stride_in_blocks);

        const int sb0 = kb0 / 4;   // super-block index for first half (4 Q8_0 blocks per super-block)

        // Half 0: super-block sb0 (K elements 0..127 of the 256-K outer iter).
        load_y_half(sb0);
        __syncthreads();

        vec_dot_q8_0_q8_1_mma_v4<mmq_x>(tile_x, tile_y, sum, /*k00=*/0);

        __syncthreads();

        // Half 1: super-block sb0+1 (K elements 128..255).
        load_y_half(sb0 + 1);
        __syncthreads();

        vec_dot_q8_0_q8_1_mma_v4<mmq_x>(tile_x, tile_y, sum, /*k00=*/MMQ_TILE_NE_K);

        __syncthreads();
    }

    // Writeback: mirrors mmq_write_back_mma for the NVIDIA branch.
    constexpr int granularity   = (mmq_x >= 48) ? 16 : 8;
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int tile_C_I      = 16;
    constexpr int tile_C_J      = 8;
    constexpr int tile_C_ne     = 4;
    constexpr int ntx           = rows_per_warp / tile_C_I;

    const int lane    = static_cast<int>(threadIdx.x);
    const int warp_id = static_cast<int>(threadIdx.y);
    const int i0_o    = (warp_id / ntx) * (ntx * tile_C_I);

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C_J) {
        #pragma unroll
        for (int n = 0; n < ntx; ++n) {
            #pragma unroll
            for (int l = 0; l < tile_C_ne; ++l) {
                // tile_C::get_j(l) = (lane%4)*2 + (l%2)
                const int j_local = j0 + (warp_id % ntx) * tile_C_J + (lane & 3) * 2 + (l % 2);
                const int j_global = col00 + j_local;
                if (need_check && j_global >= static_cast<int>(batch)) continue;
                if (j_local >= mmq_x) continue;
                // tile_C::get_i(l) = ((l/2)%2)*8 + lane/4 = (l/2)*8 + lane/4
                const int i_local  = i0_o + n * tile_C_I + (l / 2) * 8 + (lane >> 2);
                const int i_global = row0 + i_local;
                if (need_check && i_global >= static_cast<int>(rows)) continue;
                if (i_local >= MMQ_Y) continue;
                dst[static_cast<size_t>(j_global) * stride_dst_row + i_global]
                    = sum[(j0 / tile_C_J + n) * tile_C_ne + l];
            }
        }
    }
}

// Old v1 single-warp kernel — used only for tiny shapes (<64 rows / <64 batch)
// that don't reach the v2 64x64 tile granularity. Identical to the previous-
// commit kernel but folded into this file's anonymous namespace.
__launch_bounds__(V2_WARP, 4)
__global__ void mmq_q8_0_v1_kernel(
        const uint8_t   * __restrict__ wq,
        const block_q8_1 * __restrict__ ya,
        float           * __restrict__ dst,
        uint32_t          rows,
        uint32_t          cols,
        uint32_t          batch,
        uint32_t          stride_y_row,
        uint32_t          stride_dst_row)
{
    const int n_blocks = static_cast<int>(cols / MMQ_QK);
    const int row0  = static_cast<int>(blockIdx.x) * V1_M_TILE;
    const int col00 = static_cast<int>(blockIdx.y) * V1_N_PER_CTA;

    const int lane = static_cast<int>(threadIdx.x);
    const int gid  = lane >> 2;
    const int tid4 = lane & 3;

    const int row_top = row0 + gid;
    const int row_bot = row0 + gid + 8;
    const bool row_top_in_range = row_top < static_cast<int>(rows);
    const bool row_bot_in_range = row_bot < static_cast<int>(rows);

    float acc0[V1_NW] = {0.0f};
    float acc1[V1_NW] = {0.0f};
    float acc2[V1_NW] = {0.0f};
    float acc3[V1_NW] = {0.0f};

    for (int kb = 0; kb < n_blocks; ++kb) {
        unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        float d_w_top = 0.0f, d_w_bot = 0.0f;
        if (row_top_in_range) {
            const uint8_t *row_base = wq + static_cast<size_t>(row_top) * n_blocks * 34;
            const int8_t  *qs_blk  = cuda_helpers::q8_qs_plane(row_base, n_blocks) + kb * 32;
            d_w_top = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            a0 = load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk), 4 * tid4);
            a2 = load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk), 4 * tid4 + 16);
        }
        if (row_bot_in_range) {
            const uint8_t *row_base = wq + static_cast<size_t>(row_bot) * n_blocks * 34;
            const int8_t  *qs_blk  = cuda_helpers::q8_qs_plane(row_base, n_blocks) + kb * 32;
            d_w_bot = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            a1 = load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk), 4 * tid4);
            a3 = load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk), 4 * tid4 + 16);
        }

        #pragma unroll
        for (int nw = 0; nw < V1_NW; ++nw) {
            const int col0 = col00 + nw * V1_N_TILE;
            const int my_batch_row = col0 + gid;
            const bool my_batch_in_range = my_batch_row < static_cast<int>(batch);

            unsigned b0 = 0, b1 = 0;
            float d_x_my = 0.0f;
            if (my_batch_in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(my_batch_row) * stride_y_row + kb;
                d_x_my = __low2float(yblk->ds);
                b0 = *reinterpret_cast<const unsigned *>(&yblk->qs[4 * tid4]);
                b1 = *reinterpret_cast<const unsigned *>(&yblk->qs[4 * tid4 + 16]);
            }

            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma_m16n8k32_s8s8s32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

            const float d_x_left  = __shfl_sync(0xffffffffu, d_x_my, 8 * tid4,     V2_WARP);
            const float d_x_right = __shfl_sync(0xffffffffu, d_x_my, 8 * tid4 + 4, V2_WARP);

            acc0[nw] += d_w_top * d_x_left  * static_cast<float>(c0);
            acc1[nw] += d_w_top * d_x_right * static_cast<float>(c1);
            acc2[nw] += d_w_bot * d_x_left  * static_cast<float>(c2);
            acc3[nw] += d_w_bot * d_x_right * static_cast<float>(c3);
        }
    }

    #pragma unroll
    for (int nw = 0; nw < V1_NW; ++nw) {
        const int col0 = col00 + nw * V1_N_TILE;
        const int col_left  = col0 + 2 * tid4;
        const int col_right = col_left + 1;
        if (col_left < static_cast<int>(batch)) {
            if (row_top_in_range) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_top] = acc0[nw];
            if (row_bot_in_range) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_bot] = acc2[nw];
        }
        if (col_right < static_cast<int>(batch)) {
            if (row_top_in_range) dst[static_cast<size_t>(col_right) * stride_dst_row + row_top] = acc1[nw];
            if (row_bot_in_range) dst[static_cast<size_t>(col_right) * stride_dst_row + row_bot] = acc3[nw];
        }
    }
}

// ---------------------------------------------------------------------------
// v6: same geometry as v2 (64x128 tile, 4 warps) but with cp.async double-
// buffered Y (activation) tile fills, so the next K-block's Y gmem read
// overlaps the current K-block's MMA. W stays single-buffer/synchronous —
// Q8_0 weight rows have a 2-byte FP16 d-prefix that breaks 4-byte alignment
// for half of K-blocks, so cp.async (which requires 4/8/16-byte alignment)
// cannot be used on W without repacking weights. Repacking would either
// duplicate weights (memory hostile) or rewrite the in-place GGUF Q8_0
// blob (incompatible with the rest of the engine).
//
// Y is 36-byte block_q8_1 (half2 ds | int8 qs[32]); qs is 4-byte aligned for
// every block, so cp.async.ca.shared.global of 4 bytes is legal. We use
// 4-byte cp.async because larger sizes would require Y repacking too (qs at
// byte 4 is 4-byte aligned, not 16-byte aligned).
//
// Compared with v2: per K-block adds 1 cp.async commit + 1 wait_group + 1
// sync. Hides ~8 cp.async-size-4 transactions/thread (8 ints/thread × 32 K
// per block) behind W sync load + 16 MMAs/warp. Theoretical: removes the
// gmem-load tail of each K-block's critical path.
//
// Shmem footprint: doubles Y_qs (2 × 128 × 36 = 9 KB) and Y_d (2 × 128 × 4
// = 1 KB) over v2. Plus W_qs (2.3 KB) + W_d (256 B). Total ~13 KB,
// well under 99 KB optin.

__device__ __forceinline__ void cp_async_ca_4(int *smem_dst, const int *gmem_src) {
    unsigned dst = __cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(dst), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_zero_4(int *smem_dst) {
    *smem_dst = 0;
}

__device__ __forceinline__ void cp_async_commit_v6() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group_v6() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

template <bool need_check>
__launch_bounds__(V2_THREADS, 2)
__global__ void mmq_q8_0_v6_kernel(
        const uint8_t   * __restrict__ wq,
        const block_q8_1 * __restrict__ ya,
        float           * __restrict__ dst,
        uint32_t          rows,
        uint32_t          cols,
        uint32_t          batch,
        uint32_t          stride_y_row,
        uint32_t          stride_dst_row)
{
    // Single-buffered W (sync). Double-buffered Y (cp.async).
    __shared__ int8_t W_qs[V2_M_PER_CTA][V2_K_STRIDE];
    __shared__ float  W_d [V2_M_PER_CTA];
    __shared__ int8_t Y_qs[2][V2_N_PER_CTA][V2_K_STRIDE];
    __shared__ float  Y_d [2][V2_N_PER_CTA];

    const int n_blocks = static_cast<int>(cols / MMQ_QK);
    const int row0  = static_cast<int>(blockIdx.x) * V2_M_PER_CTA;
    const int col00 = static_cast<int>(blockIdx.y) * V2_N_PER_CTA;

    const int warp_id = static_cast<int>(threadIdx.y);    // 0..3
    const int lane    = static_cast<int>(threadIdx.x);    // 0..31
    const int tid     = warp_id * V2_WARP + lane;          // 0..127
    const int gid     = lane >> 2;                         // 0..7
    const int tid4    = lane & 3;                          // 0..3

    float acc[V2_NW][4];
    #pragma unroll
    for (int n = 0; n < V2_NW; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f;
        acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }

    // Issue Y[kb] cp.async. 8 passes of 16 batch rows; each thread loads
    // 1 int (4 bytes) per pass. Q8_1 qs sits at byte 4 of each 36-byte
    // block, which is 4-byte aligned (struct is alignas(4)). The Y_d
    // (FP32 scale, 4 bytes) is loaded the same way; it lives at byte 0.
    auto issue_y = [&](int kb, int buf) {
        if (kb >= n_blocks) return;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int bl       = p * 16 + tid / 8;
            const int col_byte = 4 * (tid % 8);
            const int batch_idx = col00 + bl;
            int *dst_int = reinterpret_cast<int *>(&Y_qs[buf][bl][col_byte]);
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(batch_idx) * stride_y_row + kb;
                const int *src_int = reinterpret_cast<const int *>(&yblk->qs[col_byte]);
                cp_async_ca_4(dst_int, src_int);
            } else {
                cp_async_zero_4(dst_int);
            }
        }
        // Y_d: 128 floats. tid covers all of [0..128), so each thread issues
        // exactly one cp.async for d.
        {
            const int batch_idx = col00 + tid;
            int *dst_int = reinterpret_cast<int *>(&Y_d[buf][tid]);
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1 *yblk = ya + static_cast<size_t>(batch_idx) * stride_y_row + kb;
                // half2 ds = (d, s); we want d, the low 16 bits — but we need a
                // float, so dequantize. Issue cp.async on the half2 word and
                // convert in-shmem after wait. To avoid an extra pass we just
                // write a direct float store after the qs cp.async issues —
                // that's a tiny synchronous load and doesn't dominate.
                float d = __low2float(yblk->ds);
                Y_d[buf][tid] = d;
                (void)dst_int;
            } else {
                Y_d[buf][tid] = 0.0f;
                (void)dst_int;
            }
        }
    };

    // Prologue: issue Y[0].
    issue_y(0, 0);
    cp_async_commit_v6();

    for (int kb = 0; kb < n_blocks; ++kb) {
        const int rd = kb & 1;
        const int wr = rd ^ 1;

        // Issue Y[kb+1] before waiting on Y[kb] — overlap.
        issue_y(kb + 1, wr);
        cp_async_commit_v6();

        // Load W[kb] synchronously (cannot use cp.async due to 2-byte-aligned
        // qs offset within Q8_0 weight blocks).
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int rl       = p * 16 + tid / 8;
            const int col_byte = 4 * (tid % 8);
            const int row      = row0 + rl;
            unsigned v = 0;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int8_t  *qs_blk  = cuda_helpers::q8_qs_plane(row_base, n_blocks) + kb * 32;
                v = load_int_align2(reinterpret_cast<const uint8_t *>(qs_blk), col_byte);
            }
            *reinterpret_cast<unsigned *>(&W_qs[rl][col_byte]) = v;
        }
        if (tid < V2_M_PER_CTA) {
            const int row = row0 + tid;
            float d = 0.0f;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                d = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            }
            W_d[tid] = d;
        }

        // Wait on Y[kb]: one group still in flight (Y[kb+1]).
        cp_async_wait_group_v6<1>();
        __syncthreads();

        // Compute on W[kb] / Y[kb].
        const int rl_top = warp_id * V1_M_TILE + gid;
        const int rl_bot = rl_top + 8;
        const unsigned a0 = *reinterpret_cast<unsigned *>(&W_qs[rl_top][4 * tid4]);
        const unsigned a1 = *reinterpret_cast<unsigned *>(&W_qs[rl_bot][4 * tid4]);
        const unsigned a2 = *reinterpret_cast<unsigned *>(&W_qs[rl_top][4 * tid4 + 16]);
        const unsigned a3 = *reinterpret_cast<unsigned *>(&W_qs[rl_bot][4 * tid4 + 16]);
        const float d_w_top = W_d[rl_top];
        const float d_w_bot = W_d[rl_bot];

        #pragma unroll
        for (int n = 0; n < V2_NW; ++n) {
            const int N0 = n * 8;
            const unsigned b0 = *reinterpret_cast<unsigned *>(&Y_qs[rd][N0 + gid][4 * tid4]);
            const unsigned b1 = *reinterpret_cast<unsigned *>(&Y_qs[rd][N0 + gid][4 * tid4 + 16]);
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma_m16n8k32_s8s8s32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

            const float d_y_my    = Y_d[rd][N0 + gid];
            const float d_y_left  = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4,     V2_WARP);
            const float d_y_right = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4 + 4, V2_WARP);

            acc[n][0] += d_w_top * d_y_left  * static_cast<float>(c0);
            acc[n][1] += d_w_top * d_y_right * static_cast<float>(c1);
            acc[n][2] += d_w_bot * d_y_left  * static_cast<float>(c2);
            acc[n][3] += d_w_bot * d_y_right * static_cast<float>(c3);
        }

        __syncthreads();
    }

    // Writeback (identical to v2).
    const int row_top = row0 + warp_id * V1_M_TILE + gid;
    const int row_bot = row_top + 8;
    const bool top_in = need_check ? (row_top < static_cast<int>(rows)) : true;
    const bool bot_in = need_check ? (row_bot < static_cast<int>(rows)) : true;
    #pragma unroll
    for (int n = 0; n < V2_NW; ++n) {
        const int col_left  = col00 + n * 8 + 2 * tid4;
        const int col_right = col_left + 1;
        const bool left_in  = need_check ? (col_left  < static_cast<int>(batch)) : true;
        const bool right_in = need_check ? (col_right < static_cast<int>(batch)) : true;
        if (left_in) {
            if (top_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_top] = acc[n][0];
            if (bot_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_bot] = acc[n][2];
        }
        if (right_in) {
            if (top_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_top] = acc[n][1];
            if (bot_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_bot] = acc[n][3];
        }
    }
}

} // namespace

// ===========================================================================
// v7: cp.async pipelined version of v2.
//
// Same geometry as v2 (4 warps × 64×128 tile, 1 Q8_0 block of K=32 per outer
// iter). The hot loop fetches K-block kb+1's W and Y *while* the MMA for
// kb is in flight, using cp.async.cg.shared.global. Two-stage rotating
// buffer: stage 0 holds kb, stage 1 holds kb+1.
//
// Why this can win over v2 (which lost to HGEMM): v2's hot loop is
// gmem-bound on the W tile read (64 rows × 32 int8 = 2 KB / K block / CTA,
// but issued *synchronously* — every K iter the CTA stalls waiting for
// gmem). cp.async keeps the LSUs busy in parallel with the MMA core. With
// the split-plane Q8 layout (phase A), Plane Q is 16-byte aligned, so
// cp.async.cg.shared.global with 16-byte transfers is now legal (it was
// not for the legacy interleaved 34-B blocks).
//
// Per K block, per CTA:
//   W_qs : 64 rows × 32 int8 = 2048 B → 128 cp.async.16-B issues
//          (1 issue per thread, 128 threads)
//   W_d  : 64 floats from Plane S (synchronous; 64 lanes, one .h read each)
//   Y_qs : 128 rows × 32 int8 = 4096 B → 256 cp.async.16-B issues
//          (2 issues per thread)
//   Y_d  : 128 floats from .ds (synchronous; one .h read per thread)
//
// We use one 4096-B-stage rotating buffer for W_qs and one 8192-B for Y_qs.
// Total shmem (W_qs * 2 + Y_qs * 2 + W_d * 2 + Y_d * 2) = ~25 KB.
//
// The MMA core is identical to v2's per-lane shmem reads. We could swap in
// ldmatrix.x4/x2 (v3 path) but v3 was a wash without cp.async; the win
// here is overlap, not fragment-load efficiency.

namespace {

// V7 (Phase B) constants left for reference; the kernel implementation below
// is the Phase C-1 rewrite that consumes the 144-B block_q8_1_mmq_t layout.
static constexpr int V7_STAGES = 2;

// ===========================================================================
// v7: cp.async pipelined MMQ on the 144-B block_q8_1_mmq_t activation layout.
//
// Same tile geometry as v2 (4 warps × 64×128, INT8 m16n8k32) but switches the
// activation source from 36-B block_q8_1 to 144-B block_q8_1_mmq_t. The key
// gmem-alignment win: block_q8_1_mmq_t.qs starts at +16 inside a 144-B block
// and packs 4 sub-blocks of 32 contiguously, so qs is 16-byte aligned by
// construction. That makes cp.async.cg.shared.global.16 legal on Y — closing
// the original v7 (Phase B)'s LSU bottleneck where Y had to issue 8 ×
// cp.async.4 per K block (block_q8_1.qs only 4-B aligned).
//
// Outer K-step is now V7M_K_STRIDE=128 (one super-block) instead of MMQ_QK=32.
// Per super-block we do 4 inner m16n8k32 MMAs, all consuming the same Y_qs
// load — 4× fewer outer iters and 4× MMA reuse per Y fetch.
//
// Per super-block, per CTA cp.async traffic:
//   W_qs : 64 rows × 128 int8 = 8192 B → 512 cp.async.16 issues
//          / 128 threads = 4 issues/thread
//   Y_qs : 128 rows × 128 int8 = 16384 B → 1024 cp.async.16 issues
//          / 128 threads = 8 issues/thread
//   W_d  : 64 rows × 4 fp16  = 512 B (synchronous)
//   Y_d  : 128 rows × 4 fp32 = 2048 B (synchronous)
// Total = 12 cp.async.16/thread per super-block, all 16-B aligned.
//
// vs Phase B:
//   W : 4 × 1 cp.async.16 = 4 issues/thread/super-block (same)
//   Y : 4 × 8 cp.async.4  = 32 issues/thread/super-block (4-B narrow)
// → 4× LSU pressure cut on Y and full 16-B issue width restored.
//
// Phase B's rotating 2-stage shmem buffer is preserved. Per-stage footprint:
//   W_qs  : 8 KB    W_d : 1 KB
//   Y_qs  : 16 KB   Y_d : 2 KB
//   Total per stage ≈ 27 KB, two stages ≈ 54 KB (still 1 block/SM at 100 KB cap).
// Recovering 2 blocks/SM is a separate lever (tile shrink to 64×64).

static constexpr int V7M_K_STRIDE = 128;        // 128 — qs per super-block per row (matches block_q8_1_mmq_t)
static constexpr int V7M_NSUB     = 4;         // 4 sub-blocks of 32 per super-block

template <bool need_check>
__launch_bounds__(V2_THREADS, 2)
__global__ void mmq_q8_0_v7_kernel(
        const uint8_t          * __restrict__ wq,
        const block_q8_1_mmq_t * __restrict__ ya,
        float                  * __restrict__ dst,
        uint32_t                  rows,
        uint32_t                  cols,
        uint32_t                  batch,
        uint32_t                  stride_y_sb,        // super-blocks per "row" — actually rows in batch
        uint32_t                  stride_dst_row)
{
    // 16-B-aligned shmem stages.
    __shared__ __align__(16) int8_t W_qs_buf[V7_STAGES][V7M_M_PER_CTA][V7M_K_STRIDE];
    __shared__ float                W_d_buf [V7_STAGES][V7M_M_PER_CTA][V7M_NSUB];
    __shared__ __align__(16) int8_t Y_qs_buf[V7_STAGES][V7M_N_PER_CTA][V7M_K_STRIDE];
    __shared__ float                Y_d_buf [V7_STAGES][V7M_N_PER_CTA][V7M_NSUB];

    const int n_blocks  = static_cast<int>(cols / MMQ_QK);  // 32-blocks per row of W
    const int n_sblocks = static_cast<int>(cols / V7M_K_STRIDE); // 128-super-blocks per row
    const int row0  = static_cast<int>(blockIdx.x) * V7M_M_PER_CTA;
    const int col00 = static_cast<int>(blockIdx.y) * V7M_N_PER_CTA;

    const int warp_id = static_cast<int>(threadIdx.y);
    const int lane    = static_cast<int>(threadIdx.x);
    const int tid     = warp_id * V2_WARP + lane;
    const int gid     = lane >> 2;
    const int tid4    = lane & 3;

    float acc[V7M_NW][4];
    #pragma unroll
    for (int n = 0; n < V7M_NW; ++n) {
        acc[n][0] = 0.0f; acc[n][1] = 0.0f;
        acc[n][2] = 0.0f; acc[n][3] = 0.0f;
    }

    // Issue cp.async loads for super-block `sb` into stage `s`. Caller commits.
    // Shmem layout is XOR-swizzled: chunk index c of row r lands at
    // shmem column position c ^ (r & 7). This breaks the 8-way bank conflict
    // that the row-stride-128B layout otherwise imposes (32 banks/row, so
    // 8 lanes at the same column across 8 rows all hit one bank). Within-chunk
    // bytes (low 4 bits) are preserved so cp.async.16 stays a contiguous 16-B
    // write to the swizzled position.
    auto issue_sb = [&](int sb, int s) {
        // ---- W_qs: 64 rows × 128 B per row = 8192 B = 512 cp.async.16 issues
        //     / 128 threads = 4 issues/thread. Layout same as v2's MMA reader.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int flat   = p * V2_THREADS + tid;   // 0..511
            const int rl     = flat >> 3;              // 0..63
            const int chunk  = flat & 7;
            const int row    = row0 + rl;
            // Swizzle destination chunk index by (rl & 7); source stays raw.
            const int col_byte_dst = (chunk ^ (rl & 7)) * 16;
            const int col_byte_src = chunk * 16;
            int8_t *dst_p = &W_qs_buf[s][rl][col_byte_dst];
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int8_t  *qs_sb    = cuda_helpers::q8_qs_plane(row_base, n_blocks)
                                          + sb * V7M_K_STRIDE;
                cp_async_cg_16(dst_p, qs_sb + col_byte_src);
            } else {
                int4 *zp = reinterpret_cast<int4 *>(dst_p);
                *zp = make_int4(0, 0, 0, 0);
            }
        }
        // ---- Y_qs: 64 rows × 128 B per row = 8192 B = 512 cp.async.16
        //     / 128 threads = 4 issues/thread.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int flat   = p * V2_THREADS + tid;   // 0..511
            const int bl     = flat >> 3;              // 0..63
            const int chunk  = flat & 7;
            const int batch_idx = col00 + bl;
            const int col_byte_dst = (chunk ^ (bl & 7)) * 16;
            const int col_byte_src = chunk * 16;
            int8_t *dst_p = &Y_qs_buf[s][bl][col_byte_dst];
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1_mmq_t *yblk =
                    ya + static_cast<size_t>(sb) * stride_y_sb + batch_idx;
                cp_async_cg_16(dst_p, yblk->qs + col_byte_src);
            } else {
                int4 *zp = reinterpret_cast<int4 *>(dst_p);
                *zp = make_int4(0, 0, 0, 0);
            }
        }
        // ---- W_d: 64 rows × 4 sub-d fp32 = 256 ; 2 issues/thread.
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int flat = p * V2_THREADS + tid;     // 0..255
            const int rl   = flat >> 2;                // 0..63
            const int sub  = flat & 3;
            const int row  = row0 + rl;
            float d = 0.0f;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int kb = sb * V7M_NSUB + sub;
                d = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            }
            W_d_buf[s][rl][sub] = d;
        }
        // ---- Y_d: 64 rows × 4 fp32 = 256 ; 2 issues/thread.
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int flat = p * V2_THREADS + tid;     // 0..255
            const int bl   = flat >> 2;                // 0..63
            const int sub  = flat & 3;
            const int batch_idx = col00 + bl;
            float d = 0.0f;
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1_mmq_t *yblk =
                    ya + static_cast<size_t>(sb) * stride_y_sb + batch_idx;
                d = yblk->d4[sub];
            }
            Y_d_buf[s][bl][sub] = d;
        }
    };

    // Prologue.
    if (n_sblocks > 0) {
        issue_sb(0, 0);
        cp_async_commit();
    }

    for (int sb = 0; sb < n_sblocks; ++sb) {
        const int s_cur  = sb & 1;
        const int s_next = (sb + 1) & 1;

        if (sb + 1 < n_sblocks) {
            issue_sb(sb + 1, s_next);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // ---------- 4 inner sub-blocks share the same shmem fill ----------
        const int rl_top = warp_id * V1_M_TILE + gid;
        const int rl_bot = rl_top + 8;
        // Both rl_top and rl_bot have (row & 7) == gid (warp_id*16 mod 8 == 0,
        // and the +8 leaves the low 3 bits unchanged). For Y_qs, N0+gid also
        // has (row & 7) == gid because N0 = n*8. So the swizzle XOR collapses
        // to a single `gid << 4` for every read in this loop.
        const int sw = gid << 4;

        #pragma unroll
        for (int sub = 0; sub < V7M_NSUB; ++sub) {
            const int kb_off = sub * MMQ_QK;   // byte offset within the 128-stride row
            // Swizzled byte offsets: low 4 bits (within-chunk position) preserved,
            // chunk index (bits 4..6) XORed with (row & 7) == gid.
            const int off_lo = (kb_off ^ sw) + 4 * tid4;
            const int off_hi = ((kb_off + 16) ^ sw) + 4 * tid4;

            // A fragment: same per-lane reads as v2/Phase B, just at offset kb_off.
            const unsigned a0 = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_top][off_lo]);
            const unsigned a1 = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_bot][off_lo]);
            const unsigned a2 = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_top][off_hi]);
            const unsigned a3 = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_bot][off_hi]);
            const float d_w_top = W_d_buf[s_cur][rl_top][sub];
            const float d_w_bot = W_d_buf[s_cur][rl_bot][sub];

            #pragma unroll
            for (int n = 0; n < V7M_NW; ++n) {
                const int N0 = n * 8;
                const unsigned b0 = *reinterpret_cast<unsigned *>(&Y_qs_buf[s_cur][N0 + gid][off_lo]);
                const unsigned b1 = *reinterpret_cast<unsigned *>(&Y_qs_buf[s_cur][N0 + gid][off_hi]);
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                mma_m16n8k32_s8s8s32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);

                const float d_y_my    = Y_d_buf[s_cur][N0 + gid][sub];
                const float d_y_left  = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4,     V2_WARP);
                const float d_y_right = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4 + 4, V2_WARP);

                acc[n][0] += d_w_top * d_y_left  * static_cast<float>(c0);
                acc[n][1] += d_w_top * d_y_right * static_cast<float>(c1);
                acc[n][2] += d_w_bot * d_y_left  * static_cast<float>(c2);
                acc[n][3] += d_w_bot * d_y_right * static_cast<float>(c3);
            }
        }

        __syncthreads();
    }

    // -------- Write back (identical to v2) --------
    const int row_top = row0 + warp_id * V1_M_TILE + gid;
    const int row_bot = row_top + 8;
    const bool top_in = need_check ? (row_top < static_cast<int>(rows)) : true;
    const bool bot_in = need_check ? (row_bot < static_cast<int>(rows)) : true;
    #pragma unroll
    for (int n = 0; n < V7M_NW; ++n) {
        const int col_left  = col00 + n * 8 + 2 * tid4;
        const int col_right = col_left + 1;
        const bool left_in  = need_check ? (col_left  < static_cast<int>(batch)) : true;
        const bool right_in = need_check ? (col_right < static_cast<int>(batch)) : true;
        if (left_in) {
            if (top_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_top] = acc[n][0];
            if (bot_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_bot] = acc[n][2];
        }
        if (right_in) {
            if (top_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_top] = acc[n][1];
            if (bot_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_bot] = acc[n][3];
        }
    }
}

// ===========================================================================
// v8: large-tile MMQ for chunk=2048 large-batch regime.
//
// Combines v7's plumbing (split-plane Q8 weights, 144-B block_q8_1_mmq_t
// activations, 2-stage cp.async pipeline, XOR-swizzled shmem, m16n8k32
// INT8 MMA core) with v4's tile size (8-warp, 128x128 output tile).
//
// Why v7 alone is 10% behind HGEMM at batch>=2048: the 64x64 tile reuses
// each K-axis read across only 64+64=128 lanes of work; cuBLAS HGEMM uses
// ~256x128 tiles so each K-load feeds 384 lanes (~3x arithmetic intensity).
// At large batch this becomes the ceiling.
//
// Why v4 alone is 56% behind HGEMM (measured): it has the right tile but
// none of v7's plumbing — sync K loads, no XOR swizzle, no ldmatrix.x4
// inner core, suboptimal A/B fragment cache. v8 grafts all of that on.
//
// Tile layout:
//   M_PER_CTA=128 (rows), N_PER_CTA=128 (cols), NWARPS=8 (256 threads)
//   warp(warp_id/2) -> 32-row stripe (4 stripes total: 0..3 -> rows 0,32,64,96)
//   warp(warp_id%2) -> 16-col band   (2 bands: 0..15 / 16..31, etc.)
// Per warp output sub-tile: 32 rows x 16 cols = 2 row-tiles x 2 col-tiles
// (tile_C = m16n8) with ne=4 outputs/lane each -> 16 floats/lane in sum[].
//
// Per K-iter (super-block = 128 K elements = 4 sub-blocks of 32):
//   W_qs: 128 rows * 128 B = 16384 B = 1024 cp.async.16 / 256 threads = 4/th
//   Y_qs: 128 rows * 128 B = 16384 B = 4 cp.async.16/thread
//   W_d : 128 rows * 4 fp32 = 2048 B (synchronous)
//   Y_d : 128 rows * 4 fp32 = 2048 B (synchronous)
// Total per stage shmem: 16+16+2+2 = 36 KB; 2 stages = 72 KB. 1 block/SM.
//
// XOR swizzle: identical to v7's `chunk ^ (row & 7)` (row stride = 128 B
// = 8 chunks of 16 B). Inner-loop reads collapse to a single `gid << 4`
// shift (same lane-row mapping as v7).

static constexpr int V8_M_PER_CTA = 128;
static constexpr int V8_N_PER_CTA = 128;
static constexpr int V8_NWARPS    = 8;
static constexpr int V8_THREADS   = V8_NWARPS * V2_WARP;  // 256
static constexpr int V8_K_STRIDE  = 128;   // qs per super-block per row
static constexpr int V8_NSUB      = 4;     // sub-blocks of 32 per super-block
static constexpr int V8_STAGES    = 2;
static constexpr int V8_NTX       = 2;     // 2 row-tiles per warp (32 rows)
static constexpr int V8_NTJ       = 8;     // 8 col-tiles per warp (64 cols)

template <bool need_check>
__launch_bounds__(V8_THREADS, 1)
__global__ void mmq_q8_0_v8_kernel(
        const uint8_t          * __restrict__ wq,
        const block_q8_1_mmq_t * __restrict__ ya,
        float                  * __restrict__ dst,
        uint32_t                  rows,
        uint32_t                  cols,
        uint32_t                  batch,
        uint32_t                  stride_y_sb,
        uint32_t                  stride_dst_row)
{
    extern __shared__ __align__(16) unsigned char v8_smem[];
    // Layout: W_qs_buf [2][128][128] | Y_qs_buf [2][128][128] | W_d_buf [2][128][4] | Y_d_buf [2][128][4]
    constexpr int W_qs_bytes = V8_STAGES * V8_M_PER_CTA * V8_K_STRIDE;            // 32768
    constexpr int Y_qs_bytes = V8_STAGES * V8_N_PER_CTA * V8_K_STRIDE;            // 32768
    constexpr int W_d_bytes  = V8_STAGES * V8_M_PER_CTA * V8_NSUB * 4;            // 4096
    auto W_qs_buf = reinterpret_cast<int8_t (*)[V8_M_PER_CTA][V8_K_STRIDE]>(v8_smem);
    auto Y_qs_buf = reinterpret_cast<int8_t (*)[V8_N_PER_CTA][V8_K_STRIDE]>(v8_smem + W_qs_bytes);
    auto W_d_buf  = reinterpret_cast<float  (*)[V8_M_PER_CTA][V8_NSUB]>(v8_smem + W_qs_bytes + Y_qs_bytes);
    auto Y_d_buf  = reinterpret_cast<float  (*)[V8_N_PER_CTA][V8_NSUB]>(v8_smem + W_qs_bytes + Y_qs_bytes + W_d_bytes);

    const int n_blocks  = static_cast<int>(cols / MMQ_QK);
    const int n_sblocks = static_cast<int>(cols / V8_K_STRIDE);
    const int row0  = static_cast<int>(blockIdx.x) * V8_M_PER_CTA;
    const int col00 = static_cast<int>(blockIdx.y) * V8_N_PER_CTA;

    const int warp_id  = static_cast<int>(threadIdx.y);
    const int lane     = static_cast<int>(threadIdx.x);
    const int tid      = warp_id * V2_WARP + lane;
    const int gid      = lane >> 2;
    const int tid4     = lane & 3;

    // Warp stripe (row) and band (col):
    //   stripe = warp_id / 2  -> 0..3, row range row0 + 32*stripe + [0..32)
    //   band   = warp_id % 2  -> 0..1, col range col00 + 64*band + [0..64)
    const int stripe = warp_id >> 1;
    const int band   = warp_id & 1;

    // Per-warp accumulators: NTX=2 row-tiles * NTJ=8 col-tiles * 4 outputs = 64 floats.
    float acc[V8_NTX][V8_NTJ][4];
    #pragma unroll
    for (int n = 0; n < V8_NTX; ++n) {
        #pragma unroll
        for (int t = 0; t < V8_NTJ; ++t) {
            acc[n][t][0] = 0.0f; acc[n][t][1] = 0.0f;
            acc[n][t][2] = 0.0f; acc[n][t][3] = 0.0f;
        }
    }

    // Issue cp.async loads for super-block `sb` into stage `s`.
    auto issue_sb = [&](int sb, int s) {
        // ---- W_qs: 128 rows x 128 B = 16384 B = 1024 cp.async.16 issues
        //     / 256 threads = 4 issues/thread. flat index = p*256 + tid.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int flat   = p * V8_THREADS + tid;   // 0..1023
            const int rl     = flat >> 3;              // 0..127 (row within tile)
            const int chunk  = flat & 7;               // 0..7   (16-B chunk in row)
            const int row    = row0 + rl;
            const int col_byte_dst = (chunk ^ (rl & 7)) * 16;
            const int col_byte_src = chunk * 16;
            int8_t *dst_p = &W_qs_buf[s][rl][col_byte_dst];
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int8_t  *qs_sb    = cuda_helpers::q8_qs_plane(row_base, n_blocks)
                                          + sb * V8_K_STRIDE;
                cp_async_cg_16(dst_p, qs_sb + col_byte_src);
            } else {
                int4 *zp = reinterpret_cast<int4 *>(dst_p);
                *zp = make_int4(0, 0, 0, 0);
            }
        }
        // ---- Y_qs: same shape (128 rows x 128 B), 4 issues/thread.
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int flat   = p * V8_THREADS + tid;
            const int bl     = flat >> 3;
            const int chunk  = flat & 7;
            const int batch_idx = col00 + bl;
            const int col_byte_dst = (chunk ^ (bl & 7)) * 16;
            const int col_byte_src = chunk * 16;
            int8_t *dst_p = &Y_qs_buf[s][bl][col_byte_dst];
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1_mmq_t *yblk =
                    ya + static_cast<size_t>(sb) * stride_y_sb + batch_idx;
                cp_async_cg_16(dst_p, yblk->qs + col_byte_src);
            } else {
                int4 *zp = reinterpret_cast<int4 *>(dst_p);
                *zp = make_int4(0, 0, 0, 0);
            }
        }
        // ---- W_d: 128 rows x 4 sub-d fp32 = 512 entries / 256 threads = 2/thread.
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int flat = p * V8_THREADS + tid;     // 0..511
            const int rl   = flat >> 2;                // 0..127
            const int sub  = flat & 3;
            const int row  = row0 + rl;
            float d = 0.0f;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *row_base = wq + static_cast<size_t>(row) * n_blocks * 34;
                const int kb = sb * V8_NSUB + sub;
                d = __half2float(static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kb]));
            }
            W_d_buf[s][rl][sub] = d;
        }
        // ---- Y_d: 128 rows x 4 fp32 = 512 entries; 2 issues/thread.
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int flat = p * V8_THREADS + tid;
            const int bl   = flat >> 2;
            const int sub  = flat & 3;
            const int batch_idx = col00 + bl;
            float d = 0.0f;
            const bool in_range = need_check ? (batch_idx < static_cast<int>(batch)) : true;
            if (in_range) {
                const block_q8_1_mmq_t *yblk =
                    ya + static_cast<size_t>(sb) * stride_y_sb + batch_idx;
                d = yblk->d4[sub];
            }
            Y_d_buf[s][bl][sub] = d;
        }
    };

    // Prologue.
    if (n_sblocks > 0) {
        issue_sb(0, 0);
        cp_async_commit();
    }

    for (int sb = 0; sb < n_sblocks; ++sb) {
        const int s_cur  = sb & 1;
        const int s_next = (sb + 1) & 1;

        if (sb + 1 < n_sblocks) {
            issue_sb(sb + 1, s_next);
            cp_async_commit();
            cp_async_wait_group<1>();
        } else {
            cp_async_wait_group<0>();
        }
        __syncthreads();

        // Inner MMA loop. For each of 4 sub-blocks (K=32 each):
        //   - Load A fragments for this warp's 2 row-tiles (rows = 32*stripe + 0..31)
        //   - Load B fragments for this warp's 2 col-tiles (cols = 16*band  + 0..15)
        //   - 4 m16n8k32 MMAs (NTX * NTJ)
        //   - Scale and accumulate into acc[NTX][NTJ][4]
        //
        // Lane-row mapping for tile_C m16n8 (NVIDIA-PTX layout):
        //   row_top = (i_tile_base) + lane/4         (i_tile_base + 0..7)
        //   row_bot = row_top + 8                     (i_tile_base + 8..15)
        // For row-tile n (n=0,1) within stripe `stripe`:
        //   i_tile_base = 32*stripe + 16*n
        // The shmem swizzle key is (row & 7) — for row_top = 32*stripe + 16*n + gid,
        // the low 3 bits are (gid) since 32 and 16 are multiples of 8. So sw = gid<<4
        // collapses the swizzle, identical to v7.
        //
        // For col-tile t (t=0,1) within band `band`:
        //   j_base_n = 16*band + 8*t   (col 8 lanes wide per col-tile)
        //   The B fragment row is `b_row = lane/4 = gid` (mapping into Y_qs row gid+j_base_n? no)
        // Actually for tile_B m8n8 read from Y_qs: rows-of-Y are batch entries,
        // cols-of-Y are K bytes. Lane mapping: each lane reads Y_qs[N0+gid][off+4*tid4]
        // to feed the B fragment, where N0 selects the col-tile's first batch row.
        // Same as v7, but here N0 = 16*band + 8*t for t=0..1.
        const int sw = gid << 4;    // swizzle shift for inner reads

        #pragma unroll
        for (int sub = 0; sub < V8_NSUB; ++sub) {
            const int kb_off = sub * MMQ_QK;
            const int off_lo = (kb_off ^ sw) + 4 * tid4;
            const int off_hi = ((kb_off + 16) ^ sw) + 4 * tid4;

            // Load A fragments for both row-tiles. 4 ints/lane per row-tile.
            unsigned A0[V8_NTX], A1[V8_NTX], A2[V8_NTX], A3[V8_NTX];
            float    d_w_top[V8_NTX], d_w_bot[V8_NTX];
            #pragma unroll
            for (int n = 0; n < V8_NTX; ++n) {
                const int rl_top = stripe * 32 + n * 16 + gid;
                const int rl_bot = rl_top + 8;
                A0[n] = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_top][off_lo]);
                A1[n] = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_bot][off_lo]);
                A2[n] = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_top][off_hi]);
                A3[n] = *reinterpret_cast<unsigned *>(&W_qs_buf[s_cur][rl_bot][off_hi]);
                d_w_top[n] = W_d_buf[s_cur][rl_top][sub];
                d_w_bot[n] = W_d_buf[s_cur][rl_bot][sub];
            }

            // Load B fragments and Y scales for both col-tiles. NTJ=8.
            #pragma unroll
            for (int t = 0; t < V8_NTJ; ++t) {
                const int N0 = band * 64 + t * 8;   // 0,8,...,56 within band 0; 64,...,120 within band 1
                const unsigned b0 = *reinterpret_cast<unsigned *>(&Y_qs_buf[s_cur][N0 + gid][off_lo]);
                const unsigned b1 = *reinterpret_cast<unsigned *>(&Y_qs_buf[s_cur][N0 + gid][off_hi]);

                const float d_y_my    = Y_d_buf[s_cur][N0 + gid][sub];
                const float d_y_left  = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4,     V2_WARP);
                const float d_y_right = __shfl_sync(0xffffffffu, d_y_my, 8 * tid4 + 4, V2_WARP);

                #pragma unroll
                for (int n = 0; n < V8_NTX; ++n) {
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    mma_m16n8k32_s8s8s32(c0, c1, c2, c3,
                                         A0[n], A1[n], A2[n], A3[n],
                                         b0, b1);
                    acc[n][t][0] += d_w_top[n] * d_y_left  * static_cast<float>(c0);
                    acc[n][t][1] += d_w_top[n] * d_y_right * static_cast<float>(c1);
                    acc[n][t][2] += d_w_bot[n] * d_y_left  * static_cast<float>(c2);
                    acc[n][t][3] += d_w_bot[n] * d_y_right * static_cast<float>(c3);
                }
            }
        }

        __syncthreads();
    }

    // Writeback. Output index mapping mirrors v7's per-row-tile pair.
    #pragma unroll
    for (int n = 0; n < V8_NTX; ++n) {
        const int row_top = row0 + stripe * 32 + n * 16 + gid;
        const int row_bot = row_top + 8;
        const bool top_in = need_check ? (row_top < static_cast<int>(rows)) : true;
        const bool bot_in = need_check ? (row_bot < static_cast<int>(rows)) : true;
        #pragma unroll
        for (int t = 0; t < V8_NTJ; ++t) {
            const int N0 = band * 64 + t * 8;
            const int col_left  = col00 + N0 + 2 * tid4;
            const int col_right = col_left + 1;
            const bool left_in  = need_check ? (col_left  < static_cast<int>(batch)) : true;
            const bool right_in = need_check ? (col_right < static_cast<int>(batch)) : true;
            if (left_in) {
                if (top_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_top] = acc[n][t][0];
                if (bot_in) dst[static_cast<size_t>(col_left)  * stride_dst_row + row_bot] = acc[n][t][2];
            }
            if (right_in) {
                if (top_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_top] = acc[n][t][1];
                if (bot_in) dst[static_cast<size_t>(col_right) * stride_dst_row + row_bot] = acc[n][t][3];
            }
        }
    }
}

} // namespace

bool launch_mmq_q8_0(
        const uint8_t * weight,
        const void *    y_q8_1,
        float *         dst,
        uint32_t        rows,
        uint32_t        cols,
        uint32_t        batch,
        uint32_t        stride_dst_row,
        cudaStream_t    stream) {
    if (cols % MMQ_QK != 0) return false;
    if (batch == 0 || rows == 0) return false;

    const uint32_t stride_y_row = cols / MMQ_QK;

    // Version select. v2 = 4-warp 64x128 with per-lane shmem reads (default).
    // v3 = same shape but ldmatrix.x4/x2 for fragment loads.
    // v4 = llama.cpp-style 8-warp 128x128 with cooperative tile_x/tile_y +
    //      ldmatrix MMA core. Caller is responsible for staging activations
    //      in the Q8_1_MMQ format (144 B/block, super-block-major × j-minor)
    //      when QW3_MMQ_VERSION=4 is selected. Cached on first call.
    //
    // Default is v7 (swizzled split-plane Q8 + cp.async + ldmatrix MMA core).
    // v7 strictly ≥ HGEMM at every T at chunk=512 after the XOR swizzle
    // landed (`feedback-mmq-v7-swizzle`); older v2-v6 are kept opt-in for
    // historical comparisons only.
    static const int mmq_version_env = []() {
        const char *e = std::getenv("QW3_MMQ_VERSION");
        if (!e) return 0;  // 0 = auto (per-call: v7 small, v8 large)
        if (e[0] == 'a') return 0;
        if (e[0] == '2') return 2;
        if (e[0] == '3') return 3;
        if (e[0] == '4') return 4;
        if (e[0] == '5') return 5;
        if (e[0] == '6') return 6;
        if (e[0] == '7') return 7;
        if (e[0] == '8') return 8;
        return 0;
    }();
    // Auto: v8 at batch >= V8_N_PER_CTA (and rows >= V8_M_PER_CTA),
    // else v7. v7's tail fallback handles tiny shapes already.
    const int mmq_version = (mmq_version_env != 0)
        ? mmq_version_env
        : ((batch >= V8_N_PER_CTA && rows >= V8_M_PER_CTA) ? 8 : 7);

    // v5 with smaller mmq_x — halves sum[] register pressure to eliminate
    // the 160-byte spill v4 had, and increases #CTAs for better SM
    // occupancy on Blackwell (more CTAs / less tail wave waste).
    if (mmq_version == 5 &&
        cols % MMQ_ITER_K == 0 &&
        rows >= MMQ_Y && batch >= 64) {
        // Choose mmq_x to match shape; 64 minimises spills and tail-wave
        // waste for typical Qwen prefill batches (≤256 → many CTAs,
        // ≥256 → fits multiple tiles into a wave).
        constexpr int V5_X = 64;
        const uint32_t row_tiles = (rows  + MMQ_Y - 1) / MMQ_Y;
        const uint32_t col_tiles = (batch + V5_X  - 1) / V5_X;
        const dim3 grid(row_tiles, col_tiles, 1);
        const dim3 block(V2_WARP, MMQ_NWARPS, 1);
        const bool needs_check = (rows  % MMQ_Y != 0) || (batch % V5_X != 0);
        const size_t shmem_x_bytes = static_cast<size_t>(MMQ_Y) * MMQ_MMA_TILE_X_K_Q8_0 * sizeof(int);
        const size_t shmem_y_bytes = static_cast<size_t>(V5_X)  * MMQ_TILE_Y_K          * sizeof(int);
        const size_t shmem_bytes   = shmem_x_bytes + shmem_y_bytes;
        cudaFuncSetAttribute(mmq_q8_0_v5_kernel<V5_X, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem_bytes));
        cudaFuncSetAttribute(mmq_q8_0_v5_kernel<V5_X, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem_bytes));
        const int *y_int = reinterpret_cast<const int *>(y_q8_1);
        if (needs_check) {
            mmq_q8_0_v5_kernel<V5_X, true><<<grid, block, shmem_bytes, stream>>>(
                weight, y_int, dst, rows, cols, batch, batch, stride_dst_row);
        } else {
            mmq_q8_0_v5_kernel<V5_X, false><<<grid, block, shmem_bytes, stream>>>(
                weight, y_int, dst, rows, cols, batch, batch, stride_dst_row);
        }
        return true;
    }

    // v4 path: 8-warp 128x128 tile, cols a multiple of 256 required.
    if (mmq_version == 4 &&
        cols % MMQ_ITER_K == 0 &&
        rows >= MMQ_Y && batch >= MMQ_X) {
        const uint32_t row_tiles = (rows  + MMQ_Y - 1) / MMQ_Y;
        const uint32_t col_tiles = (batch + MMQ_X - 1) / MMQ_X;
        const dim3 grid(row_tiles, col_tiles, 1);
        const dim3 block(V2_WARP, MMQ_NWARPS, 1);
        const bool needs_check = (rows  % MMQ_Y != 0) || (batch % MMQ_X != 0);
        const size_t shmem_x_bytes = static_cast<size_t>(MMQ_Y) * MMQ_MMA_TILE_X_K_Q8_0 * sizeof(int);
        const size_t shmem_y_bytes = static_cast<size_t>(MMQ_X) * MMQ_TILE_Y_K          * sizeof(int);
        const size_t shmem_bytes   = shmem_x_bytes + shmem_y_bytes;
        cudaFuncSetAttribute(mmq_q8_0_v4_kernel<MMQ_X, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem_bytes));
        cudaFuncSetAttribute(mmq_q8_0_v4_kernel<MMQ_X, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(shmem_bytes));
        const int *y_int = reinterpret_cast<const int *>(y_q8_1);
        if (needs_check) {
            mmq_q8_0_v4_kernel<MMQ_X, true><<<grid, block, shmem_bytes, stream>>>(
                weight, y_int, dst, rows, cols, batch, batch, stride_dst_row);
        } else {
            mmq_q8_0_v4_kernel<MMQ_X, false><<<grid, block, shmem_bytes, stream>>>(
                weight, y_int, dst, rows, cols, batch, batch, stride_dst_row);
        }
        return true;
    }

    // v8 path: 8-warp 128×128 tile with v7's plumbing (split-plane Q8 weight,
    // 144-B block_q8_1_mmq_t activations, 2-stage cp.async, XOR-swizzled
    // shmem, m16n8k32 INT8 MMA core). Targets the chunk=2048 large-batch
    // regime where v7's 64×64 tile leaves ~10% on the table vs HGEMM.
    if (mmq_version == 8) {
        if (cols % V8_K_STRIDE != 0) return false;
        // Tail-chunk fallback: v8 needs batch>=128 AND rows>=128. If either is
        // smaller, dispatch the v7 kernel — same 144-B activation layout, just a
        // smaller tile. Avoids silently routing to v2/v3 which expect a different
        // (36-int) layout.
        if (rows < V8_M_PER_CTA || batch < V8_N_PER_CTA) {
            if (rows < V7M_M_PER_CTA || batch < V7M_N_PER_CTA) return false;
            const uint32_t row_tiles = (rows  + V7M_M_PER_CTA - 1) / V7M_M_PER_CTA;
            const uint32_t col_tiles = (batch + V7M_N_PER_CTA - 1) / V7M_N_PER_CTA;
            const dim3 grid(row_tiles, col_tiles, 1);
            const dim3 block(V2_WARP, V2_NWARPS, 1);
            const bool needs_check = (rows  % V7M_M_PER_CTA != 0) ||
                                     (batch % V7M_N_PER_CTA != 0);
            if (needs_check) {
                mmq_q8_0_v7_kernel<true><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1_mmq_t *>(y_q8_1), dst,
                    rows, cols, batch, /*stride_y_sb=*/batch, stride_dst_row);
            } else {
                mmq_q8_0_v7_kernel<false><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1_mmq_t *>(y_q8_1), dst,
                    rows, cols, batch, /*stride_y_sb=*/batch, stride_dst_row);
            }
            return true;
        }
        const uint32_t row_tiles = (rows  + V8_M_PER_CTA - 1) / V8_M_PER_CTA;
        const uint32_t col_tiles = (batch + V8_N_PER_CTA - 1) / V8_N_PER_CTA;
        const dim3 grid(row_tiles, col_tiles, 1);
        const dim3 block(V2_WARP, V8_NWARPS, 1);
        const bool needs_check = (rows  % V8_M_PER_CTA != 0) ||
                                 (batch % V8_N_PER_CTA != 0);
        // Dynamic shmem: W_qs (32 KB) + Y_qs (32 KB) + W_d (4 KB) + Y_d (4 KB) = 72 KB.
        // Above the 48 KB default — opt in to higher carveout.
        constexpr int v8_shmem_bytes = 72 * 1024;
        cudaError_t a1 = cudaFuncSetAttribute(mmq_q8_0_v8_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v8_shmem_bytes);
        cudaError_t a2 = cudaFuncSetAttribute(mmq_q8_0_v8_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v8_shmem_bytes);
        if (a1 != cudaSuccess || a2 != cudaSuccess) {
            return false;
        }
        if (needs_check) {
            mmq_q8_0_v8_kernel<true><<<grid, block, v8_shmem_bytes, stream>>>(
                weight, reinterpret_cast<const block_q8_1_mmq_t *>(y_q8_1), dst,
                rows, cols, batch, /*stride_y_sb=*/batch, stride_dst_row);
        } else {
            mmq_q8_0_v8_kernel<false><<<grid, block, v8_shmem_bytes, stream>>>(
                weight, reinterpret_cast<const block_q8_1_mmq_t *>(y_q8_1), dst,
                rows, cols, batch, /*stride_y_sb=*/batch, stride_dst_row);
        }
        return true;
    }

    // v7 path: 4-warp 64×64 tile (vs v2's 64×128), 144-B block_q8_1_mmq_t
    // activations, two-stage cp.async pipeline. Tile shrink is what gives
    // 2 blocks/SM; v2/v3/v6 share the larger 64×128 tile below.
    if (mmq_version == 7 &&
        rows >= V7M_M_PER_CTA && batch >= V7M_N_PER_CTA) {
        if (cols % 128 != 0) return false;  // V7M_K_STRIDE
        const uint32_t row_tiles = (rows  + V7M_M_PER_CTA - 1) / V7M_M_PER_CTA;
        const uint32_t col_tiles = (batch + V7M_N_PER_CTA - 1) / V7M_N_PER_CTA;
        const dim3 grid(row_tiles, col_tiles, 1);
        const dim3 block(V2_WARP, V2_NWARPS, 1);
        const bool needs_check = (rows  % V7M_M_PER_CTA != 0) ||
                                 (batch % V7M_N_PER_CTA != 0);
        if (needs_check) {
            mmq_q8_0_v7_kernel<true><<<grid, block, 0, stream>>>(
                weight, reinterpret_cast<const block_q8_1_mmq_t *>(y_q8_1), dst,
                rows, cols, batch, /*stride_y_sb=*/batch, stride_dst_row);
        } else {
            mmq_q8_0_v7_kernel<false><<<grid, block, 0, stream>>>(
                weight, reinterpret_cast<const block_q8_1_mmq_t *>(y_q8_1), dst,
                rows, cols, batch, /*stride_y_sb=*/batch, stride_dst_row);
        }
        return true;
    }

    // v2/v3/v6 path: 4-warp 64x128 tile, kicks in whenever the problem is at least
    // one tile in each dimension. need_check handles the trailing tile.
    if (rows >= V2_M_PER_CTA && batch >= V2_N_PER_CTA) {
        const uint32_t row_tiles = (rows  + V2_M_PER_CTA - 1) / V2_M_PER_CTA;
        const uint32_t col_tiles = (batch + V2_N_PER_CTA - 1) / V2_N_PER_CTA;
        const dim3 grid(row_tiles, col_tiles, 1);
        const dim3 block(V2_WARP, V2_NWARPS, 1);
        const bool needs_check = (rows  % V2_M_PER_CTA != 0) ||
                                 (batch % V2_N_PER_CTA != 0);
        if (mmq_version == 6) {
            if (needs_check) {
                mmq_q8_0_v6_kernel<true><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                    rows, cols, batch, stride_y_row, stride_dst_row);
            } else {
                mmq_q8_0_v6_kernel<false><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                    rows, cols, batch, stride_y_row, stride_dst_row);
            }
        } else if (mmq_version == 3) {
            if (needs_check) {
                mmq_q8_0_v3_kernel<true><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                    rows, cols, batch, stride_y_row, stride_dst_row);
            } else {
                mmq_q8_0_v3_kernel<false><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                    rows, cols, batch, stride_y_row, stride_dst_row);
            }
        } else {
            if (needs_check) {
                mmq_q8_0_v2_kernel<true><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                    rows, cols, batch, stride_y_row, stride_dst_row);
            } else {
                mmq_q8_0_v2_kernel<false><<<grid, block, 0, stream>>>(
                    weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                    rows, cols, batch, stride_y_row, stride_dst_row);
            }
        }
        return true;
    }

    // v1 fallback for tiny shapes (rare in practice — every Qwen prefill
    // matmul has rows >= 3072 and batch usually >= 512).
    const uint32_t row_tiles = (rows  + V1_M_TILE     - 1) / V1_M_TILE;
    const uint32_t col_tiles = (batch + V1_N_PER_CTA  - 1) / V1_N_PER_CTA;
    const dim3 grid(row_tiles, col_tiles, 1);
    const dim3 block(V2_WARP, 1, 1);
    mmq_q8_0_v1_kernel<<<grid, block, 0, stream>>>(
        weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
        rows, cols, batch, stride_y_row, stride_dst_row);
    return true;
}

} // namespace ported
} // namespace qw3
