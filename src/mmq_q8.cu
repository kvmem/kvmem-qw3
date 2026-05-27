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

// PTX wrapper: m16n8k32.row.col.s32.s8.s8.s32.
__device__ __forceinline__ void mma_m16n8k32_s8s8s32(
        int &c0, int &c1, int &c2, int &c3,
        unsigned a0, unsigned a1, unsigned a2, unsigned a3,
        unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

// Load one int (4 int8s) from a 2-byte-aligned byte stream. Q8_0 weight
// blocks place qs at byte 2 (after the FP16 scale), so qs ints are only
// 2-byte aligned to the start of the block.
__device__ __forceinline__ unsigned load_int_align2(const uint8_t *p, int byte_off) {
    const uint16_t *p16 = reinterpret_cast<const uint16_t *>(p + byte_off);
    return (static_cast<unsigned>(p16[0]) <<  0) |
           (static_cast<unsigned>(p16[1]) << 16);
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
                const uint8_t *wblk = wq + (static_cast<size_t>(row) * n_blocks + kb) * 34;
                v = load_int_align2(wblk + 2, col_byte);
            }
            *reinterpret_cast<unsigned *>(&W_qs[rl][col_byte]) = v;
        }
        // W_d: 64 floats. First 64 threads each load one.
        if (tid < V2_M_PER_CTA) {
            const int row = row0 + tid;
            float d = 0.0f;
            const bool in_range = need_check ? (row < static_cast<int>(rows)) : true;
            if (in_range) {
                const uint8_t *wblk = wq + (static_cast<size_t>(row) * n_blocks + kb) * 34;
                d = __half2float(*reinterpret_cast<const half *>(wblk));
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

// v1 single-warp kernel (16x32 per CTA). Kept as a fallback for shapes that
// don't reach the v2 64x64 tile granularity. Identical to the previous-
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
            const uint8_t *wblk = wq + (static_cast<size_t>(row_top) * n_blocks + kb) * 34;
            d_w_top = __half2float(*reinterpret_cast<const half *>(wblk));
            a0 = load_int_align2(wblk + 2, 4 * tid4);
            a2 = load_int_align2(wblk + 2, 4 * tid4 + 16);
        }
        if (row_bot_in_range) {
            const uint8_t *wblk = wq + (static_cast<size_t>(row_bot) * n_blocks + kb) * 34;
            d_w_bot = __half2float(*reinterpret_cast<const half *>(wblk));
            a1 = load_int_align2(wblk + 2, 4 * tid4);
            a3 = load_int_align2(wblk + 2, 4 * tid4 + 16);
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

    // v2 path: 4-warp 64x64 tile, kicks in whenever the problem is at least
    // one tile in each dimension. need_check handles the trailing tile.
    if (rows >= V2_M_PER_CTA && batch >= V2_N_PER_CTA) {
        const uint32_t row_tiles = (rows  + V2_M_PER_CTA - 1) / V2_M_PER_CTA;
        const uint32_t col_tiles = (batch + V2_N_PER_CTA - 1) / V2_N_PER_CTA;
        const dim3 grid(row_tiles, col_tiles, 1);
        const dim3 block(V2_WARP, V2_NWARPS, 1);
        const bool needs_check = (rows  % V2_M_PER_CTA != 0) ||
                                 (batch % V2_N_PER_CTA != 0);
        if (needs_check) {
            mmq_q8_0_v2_kernel<true><<<grid, block, 0, stream>>>(
                weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                rows, cols, batch, stride_y_row, stride_dst_row);
        } else {
            mmq_q8_0_v2_kernel<false><<<grid, block, 0, stream>>>(
                weight, reinterpret_cast<const block_q8_1 *>(y_q8_1), dst,
                rows, cols, batch, stride_y_row, stride_dst_row);
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
