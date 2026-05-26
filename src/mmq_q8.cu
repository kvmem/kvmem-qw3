// SPDX-License-Identifier: MIT
//
// Q8_0 weight x Q8_1 activation matmul via INT8 MMA tensor cores.
//
// Goal: prefill matmul that uses the m16n8k32.s8.s8.s32 instruction directly
// instead of the cuBLAS HGEMM path that goes via FP16 tensor cores. Saves a
// Q8 -> FP16 weight dequant step (paid at upload time) and an FP32 -> FP16
// activation conversion on every call. llama.cpp's MMQ kernel (mmq.cuh)
// runs this way on Turing+ consumer GPUs and beats their HGEMM path on
// every shape we use.
//
// Design — v2, single-warp CTA, M=16 row stripe x NW=4 N-tiles (16x32 per CTA):
//
//   - One CTA = one warp (32 threads). Each CTA computes a (16 row x 32 batch)
//     output tile = 4 m16n8k32 MMA instances per K block (one per N-tile).
//   - The A fragment (16x32 weight) is loaded once per K block and reused
//     across all NW=4 MMA instances. Each B fragment (32x8 activation) is
//     loaded fresh per N-tile from the q8_1 scratch.
//   - Per K block, scaling is per-block per-row (d_w from the weight block)
//     and per-block per-batch-row (d_x from the q8_1 block). d_w broadcasts
//     across N for free (lives in registers); d_x is broadcast across lanes
//     via warp shuffle from the lane that loaded it.
//
// Trade-offs vs v1:
//   - 4x output per CTA -> 4x fewer CTAs. At rows=14336, batch=2048: 7168 CTAs
//     instead of 28672. Fits ~56 CTAs/SM on Blackwell (128 SMs).
//   - Weight reads still use gmem (no shmem tiling yet) but the L2 catches
//     reuse across the 4 batch tiles since they're scheduled close together.
//   - Activation B loads are 4 fresh gmem reads per K block, but with batch
//     coalescing across lanes the ld is one transaction per warp per B tile.
//
// Future iterations (not done):
//   - 4 warps per CTA -> 64x64 tile (matches llama.cpp's mmq.cuh stride).
//   - Cooperative shmem staging of W and X to share loads across warps.
//   - Larger NW (8 or 16) once the tile spans more of the warp's reg budget.
//
// Activation Q8_1 staging is produced by `quantize_q8_1` in mmvq_q8.cu and
// shared via the executor's q8_1_scratch buffer. Block layout is
// (batch, n_blocks_per_row) where each block is (half d, half s, int8 qs[32]).
//
// FP16 K/V analogues are NOT applicable here (matmul uses Q8_0 weight + Q8_1
// activation on the integer path).
//
// Compilation requirement: -arch=sm_75 or higher. m16n8k32.s8.s8.s32 was
// introduced on Ampere (sm_80); on sm_75 (Turing) only m16n8k16 with one
// less K factor is available. We target sm_75 for forward compat — the
// runtime-JIT'd Blackwell kernel uses Ampere+ MMA paths.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "cuda_helpers.cuh"

namespace qw3 {
namespace ported {

namespace {

// Block geometry — must match GGML's Q8_0 / Q8_1 conventions.
static constexpr int MMQ_QK     = 32;     // K elements per Q8_0 / Q8_1 block
static constexpr int MMQ_K_TILE = 32;     // K tile = one block
static constexpr int MMQ_M_TILE = 16;     // rows per CTA (m16)
static constexpr int MMQ_N_TILE = 8;      // batch cols per CTA (n8)
static constexpr int MMQ_WARP   = 32;

static constexpr int MMQ_NW     = 4;      // N-tiles per CTA (each = 8 batch cols)
static constexpr int MMQ_N_PER_CTA = MMQ_N_TILE * MMQ_NW;  // 32

// Q8_1 block layout (mirror of mmvq_q8.cu — kept in anonymous namespace to
// avoid ODR collision; identical wire layout).
struct alignas(4) block_q8_1 {
    half2  ds;        // .x = d (max/127), .y = sum of dequantized values * d
    int8_t qs[MMQ_QK];
};
static_assert(sizeof(block_q8_1) == 36, "unexpected block_q8_1 size");

// PTX wrapper: m16n8k32.row.col.s32.s8.s8.s32.
//   A is M x K = 16 x 32 int8  -> 4 .b32 / lane.
//   B is K x N = 32 x  8 int8  -> 2 .b32 / lane.
//   C is M x N = 16 x  8 int32 -> 4 .b32 / lane (accumulator).
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

// Load one int (4 int8s) from a 2-byte-aligned byte stream. The Q8_0 weight
// blocks place qs at byte offset 2 (after the FP16 scale), so consecutive
// qs ints are only 2-byte aligned to the start of the block.
__device__ __forceinline__ unsigned load_int_align2(const uint8_t *p, int byte_off) {
    const uint16_t *p16 = reinterpret_cast<const uint16_t *>(p + byte_off);
    return (static_cast<unsigned>(p16[0]) <<  0) |
           (static_cast<unsigned>(p16[1]) << 16);
}

__launch_bounds__(MMQ_WARP, 4)
__global__ void mmq_q8_0_kernel(
        const uint8_t   * __restrict__ wq,        // Q8_0 weight bytes
        const block_q8_1 * __restrict__ ya,       // Q8_1 activations
        float           * __restrict__ dst,
        uint32_t          rows,
        uint32_t          cols,
        uint32_t          batch,
        uint32_t          stride_y_row,           // blocks per activation row
        uint32_t          stride_dst_row)         // floats per dst row (== rows)
{
    const int n_blocks = static_cast<int>(cols / MMQ_QK);
    const int row0  = static_cast<int>(blockIdx.x) * MMQ_M_TILE;
    const int col00 = static_cast<int>(blockIdx.y) * MMQ_N_PER_CTA;

    const int lane = static_cast<int>(threadIdx.x);
    const int gid  = lane >> 2;        // 0..7  (groupID)
    const int tid4 = lane & 3;         // 0..3  (threadID_in_group)

    // Per PTX m16n8k32 fragment layout:
    //   a0,a2 -> row = groupID         (= row_top in our addressing)
    //   a1,a3 -> row = groupID + 8     (= row_bot)
    const int row_top = row0 + gid;
    const int row_bot = row0 + gid + 8;
    const bool row_top_in_range = row_top < static_cast<int>(rows);
    const bool row_bot_in_range = row_bot < static_cast<int>(rows);

    // Per-thread float accumulators: 4 entries per N-tile, NW tiles.
    float acc0[MMQ_NW] = {0.0f};
    float acc1[MMQ_NW] = {0.0f};
    float acc2[MMQ_NW] = {0.0f};
    float acc3[MMQ_NW] = {0.0f};

    for (int kb = 0; kb < n_blocks; ++kb) {
        // Load A fragment (16x32 weight) once per K block.
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

        // Loop over NW N-tiles, each 8 batch cols. B fragment per N-tile lives
        // in batch rows [col0..col0+8). Each lane covers one batch row (col0+gid)
        // and shares its d_x with peers via warp shuffle.
        #pragma unroll
        for (int nw = 0; nw < MMQ_NW; ++nw) {
            const int col0 = col00 + nw * MMQ_N_TILE;
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

            const float d_x_left  = __shfl_sync(0xffffffffu, d_x_my, 4 * (2 * tid4),     32);
            const float d_x_right = __shfl_sync(0xffffffffu, d_x_my, 4 * (2 * tid4) + 4, 32);

            acc0[nw] += d_w_top * d_x_left  * static_cast<float>(c0);
            acc1[nw] += d_w_top * d_x_right * static_cast<float>(c1);
            acc2[nw] += d_w_bot * d_x_left  * static_cast<float>(c2);
            acc3[nw] += d_w_bot * d_x_right * static_cast<float>(c3);
        }
    }

    // Write 4 floats per N-tile.
    #pragma unroll
    for (int nw = 0; nw < MMQ_NW; ++nw) {
        const int col0 = col00 + nw * MMQ_N_TILE;
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

// Public launcher. Q8_0 weight x pre-quantized Q8_1 activations -> FP32 dst.
//   weight        : Q8_0 weight, rows x (cols/32) x 34 bytes, row-major.
//   y_q8_1        : pre-quantized activations from launch_quantize_q8_1.
//   dst           : float [batch, rows], stride_dst_row floats per batch row.
//   rows, cols    : weight shape. cols must be a multiple of MMQ_QK (32).
//   batch         : number of activation rows (== ncols_dst).
//   stride_dst_row: floats per dst batch row (typically rows).
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
    const uint32_t row_tiles = (rows  + MMQ_M_TILE     - 1) / MMQ_M_TILE;
    const uint32_t col_tiles = (batch + MMQ_N_PER_CTA  - 1) / MMQ_N_PER_CTA;

    const dim3 grid(row_tiles, col_tiles, 1);
    const dim3 block(MMQ_WARP, 1, 1);

    mmq_q8_0_kernel<<<grid, block, 0, stream>>>(
        weight,
        reinterpret_cast<const block_q8_1 *>(y_q8_1),
        dst,
        rows, cols, batch,
        stride_y_row, stride_dst_row);
    return true;
}

} // namespace ported
} // namespace qw3
