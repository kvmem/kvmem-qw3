// SPDX-License-Identifier: MIT
//
// Q8_0 weight × Q8_1 activation matvec/GEMV — DP4A based.
//
// Ported from llama.cpp:
//   ggml/src/ggml-cuda/quantize.cu          (quantize_q8_1 kernel)
//   ggml/src/ggml-cuda/mmvq.cu              (mul_mat_vec_q kernel skeleton)
//   ggml/src/ggml-cuda/vecdotq.cuh          (vec_dot_q8_0_q8_1)
// Upstream commit 57ebaf4edd99ea675f256ae2286cd99206dbfcd1 (MIT,
// see LICENSES/llama.cpp.txt).
//
// Adaptations from upstream:
//   - Q8_0 only. The upstream switch over GGML_TYPE_* is collapsed away.
//   - No MUL_MAT_ID, no fusion (gate/bias/GLU), no multi-channel/multi-sample
//     batching. qw3 only ever calls matvec on a 2D weight × (batch, cols)
//     activation, so the kernel takes plain (rows, cols, ncols_dst) shape.
//   - Uses qw3's raw 34-byte Q8_0 layout directly: per row, blocks of
//     (FP16 scale | 32 int8 qs). No block_q8_0 type wrapper.
//   - Q8_1 staging buffer is allocated and managed by the caller; this file
//     exposes two launchers (quantize + matvec) and a size helper. That lets
//     the executor hoist quantization across multiple matvecs that share the
//     same input (e.g. Q/K/V in attention, gate/up in FFN).
//   - Pinned to 4 warps × 32 lanes (Volta+ NVIDIA generic table). RDNA / GCN
//     branches dropped. Targets Blackwell / Ada / Ampere / Turing only.
//   - ncols_dst handles the small-batch decode case (typically 1, sometimes
//     up to 8 for short prefill). For large batches the caller switches to
//     the cuBLAS HGEMM path that already exists in kernels_cuda.cu.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#include "cuda_helpers.cuh"

namespace qw3 {
namespace ported {

// Block geometry — must match GGML's Q8_0 / Q8_1 conventions.
static constexpr int QK8_0          = 32;
static constexpr int QK8_1          = 32;
static constexpr int QI8_0          = 8;   // QK8_0 / 4 (ints per block)
static constexpr int QI8_1          = 8;   // QK8_1 / 4
static constexpr int VDR_Q8_0       = 2;   // ints loaded per dp4a iter

// Q8_1 block layout: (half d, half s) | int8 qs[32]. 36 bytes total.
struct alignas(4) block_q8_1 {
    half2  ds;        // .x = d (max/127), .y = sum of dequantized values * d
    int8_t qs[QK8_1];
};
static_assert(sizeof(block_q8_1) == 36, "unexpected block_q8_1 size");

// Q8_1_MMQ block layout: 4 FP32 d-scales | 128 int8 qs. 144 bytes total.
// Tile-friendly format used by the v4 MMQ kernel — matches llama.cpp's
// block_q8_1_mmq with the D4 ds_layout (Q8_0 weight uses D4: just d, no
// per-block sum because Q8_0 has no zero-point). Four 32-element subblocks
// share a header and live contiguously, which means the 128 int8 quants are
// 16-byte aligned and ldmatrix.x4-friendly without any per-row pad.
struct alignas(16) block_q8_1_mmq {
    float  d4[4];      // per-32-element d-scales (d = amax / 127)
    int8_t qs[128];    // 4 × 32 quants, contiguous
};
static_assert(sizeof(block_q8_1_mmq) == 144, "unexpected block_q8_1_mmq size");

// ---------------------------------------------------------------------------
// quantize_q8_1: float → block_q8_1.
//
//   Grid : (ceil(ne0/CUDA_QUANTIZE_BLOCK_SIZE), batch)
//   Block: (CUDA_QUANTIZE_BLOCK_SIZE)
//
//   ne0       : padded column count (multiple of QK8_1)
//   ne00      : real column count (zero-padded above this)
//   stride_y0 : float stride between consecutive batch rows of x (== cols)
//
// One thread handles one element. 32 threads (one warp slice) cooperate on a
// single 32-wide block to compute amax and sum via warp shuffles. Lane 0 of
// each block writes the (d, sum) header; every lane writes one int8.

static constexpr int CUDA_QUANTIZE_BLOCK_SIZE = 256;

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
__global__ void quantize_q8_1_kernel(
        const float * __restrict__ x,
        block_q8_1 *  __restrict__ y,
        uint32_t      ne0,           // padded cols
        uint32_t      ne00,          // real cols
        uint32_t      stride_x_row,  // float stride per batch row in x
        uint32_t      stride_y_row)  // block stride per batch row in y (== ne0/QK8_1)
{
    const uint32_t i0 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i0 >= ne0) return;

    const uint32_t b  = blockIdx.y;
    const uint32_t ib = b * stride_y_row + i0 / QK8_1;
    const uint32_t iqs = i0 % QK8_1;

    const float xi = (i0 < ne00) ? x[b * stride_x_row + i0] : 0.0f;
    float amax = fabsf(xi);
    float sum  = xi;
    amax = cuda_helpers::warp_reduce_max<QK8_1>(amax);
    sum  = cuda_helpers::warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? 0 : __float2int_rn(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs == 0) {
        y[ib].ds = make_half2(d, sum);
    }
}

// ---------------------------------------------------------------------------
// mul_mat_vec_q8_0: Q8_0 weight × Q8_1 activations → FP32 output.
//
// Grid : (ceil(rows/rows_per_block), ncols_dst_batched)  -- but we fold batch
//        into the kernel via NCOLS_DST template constant; ncols_dst is the
//        number of activation rows (== batch size).
// Actually we use the same shape as upstream:
//   Grid : (ceil(rows / rows_per_cuda_block))     (batch handled inline)
//   Block: (warp_size, NWARPS, 1)
//
// One row per CUDA block (rows_per_cuda_block == 1 for ncols_dst==1, == 2
// otherwise). NWARPS is 4 for ncols_dst <= 4, 2 for 5..8.
//
// vdr=2: each thread handles 2 ints (8 int8s = QK8_0/4 = QI8_0/2 chunks of
// the block) per dp4a call. With QI8_0 = 8 and vdr = 2, 4 lanes cover one
// 32-wide block per outer-loop iteration.

static constexpr int Q8_VEC_WARP_SIZE = 32;

template <int NCOLS_DST>
constexpr __host__ __device__ int q8_mmvq_nwarps() {
    return NCOLS_DST <= 4 ? 4 : (NCOLS_DST <= 8 ? 2 : 1);
}

template <int NCOLS_DST>
constexpr __host__ __device__ int q8_mmvq_rows_per_block() {
    // 1 row per CUDA block for all batch widths. The 2-rows-per-block variant
    // reuses each activation column across two weight rows, but for the narrow
    // MTP verify batch (NCOLS_DST 2..6) the extra register + shared-memory
    // pressure costs more occupancy than the reuse buys. Measured against
    // qw3_ly: 1 row/block is the faster default for verify batches.
    return 1;
}

// Load one int (4 int8s) from a byte stream that is only 2-byte aligned
// (qw3's weight blocks pack as half|int8[32], so qs is at byte offset 2).
__device__ __forceinline__ int q8_load_int_align2(const uint8_t * p, int i32) {
    const uint16_t * p16 = reinterpret_cast<const uint16_t *>(p);
    return (static_cast<int>(p16[2 * i32 + 0]) <<  0) |
           (static_cast<int>(p16[2 * i32 + 1]) << 16);
}

// vec_dot_q8_0_q8_1, inlined: 2 dp4a's per call. Returns d_w * d_x * sumi.
__device__ __forceinline__ float vec_dot_q8_0_q8_1(
        const uint8_t * __restrict__ wq_bytes,    // 32 int8 weight quants (2-byte aligned)
        float          d_w,
        const block_q8_1 * __restrict__ ya,
        int            iqs)                // 0 or 4 (ints into the qs)
{
    const float d_x = __low2float(ya->ds);
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i) {
        const int v = q8_load_int_align2(wq_bytes, iqs + i);
        const int u = reinterpret_cast<const int *>(ya->qs)[iqs + i];
        sumi = __dp4a(v, u, sumi);
    }
    return d_w * d_x * static_cast<float>(sumi);
}

template <int NCOLS_DST>
__launch_bounds__(q8_mmvq_nwarps<NCOLS_DST>() * Q8_VEC_WARP_SIZE, 1)
__global__ void mul_mat_vec_q8_0_kernel(
        const uint8_t *    __restrict__ vx,         // weight: rows × (cols/32)*34 bytes
        const block_q8_1 * __restrict__ vy,         // activation
        float *            __restrict__ dst,        // [ncols_dst, rows]
        uint32_t cols,                              // == ncols_x
        uint32_t rows,                              // == nrows_x  (used for bounds)
        uint32_t stride_y_row,                      // blocks per activation row
        uint32_t stride_dst_row)                    // floats per dst row (== rows)
{
    constexpr int NWARPS              = q8_mmvq_nwarps<NCOLS_DST>();
    constexpr int rows_per_cuda_block = q8_mmvq_rows_per_block<NCOLS_DST>();
    constexpr int warp_size           = Q8_VEC_WARP_SIZE;
    constexpr int blocks_per_iter     = VDR_Q8_0 * NWARPS * warp_size / QI8_0;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = cols / QK8_0;

    // Per-thread accumulator: NCOLS_DST batch rows × rows_per_cuda_block weight rows.
    float tmp[NCOLS_DST][rows_per_cuda_block] = {{0.0f}};

    // Each thread iterates over (kbx) outer block index. tid decides which
    // 8-int chunk of the block this thread processes (kqs).
    for (int kbx = tid / (QI8_0 / VDR_Q8_0); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = VDR_Q8_0 * (tid % (QI8_0 / VDR_Q8_0));    // 0,2,4,6 (in ints)

        // Weight row (i) is the OUTER loop so each Q8 weight block is fetched
        // ONCE and reused across all NCOLS_DST activation columns. Decode/verify
        // matvec is HBM-weight-bandwidth-bound, so amortizing the weight read
        // across columns makes verify(N) cost ~one weight read instead of N.
        // (Column-outer ordering re-read the weight N times — the whole reason
        // the MTP verify batch was as slow as N separate decodes.)
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            const uint64_t row_idx = static_cast<uint64_t>(row0 + i);
            if (rows_per_cuda_block > 1 && row_idx >= rows) break;
            const uint8_t * row_base = vx + row_idx * blocks_per_row_x * 34;
            const half      d_w_h = static_cast<half>(
                cuda_helpers::q8_d_plane(row_base)[kbx]);
            const float     d_w   = __half2float(d_w_h);
            const int8_t  * qs_blk =
                cuda_helpers::q8_qs_plane(row_base, blocks_per_row_x) + kbx * 32;
            const uint8_t * qs_u8 = reinterpret_cast<const uint8_t *>(qs_blk);

            #pragma unroll
            for (int j = 0; j < NCOLS_DST; ++j) {
                const block_q8_1 * ya = vy + j * stride_y_row + kbx;
                tmp[j][i] += vec_dot_q8_0_q8_1(qs_u8, d_w, ya, kqs);
            }
        }
    }

    __shared__ float sh[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][rows_per_cuda_block][warp_size];

    if (NWARPS > 1 && threadIdx.y > 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                sh[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    // Sum partial contributions from other warps then warp-reduce, then write.
    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            #pragma unroll
            for (int l = 0; l < NWARPS - 1; ++l) {
                tmp[j][i] += sh[l][j][i][threadIdx.x];
            }
            tmp[j][i] = cuda_helpers::warp_reduce_sum<warp_size>(tmp[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block) {
            const uint64_t row_idx = static_cast<uint64_t>(row0 + threadIdx.x);
            if (rows_per_cuda_block == 1 || row_idx < rows) {
                dst[j * stride_dst_row + row_idx] = tmp[j][threadIdx.x];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_mat_vec_q8_0_two_kernel: two Q8_0 weights × one shared Q8_1 activation.
// Used to fuse FFN gate+up at decode (same input, same shape, different
// weights). Hot kernel is bandwidth-bound on weight reads, so two-weight
// fusion does NOT halve runtime — both weights are still pulled from HBM.
// What it saves is (a) one full grid-launch of CTA setup, and (b) a second
// L2 read of the activation (~6 KB per CTA). Empirically that's worth a
// few percent on FFN-heavy decode paths.
//
// Shape constraint: both weights must have identical (rows, cols). Caller
// validates this before dispatching; otherwise fall back to two separate
// matvec calls.

template <int NCOLS_DST>
__launch_bounds__(q8_mmvq_nwarps<NCOLS_DST>() * Q8_VEC_WARP_SIZE, 1)
__global__ void mul_mat_vec_q8_0_two_kernel(
        const uint8_t *    __restrict__ vx0,
        const uint8_t *    __restrict__ vx1,
        const block_q8_1 * __restrict__ vy,
        float *            __restrict__ dst0,
        float *            __restrict__ dst1,
        uint32_t cols,
        uint32_t rows,
        uint32_t stride_y_row,
        uint32_t stride_dst_row)
{
    constexpr int NWARPS              = q8_mmvq_nwarps<NCOLS_DST>();
    constexpr int rows_per_cuda_block = q8_mmvq_rows_per_block<NCOLS_DST>();
    constexpr int warp_size           = Q8_VEC_WARP_SIZE;
    constexpr int blocks_per_iter     = VDR_Q8_0 * NWARPS * warp_size / QI8_0;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = cols / QK8_0;

    float tmp0[NCOLS_DST][rows_per_cuda_block] = {{0.0f}};
    float tmp1[NCOLS_DST][rows_per_cuda_block] = {{0.0f}};

    for (int kbx = tid / (QI8_0 / VDR_Q8_0); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = VDR_Q8_0 * (tid % (QI8_0 / VDR_Q8_0));

        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            const block_q8_1 * ya = vy + j * stride_y_row + kbx;
            const float d_x = __low2float(ya->ds);
            // Hoist the activation reads once — both weights consume them.
            int u[VDR_Q8_0];
            #pragma unroll
            for (int i = 0; i < VDR_Q8_0; ++i) {
                u[i] = reinterpret_cast<const int *>(ya->qs)[kqs + i];
            }

            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const uint64_t row_idx = static_cast<uint64_t>(row0 + i);
                if (rows_per_cuda_block > 1 && row_idx >= rows) break;
                const uint8_t * row_base0 = vx0 + row_idx * blocks_per_row_x * 34;
                const uint8_t * row_base1 = vx1 + row_idx * blocks_per_row_x * 34;
                const float d_w0 = __half2float(static_cast<half>(
                    cuda_helpers::q8_d_plane(row_base0)[kbx]));
                const float d_w1 = __half2float(static_cast<half>(
                    cuda_helpers::q8_d_plane(row_base1)[kbx]));
                const int8_t * qs0 =
                    cuda_helpers::q8_qs_plane(row_base0, blocks_per_row_x) + kbx * 32;
                const int8_t * qs1 =
                    cuda_helpers::q8_qs_plane(row_base1, blocks_per_row_x) + kbx * 32;

                int sumi0 = 0, sumi1 = 0;
                #pragma unroll
                for (int k = 0; k < VDR_Q8_0; ++k) {
                    const int v0 = q8_load_int_align2(reinterpret_cast<const uint8_t *>(qs0), kqs + k);
                    const int v1 = q8_load_int_align2(reinterpret_cast<const uint8_t *>(qs1), kqs + k);
                    sumi0 = __dp4a(v0, u[k], sumi0);
                    sumi1 = __dp4a(v1, u[k], sumi1);
                }
                tmp0[j][i] += d_w0 * d_x * static_cast<float>(sumi0);
                tmp1[j][i] += d_w1 * d_x * static_cast<float>(sumi1);
            }
        }
    }

    __shared__ float sh0[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][rows_per_cuda_block][warp_size];
    __shared__ float sh1[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][rows_per_cuda_block][warp_size];

    if (NWARPS > 1 && threadIdx.y > 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                sh0[threadIdx.y - 1][j][i][threadIdx.x] = tmp0[j][i];
                sh1[threadIdx.y - 1][j][i][threadIdx.x] = tmp1[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            #pragma unroll
            for (int l = 0; l < NWARPS - 1; ++l) {
                tmp0[j][i] += sh0[l][j][i][threadIdx.x];
                tmp1[j][i] += sh1[l][j][i][threadIdx.x];
            }
            tmp0[j][i] = cuda_helpers::warp_reduce_sum<warp_size>(tmp0[j][i]);
            tmp1[j][i] = cuda_helpers::warp_reduce_sum<warp_size>(tmp1[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block) {
            const uint64_t row_idx = static_cast<uint64_t>(row0 + threadIdx.x);
            if (rows_per_cuda_block == 1 || row_idx < rows) {
                dst0[j * stride_dst_row + row_idx] = tmp0[j][threadIdx.x];
                dst1[j * stride_dst_row + row_idx] = tmp1[j][threadIdx.x];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_mat_vec_q8_0_silu_mul_kernel: fused FFN gate+up matvec with silu(g)*u
// in a single launch, writing only the (n_ff,)-sized mid buffer.
//
// Same shape constraints as the two-weight kernel — both weights must have
// identical (rows, cols) (FFN gate and up are always shape-matched in
// SwiGLU). Saves vs the gate+up + silu_mul pipeline:
//   (a) one kernel launch per layer per decode token (~64/token)
//   (b) two FP32 round-trips on the n_ff-wide intermediates (gate+up writes
//       in the matvec, then both reads in silu_mul) — small at decode batch
//       but accumulates over 64 layers
//   (c) the per-element ffn_mid_ scratch reduces to ffn_gate_ only

template <int NCOLS_DST>
__launch_bounds__(q8_mmvq_nwarps<NCOLS_DST>() * Q8_VEC_WARP_SIZE, 1)
__global__ void mul_mat_vec_q8_0_silu_mul_kernel(
        const uint8_t *    __restrict__ vx_gate,
        const uint8_t *    __restrict__ vx_up,
        const block_q8_1 * __restrict__ vy,
        float *            __restrict__ dst,
        uint32_t cols,
        uint32_t rows,
        uint32_t stride_y_row,
        uint32_t stride_dst_row)
{
    constexpr int NWARPS              = q8_mmvq_nwarps<NCOLS_DST>();
    constexpr int rows_per_cuda_block = q8_mmvq_rows_per_block<NCOLS_DST>();
    constexpr int warp_size           = Q8_VEC_WARP_SIZE;
    constexpr int blocks_per_iter     = VDR_Q8_0 * NWARPS * warp_size / QI8_0;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = cols / QK8_0;

    float tmp_g[NCOLS_DST][rows_per_cuda_block] = {{0.0f}};
    float tmp_u[NCOLS_DST][rows_per_cuda_block] = {{0.0f}};

    for (int kbx = tid / (QI8_0 / VDR_Q8_0); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = VDR_Q8_0 * (tid % (QI8_0 / VDR_Q8_0));

        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            const block_q8_1 * ya = vy + j * stride_y_row + kbx;
            const float d_x = __low2float(ya->ds);
            int u[VDR_Q8_0];
            #pragma unroll
            for (int i = 0; i < VDR_Q8_0; ++i) {
                u[i] = reinterpret_cast<const int *>(ya->qs)[kqs + i];
            }

            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const uint64_t row_idx = static_cast<uint64_t>(row0 + i);
                if (rows_per_cuda_block > 1 && row_idx >= rows) break;
                const uint8_t * row_base_g = vx_gate + row_idx * blocks_per_row_x * 34;
                const uint8_t * row_base_u = vx_up   + row_idx * blocks_per_row_x * 34;
                const float d_w_g = __half2float(static_cast<half>(
                    cuda_helpers::q8_d_plane(row_base_g)[kbx]));
                const float d_w_u = __half2float(static_cast<half>(
                    cuda_helpers::q8_d_plane(row_base_u)[kbx]));
                const int8_t * qs_g =
                    cuda_helpers::q8_qs_plane(row_base_g, blocks_per_row_x) + kbx * 32;
                const int8_t * qs_u =
                    cuda_helpers::q8_qs_plane(row_base_u, blocks_per_row_x) + kbx * 32;

                int sumi_g = 0, sumi_u = 0;
                #pragma unroll
                for (int k = 0; k < VDR_Q8_0; ++k) {
                    const int v_g = q8_load_int_align2(reinterpret_cast<const uint8_t *>(qs_g), kqs + k);
                    const int v_u = q8_load_int_align2(reinterpret_cast<const uint8_t *>(qs_u), kqs + k);
                    sumi_g = __dp4a(v_g, u[k], sumi_g);
                    sumi_u = __dp4a(v_u, u[k], sumi_u);
                }
                tmp_g[j][i] += d_w_g * d_x * static_cast<float>(sumi_g);
                tmp_u[j][i] += d_w_u * d_x * static_cast<float>(sumi_u);
            }
        }
    }

    __shared__ float sh_g[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][rows_per_cuda_block][warp_size];
    __shared__ float sh_u[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][rows_per_cuda_block][warp_size];

    if (NWARPS > 1 && threadIdx.y > 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                sh_g[threadIdx.y - 1][j][i][threadIdx.x] = tmp_g[j][i];
                sh_u[threadIdx.y - 1][j][i][threadIdx.x] = tmp_u[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            #pragma unroll
            for (int l = 0; l < NWARPS - 1; ++l) {
                tmp_g[j][i] += sh_g[l][j][i][threadIdx.x];
                tmp_u[j][i] += sh_u[l][j][i][threadIdx.x];
            }
            tmp_g[j][i] = cuda_helpers::warp_reduce_sum<warp_size>(tmp_g[j][i]);
            tmp_u[j][i] = cuda_helpers::warp_reduce_sum<warp_size>(tmp_u[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block) {
            const uint64_t row_idx = static_cast<uint64_t>(row0 + threadIdx.x);
            if (rows_per_cuda_block == 1 || row_idx < rows) {
                const float g = tmp_g[j][threadIdx.x];
                const float u = tmp_u[j][threadIdx.x];
                // silu(g) * u = g * sigmoid(g) * u
                const float silu = g / (1.0f + __expf(-g));
                dst[j * stride_dst_row + row_idx] = silu * u;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_mat_vec_q8_0_add_kernel: fused matvec + residual add. Writes
// dst[row] = dst[row] + W*x. Used for attn_output and ffn_down where the
// matvec output is immediately added back to the residual stream.
// Eliminates the per-layer add_kernel launch and the round-trip on the
// attn_out / ffn_out intermediate buffers (saves 128 launches/token).

template <int NCOLS_DST>
__launch_bounds__(q8_mmvq_nwarps<NCOLS_DST>() * Q8_VEC_WARP_SIZE, 1)
__global__ void mul_mat_vec_q8_0_add_kernel(
        const uint8_t *    __restrict__ vx,
        const block_q8_1 * __restrict__ vy,
        float *            __restrict__ dst,
        uint32_t cols,
        uint32_t rows,
        uint32_t stride_y_row,
        uint32_t stride_dst_row)
{
    constexpr int NWARPS              = q8_mmvq_nwarps<NCOLS_DST>();
    constexpr int rows_per_cuda_block = q8_mmvq_rows_per_block<NCOLS_DST>();
    constexpr int warp_size           = Q8_VEC_WARP_SIZE;
    constexpr int blocks_per_iter     = VDR_Q8_0 * NWARPS * warp_size / QI8_0;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x = cols / QK8_0;

    float tmp[NCOLS_DST][rows_per_cuda_block] = {{0.0f}};

    for (int kbx = tid / (QI8_0 / VDR_Q8_0); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = VDR_Q8_0 * (tid % (QI8_0 / VDR_Q8_0));

        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            const block_q8_1 * ya = vy + j * stride_y_row + kbx;

            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const uint64_t row_idx = static_cast<uint64_t>(row0 + i);
                if (rows_per_cuda_block > 1 && row_idx >= rows) break;
                const uint8_t * row_base = vx + row_idx * blocks_per_row_x * 34;
                const half      d_w_h = static_cast<half>(
                    cuda_helpers::q8_d_plane(row_base)[kbx]);
                const float     d_w   = __half2float(d_w_h);
                const int8_t  * qs_blk =
                    cuda_helpers::q8_qs_plane(row_base, blocks_per_row_x) + kbx * 32;
                tmp[j][i] += vec_dot_q8_0_q8_1(
                    reinterpret_cast<const uint8_t *>(qs_blk), d_w, ya, kqs);
            }
        }
    }

    __shared__ float sh[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][rows_per_cuda_block][warp_size];

    if (NWARPS > 1 && threadIdx.y > 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                sh[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            #pragma unroll
            for (int l = 0; l < NWARPS - 1; ++l) {
                tmp[j][i] += sh[l][j][i][threadIdx.x];
            }
            tmp[j][i] = cuda_helpers::warp_reduce_sum<warp_size>(tmp[j][i]);
        }
        if (threadIdx.x < rows_per_cuda_block) {
            const uint64_t row_idx = static_cast<uint64_t>(row0 + threadIdx.x);
            if (rows_per_cuda_block == 1 || row_idx < rows) {
                // Read-modify-write: dst += W*x.
                dst[j * stride_dst_row + row_idx] += tmp[j][threadIdx.x];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public launchers.

// Number of bytes needed to stage `batch` activation rows of `cols` columns
// in Q8_1 form. cols must be a multiple of QK8_1.
size_t q8_1_scratch_bytes(uint32_t batch, uint32_t cols) {
    const uint32_t blocks_per_row = (cols + QK8_1 - 1) / QK8_1;
    return static_cast<size_t>(batch) * blocks_per_row * sizeof(block_q8_1);
}

// Quantize a float input of shape (batch, cols) into Q8_1 staging buffer.
//   x          : float input, batch*stride_x_row floats.
//   stride_x_row: floats per batch row in x.
//   y_q8_1     : staging buffer, q8_1_scratch_bytes(batch, cols) bytes.
//   batch, cols: shape. cols must be a multiple of QK8_1.
bool launch_quantize_q8_1(
        const float * x,
        void *        y_q8_1,
        uint32_t      batch,
        uint32_t      cols,
        uint32_t      stride_x_row,
        cudaStream_t  stream) {
    if (cols % QK8_1 != 0) return false;
    const uint32_t ne0          = cols;
    const uint32_t blocks_per_row = ne0 / QK8_1;
    const dim3 grid((ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, batch);
    const dim3 block(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1_kernel<<<grid, block, 0, stream>>>(
        x,
        reinterpret_cast<block_q8_1 *>(y_q8_1),
        ne0, cols, stride_x_row, blocks_per_row);
    return true;
}

// ---------------------------------------------------------------------------
// quantize_mmq_q8_1: float → block_q8_1_mmq (144-byte tile-friendly format
// used by the v4 MMQ kernel).
//
// The output is laid out as super-block-major × j-minor (matches
// llama.cpp's quantize_mmq_q8_1 with the D4 ds_layout):
//
//   y[ib]  with  ib = super_block * ncols_y + j
//
// where ncols_y is the activation row count (batch size) and super_block is
// the 128-K-element index along the cols axis. Inside each super-block,
// ncols_y consecutive 144-byte blocks live contiguously, which lets the v4
// MMQ kernel read mmq_x * 36 contiguous ints per K super-iter without
// strided gathers.
//
// Grid (matches llama.cpp's quantize_mmq_q8_1_cuda layout, channel-flat):
//   gridDim.x = ncols_y (batch rows; each block handles one j)
//   gridDim.y = ceil(cols / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ)) = ceil(cols/512)
//   blockDim  = (CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1) = 128 threads
//
// One thread handles 4 floats. 8-thread groups (32 / 4) cooperate on one
// 32-element subblock to compute amax via warp shuffles. Lane 0 of each
// group writes the d-scale; every thread writes 4 int8s as a packed char4.

static constexpr int CUDA_QUANTIZE_BLOCK_SIZE_MMQ = 128;

__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1)
__global__ void quantize_mmq_q8_1_kernel(
        const float *      __restrict__ x,
        block_q8_1_mmq *   __restrict__ y,
        uint32_t ne0,             // padded col count (multiple of 4*QK8_1)
        uint32_t ne00,            // real col count (zero-padded above this)
        uint32_t stride_x_row)    // float stride per batch row in x
{
    constexpr int vals_per_scale = QK8_1;     // 32, D4 layout
    constexpr int warp_size      = 32;

    const int64_t i0 =
        (int64_t)blockDim.x * blockIdx.y * 4 + (int64_t)threadIdx.x * 4;
    if (i0 >= ne0) return;

    const int     j     = blockIdx.x;          // batch row, 0..ncols_y-1
    const int     ncols_y = gridDim.x;
    const int64_t sb    = i0 / (4 * QK8_1);    // super-block index along cols
    const int64_t ib    = sb * ncols_y + j;
    const int     iqs   = i0 % (4 * QK8_1);    // 0..127, position within block

    const float4 * x4 = reinterpret_cast<const float4 *>(x);
    const float4 xi   = (i0 < ne00)
        ? x4[((int64_t)j * stride_x_row + i0) / 4]
        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Reduce amax across the 8 threads that share a 32-element subblock
    // (vals_per_scale/4 == 8, so xor offsets 4,2,1).
    #pragma unroll
    for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset, warp_size));
    }

    const float d_inv = (amax == 0.0f) ? 0.0f : (127.0f / amax);
    char4 q;
    q.x = (amax == 0.0f) ? 0 : static_cast<int8_t>(__float2int_rn(xi.x * d_inv));
    q.y = (amax == 0.0f) ? 0 : static_cast<int8_t>(__float2int_rn(xi.y * d_inv));
    q.z = (amax == 0.0f) ? 0 : static_cast<int8_t>(__float2int_rn(xi.z * d_inv));
    q.w = (amax == 0.0f) ? 0 : static_cast<int8_t>(__float2int_rn(xi.w * d_inv));

    // Pack 4 int8s as a single 32-bit store.
    char4 * yqs4 = reinterpret_cast<char4 *>(y[ib].qs);
    yqs4[iqs / 4] = q;

    // First lane of each 32-element subblock writes the d-scale.
    if (iqs % 32 == 0) {
        const float d = (amax == 0.0f) ? 0.0f : (1.0f / d_inv);
        y[ib].d4[iqs / 32] = d;
    }
}

// Number of bytes needed to stage `batch` activation rows of `cols` columns
// in Q8_1_MMQ form. cols must be a multiple of 4*QK8_1 (= 128).
size_t q8_1_mmq_scratch_bytes(uint32_t batch, uint32_t cols) {
    const uint32_t blocks_per_row = (cols + 4 * QK8_1 - 1) / (4 * QK8_1);
    return static_cast<size_t>(batch) * blocks_per_row * sizeof(block_q8_1_mmq);
}

// Quantize a (batch, cols) float input into Q8_1_MMQ staging.
//   x          : float input, batch * stride_x_row floats.
//   stride_x_row: floats per batch row in x.
//   y_q8_1_mmq : staging buffer, q8_1_mmq_scratch_bytes(batch, cols) bytes.
//   batch, cols: shape. cols must be a multiple of 4*QK8_1 (= 128).
bool launch_quantize_mmq_q8_1(
        const float * x,
        void *        y_q8_1_mmq,
        uint32_t      batch,
        uint32_t      cols,
        uint32_t      stride_x_row,
        cudaStream_t  stream) {
    if (cols % (4 * QK8_1) != 0) return false;
    if (cols % 4 != 0) return false;
    const uint32_t ne0 = cols;
    const uint32_t block_num_y =
        (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 grid(batch, block_num_y, 1);
    const dim3 block(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    quantize_mmq_q8_1_kernel<<<grid, block, 0, stream>>>(
        x,
        reinterpret_cast<block_q8_1_mmq *>(y_q8_1_mmq),
        ne0, cols, stride_x_row);
    return true;
}

// Run Q8_0 × Q8_1 matvec.
//   weight        : Q8_0 weight, rows × (cols/32) × 34 bytes, row-major.
//   y_q8_1        : pre-quantized activations from launch_quantize_q8_1.
//   dst           : float [batch, rows], stride_dst_row floats per batch row.
//   batch         : number of activation rows (== ncols_dst). 1..8 supported.
//   stride_dst_row: floats per dst batch row (typically rows).
bool launch_mmvq_q8_0(
        const uint8_t * weight,
        const void *    y_q8_1,
        float *         dst,
        uint32_t        rows,
        uint32_t        cols,
        uint32_t        batch,
        uint32_t        stride_dst_row,
        cudaStream_t    stream) {
    if (cols % QK8_0 != 0) return false;
    if (batch == 0 || batch > 8) return false;
    const uint32_t stride_y_row = cols / QK8_1;

    auto launch = [&](auto NCOLS_C) {
        constexpr int NCOLS_DST = decltype(NCOLS_C)::value;
        constexpr int NWARPS    = q8_mmvq_nwarps<NCOLS_DST>();
        constexpr int RPB       = q8_mmvq_rows_per_block<NCOLS_DST>();
        const uint32_t nblocks  = (rows + RPB - 1) / RPB;
        const dim3 grid(nblocks, 1, 1);
        const dim3 block(Q8_VEC_WARP_SIZE, NWARPS, 1);
        mul_mat_vec_q8_0_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
            weight,
            reinterpret_cast<const block_q8_1 *>(y_q8_1),
            dst,
            cols, rows, stride_y_row, stride_dst_row);
    };

    switch (batch) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: return false;
    }
    return true;
}

// Fused two-weight matvec: gate+up FFN at decode. Caller must guarantee
// w0 and w1 have identical (rows, cols).
bool launch_mmvq_q8_0_two(
        const uint8_t * weight0,
        const uint8_t * weight1,
        const void *    y_q8_1,
        float *         dst0,
        float *         dst1,
        uint32_t        rows,
        uint32_t        cols,
        uint32_t        batch,
        uint32_t        stride_dst_row,
        cudaStream_t    stream) {
    if (cols % QK8_0 != 0) return false;
    if (batch == 0 || batch > 8) return false;
    const uint32_t stride_y_row = cols / QK8_1;

    auto launch = [&](auto NCOLS_C) {
        constexpr int NCOLS_DST = decltype(NCOLS_C)::value;
        constexpr int NWARPS    = q8_mmvq_nwarps<NCOLS_DST>();
        constexpr int RPB       = q8_mmvq_rows_per_block<NCOLS_DST>();
        const uint32_t nblocks  = (rows + RPB - 1) / RPB;
        const dim3 grid(nblocks, 1, 1);
        const dim3 block(Q8_VEC_WARP_SIZE, NWARPS, 1);
        mul_mat_vec_q8_0_two_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
            weight0, weight1,
            reinterpret_cast<const block_q8_1 *>(y_q8_1),
            dst0, dst1,
            cols, rows, stride_y_row, stride_dst_row);
    };

    switch (batch) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: return false;
    }
    return true;
}

// Fused two-weight matvec + SwiGLU: writes silu(W_gate * x) * (W_up * x) to a
// single (rows,)-shaped output. Caller must guarantee w_gate and w_up have
// identical (rows, cols).
bool launch_mmvq_q8_0_silu_mul(
        const uint8_t * weight_gate,
        const uint8_t * weight_up,
        const void *    y_q8_1,
        float *         dst,
        uint32_t        rows,
        uint32_t        cols,
        uint32_t        batch,
        uint32_t        stride_dst_row,
        cudaStream_t    stream) {
    if (cols % QK8_0 != 0) return false;
    if (batch == 0 || batch > 8) return false;
    const uint32_t stride_y_row = cols / QK8_1;

    auto launch = [&](auto NCOLS_C) {
        constexpr int NCOLS_DST = decltype(NCOLS_C)::value;
        constexpr int NWARPS    = q8_mmvq_nwarps<NCOLS_DST>();
        constexpr int RPB       = q8_mmvq_rows_per_block<NCOLS_DST>();
        const uint32_t nblocks  = (rows + RPB - 1) / RPB;
        const dim3 grid(nblocks, 1, 1);
        const dim3 block(Q8_VEC_WARP_SIZE, NWARPS, 1);
        mul_mat_vec_q8_0_silu_mul_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
            weight_gate, weight_up,
            reinterpret_cast<const block_q8_1 *>(y_q8_1),
            dst,
            cols, rows, stride_y_row, stride_dst_row);
    };

    switch (batch) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: return false;
    }
    return true;
}

// Fused matvec + add (residual). Writes dst = dst + W*x. Used for attn_output
// and ffn_down where the result is immediately summed into the residual stream.
bool launch_mmvq_q8_0_add(
        const uint8_t * weight,
        const void *    y_q8_1,
        float *         dst,
        uint32_t        rows,
        uint32_t        cols,
        uint32_t        batch,
        uint32_t        stride_dst_row,
        cudaStream_t    stream) {
    if (cols % QK8_0 != 0) return false;
    if (batch == 0 || batch > 8) return false;
    const uint32_t stride_y_row = cols / QK8_1;

    auto launch = [&](auto NCOLS_C) {
        constexpr int NCOLS_DST = decltype(NCOLS_C)::value;
        constexpr int NWARPS    = q8_mmvq_nwarps<NCOLS_DST>();
        constexpr int RPB       = q8_mmvq_rows_per_block<NCOLS_DST>();
        const uint32_t nblocks  = (rows + RPB - 1) / RPB;
        const dim3 grid(nblocks, 1, 1);
        const dim3 block(Q8_VEC_WARP_SIZE, NWARPS, 1);
        mul_mat_vec_q8_0_add_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
            weight,
            reinterpret_cast<const block_q8_1 *>(y_q8_1),
            dst,
            cols, rows, stride_y_row, stride_dst_row);
    };

    switch (batch) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: return false;
    }
    return true;
}

// mul_mat_vec_q8_0_fanout4_batch_kernel: four Q8_0 weights × one shared Q8_1
// activation, in a single launch. The four weights' row-spaces are
// concatenated along blockIdx.x; each block resolves which weight it belongs
// to and computes its local row. NCOLS_DST = batch (verify rows). One row per
// block (no rows_per_cuda_block reuse — matches the 1-row default).
template <int NCOLS_DST>
__global__ void mul_mat_vec_q8_0_fanout4_batch_kernel(
        const uint8_t * __restrict__ w0,
        const uint8_t * __restrict__ w1,
        const uint8_t * __restrict__ w2,
        const uint8_t * __restrict__ w3,
        const block_q8_1 * __restrict__ vy,
        float * __restrict__ d0,
        float * __restrict__ d1,
        float * __restrict__ d2,
        float * __restrict__ d3,
        uint32_t rows0,
        uint32_t rows1,
        uint32_t rows2,
        uint32_t rows3,
        uint32_t cols,
        uint32_t stride_y_row,
        uint32_t stride_d0,
        uint32_t stride_d1,
        uint32_t stride_d2,
        uint32_t stride_d3) {
    constexpr int NWARPS          = q8_mmvq_nwarps<NCOLS_DST>();
    constexpr int warp_size       = Q8_VEC_WARP_SIZE;
    constexpr int blocks_per_iter = VDR_Q8_0 * NWARPS * warp_size / QI8_0;

    uint32_t row = blockIdx.x;
    const uint8_t *weight = w0;
    float *dst = d0;
    uint32_t rows = rows0;
    uint32_t stride_dst_row = stride_d0;
    uint32_t local_row = row;

    if (row >= rows0) {
        row -= rows0;
        weight = w1; dst = d1; rows = rows1; stride_dst_row = stride_d1; local_row = row;
        if (row >= rows1) {
            row -= rows1;
            weight = w2; dst = d2; rows = rows2; stride_dst_row = stride_d2; local_row = row;
            if (row >= rows2) {
                row -= rows2;
                weight = w3; dst = d3; rows = rows3; stride_dst_row = stride_d3; local_row = row;
                if (row >= rows3) return;
            }
        }
    }
    if (weight == nullptr || dst == nullptr || local_row >= rows) return;

    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int blocks_per_row_x = cols / QK8_0;

    float tmp[NCOLS_DST] = {0.0f};
    for (int kbx = tid / (QI8_0 / VDR_Q8_0); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kqs = VDR_Q8_0 * (tid % (QI8_0 / VDR_Q8_0));
        // qw3 weight layout is split-plane: all block scales first, then all
        // int8 quants (NOT the interleaved 34-byte blocks). Use the same plane
        // helpers as mul_mat_vec_q8_0_kernel.
        const uint8_t *row_base =
            weight + static_cast<uint64_t>(local_row) * blocks_per_row_x * 34;
        const float d_w = __half2float(
            static_cast<half>(cuda_helpers::q8_d_plane(row_base)[kbx]));
        const int8_t *qs_blk =
            cuda_helpers::q8_qs_plane(row_base, blocks_per_row_x) + kbx * 32;
        const uint8_t *qs_u8 = reinterpret_cast<const uint8_t *>(qs_blk);

        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            const block_q8_1 *ya = vy + j * stride_y_row + kbx;
            tmp[j] += vec_dot_q8_0_q8_1(qs_u8, d_w, ya, kqs);
        }
    }

    __shared__ float sh[NWARPS - 1 > 0 ? NWARPS - 1 : 1][NCOLS_DST][warp_size];
    if (NWARPS > 1 && threadIdx.y > 0) {
        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            sh[threadIdx.y - 1][j][threadIdx.x] = tmp[j];
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) return;

    #pragma unroll
    for (int j = 0; j < NCOLS_DST; ++j) {
        #pragma unroll
        for (int l = 0; l < NWARPS - 1; ++l) {
            tmp[j] += sh[l][j][threadIdx.x];
        }
        tmp[j] = cuda_helpers::warp_reduce_sum<warp_size>(tmp[j]);
        if (threadIdx.x == 0) {
            dst[static_cast<uint64_t>(j) * stride_dst_row + local_row] = tmp[j];
        }
    }
}

bool launch_mmvq_q8_0_fanout4_batch(
        const uint8_t *weight0, const uint8_t *weight1,
        const uint8_t *weight2, const uint8_t *weight3,
        const void *y_q8_1,
        float *dst0, float *dst1, float *dst2, float *dst3,
        uint32_t rows0, uint32_t rows1, uint32_t rows2, uint32_t rows3,
        uint32_t cols, uint32_t batch,
        uint32_t stride_dst0, uint32_t stride_dst1,
        uint32_t stride_dst2, uint32_t stride_dst3,
        cudaStream_t stream) {
    if (cols % QK8_0 != 0) return false;
    if (batch == 0 || batch > 8) return false;
    const uint64_t total_rows64 =
        static_cast<uint64_t>(rows0) + rows1 + rows2 + rows3;
    if (total_rows64 == 0 || total_rows64 > 0xffffffffull) return false;
    const uint32_t stride_y_row = cols / QK8_1;
    const dim3 grid(static_cast<uint32_t>(total_rows64), 1, 1);

    auto launch = [&](auto NCOLS_C) {
        constexpr int NCOLS_DST = decltype(NCOLS_C)::value;
        constexpr int NWARPS    = q8_mmvq_nwarps<NCOLS_DST>();
        const dim3 block(Q8_VEC_WARP_SIZE, NWARPS, 1);
        mul_mat_vec_q8_0_fanout4_batch_kernel<NCOLS_DST><<<grid, block, 0, stream>>>(
            weight0, weight1, weight2, weight3,
            reinterpret_cast<const block_q8_1 *>(y_q8_1),
            dst0, dst1, dst2, dst3,
            rows0, rows1, rows2, rows3,
            cols, stride_y_row,
            stride_dst0, stride_dst1, stride_dst2, stride_dst3);
    };

    switch (batch) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: return false;
    }
    return true;
}

} // namespace ported
} // namespace qw3


