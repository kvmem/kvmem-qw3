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
    return NCOLS_DST == 1 ? 1 : 2;
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

        #pragma unroll
        for (int j = 0; j < NCOLS_DST; ++j) {
            const block_q8_1 * ya = vy + j * stride_y_row + kbx;

            #pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                const uint64_t row_idx = static_cast<uint64_t>(row0 + i);
                if (rows_per_cuda_block > 1 && row_idx >= rows) break;
                // 34 bytes per Q8_0 block, kbx blocks in.
                const uint8_t * blk = vx + (row_idx * blocks_per_row_x + kbx) * 34;
                const half     d_w_h = *reinterpret_cast<const half *>(blk);
                const float    d_w   = __half2float(d_w_h);
                tmp[j][i] += vec_dot_q8_0_q8_1(blk + 2, d_w, ya, kqs);
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
                const uint8_t * blk0 = vx0 + (row_idx * blocks_per_row_x + kbx) * 34;
                const uint8_t * blk1 = vx1 + (row_idx * blocks_per_row_x + kbx) * 34;
                const float d_w0 = __half2float(*reinterpret_cast<const half *>(blk0));
                const float d_w1 = __half2float(*reinterpret_cast<const half *>(blk1));

                int sumi0 = 0, sumi1 = 0;
                #pragma unroll
                for (int k = 0; k < VDR_Q8_0; ++k) {
                    const int v0 = q8_load_int_align2(blk0 + 2, kqs + k);
                    const int v1 = q8_load_int_align2(blk1 + 2, kqs + k);
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

} // namespace ported
} // namespace qw3
