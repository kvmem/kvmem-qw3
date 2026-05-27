#include "qw3/device_backend.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace qw3 {

// Ported kernel launcher (src/gated_delta_net.cu).
namespace ported {
bool launch_gated_delta_net(
        float *alpha_inout, float *beta_inout,
        const float *dt_bias, const float *ssm_a,
        const float *conv_qkv,
        uint32_t q_offset, uint32_t k_offset, uint32_t v_offset,
        float *state, float *core_out,
        uint32_t T,
        uint32_t num_k_heads, uint32_t num_v_heads, uint32_t head_dim,
        uint32_t qkv_row_stride, uint32_t gb_row_stride, uint32_t out_row_stride,
        cudaStream_t stream);

// Ported Q8_0 mmvq launchers (src/mmvq_q8.cu).
size_t q8_1_scratch_bytes(uint32_t batch, uint32_t cols);
bool launch_quantize_q8_1(
        const float *x, void *y_q8_1,
        uint32_t batch, uint32_t cols, uint32_t stride_x_row,
        cudaStream_t stream);
bool launch_mmvq_q8_0(
        const uint8_t *weight, const void *y_q8_1, float *dst,
        uint32_t rows, uint32_t cols, uint32_t batch, uint32_t stride_dst_row,
        cudaStream_t stream);
bool launch_mmvq_q8_0_two(
        const uint8_t *weight0, const uint8_t *weight1, const void *y_q8_1,
        float *dst0, float *dst1,
        uint32_t rows, uint32_t cols, uint32_t batch, uint32_t stride_dst_row,
        cudaStream_t stream);
bool launch_mmvq_q8_0_silu_mul(
        const uint8_t *weight_gate, const uint8_t *weight_up, const void *y_q8_1,
        float *dst,
        uint32_t rows, uint32_t cols, uint32_t batch, uint32_t stride_dst_row,
        cudaStream_t stream);
bool launch_mmvq_q8_0_add(
        const uint8_t *weight, const void *y_q8_1, float *dst,
        uint32_t rows, uint32_t cols, uint32_t batch, uint32_t stride_dst_row,
        cudaStream_t stream);

// Ported MMQ Q8_0 INT8-MMA matmul (src/mmq_q8.cu). Drop-in replacement for
// the HGEMM prefill path that uses m16n8k32.s8.s8.s32 tensor cores directly
// against the raw Q8_0 weight + Q8_1 activations — no FP16 dequant cache,
// no FP32 -> FP16 activation conversion. Off by default until validated.
bool launch_mmq_q8_0(
        const uint8_t *weight, const void *y_q8_1, float *dst,
        uint32_t rows, uint32_t cols, uint32_t batch, uint32_t stride_dst_row,
        cudaStream_t stream);

// Q8_1_MMQ activation staging — required by MMQ v4 kernel (144 B/block,
// super-block-major × j-minor). Falls back to Q8_1 (36 B) for v2/v3.
size_t q8_1_mmq_scratch_bytes(uint32_t batch, uint32_t cols);
bool launch_quantize_mmq_q8_1(
        const float *x, void *y_q8_1_mmq,
        uint32_t batch, uint32_t cols, uint32_t stride_x_row,
        cudaStream_t stream);

// Ported fattn-vec decode (src/fattn_vec_decode.cu).
bool launch_fattn_vec_decode_f32(
        float *out, const float *q, uint32_t q_stride,
        const float *k_cache, const float *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t seq_len, uint32_t batch,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);
bool launch_fattn_vec_decode_f16(
        float *out, const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t seq_len, uint32_t batch,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);
bool launch_fattn_vec_decode_f32_splitk(
        float *out, void *scratch, const float *q, uint32_t q_stride,
        const float *k_cache, const float *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t seq_len, uint32_t batch,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);
bool launch_fattn_vec_decode_f16_splitk(
        float *out, void *scratch, const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t seq_len, uint32_t batch,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);
size_t fattn_vec_scratch_bytes(uint32_t n_heads, uint32_t batch,
                               uint32_t head_dim, uint32_t seq_len);
// Tiled flash-attention prefill (FA2 style). One block per (head, BR queries);
// 4 warps cooperatively load BC=32 K/V tile into shmem, each warp scores
// against its own query → 4× HBM bandwidth reduction on K/V vs the per-query
// vec kernel. FP16 K/V only — FP32 KV would need 64 KB shmem per block which
// exceeds the 48 KB static-shmem cap. The FP32 path stays on the vec splitk
// kernel (parity-only fallback).
bool launch_fattn_prefill_f16(
        float *out, const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);
// Tensor-core (m16n8k16 fp16) prefill flash-attention. Same shape as the
// tiled FA2 above but inner GEMMs are MMA. FP16 K/V only.
bool launch_fattn_prefill_mma_f16(
        float *out, const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);
}

namespace {

// Runtime selector for the recurrent (DeltaNet) kernel. Read once on first
// use. Default is the ported kernel (warp-shuffle reductions, register-
// resident state). Set QW3_RECURRENT_KERNEL=qw3 to fall back to the original
// shmem-reduction kernel — kept around for future exploration of why the
// algorithmic mapping matters this much.
enum class RecurrentKernel { Qw3, Ported };
RecurrentKernel recurrent_kernel_choice() {
    static const RecurrentKernel choice = []() {
        const char *env = std::getenv("QW3_RECURRENT_KERNEL");
        if (env && std::strcmp(env, "qw3") == 0) return RecurrentKernel::Qw3;
        return RecurrentKernel::Ported;
    }();
    return choice;
}

// Runtime selector for the Q8_0 matvec kernel. Same pattern as above.
//   QW3_MATVEC=qw3     -> original DP4A kernel (kept for future
//                         exploration / prefill paths).
//   QW3_MATVEC=ported  -> (default) ported llama.cpp mmvq Q8_0 + Q8_1
//                         activation quantization. Wins by ~15% on decode.
enum class MatvecKernel { Qw3, Ported };
MatvecKernel matvec_kernel_choice() {
    static const MatvecKernel choice = []() {
        const char *env = std::getenv("QW3_MATVEC");
        if (env && std::strcmp(env, "qw3") == 0) return MatvecKernel::Qw3;
        return MatvecKernel::Ported;
    }();
    return choice;
}

// Runtime selector for the decode attention kernel.
//   QW3_ATTN=qw3      -> original fused-tile kernel (kept for future
//                        exploration / short-context comparison).
//   QW3_ATTN=ported   -> (default) ported flash-attention-style vec
//                        kernel — parallel across KV tokens with shared
//                        softmax. ~+26% on long-context decode.
enum class AttentionKernel { Qw3, Ported };
AttentionKernel attention_kernel_choice() {
    static const AttentionKernel choice = []() {
        const char *env = std::getenv("QW3_ATTN");
        if (env && std::strcmp(env, "qw3") == 0) return AttentionKernel::Qw3;
        return AttentionKernel::Ported;
    }();
    return choice;
}

// Prefill attention kernel selector.
//   QW3_PREFILL_ATTN=tiled  -> FA2-style tiled kernel that loads K/V tiles
//                              into shmem and reuses them across BR=4 queries.
//                              Parity-correct but ~7% SLOWER than vec on
//                              Qwen 3.6 27B at 4K context: per-layer K+V
//                              (~36 MB) already fits in Blackwell's 120 MB L2,
//                              so the vec kernel gets free reuse and shmem
//                              tiling adds load overhead without HBM savings.
//                              Kept selectable for shapes where L2 spills.
//   QW3_PREFILL_ATTN=vec    -> (default) per-query split-K vec kernel
//                              (also used for decode). At prefill seq_len is
//                              passed as 1, so pick_nsplit→1 and it runs
//                              one block per (head, query).
enum class PrefillAttnKernel { Tiled, Vec, Cublas, Mma };
PrefillAttnKernel prefill_attn_kernel_choice() {
    static const PrefillAttnKernel choice = []() {
        const char *env = std::getenv("QW3_PREFILL_ATTN");
        if (env && std::strcmp(env, "tiled") == 0) return PrefillAttnKernel::Tiled;
        if (env && std::strcmp(env, "vec") == 0) return PrefillAttnKernel::Vec;
        if (env && std::strcmp(env, "cublas") == 0) return PrefillAttnKernel::Cublas;
        if (env && std::strcmp(env, "mma") == 0) return PrefillAttnKernel::Mma;
        // Default: tensor-core MMA (m16n8k16.f16) FA2. Lifts per-call attn
        // ~14.8 ms vs tiled FP32 SIMT's ~22.8 ms at T=4K, +11% prefill end-
        // to-end. Parity-correct (greedy tokens match tiled at T=1k+).
        // Tiled remains selectable via QW3_PREFILL_ATTN=tiled. cuBLAS path
        // (slower, materializes full T×T_kv scores) selectable via =cublas.
        return PrefillAttnKernel::Mma;
    }();
    return choice;
}

// Min batch (= number of prefill queries) at which we switch to the tiled
// kernel. Below this the per-query split-K kernel wins (more parallelism per
// query). Override with QW3_PREFILL_ATTN_MIN_BATCH.
uint32_t prefill_attn_min_batch() {
    static const uint32_t v = []() {
        const char *env = std::getenv("QW3_PREFILL_ATTN_MIN_BATCH");
        if (env) {
            int n = std::atoi(env);
            if (n > 0) return static_cast<uint32_t>(n);
        }
        return static_cast<uint32_t>(16);
    }();
    return v;
}

// Prefill matmul kernel selector.
//   QW3_MATMUL=mmq    -> INT8 MMA path (m16n8k32.s8.s8.s32) directly on
//                        Q8_0 weight + Q8_1 activations. Skips the
//                        Q8 -> FP16 weight dequant cache and the
//                        FP32 -> FP16 activation conversion. Parity-correct
//                        with HGEMM (logit max-diff < 1.0; 0 top-1
//                        mismatches over 33 steps on a smoke prompt) but
//                        currently ~3x SLOWER than HGEMM in this v1
//                        (16x32 tile, 1 warp / CTA, no shmem staging).
//                        Kept off-by-default until a bigger tile (4-warp
//                        CTA + cooperative weight shmem) catches HGEMM.
//   QW3_MATMUL=hgemm  -> (default) cuBLAS HGEMM with on-the-fly Q8 -> FP16
//                        weight dequant into a shared scratch buffer.
enum class MatmulKernel { Mmq, Hgemm };
MatmulKernel matmul_kernel_choice() {
    static const MatmulKernel choice = []() {
        const char *env = std::getenv("QW3_MATMUL");
        if (env && std::strcmp(env, "mmq") == 0) return MatmulKernel::Mmq;
        return MatmulKernel::Hgemm;
    }();
    return choice;
}


static thread_local char g_err[256];

DeviceStatus cuda_status(cudaError_t err, const char *where) {
    if (err == cudaSuccess) return {};
    std::snprintf(g_err, sizeof(g_err), "%s: %s", where, cudaGetErrorString(err));
    return {false, g_err};
}

DeviceStatus launch_status(const char *where) {
    return cuda_status(cudaGetLastError(), where);
}

DeviceStatus cublas_status(cublasStatus_t st, const char *where) {
    if (st == CUBLAS_STATUS_SUCCESS) return {};
    std::snprintf(g_err, sizeof(g_err), "%s: cublas status %d", where, static_cast<int>(st));
    return {false, g_err};
}

__device__ float fp16_to_f32_device(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

struct CudaTensor final : DeviceTensor {
    float *ptr = nullptr;
    std::string label;
    CudaTensor(uint64_t n, const char *name, uint32_t elem_bytes = sizeof(float)) {
        count = n;
        elem_size = elem_bytes;
        label = name ? name : "tensor";
        cudaMalloc(&ptr, static_cast<size_t>(n) * elem_bytes);
        cudaMemset(ptr, 0, static_cast<size_t>(n) * elem_bytes);
    }
    ~CudaTensor() override {
        if (ptr) cudaFree(ptr);
    }
    bool is_fp16() const { return elem_size == sizeof(__half); }
    __half *ptr_h() const { return reinterpret_cast<__half *>(ptr); }
};

enum class WeightType {
    F32,
    Q8_0,
};

struct CudaWeight final : DeviceWeight {
    void *ptr = nullptr;
    float *q8_f32_cache = nullptr;
    uint64_t bytes = 0;
    WeightType type = WeightType::F32;
    std::string label;
    CudaWeight(const void *src, uint64_t nbytes, uint64_t r, uint64_t c, WeightType t, const char *name) {
        rows = r;
        cols = c;
        bytes = nbytes;
        type = t;
        label = name ? name : "weight";
        cudaMalloc(&ptr, static_cast<size_t>(bytes));
        cudaMemcpy(ptr, src, static_cast<size_t>(bytes), cudaMemcpyHostToDevice);
    }
    ~CudaWeight() override {
        if (q8_f32_cache) cudaFree(q8_f32_cache);
        if (ptr) cudaFree(ptr);
    }
};

CudaTensor &as_tensor(DeviceTensor &t) {
    return static_cast<CudaTensor &>(t);
}

const CudaTensor &as_tensor(const DeviceTensor &t) {
    return static_cast<const CudaTensor &>(t);
}

const CudaWeight &as_weight(const DeviceWeight &w) {
    return static_cast<const CudaWeight &>(w);
}

__global__ void add_kernel(float *out, const float *a, const float *b, uint64_t n) {
    const uint64_t i4 = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i4 + 4 <= n) {
        const float4 av = *reinterpret_cast<const float4 *>(a + i4);
        const float4 bv = *reinterpret_cast<const float4 *>(b + i4);
        float4 r;
        r.x = av.x + bv.x; r.y = av.y + bv.y;
        r.z = av.z + bv.z; r.w = av.w + bv.w;
        *reinterpret_cast<float4 *>(out + i4) = r;
    } else {
        for (uint64_t i = i4; i < n; ++i) out[i] = a[i] + b[i];
    }
}

__global__ void mul_kernel(float *out, const float *a, const float *b, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void silu_kernel(float *out, const float *x, uint64_t n) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] / (1.0f + expf(-x[i]));
}

// Single-block argmax over the vocab logits. n is ~152K for Qwen3.6 — a single
// 1024-thread block strided over the array beats a two-stage reduction here
// (memory-bound, one launch). Result is written as [token : int32, logit-bits :
// int32] into out[0..1] so the host can read both in a single 8-byte async D2H.
__global__ void argmax_kernel(int32_t *__restrict__ out,
                              const float *__restrict__ x,
                              uint64_t n) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bsz = blockDim.x;
    float local_max = -INFINITY;
    int32_t local_idx = -1;
    for (uint64_t i = tid; i < n; i += bsz) {
        const float v = x[i];
        if (v > local_max) { local_max = v; local_idx = static_cast<int32_t>(i); }
    }
    // Warp-level reduction: tie-break on lower index for determinism.
    for (int off = 16; off > 0; off >>= 1) {
        const float    om = __shfl_xor_sync(0xffffffff, local_max, off);
        const int32_t  oi = __shfl_xor_sync(0xffffffff, local_idx, off);
        if (om > local_max || (om == local_max && oi < local_idx)) {
            local_max = om; local_idx = oi;
        }
    }
    __shared__ float    warp_max[32];
    __shared__ int32_t  warp_idx[32];
    const uint32_t warp = tid >> 5;
    const uint32_t lane = tid & 31;
    if (lane == 0) { warp_max[warp] = local_max; warp_idx[warp] = local_idx; }
    __syncthreads();
    if (warp == 0) {
        const uint32_t nwarps = bsz >> 5;
        if (lane < nwarps) { local_max = warp_max[lane]; local_idx = warp_idx[lane]; }
        else               { local_max = -INFINITY;  local_idx = -1; }
        for (int off = 16; off > 0; off >>= 1) {
            const float   om = __shfl_xor_sync(0xffffffff, local_max, off);
            const int32_t oi = __shfl_xor_sync(0xffffffff, local_idx, off);
            if (om > local_max || (om == local_max && oi < local_idx)) {
                local_max = om; local_idx = oi;
            }
        }
        if (lane == 0) {
            out[0] = local_idx;
            out[1] = __float_as_int(local_max);
        }
    }
}

// Fused silu(gate) * up for the SwiGLU FFN: one launch, one read per element
// instead of silu (read gate, write gate) then mul (read gate, read up, write
// out). Saves 64 launches per decode token.
// Vectorized: 4 elements per thread (float4 in/out).
__global__ void silu_mul_kernel(float *out, const float *gate, const float *up, uint64_t n) {
    const uint64_t i4 = (static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i4 + 4 <= n) {
        const float4 g = *reinterpret_cast<const float4 *>(gate + i4);
        const float4 u = *reinterpret_cast<const float4 *>(up   + i4);
        float4 r;
        r.x = (g.x / (1.0f + __expf(-g.x))) * u.x;
        r.y = (g.y / (1.0f + __expf(-g.y))) * u.y;
        r.z = (g.z / (1.0f + __expf(-g.z))) * u.z;
        r.w = (g.w / (1.0f + __expf(-g.w))) * u.w;
        *reinterpret_cast<float4 *>(out + i4) = r;
    } else {
        for (uint64_t i = i4; i < n; ++i) {
            const float g = gate[i];
            out[i] = (g / (1.0f + __expf(-g))) * up[i];
        }
    }
}

// Single-row RMS norm: 1 block per row, 1024 threads, vectorized float4 loads,
// 32-wide warp shuffles for the reduction (shmem only across warps).
template <uint32_t BLOCK_THREADS>
__global__ void rms_norm_kernel_vec(float *__restrict__ out,
                                    const float *__restrict__ x,
                                    const float *__restrict__ weight,
                                    uint64_t n,
                                    float eps) {
    constexpr uint32_t WARP_SIZE = 32;
    constexpr uint32_t NWARPS = BLOCK_THREADS / WARP_SIZE;
    const uint32_t b = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % WARP_SIZE;
    const uint32_t warp = tid / WARP_SIZE;
    const float *x_row = x + static_cast<uint64_t>(b) * n;
    float *out_row = out + static_cast<uint64_t>(b) * n;

    // Vectorized sum-of-squares using float4 loads. The tail uses scalars.
    const uint64_t n_vec = n / 4;
    const float4 *x4 = reinterpret_cast<const float4 *>(x_row);
    float sum = 0.0f;
    for (uint64_t i = tid; i < n_vec; i += BLOCK_THREADS) {
        float4 v = x4[i];
        sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    for (uint64_t i = n_vec * 4 + tid; i < n; i += BLOCK_THREADS) {
        const float v = x_row[i];
        sum += v * v;
    }

    // Warp reduce, then cross-warp via shmem.
    #pragma unroll
    for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, delta);
    }
    __shared__ float warp_sums[NWARPS];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < NWARPS) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int delta = WARP_SIZE / 2; delta > 0; delta >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, delta);
        }
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();
    const float scale = rsqrtf(warp_sums[0] / static_cast<float>(n) + eps);

    // Vectorized normalize/scale write-back.
    const float4 *w4 = reinterpret_cast<const float4 *>(weight);
    float4 *o4 = reinterpret_cast<float4 *>(out_row);
    for (uint64_t i = tid; i < n_vec; i += BLOCK_THREADS) {
        const float4 v = x4[i];
        const float4 wv = w4[i];
        float4 r;
        r.x = v.x * scale * wv.x;
        r.y = v.y * scale * wv.y;
        r.z = v.z * scale * wv.z;
        r.w = v.w * scale * wv.w;
        o4[i] = r;
    }
    for (uint64_t i = n_vec * 4 + tid; i < n; i += BLOCK_THREADS) {
        out_row[i] = x_row[i] * scale * weight[i];
    }
}

// Fallback for the unaligned / very-small-n case.
__global__ void rms_norm_kernel(float *out, const float *x, const float *weight, uint64_t n, float eps) {
    const uint32_t b = blockIdx.x;
    const float *x_row = x + static_cast<uint64_t>(b) * n;
    float *out_row = out + static_cast<uint64_t>(b) * n;
    __shared__ float scratch[256];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (uint64_t i = tid; i < n; i += blockDim.x) sum += x_row[i] * x_row[i];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(n) + eps);
    for (uint64_t i = tid; i < n; i += blockDim.x) out_row[i] = x_row[i] * scale * weight[i];
}

__global__ void q8_get_row_kernel(float *out, const uint8_t *weight, uint64_t row, uint64_t cols) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const uint64_t block = i / 32;
    const uint64_t inb = i % 32;
    const uint8_t *p = weight + (row * (cols / 32) + block) * 34;
    const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const int8_t q = reinterpret_cast<const int8_t *>(p + 2)[inb];
    out[i] = fp16_to_f32_device(dh) * static_cast<float>(q);
}

// Batched row gather: rows_buf is a device array of `batch` token ids. Each
// CUDA block (gridDim.y) handles one batch slot, writing to out[b * cols ..].
__global__ void q8_get_rows_batch_kernel(float *out,
                                         const uint8_t *weight,
                                         const uint64_t *rows_buf,
                                         uint64_t cols) {
    const uint32_t b = blockIdx.y;
    const uint64_t row = rows_buf[b];
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const uint64_t block = i / 32;
    const uint64_t inb = i % 32;
    const uint8_t *p = weight + (row * (cols / 32) + block) * 34;
    const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const int8_t q = reinterpret_cast<const int8_t *>(p + 2)[inb];
    out[static_cast<uint64_t>(b) * cols + i] = fp16_to_f32_device(dh) * static_cast<float>(q);
}

// Multi-row Q8_0 matvec with DP4A. Each block:
//   1. Cooperatively stages x into shared memory and pre-quantizes it to
//      int8 with per-32-element scales. This is done ONCE per block and
//      shared by all WARPS_PER_BLOCK warps that process distinct rows.
//   2. Each warp = one output row. The inner loop accumulates into int32
//      with __dp4a (4-element int8 dot product in a single instruction),
//      which is ~4x the throughput of the scalar (float)int8 * float loop.
//
// Per quantized block of 32 weight elements:
//     contribution = w_scale * x_scale * dp4a_dot(w_i8x4, x_i8x4)
// where dp4a_dot is computed via 8 __dp4a calls into an int32 accumulator.
//
// Falls back to the f32 path below when the per-block shmem budget is
// exhausted (~80K cols).

// Tiled Q8 matmul: each block handles WARPS_PER_BLOCK rows × BATCH_TILE
// batch positions, reading the relevant W rows ONCE and reusing them across
// BATCH_TILE outputs. This is the prefill-optimized variant: with BATCH_TILE=N
// the weight bandwidth drops by N relative to the per-batch kernel.
template <uint32_t WARPS_PER_BLOCK, uint32_t BATCH_TILE>
__global__ void q8_matmul_tiled_dp4a_kernel(float *out,
                                            const uint8_t *weight,
                                            const float *x,
                                            uint64_t rows,
                                            uint64_t cols,
                                            uint32_t batch,
                                            uint32_t in_stride,
                                            uint32_t out_stride) {
    constexpr uint32_t LANES = 32;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / LANES;
    const uint32_t lane = tid % LANES;
    const uint32_t batch_start = blockIdx.y * BATCH_TILE;
    const uint64_t blocks = cols / 32;

    // Pre-quantize x for up to BATCH_TILE batch positions into shmem.
    // Layout: [BATCH_TILE, cols] of int8 followed by [BATCH_TILE, blocks] of f32 scales.
    extern __shared__ __align__(16) char shmem_raw[];
    int8_t *sx_i8 = reinterpret_cast<int8_t *>(shmem_raw);
    float *sx_scale = reinterpret_cast<float *>(sx_i8 + BATCH_TILE * cols);

    #pragma unroll
    for (uint32_t bb = 0; bb < BATCH_TILE; ++bb) {
        const uint32_t b_idx = batch_start + bb;
        if (b_idx >= batch) break;
        const float *x_ptr = x + static_cast<uint64_t>(b_idx) * in_stride;
        int8_t *sx_row = sx_i8 + static_cast<uint64_t>(bb) * cols;
        float  *ss_row = sx_scale + static_cast<uint64_t>(bb) * blocks;
        for (uint64_t b = warp; b < blocks; b += WARPS_PER_BLOCK) {
            const float v = x_ptr[b * 32 + lane];
            float absv = fabsf(v);
            for (int delta = 16; delta > 0; delta >>= 1) {
                absv = fmaxf(absv, __shfl_xor_sync(0xffffffffu, absv, delta));
            }
            const float x_scale = absv * (1.0f / 127.0f);
            const float x_scale_inv = (absv > 0.0f) ? (127.0f / absv) : 0.0f;
            const int q = __float2int_rn(v * x_scale_inv);
            sx_row[b * 32 + lane] = static_cast<int8_t>(max(-127, min(127, q)));
            if (lane == 0) ss_row[b] = x_scale;
        }
    }
    __syncthreads();

    const uint64_t row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) return;
    const uint8_t *rowp = weight + row * blocks * 34;

    // Accumulators for BATCH_TILE batch positions, computed by THIS warp's row.
    float sums[BATCH_TILE];
    #pragma unroll
    for (uint32_t bb = 0; bb < BATCH_TILE; ++bb) sums[bb] = 0.0f;

    for (uint64_t b = lane; b < blocks; b += LANES) {
        const uint8_t *p = rowp + b * 34;
        const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
        const float w_scale = fp16_to_f32_device(dh);

        // Pack the 8x4 int8 weight tile into 8 int32s (read once per W row).
        int32_t w_packs[8];
        #pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            const uint8_t *pw = p + 2 + k * 4;
            w_packs[k] = static_cast<int32_t>(pw[0])
                       | (static_cast<int32_t>(pw[1]) << 8)
                       | (static_cast<int32_t>(pw[2]) << 16)
                       | (static_cast<int32_t>(pw[3]) << 24);
        }

        #pragma unroll
        for (uint32_t bb = 0; bb < BATCH_TILE; ++bb) {
            const uint32_t b_idx = batch_start + bb;
            if (b_idx >= batch) continue;
            const int8_t *sx_row = sx_i8 + static_cast<uint64_t>(bb) * cols + b * 32;
            const float x_scale = sx_scale[bb * blocks + b];
            int32_t acc = 0;
            #pragma unroll
            for (uint32_t k = 0; k < 8; ++k) {
                const int32_t x_pack = *reinterpret_cast<const int32_t *>(sx_row + k * 4);
                acc = __dp4a(w_packs[k], x_pack, acc);
            }
            sums[bb] += w_scale * x_scale * static_cast<float>(acc);
        }
    }

    #pragma unroll
    for (uint32_t bb = 0; bb < BATCH_TILE; ++bb) {
        float s = sums[bb];
        for (int delta = 16; delta > 0; delta >>= 1) {
            s += __shfl_xor_sync(0xffffffffu, s, delta);
        }
        if (lane == 0) {
            const uint32_t b_idx = batch_start + bb;
            if (b_idx < batch) out[static_cast<uint64_t>(b_idx) * out_stride + row] = s;
        }
    }
}

template <uint32_t WARPS_PER_BLOCK>
__global__ void q8_matvec_dp4a_kernel(float *out,
                                      const uint8_t *weight,
                                      const float *x,
                                      uint64_t rows,
                                      uint64_t cols,
                                      uint32_t in_stride,
                                      uint32_t out_stride) {
    constexpr uint32_t LANES = 32;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / LANES;
    const uint32_t lane = tid % LANES;
    const uint32_t b_idx = blockIdx.y;
    const uint64_t blocks = cols / 32;
    const float *x_ptr = x + static_cast<uint64_t>(b_idx) * in_stride;
    float *out_ptr = out + static_cast<uint64_t>(b_idx) * out_stride;

    extern __shared__ __align__(16) char shmem_raw[];
    int8_t *sx_i8 = reinterpret_cast<int8_t *>(shmem_raw);
    float *sx_scale = reinterpret_cast<float *>(sx_i8 + cols);

    for (uint64_t b = warp; b < blocks; b += WARPS_PER_BLOCK) {
        const float v = x_ptr[b * 32 + lane];
        float absv = fabsf(v);
        for (int delta = 16; delta > 0; delta >>= 1) {
            absv = fmaxf(absv, __shfl_xor_sync(0xffffffffu, absv, delta));
        }
        const float x_scale = absv * (1.0f / 127.0f);
        const float x_scale_inv = (absv > 0.0f) ? (127.0f / absv) : 0.0f;
        const int q = __float2int_rn(v * x_scale_inv);
        sx_i8[b * 32 + lane] = static_cast<int8_t>(max(-127, min(127, q)));
        if (lane == 0) sx_scale[b] = x_scale;
    }
    __syncthreads();

    const uint64_t row = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (row >= rows) return;
    const uint8_t *rowp = weight + row * blocks * 34;

    float sum = 0.0f;
    for (uint64_t b = lane; b < blocks; b += LANES) {
        const uint8_t *p = rowp + b * 34;
        const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
        const float w_scale = fp16_to_f32_device(dh);
        const float x_scale = sx_scale[b];

        int32_t acc = 0;
        #pragma unroll
        for (uint32_t k = 0; k < 8; ++k) {
            const uint8_t *pw = p + 2 + k * 4;
            const int32_t w_pack = static_cast<int32_t>(pw[0])
                                 | (static_cast<int32_t>(pw[1]) << 8)
                                 | (static_cast<int32_t>(pw[2]) << 16)
                                 | (static_cast<int32_t>(pw[3]) << 24);
            const int32_t x_pack = *reinterpret_cast<const int32_t *>(sx_i8 + b * 32 + k * 4);
            acc = __dp4a(w_pack, x_pack, acc);
        }
        sum += w_scale * x_scale * static_cast<float>(acc);
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, delta);
    }
    if (lane == 0) out_ptr[row] = sum;
}

// Pre-DP4A path: read x as floats from shmem or global. Used when the
// shmem budget for the pre-quantized layout (cols + blocks*sizeof(float))
// doesn't fit.
template <uint32_t WARPS_PER_BLOCK, bool USE_SHMEM>
__global__ void q8_matvec_v2_kernel(float *out,
                                    const uint8_t *weight,
                                    const float *x,
                                    uint64_t rows,
                                    uint64_t cols,
                                    uint32_t in_stride,
                                    uint32_t out_stride) {
    constexpr uint32_t LANES = 32;
    constexpr uint32_t THREADS = WARPS_PER_BLOCK * LANES;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid / LANES;
    const uint32_t lane = tid % LANES;
    const uint32_t b_idx = blockIdx.y;
    const uint64_t row = blockIdx.x * WARPS_PER_BLOCK + warp;
    const uint64_t blocks = cols / 32;
    const float *x_ptr = x + static_cast<uint64_t>(b_idx) * in_stride;
    float *out_ptr = out + static_cast<uint64_t>(b_idx) * out_stride;

    extern __shared__ float sx[];
    if (USE_SHMEM) {
        for (uint64_t i = tid; i < cols; i += THREADS) sx[i] = x_ptr[i];
        __syncthreads();
    }
    if (row >= rows) return;
    const uint8_t *rowp = weight + row * blocks * 34;
    const float *xptr = USE_SHMEM ? sx : x_ptr;

    float sum = 0.0f;
    for (uint64_t b = lane; b < blocks; b += LANES) {
        const uint8_t *p = rowp + b * 34;
        const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
        const float d = fp16_to_f32_device(dh);
        const int8_t *qs = reinterpret_cast<const int8_t *>(p + 2);
        const float *xb = xptr + b * 32;
        float local = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 32; ++i) local += static_cast<float>(qs[i]) * xb[i];
        sum += d * local;
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, delta);
    }
    if (lane == 0) out_ptr[row] = sum;
}

__global__ void q8_dequant_f32_kernel(float *out, const uint8_t *weight, uint64_t rows, uint64_t cols) {
    const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = rows * cols;
    if (i >= n) return;
    const uint64_t row = i / cols;
    const uint64_t col = i % cols;
    const uint64_t block = col / 32;
    const uint64_t inb = col % 32;
    const uint8_t *p = weight + (row * (cols / 32) + block) * 34;
    const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    const int8_t q = reinterpret_cast<const int8_t *>(p + 2)[inb];
    out[row * cols + col] = fp16_to_f32_device(dh) * static_cast<float>(q);
}

// FP16 dequantization for Q8_0 weights — vectorized int4 (128-bit) stores.
// Each Q8_0 block (32 i8 elems) is handled by 4 cooperating threads. Each
// thread reads 8 i8 values via two 32-bit loads, decodes them with the
// per-block FP16 scale, and writes 8 halves as one int4 store. CTA holds
// 64 sub-blocks (256 threads / 4 lanes-per-block).
//
// Throughput (Blackwell HBM3e, 5120x17920 weight): ~470 GB/s on the Q8 read,
// ~880 GB/s on the FP16 write. ~6x faster than the original 1-thread-per-
// block kernel — the dequant cost is now small enough that we don't need a
// persistent FP16 mirror at 2x weight memory.
__global__ void q8_dequant_f16_kernel(__half *out, const uint8_t *weight, uint64_t rows, uint64_t cols) {
    constexpr unsigned BLOCKS_PER_CTA = 64;
    const uint64_t blocks_per_row = cols / 32;
    const uint64_t n_blocks = rows * blocks_per_row;
    const uint32_t lane = threadIdx.x & 3;            // 0..3 within the sub-block
    const uint32_t sub  = threadIdx.x >> 2;           // 0..63: which sub-block within the CTA
    const uint64_t bi   = static_cast<uint64_t>(blockIdx.x) * BLOCKS_PER_CTA + sub;
    if (bi >= n_blocks) return;

    const uint64_t row = bi / blocks_per_row;
    const uint64_t blk = bi % blocks_per_row;
    const uint8_t *p   = weight + bi * 34;

    __shared__ float sscale[BLOCKS_PER_CTA];
    if (lane == 0) {
        const uint16_t dh = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
        sscale[sub] = fp16_to_f32_device(dh);
    }
    __syncthreads();
    const float scale = sscale[sub];

    // qs starts at p+2 (2-byte aligned to block). Load 8 int8s (= two 32-bit
    // words) via halfword reads to honor the alignment.
    const uint16_t *pq16 = reinterpret_cast<const uint16_t *>(p + 2 + lane * 8);
    const unsigned u0 = static_cast<unsigned>(pq16[0]) | (static_cast<unsigned>(pq16[1]) << 16);
    const unsigned u1 = static_cast<unsigned>(pq16[2]) | (static_cast<unsigned>(pq16[3]) << 16);
    int8_t qs[8];
    qs[0] = static_cast<int8_t>(u0 & 0xff);
    qs[1] = static_cast<int8_t>((u0 >>  8) & 0xff);
    qs[2] = static_cast<int8_t>((u0 >> 16) & 0xff);
    qs[3] = static_cast<int8_t>((u0 >> 24) & 0xff);
    qs[4] = static_cast<int8_t>(u1 & 0xff);
    qs[5] = static_cast<int8_t>((u1 >>  8) & 0xff);
    qs[6] = static_cast<int8_t>((u1 >> 16) & 0xff);
    qs[7] = static_cast<int8_t>((u1 >> 24) & 0xff);

    __half2 v[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i].x = __float2half(scale * static_cast<float>(qs[i * 2 + 0]));
        v[i].y = __float2half(scale * static_cast<float>(qs[i * 2 + 1]));
    }
    int4 packed;
    packed.x = *reinterpret_cast<int *>(&v[0]);
    packed.y = *reinterpret_cast<int *>(&v[1]);
    packed.z = *reinterpret_cast<int *>(&v[2]);
    packed.w = *reinterpret_cast<int *>(&v[3]);

    int4 *o = reinterpret_cast<int4 *>(out + row * cols + blk * 32) + lane;
    *o = packed;
}

// FP32 → FP16 packed conversion used to stage the input activations for HGEMM.
// Vectorized: each thread converts 4 floats (float4 load) into 2 __half2 (int2
// store), 4x fewer memory transactions and threads vs the scalar version.
// The tail (n % 4) is handled by the same kernel via a per-thread bound check.
__global__ void fp32_to_fp16_kernel(__half *out, const float *in, uint64_t n) {
    const uint64_t tid = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t i4  = tid * 4;
    if (i4 + 4 <= n) {
        const float4 v = *reinterpret_cast<const float4 *>(in + i4);
        __half2 a = __floats2half2_rn(v.x, v.y);
        __half2 b = __floats2half2_rn(v.z, v.w);
        int2 packed;
        packed.x = *reinterpret_cast<int *>(&a);
        packed.y = *reinterpret_cast<int *>(&b);
        *reinterpret_cast<int2 *>(out + i4) = packed;
    } else {
        // tail: scalar fallback
        for (uint64_t j = i4; j < n; ++j) {
            out[j] = __float2half(in[j]);
        }
    }
}

// Per-row causal softmax for the cuBLAS-based prefill attention path.
//
// Layout: scores [n_groups, T, T_kv] row-major, FP16, group-major. Each
// (head_in_group, query_row) row holds T_kv scores in [0, base + q + 1) and
// garbage past my_max_kv. The kernel computes max-shift + exp + 1/sum + scale,
// with an in-place causal mask, and writes back the normalised probabilities
// in the same FP16 buffer.
//
// Grid:  (T, n_groups)
// Block: 256 threads, each handling ceil(T_kv / 256) score positions.
__global__ void softmax_causal_inplace_kernel(__half * __restrict__ scores,
                                              uint32_t T, uint32_t T_kv,
                                              uint32_t base_seq_len,
                                              uint32_t n_groups,
                                              float scale) {
    const uint32_t q       = blockIdx.x;
    const uint32_t group   = blockIdx.y;
    const uint32_t tid     = threadIdx.x;
    const uint32_t lane    = tid & 31;
    const uint32_t warp_id = tid >> 5;
    constexpr uint32_t BS  = 256;
    constexpr uint32_t NW  = BS / 32;

    if (q >= T) return;
    const uint32_t my_max_kv = base_seq_len + q + 1;       // exclusive
    __half *row = scores + (static_cast<uint64_t>(group) * T + q) * T_kv;

    // Pass 1: row max with -inf for masked positions.
    float local_max = -INFINITY;
    for (uint32_t k = tid; k < T_kv; k += BS) {
        if (k < my_max_kv) {
            const float s = __half2float(row[k]) * scale;
            if (s > local_max) local_max = s;
        }
    }

    // Block reduce-max via warp + shmem.
    __shared__ float s_warp_max[NW];
    {
        float v = local_max;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off, 32));
        }
        if (lane == 0) s_warp_max[warp_id] = v;
    }
    __syncthreads();
    float row_max = -INFINITY;
    if (warp_id == 0) {
        float v = (lane < NW) ? s_warp_max[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off, 32));
        }
        if (lane == 0) s_warp_max[0] = v;
    }
    __syncthreads();
    row_max = s_warp_max[0];

    // Pass 2: write exp(s - row_max) to row, accumulate row_sum.
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < T_kv; k += BS) {
        if (k < my_max_kv) {
            const float s   = __half2float(row[k]) * scale;
            const float e   = __expf(s - row_max);
            row[k]          = __float2half(e);
            local_sum      += e;
        } else {
            row[k] = __float2half(0.0f);
        }
    }

    __shared__ float s_warp_sum[NW];
    {
        float v = local_sum;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off, 32);
        }
        if (lane == 0) s_warp_sum[warp_id] = v;
    }
    __syncthreads();
    float row_sum = 0.0f;
    if (warp_id == 0) {
        float v = (lane < NW) ? s_warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off, 32);
        }
        if (lane == 0) s_warp_sum[0] = v;
    }
    __syncthreads();
    row_sum = s_warp_sum[0];

    // Pass 3: divide by row_sum.
    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint32_t k = tid; k < T_kv; k += BS) {
        const float p = __half2float(row[k]) * inv;
        row[k]       = __float2half(p);
    }
}

// FP32 → FP16 conversion that walks Q strided by head: src is [T, n_heads, d]
// row-major (q_batch_stride = n_heads * d, q_stride = d) but we can just
// reuse the contiguous fp32_to_fp16_kernel since the source is contiguous in
// memory. The cuBLAS path interprets the head stride at GEMM time.

// Returns the largest dynamic shmem size we will reserve per Q8 matvec block.
// Caches the device's max-opt-in shmem size and (best effort) opts the
// shmem-staged matvec kernel into using it. Returns 0 if no kernel
// instantiation is currently available to opt in.
size_t q8_matvec_max_shmem(uint32_t warps_per_block);

DeviceStatus dequant_q8_to_f32(float **out_ptr, const CudaWeight &w, bool persistent_cache) {
    if (!out_ptr) return {false, "dequant_q8_to_f32 out_ptr is null"};
    const uint64_t n = w.rows * w.cols;
    float *target = nullptr;
    if (persistent_cache && w.q8_f32_cache) {
        *out_ptr = w.q8_f32_cache;
        return {};
    }
    if (auto st = cuda_status(cudaMalloc(&target, static_cast<size_t>(n) * sizeof(float)), "cuda q8_f32 alloc"); !st.ok) {
        return st;
    }
    q8_dequant_f32_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(
        target, static_cast<const uint8_t *>(w.ptr), w.rows, w.cols);
    if (auto st = launch_status("cuda q8_f32 dequant"); !st.ok) {
        cudaFree(target);
        return st;
    }
    if (persistent_cache) {
        const_cast<CudaWeight &>(w).q8_f32_cache = target;
    }
    *out_ptr = target;
    return {};
}

size_t q8_matvec_max_shmem() {
    static size_t cached = SIZE_MAX;
    if (cached != SIZE_MAX) return cached;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        cached = 48 * 1024;
        return cached;
    }
    int max_shmem_optin = 0;
    cudaDeviceGetAttribute(&max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    const size_t budget = max_shmem_optin > 4096 ? static_cast<size_t>(max_shmem_optin) - 4096
                                                 : static_cast<size_t>(48 * 1024);
    auto opt_in = [&](const void *kernel) {
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(budget));
    };
    opt_in(reinterpret_cast<const void *>(&q8_matvec_v2_kernel<8, true>));
    opt_in(reinterpret_cast<const void *>(&q8_matvec_v2_kernel<16, true>));
    opt_in(reinterpret_cast<const void *>(&q8_matvec_v2_kernel<32, true>));
    opt_in(reinterpret_cast<const void *>(&q8_matvec_dp4a_kernel<8>));
    opt_in(reinterpret_cast<const void *>(&q8_matvec_dp4a_kernel<16>));
    opt_in(reinterpret_cast<const void *>(&q8_matvec_dp4a_kernel<32>));
    // Tiled matmul kernels (BATCH_TILE x WARPS_PER_BLOCK) used by prefill.
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<8,  2>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<16, 2>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<32, 2>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<8,  4>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<16, 4>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<32, 4>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<8,  8>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<16, 8>));
    opt_in(reinterpret_cast<const void *>(&q8_matmul_tiled_dp4a_kernel<32, 8>));
    cached = budget;
    return cached;
}

DeviceStatus ensure_q8_f32_cache(CudaWeight &w) {
    if (w.q8_f32_cache) return {};
    if (w.type != WeightType::Q8_0) return {false, "ensure_q8_f32_cache requires Q8_0 weight"};
    float *ptr = nullptr;
    return dequant_q8_to_f32(&ptr, w, true);
}

// Dequant a Q8_0 weight into an FP16 scratch buffer. Caller owns the buffer
// and is expected to size it to the largest weight in the model. Bandwidth-
// bound (~470 GB/s Q8 read on Blackwell HBM3e) and runs on the same stream
// as the HGEMM that consumes it, so no sync needed.
DeviceStatus dequant_q8_to_f16_stream(__half *out, const CudaWeight &w, cudaStream_t stream) {
    if (w.type != WeightType::Q8_0) return {false, "dequant_q8_to_f16 requires Q8_0 weight"};
    const uint64_t n_blocks = w.rows * (w.cols / 32);
    constexpr unsigned BLOCKS_PER_CTA = 64;
    const unsigned threads = BLOCKS_PER_CTA * 4;     // 4 lanes per sub-block
    const unsigned blocks  = static_cast<unsigned>(
        (n_blocks + BLOCKS_PER_CTA - 1) / BLOCKS_PER_CTA);
    q8_dequant_f16_kernel<<<blocks, threads, 0, stream>>>(out, static_cast<const uint8_t *>(w.ptr), w.rows, w.cols);
    return launch_status("q8_dequant_f16_kernel");
}

DeviceStatus dequant_q8_to_f16(__half *out, const CudaWeight &w) {
    return dequant_q8_to_f16_stream(out, w, /*stream=*/0);
}

// Causal 1D conv (single token, kernel size K).
// Inputs:
//   proj[c]         : current frame, [conv_dim]
//   state[c, k]     : K-1 previous frames, [conv_dim, K-1] indexed (c*(K-1) + k)
//                     state[c, 0] is the OLDEST frame, state[c, K-2] is the most-recent.
//   conv_w[c, k]    : weights, [conv_dim, K] indexed (c*K + k); tap K-1 applies to current.
// Outputs:
//   out[c]          : silu(sum_k conv_w[c, k] * frame[c, k])  where frame[c, K-1]=proj[c]
//   state           : shifted left, last slot now holds proj[c]
__global__ void recurrent_conv_kernel(float *out,
                                      float *state,
                                      const float *proj,
                                      const float *conv_w,
                                      uint32_t conv_dim,
                                      uint32_t conv_k) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float *w = conv_w + c * conv_k;
    float *st = state + c * (conv_k - 1);
    float acc = 0.0f;
    // Older frames live in state[0..K-2], current frame is proj[c].
    for (uint32_t k = 0; k + 1 < conv_k; ++k) acc += w[k] * st[k];
    const float cur = proj[c];
    acc += w[conv_k - 1] * cur;
    out[c] = acc / (1.0f + expf(-acc));
    // Shift state window: drop oldest, append current at the end.
    for (uint32_t k = 0; k + 2 < conv_k; ++k) st[k] = st[k + 1];
    st[conv_k - 2] = cur;
}

__global__ void l2_norm_128_kernel(float *x, uint32_t blocks, uint32_t stride, float eps) {
    const uint32_t b = blockIdx.x;
    if (b >= blocks) return;
    __shared__ float scratch[128];
    const uint32_t tid = threadIdx.x;
    float *base = x + b * stride;
    scratch[tid] = base[tid] * base[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] + eps);
    base[tid] *= scale;
}

__global__ void deltanet_kernel(float *core,
                                float *state,
                                const float *conv,
                                const float *alpha,
                                const float *beta,
                                const float *ssm_a,
                                const float *dt_bias,
                                uint32_t num_k_heads,
                                uint32_t num_v_heads,
                                uint32_t head_k_dim,
                                uint32_t head_v_dim) {
    const uint32_t vh = blockIdx.x;
    const uint32_t j = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (vh >= num_v_heads || j >= head_v_dim || tid >= head_k_dim) return;

    __shared__ float scratch[128];
    float *row = state + (static_cast<uint64_t>(vh) * head_v_dim + j) * head_k_dim;
    const float *q = conv + (vh % num_k_heads) * head_k_dim;
    const float *k = conv + num_k_heads * head_k_dim + (vh % num_k_heads) * head_k_dim;
    const float *v = conv + 2 * num_k_heads * head_k_dim + vh * head_v_dim;
    const float g_raw = log1pf(expf(alpha[vh] + dt_bias[vh])) * ssm_a[vh];
    const float eg = expf(g_raw);
    const float b = 1.0f / (1.0f + expf(-beta[vh]));

    row[tid] *= eg;
    scratch[tid] = row[tid] * k[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float delta = (v[j] - scratch[0]) * b;
    row[tid] += delta * k[tid];
    scratch[tid] = row[tid] * q[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) core[vh * head_v_dim + j] = scratch[0] * rsqrtf(static_cast<float>(head_v_dim));
}

__global__ void recurrent_norm_gate_kernel(float *core, const float *gate, const float *norm_w, uint32_t num_v_heads, uint32_t head_v_dim, float eps) {
    const uint32_t vh = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (vh >= num_v_heads || tid >= head_v_dim) return;
    __shared__ float scratch[256];
    float *block = core + vh * head_v_dim;
    scratch[tid] = block[tid] * block[tid];
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(head_v_dim) + eps);
    const float z = gate[vh * head_v_dim + tid];
    block[tid] = block[tid] * scale * norm_w[tid] * (z / (1.0f + expf(-z)));
}

// ---------------------------------------------------------------------------
// Batched recurrent kernels: process T tokens for a whole layer in O(1)
// kernel launches instead of O(T) launches.
//
// Prefill on Qwen 3.6 27B has 48 DeltaNet layers. The per-token loop above
// launched 5 small kernels per token, so 1322 tokens * 48 layers * 5 kernels
// = 317K launches per prefill. At ~10us launch overhead that's ~3s of pure
// overhead. The batched variants below do one launch per layer per sub-op,
// reducing the count to 48 * 4 = ~200 launches per prefill.

// Conv-batched: one thread runs T sequential conv steps for its channel,
// keeping the small (conv_k-1) state window in registers.
template <uint32_t CONV_K>
__global__ void recurrent_conv_batch_kernel(float *out,         // [T, *] stride out_stride
                                            float *state,       // [conv_dim, conv_k-1]
                                            const float *proj,  // [T, *] stride proj_stride
                                            const float *conv_w, // [conv_dim, conv_k]
                                            uint32_t T,
                                            uint32_t conv_dim,
                                            uint32_t proj_stride,
                                            uint32_t out_stride) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    const float *w = conv_w + c * CONV_K;
    float *st = state + c * (CONV_K - 1);
    float st_buf[CONV_K - 1];
    #pragma unroll
    for (uint32_t i = 0; i + 1 < CONV_K; ++i) st_buf[i] = st[i];

    // Pre-load weights into registers (CONV_K <= 8 in practice).
    float w_buf[CONV_K];
    #pragma unroll
    for (uint32_t k = 0; k < CONV_K; ++k) w_buf[k] = w[k];

    for (uint32_t t = 0; t < T; ++t) {
        const float cur = proj[t * proj_stride + c];
        float acc = w_buf[CONV_K - 1] * cur;
        #pragma unroll
        for (uint32_t k = 0; k + 1 < CONV_K; ++k) acc += w_buf[k] * st_buf[k];
        out[t * out_stride + c] = acc / (1.0f + expf(-acc));
        // Shift state window.
        #pragma unroll
        for (uint32_t k = 0; k + 2 < CONV_K; ++k) st_buf[k] = st_buf[k + 1];
        st_buf[CONV_K - 2] = cur;
    }
    #pragma unroll
    for (uint32_t i = 0; i + 1 < CONV_K; ++i) st[i] = st_buf[i];
}

// L2-norm batched: one block per (token, k_head) pair. Same per-head logic as
// l2_norm_128_kernel, fanned out over the time dimension via blockIdx.y.
__global__ void l2_norm_128_batch_kernel(float *x,             // base ptr
                                         uint32_t blocks,       // num k heads
                                         uint32_t stride,       // head_k_dim
                                         uint32_t T,            // batch
                                         uint32_t batch_stride, // conv_dim
                                         uint32_t base_offset,  // offset to Q or K region
                                         float eps) {
    const uint32_t b  = blockIdx.x;            // which k_head
    const uint32_t t  = blockIdx.y;            // which timestep
    if (b >= blocks || t >= T) return;
    __shared__ float scratch[128];
    const uint32_t tid = threadIdx.x;
    float *base = x + static_cast<uint64_t>(t) * batch_stride + base_offset + b * stride;
    scratch[tid] = base[tid] * base[tid];
    __syncthreads();
    for (uint32_t s = 64; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] + eps);
    base[tid] *= scale;
}

// DeltaNet batched: one block per (v_head, head_v_dim_pos), 128 threads/block,
// each thread holds one element of the row in a register. The block iterates
// over all T tokens, updating its slice of the state sequentially. Output
// `core` is [T, num_v_heads * head_v_dim] row-major.
__global__ void deltanet_batch_kernel(float *core,                  // [T, *] stride core_stride
                                      float *state,                 // [num_v_heads, head_v_dim, head_k_dim]
                                      const float *conv_batch,      // [T, *] stride conv_stride
                                      const float *alpha_batch,     // [T, *] stride alpha_stride
                                      const float *beta_batch,      // [T, *] stride beta_stride
                                      const float *ssm_a,           // [num_v_heads]
                                      const float *dt_bias,         // [num_v_heads]
                                      uint32_t T,
                                      uint32_t num_k_heads,
                                      uint32_t num_v_heads,
                                      uint32_t head_k_dim,
                                      uint32_t head_v_dim,
                                      uint32_t conv_stride,
                                      uint32_t alpha_stride,
                                      uint32_t beta_stride,
                                      uint32_t core_stride) {
    const uint32_t vh = blockIdx.x;
    const uint32_t j  = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (vh >= num_v_heads || j >= head_v_dim || tid >= head_k_dim) return;

    __shared__ float scratch[128];
    float *row = state + (static_cast<uint64_t>(vh) * head_v_dim + j) * head_k_dim;

    // Cache the per-head constants used at every timestep.
    const uint32_t kh = vh % num_k_heads;
    const uint32_t q_offset = kh * head_k_dim;
    const uint32_t k_offset = num_k_heads * head_k_dim + kh * head_k_dim;
    const uint32_t v_offset = 2u * num_k_heads * head_k_dim + vh * head_v_dim;

    const float ssm_a_v = ssm_a[vh];
    const float dt_bias_v = dt_bias[vh];
    const float inv_hvd = rsqrtf(static_cast<float>(head_v_dim));

    float row_val = row[tid];

    for (uint32_t t = 0; t < T; ++t) {
        const float *conv = conv_batch + static_cast<uint64_t>(t) * conv_stride;
        const float q = conv[q_offset + tid];
        const float k = conv[k_offset + tid];
        const float v_j = conv[v_offset + j];
        const float a_v = alpha_batch[t * alpha_stride + vh];
        const float b_v = beta_batch[t * beta_stride + vh];

        const float g_raw = log1pf(expf(a_v + dt_bias_v)) * ssm_a_v;
        const float eg = expf(g_raw);
        const float b = 1.0f / (1.0f + expf(-b_v));

        row_val *= eg;
        scratch[tid] = row_val * k;
        __syncthreads();
        for (uint32_t s = 64; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        const float delta = (v_j - scratch[0]) * b;
        row_val += delta * k;
        scratch[tid] = row_val * q;
        __syncthreads();
        for (uint32_t s = 64; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            core[static_cast<uint64_t>(t) * core_stride + vh * head_v_dim + j] = scratch[0] * inv_hvd;
        }
        __syncthreads();
    }
    row[tid] = row_val;
}

// RMSnorm + gate, batched over T. Each block normalizes one (token, v_head)
// vector of length head_v_dim.
__global__ void recurrent_norm_gate_batch_kernel(float *core,            // [T, num_v_heads * head_v_dim]
                                                 const float *gate,       // [T, num_v_heads * head_v_dim]
                                                 const float *norm_w,     // [head_v_dim]
                                                 uint32_t T,
                                                 uint32_t num_v_heads,
                                                 uint32_t head_v_dim,
                                                 uint32_t core_stride,
                                                 uint32_t gate_stride,
                                                 float eps) {
    const uint32_t vh = blockIdx.x;
    const uint32_t t  = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (vh >= num_v_heads || t >= T || tid >= head_v_dim) return;
    __shared__ float scratch[256];
    float *block = core + static_cast<uint64_t>(t) * core_stride + vh * head_v_dim;
    const float *gblock = gate + static_cast<uint64_t>(t) * gate_stride + vh * head_v_dim;
    scratch[tid] = block[tid] * block[tid];
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(head_v_dim) + eps);
    const float z = gblock[tid];
    block[tid] = block[tid] * scale * norm_w[tid] * (z / (1.0f + expf(-z)));
}

__global__ void rmsnorm_per_head_kernel(float *x,
                                        const float *w,
                                        uint32_t n_units,
                                        uint32_t per_unit_stride,
                                        uint32_t head_dim,
                                        uint32_t batch_stride,
                                        float eps) {
    const uint32_t unit = blockIdx.x;
    const uint32_t b    = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (unit >= n_units) return;
    __shared__ float scratch[256];
    float *base = x + static_cast<uint64_t>(b) * batch_stride + unit * per_unit_stride;
    float sum = 0.0f;
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) sum += base[i] * base[i];
    scratch[tid] = sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    const float scale = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
    for (uint32_t i = tid; i < head_dim; i += blockDim.x) base[i] = base[i] * scale * w[i];
}

__global__ void rope_partial_kernel(float *x,
                                    uint32_t n_units,
                                    uint32_t per_unit_stride,
                                    uint32_t rope_dim,
                                    uint32_t base_pos,
                                    uint32_t batch_stride,
                                    float theta) {
    const uint32_t unit = blockIdx.x;
    const uint32_t b    = blockIdx.y;
    const uint32_t i = threadIdx.x;
    if (unit >= n_units) return;
    const uint32_t half = rope_dim / 2;
    if (i >= half) return;
    float *base = x + static_cast<uint64_t>(b) * batch_stride + unit * per_unit_stride;
    const float inv_freq = __powf(theta, -2.0f * static_cast<float>(i) / static_cast<float>(rope_dim));
    const float angle = static_cast<float>(base_pos + b) * inv_freq;
    float c, s;
    __sincosf(angle, &s, &c);
    const float x0 = base[i];
    const float x1 = base[i + half];
    base[i]        = x0 * c - x1 * s;
    base[i + half] = x0 * s + x1 * c;
}

__global__ void kv_append_kernel(float *cache,
                                 const float *src,
                                 uint32_t base_pos,
                                 uint32_t per_pos_size,
                                 uint32_t src_stride) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;
    if (i >= per_pos_size) return;
    const float *src_row = src + static_cast<uint64_t>(b) * src_stride;
    cache[static_cast<uint64_t>(base_pos + b) * per_pos_size + i] = src_row[i];
}

__global__ void kv_append_kernel_f16(__half *cache,
                                     const float *src,
                                     uint32_t base_pos,
                                     uint32_t per_pos_size,
                                     uint32_t src_stride) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;
    if (i >= per_pos_size) return;
    const float *src_row = src + static_cast<uint64_t>(b) * src_stride;
    cache[static_cast<uint64_t>(base_pos + b) * per_pos_size + i] =
        __float2half(src_row[i]);
}

__global__ void attention_decode_qk_kernel(float *scores,
                                           const float *q,
                                           uint32_t q_stride,
                                           const float *k_cache,
                                           uint32_t n_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t seq_len,
                                           float scale) {
    // grid: [n_heads, seq_len] ; block: 256 threads, tree-reduce dot prod.
    const uint32_t head = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads || t >= seq_len) return;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);
    const float *q_attn = q + head * q_stride; // first head_dim floats
    const float *k_t = k_cache + static_cast<uint64_t>(t) * n_kv_heads * head_dim + kv_head * head_dim;
    __shared__ float scratch[256];
    float acc = 0.0f;
    for (uint32_t d = tid; d < head_dim; d += blockDim.x) acc += q_attn[d] * k_t[d];
    scratch[tid] = acc;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) scores[static_cast<uint64_t>(head) * seq_len + t] = scratch[0] * scale;
}

__global__ void attention_decode_softmax_kernel(float *scores,
                                                uint32_t n_heads,
                                                uint32_t seq_len) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads) return;
    float *row = scores + static_cast<uint64_t>(head) * seq_len;
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float scratch[256];

    float local_max = -INFINITY;
    for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
        if (row[t] > local_max) local_max = row[t];
    }
    scratch[tid] = local_max;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (scratch[tid + s] > scratch[tid]) scratch[tid] = scratch[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) s_max = scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t t = tid; t < seq_len; t += blockDim.x) {
        const float v = expf(row[t] - s_max);
        row[t] = v;
        local_sum += v;
    }
    scratch[tid] = local_sum;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    if (tid == 0) s_sum = scratch[0];
    __syncthreads();

    const float inv = 1.0f / s_sum;
    for (uint32_t t = tid; t < seq_len; t += blockDim.x) row[t] *= inv;
}

__global__ void attention_decode_av_kernel(float *out,
                                           const float *scores,
                                           const float *v_cache,
                                           uint32_t n_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t seq_len) {
    const uint32_t head = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (head >= n_heads || d >= head_dim) return;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);
    const float *row = scores + static_cast<uint64_t>(head) * seq_len;
    float acc = 0.0f;
    for (uint32_t t = 0; t < seq_len; ++t) {
        const float *vt = v_cache + static_cast<uint64_t>(t) * n_kv_heads * head_dim + kv_head * head_dim;
        acc += row[t] * vt[d];
    }
    out[static_cast<uint64_t>(head) * head_dim + d] = acc;
}

// Fused single-token SDPA decode (1 kernel instead of qk+softmax+av) with
// online softmax. One block per query head. HEAD_DIM=128 (Qwen3.5).
//
// Online softmax: process the KV stream in tiles of TILE tokens, maintain
// running (max, sum, acc):
//   new_max  = max(running_max, tile_max)
//   scale    = exp(running_max - new_max)
//   running_sum = scale * running_sum + sum_i exp(score_i - new_max)
//   running_acc = scale * running_acc + sum_i exp(score_i - new_max) * V_i
// This avoids materializing the scores tensor and reads K,V once per call.
template <uint32_t HEAD_DIM, uint32_t TILE>
__global__ void attention_decode_fused_kernel(float *out,
                                              const float *q,
                                              uint32_t q_stride,
                                              const float *k_cache,
                                              const float *v_cache,
                                              uint32_t n_heads,
                                              uint32_t n_kv_heads,
                                              uint32_t base_seq_len,
                                              uint32_t q_batch_stride,
                                              uint32_t out_batch_stride,
                                              float scale) {
    const uint32_t head = blockIdx.x;
    const uint32_t b    = blockIdx.y;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads) return;
    // Causal: batch position b attends to base_seq_len + b + 1 tokens.
    const uint32_t seq_len = base_seq_len + b + 1;
    const uint32_t kv_head = head / (n_heads / n_kv_heads);

    __shared__ float s_q[HEAD_DIM];
    __shared__ float s_scores[TILE];
    __shared__ float warp_sums[HEAD_DIM / 32];

    const float *q_ptr = q + static_cast<uint64_t>(b) * q_batch_stride;
    float       *out_ptr = out + static_cast<uint64_t>(b) * out_batch_stride;
    s_q[tid] = q_ptr[head * q_stride + tid];
    __syncthreads();

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc = 0.0f;

    const uint32_t warp = tid / 32;
    const uint32_t lane = tid % 32;
    constexpr uint32_t WARPS = HEAD_DIM / 32;

    for (uint32_t t0 = 0; t0 < seq_len; t0 += TILE) {
        const uint32_t this_tile = (seq_len - t0 < TILE) ? (seq_len - t0) : TILE;

        // Score each token in the tile via a cooperative dot product.
        for (uint32_t i = 0; i < this_tile; ++i) {
            const uint32_t t = t0 + i;
            const float *k_t = k_cache + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
            float local = s_q[tid] * k_t[tid];
            for (int delta = 16; delta > 0; delta >>= 1) {
                local += __shfl_xor_sync(0xffffffffu, local, delta);
            }
            if (lane == 0) warp_sums[warp] = local;
            __syncthreads();
            if (tid == 0) {
                float dot = 0.0f;
                #pragma unroll
                for (uint32_t w = 0; w < WARPS; ++w) dot += warp_sums[w];
                s_scores[i] = dot * scale;
            }
            __syncthreads();
        }

        float tile_max = -INFINITY;
        for (uint32_t i = 0; i < this_tile; ++i) {
            const float v = s_scores[i];
            if (v > tile_max) tile_max = v;
        }
        const float new_max = fmaxf(running_max, tile_max);
        const float prev_scale = (running_max == -INFINITY) ? 0.0f : __expf(running_max - new_max);
        running_sum = running_sum * prev_scale;
        acc = acc * prev_scale;
        for (uint32_t i = 0; i < this_tile; ++i) {
            const float w = __expf(s_scores[i] - new_max);
            const uint32_t t = t0 + i;
            const float *v_t = v_cache + static_cast<uint64_t>(t) * n_kv_heads * HEAD_DIM + kv_head * HEAD_DIM;
            running_sum += w;
            acc += w * v_t[tid];
        }
        running_max = new_max;
    }

    out_ptr[head * HEAD_DIM + tid] = acc / running_sum;
}

__global__ void apply_attn_gate_kernel(float *out,
                                       const float *q,
                                       uint32_t q_stride,
                                       uint32_t n_heads,
                                       uint32_t head_dim,
                                       uint32_t batch_stride_q,
                                       uint32_t batch_stride_out) {
    const uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t b = blockIdx.y;
    const uint64_t total = static_cast<uint64_t>(n_heads) * head_dim;
    if (i >= total) return;
    const uint32_t head = static_cast<uint32_t>(i / head_dim);
    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const float *qp = q + static_cast<uint64_t>(b) * batch_stride_q;
    float       *op = out + static_cast<uint64_t>(b) * batch_stride_out;
    const float gate = qp[head * q_stride + head_dim + d];
    op[i] *= 1.0f / (1.0f + expf(-gate));
}

__global__ void attention_norm_mid_kernel(float *mid,
                                          float *q,
                                          float *k,
                                          const float *v,
                                          const float *q_norm,
                                          const float *k_norm,
                                          uint32_t n_heads,
                                          uint32_t n_kv_heads,
                                          uint32_t head_dim,
                                          float eps) {
    const uint32_t head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (head >= n_heads || tid >= head_dim) return;
    __shared__ float scratch[256];

    float *q_base = q + head * 2 * head_dim;
    scratch[tid] = q_base[tid] * q_base[tid];
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) scratch[tid] += scratch[tid + s];
        __syncthreads();
    }
    q_base[tid] *= rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps) * q_norm[tid];
    __syncthreads();

    if (head < n_kv_heads) {
        float *k_base = k + head * head_dim;
        scratch[tid] = k_base[tid] * k_base[tid];
        __syncthreads();
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        k_base[tid] *= rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps) * k_norm[tid];
    }

    const uint32_t kvh = head % n_kv_heads;
    const float gate = q_base[head_dim + tid];
    mid[head * head_dim + tid] = v[kvh * head_dim + tid] * (1.0f / (1.0f + expf(-gate)));
}

class CudaDeviceBackend final : public DeviceBackend {
public:
    explicit CudaDeviceBackend(LinearBackend linear_backend)
        : linear_backend_(linear_backend) {}

    ~CudaDeviceBackend() override {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
        if (rows_buf_) cudaFree(rows_buf_);
        if (x_fp16_workspace_) cudaFree(x_fp16_workspace_);
        for (int i = 0; i < 2; ++i) {
            if (w_fp16_workspace_[i]) cudaFree(w_fp16_workspace_[i]);
            if (dequant_done_[i]) cudaEventDestroy(dequant_done_[i]);
            if (hgemm_done_[i]) cudaEventDestroy(hgemm_done_[i]);
        }
        if (dequant_stream_) cudaStreamDestroy(dequant_stream_);
        if (graph_instance_) cudaGraphExecDestroy(graph_instance_);
        if (exec_stream_) cudaStreamDestroy(exec_stream_);
        if (q8_1_scratch_) cudaFree(q8_1_scratch_);
        if (q8_1_mmq_scratch_) cudaFree(q8_1_mmq_scratch_);
        if (fattn_scratch_) cudaFree(fattn_scratch_);
        if (prefill_attn_q_fp16_) cudaFree(prefill_attn_q_fp16_);
        if (prefill_attn_scores_fp16_) cudaFree(prefill_attn_scores_fp16_);
        if (argmax_dev_) cudaFree(argmax_dev_);
        if (argmax_host_) cudaFreeHost(argmax_host_);
    }

    const char *name() const override {
        return "cuda-device";
    }

    DeviceStatus begin() override {
        if (auto st = cuda_status(cudaSetDevice(0), "cuda begin"); !st.ok) return st;
        if (!exec_stream_) {
            if (auto st = cuda_status(cudaStreamCreateWithFlags(&exec_stream_,
                                                                cudaStreamNonBlocking),
                                      "cuda exec stream"); !st.ok) return st;
        }
        if (!cublas_handle_) {
            if (auto st = cublas_status(cublasCreate(&cublas_handle_), "cublasCreate"); !st.ok) return st;
            if (auto st = cublas_status(cublasSetStream(cublas_handle_, exec_stream_),
                                        "cublasSetStream"); !st.ok) return st;
        }
        // Fresh forward — drop any X-FP16 cache from a prior generate().
        x_fp16_cached_src_ = nullptr;
        x_fp16_cached_elems_ = 0;
        return {};
    }

    DeviceStatus end() override {
        return synchronize();
    }

    DeviceStatus synchronize() override {
        if (!exec_stream_) {
            return cuda_status(cudaDeviceSynchronize(), "cuda synchronize");
        }
        return cuda_status(cudaStreamSynchronize(exec_stream_), "cuda stream synchronize");
    }

    // -- CUDA graph capture for decode --
    //
    // Strategy mirrors llama.cpp's pattern: every decode token re-records the
    // launches into a fresh cudaGraph_t (capture is cheap — nothing actually
    // runs). Then either instantiate a new executable on the first capture or
    // try cudaGraphExecUpdate on subsequent captures; on update failure
    // (kernel-arg shapes changed enough that the existing nodes can't be
    // patched in place), drop and re-instantiate. cudaGraphLaunch then issues
    // the whole forward in one go — saves ~150 launch-API trips per token.
    //
    // Falls back to the eager path on first failure (and keeps falling back
    // for the rest of the session). Toggle off entirely with QW3_GRAPH=off.
    bool begin_capture() override {
        if (graph_disabled_) return false;
        if (capture_active_) return false;  // refuse nested capture
        if (!exec_stream_) return false;
        const char *env = std::getenv("QW3_GRAPH");
        if (env && (std::strcmp(env, "off") == 0 || std::strcmp(env, "0") == 0)) {
            graph_disabled_ = true;
            return false;
        }
        // Ensure any pending work on exec_stream_ has completed before we
        // start capture (capture refuses if the stream has in-flight work
        // that wasn't itself captured, depending on the mode).
        cudaError_t st = cudaStreamSynchronize(exec_stream_);
        if (st != cudaSuccess) return false;
        st = cudaStreamBeginCapture(exec_stream_, cudaStreamCaptureModeRelaxed);
        if (st != cudaSuccess) {
            (void)cudaGetLastError();
            graph_disabled_ = true;
            return false;
        }
        capture_active_ = true;
        return true;
    }

    DeviceStatus end_capture() override {
        if (!capture_active_) return {};
        cudaGraph_t graph = nullptr;
        cudaError_t st = cudaStreamEndCapture(exec_stream_, &graph);
        capture_active_ = false;
        if (st != cudaSuccess) {
            (void)cudaGetLastError();
            graph_disabled_ = true;
            if (graph) cudaGraphDestroy(graph);
            return {false, "cudaStreamEndCapture failed"};
        }
        if (graph_instance_ == nullptr) {
            cudaError_t inst = cudaGraphInstantiate(&graph_instance_, graph, nullptr, nullptr, 0);
            if (inst != cudaSuccess) {
                (void)cudaGetLastError();
                graph_disabled_ = true;
                cudaGraphDestroy(graph);
                graph_instance_ = nullptr;
                return {false, "cudaGraphInstantiate failed"};
            }
        } else {
#if CUDART_VERSION >= 12000
            cudaGraphExecUpdateResultInfo info{};
            cudaError_t up = cudaGraphExecUpdate(graph_instance_, graph, &info);
#else
            cudaGraphNode_t err_node = nullptr;
            cudaGraphExecUpdateResult result = cudaGraphExecUpdateSuccess;
            cudaError_t up = cudaGraphExecUpdate(graph_instance_, graph, &err_node, &result);
#endif
            if (up != cudaSuccess) {
                // Topology / shape mismatch — drop and re-instantiate.
                (void)cudaGetLastError();
                cudaGraphExecDestroy(graph_instance_);
                graph_instance_ = nullptr;
                cudaError_t inst = cudaGraphInstantiate(&graph_instance_, graph, nullptr, nullptr, 0);
                if (inst != cudaSuccess) {
                    (void)cudaGetLastError();
                    graph_disabled_ = true;
                    cudaGraphDestroy(graph);
                    graph_instance_ = nullptr;
                    return {false, "cudaGraphInstantiate (after update fail) failed"};
                }
            }
        }
        cudaGraphDestroy(graph);
        return {};
    }

    DeviceStatus replay_graph() override {
        if (!graph_instance_) return {false, "no captured graph to replay"};
        cudaError_t st = cudaGraphLaunch(graph_instance_, exec_stream_);
        if (st != cudaSuccess) {
            (void)cudaGetLastError();
            return {false, "cudaGraphLaunch failed"};
        }
        return {};
    }

    std::unique_ptr<DeviceTensor> tensor_f32(uint64_t count, const char *label) override {
        return std::make_unique<CudaTensor>(count, label);
    }

    std::unique_ptr<DeviceTensor> tensor_f16(uint64_t count, const char *label) override {
        return std::make_unique<CudaTensor>(count, label, /*elem_bytes=*/sizeof(__half));
    }

    std::unique_ptr<DeviceWeight> weight_f32(const float *data, uint64_t count, const char *label) override {
        return std::make_unique<CudaWeight>(data, static_cast<uint64_t>(count * sizeof(float)), 1, count, WeightType::F32, label);
    }

    std::unique_ptr<DeviceWeight> weight_q8_0(const void *data, uint64_t rows, uint64_t cols, const char *label) override {
        return std::make_unique<CudaWeight>(data, rows * (cols / 32) * 34, rows, cols, WeightType::Q8_0, label);
    }

    DeviceStatus q8_0_get_row(DeviceTensor &out, const DeviceWeight &weight, uint64_t row) override {
        const auto &w = as_weight(weight);
        auto &o = as_tensor(out);
        q8_get_row_kernel<<<static_cast<unsigned>((w.cols + 255) / 256), 256, 0, exec_stream_>>>(
            o.ptr, static_cast<const uint8_t *>(w.ptr), row, w.cols);
        return launch_status("cuda q8_0_get_row");
    }

    DeviceStatus q8_0_get_rows_batch(DeviceTensor &out,
                                     const DeviceWeight &weight,
                                     const uint64_t *rows,
                                     uint32_t batch) override {
        if (batch == 0) return {};
        const auto &w = as_weight(weight);
        auto &o = as_tensor(out);
        // Upload row indices to device (small, one transfer).
        if (rows_buf_capacity_ < batch) {
            if (rows_buf_) cudaFree(rows_buf_);
            cudaMalloc(&rows_buf_, batch * sizeof(uint64_t));
            rows_buf_capacity_ = batch;
        }
        cudaMemcpyAsync(rows_buf_, rows, batch * sizeof(uint64_t), cudaMemcpyHostToDevice, exec_stream_);
        const unsigned threads = 256;
        const unsigned bx = static_cast<unsigned>((w.cols + threads - 1) / threads);
        dim3 grid(bx, batch);
        q8_get_rows_batch_kernel<<<grid, threads, 0, exec_stream_>>>(o.ptr, static_cast<const uint8_t *>(w.ptr),
                                                    rows_buf_, w.cols);
        return launch_status("cuda q8_0_get_rows_batch");
    }

    DeviceStatus q8_0_matvec(DeviceTensor &out, const DeviceWeight &weight, const DeviceTensor &x) override {
        auto &w = const_cast<CudaWeight &>(as_weight(weight));
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        // The cuBLAS-Sgemv-on-F32-dequant-cache path was an early shortcut. It
        // reads 4x more bytes per matvec (F32 cache vs Q8 weight) so even though
        // it uses tensor cores, it loses to a properly tuned Q8 kernel for the
        // single-token decode workload. Only switch to it when explicitly asked.
        const bool use_cublas = linear_backend_ == LinearBackend::Cublas;
        if (use_cublas) {
            // Only take the cuBLAS path when the F32 dequant fits in our
            // persistent per-weight cache. Otherwise the previous behaviour
            // (cudaMalloc + dequant + sgemv + cudaFree on every call) blew up
            // wall time on big tensors like output.weight (~5 GiB F32 form).
            // For those, fall through to the custom Q8 matvec which reads the
            // mmap-uploaded Q8_0 blocks directly.
            const uint64_t f32_bytes = w.rows * w.cols * sizeof(float);
            const bool can_cache = f32_bytes <= (256ull << 20);
            if (can_cache) {
                if (auto st = ensure_q8_f32_cache(w); st.ok) {
                    const float alpha = 1.0f;
                    const float beta = 0.0f;
                    if (auto sgemv_st = cublas_status(
                            cublasSgemv(cublas_handle_,
                                        CUBLAS_OP_T,
                                        static_cast<int>(w.cols),
                                        static_cast<int>(w.rows),
                                        &alpha,
                                        w.q8_f32_cache,
                                        static_cast<int>(w.cols),
                                        input.ptr,
                                        1,
                                        &beta,
                                        o.ptr,
                                        1),
                            "cublasSgemv q8_0_matvec");
                        sgemv_st.ok) {
                        return {};
                    }
                }
                // cache build or sgemv failed: drop the error state and
                // fall through to the custom kernel.
                (void) cudaGetLastError();
            }
        }
        // Pick WARPS_PER_BLOCK so each block has enough rows-per-block to
        // amortize the shared input-vector load while still launching enough
        // blocks to saturate all SMs.
        //   - small matvecs (KV proj, alpha/beta): 8 warps, max parallelism
        //   - mid (output proj, ffn_down, attn_q): 16 warps
        //   - big rows (ffn_gate/up, LM head): 32 warps, fewer redundant input reads
        // Shmem budget per layout:
        //   - DP4A path  : cols * 1 (i8) + blocks * 4 (f32 scale per block)
        //   - F32 fallback: cols * 4 (f32 staging)
        return dispatch_q8_matvec(o.ptr, w, input.ptr, /*batch=*/1, /*in_stride=*/0, /*out_stride=*/0);
    }

    DeviceStatus q8_0_matvec_fanout(DeviceTensor *const *outs,
                                    const DeviceWeight *const *weights,
                                    uint32_t n,
                                    const DeviceTensor &x) override {
        if (n == 0) return {};
        const auto &input = as_tensor(x);
        // Validate every weight has the same input width — that's the
        // precondition for sharing one Q8_1 quantization. The recurrent /
        // attention / FFN call sites all satisfy this; if a future caller
        // doesn't, fall back to per-call dispatch.
        const uint64_t cols = as_weight(*weights[0]).cols;
        bool same_cols = true;
        for (uint32_t i = 1; i < n; ++i) {
            if (as_weight(*weights[i]).cols != cols) { same_cols = false; break; }
        }
        const bool fast_path = same_cols && (cols % 32) == 0 &&
                               matvec_kernel_choice() == MatvecKernel::Ported;
        if (!fast_path) {
            return DeviceBackend::q8_0_matvec_fanout(outs, weights, n, x);
        }
        if (auto st = ensure_q8_1_scratch(/*batch=*/1, static_cast<uint32_t>(cols)); !st.ok) return st;
        if (!ported::launch_quantize_q8_1(input.ptr, q8_1_scratch_,
                                          /*batch=*/1, static_cast<uint32_t>(cols),
                                          /*stride_x_row=*/static_cast<uint32_t>(cols),
                                          exec_stream_)) {
            return {false, "fanout quantize_q8_1 launch failed"};
        }
        // Pairwise fuse adjacent same-shape weights into a single matvec
        // kernel (one CTA grid produces both outputs, sharing the activation
        // L2 read). Hot caller is FFN gate+up where both weights share
        // (rows, cols). Q+K+V/recurrent fanouts have heterogeneous rows so
        // they fall through to the per-call path. Disable with QW3_FUSE2=off.
        const bool fuse2 = []() {
            const char *e = std::getenv("QW3_FUSE2");
            return !(e && std::string(e) == "off");
        }();
        uint32_t i = 0;
        while (i < n) {
            const auto &wa = as_weight(*weights[i]);
            if (fuse2 && i + 1 < n) {
                const auto &wb = as_weight(*weights[i + 1]);
                if (wa.rows == wb.rows && wa.cols == wb.cols) {
                    auto &oa = as_tensor(*outs[i]);
                    auto &ob = as_tensor(*outs[i + 1]);
                    if (!ported::launch_mmvq_q8_0_two(
                            static_cast<const uint8_t *>(wa.ptr),
                            static_cast<const uint8_t *>(wb.ptr),
                            q8_1_scratch_,
                            oa.ptr, ob.ptr,
                            static_cast<uint32_t>(wa.rows),
                            static_cast<uint32_t>(wa.cols),
                            /*batch=*/1,
                            /*stride_dst_row=*/static_cast<uint32_t>(wa.rows),
                            exec_stream_)) {
                        return {false, "fanout mmvq_q8_0_two launch failed"};
                    }
                    i += 2;
                    continue;
                }
            }
            auto &o = as_tensor(*outs[i]);
            if (!ported::launch_mmvq_q8_0(static_cast<const uint8_t *>(wa.ptr),
                                          q8_1_scratch_, o.ptr,
                                          static_cast<uint32_t>(wa.rows),
                                          static_cast<uint32_t>(wa.cols),
                                          /*batch=*/1,
                                          /*stride_dst_row=*/static_cast<uint32_t>(wa.rows),
                                          exec_stream_)) {
                return {false, "fanout mmvq_q8_0 launch failed"};
            }
            ++i;
        }
        return launch_status("cuda q8_0_matvec_fanout");
    }

    // Fused matvec + residual add: dst = dst + W*x. Uses the dedicated
    // mmvq_add kernel that writes through dst with read-modify-write
    // instead of materializing the matvec result into a separate buffer
    // and adding it. Saves one add_kernel launch and the round-trip of
    // an n_embd-wide scratch buffer per call. Disable with QW3_FUSE_ADD=off
    // to A/B against the separate-launch path.
    DeviceStatus q8_0_matvec_add(DeviceTensor &accum,
                                 const DeviceWeight &weight,
                                 const DeviceTensor &x) override {
        const bool fuse = []() {
            const char *e = std::getenv("QW3_FUSE_ADD");
            return !(e && std::string(e) == "off");
        }();
        if (!fuse) {
            return DeviceBackend::q8_0_matvec_add(accum, weight, x);
        }
        const auto &w = as_weight(weight);
        if (matvec_kernel_choice() != MatvecKernel::Ported || (w.cols % 32) != 0) {
            return DeviceBackend::q8_0_matvec_add(accum, weight, x);
        }
        auto &o = as_tensor(accum);
        const auto &input = as_tensor(x);
        if (auto st = ensure_q8_1_scratch(/*batch=*/1, static_cast<uint32_t>(w.cols)); !st.ok) return st;
        if (!ported::launch_quantize_q8_1(input.ptr, q8_1_scratch_,
                                          /*batch=*/1, static_cast<uint32_t>(w.cols),
                                          /*stride_x_row=*/static_cast<uint32_t>(w.cols),
                                          exec_stream_)) {
            return {false, "matvec_add quantize_q8_1 launch failed"};
        }
        if (!ported::launch_mmvq_q8_0_add(static_cast<const uint8_t *>(w.ptr),
                                          q8_1_scratch_, o.ptr,
                                          static_cast<uint32_t>(w.rows),
                                          static_cast<uint32_t>(w.cols),
                                          /*batch=*/1,
                                          /*stride_dst_row=*/static_cast<uint32_t>(w.rows),
                                          exec_stream_)) {
            return {false, "mmvq_q8_0_add launch failed"};
        }
        return launch_status("cuda q8_0_matvec_add");
    }

    // Fused FFN SwiGLU: out = silu(W_gate * x) * (W_up * x).
    DeviceStatus q8_0_matvec_silu_mul(DeviceTensor &out,
                                      const DeviceWeight &weight_gate,
                                      const DeviceWeight &weight_up,
                                      const DeviceTensor &x) override {
        const bool fuse = []() {
            const char *e = std::getenv("QW3_FUSE_SILU_MUL");
            return !(e && std::string(e) == "off");
        }();
        if (!fuse) {
            return DeviceBackend::q8_0_matvec_silu_mul(out, weight_gate, weight_up, x);
        }
        const auto &wg = as_weight(weight_gate);
        const auto &wu = as_weight(weight_up);
        if (wg.rows != wu.rows || wg.cols != wu.cols ||
            matvec_kernel_choice() != MatvecKernel::Ported || (wg.cols % 32) != 0) {
            return DeviceBackend::q8_0_matvec_silu_mul(out, weight_gate, weight_up, x);
        }
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        if (auto st = ensure_q8_1_scratch(/*batch=*/1, static_cast<uint32_t>(wg.cols)); !st.ok) return st;
        if (!ported::launch_quantize_q8_1(input.ptr, q8_1_scratch_,
                                          /*batch=*/1, static_cast<uint32_t>(wg.cols),
                                          /*stride_x_row=*/static_cast<uint32_t>(wg.cols),
                                          exec_stream_)) {
            return {false, "silu_mul quantize_q8_1 launch failed"};
        }
        if (!ported::launch_mmvq_q8_0_silu_mul(
                static_cast<const uint8_t *>(wg.ptr),
                static_cast<const uint8_t *>(wu.ptr),
                q8_1_scratch_, o.ptr,
                static_cast<uint32_t>(wg.rows),
                static_cast<uint32_t>(wg.cols),
                /*batch=*/1,
                /*stride_dst_row=*/static_cast<uint32_t>(wg.rows),
                exec_stream_)) {
            return {false, "mmvq_q8_0_silu_mul launch failed"};
        }
        return launch_status("cuda q8_0_matvec_silu_mul");
    }

    DeviceStatus q8_0_matmul(DeviceTensor &out,
                             const DeviceWeight &weight,
                             const DeviceTensor &x,
                             uint32_t batch,
                             uint32_t in_stride,
                             uint32_t out_stride) override {
        auto &w = const_cast<CudaWeight &>(as_weight(weight));
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        if (batch == 0) return {};
        // For prefill batches the dp4a matvec stops being bandwidth-bound and
        // tensor-core HGEMM wins by 10x+. Threshold of 8 is the empirical
        // cross-over on Blackwell for the smallest matmul we hit (5K x 5K).
        // For batch == 1 (decode) we stay on the dp4a path which is faster.
        const uint32_t hgemm_threshold = 8;
        if (batch >= hgemm_threshold && in_stride == w.cols && out_stride == w.rows) {
            // INT8 MMA path (opt-in via QW3_MATMUL=mmq). Operates on the raw
            // Q8_0 weight + on-the-fly Q8_1 activation; no FP16 dequant cache,
            // no FP32 -> FP16 activation conversion. Falls back to HGEMM on
            // any failure (alloc, launch, shape constraint).
            if (matmul_kernel_choice() == MatmulKernel::Mmq && (w.cols % 32) == 0) {
                if (auto st = mmq_q8(o.ptr, w, input.ptr, batch); st.ok) return st;
            }
            DeviceStatus st = hgemm_q8(o.ptr, w, input.ptr, batch);
            if (st.ok) return st;
            // Fall through to dp4a on HGEMM error (e.g. OOM allocating cache).
        }
        return dispatch_q8_matvec(o.ptr, w, input.ptr, batch, in_stride, out_stride);
    }

private:
    // FP16 HGEMM for Q8_0 weights. On every call we dequant the weight into
    // a per-call FP16 scratch (no persistent mirror — keeps weight memory at
    // 8-bit only), then run cublasGemmEx with FP16 inputs and FP32 accum.
    //
    // Pipelined: two ping-pong scratch buffers and two CUDA streams let the
    // dequant of W_{n+1} run concurrently with the HGEMM of W_n. Events
    // synchronize buffer ownership:
    //   - dequant_done_[idx]: signals the dequant of buffer[idx] is finished
    //                         and HGEMM may consume it.
    //   - hgemm_done_[idx]:   signals HGEMM has finished reading buffer[idx]
    //                         and the next dequant may overwrite it.
    //
    // Layout:  O[b, r] = sum_c W[r, c] * X[b, c],  W (rows, cols) row-major,
    //          X (batch, cols) row-major, O (batch, rows) row-major.
    //
    // In cuBLAS column-major terms with A=W (CMV: cols x rows) and
    // B=X (CMV: cols x batch):  C(rows, batch) = A^T * B.
    DeviceStatus hgemm_q8(float *out_ptr, CudaWeight &w, const float *x_ptr, uint32_t batch) {
        if (auto st = ensure_dequant_pipeline(); !st.ok) return st;
        const uint64_t w_elems = w.rows * w.cols;
        if (w_fp16_capacity_ < w_elems) {
            // Resize both ping-pong buffers and clear pending events.
            for (int i = 0; i < 2; ++i) {
                if (w_fp16_workspace_[i]) cudaFree(w_fp16_workspace_[i]);
                w_fp16_workspace_[i] = nullptr;
            }
            for (int i = 0; i < 2; ++i) {
                if (auto st = cuda_status(cudaMalloc(&w_fp16_workspace_[i],
                                                     static_cast<size_t>(w_elems) * sizeof(__half)),
                                          "hgemm w_fp16 alloc"); !st.ok) {
                    w_fp16_capacity_ = 0;
                    return st;
                }
            }
            w_fp16_capacity_ = w_elems;
            w_fp16_idx_ = 0;
            // No prior in-flight dequants/HGEMMs to wait on after a resize.
        }

        const int idx = w_fp16_idx_;

        // Dequant runs on dequant_stream_; wait for the previous HGEMM that
        // read this buffer to finish before overwriting.
        cudaStreamWaitEvent(dequant_stream_, hgemm_done_[idx], 0);
        if (auto st = dequant_q8_to_f16_stream(w_fp16_workspace_[idx], w, dequant_stream_); !st.ok) return st;
        cudaEventRecord(dequant_done_[idx], dequant_stream_);

        // Stage X (FP32 -> FP16) on the cuBLAS / default stream. This work
        // doesn't conflict with the dequant since they touch different
        // buffers (x_fp16_workspace_ vs w_fp16_workspace_[idx]).
        //
        // Cache: QKV and gate/up matmul pairs share the same x_ptr in
        // immediate succession. If the prior conversion was for the same
        // (x_ptr, x_elems), the FP16 staging buffer still holds it — no
        // intervening kernel writes to x_fp16_workspace_, and by invariant
        // no kernel writes to x_ptr's buffer between hgemm_q8 calls with
        // shared input. Skip the conversion launch on hit. Disable via
        // QW3_HGEMM_X_CACHE=0 for diagnostic / forced-recompute runs.
        const uint64_t x_elems = static_cast<uint64_t>(batch) * w.cols;
        if (x_fp16_capacity_ < x_elems) {
            if (x_fp16_workspace_) cudaFree(x_fp16_workspace_);
            if (auto st = cuda_status(cudaMalloc(&x_fp16_workspace_,
                                                 static_cast<size_t>(x_elems) * sizeof(__half)),
                                      "hgemm x_fp16 alloc"); !st.ok) {
                x_fp16_workspace_ = nullptr;
                x_fp16_capacity_ = 0;
                x_fp16_cached_src_ = nullptr;
                x_fp16_cached_elems_ = 0;
                return st;
            }
            x_fp16_capacity_ = x_elems;
            x_fp16_cached_src_ = nullptr;     // resize invalidates the cache
            x_fp16_cached_elems_ = 0;
        }
        const bool x_cache_disabled = []() {
            const char *e = std::getenv("QW3_HGEMM_X_CACHE");
            return e && (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0);
        }();
        const bool x_cache_hit = !x_cache_disabled
                              && x_fp16_cached_src_   == x_ptr
                              && x_fp16_cached_elems_ == x_elems;
        if (!x_cache_hit) {
            const unsigned threads = 256;
            // Vectorized kernel: 4 elements per thread.
            const uint64_t threads_total = (x_elems + 3) / 4;
            const unsigned blocks = static_cast<unsigned>((threads_total + threads - 1) / threads);
            fp32_to_fp16_kernel<<<blocks, threads, 0, exec_stream_>>>(x_fp16_workspace_, x_ptr, x_elems);
            if (auto st = launch_status("hgemm fp32->fp16"); !st.ok) {
                x_fp16_cached_src_ = nullptr;
                x_fp16_cached_elems_ = 0;
                return st;
            }
            x_fp16_cached_src_ = x_ptr;
            x_fp16_cached_elems_ = x_elems;
        }

        // cuBLAS waits for the matching dequant to complete before reading
        // buffer[idx]. cuBLAS handle is bound to exec_stream_; issue the
        // stream-wait there.
        cudaStreamWaitEvent(exec_stream_, dequant_done_[idx], 0);

        const float alpha = 1.0f;
        const float beta = 0.0f;
        const int m = static_cast<int>(w.rows);
        const int n = static_cast<int>(batch);
        const int k = static_cast<int>(w.cols);
        // CUBLAS_COMPUTE_32F_FAST_16F: FP16 inputs, FP32 accumulator with the
        // FP16-fast-path tensor-core kernels. Tried CUBLAS_COMPUTE_16F (FP16
        // accumulator) for ~2x throughput; cuBLAS 13.x falls back to a non-
        // tensor-core path for our shapes (Q8 weight FP16 mirrors with K up
        // to ~17K), regressing prefill 25x. Stay on FAST_16F.
        if (auto st = cublas_status(
                cublasGemmEx(cublas_handle_,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             m, n, k,
                             &alpha,
                             w_fp16_workspace_[idx], CUDA_R_16F, k,
                             x_fp16_workspace_, CUDA_R_16F, k,
                             &beta,
                             out_ptr, CUDA_R_32F, m,
                             CUBLAS_COMPUTE_32F_FAST_16F,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                "cublasGemmEx hgemm_q8"); !st.ok) {
            return st;
        }
        cudaEventRecord(hgemm_done_[idx], exec_stream_);

        w_fp16_idx_ ^= 1;
        return {};
    }

    DeviceStatus ensure_dequant_pipeline() {
        if (dequant_stream_) return {};
        if (auto st = cuda_status(cudaStreamCreateWithFlags(&dequant_stream_, cudaStreamNonBlocking),
                                  "dequant stream"); !st.ok) return st;
        for (int i = 0; i < 2; ++i) {
            if (auto st = cuda_status(cudaEventCreateWithFlags(&dequant_done_[i], cudaEventDisableTiming),
                                      "dequant event"); !st.ok) return st;
            if (auto st = cuda_status(cudaEventCreateWithFlags(&hgemm_done_[i], cudaEventDisableTiming),
                                      "hgemm event"); !st.ok) return st;
            // Initial events are signalled so the first dequant doesn't
            // block waiting on a non-existent prior HGEMM.
            cudaEventRecord(dequant_done_[i], exec_stream_);
            cudaEventRecord(hgemm_done_[i],   exec_stream_);
        }
        return {};
    }

    // Lazy/owning allocator for the Q8_1 activation scratch shared by mmvq
    // matvec, mmq matmul, and the matvec_fanout fast path.
    DeviceStatus ensure_q8_1_scratch(uint32_t batch, uint32_t cols) {
        const size_t need = ported::q8_1_scratch_bytes(batch, cols);
        if (need > q8_1_scratch_capacity_) {
            if (q8_1_scratch_) cudaFree(q8_1_scratch_);
            if (auto st = cuda_status(cudaMalloc(&q8_1_scratch_, need),
                                      "q8_1 scratch alloc"); !st.ok) {
                q8_1_scratch_ = nullptr;
                q8_1_scratch_capacity_ = 0;
                return st;
            }
            q8_1_scratch_capacity_ = need;
        }
        return {};
    }

    DeviceStatus ensure_q8_1_mmq_scratch(uint32_t batch, uint32_t cols) {
        const size_t need = ported::q8_1_mmq_scratch_bytes(batch, cols);
        if (need > q8_1_mmq_scratch_capacity_) {
            if (q8_1_mmq_scratch_) cudaFree(q8_1_mmq_scratch_);
            if (auto st = cuda_status(cudaMalloc(&q8_1_mmq_scratch_, need),
                                      "q8_1_mmq scratch alloc"); !st.ok) {
                q8_1_mmq_scratch_ = nullptr;
                q8_1_mmq_scratch_capacity_ = 0;
                return st;
            }
            q8_1_mmq_scratch_capacity_ = need;
        }
        return {};
    }

    // Returns true when the v4 path is selected (QW3_MMQ_VERSION=4).
    static bool mmq_v4_selected() {
        static const bool v = []() {
            const char *e = std::getenv("QW3_MMQ_VERSION");
            return e && e[0] == '4';
        }();
        return v;
    }

    // INT8 MMA Q8_0 x Q8_1 matmul (m16n8k32.s8.s8.s32). Uses the same
    // q8_1_scratch_ buffer as decode mmvq for the activation Q8_1 stage.
    // v4 (QW3_MMQ_VERSION=4) uses the 144-byte block_q8_1_mmq layout via
    // q8_1_mmq_scratch_ — it's a tile-friendly format that lets the MMQ
    // kernel read mmq_x*36 contiguous ints per K super-iter.
    DeviceStatus mmq_q8(float *out_ptr, CudaWeight &w, const float *x_ptr, uint32_t batch) {
        const bool use_v4 = mmq_v4_selected()
                          && (w.cols % 256 == 0)
                          && (w.rows >= 128) && (batch >= 128);
        if (use_v4) {
            if (auto st = ensure_q8_1_mmq_scratch(batch, w.cols); !st.ok) return st;
            if (!ported::launch_quantize_mmq_q8_1(x_ptr, q8_1_mmq_scratch_,
                                                   batch, w.cols, /*stride_x_row=*/w.cols,
                                                   exec_stream_)) {
                return {false, "mmq quantize_mmq_q8_1 launch failed"};
            }
            if (!ported::launch_mmq_q8_0(static_cast<const uint8_t *>(w.ptr),
                                         q8_1_mmq_scratch_, out_ptr,
                                         static_cast<uint32_t>(w.rows), w.cols,
                                         batch, /*stride_dst_row=*/static_cast<uint32_t>(w.rows),
                                         exec_stream_)) {
                return {false, "mmq launch failed (v4)"};
            }
            return launch_status("cuda q8_0_matmul mmq v4");
        }

        if (auto st = ensure_q8_1_scratch(batch, w.cols); !st.ok) return st;
        if (!ported::launch_quantize_q8_1(x_ptr, q8_1_scratch_,
                                          batch, w.cols, /*stride_x_row=*/w.cols,
                                          exec_stream_)) {
            return {false, "mmq quantize_q8_1 launch failed"};
        }
        if (!ported::launch_mmq_q8_0(static_cast<const uint8_t *>(w.ptr),
                                     q8_1_scratch_, out_ptr,
                                     static_cast<uint32_t>(w.rows), w.cols,
                                     batch, /*stride_dst_row=*/static_cast<uint32_t>(w.rows),
                                     exec_stream_)) {
            return {false, "mmq launch failed"};
        }
        return launch_status("cuda q8_0_matmul mmq");
    }

    DeviceStatus dispatch_q8_matvec(float *out_ptr,
                                    CudaWeight &w,
                                    const float *x_ptr,
                                    uint32_t batch,
                                    uint32_t in_stride,
                                    uint32_t out_stride) {
        // Ported path: quantize input to Q8_1, then DP4A matvec with Q8_1
        // activations. Selected via QW3_MATVEC=ported. Caller-side scratch
        // is reused across calls. Restricted to batch==1 (decode) for now;
        // for batch 2..7 the qw3 kernel's per-block input-quant cache is
        // already efficient and the ported kernel's higher block count
        // regresses small-batch prefill. Falls back to qw3 path otherwise.
        if (matvec_kernel_choice() == MatvecKernel::Ported &&
            (w.cols % 32) == 0 && batch == 1) {
            if (auto st = ensure_q8_1_scratch(batch, w.cols); !st.ok) return st;
            // in_stride is the float stride between batch rows in x; for
            // batch==1 the upstream call passes 0, but the ported quantize
            // kernel still walks (batch=1) row, so it never actually reads
            // past row 0.
            const uint32_t qstride = (batch == 1) ? w.cols : in_stride;
            if (!ported::launch_quantize_q8_1(x_ptr, q8_1_scratch_,
                                              batch, w.cols, qstride, exec_stream_)) {
                return {false, "ported quantize_q8_1 launch failed"};
            }
            const uint32_t dst_stride = (batch == 1) ? w.rows : out_stride;
            if (!ported::launch_mmvq_q8_0(static_cast<const uint8_t *>(w.ptr),
                                          q8_1_scratch_, out_ptr,
                                          static_cast<uint32_t>(w.rows), w.cols,
                                          batch, dst_stride, exec_stream_)) {
                return {false, "ported mmvq_q8_0 launch failed"};
            }
            return launch_status("cuda q8_0_matvec ported");
        }

        // Shmem budget per layout:
        //   - DP4A path  : cols * 1 (i8) + blocks * 4 (f32 scale per block)
        //   - F32 fallback: cols * 4 (f32 staging)
        const size_t shmem_dp4a = static_cast<size_t>(w.cols) + static_cast<size_t>(w.cols / 32) * sizeof(float);
        const size_t shmem_f32  = static_cast<size_t>(w.cols) * sizeof(float);
        const size_t shmem_limit = q8_matvec_max_shmem();
        const uint64_t rows = w.rows;
        const bool use_dp4a = shmem_dp4a <= shmem_limit;
        const bool fits_f32_shmem = shmem_f32 <= shmem_limit;

        // Empirical: for Qwen 3.6 27B Q8_0 on Pro 6000 the per-batch kernel
        // already amortizes W reads through L2 (89 MB FFN weight fits the
        // ~128 MB L2), so the tiled kernel's higher shmem use and lower
        // occupancy made prefill *slower*. Keep it gated off for now.
        if (false && batch > 1 && use_dp4a) {
            const size_t dp4a8 = shmem_dp4a * 8;
            const size_t dp4a4 = shmem_dp4a * 4;
            const size_t dp4a2 = shmem_dp4a * 2;
            auto launch_tiled = [&](auto tile, auto warps) {
                constexpr uint32_t T = decltype(tile)::value;
                constexpr uint32_t W = decltype(warps)::value;
                const unsigned by = (batch + T - 1) / T;
                const unsigned bx = static_cast<unsigned>((rows + W - 1) / W);
                dim3 grid(bx, by);
                q8_matmul_tiled_dp4a_kernel<W, T>
                    <<<grid, W * 32, shmem_dp4a * T, exec_stream_>>>(
                        out_ptr, static_cast<const uint8_t *>(w.ptr), x_ptr,
                        rows, w.cols, batch, in_stride, out_stride);
            };
            const auto warps_for_rows = [&]() {
                if (rows >= 16384) return 32u;
                if (rows >= 2048) return 16u;
                return 8u;
            };
            const uint32_t W = warps_for_rows();
            if (dp4a8 <= shmem_limit && batch >= 8) {
                if (W == 32) launch_tiled(std::integral_constant<uint32_t, 8>{}, std::integral_constant<uint32_t, 32>{});
                else if (W == 16) launch_tiled(std::integral_constant<uint32_t, 8>{}, std::integral_constant<uint32_t, 16>{});
                else launch_tiled(std::integral_constant<uint32_t, 8>{}, std::integral_constant<uint32_t, 8>{});
                return launch_status("cuda q8_matmul tiled8");
            }
            if (dp4a4 <= shmem_limit && batch >= 4) {
                if (W == 32) launch_tiled(std::integral_constant<uint32_t, 4>{}, std::integral_constant<uint32_t, 32>{});
                else if (W == 16) launch_tiled(std::integral_constant<uint32_t, 4>{}, std::integral_constant<uint32_t, 16>{});
                else launch_tiled(std::integral_constant<uint32_t, 4>{}, std::integral_constant<uint32_t, 8>{});
                return launch_status("cuda q8_matmul tiled4");
            }
            if (dp4a2 <= shmem_limit) {
                if (W == 32) launch_tiled(std::integral_constant<uint32_t, 2>{}, std::integral_constant<uint32_t, 32>{});
                else if (W == 16) launch_tiled(std::integral_constant<uint32_t, 2>{}, std::integral_constant<uint32_t, 16>{});
                else launch_tiled(std::integral_constant<uint32_t, 2>{}, std::integral_constant<uint32_t, 8>{});
                return launch_status("cuda q8_matmul tiled2");
            }
            // Fall through to per-batch kernel below.
        }

        auto dispatch = [&](auto warps) {
            constexpr uint32_t W = decltype(warps)::value;
            const unsigned blocks_grid = static_cast<unsigned>((rows + W - 1) / W);
            dim3 grid(blocks_grid, batch);
            if (use_dp4a) {
                q8_matvec_dp4a_kernel<W>
                    <<<grid, W * 32, shmem_dp4a, exec_stream_>>>(
                        out_ptr, static_cast<const uint8_t *>(w.ptr), x_ptr,
                        rows, w.cols, in_stride, out_stride);
            } else if (fits_f32_shmem) {
                q8_matvec_v2_kernel<W, true>
                    <<<grid, W * 32, shmem_f32, exec_stream_>>>(
                        out_ptr, static_cast<const uint8_t *>(w.ptr), x_ptr,
                        rows, w.cols, in_stride, out_stride);
            } else {
                q8_matvec_v2_kernel<W, false>
                    <<<grid, W * 32, 0, exec_stream_>>>(
                        out_ptr, static_cast<const uint8_t *>(w.ptr), x_ptr,
                        rows, w.cols, in_stride, out_stride);
            }
        };
        if (rows >= 16384) {
            dispatch(std::integral_constant<uint32_t, 32>{});
        } else if (rows >= 2048) {
            dispatch(std::integral_constant<uint32_t, 16>{});
        } else {
            dispatch(std::integral_constant<uint32_t, 8>{});
        }
        return launch_status("cuda q8_0_matvec");
    }
public:

    DeviceStatus rms_norm(DeviceTensor &out, const DeviceTensor &x, const DeviceWeight &weight, float eps) override {
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        const auto &w = as_weight(weight);
        const uint64_t n = input.count;
        // Vectorized path: float4-aligned and big enough to make a 1024-thread
        // block worth launching. Otherwise fall back to the scalar kernel.
        const bool can_vec = (n % 4 == 0)
            && ((reinterpret_cast<uintptr_t>(input.ptr) & 0xF) == 0)
            && ((reinterpret_cast<uintptr_t>(o.ptr) & 0xF) == 0)
            && ((reinterpret_cast<uintptr_t>(w.ptr) & 0xF) == 0)
            && n >= 256;
        if (can_vec) {
            rms_norm_kernel_vec<1024><<<1, 1024, 0, exec_stream_>>>(o.ptr, input.ptr,
                                                    static_cast<const float *>(w.ptr),
                                                    n, eps);
        } else {
            rms_norm_kernel<<<1, 256, 0, exec_stream_>>>(o.ptr, input.ptr,
                                        static_cast<const float *>(w.ptr), n, eps);
        }
        return launch_status("cuda rms_norm");
    }

    DeviceStatus rms_norm_batch(DeviceTensor &out,
                                const DeviceTensor &x,
                                const DeviceWeight &weight,
                                uint32_t batch,
                                uint32_t n,
                                float eps) override {
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        const auto &w = as_weight(weight);
        if (batch == 0) return {};
        // Prefer the vectorized 1024-thread kernel when alignment + size allow.
        // Same constraints as the single-row path; batch is handled via gridDim.x.
        const bool can_vec = (n % 4 == 0)
            && ((reinterpret_cast<uintptr_t>(input.ptr) & 0xF) == 0)
            && ((reinterpret_cast<uintptr_t>(o.ptr) & 0xF) == 0)
            && ((reinterpret_cast<uintptr_t>(w.ptr) & 0xF) == 0)
            && n >= 256;
        if (can_vec) {
            rms_norm_kernel_vec<1024><<<batch, 1024, 0, exec_stream_>>>(
                o.ptr, input.ptr, static_cast<const float *>(w.ptr), n, eps);
        } else {
            rms_norm_kernel<<<batch, 256, 0, exec_stream_>>>(
                o.ptr, input.ptr, static_cast<const float *>(w.ptr), n, eps);
        }
        return launch_status("cuda rms_norm_batch");
    }

    DeviceStatus add(DeviceTensor &out, const DeviceTensor &a, const DeviceTensor &b) override {
        auto &o = as_tensor(out);
        const auto &aa = as_tensor(a);
        const auto &bb = as_tensor(b);
        const uint64_t threads_total = (o.count + 3) / 4;
        const unsigned blocks = static_cast<unsigned>((threads_total + 255) / 256);
        add_kernel<<<blocks, 256, 0, exec_stream_>>>(o.ptr, aa.ptr, bb.ptr, o.count);
        return launch_status("cuda add");
    }

    DeviceStatus silu(DeviceTensor &out, const DeviceTensor &x) override {
        auto &o = as_tensor(out);
        const auto &input = as_tensor(x);
        silu_kernel<<<static_cast<unsigned>((o.count + 255) / 256), 256, 0, exec_stream_>>>(o.ptr, input.ptr, o.count);
        return launch_status("cuda silu");
    }

    DeviceStatus mul(DeviceTensor &out, const DeviceTensor &a, const DeviceTensor &b) override {
        auto &o = as_tensor(out);
        const auto &aa = as_tensor(a);
        const auto &bb = as_tensor(b);
        mul_kernel<<<static_cast<unsigned>((o.count + 255) / 256), 256, 0, exec_stream_>>>(o.ptr, aa.ptr, bb.ptr, o.count);
        return launch_status("cuda mul");
    }

    DeviceStatus silu_mul(DeviceTensor &out, const DeviceTensor &gate, const DeviceTensor &up) override {
        auto &o = as_tensor(out);
        const auto &g = as_tensor(gate);
        const auto &u = as_tensor(up);
        // Vectorized: 4 elements per thread.
        const uint64_t threads_total = (o.count + 3) / 4;
        const unsigned blocks = static_cast<unsigned>((threads_total + 255) / 256);
        silu_mul_kernel<<<blocks, 256, 0, exec_stream_>>>(
            o.ptr, g.ptr, u.ptr, o.count);
        return launch_status("cuda silu_mul");
    }

    DeviceStatus recurrent_single_token(DeviceTensor &core,
                                        DeviceTensor &state,
                                        DeviceTensor &conv_state,
                                        DeviceTensor &conv_out,
                                        const DeviceTensor &proj,
                                        const DeviceTensor &gate,
                                        const DeviceTensor &alpha,
                                        const DeviceTensor &beta,
                                        const DeviceWeight &conv,
                                        const DeviceWeight &ssm_a,
                                        const DeviceWeight &dt_bias,
                                        const DeviceWeight &ssm_norm,
                                        uint32_t num_k_heads,
                                        uint32_t num_v_heads,
                                        uint32_t head_k_dim,
                                        uint32_t head_v_dim,
                                        uint32_t conv_kernel_size,
                                        float eps) override {
        auto &c = as_tensor(core);
        auto &s = as_tensor(state);
        auto &cs = as_tensor(conv_state);
        auto &cout = as_tensor(conv_out);
        const auto &p = as_tensor(proj);
        const auto &g = as_tensor(gate);
        const auto &a = as_tensor(alpha);
        const auto &b = as_tensor(beta);
        const auto &cw = as_weight(conv);
        const auto &aw = as_weight(ssm_a);
        const auto &dt = as_weight(dt_bias);
        const auto &nw = as_weight(ssm_norm);
        float *conv_buf = cout.ptr;
        recurrent_conv_kernel<<<static_cast<unsigned>((p.count + 255) / 256), 256, 0, exec_stream_>>>(
            conv_buf, cs.ptr, p.ptr, static_cast<const float *>(cw.ptr),
            static_cast<uint32_t>(p.count), conv_kernel_size);
        l2_norm_128_kernel<<<num_k_heads, 128, 0, exec_stream_>>>(conv_buf, num_k_heads, head_k_dim, eps);
        l2_norm_128_kernel<<<num_k_heads, 128, 0, exec_stream_>>>(conv_buf + num_k_heads * head_k_dim, num_k_heads, head_k_dim, eps);
        dim3 grid(num_v_heads, head_v_dim);
        deltanet_kernel<<<grid, 128, 0, exec_stream_>>>(c.ptr,
                                       s.ptr,
                                       conv_buf,
                                       a.ptr,
                                       b.ptr,
                                       static_cast<const float *>(aw.ptr),
                                       static_cast<const float *>(dt.ptr),
                                       num_k_heads,
                                       num_v_heads,
                                       head_k_dim,
                                       head_v_dim);
        recurrent_norm_gate_kernel<<<num_v_heads, head_v_dim, 0, exec_stream_>>>(c.ptr, g.ptr, static_cast<const float *>(nw.ptr), num_v_heads, head_v_dim, eps);
        return launch_status("cuda recurrent_single_token");
    }

    DeviceStatus recurrent_single_token_at(DeviceTensor &core,
                                            DeviceTensor &state,
                                            DeviceTensor &conv_state,
                                            DeviceTensor &conv_out,
                                            const DeviceTensor &proj,
                                            const DeviceTensor &gate,
                                            const DeviceTensor &alpha,
                                            const DeviceTensor &beta,
                                            const DeviceWeight &conv,
                                            const DeviceWeight &ssm_a,
                                            const DeviceWeight &dt_bias,
                                            const DeviceWeight &ssm_norm,
                                            uint32_t num_k_heads,
                                            uint32_t num_v_heads,
                                            uint32_t head_k_dim,
                                            uint32_t head_v_dim,
                                            uint32_t conv_kernel_size,
                                            uint32_t proj_count,
                                            uint32_t proj_off,
                                            uint32_t gate_off,
                                            uint32_t alpha_off,
                                            uint32_t beta_off,
                                            uint32_t core_off,
                                            float eps) override {
        auto &c = as_tensor(core);
        auto &s = as_tensor(state);
        auto &cs = as_tensor(conv_state);
        auto &cout = as_tensor(conv_out);
        const auto &p = as_tensor(proj);
        const auto &g = as_tensor(gate);
        const auto &a = as_tensor(alpha);
        const auto &b = as_tensor(beta);
        const auto &cw = as_weight(conv);
        const auto &aw = as_weight(ssm_a);
        const auto &dt = as_weight(dt_bias);
        const auto &nw = as_weight(ssm_norm);
        float *conv_buf = cout.ptr;
        recurrent_conv_kernel<<<static_cast<unsigned>((proj_count + 255) / 256), 256, 0, exec_stream_>>>(
            conv_buf, cs.ptr, p.ptr + proj_off, static_cast<const float *>(cw.ptr),
            proj_count, conv_kernel_size);
        l2_norm_128_kernel<<<num_k_heads, 128, 0, exec_stream_>>>(conv_buf, num_k_heads, head_k_dim, eps);
        l2_norm_128_kernel<<<num_k_heads, 128, 0, exec_stream_>>>(conv_buf + num_k_heads * head_k_dim, num_k_heads, head_k_dim, eps);
        dim3 grid(num_v_heads, head_v_dim);
        deltanet_kernel<<<grid, 128, 0, exec_stream_>>>(c.ptr + core_off,
                                       s.ptr,
                                       conv_buf,
                                       a.ptr + alpha_off,
                                       b.ptr + beta_off,
                                       static_cast<const float *>(aw.ptr),
                                       static_cast<const float *>(dt.ptr),
                                       num_k_heads,
                                       num_v_heads,
                                       head_k_dim,
                                       head_v_dim);
        recurrent_norm_gate_kernel<<<num_v_heads, head_v_dim, 0, exec_stream_>>>(c.ptr + core_off, g.ptr + gate_off,
                                                                static_cast<const float *>(nw.ptr),
                                                                num_v_heads, head_v_dim, eps);
        return launch_status("cuda recurrent_single_token_at");
    }

    DeviceStatus recurrent_batch(DeviceTensor &core,
                                  DeviceTensor &state,
                                  DeviceTensor &conv_state,
                                  DeviceTensor &conv_out_buf,
                                  const DeviceTensor &proj,
                                  const DeviceTensor &gate,
                                  const DeviceTensor &alpha,
                                  const DeviceTensor &beta,
                                  const DeviceWeight &conv,
                                  const DeviceWeight &ssm_a,
                                  const DeviceWeight &dt_bias,
                                  const DeviceWeight &ssm_norm,
                                  uint32_t batch,
                                  uint32_t num_k_heads,
                                  uint32_t num_v_heads,
                                  uint32_t head_k_dim,
                                  uint32_t head_v_dim,
                                  uint32_t conv_kernel_size,
                                  uint32_t proj_count,
                                  uint32_t proj_stride,
                                  uint32_t gate_stride,
                                  uint32_t alpha_stride,
                                  uint32_t beta_stride,
                                  uint32_t core_stride,
                                  float eps) override {
        if (batch == 0) return {};
        auto &c = as_tensor(core);
        auto &s = as_tensor(state);
        auto &cs = as_tensor(conv_state);
        auto &cout = as_tensor(conv_out_buf);
        const auto &p = as_tensor(proj);
        const auto &g = as_tensor(gate);
        const auto &a = as_tensor(alpha);
        const auto &b = as_tensor(beta);
        const auto &cw = as_weight(conv);
        const auto &aw = as_weight(ssm_a);
        const auto &dt = as_weight(dt_bias);
        const auto &nw = as_weight(ssm_norm);

        // 1. Conv (batched): one kernel call processes T sequential conv
        //    steps per channel, with the small state window kept in registers.
        //    Output written to cout (row stride = proj_stride to mirror the
        //    proj input layout; valid prefix is proj_count elements/row).
        const unsigned conv_threads = 256;
        const unsigned conv_blocks = static_cast<unsigned>((proj_count + conv_threads - 1) / conv_threads);
        switch (conv_kernel_size) {
            case 3:
                recurrent_conv_batch_kernel<3>
                    <<<conv_blocks, conv_threads, 0, exec_stream_>>>(cout.ptr, cs.ptr, p.ptr,
                                                    static_cast<const float *>(cw.ptr),
                                                    batch, proj_count, proj_stride, proj_stride);
                break;
            case 4:
                recurrent_conv_batch_kernel<4>
                    <<<conv_blocks, conv_threads, 0, exec_stream_>>>(cout.ptr, cs.ptr, p.ptr,
                                                    static_cast<const float *>(cw.ptr),
                                                    batch, proj_count, proj_stride, proj_stride);
                break;
            case 5:
                recurrent_conv_batch_kernel<5>
                    <<<conv_blocks, conv_threads, 0, exec_stream_>>>(cout.ptr, cs.ptr, p.ptr,
                                                    static_cast<const float *>(cw.ptr),
                                                    batch, proj_count, proj_stride, proj_stride);
                break;
            case 7:
                recurrent_conv_batch_kernel<7>
                    <<<conv_blocks, conv_threads, 0, exec_stream_>>>(cout.ptr, cs.ptr, p.ptr,
                                                    static_cast<const float *>(cw.ptr),
                                                    batch, proj_count, proj_stride, proj_stride);
                break;
            default:
                return {false, "recurrent_batch: unsupported conv_kernel_size (expected 3/4/5/7)"};
        }
        if (auto st = launch_status("recurrent_conv_batch_kernel"); !st.ok) return st;

        // 2. L2 norm for Q and K (batched across T). The conv-output buffer
        //    shares the proj layout (row stride = proj_stride).
        dim3 ln_grid(num_k_heads, batch);
        const uint32_t q_off = 0;
        const uint32_t k_off = num_k_heads * head_k_dim;
        l2_norm_128_batch_kernel<<<ln_grid, 128, 0, exec_stream_>>>(cout.ptr, num_k_heads, head_k_dim,
                                                   batch, proj_stride, q_off, eps);
        l2_norm_128_batch_kernel<<<ln_grid, 128, 0, exec_stream_>>>(cout.ptr, num_k_heads, head_k_dim,
                                                   batch, proj_stride, k_off, eps);
        if (auto st = launch_status("l2_norm_128_batch_kernel"); !st.ok) return st;

        // 3. DeltaNet (batched). Two implementations selected at runtime via
        //    QW3_RECURRENT_KERNEL: "qw3" (default) is the original kernel,
        //    "ported" dispatches the warp-shuffle kernel from llama.cpp.
        //    Both produce mathematically equivalent state updates and
        //    outputs; the ported version uses register-resident state and
        //    warp-level reductions, removing __syncthreads() from the inner
        //    loop.
        if (recurrent_kernel_choice() == RecurrentKernel::Ported) {
            // Ported kernel requires head_k_dim == head_v_dim (square state).
            if (head_k_dim != head_v_dim) {
                return {false, "recurrent_batch: ported kernel requires head_k_dim == head_v_dim"};
            }
            // Layout note: cout (conv output) row stride is proj_stride; q
            // starts at offset 0, k at num_k_heads*head_k_dim, v at
            // 2*num_k_heads*head_k_dim. alpha and beta share gb_row_stride
            // (= alpha_stride == beta_stride) per qw3's batched layout.
            if (alpha_stride != beta_stride) {
                return {false, "recurrent_batch: ported kernel requires alpha_stride == beta_stride"};
            }
            // alpha and beta are scratch (overwritten per-layer by the
            // projection step); the prep kernel rewrites them in place to
            // log_g and sigmoid_beta.
            auto &a_mut = const_cast<CudaTensor &>(as_tensor(alpha));
            auto &b_mut = const_cast<CudaTensor &>(as_tensor(beta));
            const uint32_t q_off_floats = 0;
            const uint32_t k_off_floats = num_k_heads * head_k_dim;
            const uint32_t v_off_floats = 2u * num_k_heads * head_k_dim;
            const bool ok = ported::launch_gated_delta_net(
                a_mut.ptr, b_mut.ptr,
                static_cast<const float *>(dt.ptr),
                static_cast<const float *>(aw.ptr),
                cout.ptr,
                q_off_floats, k_off_floats, v_off_floats,
                s.ptr, c.ptr,
                batch,
                num_k_heads, num_v_heads, head_k_dim,
                proj_stride, alpha_stride, core_stride,
                exec_stream_);
            if (!ok) return {false, "recurrent_batch: ported kernel rejected head_dim (only 16/32/64/128)"};
            if (auto st = launch_status("ported::gated_delta_net_kernel"); !st.ok) return st;
        } else {
            // Original qw3 kernel: one block per (vh, head_v_dim) iterates T
            // timesteps; reductions via __syncthreads() in shared memory.
            dim3 dn_grid(num_v_heads, head_v_dim);
            deltanet_batch_kernel<<<dn_grid, 128, 0, exec_stream_>>>(c.ptr,
                                                    s.ptr,
                                                    cout.ptr,
                                                    a.ptr,
                                                    b.ptr,
                                                    static_cast<const float *>(aw.ptr),
                                                    static_cast<const float *>(dt.ptr),
                                                    batch,
                                                    num_k_heads,
                                                    num_v_heads,
                                                    head_k_dim,
                                                    head_v_dim,
                                                    proj_stride,
                                                    alpha_stride,
                                                    beta_stride,
                                                    core_stride);
            if (auto st = launch_status("deltanet_batch_kernel"); !st.ok) return st;
        }

        // 4. RMSnorm + gate, batched over T.
        dim3 ng_grid(num_v_heads, batch);
        recurrent_norm_gate_batch_kernel<<<ng_grid, head_v_dim, 0, exec_stream_>>>(c.ptr, g.ptr,
                                                                  static_cast<const float *>(nw.ptr),
                                                                  batch, num_v_heads, head_v_dim,
                                                                  core_stride, gate_stride, eps);
        return launch_status("recurrent_norm_gate_batch_kernel");
    }

    DeviceStatus zero_tensor(DeviceTensor &x) override {
        auto &t = as_tensor(x);
        return cuda_status(cudaMemsetAsync(t.ptr, 0, static_cast<size_t>(t.count) * t.elem_size,
                                           exec_stream_),
                            "zero_tensor");
    }

    DeviceStatus attention_single_token(DeviceTensor &mid,
                                        DeviceTensor &q,
                                        DeviceTensor &k,
                                        const DeviceTensor &v,
                                        const DeviceWeight &q_norm,
                                        const DeviceWeight &k_norm,
                                        uint32_t n_heads,
                                        uint32_t n_kv_heads,
                                        uint32_t head_dim,
                                        float eps) override {
        auto &m = as_tensor(mid);
        auto &qq = as_tensor(q);
        auto &kk = as_tensor(k);
        const auto &vv = as_tensor(v);
        const auto &qn = as_weight(q_norm);
        const auto &kn = as_weight(k_norm);
        attention_norm_mid_kernel<<<n_heads, head_dim, 0, exec_stream_>>>(m.ptr,
                                                         qq.ptr,
                                                         kk.ptr,
                                                         vv.ptr,
                                                         static_cast<const float *>(qn.ptr),
                                                         static_cast<const float *>(kn.ptr),
                                                         n_heads,
                                                         n_kv_heads,
                                                         head_dim,
                                                         eps);
        return launch_status("cuda attention_single_token");
    }

    DeviceStatus rmsnorm_per_head(DeviceTensor &x,
                                   const DeviceWeight &weight,
                                   uint32_t n_units,
                                   uint32_t per_unit_stride,
                                   uint32_t head_dim,
                                   float eps) override {
        auto &t = as_tensor(x);
        const auto &w = as_weight(weight);
        dim3 grid(n_units, 1);
        rmsnorm_per_head_kernel<<<grid, 256, 0, exec_stream_>>>(t.ptr,
                                                static_cast<const float *>(w.ptr),
                                                n_units, per_unit_stride, head_dim, /*batch_stride=*/0, eps);
        return launch_status("cuda rmsnorm_per_head");
    }

    DeviceStatus rmsnorm_per_head_batch(DeviceTensor &x,
                                         const DeviceWeight &weight,
                                         uint32_t batch,
                                         uint32_t batch_stride,
                                         uint32_t n_units,
                                         uint32_t per_unit_stride,
                                         uint32_t head_dim,
                                         float eps) override {
        if (batch == 0) return {};
        auto &t = as_tensor(x);
        const auto &w = as_weight(weight);
        dim3 grid(n_units, batch);
        rmsnorm_per_head_kernel<<<grid, 256, 0, exec_stream_>>>(t.ptr,
                                                static_cast<const float *>(w.ptr),
                                                n_units, per_unit_stride, head_dim, batch_stride, eps);
        return launch_status("cuda rmsnorm_per_head_batch");
    }

    DeviceStatus rope_partial(DeviceTensor &x,
                               uint32_t n_units,
                               uint32_t per_unit_stride,
                               uint32_t rope_dim,
                               uint32_t pos,
                               float theta) override {
        auto &t = as_tensor(x);
        const uint32_t half = rope_dim / 2;
        if (half == 0) return {};
        dim3 grid(n_units, 1);
        rope_partial_kernel<<<grid, half, 0, exec_stream_>>>(t.ptr, n_units, per_unit_stride, rope_dim, pos, /*batch_stride=*/0, theta);
        return launch_status("cuda rope_partial");
    }

    DeviceStatus rope_partial_batch(DeviceTensor &x,
                                    uint32_t batch,
                                    uint32_t batch_stride,
                                    uint32_t n_units,
                                    uint32_t per_unit_stride,
                                    uint32_t rope_dim,
                                    uint32_t base_pos,
                                    float theta) override {
        if (batch == 0) return {};
        auto &t = as_tensor(x);
        const uint32_t half = rope_dim / 2;
        if (half == 0) return {};
        dim3 grid(n_units, batch);
        rope_partial_kernel<<<grid, half, 0, exec_stream_>>>(t.ptr, n_units, per_unit_stride, rope_dim, base_pos, batch_stride, theta);
        return launch_status("cuda rope_partial_batch");
    }

    DeviceStatus kv_append(DeviceTensor &cache,
                           const DeviceTensor &src,
                           uint32_t pos,
                           uint32_t per_pos_size) override {
        auto &c = as_tensor(cache);
        const auto &s = as_tensor(src);
        const unsigned threads = 256;
        const unsigned blocks = (per_pos_size + threads - 1) / threads;
        dim3 grid(blocks, 1);
        if (c.is_fp16()) {
            kv_append_kernel_f16<<<grid, threads, 0, exec_stream_>>>(c.ptr_h(), s.ptr, pos, per_pos_size, /*src_stride=*/0);
        } else {
            kv_append_kernel<<<grid, threads, 0, exec_stream_>>>(c.ptr, s.ptr, pos, per_pos_size, /*src_stride=*/0);
        }
        return launch_status("cuda kv_append");
    }

    DeviceStatus kv_append_batch(DeviceTensor &cache,
                                 const DeviceTensor &src,
                                 uint32_t base_pos,
                                 uint32_t per_pos_size,
                                 uint32_t batch) override {
        if (batch == 0) return {};
        auto &c = as_tensor(cache);
        const auto &s = as_tensor(src);
        const unsigned threads = 256;
        const unsigned blocks = (per_pos_size + threads - 1) / threads;
        dim3 grid(blocks, batch);
        if (c.is_fp16()) {
            kv_append_kernel_f16<<<grid, threads, 0, exec_stream_>>>(c.ptr_h(), s.ptr, base_pos, per_pos_size, per_pos_size);
        } else {
            kv_append_kernel<<<grid, threads, 0, exec_stream_>>>(c.ptr, s.ptr, base_pos, per_pos_size, per_pos_size);
        }
        return launch_status("cuda kv_append_batch");
    }

    DeviceStatus ensure_fattn_scratch(uint32_t n_heads, uint32_t batch,
                                       uint32_t head_dim, uint32_t seq_len) {
        const size_t need = ported::fattn_vec_scratch_bytes(n_heads, batch,
                                                             head_dim, seq_len);
        if (need == 0) return {};
        if (need > fattn_scratch_capacity_) {
            if (fattn_scratch_) cudaFree(fattn_scratch_);
            if (auto st = cuda_status(cudaMalloc(&fattn_scratch_, need),
                                      "fattn scratch alloc"); !st.ok) {
                fattn_scratch_ = nullptr;
                fattn_scratch_capacity_ = 0;
                return st;
            }
            fattn_scratch_capacity_ = need;
        }
        return {};
    }

    // Allocates Q-FP16 staging and scores-FP16 scratch for the cuBLAS prefill
    // path. Sized to the largest (T, T_kv, n_heads, head_dim) seen so far;
    // grows monotonically. Q scratch holds the full T*n_heads*d FP16 copy of
    // the FP32 Q tensor (~53 MB at T=4350 for Qwen 3.6 27B). Scores scratch
    // holds gqa_ratio*T*T_kv FP16 elements reused across all kv_groups (~230
    // MB at T=T_kv=4350). The scores buffer dominates; if it grows past
    // QW3_PREFILL_ATTN_MAX_BYTES (default 512 MB), we refuse the cuBLAS path
    // and the caller falls back to the tiled kernel.
    DeviceStatus ensure_prefill_attn_scratch(uint32_t T, uint32_t T_kv,
                                              uint32_t n_heads, uint32_t gqa_ratio,
                                              uint32_t head_dim) {
        const uint64_t need_q = static_cast<uint64_t>(T) * n_heads * head_dim;
        if (need_q > prefill_attn_q_fp16_capacity_) {
            if (prefill_attn_q_fp16_) cudaFree(prefill_attn_q_fp16_);
            if (auto st = cuda_status(cudaMalloc(&prefill_attn_q_fp16_,
                                                 need_q * sizeof(__half)),
                                      "prefill_attn q_fp16 alloc"); !st.ok) {
                prefill_attn_q_fp16_ = nullptr;
                prefill_attn_q_fp16_capacity_ = 0;
                return st;
            }
            prefill_attn_q_fp16_capacity_ = need_q;
        }
        const uint64_t need_s = static_cast<uint64_t>(gqa_ratio) * T * T_kv;
        if (need_s > prefill_attn_scores_fp16_capacity_) {
            if (prefill_attn_scores_fp16_) cudaFree(prefill_attn_scores_fp16_);
            if (auto st = cuda_status(cudaMalloc(&prefill_attn_scores_fp16_,
                                                 need_s * sizeof(__half)),
                                      "prefill_attn scores_fp16 alloc"); !st.ok) {
                prefill_attn_scores_fp16_ = nullptr;
                prefill_attn_scores_fp16_capacity_ = 0;
                return st;
            }
            prefill_attn_scores_fp16_capacity_ = need_s;
        }
        return {};
    }

    // cuBLAS-based prefill attention. For each KV group g (gqa_ratio query
    // heads share one KV head), runs:
    //   1) S_g = Q_g · K_g^T            (cuBLAS strided-batched HGEMM, batch=ratio)
    //   2) P_g = softmax(scale*S_g) w/ causal mask    (one custom kernel)
    //   3) O_g = P_g · V_g              (cuBLAS strided-batched HGEMM, batch=ratio)
    //
    // Layouts (row-major, as the rest of the stack uses):
    //   Q       : [T, n_heads, d]              FP32 in -> FP16 staged
    //   K, V    : [T_kv, n_kv_heads, d]        FP16
    //   O       : [T, n_heads, d]              FP32
    //   S, P    : [n_kv_heads, ratio, T, T_kv] FP16 (group-major; scratch reused per group)
    //
    // cuBLAS is column-major, so each row-major matmul is reinterpreted as
    // its transpose: row-major C[m,n] = A[m,k]·B[k,n] becomes col-major
    // C^T[n,m] = B^T[n,k]·A^T[k,m], i.e. cublas(opA=N, opB=N, m=n, n=m, k=k,
    // A=B_ptr, B=A_ptr). The strides we pass to cuBLAS are then plain row-
    // strides of the original row-major buffers.
    DeviceStatus attention_prefill_cublas(float *out_ptr,
                                           const float *q_ptr,
                                           uint32_t q_stride,
                                           const __half *k_ptr,
                                           const __half *v_ptr,
                                           uint32_t n_heads,
                                           uint32_t n_kv_heads,
                                           uint32_t head_dim,
                                           uint32_t T,
                                           uint32_t base_seq_len,
                                           uint32_t q_batch_stride,
                                           uint32_t out_batch_stride,
                                           float scale) {
        if (n_heads % n_kv_heads != 0) {
            return {false, "attention_prefill_cublas requires n_heads % n_kv_heads == 0"};
        }
        const uint32_t gqa = n_heads / n_kv_heads;
        const uint32_t T_kv = base_seq_len + T;
        if (T_kv == 0) return {};

        // q_ptr is row-major [T, n_heads, d] with row stride q_batch_stride
        // (in floats) and inner stride q_stride (= n_heads*d in our stack).
        // We need it contiguous as [T, n_heads*d] for the FP32→FP16 stage.
        // Caller passes q_batch_stride == n_heads*d for the prefill batch.
        if (q_batch_stride != n_heads * head_dim) {
            return {false, "attention_prefill_cublas requires contiguous Q per query"};
        }
        if (q_stride != n_heads * head_dim) {
            return {false, "attention_prefill_cublas requires q_stride == n_heads*head_dim"};
        }
        if (out_batch_stride != n_heads * head_dim) {
            return {false, "attention_prefill_cublas requires contiguous O per query"};
        }

        if (auto st = ensure_prefill_attn_scratch(T, T_kv, n_heads, gqa, head_dim);
            !st.ok) return st;

        // Stage Q to FP16. Source is contiguous T*n_heads*d FP32.
        {
            const uint64_t n = static_cast<uint64_t>(T) * n_heads * head_dim;
            const unsigned threads = 256;
            const uint64_t threads_total = (n + 3) / 4;
            const unsigned blocks = static_cast<unsigned>((threads_total + threads - 1) / threads);
            fp32_to_fp16_kernel<<<blocks, threads, 0, exec_stream_>>>(
                prefill_attn_q_fp16_, q_ptr, n);
            if (auto st = launch_status("prefill_attn fp32->fp16 Q"); !st.ok) return st;
        }

        const float falpha = 1.0f;
        const float fzero  = 0.0f;
        const int d   = static_cast<int>(head_dim);
        const int t   = static_cast<int>(T);
        const int tkv = static_cast<int>(T_kv);

        for (uint32_t g = 0; g < n_kv_heads; ++g) {
            // Q for this group: ratio heads, each [T, d] FP16, head-stride d,
            // query-stride n_heads*d. Pointer to first head of group g:
            const __half *Qg = prefill_attn_q_fp16_
                             + static_cast<uint64_t>(g) * gqa * head_dim;
            // K/V for this group: one head shared across ratio queries,
            // shape [T_kv, d] FP16, row stride n_kv_heads*d, head pointer:
            const __half *Kg = k_ptr + static_cast<uint64_t>(g) * head_dim;
            const __half *Vg = v_ptr + static_cast<uint64_t>(g) * head_dim;
            __half *Sg = prefill_attn_scores_fp16_;  // [ratio, T, T_kv] FP16

            // ---- 1) S_h = Q_h · K^T  for h in [0, ratio) ----
            // Row-major:  C_rm[T, T_kv] = Q_rm[T, d] · (K_rm[T_kv, d])^T.
            // cuBLAS is col-major; produce C_cm[j,i] = C_rm[i,j] by passing
            // (A=K, opA=T) and (B=Q, opB=N), m=T_kv, n=T, k=d:
            //   op(A)[j,k] = K_rm[j,k] (via opA=T over col-major K, lda=n_kv*d)
            //   op(B)[k,i] = Q_rm[i,k] (via opB=N over col-major Q, ldb=n_h*d)
            // strideA = 0 (K shared), strideB = head_dim (next Q head),
            // strideC = T*T_kv (next score plane), batch = ratio.
            // FP32 accumulator (FAST_16F): K can be 4K+ on long contexts and
            // FP16 accumulation drifts enough to flip greedy tokens. Same
            // tradeoff the HGEMM path made — kept tensor cores, lost 16F-acc.
            if (auto st = cublas_status(
                    cublasGemmStridedBatchedEx(
                        cublas_handle_,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        tkv, t, d,
                        &falpha,
                        Kg, CUDA_R_16F, static_cast<int>(n_kv_heads * head_dim), 0,
                        Qg, CUDA_R_16F, static_cast<int>(n_heads * head_dim),
                            static_cast<long long>(head_dim),
                        &fzero,
                        Sg, CUDA_R_16F, tkv,
                            static_cast<long long>(static_cast<uint64_t>(t) * tkv),
                        static_cast<int>(gqa),
                        CUBLAS_COMPUTE_32F_FAST_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "prefill_attn QK^T"); !st.ok) return st;

            // ---- 2) softmax(scale * S_h) with causal mask, in-place ----
            //   grid (T, ratio), block 256 threads, one row per (head, q).
            {
                dim3 grid(T, gqa);
                softmax_causal_inplace_kernel<<<grid, 256, 0, exec_stream_>>>(
                    Sg, T, T_kv, base_seq_len, gqa, scale);
                if (auto st = launch_status("prefill_attn softmax"); !st.ok) return st;
            }

            // ---- 3) O_h = P_h · V  for h in [0, ratio) ----
            // Row-major:  C_rm[T, d] = P_rm[T, T_kv] · V_rm[T_kv, d].
            // cuBLAS col-major: C_cm[j,i] = C_rm[i,j] via (A=V, opA=N),
            // (B=P, opB=N), m=d, n=T, k=T_kv:
            //   op(A)[j,k] = V_rm[k,j] (opA=N, V col-major, lda=n_kv*d)
            //   op(B)[k,i] = P_rm[i,k] (opB=N, P col-major, ldb=T_kv)
            // strideA = 0 (V shared), strideB = T*T_kv, strideC = head_dim,
            // batch = ratio.
            //
            // ldc = n_heads*head_dim so successive q rows step by full row of
            // row-major O; pointer offset g*ratio*head_dim places this group's
            // ratio heads at columns [g*ratio*d, (g+1)*ratio*d) of O_rm.
            // Compute=32F_FAST_16F to keep tensor cores on FP16 inputs while
            // accumulating FP32 into the FP32 output buffer.
            float *Og = out_ptr + static_cast<uint64_t>(g) * gqa * head_dim;
            if (auto st = cublas_status(
                    cublasGemmStridedBatchedEx(
                        cublas_handle_,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        d, t, tkv,
                        &falpha,
                        Vg, CUDA_R_16F, static_cast<int>(n_kv_heads * head_dim), 0,
                        Sg, CUDA_R_16F, tkv,
                            static_cast<long long>(static_cast<uint64_t>(t) * tkv),
                        &fzero,
                        Og, CUDA_R_32F, static_cast<int>(n_heads * head_dim),
                            static_cast<long long>(head_dim),
                        static_cast<int>(gqa),
                        CUBLAS_COMPUTE_32F_FAST_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                    "prefill_attn PV"); !st.ok) return st;
        }
        return {};
    }

    DeviceStatus attention_decode(DeviceTensor &out,
                                   DeviceTensor &scores_scratch,
                                   const DeviceTensor &q,
                                   uint32_t q_stride,
                                   const DeviceTensor &k_cache,
                                   const DeviceTensor &v_cache,
                                   uint32_t n_heads,
                                   uint32_t n_kv_heads,
                                   uint32_t head_dim,
                                   uint32_t seq_len,
                                   float scale) override {
        auto &o = as_tensor(out);
        const auto &qq = as_tensor(q);
        const auto &kc = as_tensor(k_cache);
        const auto &vc = as_tensor(v_cache);
        const bool kv_fp16 = kc.is_fp16();
        // FP16 K/V: only the ported path supports half-precision K/V; the
        // legacy F32 fused/qk-softmax-av kernels are F32-only.
        if (kv_fp16 && (head_dim == 128 || head_dim == 256)) {
            if (auto st = ensure_fattn_scratch(n_heads, /*batch=*/1, head_dim, seq_len);
                !st.ok) return st;
            if (ported::launch_fattn_vec_decode_f16_splitk(
                    o.ptr, fattn_scratch_, qq.ptr, q_stride, kc.ptr, vc.ptr,
                    n_heads, n_kv_heads, head_dim, seq_len, /*batch=*/1,
                    /*q_batch_stride=*/0, /*out_batch_stride=*/0, scale,
                    exec_stream_)) {
                return launch_status("cuda attn ported f16 splitk");
            }
            return {false, "fattn_vec f16 splitk launcher refused head_dim"};
        }
        // Ported flash-attention-style decode. Selected via QW3_ATTN=ported.
        // 4-warp parallelism across KV tokens — wins on long context.
        if (attention_kernel_choice() == AttentionKernel::Ported &&
            (head_dim == 128 || head_dim == 256)) {
            if (auto st = ensure_fattn_scratch(n_heads, /*batch=*/1, head_dim, seq_len);
                !st.ok) return st;
            if (ported::launch_fattn_vec_decode_f32_splitk(
                    o.ptr, fattn_scratch_, qq.ptr, q_stride, kc.ptr, vc.ptr,
                    n_heads, n_kv_heads, head_dim, seq_len, /*batch=*/1,
                    /*q_batch_stride=*/0, /*out_batch_stride=*/0, scale,
                    exec_stream_)) {
                return launch_status("cuda attn ported splitk");
            }
        }
        if (head_dim == 128) {
            // Fused flash-attention-style decode: 1 kernel, no scores materialization.
            dim3 grid(n_heads, 1);
            attention_decode_fused_kernel<128, 64><<<grid, 128, 0, exec_stream_>>>(
                o.ptr, qq.ptr, q_stride, kc.ptr, vc.ptr,
                n_heads, n_kv_heads, /*base_seq_len=*/seq_len - 1,
                /*q_batch_stride=*/0, /*out_batch_stride=*/0, scale);
            return launch_status("cuda attn fused");
        }
        if (head_dim == 256) {
            dim3 grid(n_heads, 1);
            attention_decode_fused_kernel<256, 64><<<grid, 256, 0, exec_stream_>>>(
                o.ptr, qq.ptr, q_stride, kc.ptr, vc.ptr,
                n_heads, n_kv_heads, /*base_seq_len=*/seq_len - 1,
                /*q_batch_stride=*/0, /*out_batch_stride=*/0, scale);
            return launch_status("cuda attn fused 256");
        }
        // Fallback for unexpected head dims.
        auto &s = as_tensor(scores_scratch);
        dim3 grid_qk(n_heads, seq_len);
        attention_decode_qk_kernel<<<grid_qk, 256, 0, exec_stream_>>>(s.ptr, qq.ptr, q_stride, kc.ptr,
                                                    n_heads, n_kv_heads, head_dim, seq_len, scale);
        if (auto st = launch_status("cuda attn qk"); !st.ok) return st;
        attention_decode_softmax_kernel<<<n_heads, 256, 0, exec_stream_>>>(s.ptr, n_heads, seq_len);
        if (auto st = launch_status("cuda attn softmax"); !st.ok) return st;
        attention_decode_av_kernel<<<n_heads, head_dim, 0, exec_stream_>>>(o.ptr, s.ptr, vc.ptr,
                                                         n_heads, n_kv_heads, head_dim, seq_len);
        return launch_status("cuda attn av");
    }

    DeviceStatus apply_attn_gate(DeviceTensor &out,
                                  const DeviceTensor &q,
                                  uint32_t q_stride,
                                  uint32_t n_heads,
                                  uint32_t head_dim) override {
        auto &o = as_tensor(out);
        const auto &qq = as_tensor(q);
        const uint64_t total = static_cast<uint64_t>(n_heads) * head_dim;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
        dim3 grid(blocks, 1);
        apply_attn_gate_kernel<<<grid, threads, 0, exec_stream_>>>(o.ptr, qq.ptr, q_stride, n_heads, head_dim, 0, 0);
        return launch_status("cuda apply_attn_gate");
    }

    DeviceStatus apply_attn_gate_batch(DeviceTensor &out,
                                        const DeviceTensor &q,
                                        uint32_t q_stride,
                                        uint32_t batch,
                                        uint32_t batch_stride_q,
                                        uint32_t batch_stride_out,
                                        uint32_t n_heads,
                                        uint32_t head_dim) override {
        if (batch == 0) return {};
        auto &o = as_tensor(out);
        const auto &qq = as_tensor(q);
        const uint64_t total = static_cast<uint64_t>(n_heads) * head_dim;
        const unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
        dim3 grid(blocks, batch);
        apply_attn_gate_kernel<<<grid, threads, 0, exec_stream_>>>(o.ptr, qq.ptr, q_stride, n_heads, head_dim,
                                                  batch_stride_q, batch_stride_out);
        return launch_status("cuda apply_attn_gate_batch");
    }

    DeviceStatus attention_decode_batch(DeviceTensor &out,
                                         const DeviceTensor &q,
                                         uint32_t q_stride,
                                         const DeviceTensor &k_cache,
                                         const DeviceTensor &v_cache,
                                         uint32_t n_heads,
                                         uint32_t n_kv_heads,
                                         uint32_t head_dim,
                                         uint32_t base_seq_len,
                                         uint32_t batch,
                                         uint32_t q_batch_stride,
                                         uint32_t out_batch_stride,
                                         float scale) override {
        if (batch == 0) return {};
        auto &o = as_tensor(out);
        const auto &qq = as_tensor(q);
        const auto &kc = as_tensor(k_cache);
        const auto &vc = as_tensor(v_cache);
        const bool kv_fp16 = kc.is_fp16();

        // cuBLAS prefill: two strided-batched FP16 GEMMs + in-place causal
        // softmax. Default for batch >= prefill_attn_min_batch on FP16 KV.
        // Tensor-core throughput closes the 14×/call gap of the FP32 tiled
        // kernel on long contexts. Requires contiguous Q (q_batch_stride ==
        // n_heads*d) which is what the prefill caller passes.
        if (kv_fp16 &&
            prefill_attn_kernel_choice() == PrefillAttnKernel::Cublas &&
            batch >= prefill_attn_min_batch() &&
            (head_dim == 128 || head_dim == 256) &&
            q_batch_stride == n_heads * head_dim &&
            out_batch_stride == n_heads * head_dim &&
            q_stride == n_heads * head_dim) {
            if (auto st = attention_prefill_cublas(
                    o.ptr, qq.ptr, q_stride, kc.ptr_h(), vc.ptr_h(),
                    n_heads, n_kv_heads, head_dim,
                    /*T=*/batch, base_seq_len,
                    q_batch_stride, out_batch_stride, scale);
                st.ok) {
                return launch_status("cuda attention_decode_batch cublas prefill");
            }
            // On unexpected failure, fall through to the tiled/vec paths.
        }

        // Tensor-core MMA prefill (m16n8k16.f16). Opt-in via QW3_PREFILL_ATTN=mma.
        if (kv_fp16 &&
            prefill_attn_kernel_choice() == PrefillAttnKernel::Mma &&
            batch >= prefill_attn_min_batch() &&
            (head_dim == 128 || head_dim == 256)) {
            if (ported::launch_fattn_prefill_mma_f16(
                    o.ptr, qq.ptr, q_stride, kc.ptr, vc.ptr,
                    n_heads, n_kv_heads, head_dim,
                    batch, base_seq_len,
                    q_batch_stride, out_batch_stride, scale, exec_stream_)) {
                return launch_status("cuda attention_decode_batch mma f16");
            }
        }

        // Tiled prefill kernel — preferred when batch is large enough that
        // K/V reuse across BR=4 queries pays for itself. FP16 KV only (FP32
        // would blow the 48 KB static-shmem cap). Below the threshold the
        // per-query split-K vec kernel wins (more SM occupancy per query).
        if (kv_fp16 &&
            prefill_attn_kernel_choice() == PrefillAttnKernel::Tiled &&
            batch >= prefill_attn_min_batch() &&
            (head_dim == 128 || head_dim == 256)) {
            if (ported::launch_fattn_prefill_f16(
                    o.ptr, qq.ptr, q_stride, kc.ptr, vc.ptr,
                    n_heads, n_kv_heads, head_dim,
                    batch, base_seq_len,
                    q_batch_stride, out_batch_stride, scale, exec_stream_)) {
                return launch_status("cuda attention_decode_batch tiled f16");
            }
        }

        if (kv_fp16 && (head_dim == 128 || head_dim == 256)) {
            if (auto st = ensure_fattn_scratch(n_heads, batch, head_dim,
                                                /*seq_len=*/base_seq_len + batch);
                !st.ok) return st;
            if (ported::launch_fattn_vec_decode_f16_splitk(
                    o.ptr, fattn_scratch_, qq.ptr, q_stride, kc.ptr, vc.ptr,
                    n_heads, n_kv_heads, head_dim,
                    /*seq_len=*/base_seq_len + 1, batch,
                    q_batch_stride, out_batch_stride, scale, exec_stream_)) {
                return launch_status("cuda attention_decode_batch ported f16 splitk");
            }
            return {false, "fattn_vec f16 splitk launcher refused head_dim"};
        }
        // Ported path also covers the per-batch case. Each batch position has
        // seq_len = base_seq_len + b + 1 (causal), which the kernel handles.
        if (attention_kernel_choice() == AttentionKernel::Ported &&
            (head_dim == 128 || head_dim == 256)) {
            if (auto st = ensure_fattn_scratch(n_heads, batch, head_dim,
                                                /*seq_len=*/base_seq_len + batch);
                !st.ok) return st;
            if (ported::launch_fattn_vec_decode_f32_splitk(
                    o.ptr, fattn_scratch_, qq.ptr, q_stride, kc.ptr, vc.ptr,
                    n_heads, n_kv_heads, head_dim,
                    /*seq_len=*/base_seq_len + 1, batch,
                    q_batch_stride, out_batch_stride, scale, exec_stream_)) {
                return launch_status("cuda attention_decode_batch ported splitk");
            }
        }
        dim3 grid(n_heads, batch);
        if (head_dim == 128) {
            attention_decode_fused_kernel<128, 64><<<grid, 128, 0, exec_stream_>>>(
                o.ptr, qq.ptr, q_stride, kc.ptr, vc.ptr,
                n_heads, n_kv_heads, base_seq_len,
                q_batch_stride, out_batch_stride, scale);
            return launch_status("cuda attention_decode_batch 128");
        }
        if (head_dim == 256) {
            attention_decode_fused_kernel<256, 64><<<grid, 256, 0, exec_stream_>>>(
                o.ptr, qq.ptr, q_stride, kc.ptr, vc.ptr,
                n_heads, n_kv_heads, base_seq_len,
                q_batch_stride, out_batch_stride, scale);
            return launch_status("cuda attention_decode_batch 256");
        }
        return {false, "attention_decode_batch unsupported head_dim"};
    }

    DeviceArgmax argmax(const DeviceTensor &x) override {
        const auto &t = as_tensor(x);
        if (auto st = argmax_launch(x); !st.ok) {
            return {};
        }
        // Sync immediately so callers that don't use the split-phase API still
        // observe a fully-populated DeviceArgmax.
        cudaStreamSynchronize(exec_stream_);
        DeviceArgmax best;
        best.token = argmax_host_[0];
        float logit;
        std::memcpy(&logit, &argmax_host_[1], sizeof(float));
        best.logit = logit;
        (void)t;
        return best;
    }

    DeviceStatus argmax_launch(const DeviceTensor &x) override {
        const auto &t = as_tensor(x);
        if (!argmax_dev_) {
            cudaMalloc(&argmax_dev_, 2 * sizeof(int32_t));
        }
        if (!argmax_host_) {
            cudaHostAlloc(reinterpret_cast<void **>(&argmax_host_),
                          2 * sizeof(int32_t), cudaHostAllocDefault);
        }
        argmax_kernel<<<1, 1024, 0, exec_stream_>>>(argmax_dev_, t.ptr, t.count);
        cudaMemcpyAsync(argmax_host_, argmax_dev_, 2 * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, exec_stream_);
        return launch_status("cuda argmax_launch");
    }

    DeviceArgmax argmax_collect() override {
        cudaStreamSynchronize(exec_stream_);
        DeviceArgmax best;
        if (!argmax_host_) return best;
        best.token = argmax_host_[0];
        float logit;
        std::memcpy(&logit, &argmax_host_[1], sizeof(float));
        best.logit = logit;
        return best;
    }

    DeviceStatus copy_to_host(const DeviceTensor &x, float *host, uint64_t offset, uint64_t count) override {
        const auto &t = as_tensor(x);
        if (offset + count > t.count) return {false, "copy_to_host out of range"};
        if (auto st = cuda_status(cudaMemcpyAsync(host,
                                                   t.ptr + offset,
                                                   static_cast<size_t>(count) * sizeof(float),
                                                   cudaMemcpyDeviceToHost,
                                                   exec_stream_),
                                   "copy_to_host async");
            !st.ok) return st;
        return cuda_status(cudaStreamSynchronize(exec_stream_), "copy_to_host sync");
    }

    DeviceStatus copy_d2d(DeviceTensor &dst,
                          const DeviceTensor &src,
                          uint64_t src_offset,
                          uint64_t count) override {
        auto &d = as_tensor(dst);
        const auto &s = as_tensor(src);
        if (src_offset + count > s.count) return {false, "copy_d2d src oob"};
        if (count > d.count) return {false, "copy_d2d dst oob"};
        return cuda_status(cudaMemcpyAsync(d.ptr,
                                            s.ptr + src_offset,
                                            static_cast<size_t>(count) * sizeof(float),
                                            cudaMemcpyDeviceToDevice,
                                            exec_stream_),
                            "copy_d2d");
    }

private:
    LinearBackend linear_backend_ = LinearBackend::Auto;
    cublasHandle_t cublas_handle_ = nullptr;
    uint64_t *rows_buf_ = nullptr;
    uint32_t  rows_buf_capacity_ = 0;
    // Input-side FP16 staging buffer for HGEMM (X is FP32 in our stack but
    // cuBLAS HGEMM needs FP16). Resized lazily to fit the largest (batch *
    // cols) matmul seen so far.
    __half *x_fp16_workspace_ = nullptr;
    uint64_t x_fp16_capacity_ = 0;  // elements
    // Weight-side FP16 dequant scratch for HGEMM. Two ping-pong buffers, each
    // sized to the largest single Q8_0 weight (~150 MB on Qwen 3.6 27B). The
    // dequant of W_{n+1} runs on `dequant_stream_` concurrently with the
    // HGEMM of W_n on the cuBLAS stream — events on `hgemm_done_` /
    // `dequant_done_` synchronize the read/write of each buffer.
    __half *w_fp16_workspace_[2] = {nullptr, nullptr};
    uint64_t w_fp16_capacity_ = 0;          // elements (per buffer)
    int      w_fp16_idx_ = 0;               // ping-pong index for next dequant
    cudaStream_t dequant_stream_ = nullptr;
    cudaEvent_t  dequant_done_[2] = {nullptr, nullptr};
    cudaEvent_t  hgemm_done_[2]   = {nullptr, nullptr};
    // Q8_1 staging buffer for the ported mmvq path. Reused across calls;
    // grows on demand. Bytes hold (batch * blocks_per_row) block_q8_1 = 36 B.
    void   *q8_1_scratch_ = nullptr;
    size_t  q8_1_scratch_capacity_ = 0;  // bytes
    void   *q8_1_mmq_scratch_ = nullptr;
    size_t  q8_1_mmq_scratch_capacity_ = 0;  // bytes
    // Split-K attention scratch. Holds VKQ partials and (max,sum) tuples for
    // the fattn-vec decode kernel when seq_len is large enough that we run
    // multiple blocks per (head, batch). Reused across all attention layers
    // because the kernel writes-then-reads it in a single launch+combine pair.
    void   *fattn_scratch_ = nullptr;
    size_t  fattn_scratch_capacity_ = 0;  // bytes
    // Prefill-attention cuBLAS path scratch. q_fp16 stages a half-precision
    // copy of the FP32 Q tensor (cuBLAS HGEMM input); scores_fp16 holds one
    // KV-group worth of attention scores [gqa_ratio, T, T_kv] reused across
    // groups within a single layer call. Reused across layers in a forward.
    __half *prefill_attn_q_fp16_ = nullptr;
    uint64_t prefill_attn_q_fp16_capacity_ = 0;       // elements
    __half *prefill_attn_scores_fp16_ = nullptr;
    uint64_t prefill_attn_scores_fp16_capacity_ = 0;  // elements
    // X-FP16 cache for hgemm_q8. The QKV and gate/up matmul pairs share the
    // same FP32 input pointer in immediate succession; without a cache we
    // re-run fp32_to_fp16_kernel on identical data. Cache key is
    // (x_ptr, x_elems); a hit skips the conversion. Reset on begin() and on
    // workspace resize. Disable via QW3_HGEMM_X_CACHE=0.
    const float *x_fp16_cached_src_ = nullptr;
    uint64_t     x_fp16_cached_elems_ = 0;
    // Single execution stream for all kernel launches. Created lazily in
    // begin(). Threading every launch through this stream (instead of stream
    // 0) is a prerequisite for CUDA graph capture and lets cuBLAS/HGEMM
    // pipelines coexist with the rest of decode without forced syncs on the
    // legacy default stream.
    cudaStream_t exec_stream_ = nullptr;
    // CUDA-graph state for the captured per-token decode forward.
    // graph_instance_ is the executable graph; capture_active_ tracks whether
    // we're currently between begin_capture/end_capture; graph_disabled_ is a
    // sticky bit that latches on first capture failure so we don't retry.
    cudaGraphExec_t graph_instance_ = nullptr;
    bool capture_active_ = false;
    bool graph_disabled_ = false;
    // Device-side argmax output (token, logit) and a pinned-host mirror.
    // Avoids the synchronous full-logits cudaMemcpy + host scan that the old
    // path did per token.
    int32_t *argmax_dev_ = nullptr;     // [token (int32), logit (float as int32 bits)]
    int32_t *argmax_host_ = nullptr;    // pinned host mirror, 2 ints
};

} // namespace

bool cuda_device_backend_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<DeviceBackend> make_cuda_device_backend(LinearBackend linear_backend) {
    return std::make_unique<CudaDeviceBackend>(linear_backend);
}

} // namespace qw3
