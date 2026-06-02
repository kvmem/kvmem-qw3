#include "flashinfer_prefill_adapter.hpp"

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

#include <cstdint>

namespace qw3 {
namespace flashinfer_adapter {
namespace {

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half(value);
}

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
    return __half2float(value);
}

template <typename T>
__global__ void pack_q_kernel(T *q_packed,
                              const float *q,
                              uint32_t q_stride,
                              uint32_t batch,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              uint32_t q_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint64_t tmp = i / head_dim;
    const uint32_t h = static_cast<uint32_t>(tmp % n_heads);
    const uint32_t b = static_cast<uint32_t>(tmp / n_heads);
    q_packed[i] = from_float<T>(q[static_cast<uint64_t>(b) * q_batch_stride
                                 + static_cast<uint64_t>(h) * q_stride + d]);
}

// Gate-fused unpack: out = sigmoid(gate) * o. qw3_ly path.
template <typename T>
__global__ void unpack_gate_kernel(float *out,
                                   const T *o_packed,
                                   const float *q,
                                   uint32_t q_stride,
                                   uint32_t batch,
                                   uint32_t n_heads,
                                   uint32_t head_dim,
                                   uint32_t q_batch_stride,
                                   uint32_t out_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint64_t tmp = i / head_dim;
    const uint32_t h = static_cast<uint32_t>(tmp % n_heads);
    const uint32_t b = static_cast<uint32_t>(tmp / n_heads);
    const uint64_t q_off = static_cast<uint64_t>(b) * q_batch_stride
                         + static_cast<uint64_t>(h) * q_stride;
    const float value = to_float<T>(o_packed[i]);
    const float gate = q[q_off + head_dim + d];
    out[static_cast<uint64_t>(b) * out_batch_stride
        + static_cast<uint64_t>(h) * head_dim + d] =
        value * (1.0f / (1.0f + expf(-gate)));
}

// Plain unpack: out = fp16 -> fp32. Caller still applies the sigmoid gate.
template <typename T>
__global__ void unpack_kernel(float *out,
                              const T *o_packed,
                              uint32_t batch,
                              uint32_t n_heads,
                              uint32_t head_dim,
                              uint32_t out_batch_stride) {
    const uint64_t total = static_cast<uint64_t>(batch) * n_heads * head_dim;
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    const uint32_t d = static_cast<uint32_t>(i % head_dim);
    const uint64_t tmp = i / head_dim;
    const uint32_t h = static_cast<uint32_t>(tmp % n_heads);
    const uint32_t b = static_cast<uint32_t>(tmp / n_heads);
    out[static_cast<uint64_t>(b) * out_batch_stride
        + static_cast<uint64_t>(h) * head_dim + d] = to_float<T>(o_packed[i]);
}

// __FA_PORT_TAIL__
template <typename TQ, typename TKV, typename TO>
bool run_prefill_typed(
        TQ *q_packed,
        TO *o_packed,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride,
        float scale, cudaStream_t stream,
        unsigned &threads_out, unsigned &blocks_out) {
    if (head_dim != 256) return false;
    if (n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0) return false;
    threads_out = 256;
    if (batch == 0) {
        blocks_out = 0;
        return true;
    }
    const uint64_t q_elems = static_cast<uint64_t>(batch) * n_heads * head_dim;
    blocks_out = static_cast<unsigned>((q_elems + threads_out - 1) / threads_out);
    pack_q_kernel<TQ><<<blocks_out, threads_out, 0, stream>>>(
        q_packed, q, q_stride, batch, n_heads, head_dim, q_batch_stride);
    if (cudaGetLastError() != cudaSuccess) return false;

    using Params = flashinfer::SinglePrefillParams<TQ, TKV, TO>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    Params params(
        q_packed,
        const_cast<TKV *>(static_cast<const TKV *>(k_cache)),
        const_cast<TKV *>(static_cast<const TKV *>(v_cache)),
        nullptr, o_packed, nullptr, nullptr,
        n_heads, n_kv_heads, batch, base_seq_len + batch,
        n_heads * head_dim, head_dim,
        n_kv_heads * head_dim, head_dim, head_dim,
        -1, 0.0f, scale, 1.0f, 1.0f);
    cudaError_t st = flashinfer::SinglePrefillWithKVCacheDispatched<
        256, 256,
        flashinfer::PosEncodingMode::kNone,
        false,
        flashinfer::MaskMode::kCausal,
        Variant>(params, nullptr, stream);
    return st == cudaSuccess;
}

} // namespace

bool launch_prefill_f16q_f16kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    if (!run_prefill_typed<half, half, half>(
            q_f16, o_f16, q, q_stride, k_cache, v_cache,
            n_heads, n_kv_heads, head_dim, batch, base_seq_len,
            q_batch_stride, scale, stream, threads, blocks)) {
        return false;
    }
    if (batch == 0) return true;
    unpack_gate_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, q, q_stride, batch, n_heads, head_dim,
        q_batch_stride, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_prefill_f16q_f16kv(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream) {
    unsigned threads = 0, blocks = 0;
    if (!run_prefill_typed<half, half, half>(
            q_f16, o_f16, q, q_stride, k_cache, v_cache,
            n_heads, n_kv_heads, head_dim, batch, base_seq_len,
            q_batch_stride, scale, stream, threads, blocks)) {
        return false;
    }
    if (batch == 0) return true;
    unpack_kernel<half><<<blocks, threads, 0, stream>>>(
        out, o_f16, batch, n_heads, head_dim, out_batch_stride);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace flashinfer_adapter
} // namespace qw3
