#include "flashinfer_vision_adapter.hpp"

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

namespace qw3::flashinfer_vision_adapter {

bool launch_bf16(
        __nv_bfloat16 *out,
        const __nv_bfloat16 *q,
        const __nv_bfloat16 *k,
        const __nv_bfloat16 *v,
        const uint32_t *cu_seqlens,
        uint32_t segments,
        uint32_t heads,
        uint32_t padded_head_dim,
        float scale,
        cudaStream_t stream) {
    if (!out || !q || !k || !v || !cu_seqlens || heads == 0 ||
        padded_head_dim != 128) {
        return false;
    }
    using T = __nv_bfloat16;
    using Params = flashinfer::SinglePrefillParams<T, T, T>;
    using Variant = flashinfer::DefaultAttention<false, false, false, false>;
    const uint32_t row_stride = heads * padded_head_dim;
    for (uint32_t i = 0; i < segments; ++i) {
        const uint32_t begin = cu_seqlens[i];
        const uint32_t end = cu_seqlens[i + 1];
        if (end < begin) return false;
        const uint32_t length = end - begin;
        if (length == 0) continue;
        Params params(
            const_cast<T *>(q + static_cast<uint64_t>(begin) * row_stride),
            const_cast<T *>(k + static_cast<uint64_t>(begin) * row_stride),
            const_cast<T *>(v + static_cast<uint64_t>(begin) * row_stride),
            nullptr,
            out + static_cast<uint64_t>(begin) * row_stride,
            nullptr, nullptr,
            heads, heads, length, length,
            row_stride, padded_head_dim,
            row_stride, padded_head_dim, padded_head_dim,
            -1, 0.0f, scale, 1.0f, 1.0f);
        const cudaError_t status =
            flashinfer::SinglePrefillWithKVCacheDispatched<
                128, 128,
                flashinfer::PosEncodingMode::kNone,
                false,
                flashinfer::MaskMode::kNone,
                Variant>(params, nullptr, stream);
        if (status != cudaSuccess) return false;
    }
    return true;
}

} // namespace qw3::flashinfer_vision_adapter
