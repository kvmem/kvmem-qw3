#pragma once
#include "flashinfer_attention_capture.hpp"
#include <flashinfer/attention/variants.cuh>

namespace qw3::flashinfer_adapter {
template<class Base> struct CaptureParams : Base {
    AttentionCapture capture;
    CaptureParams(const Base &base, AttentionCapture c) : Base(base), capture(c) {}
};

template<bool Decode> struct CaptureAttention
    : flashinfer::DefaultAttention<false, false, false, false> {
    using Base = flashinfer::DefaultAttention<false, false, false, false>;
    uint32_t capture_end;
    template<class P> __host__ __device__ CaptureAttention(
        const P &p, uint32_t batch, uint8_t *smem) : Base(p, batch, smem) {
        capture_end = kv_len;
#ifdef __CUDA_ARCH__
        // LogitsTransform precedes FlashInfer's partition/padding mask. Without
        // this bound padded lanes can overwrite the next partition's real QK.
        if (p.partition_kv)
            capture_end = min(kv_len, uint32_t(p.kv_tile_indices[blockIdx.x] + 1) *
                                     uint32_t(*p.kv_chunk_size_ptr));
#endif
    }

    template<class P> __device__ __forceinline__ float LogitsTransform(
        const P &p, float logits, uint32_t batch, uint32_t query,
        uint32_t key, uint32_t head, uint32_t /*kv_head*/) {
        const auto &c = p.capture;
        const uint32_t original_row = (Decode ? batch : query) + c.row_offset;
        const uint32_t delta = original_row - c.first_row;
        const uint32_t row = delta / c.row_step;
        // Decode replicates a dot product across its vector lanes. Prefill
        // distributes unique MMA fragments across lanes instead.
        if ((!Decode || threadIdx.x == 0) && original_row >= c.first_row &&
            delta % c.row_step == 0 && row < 64 &&
            ((c.row_mask >> row) & 1) && head < c.heads &&
            query < qo_len && key < capture_end &&
            key < kv_len - qo_len + query + 1 && key < c.stride) {
            const uint32_t slot = __popcll(c.row_mask & ((uint64_t(1) << row) - 1));
            c.logits[(uint64_t(slot) * c.heads + head) * c.stride + key] =
                logits * p.sm_scale;
        }
        return logits; // Preserve the original attention arithmetic exactly.
    }
};
template<bool D> struct CaptureTag { using type = CaptureAttention<D>; };
struct DefaultCaptureTag { using type = flashinfer::DefaultAttention<false, false, false, false>; };
}
