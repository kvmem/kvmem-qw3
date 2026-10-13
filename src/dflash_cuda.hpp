#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace qw3::dflash_cuda {

struct AttentionRequest {
    const float *keys;
    const float *values;
    uint32_t count;
    uint32_t reserved = 0;
};
static_assert(sizeof(AttentionRequest) == 24, "DFlash attention metadata layout");

cudaError_t capture_features(float *out, const void *hidden, bool bf16,
                             uint32_t rows, uint32_t width, uint32_t source_stride,
                             uint32_t taps, uint32_t tap, cudaStream_t stream);

cudaError_t grouped_conv(float *out, const float *x, const float *dynamic,
                        const float *base, uint32_t rows, uint32_t hidden,
                        uint32_t block, uint32_t taps, uint32_t group,
                        uint32_t side, cudaStream_t stream);
cudaError_t attention(float *out, const float *q, const float *k, const float *v,
                      uint32_t queries, uint32_t keys, uint32_t heads,
                      uint32_t kv_heads, uint32_t dim, uint32_t window,
                      cudaStream_t stream);
cudaError_t attention_batch(float *out, const float *q, const AttentionRequest *requests,
                            uint32_t blocks, uint32_t queries, uint32_t max_keys,
                            uint32_t heads, uint32_t kv_heads, uint32_t dim,
                            uint32_t window, cudaStream_t stream);
// Scratch has rows * ceil(vocab / 4096) * top_k elements in each plane.
cudaError_t topk(int32_t *ids, float *values, const float *logits,
                 int32_t *scratch_ids, float *scratch_values,
                 uint32_t rows, uint32_t vocab, uint32_t top_k, cudaStream_t stream);
cudaError_t select(int32_t *tokens, float *probs, const int32_t *candidates,
                   const float *unary, const float *hidden,
                   const void *predecessor, const void *successor, bool bf16,
                   const int32_t *anchors, const float *uniforms,
                   uint32_t blocks, uint32_t positions, uint32_t top_k,
                   uint32_t rank, float temperature, cudaStream_t stream);

} // namespace qw3::dflash_cuda
