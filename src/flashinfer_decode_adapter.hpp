#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qw3 {
namespace flashinfer_adapter {

// Conservative temporary storage for FlashInfer single decode when it splits a
// long KV cache into chunks. The unit is FP16 elements because FlashInfer's
// CUDA-core merge path supports FP16 partial output more broadly than FP32.
// Pass nullptr as tmp to use FlashInfer's non-partition path.
uint64_t decode_f32q_f16kv_tmp_elements(uint32_t n_heads,
                                        uint32_t head_dim,
                                        uint32_t seq_len);

// Optional adapter boundary for FlashInfer single-token decode attention.
// Inputs keep qw3's native layout:
//   q:   FP32 [n_heads, q_stride], with Q in the first head_dim values.
//   k/v: FP16 [seq_len, n_kv_heads, head_dim].
//   out: FP32 [n_heads, head_dim], without applying qw3's Q-gate.
bool launch_decode_f32q_f16kv(float *out,
                              __half *o_f16,
                              __half *tmp,
                              const float *q,
                              uint32_t q_stride,
                              const void *k_cache,
                              const void *v_cache,
                              uint32_t n_heads,
                              uint32_t n_kv_heads,
                              uint32_t head_dim,
                              uint32_t seq_len,
                              float scale,
                              cudaStream_t stream);

// FP8 (e4m3) KV variant of single-token decode. K/V read as __nv_fp8_e4m3
// (upcast to half inside FlashInfer's generic decode template), Q stays FP32,
// out FP32. Raw e4m3, no scale. tmp shape matches the f16kv variant.
bool launch_decode_f32q_fp8kv(float *out,
                              __half *o_f16,
                              __half *tmp,
                              const float *q,
                              uint32_t q_stride,
                              const void *k_cache,
                              const void *v_cache,
                              uint32_t n_heads,
                              uint32_t n_kv_heads,
                              uint32_t head_dim,
                              uint32_t seq_len,
                              float scale,
                              cudaStream_t stream);

uint32_t batch_decode_f32q_f16kv_chunk_size(uint32_t n_heads,
                                            uint32_t n_kv_heads,
                                            uint32_t head_dim,
                                            uint32_t batch,
                                            uint32_t max_seq_len);

bool launch_batch_decode_f32q_f16kv_gated(
        float *out,
        __half *o_f16,
        __half *tmp_v,
        float *tmp_s,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *request_indices,
        const int32_t *kv_tile_indices,
        const int32_t *o_indptr,
        const int32_t *o_indptr_host,
        const int32_t *kv_chunk_size,
        bool partition_kv,
        bool rowwise_merge,
        uint32_t page_size,
        uint32_t total_tiles,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream);

bool launch_batch_decode_f32q_f16kv(
        float *out,
        __half *o_f16,
        __half *tmp_v,
        float *tmp_s,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *request_indices,
        const int32_t *kv_tile_indices,
        const int32_t *o_indptr,
        const int32_t *o_indptr_host,
        const int32_t *kv_chunk_size,
        bool partition_kv,
        bool rowwise_merge,
        uint32_t page_size,
        uint32_t total_tiles,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream);

} // namespace flashinfer_adapter
} // namespace qw3
