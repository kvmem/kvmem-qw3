#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace qw3 {
namespace flashinfer_adapter {

// FlashInfer single-prefill adapter. Two variants share the same Q/KV layout:
//   q:   FP32 [batch, n_heads, q_stride] — Q in the first head_dim slots,
//        sigmoid gate in the next head_dim slots (qw3 / qw3_ly convention).
//   k/v: FP16 [seq, n_kv_heads, head_dim].
//   out: FP32 [batch, n_heads, head_dim].
//
// The _gated variants apply sigmoid(gate) inside the unpack kernel — used by
// qw3_ly which fuses the gate. The plain variants leave the gate to the
// caller's downstream apply_attn_gate path — used by qw3 which keeps the
// step separate. Pick one or the other, never both, or you'll double-gate.
bool launch_prefill_f16q_f16kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);

bool launch_prefill_f16q_f16kv(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);

// FP8 (e4m3) KV variant of the plain (non-gated) prefill. Q is packed to FP16,
// K/V are read as __nv_fp8_e4m3 (upcast to half inside FlashInfer's generic
// template), output unpacked to FP32. The gate is left to the caller. KV
// pointers point at raw e4m3 planes (no scale). Used when QW3_KV_DTYPE=fp8.
bool launch_prefill_f16q_fp8kv(
        float *out,
        __half *q_f16,
        __half *o_f16,
        const float *q, uint32_t q_stride,
        const void *k_cache, const void *v_cache,
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t batch, uint32_t base_seq_len,
        uint32_t q_batch_stride, uint32_t out_batch_stride,
        float scale, cudaStream_t stream);

bool launch_batch_prefill_paged_f16q_f16kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        uint32_t n_pages,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t base_seq_len,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream);

bool launch_batch_prefill_paged_f16q_fp8kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        uint32_t n_pages,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t base_seq_len,
        uint32_t batch,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream);

bool launch_batch_prefill_paged_ragged_f16q_f16kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *q_indptr,
        const int32_t *q_indptr_host,
        const int32_t *page_indptr_host,
        uint32_t batch,
        uint32_t total_q,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream);

bool launch_batch_prefill_paged_ragged_f16q_fp8kv_gated(
        float *out,
        __half *q_f16,
        __half *o_f16,
        int32_t *int_workspace,
        void *host_int_workspace,
        size_t int_workspace_bytes,
        float *float_workspace,
        size_t float_workspace_bytes,
        const float *q,
        uint32_t q_stride,
        const void *k_cache,
        const void *v_cache,
        const int32_t *page_indices,
        const int32_t *page_indptr,
        const int32_t *last_page_len,
        const int32_t *q_indptr,
        const int32_t *q_indptr_host,
        const int32_t *page_indptr_host,
        uint32_t batch,
        uint32_t total_q,
        uint32_t page_size,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t q_batch_stride,
        uint32_t out_batch_stride,
        float scale,
        cudaStream_t stream);

} // namespace flashinfer_adapter
} // namespace qw3
