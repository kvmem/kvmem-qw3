#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace qw3::flashinfer_vision_adapter {

// Q/K/V/O are token-major [tokens, heads, 128] BF16.  Each range in
// cu_seqlens is an independent image/video-frame sequence and receives full,
// non-causal self attention.  `logical_head_dim` controls the QK scale; padded
// lanes are zero and exist only because FlashInfer specializes power-of-two
// head dimensions.
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
    cudaStream_t stream);

} // namespace qw3::flashinfer_vision_adapter
