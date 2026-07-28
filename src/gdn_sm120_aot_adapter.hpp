#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace qw3::gdn_sm120_aot {

bool launch(
    void *out_bf16,
    float *output_state,
    const void *q_bf16,
    const void *k_bf16,
    const void *v_bf16,
    const float *initial_state,
    float *decay,
    float *beta,
    int64_t *cu_seqlens,
    uint8_t *tensormaps,
    uint32_t tensormaps_bytes,
    uint32_t tokens,
    uint32_t num_q_heads,
    uint32_t num_k_heads,
    uint32_t num_v_heads,
    uint32_t head_dim,
    float scale,
    cudaStream_t stream);

} // namespace qw3::gdn_sm120_aot
