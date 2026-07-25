#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace qw3::fp8_scaled_mm_adapter {

// Runs a row-major [M,K] by column-major [K,N] FP8 GEMM. The weight buffer is
// stored row-major [N,K], which has the same physical layout as column-major
// [K,N]. Per-row activation and per-column weight scales are applied in the
// CUTLASS epilogue before the result is written as BF16.
bool launch_bf16(void *out_bf16,
                 const void *activation_fp8,
                 const void *weight_fp8,
                 const float *activation_scales,
                 const float *weight_scales,
                 uint32_t rows,
                 uint32_t out_cols,
                 uint32_t inner_cols,
                 void *workspace,
                 size_t workspace_capacity,
                 cudaStream_t stream);

bool launch_f32(float *out_f32,
                const void *activation_fp8,
                const void *weight_fp8,
                const float *activation_scales,
                const float *weight_scales,
                uint32_t rows,
                uint32_t out_cols,
                uint32_t inner_cols,
                void *workspace,
                size_t workspace_capacity,
                cudaStream_t stream);

} // namespace qw3::fp8_scaled_mm_adapter
