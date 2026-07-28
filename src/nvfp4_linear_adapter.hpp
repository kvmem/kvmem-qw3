#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

namespace qw3::nvfp4_adapter {

size_t packed_activation_bytes(uint32_t rows, uint32_t cols);
size_t activation_scale_bytes(uint32_t rows, uint32_t cols);
size_t workspace_bytes(uint32_t rows, uint32_t out_cols, uint32_t inner_cols);

bool quantize_activation(void *packed_activation,
                         void *activation_scales,
                         const float *input_f32,
                         const float *input_global_scale_divisor,
                         uint32_t rows,
                         uint32_t inner_cols,
                         uint32_t input_stride,
                         bool clear_padded_scales,
                         cudaStream_t stream);

bool quantize_activation_bf16(void *packed_activation,
                              void *activation_scales,
                              const void *input_bf16,
                              const float *input_global_scale_divisor,
                              uint32_t rows,
                              uint32_t inner_cols,
                              uint32_t input_stride,
                              bool clear_padded_scales,
                              cudaStream_t stream);

// Fuse SwiGLU with the activation quantizer used by the following NVFP4
// projection. `pair_bf16` is row-major [rows, 2 * cols] with gate followed by
// up in each row. This avoids materializing the BF16/FP32 SwiGLU result.
bool quantize_silu_mul_bf16_pair(void *packed_activation,
                                 void *activation_scales,
                                 const void *pair_bf16,
                                 const float *input_global_scale_divisor,
                                 uint32_t rows,
                                 uint32_t cols,
                                 bool clear_padded_scales,
                                 cudaStream_t stream);

// Fuse FP32 RMSNorm with NVFP4 activation quantization. The normalization
// reduction accumulates in FP32 and the normalized values are never written
// to global memory.
bool quantize_rms_norm_activation(void *packed_activation,
                                  void *activation_scales,
                                  const float *input_f32,
                                  const float *weight_f32,
                                  const float *input_global_scale_divisor,
                                  uint32_t rows,
                                  uint32_t cols,
                                  float eps,
                                  bool clear_padded_scales,
                                  cudaStream_t stream);

bool quantize_rms_norm_activation_bf16(
        void *packed_activation,
        void *activation_scales,
        const void *input_bf16,
        const float *weight_f32,
        const float *input_global_scale_divisor,
        uint32_t rows,
        uint32_t cols,
        float eps,
        bool clear_padded_scales,
        cudaStream_t stream);

// Execute an NVFP4 GEMM when the caller has already produced the packed
// activation and swizzled block scales.
bool launch_prequantized(float *out_f32,
                         void *workspace,
                         size_t workspace_capacity,
                         const void *packed_activation,
                         const void *activation_scales,
                         const void *packed_weight,
                         const void *weight_scales,
                         const float *combined_global_scale,
                         uint32_t rows,
                         uint32_t out_cols,
                         uint32_t inner_cols,
                         cudaStream_t stream);

bool launch_prequantized_bf16(void *out_bf16,
                              void *workspace,
                              size_t workspace_capacity,
                              const void *packed_activation,
                              const void *activation_scales,
                              const void *packed_weight,
                              const void *weight_scales,
                              const float *combined_global_scale,
                              uint32_t rows,
                              uint32_t out_cols,
                              uint32_t inner_cols,
                              cudaStream_t stream);

bool launch_small_f32(float *out_f32,
                      const void *packed_activation,
                      const void *activation_scales,
                      const void *packed_weight,
                      const void *weight_scales,
                      const float *combined_global_scale,
                      uint32_t rows,
                      uint32_t out_cols,
                      uint32_t inner_cols,
                      uint32_t output_stride,
                      bool add_to_output,
                      cudaStream_t stream);

bool launch_small_bf16(void *out_bf16,
                       const void *packed_activation,
                       const void *activation_scales,
                       const void *packed_weight,
                       const void *weight_scales,
                       const float *combined_global_scale,
                       uint32_t rows,
                       uint32_t out_cols,
                       uint32_t inner_cols,
                       uint32_t output_stride,
                       bool add_to_output,
                       cudaStream_t stream);

bool launch_small_pair_f32(float *out0_f32,
                           float *out1_f32,
                           const void *packed_activation,
                           const void *activation_scales,
                           const void *packed_weight0,
                           const void *weight_scales0,
                           const float *combined_global_scale0,
                           const void *packed_weight1,
                           const void *weight_scales1,
                           const float *combined_global_scale1,
                           uint32_t rows,
                           uint32_t out_cols,
                           uint32_t inner_cols,
                           uint32_t output_stride0,
                           uint32_t output_stride1,
                           bool silu_mul_output,
                           cudaStream_t stream);

bool launch(float *out_f32,
            void *packed_activation,
            void *activation_scales,
            void *workspace,
            size_t workspace_capacity,
            const float *input_f32,
            uint32_t input_stride,
            const void *packed_weight,
            const void *weight_scales,
            const float *input_global_scale_divisor,
            const float *combined_global_scale,
            uint32_t rows,
            uint32_t out_cols,
            uint32_t inner_cols,
            cudaStream_t stream);

bool launch_bf16(void *out_bf16,
                 void *packed_activation,
                 void *activation_scales,
                 void *workspace,
                 size_t workspace_capacity,
                 const float *input_f32,
                 uint32_t input_stride,
                 const void *packed_weight,
                 const void *weight_scales,
                 const float *input_global_scale_divisor,
                 const float *combined_global_scale,
                 uint32_t rows,
                 uint32_t out_cols,
                 uint32_t inner_cols,
                 cudaStream_t stream);

} // namespace qw3::nvfp4_adapter
