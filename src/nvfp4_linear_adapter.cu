#include "nvfp4_linear_adapter.hpp"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h"

namespace flashinfer {
namespace gemm {

#if defined(QW3_FLASHINFER_SM120_SWAP_AB)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    float, 128, 128, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    float, 128, 64, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    __nv_bfloat16, 128, 128, 128, 1, 1, 1, _1SM, false)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    float, 128, 32, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    float, 128, 32, 256, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    __nv_bfloat16, 128, 32, 128, 1, 1, 1, _1SM, true)
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    __nv_bfloat16, 128, 32, 256, 1, 1, 1, _1SM, true)
#else
INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(
    float, 128, 128, 128, 1, 1, 1, _1SM)
#endif

} // namespace gemm
} // namespace flashinfer

namespace qw3::nvfp4_adapter {
namespace {

constexpr uint32_t kScaleGroup = 16;

uint32_t round_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

#if defined(QW3_FLASHINFER_SM120_SWAP_AB)
uint32_t cutlass_tile_n() {
    static const uint32_t value = []() {
        const char *env = std::getenv("QW3_NVFP4_CUTLASS_TILE_N");
        if (!env || !*env) return 128U;
        const unsigned long parsed = std::strtoul(env, nullptr, 10);
        return parsed == 64 ? 64U : 128U;
    }();
    return value;
}

bool use_vllm_decode_tactics() {
    static const bool enabled = []() {
        const char *env = std::getenv("QW3_NVFP4_VLLM_TACTICS");
        return !env ||
               (std::strcmp(env, "0") != 0 &&
                std::strcmp(env, "off") != 0 &&
                std::strcmp(env, "false") != 0);
    }();
    return enabled;
}

template <typename Output>
size_t dispatch_sm120_vllm_decode_tactic(
        Output *out,
        const void *packed_activation,
        const void *packed_weight,
        const void *activation_scales,
        const void *weight_scales,
        const float *combined_global_scale,
        uint32_t rows,
        uint32_t out_cols,
        uint32_t inner_cols,
        char *workspace,
        size_t workspace_bytes,
        cudaStream_t stream) {
    using namespace flashinfer::gemm;
    if (out_cols >= 32768) {
        const CutlassGemmConfig config(
            CutlassTileConfigSM120::CtaShape128x32x128B,
            MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
            ClusterShape::ClusterShape_1x1x1,
            /*swap_ab=*/true, /*use_stream_k=*/false);
        return dispatchNVFP4xNVFP4GemmClusterShapeSm120<
            Output, cute::Int<128>, cute::Int<32>, cute::Int<256>,
            true, false>(
                out, packed_activation, packed_weight,
                activation_scales, weight_scales, combined_global_scale,
                rows, out_cols, inner_cols, 1, config,
                workspace, workspace_bytes, stream);
    }
    const CutlassGemmConfig config(
        CutlassTileConfigSM120::CtaShape128x32x64B,
        MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
        ClusterShape::ClusterShape_1x1x1,
        /*swap_ab=*/true, /*use_stream_k=*/true);
    return dispatchNVFP4xNVFP4GemmClusterShapeSm120<
        Output, cute::Int<128>, cute::Int<32>, cute::Int<128>,
        true, true>(
            out, packed_activation, packed_weight,
            activation_scales, weight_scales, combined_global_scale,
            rows, out_cols, inner_cols, 1, config,
            workspace, workspace_bytes, stream);
}

template <typename Output, int CtaN>
size_t dispatch_sm120_tile(Output *out,
                           const void *packed_activation,
                           const void *packed_weight,
                           const void *activation_scales,
                           const void *weight_scales,
                           const float *combined_global_scale,
                           uint32_t rows,
                           uint32_t out_cols,
                           uint32_t inner_cols,
                           bool use_stream_k,
                           char *workspace,
                           size_t workspace_bytes,
                           cudaStream_t stream) {
    using namespace flashinfer::gemm;
    constexpr CutlassTileConfigSM120 tile =
        CtaN == 64
        ? CutlassTileConfigSM120::CtaShape128x64x64B
        : CutlassTileConfigSM120::CtaShape128x128x64B;
    const CutlassGemmConfig config(
        tile, MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
        ClusterShape::ClusterShape_1x1x1, false, use_stream_k);
    if (use_stream_k) {
        return dispatchNVFP4xNVFP4GemmClusterShapeSm120<
            Output, cute::Int<128>, cute::Int<CtaN>, cute::Int<128>, false, true>(
            out, packed_activation,
            packed_weight, activation_scales, weight_scales,
            combined_global_scale, rows, out_cols, inner_cols, 1, config,
            workspace, workspace_bytes, stream);
    }
    return dispatchNVFP4xNVFP4GemmClusterShapeSm120<
        Output, cute::Int<128>, cute::Int<CtaN>, cute::Int<128>, false, false>(
        out, packed_activation,
        packed_weight, activation_scales, weight_scales,
        combined_global_scale, rows, out_cols, inner_cols, 1, config,
        workspace, workspace_bytes, stream);
}
#endif

size_t dispatch_sm120(float *out_f32,
                      const void *packed_activation,
                      const void *packed_weight,
                      const void *activation_scales,
                      const void *weight_scales,
                      const float *combined_global_scale,
                      uint32_t rows,
                      uint32_t out_cols,
                      uint32_t inner_cols,
                      bool use_stream_k,
                      char *workspace,
                      size_t workspace_bytes,
                      cudaStream_t stream) {
    using namespace flashinfer::gemm;
#if defined(QW3_FLASHINFER_SM120_SWAP_AB)
    if (rows < 128 && use_vllm_decode_tactics()) {
        return dispatch_sm120_vllm_decode_tactic(
            out_f32, packed_activation, packed_weight,
            activation_scales, weight_scales, combined_global_scale,
            rows, out_cols, inner_cols, workspace, workspace_bytes, stream);
    }
    if (cutlass_tile_n() == 64) {
        return dispatch_sm120_tile<float, 64>(
            out_f32, packed_activation, packed_weight,
            activation_scales, weight_scales, combined_global_scale,
            rows, out_cols, inner_cols, use_stream_k,
            workspace, workspace_bytes, stream);
    }
    return dispatch_sm120_tile<float, 128>(
        out_f32, packed_activation, packed_weight,
        activation_scales, weight_scales, combined_global_scale,
        rows, out_cols, inner_cols, use_stream_k,
        workspace, workspace_bytes, stream);
#else
    const CutlassGemmConfig config(
        CutlassTileConfigSM120::CtaShape128x128x128B,
        MainloopScheduleType::AUTO, EpilogueScheduleType::AUTO,
        ClusterShape::ClusterShape_1x1x1, use_stream_k);
    if (use_stream_k) {
        return dispatchNVFP4xNVFP4GemmClusterShapeSm120<
            float, cute::Int<128>, cute::Int<128>, cute::Int<128>, true>(
            out_f32, packed_activation,
            packed_weight, activation_scales, weight_scales,
            combined_global_scale, rows, out_cols, inner_cols, 1, config,
            workspace, workspace_bytes, stream);
    }
    return dispatchNVFP4xNVFP4GemmClusterShapeSm120<
        float, cute::Int<128>, cute::Int<128>, cute::Int<128>, false>(
        out_f32, packed_activation,
        packed_weight, activation_scales, weight_scales,
        combined_global_scale, rows, out_cols, inner_cols, 1, config,
        workspace, workspace_bytes, stream);
#endif
}

#if defined(QW3_FLASHINFER_SM120_SWAP_AB)
size_t dispatch_sm120_bf16(__nv_bfloat16 *out_bf16,
                           const void *packed_activation,
                           const void *packed_weight,
                           const void *activation_scales,
                           const void *weight_scales,
                           const float *combined_global_scale,
                           uint32_t rows,
                           uint32_t out_cols,
                           uint32_t inner_cols,
                           bool use_stream_k,
                           char *workspace,
                           size_t workspace_bytes,
                           cudaStream_t stream) {
    if (rows < 128 && use_vllm_decode_tactics()) {
        return dispatch_sm120_vllm_decode_tactic(
            out_bf16, packed_activation, packed_weight,
            activation_scales, weight_scales, combined_global_scale,
            rows, out_cols, inner_cols, workspace, workspace_bytes, stream);
    }
    return dispatch_sm120_tile<__nv_bfloat16, 128>(
        out_bf16, packed_activation, packed_weight,
        activation_scales, weight_scales, combined_global_scale,
        rows, out_cols, inner_cols, use_stream_k,
        workspace, workspace_bytes, stream);
}
#endif

__global__ void quantize_fp32_nvfp4_kernel(uint8_t *packed,
                                           uint8_t *scales,
                                           const float *input,
                                           const float *global_scale_divisor,
                                           uint32_t rows,
                                           uint32_t cols,
                                           uint32_t input_stride,
                                           uint32_t scale_cols) {
    const uint64_t group_index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groups_per_row = cols / kScaleGroup;
    const uint64_t group_count = static_cast<uint64_t>(rows) * groups_per_row;
    if (group_index >= group_count) return;

    const uint32_t row = static_cast<uint32_t>(group_index / groups_per_row);
    const uint32_t group = static_cast<uint32_t>(group_index % groups_per_row);
    const float *src = input + static_cast<uint64_t>(row) * input_stride +
                       static_cast<uint64_t>(group) * kScaleGroup;
    float values[kScaleGroup];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; ++i) {
        values[i] = src[i];
        amax = fmaxf(amax, fabsf(values[i]));
    }

    const float divisor = global_scale_divisor[0];
    const float unrounded_sf = divisor * (amax / 6.0f);
    const __nv_fp8_e4m3 sf(unrounded_sf);
    const float rounded_sf = static_cast<float>(sf);
    const float quant_multiplier =
        amax > 0.0f && rounded_sf > 0.0f ? divisor / rounded_sf : 0.0f;

    uint8_t *dst = packed + static_cast<uint64_t>(row) * (cols / 2) +
                   static_cast<uint64_t>(group) * (kScaleGroup / 2);
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; i += 2) {
        dst[i / 2] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[i] * quant_multiplier,
                        values[i + 1] * quant_multiplier),
            __NV_E2M1, cudaRoundNearest);
    }

    const uint64_t tiles_k = scale_cols / 4;
    const uint64_t scale_offset =
        ((static_cast<uint64_t>(row) / 128) * tiles_k + group / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    scales[scale_offset] = sf.__x;
}

__global__ void quantize_bf16_nvfp4_kernel(
        uint8_t *packed,
        uint8_t *scales,
        const __nv_bfloat16 *input,
        const float *global_scale_divisor,
        uint32_t rows,
        uint32_t cols,
        uint32_t input_stride,
        uint32_t scale_cols) {
    const uint64_t group_index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groups_per_row = cols / kScaleGroup;
    const uint64_t group_count = static_cast<uint64_t>(rows) * groups_per_row;
    if (group_index >= group_count) return;

    const uint32_t row = static_cast<uint32_t>(group_index / groups_per_row);
    const uint32_t group =
        static_cast<uint32_t>(group_index % groups_per_row);
    const __nv_bfloat16 *src =
        input + static_cast<uint64_t>(row) * input_stride +
        static_cast<uint64_t>(group) * kScaleGroup;
    float values[kScaleGroup];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; ++i) {
        values[i] = __bfloat162float(src[i]);
        amax = fmaxf(amax, fabsf(values[i]));
    }

    const float divisor = global_scale_divisor[0];
    const float unrounded_sf = divisor * (amax / 6.0f);
    const __nv_fp8_e4m3 sf(unrounded_sf);
    const float rounded_sf = static_cast<float>(sf);
    const float quant_multiplier =
        amax > 0.0f && rounded_sf > 0.0f ? divisor / rounded_sf : 0.0f;
    uint8_t *dst =
        packed + static_cast<uint64_t>(row) * (cols / 2) +
        static_cast<uint64_t>(group) * (kScaleGroup / 2);
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; i += 2) {
        dst[i / 2] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[i] * quant_multiplier,
                        values[i + 1] * quant_multiplier),
            __NV_E2M1, cudaRoundNearest);
    }
    const uint64_t tiles_k = scale_cols / 4;
    const uint64_t scale_offset =
        ((static_cast<uint64_t>(row) / 128) * tiles_k + group / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    scales[scale_offset] = sf.__x;
}

__global__ void silu_mul_bf16_pair_quantize_nvfp4_kernel(
        uint8_t *packed,
        uint8_t *scales,
        const __nv_bfloat16 *pair,
        const float *global_scale_divisor,
        uint32_t rows,
        uint32_t cols,
        uint32_t scale_cols) {
    const uint64_t group_index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t groups_per_row = cols / kScaleGroup;
    const uint64_t group_count = static_cast<uint64_t>(rows) * groups_per_row;
    if (group_index >= group_count) return;

    const uint32_t row = static_cast<uint32_t>(group_index / groups_per_row);
    const uint32_t group = static_cast<uint32_t>(group_index % groups_per_row);
    const uint64_t pair_row_offset =
        static_cast<uint64_t>(row) * (2ULL * cols);
    const uint64_t group_offset =
        static_cast<uint64_t>(group) * kScaleGroup;
    const __nv_bfloat16 *gate = pair + pair_row_offset + group_offset;
    const __nv_bfloat16 *up = gate + cols;

    float values[kScaleGroup];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; ++i) {
        const float gate_value = __bfloat162float(gate[i]);
        const float up_value = __bfloat162float(up[i]);
        const float value =
            (gate_value / (1.0f + __expf(-gate_value))) * up_value;
        values[i] = value;
        amax = fmaxf(amax, fabsf(value));
    }

    const float divisor = global_scale_divisor[0];
    const float unrounded_sf = divisor * (amax / 6.0f);
    const __nv_fp8_e4m3 sf(unrounded_sf);
    const float rounded_sf = static_cast<float>(sf);
    const float quant_multiplier =
        amax > 0.0f && rounded_sf > 0.0f ? divisor / rounded_sf : 0.0f;

    uint8_t *dst = packed + static_cast<uint64_t>(row) * (cols / 2) +
                   group_offset / 2;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; i += 2) {
        dst[i / 2] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[i] * quant_multiplier,
                        values[i + 1] * quant_multiplier),
            __NV_E2M1, cudaRoundNearest);
    }

    const uint64_t tiles_k = scale_cols / 4;
    const uint64_t scale_offset =
        ((static_cast<uint64_t>(row) / 128) * tiles_k + group / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    scales[scale_offset] = sf.__x;
}

__global__ void rms_norm_quantize_nvfp4_kernel(
        uint8_t *packed,
        uint8_t *scales,
        const float *input,
        const float *weight,
        const float *global_scale_divisor,
        uint32_t cols,
        uint32_t scale_cols,
        float eps) {
    constexpr uint32_t kThreads = 1024;
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kWarps = kThreads / kWarpSize;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % kWarpSize;
    const uint32_t warp = tid / kWarpSize;
    const float *src = input + static_cast<uint64_t>(row) * cols;

    // Match the existing vectorized FP32 RMSNorm reduction.
    const uint32_t n_vec = cols / 4;
    const float4 *src4 = reinterpret_cast<const float4 *>(src);
    float sum = 0.0f;
    for (uint32_t i = tid; i < n_vec; i += kThreads) {
        const float4 value = src4[i];
        sum += value.x * value.x + value.y * value.y +
               value.z * value.z + value.w * value.w;
    }
#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        sum += __shfl_xor_sync(0xffffffffU, sum, delta);
    }
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < kWarps ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
            sum += __shfl_xor_sync(0xffffffffU, sum, delta);
        }
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();
    const float norm_scale =
        rsqrtf(warp_sums[0] / static_cast<float>(cols) + eps);

    const uint32_t groups_per_row = cols / kScaleGroup;
    if (tid >= groups_per_row) return;
    const uint32_t group = tid;
    const uint64_t group_offset =
        static_cast<uint64_t>(group) * kScaleGroup;
    float values[kScaleGroup];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; ++i) {
        const uint32_t col = static_cast<uint32_t>(group_offset) + i;
        const float value = src[col] * norm_scale * weight[col];
        values[i] = value;
        amax = fmaxf(amax, fabsf(value));
    }

    const float divisor = global_scale_divisor[0];
    const float unrounded_sf = divisor * (amax / 6.0f);
    const __nv_fp8_e4m3 sf(unrounded_sf);
    const float rounded_sf = static_cast<float>(sf);
    const float quant_multiplier =
        amax > 0.0f && rounded_sf > 0.0f ? divisor / rounded_sf : 0.0f;
    uint8_t *dst = packed + static_cast<uint64_t>(row) * (cols / 2) +
                   group_offset / 2;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; i += 2) {
        dst[i / 2] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[i] * quant_multiplier,
                        values[i + 1] * quant_multiplier),
            __NV_E2M1, cudaRoundNearest);
    }

    const uint64_t tiles_k = scale_cols / 4;
    const uint64_t scale_offset =
        ((static_cast<uint64_t>(row) / 128) * tiles_k + group / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    scales[scale_offset] = sf.__x;
}

__global__ void rms_norm_bf16_quantize_nvfp4_kernel(
        uint8_t *packed,
        uint8_t *scales,
        const __nv_bfloat16 *input,
        const float *weight,
        const float *global_scale_divisor,
        uint32_t cols,
        uint32_t scale_cols,
        float eps) {
    constexpr uint32_t kThreads = 1024;
    constexpr uint32_t kWarpSize = 32;
    constexpr uint32_t kWarps = kThreads / kWarpSize;
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid % kWarpSize;
    const uint32_t warp = tid / kWarpSize;
    const __nv_bfloat16 *src =
        input + static_cast<uint64_t>(row) * cols;

    float sum = 0.0f;
    for (uint32_t col = tid; col < cols; col += kThreads) {
        const float value = __bfloat162float(src[col]);
        sum = fmaf(value, value, sum);
    }
#pragma unroll
    for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
        sum += __shfl_xor_sync(0xffffffffU, sum, delta);
    }
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < kWarps ? warp_sums[lane] : 0.0f;
#pragma unroll
        for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
            sum += __shfl_xor_sync(0xffffffffU, sum, delta);
        }
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();
    const float norm_scale =
        rsqrtf(warp_sums[0] / static_cast<float>(cols) + eps);

    const uint32_t groups_per_row = cols / kScaleGroup;
    if (tid >= groups_per_row) return;
    const uint32_t group = tid;
    const uint64_t group_offset =
        static_cast<uint64_t>(group) * kScaleGroup;
    float values[kScaleGroup];
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; ++i) {
        const uint32_t col = static_cast<uint32_t>(group_offset) + i;
        const float value =
            __bfloat162float(src[col]) * norm_scale * weight[col];
        values[i] = value;
        amax = fmaxf(amax, fabsf(value));
    }

    const float divisor = global_scale_divisor[0];
    const float unrounded_sf = divisor * (amax / 6.0f);
    const __nv_fp8_e4m3 sf(unrounded_sf);
    const float rounded_sf = static_cast<float>(sf);
    const float quant_multiplier =
        amax > 0.0f && rounded_sf > 0.0f ? divisor / rounded_sf : 0.0f;
    uint8_t *dst =
        packed + static_cast<uint64_t>(row) * (cols / 2) +
        group_offset / 2;
#pragma unroll
    for (uint32_t i = 0; i < kScaleGroup; i += 2) {
        dst[i / 2] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[i] * quant_multiplier,
                        values[i + 1] * quant_multiplier),
            __NV_E2M1, cudaRoundNearest);
    }
    const uint64_t tiles_k = scale_cols / 4;
    const uint64_t scale_offset =
        ((static_cast<uint64_t>(row) / 128) * tiles_k + group / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
    scales[scale_offset] = sf.__x;
}

__device__ __forceinline__ float decode_e4m3(uint8_t raw) {
    __nv_fp8_e4m3 value;
    value.__x = raw;
    return static_cast<float>(value);
}

__device__ __forceinline__ int unpack_e2m1x4_i2(uint16_t packed) {
    uint32_t result = 0;
#pragma unroll
    for (uint32_t index = 0; index < 4; ++index) {
        const uint32_t nibble = (packed >> (index * 4)) & 0xfU;
        const uint32_t code = nibble & 0x7U;
        uint32_t magnitude = code;
        magnitude += code >= 5 ? code - 4 : 0;
        magnitude += code == 7 ? 2 : 0;
        const int32_t sign_mask = -static_cast<int32_t>(nibble >> 3);
        const int32_t signed_value =
            (static_cast<int32_t>(magnitude) ^ sign_mask) - sign_mask;
        result |= static_cast<uint32_t>(
                      static_cast<uint8_t>(static_cast<int8_t>(signed_value)))
                  << (index * 8);
    }
    return static_cast<int>(result);
}

__device__ __forceinline__ uint64_t scale_offset(uint32_t row,
                                                  uint32_t group,
                                                  uint32_t scale_cols) {
    const uint64_t tiles_k = scale_cols / 4;
    return ((static_cast<uint64_t>(row) / 128) * tiles_k + group / 4) * 512 +
           (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <int Batch, bool AddToOutput>
__launch_bounds__(128, 2)
__global__ void nvfp4_mmvq_f32_kernel(
        float *out,
        const uint8_t *packed_activation,
        const uint8_t *activation_scales,
        const uint8_t *packed_weight,
        const uint8_t *weight_scales,
        const float *combined_global_scale,
        uint32_t out_cols,
        uint32_t inner_cols,
        uint32_t output_stride) {
    const uint32_t out_col = blockIdx.x;
    if (out_col >= out_cols) return;

    constexpr uint32_t kThreads = 128;
    constexpr uint32_t kWarps = kThreads / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t groups = inner_cols / kScaleGroup;
    const uint32_t scale_cols = groups;
    const uint32_t packed_stride = inner_cols / 2;
    const float global_scale = combined_global_scale[0];
    float accum[Batch] = {};

    for (uint32_t group = threadIdx.x; group < groups; group += kThreads) {
        const uint64_t weight_bits =
            reinterpret_cast<const uint64_t *>(
                packed_weight + static_cast<uint64_t>(out_col) * packed_stride +
                static_cast<uint64_t>(group) * (kScaleGroup / 2))[0];
        const float weight_scale = decode_e4m3(
            weight_scales[scale_offset(out_col, group, scale_cols)]);

        uint64_t activation_bits[Batch];
        float group_scale[Batch];
#pragma unroll
        for (int item = 0; item < Batch; ++item) {
            activation_bits[item] =
                reinterpret_cast<const uint64_t *>(
                    packed_activation + static_cast<uint64_t>(item) * packed_stride +
                    static_cast<uint64_t>(group) * (kScaleGroup / 2))[0];
            group_scale[item] =
                weight_scale *
                decode_e4m3(activation_scales[
                    scale_offset(static_cast<uint32_t>(item), group, scale_cols)]) *
                global_scale;
        }

        int group_dot[Batch] = {};
#pragma unroll
        for (uint32_t chunk = 0; chunk < kScaleGroup / 4; ++chunk) {
            const int weight_values = unpack_e2m1x4_i2(
                static_cast<uint16_t>(weight_bits >> (chunk * 16)));
#pragma unroll
            for (int item = 0; item < Batch; ++item) {
                const int activation_values = unpack_e2m1x4_i2(
                    static_cast<uint16_t>(
                        activation_bits[item] >> (chunk * 16)));
                group_dot[item] =
                    __dp4a(weight_values, activation_values, group_dot[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < Batch; ++item) {
            accum[item] = fmaf(static_cast<float>(group_dot[item]) * 0.25f,
                               group_scale[item], accum[item]);
        }
    }

    __shared__ float partial[kWarps][Batch];
#pragma unroll
    for (int item = 0; item < Batch; ++item) {
        const float reduced = warp_sum(accum[item]);
        if (lane == 0) partial[warp][item] = reduced;
    }
    __syncthreads();

    if (warp == 0 && lane < Batch) {
        float result = 0.0f;
#pragma unroll
        for (uint32_t source_warp = 0; source_warp < kWarps; ++source_warp) {
            result += partial[source_warp][lane];
        }
        float *dst = out + static_cast<uint64_t>(lane) * output_stride + out_col;
        if constexpr (AddToOutput) {
            *dst += result;
        } else {
            *dst = result;
        }
    }
}

template <int Batch, bool AddToOutput>
__launch_bounds__(128, 2)
__global__ void nvfp4_mmvq_bf16_kernel(
        __nv_bfloat16 *out,
        const uint8_t *packed_activation,
        const uint8_t *activation_scales,
        const uint8_t *packed_weight,
        const uint8_t *weight_scales,
        const float *combined_global_scale,
        uint32_t out_cols,
        uint32_t inner_cols,
        uint32_t output_stride) {
    const uint32_t out_col = blockIdx.x;
    if (out_col >= out_cols) return;

    constexpr uint32_t kThreads = 128;
    constexpr uint32_t kWarps = kThreads / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t groups = inner_cols / kScaleGroup;
    const uint32_t scale_cols = groups;
    const uint32_t packed_stride = inner_cols / 2;
    const float global_scale = combined_global_scale[0];
    float accum[Batch] = {};

    for (uint32_t group = threadIdx.x; group < groups; group += kThreads) {
        const uint64_t weight_bits =
            reinterpret_cast<const uint64_t *>(
                packed_weight + static_cast<uint64_t>(out_col) * packed_stride +
                static_cast<uint64_t>(group) * (kScaleGroup / 2))[0];
        const float weight_scale = decode_e4m3(
            weight_scales[scale_offset(out_col, group, scale_cols)]);
        uint64_t activation_bits[Batch];
        float group_scale[Batch];
#pragma unroll
        for (int item = 0; item < Batch; ++item) {
            activation_bits[item] =
                reinterpret_cast<const uint64_t *>(
                    packed_activation +
                    static_cast<uint64_t>(item) * packed_stride +
                    static_cast<uint64_t>(group) *
                        (kScaleGroup / 2))[0];
            group_scale[item] =
                weight_scale *
                decode_e4m3(activation_scales[
                    scale_offset(static_cast<uint32_t>(item), group,
                                 scale_cols)]) *
                global_scale;
        }
        int group_dot[Batch] = {};
#pragma unroll
        for (uint32_t chunk = 0; chunk < kScaleGroup / 4; ++chunk) {
            const int weight_values = unpack_e2m1x4_i2(
                static_cast<uint16_t>(weight_bits >> (chunk * 16)));
#pragma unroll
            for (int item = 0; item < Batch; ++item) {
                const int activation_values = unpack_e2m1x4_i2(
                    static_cast<uint16_t>(
                        activation_bits[item] >> (chunk * 16)));
                group_dot[item] = __dp4a(
                    weight_values, activation_values, group_dot[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < Batch; ++item) {
            accum[item] = fmaf(
                static_cast<float>(group_dot[item]) * 0.25f,
                group_scale[item], accum[item]);
        }
    }

    __shared__ float partial[kWarps][Batch];
#pragma unroll
    for (int item = 0; item < Batch; ++item) {
        const float reduced = warp_sum(accum[item]);
        if (lane == 0) partial[warp][item] = reduced;
    }
    __syncthreads();
    if (warp == 0 && lane < Batch) {
        float result = 0.0f;
#pragma unroll
        for (uint32_t source_warp = 0; source_warp < kWarps;
             ++source_warp) {
            result += partial[source_warp][lane];
        }
        __nv_bfloat16 *dst =
            out + static_cast<uint64_t>(lane) * output_stride + out_col;
        if constexpr (AddToOutput) {
            result += __bfloat162float(*dst);
        }
        *dst = __float2bfloat16(result);
    }
}

template <int Batch, bool SiluMulOutput>
__launch_bounds__(128, 1)
__global__ void nvfp4_mmvq_pair_f32_kernel(
        float *out0,
        float *out1,
        const uint8_t *packed_activation,
        const uint8_t *activation_scales,
        const uint8_t *packed_weight0,
        const uint8_t *weight_scales0,
        const float *combined_global_scale0,
        const uint8_t *packed_weight1,
        const uint8_t *weight_scales1,
        const float *combined_global_scale1,
        uint32_t out_cols,
        uint32_t inner_cols,
        uint32_t output_stride0,
        uint32_t output_stride1) {
    const uint32_t out_col = blockIdx.x;
    if (out_col >= out_cols) return;

    constexpr uint32_t kThreads = 128;
    constexpr uint32_t kWarps = kThreads / 32;
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t warp = threadIdx.x >> 5U;
    const uint32_t groups = inner_cols / kScaleGroup;
    const uint32_t scale_cols = groups;
    const uint32_t packed_stride = inner_cols / 2;
    const float global_scale0 = combined_global_scale0[0];
    const float global_scale1 = combined_global_scale1[0];
    float accum0[Batch] = {};
    float accum1[Batch] = {};

    for (uint32_t group = threadIdx.x; group < groups; group += kThreads) {
        const uint64_t weight_bits0 =
            reinterpret_cast<const uint64_t *>(
                packed_weight0 + static_cast<uint64_t>(out_col) * packed_stride +
                static_cast<uint64_t>(group) * (kScaleGroup / 2))[0];
        const uint64_t weight_bits1 =
            reinterpret_cast<const uint64_t *>(
                packed_weight1 + static_cast<uint64_t>(out_col) * packed_stride +
                static_cast<uint64_t>(group) * (kScaleGroup / 2))[0];
        const uint64_t sf_index = scale_offset(out_col, group, scale_cols);
        const float weight_scale0 = decode_e4m3(weight_scales0[sf_index]);
        const float weight_scale1 = decode_e4m3(weight_scales1[sf_index]);

        uint64_t activation_bits[Batch];
        float group_scale0[Batch];
        float group_scale1[Batch];
#pragma unroll
        for (int item = 0; item < Batch; ++item) {
            activation_bits[item] =
                reinterpret_cast<const uint64_t *>(
                    packed_activation + static_cast<uint64_t>(item) * packed_stride +
                    static_cast<uint64_t>(group) * (kScaleGroup / 2))[0];
            const float activation_scale = decode_e4m3(
                activation_scales[
                    scale_offset(static_cast<uint32_t>(item), group, scale_cols)]);
            group_scale0[item] =
                weight_scale0 * activation_scale * global_scale0;
            group_scale1[item] =
                weight_scale1 * activation_scale * global_scale1;
        }

        int group_dot0[Batch] = {};
        int group_dot1[Batch] = {};
#pragma unroll
        for (uint32_t chunk = 0; chunk < kScaleGroup / 4; ++chunk) {
            const int weight_values0 = unpack_e2m1x4_i2(
                static_cast<uint16_t>(weight_bits0 >> (chunk * 16)));
            const int weight_values1 = unpack_e2m1x4_i2(
                static_cast<uint16_t>(weight_bits1 >> (chunk * 16)));
#pragma unroll
            for (int item = 0; item < Batch; ++item) {
                const int activation_values = unpack_e2m1x4_i2(
                    static_cast<uint16_t>(
                        activation_bits[item] >> (chunk * 16)));
                group_dot0[item] =
                    __dp4a(weight_values0, activation_values, group_dot0[item]);
                group_dot1[item] =
                    __dp4a(weight_values1, activation_values, group_dot1[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < Batch; ++item) {
            accum0[item] =
                fmaf(static_cast<float>(group_dot0[item]) * 0.25f,
                     group_scale0[item], accum0[item]);
            accum1[item] =
                fmaf(static_cast<float>(group_dot1[item]) * 0.25f,
                     group_scale1[item], accum1[item]);
        }
    }

    __shared__ float partial0[kWarps][Batch];
    __shared__ float partial1[kWarps][Batch];
#pragma unroll
    for (int item = 0; item < Batch; ++item) {
        const float reduced0 = warp_sum(accum0[item]);
        const float reduced1 = warp_sum(accum1[item]);
        if (lane == 0) {
            partial0[warp][item] = reduced0;
            partial1[warp][item] = reduced1;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < Batch) {
        float result0 = 0.0f;
        float result1 = 0.0f;
#pragma unroll
        for (uint32_t source_warp = 0; source_warp < kWarps; ++source_warp) {
            result0 += partial0[source_warp][lane];
            result1 += partial1[source_warp][lane];
        }
        if constexpr (SiluMulOutput) {
            out0[static_cast<uint64_t>(lane) * output_stride0 + out_col] =
                (result0 / (1.0f + expf(-result0))) * result1;
        } else {
            out0[static_cast<uint64_t>(lane) * output_stride0 + out_col] = result0;
            out1[static_cast<uint64_t>(lane) * output_stride1 + out_col] = result1;
        }
    }
}

template <bool AddToOutput>
bool dispatch_small_f32(float *out_f32,
                        const void *packed_activation,
                        const void *activation_scales,
                        const void *packed_weight,
                        const void *weight_scales,
                        const float *combined_global_scale,
                        uint32_t rows,
                        uint32_t out_cols,
                        uint32_t inner_cols,
                        uint32_t output_stride,
                        cudaStream_t stream) {
    const dim3 grid(out_cols);
    constexpr dim3 block(128);
#define QW3_NVFP4_SMALL_CASE(BATCH)                                                \
    case BATCH:                                                                    \
        nvfp4_mmvq_f32_kernel<BATCH, AddToOutput><<<grid, block, 0, stream>>>(     \
            out_f32, static_cast<const uint8_t *>(packed_activation),              \
            static_cast<const uint8_t *>(activation_scales),                       \
            static_cast<const uint8_t *>(packed_weight),                           \
            static_cast<const uint8_t *>(weight_scales),                           \
            combined_global_scale, out_cols, inner_cols, output_stride);           \
        break
    switch (rows) {
        QW3_NVFP4_SMALL_CASE(1);
        QW3_NVFP4_SMALL_CASE(2);
        QW3_NVFP4_SMALL_CASE(3);
        QW3_NVFP4_SMALL_CASE(4);
        QW3_NVFP4_SMALL_CASE(5);
        QW3_NVFP4_SMALL_CASE(6);
        QW3_NVFP4_SMALL_CASE(7);
        QW3_NVFP4_SMALL_CASE(8);
        default: return false;
    }
#undef QW3_NVFP4_SMALL_CASE
    return cudaGetLastError() == cudaSuccess;
}

template <bool AddToOutput>
bool dispatch_small_bf16(void *out_bf16,
                         const void *packed_activation,
                         const void *activation_scales,
                         const void *packed_weight,
                         const void *weight_scales,
                         const float *combined_global_scale,
                         uint32_t rows,
                         uint32_t out_cols,
                         uint32_t inner_cols,
                         uint32_t output_stride,
                         cudaStream_t stream) {
    const dim3 grid(out_cols);
    constexpr dim3 block(128);
    auto *out = static_cast<__nv_bfloat16 *>(out_bf16);
#define QW3_NVFP4_SMALL_BF16_CASE(BATCH)                                      \
    case BATCH:                                                               \
        nvfp4_mmvq_bf16_kernel<BATCH, AddToOutput>                            \
            <<<grid, block, 0, stream>>>(                                     \
                out, static_cast<const uint8_t *>(packed_activation),         \
                static_cast<const uint8_t *>(activation_scales),              \
                static_cast<const uint8_t *>(packed_weight),                  \
                static_cast<const uint8_t *>(weight_scales),                  \
                combined_global_scale, out_cols, inner_cols, output_stride);  \
        break
    switch (rows) {
        QW3_NVFP4_SMALL_BF16_CASE(1);
        QW3_NVFP4_SMALL_BF16_CASE(2);
        QW3_NVFP4_SMALL_BF16_CASE(3);
        QW3_NVFP4_SMALL_BF16_CASE(4);
        QW3_NVFP4_SMALL_BF16_CASE(5);
        QW3_NVFP4_SMALL_BF16_CASE(6);
        QW3_NVFP4_SMALL_BF16_CASE(7);
        QW3_NVFP4_SMALL_BF16_CASE(8);
        default: return false;
    }
#undef QW3_NVFP4_SMALL_BF16_CASE
    return cudaGetLastError() == cudaSuccess;
}

template <bool SiluMulOutput>
bool dispatch_small_pair_f32(
        float *out0_f32,
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
        cudaStream_t stream) {
    const dim3 grid(out_cols);
    constexpr dim3 block(128);
#define QW3_NVFP4_PAIR_CASE(BATCH)                                                \
    case BATCH:                                                                   \
        nvfp4_mmvq_pair_f32_kernel<BATCH, SiluMulOutput>                          \
            <<<grid, block, 0, stream>>>(                                         \
                out0_f32, out1_f32,                                               \
                static_cast<const uint8_t *>(packed_activation),                  \
                static_cast<const uint8_t *>(activation_scales),                  \
                static_cast<const uint8_t *>(packed_weight0),                     \
                static_cast<const uint8_t *>(weight_scales0),                     \
                combined_global_scale0,                                           \
                static_cast<const uint8_t *>(packed_weight1),                     \
                static_cast<const uint8_t *>(weight_scales1),                     \
                combined_global_scale1, out_cols, inner_cols,                     \
                output_stride0, output_stride1);                                  \
        break
    switch (rows) {
        QW3_NVFP4_PAIR_CASE(1);
        QW3_NVFP4_PAIR_CASE(2);
        QW3_NVFP4_PAIR_CASE(3);
        QW3_NVFP4_PAIR_CASE(4);
        QW3_NVFP4_PAIR_CASE(5);
        QW3_NVFP4_PAIR_CASE(6);
        QW3_NVFP4_PAIR_CASE(7);
        QW3_NVFP4_PAIR_CASE(8);
        default: return false;
    }
#undef QW3_NVFP4_PAIR_CASE
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

size_t packed_activation_bytes(uint32_t rows, uint32_t cols) {
    return static_cast<size_t>(rows) * cols / 2;
}

size_t activation_scale_bytes(uint32_t rows, uint32_t cols) {
    const uint32_t scale_rows = round_up(rows, 128);
    const uint32_t scale_cols = round_up(cols / kScaleGroup, 4);
    return static_cast<size_t>(scale_rows) * scale_cols;
}

size_t workspace_bytes(uint32_t rows, uint32_t out_cols, uint32_t inner_cols) {
    struct Entry {
        uint32_t rows;
        uint32_t out_cols;
        uint32_t inner_cols;
        size_t bytes;
    };
    static thread_local std::vector<Entry> cache;
    for (const Entry &entry : cache) {
        if (entry.rows == rows && entry.out_cols == out_cols &&
            entry.inner_cols == inner_cols) {
            return entry.bytes;
        }
    }
    const size_t bytes =
        dispatch_sm120(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       rows, out_cols, inner_cols, rows < 128,
                       nullptr, 0, nullptr);
    cache.push_back({rows, out_cols, inner_cols, bytes});
    return bytes;
}

bool quantize_activation(void *packed_activation,
                         void *activation_scales,
                         const float *input_f32,
                         const float *input_global_scale_divisor,
                         uint32_t rows,
                         uint32_t inner_cols,
                         uint32_t input_stride,
                         bool clear_padded_scales,
                         cudaStream_t stream) {
    if (!packed_activation || !activation_scales || !input_f32 ||
        !input_global_scale_divisor || rows == 0 || inner_cols == 0 ||
        inner_cols % 64 != 0 || input_stride < inner_cols) {
        return false;
    }
    if (clear_padded_scales) {
        const size_t scale_bytes = activation_scale_bytes(rows, inner_cols);
        if (cudaMemsetAsync(activation_scales, 0, scale_bytes, stream) != cudaSuccess) {
            return false;
        }
    }
    const uint64_t group_count =
        static_cast<uint64_t>(rows) * inner_cols / kScaleGroup;
    const unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((group_count + threads - 1) / threads);
    quantize_fp32_nvfp4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t *>(packed_activation),
        static_cast<uint8_t *>(activation_scales), input_f32,
        input_global_scale_divisor, rows, inner_cols, input_stride,
        round_up(inner_cols / kScaleGroup, 4));
    return cudaGetLastError() == cudaSuccess;
}

bool quantize_activation_bf16(void *packed_activation,
                              void *activation_scales,
                              const void *input_bf16,
                              const float *input_global_scale_divisor,
                              uint32_t rows,
                              uint32_t inner_cols,
                              uint32_t input_stride,
                              bool clear_padded_scales,
                              cudaStream_t stream) {
    if (!packed_activation || !activation_scales || !input_bf16 ||
        !input_global_scale_divisor || rows == 0 || inner_cols == 0 ||
        inner_cols % 64 != 0 || input_stride < inner_cols) {
        return false;
    }
    if (clear_padded_scales) {
        const size_t scale_bytes = activation_scale_bytes(rows, inner_cols);
        if (cudaMemsetAsync(activation_scales, 0, scale_bytes, stream) !=
            cudaSuccess) {
            return false;
        }
    }
    const uint64_t group_count =
        static_cast<uint64_t>(rows) * inner_cols / kScaleGroup;
    const unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((group_count + threads - 1) / threads);
    quantize_bf16_nvfp4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t *>(packed_activation),
        static_cast<uint8_t *>(activation_scales),
        static_cast<const __nv_bfloat16 *>(input_bf16),
        input_global_scale_divisor, rows, inner_cols, input_stride,
        round_up(inner_cols / kScaleGroup, 4));
    return cudaGetLastError() == cudaSuccess;
}

bool quantize_silu_mul_bf16_pair(void *packed_activation,
                                 void *activation_scales,
                                 const void *pair_bf16,
                                 const float *input_global_scale_divisor,
                                 uint32_t rows,
                                 uint32_t cols,
                                 bool clear_padded_scales,
                                 cudaStream_t stream) {
    if (!packed_activation || !activation_scales || !pair_bf16 ||
        !input_global_scale_divisor || rows == 0 || cols == 0 ||
        cols % 64 != 0) {
        return false;
    }
    if (clear_padded_scales) {
        const size_t scale_bytes = activation_scale_bytes(rows, cols);
        if (cudaMemsetAsync(activation_scales, 0, scale_bytes, stream) !=
            cudaSuccess) {
            return false;
        }
    }
    const uint64_t group_count =
        static_cast<uint64_t>(rows) * cols / kScaleGroup;
    const unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((group_count + threads - 1) / threads);
    silu_mul_bf16_pair_quantize_nvfp4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint8_t *>(packed_activation),
        static_cast<uint8_t *>(activation_scales),
        static_cast<const __nv_bfloat16 *>(pair_bf16),
        input_global_scale_divisor, rows, cols,
        round_up(cols / kScaleGroup, 4));
    return cudaGetLastError() == cudaSuccess;
}

bool quantize_rms_norm_activation(void *packed_activation,
                                  void *activation_scales,
                                  const float *input_f32,
                                  const float *weight_f32,
                                  const float *input_global_scale_divisor,
                                  uint32_t rows,
                                  uint32_t cols,
                                  float eps,
                                  bool clear_padded_scales,
                                  cudaStream_t stream) {
    if (!packed_activation || !activation_scales || !input_f32 ||
        !weight_f32 || !input_global_scale_divisor || rows == 0 ||
        cols == 0 || cols % 64 != 0 || cols / kScaleGroup > 1024) {
        return false;
    }
    if (clear_padded_scales) {
        const size_t scale_bytes = activation_scale_bytes(rows, cols);
        if (cudaMemsetAsync(activation_scales, 0, scale_bytes, stream) !=
            cudaSuccess) {
            return false;
        }
    }
    rms_norm_quantize_nvfp4_kernel<<<rows, 1024, 0, stream>>>(
        static_cast<uint8_t *>(packed_activation),
        static_cast<uint8_t *>(activation_scales), input_f32, weight_f32,
        input_global_scale_divisor, cols,
        round_up(cols / kScaleGroup, 4), eps);
    return cudaGetLastError() == cudaSuccess;
}

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
        cudaStream_t stream) {
    if (!packed_activation || !activation_scales || !input_bf16 ||
        !weight_f32 || !input_global_scale_divisor || rows == 0 ||
        cols == 0 || cols % 64 != 0 || cols / kScaleGroup > 1024) {
        return false;
    }
    if (clear_padded_scales) {
        const size_t scale_bytes = activation_scale_bytes(rows, cols);
        if (cudaMemsetAsync(activation_scales, 0, scale_bytes, stream) !=
            cudaSuccess) {
            return false;
        }
    }
    rms_norm_bf16_quantize_nvfp4_kernel<<<rows, 1024, 0, stream>>>(
        static_cast<uint8_t *>(packed_activation),
        static_cast<uint8_t *>(activation_scales),
        static_cast<const __nv_bfloat16 *>(input_bf16), weight_f32,
        input_global_scale_divisor, cols,
        round_up(cols / kScaleGroup, 4), eps);
    return cudaGetLastError() == cudaSuccess;
}

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
                         cudaStream_t stream) {
    if (!out_f32 || !packed_activation || !activation_scales || !packed_weight ||
        !weight_scales || !combined_global_scale || rows == 0 ||
        out_cols == 0 || inner_cols == 0 || inner_cols % 64 != 0 ||
        out_cols % 32 != 0) {
        return false;
    }
    try {
        const bool use_stream_k = rows < 128;
        const size_t required = workspace_bytes(rows, out_cols, inner_cols);
        if (required > workspace_capacity) return false;
        dispatch_sm120(out_f32, packed_activation, packed_weight,
                       activation_scales, weight_scales, combined_global_scale,
                       rows, out_cols, inner_cols, use_stream_k,
                       static_cast<char *>(workspace), required, stream);
        return cudaGetLastError() == cudaSuccess;
    } catch (const std::runtime_error &) {
        return false;
    }
}

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
                              cudaStream_t stream) {
#if !defined(QW3_FLASHINFER_SM120_SWAP_AB)
    (void)out_bf16;
    (void)workspace;
    (void)workspace_capacity;
    (void)packed_activation;
    (void)activation_scales;
    (void)packed_weight;
    (void)weight_scales;
    (void)combined_global_scale;
    (void)rows;
    (void)out_cols;
    (void)inner_cols;
    (void)stream;
    return false;
#else
    if (!out_bf16 || !packed_activation || !activation_scales ||
        !packed_weight || !weight_scales || !combined_global_scale ||
        rows == 0 || out_cols == 0 || inner_cols == 0 ||
        inner_cols % 64 != 0 || out_cols % 32 != 0) {
        return false;
    }
    try {
        const bool use_stream_k = rows < 128;
        const size_t required = workspace_bytes(rows, out_cols, inner_cols);
        if (required > workspace_capacity) return false;
        dispatch_sm120_bf16(
            static_cast<__nv_bfloat16 *>(out_bf16),
            packed_activation, packed_weight, activation_scales, weight_scales,
            combined_global_scale, rows, out_cols, inner_cols, use_stream_k,
            static_cast<char *>(workspace), required, stream);
        return cudaGetLastError() == cudaSuccess;
    } catch (const std::runtime_error &) {
        return false;
    }
#endif
}

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
                      cudaStream_t stream) {
    if (!out_f32 || !packed_activation || !activation_scales || !packed_weight ||
        !weight_scales || !combined_global_scale || rows == 0 || rows > 8 ||
        out_cols == 0 || inner_cols == 0 || inner_cols % 64 != 0 ||
        out_cols % 32 != 0 || output_stride < out_cols) {
        return false;
    }
    if (add_to_output) {
        return dispatch_small_f32<true>(
            out_f32, packed_activation, activation_scales, packed_weight,
            weight_scales, combined_global_scale, rows, out_cols, inner_cols,
            output_stride, stream);
    }
    return dispatch_small_f32<false>(
        out_f32, packed_activation, activation_scales, packed_weight,
        weight_scales, combined_global_scale, rows, out_cols, inner_cols,
        output_stride, stream);
}

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
                       cudaStream_t stream) {
    if (!out_bf16 || !packed_activation || !activation_scales ||
        !packed_weight || !weight_scales || !combined_global_scale ||
        rows == 0 || rows > 8 || out_cols == 0 || inner_cols == 0 ||
        inner_cols % 64 != 0 || out_cols % 32 != 0 ||
        output_stride < out_cols) {
        return false;
    }
    if (add_to_output) {
        return dispatch_small_bf16<true>(
            out_bf16, packed_activation, activation_scales, packed_weight,
            weight_scales, combined_global_scale, rows, out_cols, inner_cols,
            output_stride, stream);
    }
    return dispatch_small_bf16<false>(
        out_bf16, packed_activation, activation_scales, packed_weight,
        weight_scales, combined_global_scale, rows, out_cols, inner_cols,
        output_stride, stream);
}

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
                           cudaStream_t stream) {
    if (!out0_f32 || (!silu_mul_output && !out1_f32) ||
        !packed_activation || !activation_scales ||
        !packed_weight0 || !weight_scales0 || !combined_global_scale0 ||
        !packed_weight1 || !weight_scales1 || !combined_global_scale1 ||
        rows == 0 || rows > 8 || out_cols == 0 || inner_cols == 0 ||
        inner_cols % 64 != 0 || out_cols % 32 != 0 ||
        output_stride0 < out_cols ||
        (!silu_mul_output && output_stride1 < out_cols)) {
        return false;
    }
    if (silu_mul_output) {
        return dispatch_small_pair_f32<true>(
            out0_f32, out1_f32, packed_activation, activation_scales,
            packed_weight0, weight_scales0, combined_global_scale0,
            packed_weight1, weight_scales1, combined_global_scale1,
            rows, out_cols, inner_cols, output_stride0, output_stride1, stream);
    }
    return dispatch_small_pair_f32<false>(
        out0_f32, out1_f32, packed_activation, activation_scales,
        packed_weight0, weight_scales0, combined_global_scale0,
        packed_weight1, weight_scales1, combined_global_scale1,
        rows, out_cols, inner_cols, output_stride0, output_stride1, stream);
}

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
            cudaStream_t stream) {
    if (!out_f32 || !packed_activation || !activation_scales || !input_f32 ||
        !packed_weight || !weight_scales || !input_global_scale_divisor ||
        !combined_global_scale || rows == 0 || out_cols == 0 || inner_cols == 0 ||
        inner_cols % 64 != 0 || out_cols % 32 != 0 || input_stride < inner_cols) {
        return false;
    }

    if (!quantize_activation(packed_activation, activation_scales, input_f32,
                             input_global_scale_divisor, rows, inner_cols,
                             input_stride, false, stream)) {
        return false;
    }

    try {
        const bool use_stream_k = rows < 128;
        const size_t required = workspace_bytes(rows, out_cols, inner_cols);
        if (required > workspace_capacity) return false;
        dispatch_sm120(out_f32, packed_activation, packed_weight,
                       activation_scales, weight_scales, combined_global_scale,
                       rows, out_cols, inner_cols, use_stream_k,
                       static_cast<char *>(workspace), required, stream);
        return cudaGetLastError() == cudaSuccess;
    } catch (const std::runtime_error &) {
        return false;
    }
}

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
                 cudaStream_t stream) {
#if !defined(QW3_FLASHINFER_SM120_SWAP_AB)
    (void)out_bf16;
    (void)packed_activation;
    (void)activation_scales;
    (void)workspace;
    (void)workspace_capacity;
    (void)input_f32;
    (void)input_stride;
    (void)packed_weight;
    (void)weight_scales;
    (void)input_global_scale_divisor;
    (void)combined_global_scale;
    (void)rows;
    (void)out_cols;
    (void)inner_cols;
    (void)stream;
    return false;
#else
    if (!out_bf16 || !packed_activation || !activation_scales || !input_f32 ||
        !packed_weight || !weight_scales || !input_global_scale_divisor ||
        !combined_global_scale || rows == 0 || out_cols == 0 || inner_cols == 0 ||
        input_stride < inner_cols) {
        return false;
    }
    if (!quantize_activation(packed_activation, activation_scales, input_f32,
                             input_global_scale_divisor, rows, inner_cols,
                             input_stride, true, stream)) {
        return false;
    }
    try {
        const bool use_stream_k = rows < 128;
        const size_t required = workspace_bytes(rows, out_cols, inner_cols);
        if (required > workspace_capacity) return false;
        dispatch_sm120_bf16(
            static_cast<__nv_bfloat16 *>(out_bf16),
            packed_activation, packed_weight, activation_scales, weight_scales,
            combined_global_scale, rows, out_cols, inner_cols, use_stream_k,
            static_cast<char *>(workspace), workspace_capacity, stream);
        return true;
    } catch (const std::runtime_error &) {
        return false;
    }
#endif
}

} // namespace qw3::nvfp4_adapter
