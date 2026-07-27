#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace qw3::rms_fp4_sm120_aot {

bool launch(void *packed_fp4,
            void *block_scales,
            const void *input_bf16,
            const void *weight_bf16,
            const float *global_scale_divisor,
            uint32_t rows,
            uint32_t hidden,
            float eps,
            cudaStream_t stream);

} // namespace qw3::rms_fp4_sm120_aot
