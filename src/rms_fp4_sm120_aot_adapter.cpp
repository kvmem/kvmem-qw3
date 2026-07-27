#include "rms_fp4_sm120_aot_adapter.hpp"

#include "rms_fp4_sm120_0.h"

#include <mutex>

namespace qw3::rms_fp4_sm120_aot {
namespace {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kScaleBlocks = kHidden / 16;
constexpr uint32_t kScaleTiles = (kScaleBlocks + 3) / 4;

qw3_rms_fp4_sm120_0_Kernel_Module_t g_module{};
std::once_flag g_module_once;

void load_module() {
    qw3_rms_fp4_sm120_0_Kernel_Module_Load(&g_module);
}

} // namespace

bool launch(void *packed_fp4,
            void *block_scales,
            const void *input_bf16,
            const void *weight_bf16,
            const float *global_scale_divisor,
            uint32_t rows,
            uint32_t hidden,
            float eps,
            cudaStream_t stream) {
    if (!packed_fp4 || !block_scales || !input_bf16 || !weight_bf16 ||
        !global_scale_divisor || rows == 0 || hidden != kHidden) {
        return false;
    }

    int device = 0;
    int major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        major != 12) {
        return false;
    }
    std::call_once(g_module_once, load_module);

    const uint32_t padded_rows = ((rows + 127U) / 128U) * 128U;
    const uint64_t scale_bytes =
        static_cast<uint64_t>(padded_rows) * kScaleTiles * 4U;
    if (scale_bytes > INT32_MAX || rows > INT32_MAX) {
        return false;
    }

    qw3_rms_fp4_sm120_0_Tensor_mX_t x_desc{
        const_cast<void *>(input_bf16),
        {static_cast<int32_t>(rows)}};
    qw3_rms_fp4_sm120_0_Tensor_mW_t weight_desc{
        const_cast<void *>(weight_bf16)};
    qw3_rms_fp4_sm120_0_Tensor_mY_t packed_desc{
        packed_fp4,
        {static_cast<int32_t>(rows)}};
    qw3_rms_fp4_sm120_0_Tensor_mS_t scale_desc{
        block_scales,
        {static_cast<int32_t>(scale_bytes)}};
    qw3_rms_fp4_sm120_0_Tensor_mGlobalScale_t global_scale_desc{
        const_cast<float *>(global_scale_divisor)};

    const int32_t status = cute_dsl_qw3_rms_fp4_sm120_0_wrapper(
        &g_module,
        &x_desc,
        &weight_desc,
        &packed_desc,
        &scale_desc,
        &global_scale_desc,
        static_cast<int32_t>(rows),
        eps,
        stream);
    return status == 0 && cudaPeekAtLastError() == cudaSuccess;
}

} // namespace qw3::rms_fp4_sm120_aot
