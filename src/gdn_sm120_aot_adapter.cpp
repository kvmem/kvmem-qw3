#include "gdn_sm120_aot_adapter.hpp"

#include "gdn_sm120_0.h"

#include <mutex>

namespace qw3::gdn_sm120_aot {
namespace {

qw3_gdn_sm120_0_Kernel_Module_t g_module{};
std::once_flag g_module_once;

void load_module() {
    qw3_gdn_sm120_0_Kernel_Module_Load(&g_module);
}

} // namespace

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
    cudaStream_t stream) {
    if (!out_bf16 || !output_state || !q_bf16 || !k_bf16 || !v_bf16 ||
        !initial_state || !decay || !beta || !cu_seqlens || !tensormaps ||
        tokens == 0 || num_q_heads == 0 || num_k_heads == 0 ||
        num_v_heads == 0 || head_dim != 128) {
        return false;
    }

    int major = 0;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        major != 12) {
        return false;
    }

    std::call_once(g_module_once, load_module);

    const int32_t T = static_cast<int32_t>(tokens);
    const int32_t D = static_cast<int32_t>(head_dim);
    const int32_t HQ = static_cast<int32_t>(num_q_heads);
    const int32_t HK = static_cast<int32_t>(num_k_heads);
    const int32_t HV = static_cast<int32_t>(num_v_heads);

    qw3_gdn_sm120_0_Tensor_g_q_t q_desc{
        const_cast<void *>(q_bf16),
        {T, D, HQ},
        {static_cast<int64_t>(HQ) * D, D}};
    qw3_gdn_sm120_0_Tensor_g_k_t k_desc{
        const_cast<void *>(k_bf16),
        {D, T, HK},
        {static_cast<int64_t>(HK) * D, D}};
    qw3_gdn_sm120_0_Tensor_g_v_t v_desc{
        const_cast<void *>(v_bf16),
        {D, T, HV},
        {static_cast<int64_t>(HV) * D, D}};
    qw3_gdn_sm120_0_Tensor_g_o_t out_desc{
        out_bf16,
        {D, T, HV},
        {static_cast<int64_t>(HV) * D, D}};
    qw3_gdn_sm120_0_Tensor_g_alpha_t decay_desc{
        decay, {T * HV}};
    qw3_gdn_sm120_0_Tensor_g_beta_t beta_desc{
        beta, {T * HV}};
    qw3_gdn_sm120_0_Tensor_g_state_t output_state_desc{
        output_state, {HV * D * D}};
    qw3_gdn_sm120_0_Tensor_g_init_state_t initial_state_desc{
        const_cast<float *>(initial_state), {HV * D * D}};
    qw3_gdn_sm120_0_Tensor_g_tensormaps_t tensormaps_desc{
        tensormaps, {static_cast<int32_t>(tensormaps_bytes)}};
    qw3_gdn_sm120_0_Tensor_cu_seqlens_t cu_desc{
        cu_seqlens, {2}};

    const int32_t status = cute_dsl_qw3_gdn_sm120_0_wrapper(
        &g_module,
        &q_desc,
        &k_desc,
        &v_desc,
        &out_desc,
        &decay_desc,
        &beta_desc,
        &output_state_desc,
        &initial_state_desc,
        &tensormaps_desc,
        &cu_desc,
        scale,
        HQ,
        HK,
        HV,
        HV,
        1,
        1,
        0,
        HV,
        stream);
    return status == 0 && cudaPeekAtLastError() == cudaSuccess;
}

} // namespace qw3::gdn_sm120_aot
