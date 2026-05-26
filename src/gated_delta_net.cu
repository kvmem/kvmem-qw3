// SPDX-License-Identifier: MIT
//
// Gated DeltaNet recurrent layer — single kernel, batched over T tokens.
//
// Ported from llama.cpp ggml/src/ggml-cuda/gated_delta_net.cu @ commit
// 57ebaf4edd99ea675f256ae2286cd99206dbfcd1 (MIT, see LICENSES/llama.cpp.txt).
//
// Adaptations from upstream:
//   - Single-sequence (n_seqs = 1, sequence index always 0).
//   - Drop the KDA branch (Qwen 3.6 uses a scalar gate per v-head).
//   - Drop the keep_rs_t branch (qw3 only stores final state, K = 1).
//   - Drop the ggml_tensor wrapper; expose a raw-pointer launcher that
//     callers in src/kernels_cuda.cu use directly.
//   - Add prep_log_g_sigmoid_beta_kernel: qw3 stores raw alpha and beta;
//     the upstream kernel expects log(g) and sigmoid(beta) ready to use.
//     This small per-(t, vh) kernel converts in place before the main
//     kernel runs.
//
// The state and (q, k, v, alpha, beta) layouts already match qw3's
// recurrent_batch convention — see kernels_cuda.cu recurrent_batch().

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

#include "cuda_helpers.cuh"

namespace qw3 {
namespace ported {

// In-place prep: convert qw3's raw alpha and beta into the (log_g,
// sigmoid_beta) form expected by gated_delta_net_kernel.
//
//   log_g[t, vh]      = softplus(alpha[t, vh] + dt_bias[vh]) * ssm_a[vh]
//   sigmoid_beta[t,v] = 1 / (1 + exp(-beta[t, vh]))
//
// Both arrays are read with stride [T, num_v_heads]; rewritten in place.
__global__ void prep_log_g_sigmoid_beta_kernel(
        float *       alpha,        // [T, num_v_heads], stride alpha_stride
        float *       beta,         // [T, num_v_heads], stride beta_stride
        const float * dt_bias,      // [num_v_heads]
        const float * ssm_a,        // [num_v_heads]
        uint32_t      T,
        uint32_t      num_v_heads,
        uint32_t      alpha_stride,
        uint32_t      beta_stride) {
    const uint32_t t  = blockIdx.x;
    const uint32_t vh = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || vh >= num_v_heads) return;

    const uint32_t a_idx = t * alpha_stride + vh;
    const uint32_t b_idx = t * beta_stride  + vh;

    // softplus(x) = log1p(exp(x)) is unstable for large x; the math library
    // log1pf + expf combo handles the moderate range we hit here cleanly.
    const float a_raw = alpha[a_idx];
    const float bias  = dt_bias[vh];
    const float a_v   = ssm_a[vh];
    alpha[a_idx] = log1pf(expf(a_raw + bias)) * a_v;

    const float b_raw = beta[b_idx];
    beta[b_idx] = 1.0f / (1.0f + expf(-b_raw));
}

// One warp owns one column of the per-head S_v×S_v state; rows distributed
// across the warp's 32 lanes. State held in registers across all T steps.
// No __syncthreads() in the inner loop — both reductions are warp-shuffles.
template <int S_v>
__global__ void
__launch_bounds__((32 < S_v ? 32 : S_v) * 4, 2)
gated_delta_net_kernel(
        const float * q,            // [T, num_k_heads, S_v]
        const float * k,            // [T, num_k_heads, S_v]
        const float * v,            // [T, num_v_heads, S_v]
        const float * log_g,        // [T, num_v_heads]
        const float * sigmoid_beta, // [T, num_v_heads]
        float *       state,        // [num_v_heads, S_v, S_v]
        float *       out,          // [T, num_v_heads, S_v]
        uint32_t      T,
        uint32_t      qkv_row_stride,    // stride between t-rows in q/k (floats)
        uint32_t      v_row_stride,      // stride between t-rows in v (floats)
        uint32_t      gb_row_stride,     // stride between t-rows in log_g/beta
        uint32_t      out_row_stride,    // stride between t-rows in out (floats)
        uint3         num_k_heads_fastdiv,
        float         scale) {
    constexpr int warp_size    = 32 < S_v ? 32 : S_v;
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");

    const uint32_t vh   = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t col  = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= S_v) return;

    const uint32_t kh = cuda_helpers::fastmodulo(vh, num_k_heads_fastdiv);

    // State is laid out as state[vh][col*S_v + i]. Column-major within a head:
    // each warp's column is contiguous; rows striped across lanes.
    float * state_head = state + static_cast<uint64_t>(vh) * S_v * S_v;

    // Load this warp's column into registers.
    float s_shard[rows_per_lane];
    #pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        s_shard[r] = state_head[col * S_v + i];
    }

    // Per-token strides for this (kh, vh).
    const uint32_t q_head_off = kh * S_v;
    const uint32_t k_head_off = kh * S_v;
    const uint32_t v_head_off = vh * S_v;

    for (uint32_t t = 0; t < T; ++t) {
        const float * q_t = q + t * qkv_row_stride + q_head_off;
        const float * k_t = k + t * qkv_row_stride + k_head_off;
        const float * v_t = v + t * v_row_stride   + v_head_off;

        const float g_val    = expf(log_g[t * gb_row_stride + vh]);
        const float beta_val = sigmoid_beta[t * gb_row_stride + vh];

        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        // kv[col] = sum_i S[i][col] * k[i]
        float kv_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            kv_partial += s_shard[r] * k_reg[r];
        }
        const float kv_col = cuda_helpers::warp_reduce_sum<warp_size>(kv_partial);

        // delta = (v[col] - g * kv[col]) * beta
        const float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

        // Fused decay + rank-1 update + S^T @ q.
        float attn_partial = 0.0f;
        #pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            s_shard[r]    = g_val * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = cuda_helpers::warp_reduce_sum<warp_size>(attn_partial);

        if (lane == 0) {
            out[t * out_row_stride + vh * S_v + col] = attn_col * scale;
        }
    }

    // Persist final state.
    #pragma unroll
    for (int r = 0; r < rows_per_lane; ++r) {
        const int i = r * warp_size + lane;
        state_head[col * S_v + i] = s_shard[r];
    }
}

// qw3-facing launcher. Returns true on success; false if S_v is unsupported.
bool launch_gated_delta_net(
        float *       alpha_inout,        // raw alpha → overwritten with log_g
        float *       beta_inout,         // raw beta  → overwritten with sigmoid_beta
        const float * dt_bias,
        const float * ssm_a,
        const float * conv_qkv,           // packed [T, recurrent_qkv_dim]
        uint32_t      q_offset,           // start of q rows (floats)
        uint32_t      k_offset,           // start of k rows (floats)
        uint32_t      v_offset,           // start of v rows (floats)
        float *       state,              // [num_v_heads, S_v, S_v]
        float *       core_out,           // [T, num_v_heads * head_v_dim]
        uint32_t      T,
        uint32_t      num_k_heads,
        uint32_t      num_v_heads,
        uint32_t      head_dim,           // == head_k_dim == head_v_dim
        uint32_t      qkv_row_stride,
        uint32_t      gb_row_stride,
        uint32_t      out_row_stride,
        cudaStream_t  stream) {
    if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return false;
    }

    // Step 1: prep alpha/beta in-place.
    {
        const uint32_t threads = 32;
        const dim3 grid(T, (num_v_heads + threads - 1) / threads);
        prep_log_g_sigmoid_beta_kernel<<<grid, threads, 0, stream>>>(
            alpha_inout, beta_inout, dt_bias, ssm_a,
            T, num_v_heads, gb_row_stride, gb_row_stride);
    }

    // Step 2: ported delta kernel.
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const uint3 fd = cuda_helpers::init_fastdiv_values(num_k_heads);
    constexpr int num_warps = 4;

    auto launch = [&](auto S_v_const) {
        constexpr int S_v = decltype(S_v_const)::value;
        const int warp_size = 32 < S_v ? 32 : S_v;
        const dim3 grid(num_v_heads, 1, (S_v + num_warps - 1) / num_warps);
        const dim3 block(warp_size, num_warps, 1);
        gated_delta_net_kernel<S_v><<<grid, block, 0, stream>>>(
            conv_qkv + q_offset,
            conv_qkv + k_offset,
            conv_qkv + v_offset,
            alpha_inout,    // now log_g
            beta_inout,     // now sigmoid_beta
            state, core_out,
            T,
            qkv_row_stride, qkv_row_stride, gb_row_stride, out_row_stride,
            fd, scale);
    };

    switch (head_dim) {
        case 16:  launch(std::integral_constant<int, 16>{});  break;
        case 32:  launch(std::integral_constant<int, 32>{});  break;
        case 64:  launch(std::integral_constant<int, 64>{});  break;
        case 128: launch(std::integral_constant<int, 128>{}); break;
        default:  return false;
    }
    return true;
}

} // namespace ported
} // namespace qw3
