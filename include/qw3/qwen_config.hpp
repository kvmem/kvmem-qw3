#pragma once

#include "qw3/gguf.hpp"

#include <cstdint>
#include <string>

namespace qw3 {

/* Qwen35 hyperparameters resolved from GGUF metadata.
 *
 * All values are read from the file at load time rather than baked in as
 * constexpr. If a key is missing the constructor throws; we want a hard
 * failure on shape mismatches instead of silently producing wrong logits. */
struct QwenConfig {
    std::string architecture; // "qwen35"

    // Block layout
    uint32_t block_count = 0; // GGUF block_count, including trailing MTP blocks
    uint32_t nextn_predict_layers = 0;
    uint32_t n_layers = 0; // main transformer layers executed by qwen-native
    uint32_t n_embd = 0;
    uint32_t n_ff = 0;
    uint32_t n_ctx_train = 0;

    // Standard attention
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;  // key_length; the model also uses this as v dim
    uint32_t head_v_dim = 0; // value_length
    float    rms_eps = 1e-6f;

    // RoPE
    uint32_t rope_dim = 0;     // qwen35.rope.dimension_count (partial RoPE)
    float    rope_theta = 1e7f;

    // SSM / DeltaNet (recurrent layers)
    uint32_t ssm_conv_kernel = 0;
    uint32_t ssm_inner_size = 0;     // sum of K+V channels into the SSM core
    uint32_t ssm_state_size = 0;     // per-head state dim (head_k_dim==head_v_dim==128)
    uint32_t ssm_time_step_rank = 0; // num_v_heads
    uint32_t ssm_group_count = 0;    // num_k_heads

    // Hybrid layout
    uint32_t full_attention_interval = 0; // every k-th layer is full attention

    // Vocabulary
    uint32_t vocab_size = 0;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    bool     add_bos = false;

    explicit QwenConfig(const GgufFile &gguf);

    bool is_standard_attention_layer(uint32_t i) const {
        if (full_attention_interval == 0) return false;
        // Pattern observed in Qwen3.6-27B: 3 recurrent, 1 standard, ...
        // i.e. layer is standard iff (i+1) % full_attention_interval == 0.
        return ((i + 1) % full_attention_interval) == 0;
    }

    // Recurrent / DeltaNet derived shapes.
    uint32_t num_v_heads() const { return ssm_time_step_rank; }
    uint32_t num_k_heads() const { return ssm_group_count; }
    uint32_t head_k_dim() const { return ssm_state_size; }
    uint32_t head_v_dim_ssm() const { return ssm_state_size; }
};

} // namespace qw3
