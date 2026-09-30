#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

struct DFlashConfig {
    explicit DFlashConfig(const std::string &directory);

    void validate_target(uint32_t layers, uint32_t hidden, uint32_t vocab) const;

    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_target_layers = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t vocab_size = 0;
    uint32_t max_position_embeddings = 0;
    uint32_t sliding_window = 0;
    uint32_t block_size = 0;  // Includes the committed anchor.
    uint32_t mask_token_id = 0;
    uint32_t conv_kernel_size = 0;
    uint32_t conv_group_size = 0;
    uint32_t selector_rank = 0;
    uint32_t selector_top_k = 0;
    float rms_norm_eps = 0;
    float rope_theta = 0;
    std::vector<uint32_t> target_layer_ids; // Zero-based, post-residual outputs.
};

} // namespace qw3
