#pragma once
#include "qw3/kvmem_attention.hpp"

namespace qw3 {
// Observation only. None of these fields are inputs to the selection policy.
struct KvMemMismatchConfig {
    std::string log_path;
    std::string layers = "spread3";
    std::string query_mode = "sampled";
    uint32_t interval = 1024;
    uint32_t samples = 4;
    double half_life = 8192;
    double epsilon = 0.10;
    bool enabled() const { return !log_path.empty(); }
    void validate() const {
        if (query_mode != "sampled" && query_mode != "block-mean")
            throw std::invalid_argument("--kvmem-mismatch-query-mode must be sampled|block-mean");
        if (layers != "last" && layers != "spread3" && layers != "all")
            (void)detail::parse_attention_layer_ids(layers);
        if (!interval || interval > 1000000000)
            throw std::invalid_argument("--kvmem-mismatch-interval must be in [1,1000000000]");
        if (!samples || samples > 4)
            throw std::invalid_argument("--kvmem-mismatch-samples must be in [1,4]");
        if (!std::isfinite(half_life) || half_life < 1 || half_life > 1e9)
            throw std::invalid_argument("--kvmem-mismatch-half-life must be in [1,1000000000]");
        if (!std::isfinite(epsilon) || epsilon < 0 || epsilon >= 1)
            throw std::invalid_argument("--kvmem-mismatch-epsilon must be in [0,1)");
    }
};
}
