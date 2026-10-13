#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qw3 {

namespace detail {
inline std::vector<int32_t> parse_attention_layer_ids(std::string_view text) {
    std::vector<int32_t> ids;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto part = text.substr(0, comma);
        int32_t id = -1;
        const auto [end, error] = std::from_chars(part.data(), part.data() + part.size(), id);
        if (part.empty() || error != std::errc{} || end != part.data() + part.size() || id < 0)
            throw std::invalid_argument("--kvmem-action-layers must be last|spread3|all|comma-separated standard layer IDs");
        ids.push_back(id);
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
        if (text.empty()) throw std::invalid_argument("empty attention layer ID");
    }
    if (ids.empty()) throw std::invalid_argument("empty attention layer policy");
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::invalid_argument("duplicate attention layer ID");
    return ids;
}

inline std::vector<int32_t> resolve_attention_layers(std::string_view policy,
        const std::vector<int32_t> &standard) {
    if (standard.empty()) throw std::invalid_argument("attention capture needs a standard attention layer");
    if (policy == "last") return {standard.back()};
    if (policy == "all") return standard;
    if (policy == "spread3") {
        std::vector<int32_t> ids{standard.front(), standard[(standard.size()-1)/2], standard.back()};
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        return ids;
    }
    auto ids = parse_attention_layer_ids(policy);
    for (int32_t id : ids)
        if (!std::binary_search(standard.begin(), standard.end(), id))
            throw std::invalid_argument("attention layer ID is not a standard attention layer in this model");
    return ids;
}
} // namespace detail

// Shared by the public engine and executor; no process-global environment state.
struct KvMemAttentionConfig {
    bool enabled = false;
    std::string scope = "tool-call";
    std::string layers = "spread3"; // equal mean across selected standard attention layers
    std::string combine = "quota"; // quota allocation or normalized score fusion
    double fraction = 0.25; // attention quota share or fused-score weight
    double decay = 0.5;    // per eligible output response
    uint32_t interval = 16;
    bool pressure = false;
    bool prefill = false;
    uint32_t prefill_interval = 1024;
    uint32_t prefill_stride = 32; // 0: last row only; N: one causal query every N rows
    double prefill_half_life = 8192; // input tokens

    void validate() const {
        if (layers != "last" && layers != "spread3" && layers != "all")
            (void)detail::parse_attention_layer_ids(layers);
        if (scope != "tool-call" && scope != "non-thinking")
            throw std::invalid_argument("--kvmem-action-scope must be tool-call|non-thinking");
        if (combine != "quota" && combine != "score")
            throw std::invalid_argument("--kvmem-action-combine must be quota|score");
        if (!std::isfinite(fraction) || fraction < 0 || fraction > 1)
            throw std::invalid_argument("--kvmem-action-fraction must be in [0,1]");
        if (!std::isfinite(decay) || decay < 0 || decay > 1)
            throw std::invalid_argument("--kvmem-action-decay must be in [0,1]");
        if (!interval || interval > 4096)
            throw std::invalid_argument("--kvmem-action-interval must be in [1,4096]");
        if (!prefill_interval || prefill_interval > 1000000000)
            throw std::invalid_argument("--kvmem-prefill-attention-interval must be in [1,1000000000]");
        if (prefill_stride > 1000000000)
            throw std::invalid_argument("--kvmem-prefill-attention-stride must be in [0,1000000000]");
        if (!std::isfinite(prefill_half_life) || prefill_half_life < 1 || prefill_half_life > 1e9)
            throw std::invalid_argument("--kvmem-prefill-attention-half-life must be in [1,1000000000]");
        if ((pressure || prefill) && !enabled)
            throw std::invalid_argument("pressure/prefill attention requires --kvmem-action-attention on");
    }
};

} // namespace qw3
