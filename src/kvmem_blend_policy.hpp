#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

struct KvMemBlendSpan {
    uint64_t history = 0;
    uint32_t window = 0;
    uint32_t tokens = 0;
    double score = 0;
};

// Whole retrieval blocks keep the existing contiguous causal prefill kernels.
// Rank by the already computed query score, then execute in causal order.
inline std::vector<KvMemBlendSpan> kvmem_blend_plan(
        std::vector<KvMemBlendSpan> candidates, uint32_t window_tokens,
        double ratio, uint32_t max_tokens) {
    if (!std::isfinite(ratio) || ratio < 0 || ratio > 1)
        throw std::invalid_argument("KVMem blend ratio must be in [0,1]");
    const uint32_t budget = std::min<uint32_t>(
        max_tokens, static_cast<uint32_t>(std::floor(window_tokens * ratio)));
    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
        const double sa = std::isfinite(a.score) ? a.score : 0;
        const double sb = std::isfinite(b.score) ? b.score : 0;
        return sa != sb ? sa > sb : a.window < b.window;
    });
    std::vector<KvMemBlendSpan> selected;
    uint32_t used = 0;
    for (const auto &span : candidates) {
        if (span.tokens == 0 || span.window > window_tokens ||
            span.tokens > window_tokens - span.window || span.tokens > budget - used)
            continue;
        selected.push_back(span);
        used += span.tokens;
    }
    std::sort(selected.begin(), selected.end(), [](const auto &a, const auto &b) {
        return a.window < b.window;
    });
    return selected;
}

}  // namespace qw3::detail
