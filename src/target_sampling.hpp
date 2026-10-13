#pragma once

#include "qw3/qw3.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace qw3::detail {

// Full-vocab probability vector that token sampling draws from: softmax(logits/temp)
// with top-k/top-p/min-p truncation applied and renormalized over the kept set.
// Entries outside the kept set are 0. Penalties are NOT applied here — callers
// penalize the logits first (matching apply_token_penalties usage). This is the
// single source of truth for the sampling distribution, shared by the non-MTP
// sampler (sample_token) and the MTP speculative accept test, so MTP under temp>0
// reproduces exactly the same distribution as plain decode.
inline std::vector<float> sampling_distribution(const std::vector<float> &logits,
                                         float temp, float top_p, int top_k,
                                         float min_p, SamplingFilterOrder filter_order) {
    const int n = static_cast<int>(logits.size());
    std::vector<float> out(n > 0 ? static_cast<size_t>(n) : 0, 0.0f);
    if (n <= 0) return out;
    // softmax(logits / temp), numerically stabilized by max subtraction.
    const float inv_t = 1.0f / (temp > 0.0f ? temp : 1.0f);
    float maxv = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];
    // Top-k-first uses only the retained top-k mass for its top-p cutoff.
    // Joint keeps the historical full-vocabulary normalization.
    std::vector<int> idx;
    size_t cap = static_cast<size_t>(n);
    if (top_k > 0 && static_cast<size_t>(top_k) < cap) cap = static_cast<size_t>(top_k);
    auto by_logit_desc = [&](int a, int b) { return logits[a] > logits[b]; };
    const bool normalize_top_k = filter_order == SamplingFilterOrder::TopKFirst && cap < size_t(n);
    double sum = 0.0;
    if (normalize_top_k) {
        idx.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) idx[i] = i;
        std::nth_element(idx.begin(), idx.begin() + cap, idx.end(), by_logit_desc);
        for (size_t i = 0; i < cap; ++i) sum += std::exp((logits[idx[i]] - maxv) * inv_t);
        idx.resize(cap);
    } else {
        for (int i = 0; i < n; ++i) sum += std::exp((logits[i] - maxv) * inv_t);
    }
    const float norm = static_cast<float>(1.0 / (sum > 0.0 ? sum : 1.0));
    auto prob_of = [&](int i) -> float {
        return std::exp((logits[i] - maxv) * inv_t) * norm;
    };
    const bool need_sort =
        (top_k > 0 && top_k < n) ||
        (top_p < 1.0f && top_p > 0.0f) ||
        (min_p > 0.0f);
    if (!need_sort) {
        // No truncation: emit the full-vocab softmax directly.
        for (int i = 0; i < n; ++i) out[i] = prob_of(i);
        return out;
    }
    // Truncation only needs the highest-probability tokens, so avoid the O(V log V)
    // full sort: isolate the top candidates with O(V) partial selection and sort
    // only that window. exp is monotonic, so ranking by logit equals ranking by
    // probability; the kept set + renormalization below is identical to sorting the
    // whole vocab. If a filter would keep the entire window, widen it and retry —
    // that bounds the answer exactly. Truncation order matches the original
    // sample_token: top_k, then min_p, then nucleus (top_p); each keeps a prefix.
    if (idx.empty()) {
        idx.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) idx[i] = i;
    }
    size_t K = std::min<size_t>(cap, 128);
    if (K == 0) K = std::min<size_t>(cap, 1);
    size_t keep = 0;
    for (;;) {
        if (K > cap) K = cap;
        if (K < idx.size())
            std::nth_element(idx.begin(), idx.begin() + K, idx.end(), by_logit_desc);
        std::sort(idx.begin(), idx.begin() + K, by_logit_desc);
        keep = K;
        if (top_k > 0 && static_cast<size_t>(top_k) < keep) keep = static_cast<size_t>(top_k);
        if (min_p > 0.0f && keep > 0) {
            const float cutoff = prob_of(idx[0]) * min_p;
            size_t k = 0;
            while (k < keep && prob_of(idx[k]) >= cutoff) ++k;
            keep = k;
        }
        if (top_p < 1.0f && top_p > 0.0f) {
            double cum = 0.0;
            size_t k = keep;
            for (size_t i = 0; i < keep; ++i) {
                cum += prob_of(idx[i]);
                if (cum >= top_p) { k = i + 1; break; }
            }
            // FlashInfer's nucleus filter keeps all candidates tied with the
            // cutoff probability. Preserve the historical joint policy, while
            // making the explicit upstream-compatible policy match this rule.
            if (filter_order == SamplingFilterOrder::TopKFirst && k > 0) {
                const float cutoff = prob_of(idx[k - 1]);
                while (k < keep && prob_of(idx[k]) == cutoff) ++k;
            }
            keep = k;
        }
        // Exact once the kept set fits strictly inside the window (or the window
        // already spans the whole eligible vocab / top_k cap).
        if (keep < K || K >= cap) break;
        K = std::min<size_t>(cap, K * 2);
    }
    double kept_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) kept_sum += prob_of(idx[i]);
    if (kept_sum <= 0.0) {
        // Degenerate (e.g. min_p removed everything): collapse to the top token.
        out[idx[0]] = 1.0f;
        return out;
    }
    const float kn = static_cast<float>(1.0 / kept_sum);
    for (size_t i = 0; i < keep; ++i) out[idx[i]] = prob_of(idx[i]) * kn;
    return out;
}

} // namespace qw3::detail
