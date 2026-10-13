#pragma once
#include <cstdint>
#include <vector>

namespace qw3 {
// One target row per proposed token, followed by the all-accept bonus row.
// Zero proposals is ordinary sampling. Penalties include the committed prefix
// plus the hypothetical accepted proposal prefix for each independent row.
struct DeviceSamplingRequest {
    uint32_t vocab = 0;
    float temperature = 1, top_p = 1, min_p = 0;
    int top_k = 0;
    float presence_penalty = 0, repetition_penalty = 1;
    bool top_k_first = false;
    std::vector<uint32_t> seen;
    std::vector<int32_t> drafts, candidate_ids;
    std::vector<float> proposal_probabilities;
    uint32_t candidates_per_row = 0;
    bool residual_always_draw = true; // DFlash; point-mass MTP skips singleton draws.
    // Consecutive draws from a COPY of the request RNG. Commit only rng_draws
    // after a successful GPU result; a CPU fallback must see the original RNG.
    std::vector<double> uniforms;
    bool collect_probabilities = false; // Numerical oracle only, never serving.
};
struct DeviceSamplingResult {
    int32_t accepted = 0, token = -1, rng_draws = 0, fallback = 0;
    std::vector<float> probabilities;
    std::vector<int32_t> row_fallbacks; // Only populated with collect_probabilities.
};
} // namespace qw3
