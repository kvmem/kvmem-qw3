#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

// One conditional rejection-sampling step. Target probabilities must already
// include the request's temperature, truncation and prefix penalties. Draft
// probabilities are the actual selector distribution for this predecessor,
// including zero mass outside its sparse candidate set. This also works for
// point-mass proposals and greedy target distributions.
struct SparseRejectionStep {
    double accept_probability = 0;
    std::vector<double> residual; // Unnormalized max(p - q, 0).

    bool accepts(double uniform) const {
        if (!(uniform >= 0 && uniform < 1))
            throw std::invalid_argument("acceptance uniform outside [0, 1)");
        return uniform < accept_probability;
    }

    int32_t residual_token(double uniform) const {
        if (!(uniform >= 0 && uniform < 1))
            throw std::invalid_argument("residual uniform outside [0, 1)");
        double total = 0;
        for (double p : residual) total += p;
        if (!(total > 0)) throw std::logic_error("rejection has no residual mass");
        const double draw = uniform * total;
        double cumulative = 0;
        int32_t last = -1;
        for (size_t i = 0; i < residual.size(); ++i) {
            if (residual[i] <= 0) continue;
            last = static_cast<int32_t>(i);
            cumulative += residual[i];
            if (draw < cumulative) return last;
        }
        return last; // Floating-point summation at the upper CDF boundary.
    }
};

inline SparseRejectionStep sparse_rejection_step(
        const std::vector<float> &target,
        const int32_t *candidate_ids, const float *proposal,
        uint32_t candidate_count, int32_t proposed_token) {
    if (target.empty() || !candidate_ids || !proposal || !candidate_count ||
        proposed_token < 0 || size_t(proposed_token) >= target.size())
        throw std::invalid_argument("invalid sparse rejection shape or token");
    double p_total = 0, q_total = 0, q_proposed = 0;
    for (float p : target) {
        if (!std::isfinite(p) || p < 0)
            throw std::invalid_argument("invalid target probability");
        p_total += p;
    }
    for (uint32_t i = 0; i < candidate_count; ++i) {
        if (candidate_ids[i] < 0 || size_t(candidate_ids[i]) >= target.size() ||
            !std::isfinite(proposal[i]) || proposal[i] < 0)
            throw std::invalid_argument("invalid sparse proposal probability or token");
        for (uint32_t j = 0; j < i; ++j)
            if (candidate_ids[j] == candidate_ids[i])
                throw std::invalid_argument("duplicate sparse proposal token");
        q_total += proposal[i];
        if (candidate_ids[i] == proposed_token) q_proposed = proposal[i];
    }
    if (!(p_total > 0) || !(q_total > 0) || !(q_proposed > 0))
        throw std::invalid_argument("empty distribution or impossible proposed token");
    // Normalize in double precision, matching categorical sampling of weights
    // even when a float softmax sums a few ULPs away from one.
    SparseRejectionStep result;
    result.accept_probability = std::min(1.0,
        (target[proposed_token] / p_total) / (q_proposed / q_total));
    result.residual.reserve(target.size());
    for (float p : target) result.residual.push_back(p / p_total);
    for (uint32_t i = 0; i < candidate_count; ++i) {
        double &p = result.residual[candidate_ids[i]];
        p = std::max(0.0, p - proposal[i] / q_total);
    }
    return result;
}

} // namespace qw3::detail
