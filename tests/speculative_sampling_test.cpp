#include "speculative_sampling.hpp"

#include <array>
#include <iostream>
#include <numeric>
#include <random>

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void analytic(const std::vector<float> &p, const std::vector<int32_t> &ids,
              const std::vector<float> &q) {
    const double ps = std::accumulate(p.begin(), p.end(), 0.0);
    const double qs = std::accumulate(q.begin(), q.end(), 0.0);
    std::vector<double> actual(p.size(), 0);
    for (size_t d = 0; d < ids.size(); ++d) {
        if (!q[d]) continue;
        const auto step = qw3::detail::sparse_rejection_step(p, ids.data(), q.data(), ids.size(), ids[d]);
        const double mass = q[d] / qs;
        actual[ids[d]] += mass * step.accept_probability;
        const double residual_total = std::accumulate(step.residual.begin(), step.residual.end(), 0.0);
        if (step.accept_probability == 1) continue;
        require(residual_total > 0, "rejection without a residual");
        for (size_t v = 0; v < p.size(); ++v)
            actual[v] += mass * (1 - step.accept_probability) * step.residual[v] / residual_total;
        require(step.residual_token(0) >= 0 && step.residual_token(std::nextafter(1.0, 0.0)) >= 0,
                "residual CDF boundary lost a token");
    }
    for (size_t v = 0; v < p.size(); ++v)
        require(std::abs(actual[v] - p[v] / ps) < 1e-12, "sparse rejection changes the target distribution");
}
} // namespace

int main() try {
    analytic({.1f, .2f, .3f, .4f}, {2, 0}, {.7f, .3f});
    analytic({0, 0, .5f, .5f}, {0, 1}, {.2f, .8f});
    analytic({.1f, .2f, .3f, .4f}, {0}, {1});
    analytic({0, 1, 0, 0}, {0, 1, 2}, {.6f, .2f, .2f});
    analytic({.1f, .2f, .3f, .4f}, {0, 1, 2, 3}, {.1f, .2f, .3f, .4f});
    analytic({1e-30f, .1f, .9f}, {0, 1}, {1e-20f, 1});

    // Two-token joint distribution with predecessor-dependent proposals. At a
    // first-position rejection the old second proposal is invalid and must be
    // discarded. Check the joint law, not only marginal token histograms.
    std::mt19937_64 rng(8942);
    std::uniform_real_distribution<double> unit(0, 1);
    const std::vector<float> p0{.2f, .3f, .5f}, q0{.8f, .2f};
    const std::array<std::vector<float>, 3> p1{{{.1f, .2f, .7f}, {.6f, .1f, .3f}, {.2f, .7f, .1f}}};
    const std::array<std::vector<float>, 3> q1{{{.4f, .6f}, {.9f, .1f}, {.3f, .7f}}};
    const std::array<int32_t, 2> ids{0, 2};
    const auto draw = [&](const std::vector<float> &p) {
        std::discrete_distribution<int> d(p.begin(), p.end()); return d(rng);
    };
    std::array<uint32_t, 9> counts{};
    constexpr uint32_t trials = 400000;
    for (uint32_t n = 0; n < trials; ++n) {
        const int d0 = ids[draw(q0)], d1 = ids[draw(q1[d0])];
        const auto s0 = qw3::detail::sparse_rejection_step(p0, ids.data(), q0.data(), 2, d0);
        int a, b;
        if (!s0.accepts(unit(rng))) {
            a = s0.residual_token(unit(rng));
            b = draw(p1[a]);
        } else {
            a = d0;
            const auto s1 = qw3::detail::sparse_rejection_step(p1[a], ids.data(), q1[a].data(), 2, d1);
            b = s1.accepts(unit(rng)) ? d1 : s1.residual_token(unit(rng));
        }
        ++counts[a * 3 + b];
    }
    double max_z = 0;
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
        const double probability = double(p0[a]) * p1[a][b];
        const double z = std::abs(counts[a * 3 + b] - trials * probability) /
                         std::sqrt(trials * probability * (1 - probability));
        max_z = std::max(max_z, z);
        require(z < 6, "conditional rejection joint distribution is biased");
    }
    const auto invalid = [&](const std::vector<int32_t> &ids, const std::vector<float> &q, int32_t token) {
        bool rejected = false;
        try { qw3::detail::sparse_rejection_step(p0, ids.data(), q.data(), ids.size(), token); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "invalid sparse proposal accepted");
    };
    invalid({0, 0}, {.5f, .5f}, 0);
    invalid({0, 3}, {.5f, .5f}, 0);
    invalid({0, 2}, {0, 1}, 0);
    invalid({0, 2}, {NAN, 1}, 2);
    std::cout << "sparse_rejection analytic_cases=6 joint_trials=" << trials << " max_z=" << max_z << " passed\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
}
