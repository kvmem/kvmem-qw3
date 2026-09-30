#include "dflash_deferred_refresh.hpp"
#include "speculative_sampling.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace qw3::detail;

void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void boundary(std::string_view prefix, const std::vector<std::string> &pieces,
              uint32_t expected, bool closes) {
    const auto result = dflash_refresh_boundary(prefix, pieces.size(),
        [&](uint32_t index) { return pieces[index]; });
    require(result.candidates == expected && result.includes_boundary == closes,
            "lost the first safe boundary or discarded its proposal");
}

void closure_distribution(const std::vector<float> &p, const std::vector<float> &q) {
    const std::array<int32_t, 2> ids{0, 1};  // ordinary content, closing newline
    std::array<double, 2> emitted{};
    for (size_t proposed = 0; proposed < ids.size(); ++proposed) {
        const auto cut = dflash_refresh_boundary("unfinished", 2, [&](uint32_t index) {
            return index == 0 && proposed == 1 ? "\n" : " more";
        });
        require(cut.candidates >= 1, "closing proposal bypassed verification");
        require(cut.includes_boundary == (proposed == 1), "unexpected boundary decision");
        const auto step = sparse_rejection_step(p, ids.data(), q.data(), 2, ids[proposed]);
        emitted[proposed] += q[proposed] * step.accept_probability;
        if (step.accept_probability == 1) continue;
        const double norm = std::accumulate(step.residual.begin(), step.residual.end(), 0.0);
        for (size_t token = 0; token < emitted.size(); ++token)
            emitted[token] += q[proposed] * (1 - step.accept_probability) * step.residual[token] / norm;
    }
    for (size_t token = 0; token < emitted.size(); ++token)
        require(std::abs(emitted[token] - p[token]) < 1e-12,
                "proposal-dependent closure changed the target marginal");
}

int main() try {
    for (uint32_t decoded = 0; decoded < 40; ++decoded) {
        const auto result = dflash_deferred_refresh_limit(7, decoded, 0, 13, 0, 2, 40, false, 8);
        if (decoded < 13) require(decoded + result.drafts + 1 <= 13, "skipped initial trigger");
        else if (decoded < 32) {
            require(result.needs_text_boundary_check, "pending boundary fell back indefinitely");
            require(decoded + result.drafts + 1 <= 32, "skipped emergency reserve boundary");
        } else require(result.drafts == 0, "speculated into emergency reserve");
    }
    const auto actual = dflash_deferred_refresh_limit(7, 2534, 0, 2534, 0, 2, 6630, false);
    require(actual.drafts == 7 && actual.needs_text_boundary_check,
            "recorded DeepSWE open tool body still forces serial execution");
    require(dflash_deferred_refresh_limit(7, 2534, 0, 2534, 0, 2, 6630, true).drafts == 0,
            "safe boundary was postponed");
    require(dflash_deferred_refresh_limit(7, 110, 100, 13, 1, 2, 40, false, 8).drafts == 2,
            "epoch offset changed initial trigger");
    require(dflash_deferred_refresh_limit(7, UINT32_MAX, 0, 13, 2, 2, 40, false).drafts == 7,
            "completed policy continued throttling");
    require(dflash_deferred_refresh_limit(7, 13, 0, 13, 0, 2, 0, false).drafts == 0,
            "unknown reserve permitted deferred speculation");
    boundary("<tool_call><function=bash>code", {"\n", "</function>", "</tool", "_call>", "after"}, 4, true);
    // Preserve the current serial predicate: a bare closing fence still has
    // a partial-marker suffix, so its following newline is the safe boundary.
    boundary("```cpp\nint x", {" = 0;", "\n", "``", "`", "\n"}, 5, true);
    boundary("unfinished", {" text", ".", "after"}, 2, true);
    boundary("<tool_call>code", {"\n", ".", "```"}, 3, false);
    closure_distribution({.75f, .25f}, {.25f, .75f});
    closure_distribution({.75f, .25f}, {.875f, .125f});
    closure_distribution({0.f, 1.f}, {.5f, .5f});
    // A negative control: skipping verification when q proposes the boundary
    // produces 0.1875 rather than the target's 0.25 closure probability.
    require(std::abs(.75 * .25 - .25) > .05, "negative control did not expose biased closure");
    std::cout << "Deferred refresh policy passed: trigger/reserve limits, split tool/fence boundaries, 3 exact closure marginals\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
