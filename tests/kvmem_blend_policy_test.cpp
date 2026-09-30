#include "kvmem_blend_policy.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>

int main() {
    using namespace qw3::detail;
    const auto require = [](bool ok) { if (!ok) std::abort(); };
    const std::vector<KvMemBlendSpan> spans{{1000, 32, 32, 1}, {2000, 64, 32, 9},
                                         {3000, 96, 32, 9}, {4000, 128, 32, 2}};
    require(kvmem_blend_plan(spans, 160, 0, 160).empty());
    require(kvmem_blend_plan(spans, 160, .1, 160).empty());
    const auto top = kvmem_blend_plan(spans, 160, 1, 64);
    require(top.size() == 2 && top[0].history == 2000 && top[1].history == 3000);
    std::vector<KvMemBlendSpan> window32k;
    for (uint32_t i = 0; i < 1024; ++i)
        window32k.push_back({65536 + i * 32, i * 32, 32, double(i)});
    const auto ten_percent = kvmem_blend_plan(window32k, 32768, .10, UINT32_MAX);
    require(ten_percent.size() == 102 && ten_percent.front().window == 922 * 32 &&
            ten_percent.back().window == 1023 * 32);
    auto invalid = spans;
    invalid.push_back({0, UINT32_MAX, 32, 100});
    invalid.push_back({0, 150, 32, 100});
    invalid.push_back({0, 0, 0, 100});
    invalid.push_back({0, 0, 32, std::numeric_limits<double>::quiet_NaN()});
    const auto bounded = kvmem_blend_plan(invalid, 160, .4, UINT32_MAX);
    require(bounded.size() == 2 && bounded[0].history == 2000 && bounded[1].history == 3000);
    for (double ratio : {-1., 1.01, std::numeric_limits<double>::quiet_NaN()}) {
        bool caught = false;
        try { kvmem_blend_plan(spans, 160, ratio, 160); }
        catch (const std::invalid_argument &) { caught = true; }
        require(caught);
    }
    std::cout << "blend policy: budget, causal order, deterministic ties, invalid spans passed\n";
}
