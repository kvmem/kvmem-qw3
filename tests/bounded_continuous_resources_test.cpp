#include "bounded_continuous_resources.hpp"
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main() try {
    using qw3::detail::BoundedContinuousResources;
    for (uint32_t slots : {1u, 2u, 4u}) {
        const auto p = BoundedContinuousResources::plan(slots, 32, 16, 131072, 131072,
            65536, 8192, 64ull << 30, 100ull << 30);
        check(p.pages_per_request == 12800, "K/G/private reservation changed");
        check(p.arena_pages == slots * 12800, "GPU arena under-reserves concurrent working sets");
        check(p.cpu_bytes_per_owner * (slots + 1) <= (64ull << 30), "CPU owners exceed process tier budget");
        check(p.nvme_bytes_per_owner * (slots + 1) <= (100ull << 30), "NVMe owners exceed process tier budget");
    }
    const auto rounded = BoundedContinuousResources::plan(2, 32, 16, 32, 64, 33, 17, 12, 0);
    check(rounded.pages_per_request == 10 && rounded.arena_pages == 20,
          "partial generation/private blocks were not reserved");
    const auto histories = BoundedContinuousResources::plan(2, 32, 16, 131072, 131072,
        65536, 8192, 90ull << 30, 180ull << 30, 8);
    check(histories.arena_pages == 25600 && histories.pages_per_request == 12800,
          "idle histories must not reserve additional GPU slots");
    check(histories.cpu_bytes_per_owner == (10ull << 30) &&
          histories.nvme_bytes_per_owner == (20ull << 30),
          "retained histories exceeded total tier quotas");
    bool too_few_histories = false;
    try { (void)BoundedContinuousResources::plan(4, 32, 16, 32, 32, 32, 0, 0, 0, 2); }
    catch (const std::invalid_argument &) { too_few_histories = true; }
    check(too_few_histories, "history capacity below active slots was accepted");
    for (unsigned invalid = 0; invalid < 5; ++invalid) {
        bool threw = false;
        try {
            (void)BoundedContinuousResources::plan(invalid == 0 ? 0 : invalid == 1 ? UINT32_MAX : 4,
                32, invalid == 2 ? 3 : 16, 131072, 131072, 65536,
                invalid == 3 ? uint64_t(UINT32_MAX) + 1 : 8192,
                invalid == 4 ? 4 : (64ull << 30), 0);
        } catch (const std::invalid_argument &) { threw = true; }
        check(threw, "invalid resource reservation accepted");
    }
    std::cout << "bounded continuous resource bounds status=pass\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
