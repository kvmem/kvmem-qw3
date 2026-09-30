#include "continuous_prefill_policy.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

static void check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    using qw3::detail::ContinuousPrefillPolicy;
    // An indefinitely ready decoder must not starve any of three prefilling
    // slots, including gaps left by cancelled requests. Conversely each chunk
    // must give the ready decoder another turn before the next prefill chunk.
    for (uint32_t quantum : {1u, 4u, 8u}) {
        ContinuousPrefillPolicy policy(quantum);
        std::array<unsigned, 5> serviced{};
        unsigned verifies = 0, prefills = 0, gap = 0;
        bool previous_prefill = false;
        for (unsigned i = 0; i < 300; ++i) {
            if (policy.run_prefill(true, true)) {
                check(!previous_prefill, "prefill starved ready decode");
                ++serviced.at(policy.select_prefill({0, 2, 4}));
                ++prefills; gap = 0; previous_prefill = true;
            } else {
                policy.verified(); ++verifies; ++gap;
                check(gap <= quantum, "decode starved prefill");
                previous_prefill = false;
            }
        }
        check(verifies && prefills && serviced[0] && serviced[2] && serviced[4],
              "runnable request made no progress");
        check(serviced[0] - serviced[4] <= 1, "prefill slots were not served fairly");
    }
    ContinuousPrefillPolicy policy(4);
    check(policy.run_prefill(true, false), "cold prefill waited for absent decode");
    check(policy.select_prefill({1, 3}) == 1, "first pending slot was skipped");
    // Cancel slot 3 before its turn, then replenish a different slot.
    check(policy.select_prefill({0, 1}) == 0, "cancelled slot retained ownership");
    check(policy.select_prefill({1, 4}) == 1, "replenishment reordered older pending slot");
    check(policy.select_prefill({4}) == 4, "replacement slot was starved");
    check(!policy.run_prefill(false, true) && !policy.run_prefill(false, false),
          "scheduler selected absent prefill");
    bool threw = false;
    try { policy.select_prefill({}); } catch (const std::logic_error &) { threw = true; }
    check(threw, "empty prefill selection was accepted");
    // Equal-sized cold prompts should produce the first ready request after
    // one prompt's work, not after almost all of every prompt. Ready decoding
    // continues between chunks; slot reuse must not reset request age.
    ContinuousPrefillPolicy oldest(4);
    std::array<unsigned, 3> remaining{3, 3, 3};
    std::array<uint64_t, 3> age{12, 10, 11};
    unsigned total_chunks = 0;
    while (remaining[1]) {
        std::vector<std::pair<size_t, uint64_t>> slots;
        for (size_t slot = 0; slot < remaining.size(); ++slot)
            if (remaining[slot]) slots.emplace_back(slot, age[slot]);
        --remaining[oldest.select_oldest_prefill(slots)];
        ++total_chunks;
    }
    check(total_chunks == 3 && remaining[0] == 3 && remaining[2] == 3,
          "cold prompts delayed first ready request");
    // The freed slot is reused immediately by a new request. The older slot
    // must win, even when the replacement appears first in the input vector.
    check(oldest.select_oldest_prefill({{1, 13}, {0, 12}, {2, 11}}) == 2,
          "replacement request overtook old prefill");
    check(!oldest.run_prefill(true, true), "oldest prefill starved ready decode");
    for (unsigned i = 0; i < 4; ++i) oldest.verified();
    check(oldest.run_prefill(true, true), "decode starved oldest prefill");
    check(oldest.select_oldest_prefill({{1, 13}, {0, 12}}) == 0,
          "cancelled oldest prompt retained ownership");
    threw = false;
    try { oldest.select_oldest_prefill({}); } catch (const std::logic_error &) { threw = true; }
    check(threw, "empty oldest prefill selection was accepted");
    std::cout << "continuous_prefill_policy fairness/cancellation/replenishment/age status=pass\n";
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
