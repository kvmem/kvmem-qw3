#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qw3::detail {
// Bound the number of ready decode rounds between independent prefill chunks.
// This is a cooperative policy; it does not preempt an executing GPU operation.
class ContinuousPrefillPolicy {
public:
    explicit ContinuousPrefillPolicy(uint32_t decode_rounds)
        : budget_(std::max(1u, decode_rounds)) {}

    bool run_prefill(bool has_prefill, bool has_verify) const {
        return has_prefill && (!has_verify || rounds_ >= budget_);
    }
    void verified() { if (rounds_ < budget_) ++rounds_; }

    // Slot IDs are stable across suspension and may be reused after teardown.
    size_t select_prefill(const std::vector<size_t> &slots) {
        if (slots.empty()) throw std::logic_error("no pending prefill slot");
        auto next = std::find_if(slots.begin(), slots.end(),
                                 [&](size_t slot) { return slot >= cursor_; });
        const size_t row = next == slots.end() ? slots.front() : *next;
        cursor_ = row + 1;
        rounds_ = 0;
        return row;
    }

    // Finish the oldest pending prompt before advancing another prompt, while
    // run_prefill() still interleaves every chunk with ready decode. Request
    // IDs, rather than recycled slot numbers, prevent new arrivals from
    // overtaking older prompts. Age ordering resumes after cancellation too.
    size_t select_oldest_prefill(const std::vector<std::pair<size_t, uint64_t>> &slots) {
        if (slots.empty()) throw std::logic_error("no pending prefill slot");
        const auto next = std::min_element(slots.begin(), slots.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
        cursor_ = next->first + 1;
        rounds_ = 0;
        return next->first;
    }
private:
    uint32_t budget_, rounds_ = 0;
    size_t cursor_ = 0;
};
} // namespace qw3::detail
