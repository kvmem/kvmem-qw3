#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

inline bool batch_needs_logits(float temperature, float presence_penalty,
                               float repetition_penalty) {
    return temperature > 0.0f || presence_penalty != 0.0f ||
           (repetition_penalty > 0.0f && repetition_penalty != 1.0f);
}

// A copied LM-head matrix owns its rows until every request has consumed them.
// Never fall back to an executor's old logits when a batch row is missing.
inline std::vector<float> batch_logits_row(const std::vector<float> &matrix,
                                         uint32_t vocab, uint64_t row) {
    if (!vocab || matrix.size() % vocab || row >= matrix.size() / vocab)
        throw std::runtime_error("continuous batch logits row out of range");
    const auto begin = matrix.begin() + static_cast<std::ptrdiff_t>(row * vocab);
    return {begin, begin + vocab};
}

struct DecodePageSpan {
    uint32_t position = 0;
    uint32_t seq_len = 0;
    uint32_t pages = 0;
    uint32_t last_page_len = 0;
    const int32_t *page_indices = nullptr;
};

// Select the attention coordinate and its matching page table together.
// Historical positions are allowed to exceed u32 when a compact window is used.
inline DecodePageSpan decode_page_span(
        uint64_t historical_position, uint32_t page_size,
        const int32_t *history_pages, uint32_t history_page_count,
        bool window_active, uint32_t window_position,
        const int32_t *window_pages, uint32_t window_page_count) {
    const uint64_t pos = window_active ? window_position : historical_position;
    const auto *pages = window_active ? window_pages : history_pages;
    const uint32_t count = window_active ? window_page_count : history_page_count;
    if (!page_size || !pages || pos >= uint64_t(INT32_MAX))
        throw std::runtime_error("continuous decode attention position invalid");
    const uint32_t seq_len = static_cast<uint32_t>(pos + 1);
    const uint32_t needed = static_cast<uint32_t>((pos + page_size) / page_size);
    if (needed > count)
        throw std::runtime_error("continuous decode page table is incomplete");
    for (uint32_t i = 0; i < needed; ++i)
        if (pages[i] < 0)
            throw std::runtime_error("continuous decode page is not resident");
    return {static_cast<uint32_t>(pos), seq_len, needed,
            seq_len % page_size ? seq_len % page_size : page_size, pages};
}

} // namespace qw3::detail
