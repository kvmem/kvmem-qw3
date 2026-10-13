#pragma once
#include <cstdint>

namespace qw3::flashinfer_adapter {
// Only sampled rows are materialized. No second Q*K or KV-cache traversal.
// Row slots are popcount(row_mask below row), each [heads, stride] floats.
struct AttentionCapture {
    float *logits = nullptr;
    uint64_t row_mask = 0;
    uint32_t stride = 0;
    uint32_t heads = 0;
    uint32_t row_offset = 0;
    uint32_t first_row = 0; // row_mask is relative to this row in the original call
    uint32_t row_step = 1; // physical row = first_row + mask bit * row_step
};
}
