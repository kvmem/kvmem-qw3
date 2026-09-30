#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace qw3::detail {
// Fixed owner quotas include idle retained sessions and the setup executor.
// Controller admission limits live + idle request owners to max_active.
struct BoundedContinuousResources {
    uint64_t cpu_bytes_per_owner, nvme_bytes_per_owner;
    uint32_t pages_per_request, arena_pages;

    static BoundedContinuousResources plan(uint32_t max_active, uint32_t block_tokens,
            uint32_t page_tokens, uint32_t k, uint32_t prefill, uint32_t g,
            uint64_t private_tokens, uint64_t cpu_bytes, uint64_t nvme_bytes) {
        if (!max_active || !block_tokens || !page_tokens || block_tokens % page_tokens ||
            !k || !g || prefill < k || private_tokens > UINT32_MAX)
            throw std::invalid_argument("invalid bounded continuous resource plan");
        const uint64_t resident_blocks = (uint64_t(prefill) + block_tokens - 1) / block_tokens +
            (uint64_t(g) + block_tokens - 1) / block_tokens;
        const uint64_t pages = resident_blocks * (block_tokens / page_tokens) +
            (private_tokens + page_tokens - 1) / page_tokens;
        if (pages > INT32_MAX || pages * max_active > INT32_MAX)
            throw std::invalid_argument("bounded continuous arena exceeds signed page-index capacity");
        const uint64_t owners = uint64_t(max_active) + 1;
        if ((cpu_bytes && cpu_bytes < owners) || (nvme_bytes && nvme_bytes < owners))
            throw std::invalid_argument("bounded continuous tier budget is too small for all owners");
        return {cpu_bytes / owners, nvme_bytes / owners, uint32_t(pages), uint32_t(pages * max_active)};
    }
};
} // namespace qw3::detail
