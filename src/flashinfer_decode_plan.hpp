#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qw3::detail {

// Shape-only metadata for causal rows sharing a device page table. Physical
// page IDs are intentionally absent: callers must gather them on every call,
// even if the page-table pointer and logical sequence lengths did not change.
struct FlashinferDecodePlan {
    std::vector<int32_t> metadata, output_indptr;
    uint64_t page_indptr_offset = 0, last_page_len_offset = 0;
    uint64_t request_indices_offset = 0, kv_tile_indices_offset = 0;
    uint64_t output_indptr_offset = 0, chunk_size_offset = 0;
    uint32_t total_tiles = 0;
    bool valid = false;
    std::array<uint32_t, 5> key{};

    uint64_t storage_elements() const { return page_indptr_offset + metadata.size(); }

    bool update(uint32_t base, uint32_t batch, uint32_t page_size,
                uint32_t chunk_size, bool partition) {
        const std::array<uint32_t, 5> next{base, batch, page_size, chunk_size, uint32_t(partition)};
        if (valid && key == next) return false;
        valid = false;
        constexpr uint64_t limit = std::numeric_limits<int32_t>::max();
        if (!batch || !page_size || !chunk_size || uint64_t(base) + batch > limit)
            throw std::invalid_argument("invalid FlashInfer decode plan dimensions");
        const uint64_t max_pages = (uint64_t(base) + batch + page_size - 1) / page_size;
        const uint64_t max_tiles = partition
            ? (uint64_t(base) + batch + chunk_size - 1) / chunk_size : 1;
        if (max_pages * batch > limit || max_tiles * batch > limit)
            throw std::overflow_error("FlashInfer decode plan exceeds int32 indexing");

        std::vector<int32_t> page_indptr(batch + 1), last_page_len(batch);
        std::vector<int32_t> requests, tiles;
        output_indptr.assign(batch + 1, 0);
        uint32_t pages_total = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t length = base + b + 1;
            const uint32_t pages = (uint64_t(length) + page_size - 1) / page_size;
            page_indptr[b] = pages_total;
            pages_total += pages;
            last_page_len[b] = length - (pages - 1) * page_size;
            output_indptr[b] = requests.size();
            const uint32_t chunks = partition
                ? (uint64_t(length) + chunk_size - 1) / chunk_size : 1;
            for (uint32_t t = 0; t < chunks; ++t) {
                requests.push_back(b);
                tiles.push_back(t);
            }
        }
        page_indptr[batch] = pages_total;
        output_indptr[batch] = requests.size();
        total_tiles = requests.size();
        metadata.clear();
        // Leave device-only space for expanded physical page IDs; uploading
        // zeros here is redundant because the gather overwrites every entry.
        page_indptr_offset = batch > 1 ? pages_total : 0;
        auto append = [&](const std::vector<int32_t> &values) {
            const uint64_t offset = page_indptr_offset + metadata.size();
            metadata.insert(metadata.end(), values.begin(), values.end());
            return offset;
        };
        append(page_indptr);
        last_page_len_offset = append(last_page_len);
        request_indices_offset = append(requests);
        kv_tile_indices_offset = append(tiles);
        output_indptr_offset = append(output_indptr);
        chunk_size_offset = page_indptr_offset + metadata.size();
        if (chunk_size > limit) throw std::overflow_error("FlashInfer chunk size exceeds int32");
        metadata.push_back(chunk_size);
        key = next;
        valid = true;
        return true;
    }
};

}  // namespace qw3::detail
