#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qw3::detail {

// Exact bytes of a request's allocated pages, including partial-page tails.
// Logical directories and RoPE coordinates stay with the paused executor.
// No backing-tier selection/re-RoPE is needed when these pages are restored.
struct KvMemIdlePages {
    struct Tensor {
        uint64_t page_bytes = 0;
        std::vector<uint8_t> bytes;
    };
    std::vector<int32_t> pages;
    std::vector<Tensor> tensors;

    static uint64_t payload_bytes(const std::vector<uint64_t> &page_bytes,
                                  size_t page_count) {
        uint64_t total = 0;
        for (uint64_t bytes : page_bytes) {
            if (!bytes || page_count > (std::numeric_limits<uint64_t>::max() - total) / bytes)
                throw std::overflow_error("invalid idle KV snapshot size");
            total += bytes * page_count;
        }
        if (total > std::numeric_limits<size_t>::max())
            throw std::overflow_error("idle KV snapshot exceeds host address space");
        return total;
    }

    uint64_t payload_bytes() const {
        uint64_t total = 0;
        for (const auto &tensor : tensors) total += tensor.bytes.size();
        return total;
    }

    template<class Read>
    static KvMemIdlePages capture(std::vector<int32_t> ids,
                                 const std::vector<uint64_t> &page_bytes,
                                 uint32_t arena_pages, Read read) {
        payload_bytes(page_bytes, ids.size());
        std::unordered_set<int32_t> unique;
        for (int32_t page : ids)
            if (page < 0 || uint32_t(page) >= arena_pages || !unique.insert(page).second)
                throw std::invalid_argument("invalid idle KV physical page directory");
        KvMemIdlePages out;
        out.pages = std::move(ids);
        out.tensors.reserve(page_bytes.size());
        for (size_t t = 0; t < page_bytes.size(); ++t) {
            Tensor image;
            image.page_bytes = page_bytes[t];
            image.bytes.resize(image.page_bytes * out.pages.size());
            for (size_t p = 0; p < out.pages.size(); ++p)
                read(t, uint64_t(out.pages[p]) * image.page_bytes,
                     image.bytes.data() + p * image.page_bytes, image.page_bytes);
            out.tensors.push_back(std::move(image));
        }
        return out;
    }

    template<class Write>
    void restore(const std::vector<int32_t> &destination,
                 uint32_t arena_pages, Write write) const {
        if (destination.size() != pages.size())
            throw std::invalid_argument("idle KV restore page count mismatch");
        std::unordered_set<int32_t> unique;
        for (int32_t page : destination)
            if (page < 0 || uint32_t(page) >= arena_pages || !unique.insert(page).second)
                throw std::invalid_argument("invalid idle KV restore directory");
        for (size_t t = 0; t < tensors.size(); ++t) {
            const auto &image = tensors[t];
            for (size_t p = 0; p < destination.size(); ++p)
                write(t, uint64_t(destination[p]) * image.page_bytes,
                      image.bytes.data() + p * image.page_bytes, image.page_bytes);
        }
    }

    // Negative holes are retained. Every nonnegative entry must name a page
    // that this executor owned at suspension; borrowed/pinned pages are refused.
    std::vector<int32_t> remap(const std::vector<int32_t> &directory,
                               const std::vector<int32_t> &destination) const {
        if (destination.size() != pages.size())
            throw std::invalid_argument("idle KV remap page count mismatch");
        std::unordered_map<int32_t, int32_t> map;
        for (size_t p = 0; p < pages.size(); ++p) map.emplace(pages[p], destination[p]);
        auto result = directory;
        for (auto &page : result) if (page >= 0) {
            const auto it = map.find(page);
            if (it == map.end()) throw std::logic_error("idle KV directory contains an unowned page");
            page = it->second;
        }
        return result;
    }
};

} // namespace qw3::detail
