#pragma once
#include "qwen_executor.hpp"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace qw3 {
// The physical allocation includes a bounded private-query scratch reserve.
// free_pages()/allocate_physical_page() hide it from ordinary inference so
// adding scratch cannot move the same exhaustion bug to a later boundary.
class LocalKvPagePool final : public KvPhysicalPageAllocator {
public:
    LocalKvPagePool(uint32_t total_pages, uint32_t page_size,
                    const char *label, uint32_t reserve_pages = 0)
        : total_pages_(checked_total(total_pages, reserve_pages)),
          reserve_pages_(reserve_pages),
          page_size_(std::max<uint32_t>(1, page_size)),
          label_(label ? label : "unknown"),
          in_free_(total_pages_, true) {
        free_pages_.reserve(total_pages_);
        for (uint32_t i = 0; i < total_pages_; ++i) {
            free_pages_.push_back(static_cast<int32_t>(total_pages_ - 1U - i));
        }
    }

    int32_t allocate_physical_page() override {
        if (free_pages() == 0) {
            throw std::runtime_error(
                "local KVMem GPU page pool exhausted: pool=" + label_ +
                " free=0 total=" +
                std::to_string(total_pages_) +
                " page_size=" + std::to_string(page_size_) +
                " reserve=" + std::to_string(reserve_pages_) +
                " transient=" + std::to_string(transient_allocation_));
        }
        const int32_t page = free_pages_.back();
        free_pages_.pop_back();
        if (page >= 0 && static_cast<uint32_t>(page) < in_free_.size()) {
            in_free_[page] = false;
        }
        ++used_pages_;
        return page;
    }

    void release_physical_pages(const std::vector<int32_t> &pages) override {
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            if (in_free_[page]) continue;
            in_free_[page] = true;
            free_pages_.push_back(page);
            if (used_pages_ > 0) --used_pages_;
        }
    }

    uint32_t free_pages() const override {
        const uint32_t free = static_cast<uint32_t>(free_pages_.size());
        return transient_allocation_ ? free :
            (free > reserve_pages_ ? free - reserve_pages_ : 0);
    }

    uint32_t used_pages() const override { return used_pages_; }
    uint32_t total_pages() const override { return total_pages_; }

    void set_transient_allocation(bool enabled) override {
        transient_allocation_ = enabled;
    }

private:
    static uint32_t checked_total(uint32_t ordinary, uint32_t reserve) {
        const uint64_t total = uint64_t(std::max<uint32_t>(1, ordinary)) + reserve;
        if (total > INT32_MAX) throw std::invalid_argument("KVMem page pool too large");
        return static_cast<uint32_t>(total);
    }
    uint32_t total_pages_ = 0;
    uint32_t reserve_pages_ = 0;
    bool transient_allocation_ = false;
    uint32_t page_size_ = 0;
    std::string label_;
    std::vector<int32_t> free_pages_;
    std::vector<bool> in_free_;
    uint32_t used_pages_ = 0;
};

} // namespace qw3
