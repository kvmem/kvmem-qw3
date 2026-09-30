#pragma once

#include "qwen_executor.hpp"
#include <algorithm>
#include <climits>
#include <unordered_map>

namespace qw3 {
// Per-request lease over the serving arena. Resident pages are reserved at
// admission so another request cannot consume the working-set guarantee.
// Private-query pages are borrowed only while a reversible probe is active.
// The parent outlives this pool; the executor drains GPU work before teardown.
class SharedKvPagePool final : public KvPhysicalPageAllocator {
public:
    SharedKvPagePool(KvPhysicalPageAllocator &parent, uint32_t resident_pages,
                     uint32_t reserve_pages, const char *label)
        : parent_(parent), resident_(resident_pages), reserve_(reserve_pages),
          label_(label ? label : "unknown") {
        if (!resident_ || uint64_t(resident_) + reserve_ > INT32_MAX)
            throw std::invalid_argument("invalid shared KVMem page quota");
        pages_.reserve(resident_);
        free_.reserve(resident_);
        private_.reserve(reserve_);
        in_use_.assign(resident_, false);
        slots_.reserve(resident_);
        try {
            for (uint32_t i = 0; i < resident_; ++i) {
                const int32_t page = parent_.allocate_physical_page();
                pages_.push_back(page);
                if (page < 0 || uint32_t(page) >= physical_page_capacity() ||
                    !slots_.emplace(page, i).second)
                    throw std::runtime_error("invalid shared KVMem physical page");
            }
            for (uint32_t i = 0; i < resident_; ++i) free_.push_back(resident_ - 1 - i);
        } catch (...) {
            parent_.release_physical_pages(pages_);
            throw;
        }
    }
    ~SharedKvPagePool() override {
        // Pinned prefix pages remain owned by the parent's cache until eviction.
        parent_.release_physical_pages(private_);
        parent_.release_physical_pages(pages_);
    }
    SharedKvPagePool(const SharedKvPagePool &) = delete;
    SharedKvPagePool &operator=(const SharedKvPagePool &) = delete;

    int32_t allocate_physical_page() override {
        if (!free_.empty()) {
            const auto slot = free_.back(); free_.pop_back();
            in_use_[slot] = true;
            return pages_[slot];
        }
        if (transient_ && private_.size() < reserve_) {
            const auto page = parent_.allocate_physical_page();
            private_.push_back(page); // Capacity reserved in the constructor.
            return page;
        }
        throw std::runtime_error("shared KVMem request page quota exhausted: pool=" + label_ +
            " resident=" + std::to_string(resident_) + " reserve=" + std::to_string(reserve_));
    }
    void release_physical_pages(const std::vector<int32_t> &pages) override {
        for (int32_t page : pages) {
            const auto it = slots_.find(page);
            if (it != slots_.end()) {
                const auto slot = it->second;
                if (in_use_[slot]) { in_use_[slot] = false; free_.push_back(slot); }
            } else {
                const auto found = std::find(private_.begin(), private_.end(), page);
                if (found != private_.end()) {
                    parent_.release_physical_pages({page});
                    private_.erase(found);
                }
            }
        }
    }
    uint32_t free_pages() const override {
        return uint32_t(free_.size()) + (transient_
            ? std::min<uint32_t>(reserve_ - private_.size(), parent_.free_pages()) : 0);
    }
    uint32_t used_pages() const override {
        return resident_ - uint32_t(free_.size()) + uint32_t(private_.size());
    }
    uint32_t total_pages() const override { return resident_ + reserve_; }
    uint32_t physical_page_capacity() const override { return parent_.physical_page_capacity(); }
    void set_transient_allocation(bool enabled) override { transient_ = enabled; }
private:
    KvPhysicalPageAllocator &parent_;
    uint32_t resident_, reserve_;
    std::string label_;
    std::vector<int32_t> pages_, private_;
    std::vector<uint32_t> free_;
    std::vector<bool> in_use_;
    std::unordered_map<int32_t, uint32_t> slots_;
    bool transient_ = false;
};
} // namespace qw3
