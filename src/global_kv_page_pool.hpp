#pragma once

#include "qwen_executor.hpp"
#include <functional>
#include <mutex>
#include <stdexcept>

namespace qw3 {

class GlobalKvPagePool final : public KvPhysicalPageAllocator {
public:
    GlobalKvPagePool(uint32_t total_pages, uint32_t page_size)
        : total_pages_(total_pages), page_size_(page_size) {
        free_pages_.reserve(total_pages_);
        for (uint32_t i = 0; i < total_pages_; ++i) {
            free_pages_.push_back(static_cast<int32_t>(total_pages_ - 1U - i));
        }
        page_pin_refcount_.assign(total_pages_, 0);
        allocated_.assign(total_pages_, false);
    }

    // Installed by the backend's prefix cache. On free-stack exhaustion the
    // pool invokes this to reclaim pinned (refcount==0 entry) cache pages back
    // onto the free stack. Must return the number of entries evicted (0 = nothing
    // evictable). The callback runs WITHOUT the pool mutex held (it itself
    // calls unpin_pages/release_physical_pages, which take the lock).
    void set_evict_callback(std::function<uint32_t()> cb) {
        evict_cb_ = std::move(cb);
    }

    int32_t allocate_physical_page() override {
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!free_pages_.empty()) {
                    const int32_t page = free_pages_.back();
                    free_pages_.pop_back();
                    allocated_[page] = true;
                    ++used_pages_;
                    return page;
                }
            }
            // Free stack empty: ask the prefix cache to evict the LRU entry
            // whose refcount is 0, then retry. A single eviction may reclaim
            // ZERO physical pages when the victim shares all its pages with a
            // longer cached entry (extension/multi-turn). The evict callback
            // returns the number of ENTRIES it evicted (progress), not pages,
            // so we keep evicting until a page actually frees up or there is
            // nothing left to evict. Each iteration drops one entry's pin, so
            // the entry that uniquely owns the shared region eventually frees.
            if (!evict_cb_ || evict_cb_() == 0) break;
        }
        std::lock_guard<std::mutex> lk(mu_);
        throw std::runtime_error(
            "global KV page pool exhausted: free=0 total=" +
            std::to_string(total_pages_) +
            " page_size=" + std::to_string(page_size_));
    }

    void release_physical_pages(const std::vector<int32_t> &pages) override {
        if (pages.empty()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            // Defense in depth: a page still pinned by a prefix-cache entry
            // must never re-enter the free stack (the owning executor's
            // `owned` flag should already prevent this).
            if (page_pin_refcount_[page] > 0 || !allocated_[page]) continue;
            allocated_[page] = false;
            free_pages_.push_back(page);
            if (used_pages_ > 0) --used_pages_;
        }
    }

    // Pin pages held by a prefix-cache entry so they can never be handed out
    // by allocate_physical_page or returned to the free stack while live.
    void pin_pages(const std::vector<int32_t> &pages) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            if (page_pin_refcount_[page]++ == 0) ++pinned_pages_;
        }
    }

    // Drop the cache's pin on pages (called just before releasing them at
    // eviction). Does NOT return them to the free stack; pair with
    // release_physical_pages.
    void unpin_pages(const std::vector<int32_t> &pages) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            if (page_pin_refcount_[page] > 0 && --page_pin_refcount_[page] == 0) {
                --pinned_pages_;
            }
        }
    }

    uint32_t pinned_pages() const {
        std::lock_guard<std::mutex> lk(mu_);
        return pinned_pages_;
    }

    uint32_t free_pages() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(free_pages_.size());
    }

    uint32_t used_pages() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return used_pages_;
    }

    uint32_t total_pages() const override { return total_pages_; }
    uint32_t page_size() const { return page_size_; }

private:
    uint32_t total_pages_ = 0;
    uint32_t page_size_ = 0;
    mutable std::mutex mu_;
    std::vector<int32_t> free_pages_;
    std::vector<uint32_t> page_pin_refcount_;
    std::vector<bool> allocated_;
    uint32_t pinned_pages_ = 0;
    std::function<uint32_t()> evict_cb_;
    uint32_t used_pages_ = 0;
};

} // namespace qw3
