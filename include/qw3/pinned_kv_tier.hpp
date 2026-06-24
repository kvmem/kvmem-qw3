#pragma once

// CPU pinned (page-locked) KV tier — host-side slot bookkeeping.
//
// Block-sparse KV attention (see kv_block_store.hpp) keeps the whole context's
// KV resident, but a long agent session can exceed GPU memory. The CPU tier is
// a second-level cache between GPU and NVMe: cold blocks spill from GPU into a
// fixed pool of page-locked host slots (one slot holds one block's KV across
// all attention layers), and a selected block stages back to GPU on demand.
//
// This module is PURE HOST LOGIC: a fixed-count slab allocator (LIFO free list,
// mirroring GlobalKvPagePool) plus a block_id <-> slot map. It owns NO memory
// and issues NO copies — the CUDA backend supplies the cudaHostAlloc'd buffer
// and performs the actual D2H/H2D over copy_stream_. Keeping it host-only makes
// the eviction/placement math unit-testable without a GPU (see
// tests/pinned_kv_tier_test.cpp).
//
// Slot sizing is the caller's contract: slot_bytes must be >= the byte size of
// one block's KV summed across all standard-attention layers. The pool only
// tracks slot indices; byte offsets are slot_index * slot_bytes.

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qw3 {

struct PinnedKvTierConfig {
    uint64_t total_bytes = 0;  // --kv-store-cpu-bytes (0 = tier disabled)
    uint64_t slot_bytes = 0;   // bytes per block KV across all attention layers
};

// Result of placing a block into the CPU tier. When the pool is full and a
// victim must be evicted to NVMe (or dropped), `evicted_block` names it
// (-1 = none evicted) so the caller can spill it onward before reusing its
// slot. v1 skeleton never evicts (returns slot=-1 when full); the eviction
// hook is reserved for the NVMe tier task.
struct PinnedSlotPlacement {
    int32_t slot = -1;            // assigned host slot index (-1 = pool full)
    int32_t evicted_block = -1;   // block displaced to make room (-1 = none)
};

class PinnedKvTier {
public:
    explicit PinnedKvTier(PinnedKvTierConfig cfg) : cfg_(cfg) {
        if (cfg_.slot_bytes > 0 && cfg_.total_bytes >= cfg_.slot_bytes) {
            slot_count_ = static_cast<uint32_t>(cfg_.total_bytes / cfg_.slot_bytes);
        }
        free_slots_.reserve(slot_count_);
        // LIFO free list: hand out low indices first (push high..low).
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(static_cast<int32_t>(slot_count_ - 1U - i));
        }
    }

    const PinnedKvTierConfig &config() const { return cfg_; }
    bool enabled() const { return slot_count_ > 0; }
    uint32_t slot_count() const { return slot_count_; }
    uint32_t free_slots() const { return static_cast<uint32_t>(free_slots_.size()); }
    uint32_t used_slots() const { return slot_count_ - free_slots(); }

    // Byte offset of a slot within the pinned buffer.
    uint64_t slot_offset(int32_t slot) const {
        return static_cast<uint64_t>(slot) * cfg_.slot_bytes;
    }

    // Is this block currently resident in the CPU tier? Returns its slot or -1.
    int32_t block_slot(uint32_t block_id) const {
        auto it = block_to_slot_.find(block_id);
        return it == block_to_slot_.end() ? -1 : it->second;
    }

    // Reserve a slot for `block_id` (the caller then D2H-copies the block's KV
    // into slot_offset(slot)). If the block is already resident its existing
    // slot is returned unchanged. When the pool is full, v1 returns slot=-1
    // (no eviction); the caller must keep the block on GPU. Marks the block
    // resident on success.
    PinnedSlotPlacement place_block(uint32_t block_id) {
        PinnedSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            return out;
        }
        if (free_slots_.empty()) return out;  // full: slot stays -1
        const int32_t slot = free_slots_.back();
        free_slots_.pop_back();
        block_to_slot_[block_id] = slot;
        touch(block_id);
        out.slot = slot;
        return out;
    }

    // Reserve a slot and evict the LRU resident block if the pool is full.
    // The caller must spill `evicted_block` onward (e.g. to NVMe) before
    // overwriting the returned slot's bytes.
    PinnedSlotPlacement place_block_evicting(uint32_t block_id) {
        PinnedSlotPlacement out;
        auto it = block_to_slot_.find(block_id);
        if (it != block_to_slot_.end()) {
            out.slot = it->second;
            touch(block_id);
            return out;
        }
        if (!free_slots_.empty()) return place_block(block_id);
        if (lru_.empty()) return out;

        const uint32_t victim = lru_.front();
        auto vit = block_to_slot_.find(victim);
        if (vit == block_to_slot_.end()) {
            erase_lru(victim);
            return out;
        }
        const int32_t slot = vit->second;
        block_to_slot_.erase(vit);
        erase_lru(victim);
        block_to_slot_[block_id] = slot;
        touch(block_id);
        out.slot = slot;
        out.evicted_block = static_cast<int32_t>(victim);
        return out;
    }

    // Release a block's slot back to the pool (it was staged back to GPU, or
    // the session ended). No-op if the block is not resident.
    void release_block(uint32_t block_id) {
        auto it = block_to_slot_.find(block_id);
        if (it == block_to_slot_.end()) return;
        free_slots_.push_back(it->second);
        block_to_slot_.erase(it);
        erase_lru(block_id);
    }

    // Mark a block most-recently-used (called on stage-in / access) so the LRU
    // victim picked by evict_lru_victim() is the coldest resident block.
    void touch(uint32_t block_id) {
        erase_lru(block_id);
        lru_.push_back(block_id);
    }

    // Pick (without removing) the least-recently-used resident block, for the
    // NVMe spill path to displace. Returns -1 when the tier is empty.
    int32_t lru_victim() const {
        return lru_.empty() ? -1 : static_cast<int32_t>(lru_.front());
    }

    void clear() {
        free_slots_.clear();
        for (uint32_t i = 0; i < slot_count_; ++i) {
            free_slots_.push_back(static_cast<int32_t>(slot_count_ - 1U - i));
        }
        block_to_slot_.clear();
        lru_.clear();
    }

private:
    void erase_lru(uint32_t block_id) {
        for (auto it = lru_.begin(); it != lru_.end(); ++it) {
            if (*it == block_id) { lru_.erase(it); return; }
        }
    }

    PinnedKvTierConfig cfg_;
    uint32_t slot_count_ = 0;
    std::vector<int32_t> free_slots_;
    std::unordered_map<uint32_t, int32_t> block_to_slot_;
    std::vector<uint32_t> lru_;  // front = least recently used
};

} // namespace qw3
