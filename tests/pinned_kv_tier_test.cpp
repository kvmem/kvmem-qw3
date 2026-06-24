// PinnedKvTier host-logic unit test (CPU pinned KV tier, Task #43).
//
// Pure host logic — no GPU. Covers: slot sizing from byte budget, slot
// allocation/release (LIFO), block<->slot residency map, full-pool behavior
// (no eviction in v1 skeleton), and LRU victim ordering.

#include "qw3/pinned_kv_tier.hpp"

#include <cstdio>

using namespace qw3;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static void test_sizing() {
    // 100 MB budget, 8 MB slots -> 12 whole slots (96 MB), remainder unused.
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 100ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;
    PinnedKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.slot_count() == 12);
    CHECK(t.free_slots() == 12);
    CHECK(t.used_slots() == 0);
    // slot byte offsets are slot_index * slot_bytes.
    CHECK(t.slot_offset(0) == 0);
    CHECK(t.slot_offset(3) == 3ull * cfg.slot_bytes);
}

static void test_disabled() {
    PinnedKvTierConfig cfg;  // total_bytes=0 -> disabled
    PinnedKvTier t(cfg);
    CHECK(!t.enabled());
    CHECK(t.slot_count() == 0);
    CHECK(t.place_block(0).slot == -1);  // nothing to hand out

    PinnedKvTierConfig small;
    small.total_bytes = 4ull * 1024 * 1024;   // smaller than one slot
    small.slot_bytes = 8ull * 1024 * 1024;
    PinnedKvTier t2(small);
    CHECK(!t2.enabled());
    CHECK(t2.slot_count() == 0);
}

static void test_place_release() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 32ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 4 slots
    PinnedKvTier t(cfg);
    CHECK(t.slot_count() == 4);

    // Place three blocks; LIFO hands out 0,1,2.
    auto p0 = t.place_block(10);
    auto p1 = t.place_block(11);
    auto p2 = t.place_block(12);
    CHECK(p0.slot == 0 && p0.evicted_block == -1);
    CHECK(p1.slot == 1);
    CHECK(p2.slot == 2);
    CHECK(t.used_slots() == 3 && t.free_slots() == 1);

    // Residency lookups.
    CHECK(t.block_slot(10) == 0);
    CHECK(t.block_slot(11) == 1);
    CHECK(t.block_slot(99) == -1);

    // Re-placing an already-resident block returns its slot, no new alloc.
    auto p0b = t.place_block(10);
    CHECK(p0b.slot == 0);
    CHECK(t.used_slots() == 3);

    // Release block 11 -> its slot returns to the pool, reused next.
    t.release_block(11);
    CHECK(t.block_slot(11) == -1);
    CHECK(t.free_slots() == 2);
    auto p3 = t.place_block(13);
    CHECK(p3.slot == 1);  // freed slot 1 reused (LIFO)

    // Releasing a non-resident block is a no-op.
    t.release_block(12345);
    CHECK(t.free_slots() == 1);
}

static void test_full_no_evict() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 16ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 2 slots
    PinnedKvTier t(cfg);
    CHECK(t.place_block(1).slot == 0);
    CHECK(t.place_block(2).slot == 1);
    // Full: v1 skeleton returns slot=-1, no eviction.
    auto full = t.place_block(3);
    CHECK(full.slot == -1);
    CHECK(full.evicted_block == -1);
    CHECK(t.block_slot(3) == -1);
    CHECK(t.used_slots() == 2);
}

static void test_full_with_explicit_evict() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 16ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 2 slots
    PinnedKvTier t(cfg);
    CHECK(t.place_block(1).slot == 0);
    CHECK(t.place_block(2).slot == 1);
    t.touch(1);  // block 2 is now LRU.

    auto p = t.place_block_evicting(3);
    CHECK(p.slot == 1);
    CHECK(p.evicted_block == 2);
    CHECK(t.block_slot(2) == -1);
    CHECK(t.block_slot(3) == 1);
    CHECK(t.used_slots() == 2);
    CHECK(t.lru_victim() == 1);
}

static void test_lru_order() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 32ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 4 slots
    PinnedKvTier t(cfg);
    t.place_block(1);
    t.place_block(2);
    t.place_block(3);
    // LRU victim is the oldest placed: block 1.
    CHECK(t.lru_victim() == 1);
    // Touch block 1 -> it becomes most recent, victim is now block 2.
    t.touch(1);
    CHECK(t.lru_victim() == 2);
    // Release block 2 -> victim is block 3.
    t.release_block(2);
    CHECK(t.lru_victim() == 3);

    t.clear();
    CHECK(t.lru_victim() == -1);
    CHECK(t.free_slots() == 4);
    CHECK(t.block_slot(1) == -1);
}

int main() {
    test_sizing();
    test_disabled();
    test_place_release();
    test_full_no_evict();
    test_full_with_explicit_evict();
    test_lru_order();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
