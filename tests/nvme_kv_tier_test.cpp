// NvmeKvTier host-logic test. Covers slot sizing, block residency, read/write,
// explicit LRU eviction, and release/reuse.

#include "qw3/nvme_kv_tier.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace qw3;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static std::string temp_dir() {
    const char *base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    return std::string(base) + "/qw3_nvme_kv_tier_test";
}

static void test_disabled() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 0;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    CHECK(!t.enabled());
    CHECK(t.slot_count() == 0);
}

static void test_write_read_release() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 256;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.slot_count() == 4);

    std::vector<uint8_t> a(64), b(64), out(64);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<uint8_t>(i);
        b[i] = static_cast<uint8_t>(255 - i);
    }

    t.write_block(10, a.data(), a.size());
    CHECK(t.block_slot(10) == 0);
    t.read_block(10, out.data(), out.size());
    CHECK(out == a);

    t.write_block(10, b.data(), b.size());
    t.read_block(10, out.data(), out.size());
    CHECK(out == b);

    t.release_block(10);
    CHECK(t.block_slot(10) == -1);
    CHECK(t.free_slots() == 4);
}

static void test_evicting_place() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 128;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);
    std::vector<uint8_t> a(64, 1), b(64, 2), c(64, 3), out(64);
    t.write_block(1, a.data(), a.size());
    t.write_block(2, b.data(), b.size());
    t.touch(1);  // block 2 becomes LRU.

    auto p = t.place_block_evicting(3);
    CHECK(p.slot == 1);
    CHECK(p.evicted_block == 2);
    CHECK(t.block_slot(2) == -1);
    CHECK(t.block_slot(3) == 1);

    t.write_block(3, c.data(), c.size());
    t.read_block(3, out.data(), out.size());
    CHECK(out == c);
}

int main() {
    test_disabled();
    test_write_read_release();
    test_evicting_place();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
