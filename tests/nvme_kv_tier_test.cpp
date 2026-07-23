// NvmeKvTier host-logic test. Covers slot sizing, block residency, read/write,
// explicit LRU eviction, and release/reuse.

#include "qw3/nvme_kv_tier.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <future>
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
    cfg.drop_page_cache = true;
    NvmeKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.drops_page_cache());
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

static void test_coalesced_batch_io() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 64 * 8;
    cfg.slot_bytes = 64;
    cfg.drop_page_cache = true;
    NvmeKvTier t(cfg);
    CHECK(t.drops_page_cache());

    std::vector<uint8_t> input(64 * 3), output(64 * 3, 0);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint8_t>((i * 17) & 0xff);
    }
    std::vector<NvmeIoSpan> spans;
    for (uint32_t block = 0; block < 3; ++block) {
        const auto p = t.place_block(100 + block);
        CHECK(p.slot == static_cast<int32_t>(block));
        spans.push_back(NvmeIoSpan{
            p.slot, static_cast<uint64_t>(block) * 64, 64});
    }

    NvmeBatchIoStats writes;
    t.write_spans(spans, input.data(), input.size(), &writes);
    CHECK(writes.bytes == input.size());
    CHECK(writes.syscalls == 1);
    CHECK(writes.cache_drop_bytes == input.size());
    CHECK(writes.cache_drop_failures == 0);

    NvmeBatchIoStats reads;
    t.read_spans(spans, output.data(), output.size(), &reads);
    CHECK(reads.bytes == output.size());
    CHECK(reads.syscalls == 1);
    CHECK(reads.cache_drop_bytes == output.size());
    CHECK(reads.cache_drop_failures == 0);
    CHECK(output == input);
}

static void test_concurrent_positional_batches() {
    NvmeKvTierConfig cfg;
    cfg.dir = temp_dir();
    cfg.total_bytes = 64 * 8;
    cfg.slot_bytes = 64;
    NvmeKvTier t(cfg);

    std::vector<uint8_t> a(64 * 3, 0x35);
    std::vector<uint8_t> b(64 * 3, 0xca);
    std::vector<uint8_t> output(64 * 6, 0);
    std::vector<NvmeIoSpan> a_spans;
    std::vector<NvmeIoSpan> b_spans;
    for (uint32_t block = 0; block < 6; ++block) {
        const auto p = t.place_block(200 + block);
        CHECK(p.slot == static_cast<int32_t>(block));
        auto &spans = block < 3 ? a_spans : b_spans;
        spans.push_back(NvmeIoSpan{
            p.slot, static_cast<uint64_t>(block % 3) * 64, 64});
    }

    auto aw = std::async(std::launch::async, [&]() {
        t.write_spans(a_spans, a.data(), a.size());
    });
    auto bw = std::async(std::launch::async, [&]() {
        t.write_spans(b_spans, b.data(), b.size());
    });
    aw.get();
    bw.get();

    std::vector<NvmeIoSpan> all;
    for (uint32_t slot = 0; slot < 6; ++slot) {
        all.push_back(NvmeIoSpan{
            static_cast<int32_t>(slot),
            static_cast<uint64_t>(slot) * 64, 64});
    }
    NvmeBatchIoStats reads;
    t.read_spans(all, output.data(), output.size(), &reads);
    CHECK(reads.syscalls == 1);
    CHECK(std::equal(a.begin(), a.end(), output.begin()));
    CHECK(std::equal(
        b.begin(), b.end(), output.begin() +
            static_cast<std::ptrdiff_t>(a.size())));
}

int main() {
    test_disabled();
    test_write_read_release();
    test_evicting_place();
    test_coalesced_batch_io();
    test_concurrent_positional_batches();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
