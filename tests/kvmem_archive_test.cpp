// KVMem Context Archive host-logic test. Covers the layout key and its
// mismatch diagnosis, manifest round-trip and commit-on-rename, validity
// bitmaps, the token stream, ladder state files, sealing preconditions,
// prefix-truncation ladder lookup, resumable-build detection, and the
// durable + direct-mapped NvmeKvTier modes the archive relies on.

#include "qw3/kvmem_archive.hpp"
#include "qw3/nvme_kv_tier.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

using namespace qw3;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

#define CHECK_THROWS(expr) do {                                            \
    bool threw = false;                                                    \
    try { expr; } catch (const std::exception &) { threw = true; }         \
    if (!threw) {                                                          \
        std::printf("FAIL %s:%d  expected throw: %s\n",                    \
                    __FILE__, __LINE__, #expr);                            \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static std::string temp_root() {
    const char *base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    return std::string(base) + "/qw3_kvmem_archive_test";
}

static void remove_tree(const std::string &dir) {
    const std::string cmd = "rm -rf '" + dir + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::printf("warning: could not clean %s\n", dir.c_str());
    }
}

// Qwen3.6-27B under fp8: 64 layers, every 4th standard, 4 KV heads, 256 dims.
static KvMemArchiveLayout sample_layout() {
    KvMemArchiveLayout l;
    l.architecture = "qwen35";
    l.model_name = "Qwen3.6-27B-Q8_0.gguf";
    l.model_bytes = 29047084160ull;
    l.model_sha256 = std::string(64, 'a');
    l.n_layers = 64;
    l.full_attention_interval = 4;
    l.n_standard_layers = 16;
    l.n_kv_heads = 4;
    l.head_dim = 256;
    l.head_v_dim = 256;
    l.rope_dim = 64;
    l.rope_theta = 1e7;
    l.kv_dtype = "fp8";
    l.block_tokens = 128;
    l.raw_chunk_tokens = 2048;
    l.kv_page_size = 16;
    l.immutable_source_k = true;
    l.raw_k_block_major = true;
    l.mtp_archived = false;
    l.raw_k_row_bytes = 4ull * 256ull;             // n_kv_heads * head_dim * 1
    l.raw_chunk_bytes = 16ull * 2048ull * 1024ull; // layers * tokens * row
    l.mtp_chunk_bytes = 0;
    l.v_block_bytes = 16ull * 128ull * 1024ull;
    return l;
}

static void test_layout_key_and_mismatch() {
    const KvMemArchiveLayout a = sample_layout();
    KvMemArchiveLayout b = a;
    CHECK(a.key() == b.key());
    CHECK(a.explain_mismatch(b).empty());

    // Policy-side settings are not part of the layout at all, so the only way
    // to differ is through a genuine layout field.
    b.block_tokens = 32;
    CHECK(a.key() != b.key());
    const std::string why = a.explain_mismatch(b);
    CHECK(why.find("block_tokens") != std::string::npos);
    CHECK(why.find("archive=128") != std::string::npos);
    CHECK(why.find("engine=32") != std::string::npos);

    KvMemArchiveLayout c = a;
    c.kv_dtype = "fp16";
    CHECK(a.explain_mismatch(c).find("kv_dtype") != std::string::npos);

    KvMemArchiveLayout d = a;
    d.rope_theta = 5e5;
    CHECK(a.explain_mismatch(d).find("rope_theta") != std::string::npos);

    KvMemArchiveLayout e = a;
    e.model_sha256 = std::string(64, 'b');
    CHECK(a.explain_mismatch(e).find("model_sha256") != std::string::npos);
}

static void test_model_sha256_and_cache() {
    const std::string dir = temp_root() + "/digest";
    remove_tree(dir);
    CHECK(::mkdir(temp_root().c_str(), 0755) == 0 || errno == EEXIST);
    CHECK(::mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST);
    const std::string path = dir + "/tiny.gguf";
    const int fd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC,
                          0644);
    CHECK(fd >= 0);
    if (fd < 0) return;
    CHECK(::write(fd, "abc", 3) == 3);
    CHECK(::close(fd) == 0);
    const std::string cache = dir + "/cache";
    CHECK(::setenv("QW3_MODEL_DIGEST_CACHE_DIR", cache.c_str(), 1) == 0);
    const std::string expected =
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    CHECK(kvmem_archive_model_sha256(path) == expected);
    // The second call exercises the stat-validated sidecar hit.
    CHECK(kvmem_archive_model_sha256(path) == expected);
    // Mutating the same inode must invalidate the sidecar through size/time
    // metadata rather than returning the stale digest.
    const int append_fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CLOEXEC);
    CHECK(append_fd >= 0);
    CHECK(::write(append_fd, "d", 1) == 1);
    CHECK(::close(append_fd) == 0);
    CHECK(kvmem_archive_model_sha256(path) ==
          "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589");
    CHECK(::unsetenv("QW3_MODEL_DIGEST_CACHE_DIR") == 0);
    remove_tree(dir);
}

static void test_new_archive_rejects_non_hex_model_digest() {
    const std::string dir = temp_root() + "/bad_digest";
    remove_tree(dir);
    KvMemArchiveLayout layout = sample_layout();
    layout.model_sha256 = std::string(64, 'z');
    CHECK_THROWS((void)KvMemArchive::open_for_build(dir, layout, ""));
    remove_tree(dir);
}

static void test_build_seal_attach() {
    const std::string dir = temp_root() + "/basic";
    remove_tree(dir);
    const KvMemArchiveLayout layout = sample_layout();

    // 4 blocks of 128 tokens = 512 tokens, one 2048-token chunk.
    const uint64_t total_tokens = 512;
    const uint32_t total_blocks = 4;
    const uint32_t raw_chunks = 1;

    {
        auto a = KvMemArchive::open_for_build(dir, layout, "budget=131072");
        CHECK(a->writable());
        CHECK(!a->manifest().sealed);
        CHECK(a->manifest().layout.key() == layout.key());

        std::vector<uint32_t> tokens(total_tokens);
        for (uint64_t i = 0; i < total_tokens; ++i) {
            tokens[static_cast<size_t>(i)] = static_cast<uint32_t>(i * 7 + 1);
        }
        a->append_tokens(tokens.data(), tokens.size());
        CHECK(a->token_count() == total_tokens);
        const uint32_t speculative[] = {0xdeadbeefu, 0xcafebabeu};
        a->append_tokens(speculative, 2);
        CHECK(a->token_count() == total_tokens + 2);
        a->truncate_tokens(total_tokens);
        CHECK(a->token_count() == total_tokens);

        // Sealing must refuse while any payload is still missing.
        CHECK_THROWS(a->seal(total_tokens, total_blocks, raw_chunks));
        a->mark_chunk_valid(0);
        CHECK_THROWS(a->seal(total_tokens, total_blocks, raw_chunks));
        for (uint32_t b = 0; b < total_blocks; ++b) a->mark_block_valid(b);

        KvMemArchiveState st;
        st.position = 256;
        st.kvmem_registered_pos = 256;
        st.mtp_prefix_len = 0;
        st.entries.push_back({KvMemArchiveStateEntry::Hidden, 0, 8});
        st.entries.push_back({KvMemArchiveStateEntry::Recurrent, 3, 16});
        st.payload.assign(24, 0);
        for (size_t i = 0; i < st.payload.size(); ++i) {
            st.payload[i] = static_cast<uint8_t>(i + 1);
        }
        a->write_state(st);

        KvMemArchiveState st2 = st;
        st2.position = 512;
        st2.kvmem_registered_pos = 512;
        st2.payload.assign(24, 0xab);
        a->write_state(st2);

        a->seal(total_tokens, total_blocks, raw_chunks);
        CHECK(a->manifest().sealed);
    }

    {
        auto a = KvMemArchive::attach(dir);
        CHECK(!a->writable());
        CHECK(a->manifest().format_version == 3);
        CHECK(a->manifest().sealed);
        CHECK(a->manifest().total_tokens == total_tokens);
        CHECK(a->manifest().total_blocks == total_blocks);
        CHECK(a->manifest().raw_chunks == raw_chunks);
        CHECK(a->manifest().policy_snapshot == "budget=131072");
        CHECK(a->manifest().layout.explain_mismatch(layout).empty());
        CHECK(a->manifest().ladder.size() == 2);
        CHECK(a->manifest().ladder[0] == 256);
        CHECK(a->manifest().ladder[1] == 512);

        CHECK(a->chunk_valid(0));
        CHECK(!a->chunk_valid(1));
        for (uint32_t b = 0; b < total_blocks; ++b) CHECK(a->block_valid(b));
        CHECK(!a->block_valid(total_blocks));

        const std::vector<uint32_t> head = a->read_tokens(0, 4);
        CHECK(head.size() == 4);
        CHECK(head[0] == 1);
        CHECK(head[3] == 22);
        const std::vector<uint32_t> tail = a->read_tokens(total_tokens - 1, 1);
        CHECK(tail[0] == static_cast<uint32_t>((total_tokens - 1) * 7 + 1));
        CHECK_THROWS((void)a->read_tokens(total_tokens, 1));

        const KvMemArchiveState st = a->read_state(256);
        CHECK(st.position == 256);
        CHECK(st.kvmem_registered_pos == 256);
        CHECK(st.entries.size() == 2);
        CHECK(st.entries[0].kind == KvMemArchiveStateEntry::Hidden);
        CHECK(st.entries[1].kind == KvMemArchiveStateEntry::Recurrent);
        CHECK(st.entries[1].index == 3);
        CHECK(st.payload.size() == 24);
        CHECK(st.payload[0] == 1);
        CHECK(st.payload[23] == 24);

        const KvMemArchiveState st2 = a->read_state(512);
        CHECK(st2.payload[0] == 0xab);

        // A read-only attach must refuse every mutation.
        CHECK_THROWS(a->append_tokens(head.data(), 1));
        CHECK_THROWS(a->truncate_tokens(total_tokens - 1));
        CHECK_THROWS(a->write_state(st));
    }

    remove_tree(dir);
}

static void test_ladder_lookup_for_truncation() {
    const std::string dir = temp_root() + "/ladder";
    remove_tree(dir);
    KvMemArchiveLayout layout = sample_layout();

    auto a = KvMemArchive::open_for_build(dir, layout, "");
    std::vector<uint32_t> tokens(1024, 5);
    a->append_tokens(tokens.data(), tokens.size());
    for (uint64_t pos : {256ull, 512ull, 768ull}) {
        KvMemArchiveState st;
        st.position = pos;
        st.kvmem_registered_pos = pos;
        a->write_state(st);
    }
    a->mark_chunk_valid(0);
    for (uint32_t b = 0; b < 8; ++b) a->mark_block_valid(b);
    a->seal(1024, 8, 1);

    // Truncating to N restores the largest ladder point at or below N and
    // replays the residual, so the lookup must never overshoot.
    CHECK(a->ladder_at_or_below(768) == 768);
    CHECK(a->ladder_at_or_below(700) == 512);
    CHECK(a->ladder_at_or_below(512) == 512);
    CHECK(a->ladder_at_or_below(511) == 256);
    CHECK(a->ladder_at_or_below(1024) == 768);
    CHECK(a->ladder_at_or_below(255) == UINT64_MAX);

    remove_tree(dir);
}

static void test_resume_unsealed_build() {
    const std::string dir = temp_root() + "/resume";
    remove_tree(dir);
    const KvMemArchiveLayout layout = sample_layout();

    {
        auto a = KvMemArchive::open_for_build(dir, layout, "p1");
        std::vector<uint32_t> tokens(512, 3);
        a->append_tokens(tokens.data(), tokens.size());
        a->mark_chunk_valid(0);
        for (uint32_t b = 0; b < 4; ++b) a->mark_block_valid(b);
        KvMemArchiveState st;
        st.position = 512;
        st.kvmem_registered_pos = 512;
        a->write_state(st);
        // A ladder point beyond the durable payload must not be a resume point.
        KvMemArchiveState ahead;
        ahead.position = 4096;
        ahead.kvmem_registered_pos = 4096;
        a->write_state(ahead);
        a->checkpoint_metadata(512, 4, 1);
    }

    {
        // Reopening an unsealed archive resumes it instead of starting over.
        auto a = KvMemArchive::open_for_build(dir, layout, "p2");
        CHECK(a->token_count() == 512);
        CHECK(a->chunk_valid(0));
        CHECK(a->block_valid(3));
        CHECK(a->resume_position() == 512);
        CHECK(a->manifest().policy_snapshot == "p2");
        CHECK_THROWS((void)KvMemArchive::attach(dir));

        std::vector<uint32_t> more(512, 9);
        a->append_tokens(more.data(), more.size());
        for (uint32_t b = 4; b < 8; ++b) a->mark_block_valid(b);
        a->seal(1024, 8, 1);
    }

    {
        auto a = KvMemArchive::attach(dir);
        CHECK(a->manifest().total_tokens == 1024);
        const std::vector<uint32_t> t = a->read_tokens(600, 1);
        CHECK(t[0] == 9);
    }

    // A layout change must refuse to resume rather than corrupt the payload.
    KvMemArchiveLayout other = layout;
    other.block_tokens = 32;
    CHECK_THROWS((void)KvMemArchive::open_for_build(dir, other, ""));
    // And a sealed archive is never reopened for writing.
    CHECK_THROWS((void)KvMemArchive::open_for_build(dir, layout, ""));

    remove_tree(dir);
}

static void test_direct_mapped_durable_tier() {
    const std::string dir = temp_root() + "/tier";
    remove_tree(dir);
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::printf("FAIL could not create %s\n", dir.c_str());
        ++g_fail;
        return;
    }
    const std::string path = dir + "/v.bin";

    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = "v.bin";
        cfg.slot_bytes = 64;
        cfg.total_bytes = 64 * 8;
        cfg.durable = true;
        cfg.direct_mapped = true;
        cfg.preallocate = true;
        NvmeKvTier t(cfg);
        CHECK(t.enabled());
        CHECK(t.direct_mapped());
        CHECK(t.slot_count() == 8);
        // Durable arenas keep their directory entry; ephemeral ones unlink it.
        CHECK(::access(path.c_str(), F_OK) == 0);
        struct stat st {};
        CHECK(::stat(path.c_str(), &st) == 0);
        CHECK(static_cast<uint64_t>(st.st_size) == cfg.total_bytes);
        CHECK(static_cast<uint64_t>(st.st_blocks) * 512 >= cfg.total_bytes);

        // Identity placement: the slot is the block id, in any order, with no
        // free list to exhaust and no eviction.
        for (uint32_t b : {5u, 0u, 7u}) {
            const NvmeSlotPlacement p = t.place_block(b);
            CHECK(p.slot == static_cast<int32_t>(b));
            CHECK(p.evicted_block == -1);
        }
        CHECK(t.block_slot(5) == 5);
        CHECK(t.block_slot(3) == -1);
        CHECK(t.used_slots() == 3);

        std::vector<uint8_t> payload(64);
        for (size_t i = 0; i < payload.size(); ++i) {
            payload[i] = static_cast<uint8_t>(i ^ 0x5a);
        }
        t.write_block(7, payload.data(), payload.size());
        std::vector<uint8_t> back(64, 0);
        t.read_block(7, back.data(), back.size());
        CHECK(back == payload);
        // Out-of-range ids fail placement instead of evicting a live block.
        CHECK(t.place_block(8).slot == -1);
    }

    {
        // Reattach read-only: mappings are recovered from the id, not a table.
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = "v.bin";
        cfg.slot_bytes = 64;
        cfg.total_bytes = 64 * 8;
        cfg.durable = true;
        cfg.direct_mapped = true;
        cfg.read_only = true;
        NvmeKvTier t(cfg);
        CHECK(t.read_only());
        t.mark_present_range(0, 8);
        CHECK(t.block_slot(7) == 7);
        CHECK(t.used_slots() == 8);

        std::vector<uint8_t> back(64, 0);
        t.read_block(7, back.data(), back.size());
        CHECK(back[0] == 0x5a);
        CHECK(back[1] == (1 ^ 0x5a));

        std::vector<uint8_t> payload(64, 1);
        CHECK_THROWS(t.write_block(1, payload.data(), payload.size()));
    }

    remove_tree(dir);
}

static void test_read_only_overlay_copy_on_write() {
    const std::string dir = temp_root() + "/overlay";
    remove_tree(dir);
    if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
        std::printf("FAIL could not create %s\n", dir.c_str());
        ++g_fail;
        return;
    }
    const std::string base_path = dir + "/base.bin";
    const std::string overlay_path = dir + "/scratch.bin";
    constexpr uint64_t slot_bytes = 64;
    constexpr uint32_t slots = 4;

    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = "base.bin";
        cfg.slot_bytes = slot_bytes;
        cfg.total_bytes = slot_bytes * slots;
        cfg.durable = true;
        cfg.direct_mapped = true;
        NvmeKvTier t(cfg);
        for (uint32_t slot = 0; slot < slots; ++slot) {
            std::vector<uint8_t> bytes(
                slot_bytes, static_cast<uint8_t>(0x10 + slot));
            t.write_block(slot, bytes.data(), bytes.size());
        }
    }

    std::vector<uint8_t> base_before(slot_bytes * slots);
    {
        const int fd = ::open(base_path.c_str(), O_RDONLY | O_CLOEXEC);
        CHECK(fd >= 0);
        if (fd >= 0) {
            CHECK(::pread(fd, base_before.data(), base_before.size(), 0) ==
                  static_cast<ssize_t>(base_before.size()));
            ::close(fd);
        }
    }

    {
        NvmeKvTierConfig cfg;
        cfg.dir = dir;
        cfg.file_name = "base.bin";
        cfg.slot_bytes = slot_bytes;
        cfg.total_bytes = slot_bytes * slots;
        cfg.durable = true;
        cfg.direct_mapped = true;
        cfg.read_only = true;
        cfg.overlay_dir = dir;
        cfg.overlay_file_name = "scratch.bin";
        NvmeKvTier t(cfg);
        CHECK(t.has_overlay());
        CHECK(!t.read_only());
        CHECK(::access(overlay_path.c_str(), F_OK) != 0);
        t.mark_present_range(0, slots);

        const std::vector<uint8_t> patch(8, 0xa5);
        t.write_slot_range(1, 16, patch.data(), patch.size());
        std::vector<uint8_t> slot1(slot_bytes, 0);
        t.read_block(1, slot1.data(), slot1.size());
        for (size_t i = 0; i < slot1.size(); ++i) {
            CHECK(slot1[i] == (i >= 16 && i < 24 ? 0xa5 : 0x11));
        }

        const std::vector<uint8_t> replacement(slot_bytes, 0x7c);
        t.write_block(2, replacement.data(), replacement.size());
        std::vector<uint8_t> all(slot_bytes * slots, 0);
        std::vector<NvmeIoSpan> spans;
        for (uint32_t slot = 0; slot < slots; ++slot) {
            spans.push_back({static_cast<int32_t>(slot),
                             slot * slot_bytes, slot_bytes});
        }
        t.read_spans(spans, all.data(), all.size());
        CHECK(all[0] == 0x10);
        CHECK(all[slot_bytes + 16] == 0xa5);
        CHECK(all[2 * slot_bytes] == 0x7c);
        CHECK(all[3 * slot_bytes] == 0x13);
    }

    std::vector<uint8_t> base_after(slot_bytes * slots);
    {
        const int fd = ::open(base_path.c_str(), O_RDONLY | O_CLOEXEC);
        CHECK(fd >= 0);
        if (fd >= 0) {
            CHECK(::pread(fd, base_after.data(), base_after.size(), 0) ==
                  static_cast<ssize_t>(base_after.size()));
            ::close(fd);
        }
    }
    CHECK(base_after == base_before);
    remove_tree(dir);
}

int main() {
    test_layout_key_and_mismatch();
    test_model_sha256_and_cache();
    test_new_archive_rejects_non_hex_model_digest();
    test_build_seal_attach();
    test_ladder_lookup_for_truncation();
    test_resume_unsealed_build();
    test_direct_mapped_durable_tier();
    test_read_only_overlay_copy_on_write();
    if (g_fail == 0) std::printf("qw3-kvmem-archive: all checks passed\n");
    return g_fail == 0 ? 0 : 1;
}
