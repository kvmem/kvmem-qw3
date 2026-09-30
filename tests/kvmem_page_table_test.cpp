#include "qwen_executor.hpp"
#include "local_kv_page_pool.hpp"
#include "shared_kv_page_pool.hpp"
#include "global_kv_page_pool.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <stdexcept>

namespace qw3 {
struct KvPageTableTestAccess {
    using Table = QwenExecutor::KvPageTable;
    static void raw_storage(const QwenNativeModel &, const QwenWeights &, DeviceBackend &, const std::string &);
    static void mtp_recent_window(const QwenNativeModel &, const QwenWeights &, DeviceBackend &);
    static void historical_pins(const QwenNativeModel &, const QwenWeights &, DeviceBackend &);
    static void bounded_mrope(const QwenNativeModel &, const QwenWeights &, DeviceBackend &);
    static void bounded_config_guards(const QwenNativeModel &, const QwenWeights &, DeviceBackend &);
};
}
using namespace qw3;
using Table = KvPageTableTestAccess::Table;
static void check(bool v, const char *why) { if (!v) throw std::runtime_error(why); }

// Exercise the production page table's CPU allocation/H2D planning. No fake
// attention model is evaluated; numerical GPU/model tests remain separate.
struct RecordingBackend final : DeviceBackend {
    struct Tensor : DeviceTensor { std::vector<int32_t> words; };
    std::vector<std::pair<uint64_t, uint64_t>> uploads;
    uint32_t transfer_fences = 0;
    DeviceStatus record_kv_transfer_fence(std::unique_ptr<DeviceTransferFence> &fence) override {
        ++transfer_fences; fence.reset(); return {};
    }
    const char *name() const override { return "page-table-recorder"; }
    DeviceStatus begin() override { return {}; }
    DeviceStatus end() override { return {}; }
    DeviceStatus synchronize() override { return {}; }
    std::unique_ptr<DeviceTensor> tensor_f32(uint64_t count, const char *) override {
        auto t = std::make_unique<Tensor>(); t->count = count; t->words.resize(count); return t;
    }
    DeviceStatus copy_i32_from_host(DeviceTensor &dst, uint64_t offset, const int32_t *src, uint64_t n) override {
        auto &t = dynamic_cast<Tensor &>(dst);
        check(offset <= t.words.size() && n <= t.words.size() - offset, "page-table upload out of bounds");
        std::copy(src, src + n, t.words.begin() + offset);
        uploads.emplace_back(offset, n);
        return {};
    }
    std::unique_ptr<DeviceWeight> weight_f32(const float *, uint64_t n, const char *) override {
        auto w = std::make_unique<DeviceWeight>(); w->rows = n; w->cols = 1; return w;
    }
    std::unique_ptr<DeviceWeight> weight_q8_0(const void *, uint64_t, uint64_t, const char *) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus q8_0_get_row(DeviceTensor &, const DeviceWeight &, uint64_t) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus q8_0_matvec(DeviceTensor &, const DeviceWeight &, const DeviceTensor &) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus rms_norm(DeviceTensor &, const DeviceTensor &, const DeviceWeight &, float) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus add(DeviceTensor &, const DeviceTensor &, const DeviceTensor &) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus silu(DeviceTensor &, const DeviceTensor &) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus mul(DeviceTensor &, const DeviceTensor &, const DeviceTensor &) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus recurrent_single_token(DeviceTensor &, DeviceTensor &, DeviceTensor &, DeviceTensor &, const DeviceTensor &, const DeviceTensor &, const DeviceTensor &, const DeviceTensor &, const DeviceWeight &, const DeviceWeight &, const DeviceWeight &, const DeviceWeight &, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus attention_single_token(DeviceTensor &, DeviceTensor &, DeviceTensor &, const DeviceTensor &, const DeviceWeight &, const DeviceWeight &, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus rmsnorm_per_head(DeviceTensor &, const DeviceWeight &, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus rope_partial(DeviceTensor &, uint32_t, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus kv_append(DeviceTensor &, const DeviceTensor &, uint32_t, uint32_t) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus attention_decode(DeviceTensor &, DeviceTensor &, const DeviceTensor &, uint32_t, const DeviceTensor &, const DeviceTensor &, uint32_t, uint32_t, uint32_t, uint32_t, float) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus apply_attn_gate(DeviceTensor &, const DeviceTensor &, uint32_t, uint32_t, uint32_t) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceStatus zero_tensor(DeviceTensor &) override { throw std::logic_error("unexpected compute in page-table test"); }
    DeviceArgmax argmax(const DeviceTensor &) override { throw std::logic_error("unexpected compute in page-table test"); }
};

struct PagePool : KvPhysicalPageAllocator {
    explicit PagePool(uint32_t n) : live(n, false) {
        for (uint32_t i = 0; i < n; ++i) free.push_back(static_cast<int32_t>(i));
    }
    std::vector<int32_t> free;
    std::vector<bool> live;
    int32_t allocate_physical_page() override {
        if (free.empty()) throw std::runtime_error("test physical page pool exhausted");
        const auto p = free.back(); free.pop_back();
        check(!live[p], "physical page allocated twice"); live[p] = true; return p;
    }
    void release_physical_pages(const std::vector<int32_t> &pages) override {
        for (auto p : pages) {
            check(p >= 0 && size_t(p) < live.size() && live[p], "invalid or duplicate page release");
            live[p] = false; free.push_back(p);
        }
    }
    uint32_t free_pages() const override { return static_cast<uint32_t>(free.size()); }
    uint32_t used_pages() const override { return total_pages() - free_pages(); }
    uint32_t total_pages() const override { return static_cast<uint32_t>(live.size()); }
};

static void sparse_allocation_and_legacy_control() {
    RecordingBackend backend;
    PagePool pool(3);
    Table table; table.configure_bounded(128, &pool);
    check(table.pages.size() == 8 && pool.used_pages() == 0, "directory allocated payload eagerly");
    table.ensure_pages(backend, 128, 112, 16);
    check(pool.used_pages() == 1, "high slot allocated intervening holes");
    check(std::count(table.pages.begin(), table.pages.end(), -1) == 7, "unexpected resident pages");
    table.ensure_logical_page_ranges_resident(backend, {{2, 2}});
    check(pool.used_pages() == 3, "sparse range did not allocate only touched pages");
    const auto &device = dynamic_cast<RecordingBackend::Tensor &>(*table.device_pages).words;
    check(device == table.pages, "H2D image contains unpublished holes or stale aliases");
    table.release_logical_page_ranges(backend, {{7, 1}});
    check(pool.used_pages() == 2 && device[7] == -1, "stage-out did not invalidate the GPU page alias");
    table.ensure_pages(backend, 128, 115, 1);
    check(pool.used_pages() == 3 && table.pages[7] >= 0, "recycled sparse page not allocated");
    bool rejected = false;
    try { table.truncate_to_logical_pages(2); } catch (const std::logic_error &) { rejected = true; }
    check(rejected, "historical truncate incorrectly shortened a physical slot directory");
    table.reset();
    check(pool.used_pages() == 0 && table.pages.size() == 8, "bounded reset changed directory capacity");
    table.ensure_pages(backend, 128, 96, 16);
    check(pool.used_pages() == 1, "post-reset allocation filled holes");
    table.reset();

    PagePool dense_pool(8);
    Table dense; dense.configure(128, &dense_pool);
    dense.ensure_pages(backend, 128, 112, 16);
    check(dense_pool.used_pages() == 8, "legacy dense page-allocation behavior changed");
    dense.reset();
}

static void partial_batch_recycled_aliases() {
    RecordingBackend backend;
    PagePool pool(5);
    Table table; table.configure_bounded(128, &pool);
    table.ensure_logical_page_ranges_resident(backend, {{7, 1}, {2, 2}, {0, 1}});
    const uint64_t high = (uint64_t{1} << 34) + 31;
    std::vector<KvMemStorageSpan> spans = {
        {high, 127, 1, {3, 7}}, {high + 1, 32, 32, {1, 4}}, {high + 33, 0, 5, {0, 9}}};
    const auto aliases = table.batch_storage_indices(spans);
    check(aliases == std::vector<int32_t>({table.pages[7], table.pages[2], table.pages[3], table.pages[0]}),
          "MTP batch-local aliases were sorted by physical address");
    for (uint32_t i = 0; i < 38; ++i) {
        const auto batch_page = aliases[(15 + i) / 16];
        const uint32_t storage_token = i == 0 ? 127 : i <= 32 ? 31 + i : i - 33;
        check(batch_page == table.pages[storage_token / 16] &&
              (15 + i) % 16 == storage_token % 16, "partial-page offset changed during batch mapping");
    }
    table.reset();
}

static void accepted_prefix_releases_only_rejected_pages() {
    RecordingBackend backend;
    PagePool main_pool(6), mtp_pool(6);
    Table main, mtp; main.configure_bounded(96, &main_pool); mtp.configure_bounded(96, &mtp_pool);
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{3, 32});
    KvMemStoreConfig cfg; cfg.block_tokens = 32;
    KvMemStore store(cfg, repo);
    auto a = store.prepare_repository_append(69, 0);
    auto io = a->storage_write(64, 5);
    main.ensure_logical_page_ranges_resident(backend, {{0, 5}});
    mtp.ensure_logical_page_ranges_resident(backend, {{0, 5}});
    const auto rejected_handle = a->spans()[2].handle;
    const auto discarded = a->discarded_page_ranges(33, 16);
    main.release_logical_page_ranges(backend, discarded);
    mtp.release_logical_page_ranges(backend, discarded);
    check(main_pool.used_pages() == 3 && mtp_pool.used_pages() == 3,
          "partially accepted tail page was freed or rejected pages leaked");
    a->commit(33);
    bool callback = false;
    auto cleanup = [&](const auto &) {
        callback = true;
        check(main.pages[4] == -1 && mtp.pages[4] == -1, "CPU cleanup raced old GPU aliases");
        return true;
    };
    check(repo->try_reclaim(rejected_handle, cleanup) == KvMemRepository::ReclaimResult::DeferredReaders &&
          !callback, "CPU cleanup ran before rejected IO completed");
    io = {};
    check(repo->try_reclaim(rejected_handle, cleanup) == KvMemRepository::ReclaimResult::Reclaimed &&
          callback, "rejected bundle was not returned after IO");
    a = store.prepare_repository_append(32, 33);
    check(a->spans().back().handle.slot == rejected_handle.slot &&
          a->spans().back().handle.generation != rejected_handle.generation,
          "rejected physical slot did not enter the ready queue");
    // Speculative bytes in the preserved final page never expanded valid KV.
    check(store.history_end() == 33 && store.blocks()[1].n_tokens == 1,
          "rejected bytes were committed with the accepted tail page");
    a.reset();
    main.reset(); mtp.reset();
}

static void recycled_history_to_main_and_mtp_pages() {
    RecordingBackend backend;
    PagePool main_pool(4), mtp_pool(6);
    Table main, mtp; main.configure_bounded(128, &main_pool); mtp.configure_bounded(128, &mtp_pool);
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    KvMemStoreConfig cfg; cfg.block_tokens = 32;
    KvMemStore store(cfg, repo);
    std::vector<uint64_t> main_bytes(4 * 16), mtp_bytes(6 * 16);
    const uint64_t history_base = uint64_t{1} << 34;
    auto initial = repo->try_reserve(history_base - 32, 32).commit();
    const auto seed = initial.handle();
    store.register_repository_block(std::move(initial), 0);
    store.set_block_tier(seed.slot, KvTier::CPU, 0, -1, seed.generation);
    check(repo->retire_if_unpinned(seed), "initial block still pinned");
    repo->try_reclaim(seed, [](const auto &) { return true; });
    for (uint32_t step = 0; step < 1024; ++step) {
        auto a = store.prepare_repository_append(32, 0);
        check(bool(a), "test repository ran out of recycled slots");
        const auto view = a->storage_write(a->begin(), 32);
        const auto aliases = kvmem_storage_page_spans(view.spans, 16, 0);
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (const auto &p : aliases) ranges.emplace_back(p.storage_page, p.page_count);
        main.ensure_logical_page_ranges_resident(backend, ranges);
        mtp.ensure_logical_page_ranges_resident(backend, ranges);
        const auto main_window = main.batch_storage_indices(view.spans);
        const auto mtp_window = mtp.batch_storage_indices(view.spans);
        for (uint32_t i = 0; i < 32; ++i) {
            main_bytes[main_window[i / 16] * 16 + i % 16] = a->begin() + i;
            mtp_bytes[mtp_window[i / 16] * 16 + i % 16] = a->begin() + i + 17;
        }
        const auto h = view.spans[0].handle;
        a->commit(32);
        for (uint32_t i = 0; i < 32; ++i) {
            check(main_bytes[main_window[i / 16] * 16 + i % 16] == history_base + step * 32 + i, "main alias mismatch");
            check(mtp_bytes[mtp_window[i / 16] * 16 + i % 16] == history_base + step * 32 + i + 17, "MTP alias mismatch");
        }
        // GPU pages are released by the foreground tier transition. The CPU
        // reclaimer subsequently returns repository capacity without any GPU
        // allocation, upload, or synchronization in its callback.
        main.release_logical_page_ranges(backend, ranges);
        mtp.release_logical_page_ranges(backend, ranges);
        bool rejected = false;
        try { mtp.batch_storage_indices(view.spans); }
        catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "MTP alias used a page that was already staged out");
        store.set_block_tier(h.slot, KvTier::CPU, h.slot, -1, h.generation);
        check(repo->retire(h), "completed page bundle could not retire");
        // view keeps the backing pinned until this iteration ends; reclaim it
        // at the next iteration's scan rather than overriding that reference.
        for (uint32_t slot = 0; slot < 4; ++slot) {
            const auto retired = repo->retired_at(slot);
            if (retired.generation) repo->try_reclaim(retired, [](const auto &) { return true; });
        }
        check(main.pages.size() == 8 && mtp.pages.size() == 8, "GPU page metadata grew with history");
    }
    main.reset(); mtp.reset();
}

// Only constructs the production executor's storage owners. No forward,
// attention, recurrent or MTP numerical kernels run against these dummy weights.
struct StorageModelSource final : ModelSource {
    std::string directory;
    ModelSourceInfo description{"storage-test", "qwen35", 1, 4};
    mutable std::map<std::string, ModelTensorInfo> tensors;
    float value = 0;
    explicit StorageModelSource(std::string dir) : directory(std::move(dir)) {}
    const ModelTensorInfo *find_tensor(const std::string &name) const override {
        auto &t = tensors[name]; t.name = name; t.source_name = name;
        t.type = ModelTensorType::F32; t.dims = {1}; t.data = &value; t.bytes = sizeof(value);
        return &t;
    }
    const ModelSourceInfo &info() const override { return description; }
    const std::string &model_directory() const override { return directory; }
};

void qw3::KvPageTableTestAccess::raw_storage(const QwenNativeModel &model,
        const QwenWeights &weights, DeviceBackend &backend, const std::string &directory) {
    constexpr uint32_t P = 4, B = 32, C = 64;
    QwenExecutor e(model, weights, backend, P * B);
    KvMemStoreConfig cfg; cfg.block_tokens = B; cfg.sink_blocks = 0;
    cfg.repository_blocks = P; cfg.mtp_enabled = true;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{P, B});
    e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
    e.kvmem_enabled_ = true;
    e.kvmem_repository_storage_generation_.resize(P);
    e.kvmem_repository_index_.resize(P);
    e.kv_pages_.configure_bounded(P * B, nullptr);
    e.mtp_kv_pages_.configure_bounded(P * B, nullptr);
    e.kvmem_raw_k_nvme_enabled_ = true;
    e.kvmem_raw_k_nvme_block_slots_ = true;
    e.kvmem_raw_k_block_major_ = true;
    e.kvmem_raw_k_row_bytes_ = 1;
    e.kvmem_raw_layers_ = {0};
    e.kvmem_raw_k_chunk_tokens_ = C;
    e.kvmem_raw_k_chunk_bytes_ = e.kvmem_raw_mtp_k_chunk_bytes_ = C;
    e.kvmem_raw_k_nvme_main_block_bytes_ = e.kvmem_raw_k_nvme_mtp_block_bytes_ = B;
    e.kvmem_raw_k_nvme_slot_bytes_ = 2 * B;
    e.kvmem_raw_k_nvme_block_backed_.resize(P);
    e.kvmem_raw_mtp_k_nvme_block_backed_.resize(P);
    e.kvmem_raw_k_nvme_backed_.resize(2); e.kvmem_raw_mtp_k_nvme_backed_.resize(2);
    e.kvmem_raw_k_dirty_.resize(2); e.kvmem_raw_mtp_k_dirty_.resize(2);
    e.kvmem_raw_k_write_pending_.resize(2);
    e.kvmem_raw_k_valid_tokens_.resize(P * B); e.kvmem_raw_mtp_k_valid_tokens_.resize(P * B);
    e.kvmem_raw_k_chunks_.resize(2); e.kvmem_raw_mtp_k_chunks_.resize(2);
    for (uint32_t i = 0; i < 2; ++i) {
        e.kvmem_raw_k_chunks_[i] = std::make_unique<uint8_t[]>(C);
        e.kvmem_raw_mtp_k_chunks_[i] = std::make_unique<uint8_t[]>(C);
    }
    e.kvmem_raw_writeback_record_bytes_ = 2 * C;
    e.kvmem_raw_writeback_chunk_slot_.assign(2, -1);
    e.kvmem_raw_writeback_chunk_offset_.resize(2);
    e.kvmem_raw_writeback_main_ready_.resize(2); e.kvmem_raw_writeback_mtp_ready_.resize(2);
    for (auto &slot : e.kvmem_raw_writeback_slots_) {
        slot.capacity_bytes = 4 * C;
        slot.host = backend.host_buffer(slot.capacity_bytes, "raw-storage-test");
    }
    NvmeKvTierConfig nc; nc.dir = directory; nc.slot_bytes = 2 * B; nc.total_bytes = P * nc.slot_bytes;
    e.kvmem_raw_k_nvme_tier_ = std::make_unique<NvmeKvTier>(nc);
    auto release = kvmem_repository_storage_release(P, e.kvmem_raw_k_nvme_tier_.get(), nullptr);
    // A cold, unchanged neighbor shares the physical chunk but must not be
    // rewritten when only one block changes.
    e.kvmem_prepare_repository_range(0, 2 * B, 0);
    const auto initial = e.kvmem_repository_append_->spans();
    std::array<uint8_t, B> original{}; original.fill(73);
    for (const auto &span : initial) {
        e.kvmem_write_raw_k(0, span.storage_begin, original.data(), B, false);
        e.kvmem_write_raw_k(0, span.storage_begin, original.data(), B, true);
        std::fill_n(e.kvmem_raw_k_valid_tokens_.begin() + span.storage_begin, B, 1);
        std::fill_n(e.kvmem_raw_mtp_k_valid_tokens_.begin() + span.storage_begin, B, 1);
    }
    check(e.kvmem_copy_raw_k_chunk_to_writeback(0), "initial raw chunk not persisted");
    e.kvmem_register_until(2 * B); e.kvmem_flush_raw_k_writes(true);
    const auto before_write = e.kvmem_raw_k_nvme_write_bytes_;
    std::array<uint8_t, B> changed{}; changed.fill(119);
    e.kvmem_write_raw_k(0, initial[0].storage_begin, changed.data(), B, false);
    e.kvmem_write_raw_k(0, initial[0].storage_begin, changed.data(), B, true);
    auto temporarily_missing = std::move(e.kvmem_raw_k_chunks_[0]);
    bool packing_failure = false;
    try { e.kvmem_copy_raw_k_chunk_to_writeback(0); }
    catch (const std::logic_error &) { packing_failure = true; }
    check(packing_failure && e.kvmem_raw_writeback_chunk_slot_[0] == -1,
          "failed CPU packing left a reserved writeback group");
    for (const auto &slot : e.kvmem_raw_writeback_slots_)
        check(slot.spans.empty() && !slot.used_bytes && slot.repository_capture.leases.empty(),
              "failed CPU packing left a partially publishable batch");
    e.kvmem_raw_k_chunks_[0] = std::move(temporarily_missing);
    check(e.kvmem_copy_raw_k_chunk_to_writeback(0), "changed block not persisted");
    e.kvmem_flush_raw_k_writes(true);
    check(e.kvmem_raw_k_nvme_write_bytes_ - before_write == 2 * B,
          "recycled raw writeback rewrote unchanged neighboring blocks");
    std::array<uint8_t, 2 * B> neighbor{};
    e.kvmem_raw_k_nvme_tier_->read_block(initial[1].handle.slot, neighbor.data(), neighbor.size());
    check(std::all_of(neighbor.begin(), neighbor.end(), [](uint8_t v) { return v == 73; }),
          "unchanged raw/MTP neighbor was corrupted");
    std::array<uint8_t, B> actual{};
    e.kvmem_read_raw_k(0, initial[0].storage_begin, actual.data(), B, false);
    check(actual == changed, "bounded raw reader did not read the current generation");
    // Test the synchronous cache-eviction fallback as well as the packed path.
    e.kvmem_write_raw_k(0, initial[0].storage_begin, original.data(), B, false);
    const auto sync_before = e.kvmem_raw_k_nvme_write_bytes_;
    e.kvmem_persist_raw_k_chunk(0, false);
    check(e.kvmem_raw_k_nvme_write_bytes_ - sync_before == B,
          "synchronous raw spill rewrote a clean neighbor");
    for (const auto &span : initial) {
        e.block_store_->set_block_tier(span.handle.slot, KvTier::CPU, span.handle.slot, -1, span.handle.generation);
        check(repo->retire_if_unpinned(span.handle), "clean initial block could not retire");
        bool deleted_read_rejected = false;
        try { e.kvmem_read_raw_k(0, span.storage_begin, actual.data(), B, false); }
        catch (const std::runtime_error &) { deleted_read_rejected = true; }
        check(deleted_read_rejected, "raw reader exposed bytes from retired history");
        check(repo->try_reclaim(span.handle, release) == KvMemRepository::ReclaimResult::Reclaimed,
              "initial raw resource release failed");
    }
    for (uint32_t round = 0; round < 128; ++round) {
        const auto begin = e.block_store_->history_end();
        e.kvmem_prepare_repository_range(begin, B, 0);
        const auto span = e.kvmem_repository_append_->spans().front();
        const auto h = span.handle;
        std::array<uint8_t, B> k{}, m{}; k.fill(uint8_t(round)); m.fill(uint8_t(255 - round));
        e.kvmem_write_raw_k(0, span.storage_begin, k.data(), B, false);
        std::fill_n(e.kvmem_raw_k_valid_tokens_.begin() + span.storage_begin, B, 1);
        check(!e.kvmem_copy_raw_k_chunk_to_writeback(span.storage_begin / C), "MTP incomplete raw bundle was submitted");
        e.kvmem_write_raw_k(0, span.storage_begin, m.data(), B, true);
        std::fill_n(e.kvmem_raw_mtp_k_valid_tokens_.begin() + span.storage_begin, B, 1);
        check(e.kvmem_copy_raw_k_chunk_to_writeback(span.storage_begin / C), "ready raw/MTP bundle was not batched");
        e.kvmem_register_until(static_cast<uint32_t>(begin + B));
        e.block_store_->set_block_tier(h.slot, KvTier::CPU, h.slot, -1, h.generation);
        check(repo->retire(h), "raw bundle could not retire");
        check(repo->try_reclaim(h, release) == KvMemRepository::ReclaimResult::DeferredReaders,
              "queued raw writeback failed to pin payload");
        // A different free slot in the same chunk can be claimed before reap.
        e.kvmem_prepare_repository_range(begin + B, B, 0);
        const auto next = e.kvmem_repository_append_->spans().front();
        const auto next_chunk = next.storage_begin / C, offset = next.storage_begin % C;
        std::memset(e.kvmem_raw_k_chunks_[next_chunk].get() + offset, 197, B);
        e.kvmem_flush_raw_k_writes(true);
        check(e.kvmem_raw_k_nvme_tier_->used_slots() == 1, "raw writeback resurrected a physical hole");
        std::array<uint8_t, 2 * B> bytes{};
        e.kvmem_raw_k_nvme_tier_->read_block(h.slot, bytes.data(), bytes.size());
        check(std::equal(k.begin(), k.end(), bytes.begin()) && std::equal(m.begin(), m.end(), bytes.begin() + B),
              "production raw/MTP batch persisted incorrect bytes");
        check(std::all_of(e.kvmem_raw_k_chunks_[next_chunk].get() + offset,
                         e.kvmem_raw_k_chunks_[next_chunk].get() + offset + B,
                         [](uint8_t b) { return b == 197; }), "reap overwrote newly reserved raw chunk hole");
        e.kvmem_repository_append_.reset();
        check(repo->try_reclaim(next.handle, release) == KvMemRepository::ReclaimResult::Reclaimed,
              "unused reservation was not released");
        check(repo->try_reclaim(h, release) == KvMemRepository::ReclaimResult::Reclaimed,
              "completed raw writeback still held retired payload");
        check(repo->ready_slots() == P && e.kvmem_raw_k_nvme_tier_->used_slots() == 0,
              "raw resource capacity leaked across slot reuse");
    }
    // Exercise the production physical rewind past the finite storage length.
    const uint64_t begin = e.block_store_->history_end();
    e.kvmem_gpu_page_pool_ = std::make_unique<PagePool>(P * B / 16);
    e.kvmem_mtp_gpu_page_pool_ = std::make_unique<PagePool>(P * B / 16);
    e.kv_pages_.configure_bounded(P * B, e.kvmem_gpu_page_pool_.get());
    e.mtp_kv_pages_.configure_bounded(P * B, e.kvmem_mtp_gpu_page_pool_.get());
    e.kvmem_prepare_repository_range(begin, 70, 0);
    e.kvmem_ensure_repository_pages(e.kv_pages_, begin, 70);
    e.kvmem_ensure_repository_pages(e.mtp_kv_pages_, begin, 70);
    const auto written = e.kvmem_repository_append_->spans();
    for (const auto &span : written) {
        e.kvmem_write_raw_k(0, span.storage_begin, original.data(), span.token_count, false);
        e.kvmem_write_raw_k(0, span.storage_begin, original.data(), span.token_count, true);
        std::fill_n(e.kvmem_raw_k_valid_tokens_.begin() + span.storage_begin, span.token_count, 1);
        std::fill_n(e.kvmem_raw_mtp_k_valid_tokens_.begin() + span.storage_begin, span.token_count, 1);
        e.kvmem_repository_index_[span.handle.slot] = {span.handle, span.token_count};
    }
    e.kvmem_register_until(static_cast<uint32_t>(begin + 70));
    auto preserved = e.block_store_->repository_checkpoint(begin + 37);
    auto invalidated = e.block_store_->repository_checkpoint(begin + 70);
    check(e.kvmem_gpu_page_pool_->used_pages() == 5, "unexpected main page count before rewind");
    e.kvmem_truncate_to(static_cast<uint32_t>(begin + 37));
    check(e.kvmem_gpu_page_pool_->used_pages() == 3 && e.kvmem_mtp_gpu_page_pool_->used_pages() == 3,
          "bounded rewind failed to keep the partial page and release rejected main/MTP pages");
    check(preserved->valid() && !invalidated->valid(), "executor rewind did not invalidate the discarded checkpoint");
    e.kvmem_read_raw_k(0, written[1].storage_begin, actual.data(), 5, false);
    bool beyond_tail_rejected = false;
    try { e.kvmem_read_raw_k(0, written[1].storage_begin, actual.data(), 6, false); }
    catch (const std::runtime_error &) { beyond_tail_rejected = true; }
    check(beyond_tail_rejected, "rewind left discarded raw tokens readable");
    check(!e.kvmem_repository_index_[written[1].handle.slot].handle.generation,
          "rewind retained a stale partial-tail mean-K index");
    check(repo->try_reclaim(written[2].handle, release) == KvMemRepository::ReclaimResult::Reclaimed,
          "discarded checkpoint still pins suffix resources");
    e.kvmem_registered_pos_ = static_cast<uint32_t>(begin + 37);
    e.kvmem_prepare_repository_range(begin + 37, 20, 37);
    e.kvmem_ensure_repository_pages(e.kv_pages_, begin + 37, 20);
    e.kvmem_ensure_repository_pages(e.mtp_kv_pages_, begin + 37, 20);
    check(e.kvmem_gpu_page_pool_->used_pages() == 4, "new branch did not reuse the partial prefix");
    e.kvmem_truncate_to(static_cast<uint32_t>(begin + 37)); // cancel unpublished speculation
    check(e.kvmem_gpu_page_pool_->used_pages() == 3 && !e.kvmem_repository_append_,
          "speculative cancellation leaked repository pages");
    check(dynamic_cast<RecordingBackend &>(backend).transfer_fences == 0,
          "CPU-packed raw persistence unnecessarily touched a GPU transfer fence");
}

void qw3::KvPageTableTestAccess::mtp_recent_window(const QwenNativeModel &model,
        const QwenWeights &weights, DeviceBackend &backend) {
    QwenExecutor e(model, weights, backend, 128);
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.repository_blocks = 4;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
    e.kvmem_enabled_ = true;
    e.kvmem_repository_storage_generation_.resize(4); e.kvmem_repository_index_.resize(4);
    e.kv_pages_.configure_bounded(128, nullptr); e.mtp_kv_pages_.configure_bounded(128, nullptr);
    e.ensure_mtp_kv_pages(0, 64);
    e.kvmem_register_until(64);
    e.position_ = e.window_query_pos_ = 64; e.kvmem_active_ = true;
    // Only the previously assembled first 32 tokens have window aliases;
    // the second 32 are primed KV produced by a subsequent target batch.
    e.mtp_window_pages_host_.assign(e.mtp_kv_pages_.pages.begin(), e.mtp_kv_pages_.pages.begin() + 2);
    e.mtp_window_page_count_ = 2;
    e.ensure_mtp_kv_pages(64, 1);
    e.kvmem_extend_mtp_window_for_decode_n(1, 64);
    check(e.mtp_window_page_count_ == 5 &&
          std::equal(e.mtp_window_pages_host_.begin(), e.mtp_window_pages_host_.end(), e.mtp_kv_pages_.pages.begin()),
          "MTP draft did not fill its already-primed recent window tail");
    e.kvmem_extend_mtp_window_for_decode_n(1, 64);
    check(e.mtp_window_page_count_ == 5, "repeated MTP draft extension duplicated its tail");
}

void qw3::KvPageTableTestAccess::historical_pins(const QwenNativeModel &model,
        const QwenWeights &weights, DeviceBackend &backend) {
    for (const uint64_t origin : {4096ull, 1ull << 40}) {
    QwenExecutor e(model, weights, backend, 128);
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.repository_blocks = 4;
    cfg.sink_blocks = 0; cfg.select_budget = 64;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
    std::vector<KvMemBlockHandle> old;
    for (uint32_t i = 0; i < 4; ++i) {
        auto lease = repo->try_reserve(origin + i * 32, 32).commit();
        old.push_back(lease.handle());
        e.block_store_->register_repository_block(std::move(lease), 0);
        e.block_store_->set_block_tier(old.back().slot, KvTier::CPU, 0, -1, old.back().generation);
    }
    check(repo->retire_if_unpinned(old[0]), "test old pin block was still held");
    repo->try_reclaim(old[0], [](const auto &) { return true; });
    auto append = e.block_store_->prepare_repository_append(32, 0);
    const auto newest = append->spans().front().handle;
    append->commit(32);
    check(newest.slot == old[0].slot, "test did not reuse a low physical slot for new history");
    e.kvmem_set_pin_from_block((origin + 96) / 32);
    check(e.kvmem_materialize_mandatory_blocks() == std::vector<uint32_t>({0, 3}),
          "query suffix pin used physical IDs instead of history coordinates");
    check(e.kvmem_set_bounded_recent_pin(64) == 64 && e.kvmem_qc_pin_from_block_ == (origin + 96) / 32,
          "recent tail was not mapped back from chronological ordinals");
    e.kvmem_set_mandatory_token_spans({{origin + 33, origin + 34}});
    check(!repo->retire_if_unpinned(old[1]), "CPU mandatory block was not protected across requests");
    check(e.kvmem_set_bounded_recent_pin(64) == 32 &&
          e.kvmem_materialize_mandatory_blocks() == std::vector<uint32_t>({0, 1}),
          "recent tail failed to account for an older mandatory historical span");
    e.kvmem_set_oracle_token_spans({{origin, origin + 1}});
    check(e.kvmem_materialize_mandatory_blocks() == std::vector<uint32_t>({0, 1}),
          "deleted oracle evidence aliased a new block in the same slot");
    bool rejected = false;
    try {
        e.kvmem_set_mandatory_token_spans({{origin, origin + 1}});
        e.kvmem_materialize_mandatory_blocks();
    } catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "deleted required history was silently accepted");
    check(e.kvmem_materialize_mandatory_blocks() == std::vector<uint32_t>({0, 1}),
          "failed mandatory update destroyed the previously valid request pins");
    e.kvmem_set_mandatory_token_spans({});
    check(repo->retire_if_unpinned(old[1]), "cleared mandatory span leaked a repository lease");    bool overflow_rejected = false;
    try { e.kvmem_set_pin_from_block(UINT64_MAX - 1); }
    catch (const std::overflow_error &) { overflow_rejected = true; }
    check(overflow_rejected, "overflowed historical tail pin was accepted");
    }
}

void qw3::KvPageTableTestAccess::bounded_mrope(const QwenNativeModel &model,
        const QwenWeights &weights, DeviceBackend &backend) {
    constexpr uint32_t P = 4, B = 32;
    QwenExecutor e(model, weights, backend, P * B);
    KvMemStoreConfig cfg; cfg.block_tokens = B; cfg.repository_blocks = P; cfg.sink_blocks = 0;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{P, B});
    e.block_store_ = std::make_unique<KvMemStore>(cfg, repo);
    e.kvmem_enabled_ = e.kvmem_active_ = true;
    e.kvmem_repository_storage_generation_.resize(P); e.kvmem_repository_index_.resize(P);
    e.kv_pages_.configure_bounded(P * B, nullptr); e.mtp_kv_pages_.configure_bounded(P * B, nullptr);
    GenerationOptions::InputEmbeddingOverride image;
    image.token_id = 0x80000001u; image.source_token_id = 1; image.embedding.assign(4, 0.25f);
    std::vector<uint32_t> prompt;
    std::vector<std::array<uint32_t, 3>> source;
    for (uint32_t round = 0; round < 32; ++round) {
        const uint32_t begin = round * 2 * B, end = begin + 2 * B;
        prompt.resize(end, 1); source.resize(end);
        for (uint32_t i = begin; i < end; ++i) source[i] = {2*i, 2*i + i%4, 2*i + i%2};
        prompt[begin + 8] = image.token_id;
        e.set_input_embedding_overrides({image}, prompt, source);
        check(e.input_compact_mrope_positions_.size() == P * B, "compact coordinates grew with history");
        e.window_mrope_next_ = 0; e.window_mrope_append_active_ = false;
        e.kvmem_prepare_repository_range(begin, 2 * B, 0);
        const auto spans = e.kvmem_repository_append_->spans();
        for (const auto &span : spans) for (uint32_t j = 0; j < B; ++j)
            check(!e.input_compact_mrope_valid_[span.storage_begin + j], "reused generation retained visual coordinates");
        check(e.prepare_input_mrope_batch(prompt.data() + begin, 2 * B, begin), "visual batch not prepared");
        for (const auto &span : spans) for (uint32_t j = 0; j < B; ++j) {
            const auto pos = span.history_begin + j;
            for (uint32_t axis = 0; axis < 3; ++axis)
                check(e.input_compact_mrope_positions_[span.storage_begin+j][axis] == source[pos][axis] - 2*begin,
                      "visual append used history as physical address");
        }
        e.position_ = end; e.kvmem_register_until(end);
        KvMemPlan plan; plan.total_window_tokens = 2 * B;
        for (uint32_t j = 0; j < spans.size(); ++j)
            plan.remaps.push_back({spans[j].handle.slot, B, 0, int32_t(j*B)});
        e.build_kvmem_window_mrope_positions(plan);
        for (uint32_t i = 0; i < 2 * B; ++i) for (uint32_t axis = 0; axis < 3; ++axis)
            check(e.window_mrope_positions_host_[axis*2*B+i] == int32_t(source[begin+i][axis] - 2*begin),
                  "visual assembly used a recycled slot as historical coordinate");
        e.kvmem_truncate_repository(end - 16);
        const auto tail = spans.back().storage_begin;
        check(e.input_compact_mrope_valid_[tail+15] && !e.input_compact_mrope_valid_[tail+16],
              "visual rewind did not preserve prefix and invalidate rejected suffix");
        e.position_ = end - 16; e.kvmem_registered_pos_ = end - 16;
        e.kvmem_prepare_repository_range(end - 16, 16, 48);
        e.position_ = end; e.kvmem_register_until(end);
        for (const auto &span : spans) {
            e.block_store_->set_block_tier(span.handle.slot, KvTier::CPU, -1, -1, span.handle.generation);
            check(repo->retire_if_unpinned(span.handle), "visual test leaked a storage lease");
            check(repo->try_reclaim(span.handle, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed,
                  "visual test could not recycle payload");
        }
        e.release_bounded_input_metadata();
        check(e.input_mrope_positions_.capacity() == 0 && e.input_multimodal_prompt_tokens_.capacity() == 0 &&
              e.input_embedding_overrides_.capacity() == 0, "bounded executor retained request-sized visual input");
    }
}

void KvPageTableTestAccess::bounded_config_guards(
        const QwenNativeModel &model, const QwenWeights &weights, DeviceBackend &backend) {
    QwenExecutor e(model, weights, backend, 128);
    KvMemStoreConfig base;
    base.block_tokens = 32; base.repository_blocks = 4;
    base.immutable_source_k = true; base.index_placement = KvMemIndexPlacement::CPU;
    for (uint32_t invalid = 0; invalid < 6; ++invalid) {
        auto cfg = base;
        if (invalid == 0) cfg.n_subblocks = 0;
        if (invalid == 1) cfg.n_subblocks = 3;
        if (invalid == 2) cfg.adaptive_score_mode = KvMemAdaptiveScoreMode::TiledTwoPass;
        if (invalid == 3) cfg.adaptive_score_mode = KvMemAdaptiveScoreMode::LayerOnePass;
        if (invalid == 4) cfg.prototype_mode = KvMemPrototypeMode::KeyDirectionAdaptive; // width must be 4
        if (invalid == 5) { cfg.n_subblocks = 4; cfg.subblock_reduce = KvMemSubblockReduce::Sum; }
        bool rejected = false;
        try { e.configure_kvmem(cfg); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected && !e.block_store_, "unsafe bounded index layout reached allocation or scoring");
    }
}

static void executor_raw_storage() {
    const auto dir = std::filesystem::temp_directory_path() / ("qw3-executor-storage-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "config.json") << R"({"text_config":{"model_type":"qwen3_5_text",
      "num_hidden_layers":1,"hidden_size":4,"intermediate_size":4,"max_position_embeddings":128,
      "num_attention_heads":1,"num_key_value_heads":1,"head_dim":4,"linear_conv_kernel_dim":4,
      "linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":4,"linear_value_head_dim":4,
      "full_attention_interval":1,"vocab_size":4}})";
    try {
        RecordingBackend backend;
        QwenNativeModel model(std::make_unique<StorageModelSource>(dir.string()));
        QwenWeights weights(model, backend, false, false);
        KvPageTableTestAccess::raw_storage(model, weights, backend, dir.string());
        KvPageTableTestAccess::mtp_recent_window(model, weights, backend);
        KvPageTableTestAccess::historical_pins(model, weights, backend);
        KvPageTableTestAccess::bounded_mrope(model, weights, backend);
        KvPageTableTestAccess::bounded_config_guards(model, weights, backend);
    } catch (...) { std::filesystem::remove_all(dir); throw; }
    std::filesystem::remove_all(dir);
}

static void private_query_pool_reserve() {
    RecordingBackend backend;
    for (const char *label : {"main", "mtp"}) {
        LocalKvPagePool pool(4, 16, label, 3);
        Table table;
        table.configure_bounded(128, &pool);
        table.ensure_logical_page_ranges_resident(backend, {{0, 4}});
        const auto committed = table.pages;
        check(pool.free_pages() == 0 && pool.total_pages() == 7,
              "ordinary inference consumed private-query headroom");
        bool rejected = false;
        try { pool.allocate_physical_page(); } catch (const std::runtime_error &) { rejected = true; }
        check(rejected, "ordinary allocation bypassed the resident limit");
        for (int iteration = 0; iteration < 100; ++iteration) {
            pool.set_transient_allocation(true);
            table.ensure_logical_page_ranges_resident(backend, {{4, 3}});
            check(pool.used_pages() == 7 && pool.free_pages() == 0, "probe did not borrow scratch");
            rejected = false;
            try { pool.allocate_physical_page(); } catch (const std::runtime_error &) { rejected = true; }
            check(rejected, "private allocation exceeded bounded scratch");
            // Both success and exception paths disable borrowing before rollback.
            pool.set_transient_allocation(false);
            table.release_logical_page_ranges(backend, {{4, 3}});
            check(table.pages == committed && pool.used_pages() == 4 && pool.free_pages() == 0,
                  "private rollback leaked pages or changed the committed working set");
        }
        table.reset();
        check(pool.free_pages() == 4, "reset exposed scratch to ordinary inference");
    }
}

static void shared_request_pool_isolation() {
    RecordingBackend backend;
    PagePool parent(32); // Allocates high IDs first, above each request's quota.
    {
        SharedKvPagePool a(parent, 4, 3, "request-a"), b(parent, 4, 3, "request-b");
        check(parent.used_pages() == 8, "resident quotas were not reserved at admission");
        Table at, bt;
        at.configure(128, &a); bt.configure_bounded(128, &b);
        at.ensure_pages(backend, 128, 0, 64);
        bt.ensure_logical_page_ranges_resident(backend, {{0, 4}});
        const auto a_prefix = at.pages, b_prefix = bt.pages;
        check(at.pages.front() > int32_t(a.total_pages()) &&
              a.physical_page_capacity() == 32, "high global page IDs were not exercised");
        for (auto p : at.pages)
            check(std::find(bt.pages.begin(), bt.pages.end(), p) == bt.pages.end(), "request pages alias");
        for (unsigned i = 0; i < 100; ++i) {
            bool exhausted = false;
            try { a.allocate_physical_page(); } catch (const std::runtime_error &) { exhausted = true; }
            check(exhausted && a.used_pages() == 4 && b.used_pages() == 4,
                  "ordinary inference borrowed another request or private quota");
            a.set_transient_allocation(true);
            at.ensure_pages(backend, 128, 64, 48);
            check(a.used_pages() == 7 && parent.used_pages() == 11 && b.free_pages() == 0,
                  "private borrow affected sibling quota");
            a.set_transient_allocation(false);
            at.truncate_to_logical_pages(4);
            check(at.pages == a_prefix && bt.pages == b_prefix && parent.used_pages() == 8,
                  "private rollback leaked or changed committed pages");
            // Invalid/foreign/double frees must not return any sibling pages.
            a.release_physical_pages({-1, 1000, bt.pages[0]});
            check(a.used_pages() == 4 && b.used_pages() == 4, "foreign release corrupted a lease");
        }
        at.reset(); bt.reset();
        check(a.free_pages() == 4 && b.free_pages() == 4 && parent.used_pages() == 8,
              "reset failed to retain the bounded resident lease");
    }
    check(parent.used_pages() == 0, "request destruction leaked its resident lease");
    // Failed admission must return the partially acquired resident quota.
    {
        SharedKvPagePool live(parent, 30, 0, "live");
        bool failed = false;
        try { SharedKvPagePool rejected(parent, 3, 0, "rejected"); }
        catch (const std::runtime_error &) { failed = true; }
        check(failed && parent.used_pages() == 30 && parent.free_pages() == 2,
              "failed admission leaked pages or disturbed a live request");
    }
    check(parent.used_pages() == 0, "admission failure cleanup leaked pages");
    // The parent cache keeps promoted pages pinned across request teardown.
    GlobalKvPagePool cached(8, 16);
    int32_t promoted = -1;
    {
        SharedKvPagePool request(cached, 4, 2, "cached");
        promoted = request.allocate_physical_page();
        cached.pin_pages({promoted});
    }
    check(cached.used_pages() == 1 && cached.pinned_pages() == 1,
          "request destruction freed a live prefix page");
    cached.unpin_pages({promoted}); cached.release_physical_pages({promoted});
    check(cached.used_pages() == 0 && cached.free_pages() == 8, "prefix eviction leaked its final page");
}

int main() {
    shared_request_pool_isolation();
    private_query_pool_reserve();
    try {
        setenv("QW3_PAGED_KV_PAGE_SIZE", "16", 1);
        executor_raw_storage();
        sparse_allocation_and_legacy_control();
        partial_batch_recycled_aliases();
        accepted_prefix_releases_only_rejected_pages();
        recycled_history_to_main_and_mtp_pages();
        std::cout << "OK\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
