#include "qw3/kvmem_gc.hpp"
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include <algorithm>
#include <future>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace qw3;
static void check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
template<class F> static void eventually(F f, const char *why) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!f()) {
        if (std::chrono::steady_clock::now() >= end) throw std::runtime_error(why);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}
static KvMemGcConfig config() {
    KvMemGcConfig c; c.max_live_blocks = 2; c.scan_blocks = 2;
    c.batch_blocks = 1; c.poll_ms = 1; c.grace_steps = 1; c.candidate_steps = 1;
    return c;
}
static void populate(KvMemGc &gc) {
    for (uint32_t id = 0; id < 3; ++id) {
        gc.register_block(id, true);
        gc.update_flags(id, KvMemGc::Gpu | KvMemGc::RawBacked | KvMemGc::ValueBacked,
                        KvMemGc::RawBacked | KvMemGc::ValueBacked);
    }
    gc.update_flags(0, KvMemGc::Anchor, KvMemGc::Anchor);
    gc.update_flags(1, KvMemGc::Anchor, KvMemGc::Anchor);
    gc.update_flags(2, KvMemGc::Duplicate, KvMemGc::Duplicate);
    for (int i = 0; i < 10; ++i) gc.begin_step();
    eventually([&] { return gc.stats().candidates > 0; }, "candidate not observed");
    gc.begin_step();
}
static void policy() {
    KvMemGcConfig c;
    KvMemGcFeatures f; f.step = 100;
    auto d = kvmem_gc_score(f, c);
    check(d.level == 2 && (d.reasons & GcUnknownScore), "missing score must be ordinary");
    f.low_streak = 100;
    check(kvmem_gc_score(f, c).level == 2, "missing is not low");
    f.score_valid = true;
    check(kvmem_gc_score(f, c).level == 1, "valid low evidence ignored");
    f.duplicate = true;
    check(kvmem_gc_score(f, c).level == 0, "duplicate priority");
    f.last_use = 99;
    check(kvmem_gc_score(f, c).level == 3, "recent duplicate must be protected");
    f.anchor = true;
    check(kvmem_gc_score(f, c).level == 4, "anchor must dominate");
    f.anchor = f.duplicate = f.score_valid = false; f.last_use = 0; f.frequency = 0;
    const double cold = kvmem_gc_score(f, c).score;
    f.frequency = 1;
    check(kvmem_gc_score(f, c).score > cold, "heat must increase retention");
}
static void reader_and_rescue() {
    std::atomic<unsigned> releases{0};
    KvMemGc gc(config(), 8, [&](uint32_t) { return [&] { ++releases; return true; }; });
    auto reader = gc.read_view();
    populate(gc);
    eventually([&] { return !gc.is_live(2, gc.read_view()); }, "victim not retired");
    auto new_reader = gc.read_view();
    check(!gc.is_live(2, new_reader), "new directory must hide victim");
    check(gc.is_live(2, reader), "old directory must remain readable");
    check(releases == 0, "payload freed under old reader");
    auto pin = gc.pin({2}); // legal transfer from the old reader
    reader.reset();
    eventually([&] { return gc.stats().cancelled > 0; }, "late selection not rescued");
    check(releases == 0 && gc.is_live(2, new_reader), "pinned victim lost");
    pin.reset();
}
static void deferred_release_and_reuse() {
    std::atomic<unsigned> attempts{0}, releases{0};
    std::atomic<bool> ready{false};
    const auto foreground = std::this_thread::get_id();
    std::atomic<bool> wrong_thread{false};
    KvMemGc gc(config(), 8, [&](uint32_t) { return [&] {
        if (std::this_thread::get_id() == foreground) wrong_thread = true;
        ++attempts;
        if (!ready) return false;
        ++releases; return true;
    }; });
    auto reader = gc.read_view();
    populate(gc);
    eventually([&] { return gc.stats().retired > 0; }, "retirement missing");
    check(attempts == 0, "release attempted before grace period");
    reader.reset();
    eventually([&] { return attempts > 0; }, "release not retried in background");
    ready = true;
    eventually([&] { return gc.stats().reclaimed == 1; }, "physical release missing");
    check(releases == 1 && !wrong_thread, "wrong release count/thread");
    auto stale = gc.read_view();
    check(!gc.is_live(2, stale), "dead block resurrected");
    const uint64_t old_generation = gc.generation(2);
    gc.forget_tail(2);
    gc.register_block(2, false);
    check(gc.generation(2) != old_generation && gc.is_live(2, stale), "rewind generation mismatch");
}
static void score_gaps_and_ties() {
    auto c = config(); c.max_live_blocks = 100;
    KvMemGc gc(c, 8, [](uint32_t) { return [] { return true; }; });
    for (uint32_t i = 0; i < 3; ++i) gc.register_block(i, true);
    auto event = std::make_shared<KvMemGc::ScoreEvent>();
    event->sequence = 3; event->step = gc.begin_step();
    event->scores = {1,1,1}; event->eligible = {1,1,1}; event->selected = {1,0,0};
    event->generations = {gc.generation(0),gc.generation(1),gc.generation(2)};
    gc.submit_scores(event);
    eventually([&] { return gc.stats().score_gaps == 1; }, "dropped score event not marked");
    check(gc.stats().retired == 0, "unpressured/tied blocks retired");
}

static void hard_protection_and_shadow() {
    auto c = config(); c.max_live_blocks = 1; c.shadow = true;
    KvMemGc gc(c, 8, [](uint32_t) -> std::function<bool()> {
        throw std::runtime_error("shadow invoked physical allocator");
    });
    for (uint32_t id = 0; id < 7; ++id) {
        gc.register_block(id, id != 0);
        gc.update_flags(id, KvMemGc::Gpu | KvMemGc::RawBacked | KvMemGc::ValueBacked,
                        KvMemGc::RawBacked | KvMemGc::ValueBacked);
    }
    gc.update_flags(1, KvMemGc::Io, KvMemGc::Io);
    gc.update_flags(2, KvMemGc::Gpu, KvMemGc::Gpu);
    gc.update_flags(3, KvMemGc::CpuCached, KvMemGc::CpuCached);
    gc.update_flags(4, KvMemGc::Mandatory, KvMemGc::Mandatory);
    gc.set_recent_blocks(1); // protects block 6; only block 5 may be nominated
    for (int i = 0; i < 20; ++i) gc.begin_step();
    eventually([&] { return gc.stats().candidates == 1; }, "hard protections not applied");
    gc.begin_step();
    eventually([&] { return gc.stats().shadow_candidates == 1; }, "shadow did not nominate cold block");
    const auto examined = gc.stats().examined;
    eventually([&] { return gc.stats().examined > examined + 14; }, "shadow sweep stalled");
    check(gc.stats().shadow_candidates == 1 && gc.stats().failures == 0,
          "shadow repeatedly nominated same block or called allocator");
    auto reader = gc.read_view();
    for (uint32_t id = 0; id < 7; ++id) check(gc.is_live(id, reader), "shadow deleted history");
    gc.begin_step(); gc.used(5);
    eventually([&] { return gc.stats().shadow_reuses == 1; }, "shadow reuse regret missing");
}

static void fresh_scores_cancel_pending() {
    KvMemGc gc(config(), 8, [](uint32_t) { return [] { return true; }; });
    auto reader = gc.read_view();
    populate(gc);
    eventually([&] { return gc.stats().retired == 1; }, "retirement missing");
    auto event = std::make_shared<KvMemGc::ScoreEvent>();
    event->sequence = 1; event->scorer_epoch = 1; event->step = gc.step();
    event->scores = {1,2,100}; event->eligible = {1,1,1}; event->selected = {0,0,0};
    event->generations = {gc.generation(0),gc.generation(1),gc.generation(2)};
    gc.submit_scores(event);
    eventually([&] { return gc.stats().cancelled == 1; }, "new high score did not cancel retirement");
    reader.reset();
    check(gc.stats().reclaimed == 0, "freshly important block was freed");
}

static void scores_rank_across_sweeps() {
    auto c = config(); c.max_live_blocks = 7; c.candidate_steps = 5;
    KvMemGc gc(c, 8, [](uint32_t) { return [] { return true; }; });
    auto reader = gc.read_view();
    for (uint32_t id = 0; id < 8; ++id) {
        gc.register_block(id, true);
        gc.update_flags(id, KvMemGc::Gpu | KvMemGc::RawBacked | KvMemGc::ValueBacked,
                        KvMemGc::RawBacked | KvMemGc::ValueBacked);
    }
    for (int i = 0; i < 100; ++i) gc.begin_step();
    eventually([&] { return gc.stats().examined >= 16; }, "full sweep not finished");
    for (uint64_t seq = 1; seq <= 3; ++seq) {
        auto event = std::make_shared<KvMemGc::ScoreEvent>();
        event->sequence = seq; event->scorer_epoch = 1; event->step = gc.begin_step();
        event->scores = {3,4,5,6,7,8,9,1};
        event->eligible.assign(8,1); event->selected = {0,0,0,0,0,0,1,0};
        for (uint32_t id = 0; id < 8; ++id) event->generations.push_back(gc.generation(id));
        const auto examined = gc.stats().examined;
        gc.submit_scores(event);
        eventually([&] { return gc.stats().examined >= examined + 16; }, "score sweep stalled");
    }
    for (int i = 0; i < 5; ++i) gc.begin_step();
    eventually([&] { return !gc.is_live(7, gc.read_view()); }, "cold score candidate not retired");
    const auto view = gc.read_view();
    check(!gc.is_live(7, view) && gc.is_live(0, view), "scan order beat sustained low-score priority");
}

static void nvme_store_integration() {
    NvmeKvTierConfig nc; nc.dir = "/tmp/qw3_gc_test_" + std::to_string(getpid());
    nc.total_bytes = 8 * 64; nc.slot_bytes = 64;
    nc.file_name = "raw.bin"; NvmeKvTier raw(nc);
    nc.file_name = "values.bin"; NvmeKvTier values(nc);
    KvMemStoreConfig sc; sc.block_tokens = 4; sc.select_budget = 16;
    sc.prefill_budget = 16; sc.sink_blocks = 1; sc.recent_blocks = 1;
    sc.optimize_stage_out = true; sc.gc = config(); sc.gc.max_live_blocks = 3;
    auto gc = std::make_shared<KvMemGc>(sc.gc, 8, [&](uint32_t id) -> std::function<bool()> {
        const auto k = raw.allocation(id), v = values.allocation(id);
        if (k.slot < 0 || v.slot < 0) return {};
        return [&, k, v] { return raw.try_release_allocation(k) && values.try_release_allocation(v); };
    });
    KvMemStore store(sc); store.attach_gc(gc); store.register_append(16);
    std::vector<uint8_t> payload(64, 0xa5), out(64);
    for (uint32_t id = 0; id < 4; ++id) {
        raw.write_block(id, payload.data(), payload.size()); values.write_block(id, payload.data(), payload.size());
        store.set_block_ssd_backing(id, values.block_slot(id), true);
        store.set_block_tier(id, KvTier::SSD, -1, values.block_slot(id));
        gc->update_flags(id, KvMemGc::RawBacked, KvMemGc::RawBacked);
    }
    auto plan = store.set_selection({0,2,3});
    auto checkpoint = store.gc_checkpoint_lease();
    // Simulated async read retains the same view as a real I/O batch.
    auto io_view = store.gc_view();
    for (int i = 0; i < 100; ++i) store.gc_begin_step();
    eventually([&] { return gc->stats().candidates > 0; }, "store produced no candidate");
    store.gc_begin_step();
    eventually([&] { return !gc->is_live(1, gc->read_view()); }, "store failed to retire");
    store.gc_begin_step();
    const auto selected = store.pick_topk_blocks(); // all-fit fast path must respect holes
    check(selected == std::vector<uint32_t>({0,2,3}), "all-fit selector resurrected hole");
    check(store.pick_prefill_pressure_blocks() == selected, "pressure selector resurrected hole");
    check(store.gc_missing_required_prefix_block(4, 8, 4) == UINT32_MAX,
          "a suffix to be recomputed must not invalidate a shorter checkpoint");
    check(store.gc_missing_required_prefix_block(4, 8, 8) == 1,
          "required deleted prefix must reject checkpoint reuse");
    check(store.gc_missing_required_prefix_block(8, 12, 16) == UINT32_MAX,
          "unrelated deleted history must not force cold prefill");
    check(raw.block_slot(1) >= 0 && values.block_slot(1) >= 0, "pending read lost payload");
    raw.read_block(1, out.data(), out.size()); check(out == payload, "pending read corruption");
    bool threw = false;
    try { store.set_selection({1}); } catch (const std::runtime_error &) { threw = true; }
    check(threw, "explicit deleted selection did not fail closed");
    store.gc_end_step(); // idle requests must not retain an obsolete directory
    io_view.reset();
    eventually([&] { return gc->stats().reclaimed == 1; }, "NVMe slots not reclaimed");
    check(raw.block_slot(1) < 0 && values.block_slot(1) < 0 && raw.free_slots() == 5,
          "K/V authority not physically released");
    store.gc_begin_step();
    auto next = store.set_selection({0,3});
    store.gc_restore_lease(checkpoint);
    check(store.is_live(2), "checkpoint lost preserved block");
    // Recycle the physical slot for the same logical ID and prove an older
    // allocation token cannot free the replacement (including after clear).
    raw.write_block(1, payload.data(), payload.size()); const auto old = raw.allocation(1);
    raw.release_block(1); raw.write_block(1, payload.data(), payload.size());
    check(raw.try_release_allocation(old) && raw.block_slot(1) >= 0, "allocator ABA after release");
    const auto prior_clear = raw.allocation(1); raw.clear();
    raw.write_block(1, payload.data(), payload.size());
    check(raw.try_release_allocation(prior_clear) && raw.block_slot(1) >= 0, "allocator ABA after reset");
    gc->stop();
    ::rmdir(nc.dir.c_str()); // payload files were unlinked on open
}

int main() {
    try { policy(); reader_and_rescue(); deferred_release_and_reuse(); score_gaps_and_ties();
          hard_protection_and_shadow(); fresh_scores_cancel_pending();
          scores_rank_across_sweeps(); nvme_store_integration(); }
    catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    std::cout << "KVMem GC policy and retirement tests passed\n";
}
