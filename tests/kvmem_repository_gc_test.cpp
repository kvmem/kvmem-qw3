#include "qw3/kvmem_repository_gc.hpp"
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace qw3;
using Mode = KvMemRepositoryGc::Mode;
constexpr uint32_t backed = KvMemGc::Sealed | KvMemGc::RawBacked |
                            KvMemGc::ValueBacked | KvMemRepositoryGc::Indexed;
static void check(bool value, const char *why) { if (!value) throw std::runtime_error(why); }
static KvMemRepositoryGc::Config config(uint32_t ready, Mode mode = Mode::Delete) {
    KvMemRepositoryGc::Config c;
    c.ready_low = c.ready_high = ready;
    c.mode = mode; c.background = false; c.tick_budget_us = 100000;
    c.policy.scan_blocks = 64; c.policy.batch_blocks = 64;
    c.policy.poll_ms = 1; c.policy.grace_steps = 2; c.policy.candidate_steps = 1;
    return c;
}
static void advance(KvMemRepositoryGc &gc, uint32_t steps) {
    for (uint32_t i = 0; i < steps; ++i) gc.begin_step();
}

static void protected_blocks_and_watermarks() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    auto c = config(3); c.recent_tokens = 32;
    std::vector<KvMemBlockHandle> freed, handles;
    KvMemRepositoryGc gc(c, repo, [&](const auto &r) { freed.push_back(r.handle); return true; });
    gc.begin_step();
    KvMemRepository::Lease checkpoint;
    for (uint32_t i = 0; i < 8; ++i) {
        auto l = repo->try_reserve(i * 32, 32).commit();
        handles.push_back(l.handle());
        uint32_t flags = backed;
        if (i == 0) flags |= KvMemGc::Mandatory;
        if (i == 1) flags |= KvMemGc::Io;
        if (i == 2) flags &= ~KvMemRepositoryGc::Indexed;
        if (i == 3) checkpoint = l.clone();
        gc.publish(l, flags);
    }
    gc.set_history_end(256);
    advance(gc, 10); gc.poll();
    check(freed.empty(), "candidate delay was skipped");
    gc.begin_step(); gc.poll();
    check(repo->ready_slots() == 3, "worker did not replenish the ready watermark");
    check(freed.size() == 3 && gc.stats().reclaimed == 3, "wrong number of reclaimed bundles");
    for (const auto h : freed) check(h.slot >= 4 && h.slot <= 6, "worker deleted protected history");
    advance(gc, 10); gc.poll();
    check(freed.size() == 3, "worker reclaimed beyond its high watermark");
    check(bool(repo->try_pin(handles[3])), "checkpoint block was hidden while pinned");
}

static void candidates_mature_before_pressure() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    auto c = config(2); c.policy.candidate_steps = 3;
    KvMemRepositoryGc gc(c, repo, [](const auto &) { return true; });
    gc.begin_step();
    for (uint32_t i = 0; i < 4; ++i) {
        auto lease = repo->try_reserve(i * 32, 32).commit();
        gc.publish(lease, backed);
    }
    advance(gc, 5); gc.poll();
    advance(gc, 3); gc.poll();
    check(repo->ready_slots() == 4 && gc.stats().retired == 0,
          "candidate observation deleted above the watermark");
    for (uint32_t i = 4; i < 7; ++i) {
        auto lease = repo->try_reserve(i * 32, 32).commit();
        gc.publish(lease, backed | KvMemGc::Mandatory);
    }
    gc.poll(); // no additional request steps after pressure first appeared
    check(repo->ready_slots() == 2 && gc.stats().reclaimed == 1,
          "pressure incorrectly restarted the already-observed candidate delay");
}

static void hot_history_still_has_a_finite_capacity() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    auto c = config(2);
    std::vector<KvMemBlockHandle> handles;
    KvMemRepositoryGc gc(c, repo, [](const auto &) { return true; });
    gc.begin_step();
    for (uint32_t i = 0; i < 8; ++i) {
        auto lease = repo->try_reserve(i * 32, 32).commit();
        handles.push_back(lease.handle());
        gc.publish(lease, backed | KvMemGc::Mandatory);
    }
    for (uint32_t i = 0; i < 10; ++i) {
        gc.begin_step();
        for (auto h : handles) gc.publish(repo->try_pin(h), backed | KvMemGc::Mandatory, true);
        gc.poll();
    }
    for (uint32_t i = 2; i < 8; ++i) gc.publish(repo->try_pin(handles[i]), backed);
    gc.poll(); advance(gc, 1); gc.poll();
    check(repo->ready_slots() == 2 && gc.stats().policy_fallback_retired == 2,
          "soft frequency preference deadlocked the finite repository");
    check(bool(repo->try_pin(handles[0])) && bool(repo->try_pin(handles[1])),
          "capacity fallback removed a mandatory block");
}

static void one_request_ingest_grace_and_protection() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    auto c = config(2); c.candidate_ingest_tokens = 64;
    c.recent_tokens = 32; c.policy.candidate_steps = 100;
    KvMemRepositoryGc gc(c, repo, [](const auto &) { return true; });
    // Also exercise step zero: candidate observation cannot use step=0 as an
    // unset sentinel, since no subsequent request is needed for long ingest.
    std::vector<KvMemBlockHandle> handles;
    KvMemRepository::Lease checkpoint;
    for (uint32_t i = 0; i < 5; ++i) {
        auto l = repo->try_reserve(i * 32, 32).commit(); handles.push_back(l.handle());
        if (i == 1) checkpoint = l.clone();
        gc.publish(l, backed | (i == 0 ? KvMemGc::Mandatory : 0));
    }
    gc.set_history_end(160); gc.poll();
    for (uint32_t i = 5; i < 8; ++i) {
        auto l = repo->try_reserve(i * 32, 32).commit(); handles.push_back(l.handle());
        gc.publish(l, backed | KvMemGc::Io);
        gc.set_history_end((i + 1) * 32);
        if (i == 5) {
            gc.poll();
            check(gc.stats().retired == 0, "single-request candidates matured too early");
            // A use resets candidate grace despite later history progress.
            gc.publish(repo->try_pin(handles[3]), backed, true);
        }
    }
    gc.poll();
    check(gc.step() == 0 && repo->ready_slots() == 2,
          "single-request history did not replenish the reserve");
    check(bool(repo->try_pin(handles[0])) && bool(repo->try_pin(handles[1])),
          "ingest grace bypassed mandatory/checkpoint protection");
    for (uint32_t i = 5; i < 8; ++i)
        check(bool(repo->try_pin(handles[i])), "ingest grace bypassed IO protection");
    check(gc.stats().policy_fallback_retired == 2,
          "same-request age must remain a soft preference, not fabricated old heat");
}

static void released_reader_notifies_the_worker() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{2, 32});
    auto c = config(1); c.background = true; c.policy.poll_ms = 60000;
    c.policy.candidate_steps = 0;
    KvMemRepositoryGc gc(c, repo, [](const auto &) { return true; });
    gc.begin_step();
    auto pinned = repo->try_reserve(0, 32).commit();
    auto other = repo->try_reserve(32, 32).commit();
    gc.publish(pinned, backed); gc.publish(other, backed | KvMemGc::Mandatory);
    other.reset(); advance(gc, 10);
    const auto wait = [&](auto ready) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!ready() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(ready(), "CPU worker ignored released-reader notification");
    };
    const auto ticks = gc.stats().ticks;
    gc.notify_progress(); wait([&] { return gc.stats().ticks > ticks; });
    check(repo->ready_slots() == 0, "notified GC retired a pinned reader");
    pinned.reset(); gc.notify_progress();
    wait([&] { return repo->ready_slots() == 1; });
}

static void semantic_low_scores_and_generation_reset() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{6, 32});
    auto c = config(1); c.policy.candidate_steps = 0;
    std::vector<KvMemBlockHandle> handles, freed;
    KvMemRepositoryGc gc(c, repo, [&](const auto &r) { freed.push_back(r.handle); return true; });
    gc.begin_step();
    for (uint32_t i = 0; i < 6; ++i) {
        auto l = repo->try_reserve(i * 32, 32).commit();
        handles.push_back(l.handle()); gc.publish(l, backed | KvMemGc::Mandatory);
    }
    gc.set_history_end(192); advance(gc, 4);
    for (uint64_t seq = 1; seq <= 3; ++seq) {
        gc.begin_step();
        auto scores = std::make_shared<KvMemRepositoryGc::Scores>();
        scores->step = gc.step(); scores->sequence = seq;
        for (uint32_t i = 0; i < 6; ++i) scores->entries.push_back({handles[i], double(i), i == 5});
        gc.submit_scores(scores); gc.poll();
    }
    for (const auto h : handles) gc.publish(repo->try_pin(h), backed);
    gc.poll();
    check(freed.size() == 1 && freed.front() == handles.front(), "sustained low-score block was not preferred");
    auto replacement = repo->try_reserve(192, 32).commit();
    check(replacement.handle().slot == handles.front().slot, "victim slot was not reused");
    const auto fresh = replacement.handle();
    gc.publish(replacement, backed | KvMemGc::Anchor);
    replacement.reset();
    auto stale = std::make_shared<KvMemRepositoryGc::Scores>();
    stale->sequence = 5; stale->step = gc.step(); stale->entries.push_back({handles[0], -100, false});
    gc.submit_scores(stale); gc.poll();
    check(gc.stats().score_gaps == 1 && gc.stats().stale_scores == 1, "lost/stale score evidence was accepted");
    check(bool(repo->try_pin(fresh)), "old score/candidate deleted replacement anchor");
}

static void cancelled_io_and_partial_cleanup() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{2, 32});
    uint32_t calls = 0;
    KvMemRepositoryGc gc(config(1, Mode::Off), repo, [&](const auto &) {
        ++calls;
        if (calls == 1) return false;
        if (calls == 2) throw std::runtime_error("injected allocator busy/failure");
        return true;
    });
    auto reservation = repo->try_reserve(0, 32);
    auto io = reservation.lease();
    reservation.cancel();
    gc.poll();
    check(!calls && repo->ready_slots() == 1, "cancelled in-flight IO was reclaimed");
    io.reset(); gc.poll();
    check(calls == 1 && repo->stats().releasing == 1, "partial cleanup was published as free");
    gc.poll();
    check(calls == 2 && gc.stats().failures == 1, "cleanup exception lost retry state");
    gc.poll();
    check(calls == 3 && repo->ready_slots() == 2, "cancelled bundle was not reclaimed with GC policy off");
}

static void shadow_and_off_do_not_delete() {
    for (auto mode : {Mode::Shadow, Mode::Off}) {
        auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
        auto c = config(1, mode); c.policy.candidate_steps = 0;
        uint32_t callbacks = 0;
        KvMemRepositoryGc gc(c, repo, [&](const auto &) { ++callbacks; return true; });
        std::vector<KvMemBlockHandle> handles;
        gc.begin_step();
        for (uint32_t i = 0; i < 4; ++i) {
            auto l = repo->try_reserve(i * 32, 32).commit();
            handles.push_back(l.handle()); gc.publish(l, backed);
        }
        advance(gc, 5); gc.poll();
        check(callbacks == 0 && repo->stats().live == 4, "nondelete mode deleted live blocks");
        if (mode == Mode::Shadow) {
            check(gc.stats().shadow_candidates > 0, "shadow mode did not record candidates");
            gc.begin_step(); gc.publish(repo->try_pin(handles[0]), backed, true); gc.poll();
            check(gc.stats().shadow_reuses == 1, "shadow counter missed a later reuse");
        }
    }
}

static void store_background_continuation() {
    constexpr uint32_t capacity = 16, steps = 512;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{capacity, 32});
    KvMemStoreConfig sc; sc.block_tokens = 32; sc.select_budget = 64; sc.sink_blocks = 1; sc.recent_blocks = 1;
    KvMemStore store(sc, repo);
    auto c = config(8); c.ready_low = 6; c.recent_tokens = 32; c.background = true;
    std::vector<uint64_t> payload(capacity * 32, UINT64_MAX);
    std::vector<uint64_t> generations(capacity, 0);
    // Only the producer writes live payload and only the worker frees retired
    // payload. Repository lease/ready-queue handoff is the lifetime fence.
    auto gc = std::make_shared<KvMemRepositoryGc>(c, repo, [&](const auto &r) {
        if (generations[r.handle.slot] != r.handle.generation)
            throw std::runtime_error("release callback observed a recycled generation");
        for (uint32_t i = 0; i < 32; ++i) payload[r.handle.slot * 32 + i] = UINT64_MAX;
        generations[r.handle.slot] = 0;
        return true;
    });
    store.attach_repository_gc(gc);
    const auto metadata_bytes = gc->stats().metadata_bytes;
    for (uint32_t step = 0; step < steps; ++step) {
        store.gc_begin_step();
        std::unique_ptr<KvMemStore::RepositoryAppend> append;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!(append = store.prepare_repository_append(32, 0))) {
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("background GC admission stalled");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto span = append->spans().front();
        generations[span.handle.slot] = span.handle.generation;
        for (uint32_t i = 0; i < 32; ++i) payload[span.storage_begin + i] = span.history_begin + i;
        append->commit(32);
        auto reader = store.repository_storage_read(uint64_t(step) * 32, 32);
        store.set_repository_gc_flags(span.handle, backed & ~KvMemGc::Sealed, backed);
        store.set_block_tier(span.handle.slot, KvTier::CPU, span.handle.slot, -1, span.handle.generation);
        for (uint32_t i = 0; i < 32; ++i) check(payload[span.storage_begin + i] == span.history_begin + i,
                                              "background GC corrupted a leased payload");
        reader = {};
        check(store.repository_directory_entries() <= capacity && gc->stats().metadata_bytes == metadata_bytes,
              "worker metadata grew with cumulative history");
        // Simulate separated requests, allowing candidate_steps to advance
        // between worker observations (production requests take much longer).
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    gc->stop();
    check(gc->stats().failures == 0, "background cleanup saw a stale/wrong allocation");
    check(gc->stats().reclaimed >= steps - capacity, "background did not reuse fixed capacity");
    check(store.history_end() == uint64_t(steps) * 32, "history stopped at physical capacity");
    check(store.repository_storage_read(0, 32).spans.front().handle.slot == 0, "sink history was deleted");
    std::cout << "background continuation: slots=" << capacity << " blocks=" << steps
              << " reclaimed=" << gc->stats().reclaimed << " max_tick_ns=" << gc->stats().max_tick_ns << '\n';
}

static void nvme_background_resource_bundles() {
    constexpr uint32_t capacity = 8, steps = 256;
    NvmeKvTierConfig nc;
    nc.dir = "/tmp/qw3_repository_gc_test_" + std::to_string(getpid());
    nc.total_bytes = capacity * 64; nc.slot_bytes = 64;
    {
        nc.file_name = "raw.bin"; NvmeKvTier raw(nc);
        nc.file_name = "values.bin"; NvmeKvTier values(nc);
        nc.file_name = "mtp.bin"; NvmeKvTier mtp(nc);
        auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{capacity, 32});
        KvMemStoreConfig sc; sc.block_tokens = 32; sc.sink_blocks = 0; sc.recent_blocks = 1;
        KvMemStore store(sc, repo);
        auto c = config(4); c.ready_low = 3; c.recent_tokens = 32; c.background = true;
        c.policy.grace_steps = 0; c.policy.candidate_steps = 0;
        std::array<NvmeAllocation, capacity> raw_ids{}, value_ids{}, mtp_ids{};
        std::array<uint64_t, capacity> generations{};
        auto gc = std::make_shared<KvMemRepositoryGc>(c, repo, [&](const auto &r) {
            if (generations[r.handle.slot] != r.handle.generation)
                throw std::runtime_error("NVMe resource bundle generation mismatch");
            const auto s = r.handle.slot;
            return raw.try_release_allocation(raw_ids[s]) && values.try_release_allocation(value_ids[s]) &&
                   mtp.try_release_allocation(mtp_ids[s]);
        });
        store.attach_repository_gc(gc);
        std::array<uint8_t, 64> bytes{}, readback{};
        for (uint32_t i = 0; i < steps; ++i) {
            store.gc_begin_step();
            std::unique_ptr<KvMemStore::RepositoryAppend> a;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!(a = store.prepare_repository_append(32, 0))) {
                if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("NVMe bundle refill stalled");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto h = a->spans().front().handle;
            generations[h.slot] = h.generation;
            bytes.fill(uint8_t(i % 251));
            const auto stale_raw = raw_ids[h.slot];
            raw.write_block(h.slot, bytes.data(), bytes.size());
            values.write_block(h.slot, bytes.data(), bytes.size());
            mtp.write_block(h.slot, bytes.data(), bytes.size());
            raw_ids[h.slot] = raw.allocation(h.slot); value_ids[h.slot] = values.allocation(h.slot);
            mtp_ids[h.slot] = mtp.allocation(h.slot);
            a->commit(32);
            auto read = store.repository_storage_read(uint64_t(i) * 32, 32);
            store.set_repository_gc_flags(h, backed & ~KvMemGc::Sealed, backed);
            store.set_block_tier(h.slot, KvTier::SSD, -1, value_ids[h.slot].slot, h.generation);
            if (i >= capacity) {
                const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                while (!raw.try_release_allocation(stale_raw)) {
                    // false is allocator contention, not an identity failure;
                    // the real CPU worker likewise retries on its next tick.
                    if (std::chrono::steady_clock::now() >= cleanup_deadline)
                        throw std::runtime_error("stale NVMe cleanup stayed busy");
                    std::this_thread::yield();
                }
            }
            for (auto *tier : {&raw, &values, &mtp}) {
                tier->read_block(h.slot, readback.data(), readback.size());
                check(bytes == readback, "main/MTP payload was corrupted after slot reuse");
                check(tier->used_slots() <= capacity, "NVMe allocation exceeded fixed capacity");
            }
        }
        gc->stop();
        check(gc->stats().failures == 0 && gc->stats().reclaimed >= steps - capacity,
              "background NVMe cleanup failed or did not reuse storage");
        std::cout << "NVMe background: slots=" << capacity << " blocks=" << steps
                  << " reclaimed=" << gc->stats().reclaimed << '\n';
    }
    ::rmdir(nc.dir.c_str());
}

// Exercise the production resource callback (raw includes main/MTP K,
// values includes main/MTP V), including cancelled partial writes.
static void production_storage_release() {
    constexpr uint32_t capacity = 2;
    NvmeKvTierConfig nc;
    nc.dir = "/tmp/qw3_repository_storage_test_" + std::to_string(getpid());
    nc.total_bytes = capacity * 64; nc.slot_bytes = 64;
    {
        nc.file_name = "raw.bin"; NvmeKvTier raw(nc);
        nc.file_name = "value.bin"; NvmeKvTier values(nc);
        auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{capacity, 32});
        auto release = kvmem_repository_storage_release(capacity, &raw, &values);
        KvMemRepositoryGc gc(config(1, Mode::Off), repo, release);
        std::array<uint8_t, 64> bytes{}, actual{};
        KvMemRepository::Record old;
        for (uint32_t i = 0; i < 128; ++i) {
            auto a = repo->try_reserve(uint64_t(i) * 32, 32);
            check(bool(a), "production cleanup did not return ready storage");
            const auto record = a.record();
            bytes.fill(uint8_t(i));
            raw.write_block(record.handle.slot, bytes.data(), bytes.size());
            // Every third cancellation happens before writing any V.
            if (i % 3) values.write_block(record.handle.slot, bytes.data(), bytes.size());
            auto io = a.lease();
            a.cancel();
            gc.poll();
            check(raw.used_slots() == 1, "in-flight raw bundle was released");
            raw.read_block(record.handle.slot, actual.data(), actual.size());
            check(actual == bytes, "IO lease did not preserve raw/MTP bytes");
            io.reset(); gc.poll();
            check(repo->ready_slots() == capacity && !raw.used_slots() && !values.used_slots(),
                  "partial main/MTP resource bundle was not completely released");
            if (i == 0) old = record;
        }
        check(gc.stats().failures == 0 && gc.stats().reclaimed == 128,
              "production cleanup has failures or lost cancelled writes");
        // CPU-only and one optional SSD tier use the same completion contract.
        auto cpu = kvmem_repository_storage_release(capacity, nullptr, nullptr);
        check(cpu(old), "CPU fixed arena could not return an unused slot");
        auto mixed = kvmem_repository_storage_release(capacity, &raw, nullptr);
        raw.write_block(old.handle.slot, bytes.data(), bytes.size());
        check(mixed(old) && !raw.used_slots(), "optional raw tier was not released");
    }
    ::rmdir(nc.dir.c_str());
}

int main() {
    try {
        protected_blocks_and_watermarks();
        candidates_mature_before_pressure();
        hot_history_still_has_a_finite_capacity();
        one_request_ingest_grace_and_protection();
        released_reader_notifies_the_worker();
        semantic_low_scores_and_generation_reset();
        cancelled_io_and_partial_cleanup();
        shadow_and_off_do_not_delete();
        store_background_continuation();
        nvme_background_resource_bundles();
        production_storage_release();
        std::cout << "OK\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
