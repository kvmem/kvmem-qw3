#include "qw3/kvmem_repository.hpp"
#include "qw3/nvme_kv_tier.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using qw3::KvMemRepository;
using qw3::KvMemBlockHandle;
using Result = KvMemRepository::ReclaimResult;

static void check(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}
static bool freed(const KvMemRepository::Record &) { return true; }

static void capacity_and_long_history() {
    KvMemRepository repo({4, 32});
    const auto bytes = repo.stats().metadata_bytes;
    check(repo.ready_slots() == 4, "initial ready count wrong");
    std::array<KvMemBlockHandle, 4> oldest{};
    // Keep the pool full, repeatedly delete an interior/old block, and append
    // at positions beyond uint32. Neither directory nor slot count can grow.
    for (uint64_t i = 0; i < 4096; ++i) {
        if (i >= 4) {
            const auto victim = oldest[i % 4];
            check(repo.retire(victim), "failed to retire valid block");
            check(repo.try_reclaim(victim, freed) == Result::Reclaimed, "slot not returned");
            check(!repo.try_pin(victim), "deleted block remained accessible");
        }
        auto r = repo.try_reserve((uint64_t{1} << 33) + i * 32, 32);
        check(bool(r), "recycled pool exhausted");
        auto lease = r.commit();
        const auto h = lease.handle();
        check(lease.record().logical_id == i + 1, "history identity was reused");
        check(lease.record().token_begin == (uint64_t{1} << 33) + i * 32,
              "64-bit history position truncated");
        if (i >= 4) {
            check(h.slot == oldest[i % 4].slot, "freed slot not reused");
            check(h.generation != oldest[i % 4].generation, "generation reused");
            check(!repo.try_pin(oldest[i % 4]), "old handle pins new payload");
            check(!repo.retire(oldest[i % 4]), "old handle retires new payload");
        }
        oldest[i % 4] = h;
        if (i >= 3) check(!repo.try_reserve(0, 1), "capacity is not bounded");
        check(repo.stats().metadata_bytes == bytes, "metadata grows with history");
        check(repo.ready_slots() == repo.stats().ready, "wrapped ready admission count wrong");
    }
    const auto s = repo.stats();
    check(s.capacity == 4 && s.live == 4 && s.leases == 0 && s.ready == 0,
          "live/lease accounting mismatch");
    check(s.reservations == 4096 && s.reclaimed == 4092, "reuse accounting mismatch");
    bool bad = false;
    try { repo.try_reserve(UINT64_MAX - 3, 32); }
    catch (const std::invalid_argument &) { bad = true; }
    check(bad, "token interval overflow accepted");
}

static void leases_rescue_and_cancellation() {
    KvMemRepository repo({1, 32});
    auto r = repo.try_reserve(123, 5);
    const auto h = r.record().handle;
    check(!repo.try_pin(h), "unpublished payload visible");
    auto io = r.lease();
    auto active = r.commit();
    auto checkpoint = active.clone();
    check(!repo.retire_if_unpinned(h), "GC retired a checkpoint/IO protected entry");
    check(repo.retire(h), "retire failed");
    check(!repo.try_pin(h), "new reader entered retired block");
    check(io.record().token_begin == 123, "old IO view invalidated");
    check(repo.try_reclaim(h, freed) == Result::DeferredReaders, "live IO was freed");
    check(repo.rescue(checkpoint), "old checkpoint could not rescue payload");
    check(bool(repo.try_pin(h)), "rescued block not visible");
    check(repo.retire(h), "rescued block failed second retirement");
    active.reset(); checkpoint.reset();
    check(repo.try_reclaim(h, freed) == Result::DeferredReaders, "IO reference ignored");
    io.reset();
    check(repo.try_reclaim(h, freed) == Result::Reclaimed, "last IO did not unblock release");

    KvMemBlockHandle aborted;
    {
        auto pending = repo.try_reserve(999, 1);
        aborted = pending.record().handle;
        io = pending.lease();
        // Scope exit must retire rather than leak an incomplete write.
    }
    check(repo.stats().cancelled == 1, "cancelled reservation leaked");
    check(!repo.rescue(io), "cancelled uncommitted bytes were published by rescue");
    check(repo.try_reclaim(aborted, freed) == Result::DeferredReaders,
          "cancelled write reclaimed before IO completion");
    io.reset();
    check(repo.try_reclaim(aborted, freed) == Result::Reclaimed, "cancelled slot not reclaimed");
}

static void partial_release_and_exceptions() {
    KvMemRepository repo({1, 32});
    auto r = repo.try_reserve(0, 32);
    auto h = r.record().handle;
    r.commit().reset();
    check(repo.retire(h), "retire failed");
    unsigned calls = 0;
    auto release = [&](const KvMemRepository::Record &v) {
        check(v.handle == h, "release callback observed replacement");
        ++calls;
        if (calls == 1) return false; // e.g. raw K freed, V metadata lock busy
        if (calls == 2) throw std::runtime_error("injected release failure");
        return true;
    };
    check(repo.try_reclaim(h, release) == Result::DeferredRelease, "partial release became ready");
    check(repo.stats().releasing == 1 && !repo.try_reserve(32, 32),
          "partially released slot reused");
    bool threw = false;
    try { repo.try_reclaim(h, release); } catch (const std::runtime_error &) { threw = true; }
    check(threw && repo.stats().releasing == 1, "release exception lost lifetime state");
    check(repo.try_reclaim(h, release) == Result::Reclaimed && calls == 3,
          "release retry did not preserve ownership");
    auto replacement = repo.try_reserve(32, 32).commit();
    check(replacement.handle() != h, "replacement has old identity");
    check(repo.try_reclaim(h, release) == Result::NotRetired && calls == 3,
          "stale release touched replacement");
    const auto fresh = replacement.handle();
    check(repo.retire(fresh), "replacement retirement failed");
    replacement.reset();
    check(repo.try_reclaim(h, release) == Result::StaleHandle && calls == 3,
          "stale handle reclaimed a retired replacement");
    check(repo.try_reclaim(fresh, freed) == Result::Reclaimed, "replacement leaked");
}

static void nvme_resource_bundle() {
    qw3::NvmeKvTierConfig c;
    c.dir = "/tmp/qw3_repository_test_" + std::to_string(getpid());
    c.total_bytes = 4 * 64; c.slot_bytes = 64;
    {
        c.file_name = "raw.bin"; qw3::NvmeKvTier raw(c);
        c.file_name = "values.bin"; qw3::NvmeKvTier values(c);
        c.file_name = "mtp.bin"; qw3::NvmeKvTier mtp(c);
        KvMemRepository repo({4, 32});
        std::array<qw3::NvmeAllocation, 4> raw_ids{}, value_ids{}, mtp_ids{};
        std::array<KvMemBlockHandle, 4> handles{};
        std::array<uint8_t, 64> payload{}, readback{};
        auto release = [&](const KvMemRepository::Record &r) {
            const auto s = r.handle.slot;
            return raw.try_release_allocation(raw_ids[s]) &&
                   values.try_release_allocation(value_ids[s]) &&
                   mtp.try_release_allocation(mtp_ids[s]);
        };
        for (uint64_t i = 0; i < 400; ++i) {
            if (i >= 4) {
                check(repo.retire(handles[i % 4]), "NVMe retirement failed");
                check(repo.try_reclaim(handles[i % 4], release) == Result::Reclaimed,
                      "NVMe bundle not freed");
            }
            auto r = repo.try_reserve(i * 32, 32);
            const auto h = r.record().handle;
            const auto s = h.slot;
            const auto old_raw = raw_ids[s];
            payload.fill(uint8_t(i % 251));
            raw.write_block(s, payload.data(), payload.size());
            values.write_block(s, payload.data(), payload.size());
            mtp.write_block(s, payload.data(), payload.size());
            raw_ids[s] = raw.allocation(s); value_ids[s] = values.allocation(s);
            mtp_ids[s] = mtp.allocation(s);
            auto live = r.commit();
            handles[i % 4] = h;
            if (i >= 4) check(raw.try_release_allocation(old_raw), "stale NVMe cleanup failed");
            for (auto *tier : {&raw, &values, &mtp}) {
                tier->read_block(s, readback.data(), readback.size());
                check(readback == payload, "recycled K/V/MTP payload corrupted");
                check(tier->used_slots() == std::min<uint64_t>(4, i + 1),
                      "NVMe physical allocation grew with history");
            }
        }
    }
    ::rmdir(c.dir.c_str());
}

static void concurrent_payload_reuse() {
    constexpr unsigned cap = 8, iterations = 20000;
    KvMemRepository repo({cap, 32});
    std::array<std::array<uint64_t, 8>, cap> data{};
    std::atomic<bool> writer_done{false}, stop_readers{false}, corrupt{false};
    std::atomic<uint64_t> observed{0};
    std::vector<std::thread> readers;
    for (unsigned n = 0; n < 3; ++n) readers.emplace_back([&] {
        while (!stop_readers.load(std::memory_order_acquire)) {
            for (unsigned s = 0; s < cap; ++s) {
                auto l = repo.try_pin_slot(s);
                if (!l) continue;
                for (auto v : data[s]) if (v != l.record().logical_id) corrupt = true;
                if ((l.record().logical_id % 127) == 0) std::this_thread::yield();
                for (auto v : data[s]) if (v != l.record().logical_id) corrupt = true;
                ++observed;
            }
        }
    });
    std::thread worker([&] {
        while (!writer_done.load(std::memory_order_acquire) || repo.stats().ready != cap) {
            for (unsigned s = 0; s < cap; ++s) {
                const auto h = repo.retired_at(s);
                if (h.generation) repo.try_reclaim(h, [&](const KvMemRepository::Record &r) {
                    for (auto v : data[r.handle.slot]) if (v != r.logical_id) corrupt = true;
                    data[r.handle.slot].fill(0); // teardown must not race a reader
                    return true;
                });
            }
            std::this_thread::yield();
        }
    });
    for (unsigned i = 0; i < iterations; ++i) {
        KvMemRepository::Reservation r;
        while (!(r = repo.try_reserve((uint64_t{1} << 34) + uint64_t(i) * 32, 32)))
            std::this_thread::yield();
        data[r.record().handle.slot].fill(r.record().logical_id);
        auto l = r.commit();
        if ((i % 7) == 0) std::this_thread::yield();
        check(repo.retire(l.handle()), "concurrent retirement failed");
    }
    writer_done.store(true, std::memory_order_release);
    worker.join();
    stop_readers.store(true, std::memory_order_release);
    for (auto &r : readers) r.join();
    check(!corrupt && observed > 0, "concurrent reader saw overwritten payload");
    const auto s = repo.stats();
    check(s.ready == cap && s.reclaimed == iterations && s.leases == 0,
          "concurrent run leaked slots or leases");
}

int main() {
    try {
        capacity_and_long_history();
        leases_rescue_and_cancellation();
        partial_release_and_exceptions();
        nvme_resource_bundle();
        concurrent_payload_reuse();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
    std::cout << "KVMem bounded repository tests passed (4096 appends / 4 slots, "
                 "400 NVMe bundles / 4 slots, 20000 concurrent reuses)\n";
}
