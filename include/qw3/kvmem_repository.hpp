#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace qw3 {

// A physical slot is NOT a history position. A slot can be recycled only after
// every payload consumer has dropped its lease and the complete resource bundle
// (main/MTP K/V and index backing) has been released by the CPU owner.
struct KvMemBlockHandle {
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 0;
    bool operator==(const KvMemBlockHandle &b) const {
        return slot == b.slot && generation == b.generation;
    }
    bool operator!=(const KvMemBlockHandle &b) const { return !(*this == b); }
};

class KvMemRepository {
    struct Shared;
public:
    struct Record {
        KvMemBlockHandle handle;
        uint64_t logical_id = 0;
        uint64_t token_begin = 0;
        // Reserved storage extent, immutable while leased. A partial append
        // can publish fewer valid tokens; KvMemStore owns the committed count.
        uint32_t token_count = 0;
        uint64_t token_end() const { return token_begin + token_count; }
    };
    struct Config {
        uint32_t capacity = 0;
        uint32_t block_tokens = 32;
    };
    // Obtaining an existing lease never runs GC, waits for IO, takes a mutex,
    // or allocates payload. Lease objects themselves are thread confined; move
    // or clone one to hand ownership to an asynchronous consumer.
    class Lease {
    public:
        Lease() = default;
        ~Lease();
        Lease(Lease &&) noexcept;
        Lease &operator=(Lease &&) noexcept;
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        explicit operator bool() const { return bool(shared_); }
        KvMemBlockHandle handle() const { return handle_; }
        const Record &record() const;
        Lease clone() const;
        void reset() noexcept;
    private:
        friend class KvMemRepository;
        Lease(std::shared_ptr<Shared>, KvMemBlockHandle);
        std::shared_ptr<Shared> shared_;
        KvMemBlockHandle handle_;
    };
    // Unpublished data is invisible to new readers. Cancellation also retires
    // its resource bundle; only the background reclaimer returns it to ready.
    // A writeback/DMA must hold lease() until actual completion, not just until
    // submission. A generation comparison cannot replace this lifetime fence.
    class Reservation {
    public:
        Reservation() = default;
        ~Reservation();
        Reservation(Reservation &&) noexcept;
        Reservation &operator=(Reservation &&) noexcept;
        Reservation(const Reservation &) = delete;
        Reservation &operator=(const Reservation &) = delete;
        explicit operator bool() const { return bool(lease_); }
        const Record &record() const { return lease_.record(); }
        Lease lease() const { return lease_.clone(); }
        Lease commit();
        void cancel() noexcept;
    private:
        friend class KvMemRepository;
        explicit Reservation(Lease);
        Lease lease_;
    };
    enum class ReclaimResult {
        Reclaimed, DeferredReaders, DeferredRelease, NotRetired, StaleHandle,
        Busy
    };
    struct Stats {
        uint32_t capacity = 0, ready = 0, writing = 0, live = 0;
        uint32_t retired = 0, releasing = 0, cancelled = 0;
        uint64_t leases = 0, reservations = 0, reclaimed = 0;
        // Actual fixed directory + ready-queue storage; excludes shared_ptr
        // control block, allocator bookkeeping and externally owned payloads.
        uint64_t metadata_bytes = 0;
    };

    explicit KvMemRepository(Config);
    KvMemRepository(const KvMemRepository &) = delete;
    KvMemRepository &operator=(const KvMemRepository &) = delete;

    // Single append producer. Empty means ready capacity is exhausted: callers
    // must apply explicit admission, never make decode perform a GC scan.
    // Histories/rewinds may reuse token positions, but never a logical identity.
    Reservation try_reserve(uint64_t token_begin, uint32_t token_count);
    // Concurrent readers. Only Live entries can acquire a fresh lease. An old
    // reader may clone its lease across retirement, and rescue that same entry.
    Lease try_pin(KvMemBlockHandle) const;
    Lease try_pin_slot(uint32_t slot) const;
    bool retire(KvMemBlockHandle);
    // Policy retirement must not hide blocks currently being scored, used by
    // a checkpoint, or copied asynchronously. Unlike explicit invalidation
    // above, this succeeds only if its own temporary pin is the sole reader.
    bool retire_if_unpinned(KvMemBlockHandle);
    // A policy worker may retain its own lease through retirement, recheck its
    // metadata snapshot, and rescue on a raced update before dropping the pin.
    bool retire_if_only_reader(const Lease &);
    bool rescue(const Lease &);
    // CPU owner only. The callback must release the whole resource bundle and
    // return true ONLY when safe to overwrite. On false or exception the slot
    // stays Releasing, cannot be rescued, and is retried with the same handle.
    // It must not launch GPU work. Concurrent reclaim calls return Busy.
    ReclaimResult try_reclaim(KvMemBlockHandle,
                             const std::function<bool(const Record &)> &);
    // Bounded worker scan, including cancelled unpublished reservations.
    // A returned handle is only a hint; try_reclaim revalidates it.
    KvMemBlockHandle retired_at(uint32_t slot) const;
    Stats stats() const;
    uint32_t capacity() const;
    uint32_t block_tokens() const;
    // Append-producer only: conservative O(1) admission snapshot. The single
    // consumer owns head, so the reclaimer can only add capacity after this
    // read. No scan of historical blocks is needed before a batched append.
    uint32_t ready_slots() const;
    // Same O(1) snapshot with the SPSC producer's load order. Reclaimer only.
    uint32_t ready_slots_for_reclaimer() const;
private:
    std::shared_ptr<Shared> shared_;
};

} // namespace qw3
