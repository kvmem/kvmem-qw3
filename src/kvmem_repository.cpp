#include "qw3/kvmem_repository.hpp"

#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qw3 {
namespace {
constexpr unsigned phase_shift = 60;
constexpr uint64_t ref_mask = (uint64_t{1} << phase_shift) - 1;
enum class Phase : uint64_t { Free, Writing, Live, Retired, Releasing, Cancelled };
constexpr uint64_t state(Phase p, uint64_t refs = 0) {
    return (uint64_t(p) << phase_shift) | refs;
}
Phase phase(uint64_t v) { return Phase(v >> phase_shift); }
uint64_t refs(uint64_t v) { return v & ref_mask; }
// Retirement and pin acquisition share a single CAS, so readers cannot race
// between a separate 'is_live' check and incrementing the reference count.
bool acquire(std::atomic<uint64_t> &control, bool existing) {
    auto prior = control.load(std::memory_order_acquire);
    do {
        const Phase p = phase(prior);
        if (existing) {
            if (!refs(prior) || p == Phase::Free || p == Phase::Releasing) return false;
        } else if (p != Phase::Live) return false;
        if (refs(prior) == ref_mask) return false;
    } while (!control.compare_exchange_weak(prior, prior + 1,
                 std::memory_order_acq_rel, std::memory_order_acquire));
    return true;
}
} // namespace

struct KvMemRepository::Shared {
    struct Slot {
        std::atomic<uint64_t> control{state(Phase::Free)};
        // Generation is separately atomic only for the worker's scan hint.
        // Actual record reads always hold a lease or exclusive Releasing state.
        std::atomic<uint64_t> generation{0};
        Record record;
    };
    explicit Shared(Config c)
        : config(c), slots(new Slot[c.capacity]), ready(c.capacity + 1) {
        for (uint32_t i = 0; i < c.capacity; ++i) ready[i] = i;
        tail.store(c.capacity, std::memory_order_relaxed);
    }
    Config config;
    std::unique_ptr<Slot[]> slots;
    // Single producer (reclaimer), single consumer (append). Prepopulated once
    // at construction; no vector resizing or allocation when slots are reused.
    std::vector<uint32_t> ready;
    alignas(64) std::atomic<uint32_t> head{0};
    alignas(64) std::atomic<uint32_t> tail{0};
    std::atomic_flag reclaim_owner = ATOMIC_FLAG_INIT;
    uint64_t sequence = 0; // sole append producer, never rewound
    std::atomic<uint64_t> reservations{0}, reclaimed{0};
    uint32_t next(uint32_t i) const { return i == config.capacity ? 0 : i + 1; }
    bool pop(uint32_t &slot) {
        const uint32_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        slot = ready[h];
        head.store(next(h), std::memory_order_release);
        return true;
    }
    void push(uint32_t slot) {
        const uint32_t t = tail.load(std::memory_order_relaxed);
        const uint32_t n = next(t);
        if (n == head.load(std::memory_order_acquire))
            throw std::logic_error("KVMem ready queue overflow");
        ready[t] = slot;
        tail.store(n, std::memory_order_release);
    }
};

KvMemRepository::KvMemRepository(Config c) {
    if (!c.capacity || c.capacity == UINT32_MAX || !c.block_tokens)
        throw std::invalid_argument("invalid KVMem repository capacity/block size");
    shared_ = std::make_shared<Shared>(c);
}
uint32_t KvMemRepository::capacity() const { return shared_->config.capacity; }
uint32_t KvMemRepository::block_tokens() const { return shared_->config.block_tokens; }
uint32_t KvMemRepository::ready_slots() const {
    const auto &s = *shared_;
    const uint32_t h = s.head.load(std::memory_order_relaxed);
    const uint32_t t = s.tail.load(std::memory_order_acquire);
    return t >= h ? t - h : (s.config.capacity - h) + t + 1;
}
uint32_t KvMemRepository::ready_slots_for_reclaimer() const {
    const auto &s = *shared_;
    const uint32_t t = s.tail.load(std::memory_order_relaxed);
    const uint32_t h = s.head.load(std::memory_order_acquire);
    return t >= h ? t - h : (s.config.capacity - h) + t + 1;
}

KvMemRepository::Lease::Lease(std::shared_ptr<Shared> s, KvMemBlockHandle h)
    : shared_(std::move(s)), handle_(h) {}
KvMemRepository::Lease::~Lease() { reset(); }
KvMemRepository::Lease::Lease(Lease &&b) noexcept
    : shared_(std::move(b.shared_)), handle_(b.handle_) { b.handle_ = {}; }
KvMemRepository::Lease &KvMemRepository::Lease::operator=(Lease &&b) noexcept {
    if (this != &b) {
        reset(); shared_ = std::move(b.shared_); handle_ = b.handle_; b.handle_ = {};
    }
    return *this;
}
void KvMemRepository::Lease::reset() noexcept {
    if (shared_) {
        shared_->slots[handle_.slot].control.fetch_sub(1, std::memory_order_release);
        shared_.reset(); handle_ = {};
    }
}
const KvMemRepository::Record &KvMemRepository::Lease::record() const {
    if (!shared_) throw std::logic_error("empty KVMem repository lease");
    return shared_->slots[handle_.slot].record;
}
KvMemRepository::Lease KvMemRepository::Lease::clone() const {
    if (!shared_) return {};
    if (!acquire(shared_->slots[handle_.slot].control, true))
        throw std::overflow_error("KVMem repository lease overflow");
    return Lease(shared_, handle_);
}

KvMemRepository::Reservation::Reservation(Lease l) : lease_(std::move(l)) {}
KvMemRepository::Reservation::~Reservation() { cancel(); }
KvMemRepository::Reservation::Reservation(Reservation &&b) noexcept
    : lease_(std::move(b.lease_)) {}
KvMemRepository::Reservation &KvMemRepository::Reservation::operator=(Reservation &&b) noexcept {
    if (this != &b) { cancel(); lease_ = std::move(b.lease_); }
    return *this;
}
KvMemRepository::Lease KvMemRepository::Reservation::commit() {
    if (!lease_) throw std::logic_error("empty KVMem repository reservation");
    auto &control = lease_.shared_->slots[lease_.handle_.slot].control;
    auto prior = control.load(std::memory_order_acquire);
    do {
        if (phase(prior) != Phase::Writing)
            throw std::logic_error("KVMem reservation is no longer writable");
    } while (!control.compare_exchange_weak(prior, state(Phase::Live, refs(prior)),
                 std::memory_order_acq_rel, std::memory_order_acquire));
    return std::move(lease_);
}
void KvMemRepository::Reservation::cancel() noexcept {
    if (!lease_) return;
    auto &control = lease_.shared_->slots[lease_.handle_.slot].control;
    auto prior = control.load(std::memory_order_acquire);
    while (phase(prior) == Phase::Writing &&
           !control.compare_exchange_weak(prior, state(Phase::Cancelled, refs(prior)),
                 std::memory_order_acq_rel, std::memory_order_acquire)) {}
    lease_.reset();
}

KvMemRepository::Reservation KvMemRepository::try_reserve(uint64_t begin, uint32_t count) {
    auto &s = *shared_;
    if (!count || count > s.config.block_tokens || begin > UINT64_MAX - count)
        throw std::invalid_argument("invalid KVMem block token interval");
    if (s.sequence == UINT64_MAX)
        throw std::overflow_error("KVMem history identity exhausted");
    uint32_t slot;
    if (!s.pop(slot)) return {};
    auto &entry = s.slots[slot];
    // The ready queue's release/acquire handoff guarantees that all previous
    // cleanup finished. New readers cannot see a partially initialized record.
    entry.control.store(state(Phase::Writing, 1), std::memory_order_relaxed);
    const KvMemBlockHandle handle{slot, ++s.sequence};
    entry.record = {handle, handle.generation, begin, count};
    entry.generation.store(handle.generation, std::memory_order_release);
    s.reservations.fetch_add(1, std::memory_order_relaxed);
    return Reservation(Lease(shared_, handle));
}

KvMemRepository::Lease KvMemRepository::try_pin_slot(uint32_t slot) const {
    if (slot >= capacity()) return {};
    auto &entry = shared_->slots[slot];
    if (!acquire(entry.control, false)) return {};
    return Lease(shared_, entry.record.handle);
}
KvMemRepository::Lease KvMemRepository::try_pin(KvMemBlockHandle h) const {
    auto lease = try_pin_slot(h.slot);
    if (lease && lease.handle() != h) lease.reset();
    return lease;
}
bool KvMemRepository::retire(KvMemBlockHandle h) {
    auto lease = try_pin(h);
    if (!lease) return false;
    auto &control = shared_->slots[h.slot].control;
    auto prior = control.load(std::memory_order_acquire);
    do {
        if (phase(prior) != Phase::Live) return false;
    } while (!control.compare_exchange_weak(prior, state(Phase::Retired, refs(prior)),
                 std::memory_order_acq_rel, std::memory_order_acquire));
    return true;
}
bool KvMemRepository::rescue(const Lease &lease) {
    if (!lease || lease.shared_ != shared_) return false;
    auto &control = shared_->slots[lease.handle_.slot].control;
    auto prior = control.load(std::memory_order_acquire);
    do {
        if (phase(prior) == Phase::Live) return true;
        if (phase(prior) != Phase::Retired) return false;
    } while (!control.compare_exchange_weak(prior, state(Phase::Live, refs(prior)),
                 std::memory_order_acq_rel, std::memory_order_acquire));
    return true;
}

bool KvMemRepository::retire_if_unpinned(KvMemBlockHandle h) {
    auto lease = try_pin(h);
    return retire_if_only_reader(lease);
}
bool KvMemRepository::retire_if_only_reader(const Lease &lease) {
    if (!lease || lease.shared_ != shared_) return false;
    uint64_t expected = state(Phase::Live, 1);
    return shared_->slots[lease.handle_.slot].control.compare_exchange_strong(
        expected, state(Phase::Retired, 1),
        std::memory_order_acq_rel, std::memory_order_acquire);
}

KvMemRepository::ReclaimResult KvMemRepository::try_reclaim(
        KvMemBlockHandle h, const std::function<bool(const Record &)> &release) {
    if (h.slot >= capacity() || !h.generation) return ReclaimResult::StaleHandle;
    if (!release) throw std::invalid_argument("KVMem release callback is empty");
    auto &s = *shared_;
    if (s.reclaim_owner.test_and_set(std::memory_order_acquire)) return ReclaimResult::Busy;
    struct Unlock { std::atomic_flag &flag; ~Unlock() { flag.clear(std::memory_order_release); } } unlock{s.reclaim_owner};
    auto &entry = s.slots[h.slot];
    auto prior = entry.control.load(std::memory_order_acquire);
    // This thread is now the sole reclaimer. A retired record cannot be reused
    // underneath the checks; a reader rescue is resolved by the CAS below.
    if (phase(prior) != Phase::Retired && phase(prior) != Phase::Releasing &&
        phase(prior) != Phase::Cancelled)
        return ReclaimResult::NotRetired;
    if (entry.generation.load(std::memory_order_acquire) != h.generation)
        return ReclaimResult::StaleHandle;
    if (refs(prior)) return ReclaimResult::DeferredReaders;
    if (phase(prior) != Phase::Releasing &&
        !entry.control.compare_exchange_strong(prior, state(Phase::Releasing),
                 std::memory_order_acq_rel, std::memory_order_acquire))
        return ReclaimResult::DeferredReaders;
    // A false return may mean one component was already freed. Do not expose
    // this entry again or allow rescue after the first release attempt.
    if (!release(entry.record)) return ReclaimResult::DeferredRelease;
    entry.control.store(state(Phase::Free), std::memory_order_release);
    s.push(h.slot);
    s.reclaimed.fetch_add(1, std::memory_order_relaxed);
    return ReclaimResult::Reclaimed;
}
KvMemBlockHandle KvMemRepository::retired_at(uint32_t slot) const {
    if (slot >= capacity()) return {};
    const auto &e = shared_->slots[slot];
    const auto p = phase(e.control.load(std::memory_order_acquire));
    if (p != Phase::Retired && p != Phase::Releasing && p != Phase::Cancelled) return {};
    return {slot, e.generation.load(std::memory_order_acquire)};
}
KvMemRepository::Stats KvMemRepository::stats() const {
    Stats out;
    out.capacity = capacity();
    // Diagnostic scan is O(physical capacity), never O(cumulative history).
    // During concurrent updates these counts are observational, not an atomic
    // admission snapshot. Allocation uses the ready queue directly instead.
    for (uint32_t i = 0; i < out.capacity; ++i) {
        const auto v = shared_->slots[i].control.load(std::memory_order_acquire);
        out.leases += refs(v);
        switch (phase(v)) {
        case Phase::Free: ++out.ready; break;
        case Phase::Writing: ++out.writing; break;
        case Phase::Live: ++out.live; break;
        case Phase::Retired: ++out.retired; break;
        case Phase::Releasing: ++out.releasing; break;
        case Phase::Cancelled: ++out.cancelled; break;
        }
    }
    out.reservations = shared_->reservations.load(std::memory_order_relaxed);
    out.reclaimed = shared_->reclaimed.load(std::memory_order_relaxed);
    out.metadata_bytes = sizeof(Shared) + uint64_t(out.capacity) * sizeof(Shared::Slot) +
                         uint64_t(shared_->ready.capacity()) * sizeof(uint32_t);
    return out;
}
} // namespace qw3
