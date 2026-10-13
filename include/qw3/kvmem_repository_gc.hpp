#pragma once

#include "qw3/kvmem_gc.hpp"
#include "qw3/kvmem_repository.hpp"

#include <condition_variable>
#include <mutex>

namespace qw3 {

class NvmeKvTier;

// CPU-only policy worker for reusable repository slots. No Store/DeviceBackend
// pointer is accessed by the worker. Foreground metadata is published through
// versioned atomics; IO/checkpoint/GPU users own real repository leases.
class KvMemRepositoryGc {
public:
    enum class Mode { Off, Shadow, Delete };
    enum : uint32_t { Indexed = 1u << 12 };
    struct Config {
        KvMemGcConfig policy;
        uint32_t ready_low = 0, ready_high = 0;
        uint64_t recent_tokens = 0;
        // Long ingestion can span the whole repository within one request.
        // Let an unchanged, unprotected candidate mature after this much new
        // history as an alternative to request-step grace. Zero disables it.
        uint64_t candidate_ingest_tokens = 0;
        uint32_t tick_budget_us = 2000;
        Mode mode = Mode::Delete;
        bool background = true; // false only for deterministic host tests
    };
    struct Score {
        KvMemBlockHandle handle;
        double value = 0;
        bool selected = false;
    };
    struct Scores {
        uint64_t sequence = 0, step = 0, scorer_epoch = 0;
        std::vector<Score> entries;
    };
    struct Stats {
        uint64_t ticks = 0, examined = 0, retired = 0, reclaimed = 0;
        uint64_t protected_blocks = 0, deferred = 0, failures = 0;
        uint64_t raced_updates = 0, score_gaps = 0, stale_scores = 0;
        uint64_t shadow_candidates = 0, shadow_reuses = 0;
        uint64_t worker_ns = 0, max_tick_ns = 0;
        uint64_t metadata_bytes = 0;
        uint64_t policy_fallback_retired = 0;
        uint64_t incomplete_backing = 0, reader_conflicts = 0;
    };
    // Nonblocking, idempotent CPU cleanup of the complete generation-bound
    // raw-K/V/index/MTP bundle. A partial failure remains Releasing until retry.
    using Release = std::function<bool(const KvMemRepository::Record &)>;
    KvMemRepositoryGc(Config, std::shared_ptr<KvMemRepository>, Release);
    ~KvMemRepositoryGc();
    KvMemRepositoryGc(const KvMemRepositoryGc &) = delete;
    KvMemRepositoryGc &operator=(const KvMemRepositoryGc &) = delete;
    void stop();
    // Foreground only announces a released read/selection phase. CPU worker
    // performs all policy and cleanup; the caller never waits for GC.
    void notify_progress();
    void poll(); // manual mode only; the background thread is the sole poller
    uint64_t begin_step() { return step_.fetch_add(1, std::memory_order_acq_rel) + 1; }
    uint64_t step() const { return step_.load(std::memory_order_relaxed); }
    // Sole foreground publisher, with a live lease held throughout each call.
    // RawBacked|ValueBacked mean ALL configured main/MTP authorities are ready.
    // CPU-resident KV is eligible once durable; CpuCached is not a hard pin.
    void publish(const KvMemRepository::Lease &, uint32_t flags, bool used = false);
    void set_history_end(uint64_t end) { history_end_.store(end, std::memory_order_release); }
    void submit_scores(std::shared_ptr<const Scores>);
    Stats stats() const;
    const std::shared_ptr<KvMemRepository> &repository() const { return repository_; }
private:
    struct Metadata {
        std::atomic<uint64_t> version{0}, generation{0}, birth{0}, last_use{0}, uses{0};
        std::atomic<uint32_t> flags{0};
    };
    struct Snapshot {
        uint64_t version = 0, generation = 0, birth = 0, last_use = 0, uses = 0;
        uint32_t flags = 0;
    };
    struct History {
        uint64_t generation = 0, uses = 0, decay_step = 0;
        uint64_t score_step = 0, score_sequence = 0, candidate_step = 0, candidate_version = 0;
        uint64_t candidate_history_end = 0;
        uint64_t high_step = 0;
        double frequency = 0;
        std::array<double, 3> q{};
        std::array<uint64_t, 3> q_step{};
        uint32_t low_streak = 0, q_count = 0, q_cursor = 0;
        bool score_valid = false, high = false, shadow = false, pending = false;
        bool candidate_observed = false;
    };
    struct Candidate { KvMemBlockHandle handle; uint64_t version; double score; uint64_t last_use; uint32_t level; };
    bool snapshot(uint32_t, Snapshot &) const;
    bool protected_now(const KvMemRepository::Record &, const Snapshot &) const;
    History &refresh(uint32_t, const Snapshot &, uint64_t);
    void consume_scores();
    void release(KvMemBlockHandle);
    void tick();
    void run();
    Config config_;
    std::shared_ptr<KvMemRepository> repository_;
    Release release_;
    std::unique_ptr<Metadata[]> metadata_;
    std::vector<History> history_;
    std::vector<Candidate> candidates_;
    std::vector<Score> sorted_scores_;
    std::vector<uint64_t> score_seen_;
    std::shared_ptr<const Scores> scores_;
    uint64_t score_sequence_ = 0, scorer_epoch_ = 0;
    uint32_t cursor_ = 0, pending_ = 0, shadow_live_ = 0;
    uint32_t pressure_scanned_ = 0;
    bool replenishing_ = false;
    std::atomic<uint64_t> step_{0}, history_end_{0};
    std::atomic<uint64_t> submitted_sequence_{0};
    std::atomic<bool> stopping_{false};
    std::atomic<bool> wake_requested_{false};
    std::mutex sleep_mutex_;
    std::condition_variable wake_;
    std::thread worker_;
    std::atomic<uint64_t> ticks_{0}, examined_{0}, retired_{0}, reclaimed_{0};
    std::atomic<uint64_t> protected_{0}, deferred_{0}, failures_{0}, races_{0};
    std::atomic<uint64_t> gaps_{0}, stale_{0}, shadow_{0}, shadow_reuses_{0};
    std::atomic<uint64_t> worker_ns_{0}, max_tick_ns_{0};
    std::atomic<uint64_t> policy_fallback_retired_{0};
    std::atomic<uint64_t> incomplete_backing_{0}, reader_conflicts_{0};
};

// Complete the optional NVMe part of a reusable slot bundle. CPU raw-K/V/index
// arenas retain their bounded allocations; returning the repository slot makes
// those addresses reusable. Their foreground owner invalidates old generation
// metadata before writing a fresh reservation. Both tiers include MTP payloads
// when enabled. This callback has no executor/backend pointer or GPU operation.
KvMemRepositoryGc::Release kvmem_repository_storage_release(
    uint32_t capacity, NvmeKvTier *raw, NvmeKvTier *values);
} // namespace qw3
