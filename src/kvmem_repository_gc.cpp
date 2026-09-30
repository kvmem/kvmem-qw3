#include "qw3/kvmem_repository_gc.hpp"
#include "qw3/nvme_kv_tier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace qw3 {
KvMemRepositoryGc::Release kvmem_repository_storage_release(
        uint32_t capacity, NvmeKvTier *raw, NvmeKvTier *values) {
    if (!capacity || (raw && raw->direct_mapped()) ||
        (values && values->direct_mapped()))
        throw std::invalid_argument("KVMem reusable storage cannot own a direct-mapped archive");
    struct Pending {
        uint64_t generation = 0;
        NvmeAllocation raw, values;
        bool raw_captured = false, values_captured = false;
        bool raw_done = false, values_done = false;
    };
    return [raw, values, pending = std::vector<Pending>(capacity)](
            const KvMemRepository::Record &record) mutable {
        if (!record.handle.generation || record.handle.slot >= pending.size())
            throw std::invalid_argument("KVMem reclaim handle outside resource directory");
        auto &p = pending[record.handle.slot];
        if (p.generation > record.handle.generation) return true;
        if (p.generation != record.handle.generation) {
            p = {};
            p.generation = record.handle.generation;
        }
        auto release = [&](NvmeKvTier *tier, NvmeAllocation &a, bool &captured, bool &done) {
            if (done) return;
            if (!tier) { done = true; return; }
            if (!captured) {
                if (!tier->try_allocation(record.handle.slot, a)) return;
                captured = true;
            }
            // Cancelled writes may never have allocated this tier. The exact
            // allocator generation is retained across lock-busy/partial retry.
            done = tier->try_release_allocation(a);
        };
        release(raw, p.raw, p.raw_captured, p.raw_done);
        release(values, p.values, p.values_captured, p.values_done);
        return p.raw_done && p.values_done;
    };
}

namespace {
using Clock = std::chrono::steady_clock;
uint64_t age(uint64_t now, uint64_t then) { return now > then ? now - then : 0; }
}

KvMemRepositoryGc::KvMemRepositoryGc(Config c, std::shared_ptr<KvMemRepository> r, Release release)
    : config_(c), repository_(std::move(r)), release_(std::move(release)) {
    if (!repository_ || !release_ || !c.ready_low || c.ready_low > c.ready_high ||
        c.ready_high > repository_->capacity() || !c.policy.poll_ms ||
        !c.policy.scan_blocks || !c.policy.batch_blocks || !c.policy.half_life_steps ||
        !c.policy.low_observations || !c.tick_budget_us)
        throw std::invalid_argument("invalid bounded KVMem GC configuration");
#if defined(__linux__)
    if (c.policy.worker_cpu >= 0) {
        cpu_set_t allowed; CPU_ZERO(&allowed);
        if (c.policy.worker_cpu >= CPU_SETSIZE || sched_getaffinity(0, sizeof(allowed), &allowed) ||
            !CPU_ISSET(c.policy.worker_cpu, &allowed))
            throw std::invalid_argument("bounded KVMem GC CPU is outside affinity mask");
    }
#else
    if (c.policy.worker_cpu >= 0) throw std::invalid_argument("bounded KVMem GC affinity requires Linux");
#endif
    metadata_ = std::make_unique<Metadata[]>(repository_->capacity());
    history_.resize(repository_->capacity());
    candidates_.reserve(std::min(c.policy.scan_blocks, repository_->capacity()));
    sorted_scores_.reserve(repository_->capacity());
    score_seen_.resize(repository_->capacity());
    if (config_.background) worker_ = std::thread([this] { run(); });
}
KvMemRepositoryGc::~KvMemRepositoryGc() { stop(); }
void KvMemRepositoryGc::stop() {
    stopping_.store(true, std::memory_order_release);
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}
void KvMemRepositoryGc::poll() {
    if (config_.background) throw std::logic_error("bounded GC already has a background poller");
    if (!stopping_.load(std::memory_order_acquire)) tick();
}
void KvMemRepositoryGc::notify_progress() {
    if (config_.background && !wake_requested_.exchange(true, std::memory_order_acq_rel))
        wake_.notify_one();
}
void KvMemRepositoryGc::publish(const KvMemRepository::Lease &lease, uint32_t flags, bool used) {
    if (!lease) throw std::invalid_argument("bounded GC publication requires a live lease");
    auto verified = repository_->try_pin(lease.handle());
    if (!verified || &verified.record() != &lease.record())
        throw std::invalid_argument("bounded GC publication has a stale/foreign lease");
    auto &m = metadata_[lease.handle().slot];
    const uint64_t generation = lease.handle().generation;
    const bool fresh = m.generation.load() != generation;
    if (!fresh && !used && m.flags.load() == flags) return;
    // All metadata fields are atomic and sequentially consistent. The version
    // prevents observing a mix of two publications (including slot reuse).
    m.version.fetch_add(1);
    if (fresh) {
        m.generation.store(generation);
        m.birth.store(step()); m.last_use.store(step()); m.uses.store(step() ? 1 : 0);
    }
    if (used && m.last_use.load() < step()) { m.last_use.store(step()); m.uses.fetch_add(1); }
    m.flags.store(flags);
    m.version.fetch_add(1);
}
bool KvMemRepositoryGc::snapshot(uint32_t slot, Snapshot &s) const {
    const auto &m = metadata_[slot];
    s.version = m.version.load();
    if (s.version & 1) return false;
    s.generation = m.generation.load(); s.birth = m.birth.load();
    s.last_use = m.last_use.load(); s.uses = m.uses.load(); s.flags = m.flags.load();
    return s.version == m.version.load();
}
bool KvMemRepositoryGc::protected_now(const KvMemRepository::Record &r, const Snapshot &m) const {
    constexpr uint32_t required = KvMemGc::Sealed | KvMemGc::RawBacked | KvMemGc::ValueBacked | Indexed;
    constexpr uint32_t pinned = KvMemGc::Gpu | KvMemGc::Io | KvMemGc::Mandatory | KvMemGc::Anchor;
    const uint64_t end = history_end_.load(std::memory_order_acquire);
    const uint64_t recent_begin = end > config_.recent_tokens ? end - config_.recent_tokens : 0;
    return m.generation != r.handle.generation || (m.flags & required) != required ||
           (m.flags & pinned) || (config_.recent_tokens && r.token_end() > recent_begin);
}
KvMemRepositoryGc::History &KvMemRepositoryGc::refresh(uint32_t slot, const Snapshot &m, uint64_t now) {
    auto &h = history_[slot];
    if (h.generation != m.generation) {
        if (h.pending) --pending_;
        if (h.shadow) --shadow_live_;
        h = {}; h.generation = m.generation; h.decay_step = now;
    }
    h.frequency *= std::exp2(-double(age(now, h.decay_step)) / config_.policy.half_life_steps);
    h.decay_step = now;
    if (h.shadow && h.candidate_version != m.version) {
        if (m.uses > h.uses) ++shadow_reuses_;
        --shadow_live_; h.shadow = false;
    }
    if (m.uses != h.uses) {
        if (m.uses > h.uses) h.frequency += double(m.uses - h.uses);
        h.uses = m.uses;
    }
    if (h.candidate_version != m.version) {
        h.candidate_observed = false;
        h.candidate_version = m.version;
    }
    return h;
}
void KvMemRepositoryGc::submit_scores(std::shared_ptr<const Scores> scores) {
    if (!scores || !scores->sequence || scores->entries.size() > repository_->capacity())
        throw std::invalid_argument("bounded GC score frame exceeds slot capacity");
    const uint64_t sequence = scores->sequence;
    if (sequence <= submitted_sequence_.load(std::memory_order_acquire))
        throw std::invalid_argument("bounded GC score sequence must increase");
    std::atomic_store(&scores_, std::move(scores));
    submitted_sequence_.store(sequence, std::memory_order_release);
}
void KvMemRepositoryGc::consume_scores() {
    const auto scores = std::atomic_load(&scores_);
    if (!scores || scores->sequence <= score_sequence_) return;
    const bool gap = scores->sequence != score_sequence_ + 1 || scores->scorer_epoch != scorer_epoch_;
    if (gap) {
        ++gaps_;
        // Missing selection evidence never counts as a consecutive low score.
        for (auto &h : history_) h.low_streak = 0;
    }
    score_sequence_ = scores->sequence; scorer_epoch_ = scores->scorer_epoch;
    sorted_scores_.clear();
    for (const auto &score : scores->entries) {
        if (!std::isfinite(score.value) || !score.handle.generation ||
            score.handle.slot >= repository_->capacity()) { ++stale_; continue; }
        Snapshot m;
        if (!snapshot(score.handle.slot, m) || m.generation != score.handle.generation ||
            !repository_->try_pin(score.handle)) { ++stale_; continue; }
        if (score_seen_[score.handle.slot] == scores->sequence) { ++stale_; continue; }
        score_seen_[score.handle.slot] = scores->sequence;
        sorted_scores_.push_back(score);
    }
    // Runs only on a new score event in the CPU worker. The tick time cap below
    // is cooperative, not a real-time guarantee; sorting and release callbacks
    // are included in reported max_tick_ns rather than hidden from accounting.
    std::sort(sorted_scores_.begin(), sorted_scores_.end(), [](const auto &a, const auto &b) {
        return a.value < b.value;
    });
    const bool distinct = sorted_scores_.size() > 1 && sorted_scores_.front().value != sorted_scores_.back().value;
    const size_t selected = std::count_if(sorted_scores_.begin(), sorted_scores_.end(),
                                         [](const auto &s) { return s.selected; });
    const double cutoff = 1.0 - double(selected) / std::max<size_t>(1, sorted_scores_.size());
    for (size_t i = 0; i < sorted_scores_.size();) {
        size_t end = i + 1;
        while (end < sorted_scores_.size() && sorted_scores_[end].value == sorted_scores_[i].value) ++end;
        // Equal scores receive the same midpoint percentile, not slot order.
        const double heat = (double(i) + double(end) - 1) * 0.5 / std::max<size_t>(1, sorted_scores_.size() - 1);
        for (size_t j = i; j < end; ++j) {
            const auto &score = sorted_scores_[j];
            Snapshot m;
            if (!snapshot(score.handle.slot, m) || m.generation != score.handle.generation) { ++stale_; continue; }
            auto &h = refresh(score.handle.slot, m, step());
            if (h.score_sequence == scores->sequence) { ++stale_; continue; } // duplicate handle
            const bool fresh = h.score_valid && age(scores->step, h.score_step) <= config_.policy.half_life_steps;
            const bool new_step = h.score_sequence == 0 || scores->step > h.score_step;
            const bool near = selected > 0 && heat >= cutoff - 0.05;
            if (gap || !fresh || h.score_sequence + 1 != scores->sequence) h.low_streak = 0;
            if (distinct && heat < 0.2 && !near && !score.selected) {
                if (new_step && h.low_streak < UINT32_MAX) ++h.low_streak;
            } else h.low_streak = 0;
            if (gap || !fresh) { h.q_count = 0; h.q_cursor = 0; }
            if (distinct) {
                if (new_step || !h.q_count) {
                    h.q_cursor = h.q_count ? (h.q_cursor + 1) % h.q.size() : 0;
                    h.q_count = std::min<uint32_t>(h.q_count + 1, h.q.size());
                    h.q[h.q_cursor] = heat; h.q_step[h.q_cursor] = scores->step;
                } else h.q[h.q_cursor] = std::max(h.q[h.q_cursor], heat);
            }
            if (distinct && (heat >= 0.8 || near)) { h.high = true; h.high_step = scores->step; }
            h.score_valid = distinct; h.score_step = scores->step; h.score_sequence = scores->sequence;
        }
        i = end;
    }
}
void KvMemRepositoryGc::release(KvMemBlockHandle handle) {
    auto &h = history_[handle.slot];
    if (h.shadow) { h.shadow = false; --shadow_live_; }
    if (h.generation != handle.generation) {
        if (h.pending) --pending_;
        h = {}; h.generation = handle.generation;
    }
    if (!h.pending) { h.pending = true; ++pending_; }
    try {
        const auto result = repository_->try_reclaim(handle, release_);
        if (result == KvMemRepository::ReclaimResult::Reclaimed) {
            ++reclaimed_; h.pending = false; --pending_;
        } else if (result == KvMemRepository::ReclaimResult::NotRetired ||
                   result == KvMemRepository::ReclaimResult::StaleHandle) {
            h.pending = false; --pending_;
        } else ++deferred_;
    } catch (...) {
        ++failures_; // Releasing is retained; retry the same bundle next scan
    }
}
void KvMemRepositoryGc::tick() {
    const auto start = Clock::now();
    const auto deadline = start + std::chrono::microseconds(config_.tick_budget_us);
    if (config_.mode != Mode::Off) consume_scores();
    const uint64_t now = step();
    const uint32_t ready = repository_->ready_slots_for_reclaimer() + shadow_live_;
    if (ready < config_.ready_low) replenishing_ = true;
    if (ready >= config_.ready_high) { replenishing_ = false; pressure_scanned_ = 0; }
    candidates_.clear();
    const uint32_t scan = std::min(config_.policy.scan_blocks, repository_->capacity());
    uint32_t releases = 0;
    for (uint32_t i = 0; i < scan; ++i) {
        if (i && Clock::now() >= deadline) break;
        const uint32_t slot = cursor_;
        cursor_ = cursor_ + 1 == repository_->capacity() ? 0 : cursor_ + 1;
        if (replenishing_ && pressure_scanned_ < repository_->capacity()) ++pressure_scanned_;
        ++examined_;
        const auto retired = repository_->retired_at(slot);
        if (retired.generation) {
            if (releases < config_.policy.batch_blocks) { release(retired); ++releases; }
            continue;
        }
        if (config_.mode == Mode::Off) continue;
        auto lease = repository_->try_pin_slot(slot);
        Snapshot m;
        if (!lease || !snapshot(slot, m) || m.generation != lease.handle().generation) continue;
        auto &h = refresh(slot, m, now);
        if (repository_->ready_slots_for_reclaimer() + shadow_live_ < config_.ready_low) replenishing_ = true;
        if (protected_now(lease.record(), m)) {
            constexpr uint32_t required = KvMemGc::Sealed | KvMemGc::RawBacked |
                KvMemGc::ValueBacked | Indexed;
            if ((m.flags & required) != required) ++incomplete_backing_;
            ++protected_; h.candidate_observed = false; continue;
        }
        KvMemGcFeatures f;
        f.birth = m.birth; f.last_use = m.last_use; f.step = now;
        f.frequency = h.frequency;
        f.score_valid = h.score_valid && age(now, h.score_step) <= config_.policy.half_life_steps;
        f.high_score = h.high && age(now, h.high_step) <= config_.policy.half_life_steps;
        f.low_streak = h.low_streak;
        for (uint32_t j = 0; j < h.q_count; ++j)
            if (age(now, h.q_step[j]) <= config_.policy.half_life_steps)
                f.semantic_heat = std::max(f.semantic_heat, h.q[j] *
                    std::exp2(-double(age(now, h.q_step[j])) / config_.policy.half_life_steps));
        f.duplicate = m.flags & KvMemGc::Duplicate; f.superseded = m.flags & KvMemGc::Superseded;
        const auto decision = kvmem_gc_score(f, config_.policy);
        if (decision.level >= 4) { h.candidate_observed = false; continue; }
        const uint64_t history_end = history_end_.load(std::memory_order_acquire);
        if (!h.candidate_observed) {
            h.candidate_observed = true;
            h.candidate_step = now;
            h.candidate_history_end = history_end;
        }
        const bool ingest_grace = config_.candidate_ingest_tokens &&
            age(history_end, h.candidate_history_end) >= config_.candidate_ingest_tokens;
        if (!ingest_grace && age(now, h.candidate_step) < config_.policy.candidate_steps) continue;
        // Observe candidates before pressure, within the same bounded scan.
        // Starting their multi-step grace only below the low watermark can
        // consume the entire reserve before the first victim is eligible.
        // Readiness never itself authorizes deletion above the watermark.
        if (!replenishing_) continue;
        // Recency/frequency/query heat are retention preferences. They must
        // not permanently protect the entire finite repository after repeated
        // selections. Level 3 sorts after every ordinary cold candidate and
        // can fill the reserve only if those candidates are insufficient.
        // Actual GPU/IO/checkpoint/mandatory/anchor protections remain strict.
        candidates_.push_back({lease.handle(), m.version, decision.score, m.last_use, decision.level});
    }
    std::sort(candidates_.begin(), candidates_.end(), [](const auto &a, const auto &b) {
        if (a.score != b.score) return a.score < b.score;
        return a.last_use < b.last_use;
    });
    uint32_t picked = 0;
    for (const auto &candidate : candidates_) {
        if (picked >= config_.policy.batch_blocks || Clock::now() >= deadline) break;
        if (repository_->ready_slots_for_reclaimer() + pending_ + shadow_live_ >= config_.ready_high) break;
        // A partial scan containing hot blocks is insufficient evidence that
        // ordinary cold candidates elsewhere cannot satisfy the reserve.
        if (candidate.level == 3 && pressure_scanned_ < repository_->capacity()) continue;
        auto lease = repository_->try_pin(candidate.handle);
        Snapshot m;
        if (!lease || !snapshot(candidate.handle.slot, m) || m.version != candidate.version ||
            protected_now(lease.record(), m)) { ++races_; continue; }
        auto &h = history_[candidate.handle.slot];
        if (config_.mode == Mode::Shadow) {
            if (!h.shadow) { h.shadow = true; ++shadow_; ++shadow_live_; }
            ++picked; continue;
        }
        if (!repository_->retire_if_only_reader(lease)) { ++protected_; ++reader_conflicts_; continue; }
        // A complete foreground update can race between the snapshot and CAS.
        // Once Retired, no fresh publisher can pin it. Rescue under our own
        // lease if any metadata changed; never free on the stale decision.
        Snapshot after;
        if (!snapshot(candidate.handle.slot, after) || after.version != m.version ||
            submitted_sequence_.load(std::memory_order_acquire) != score_sequence_ ||
            protected_now(lease.record(), after)) {
            repository_->rescue(lease); ++races_; continue;
        }
        ++retired_; ++picked;
        if (candidate.level == 3) ++policy_fallback_retired_;
        lease.reset();
        if (releases < config_.policy.batch_blocks) { release(candidate.handle); ++releases; }
        else if (!h.pending) { h.pending = true; ++pending_; }
    }
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    worker_ns_.fetch_add(static_cast<uint64_t>(ns));
    max_tick_ns_.store(std::max(max_tick_ns_.load(), static_cast<uint64_t>(ns)));
    ++ticks_;
}
void KvMemRepositoryGc::run() {
#if defined(__linux__)
    if (config_.policy.worker_cpu >= 0) {
        cpu_set_t cpus; CPU_ZERO(&cpus); CPU_SET(config_.policy.worker_cpu, &cpus);
        if (pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus)) ++failures_;
    }
    // This affects only the worker thread, not the serving process/GPU owner.
    (void)setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), 10);
#endif
    while (!stopping_.load(std::memory_order_acquire)) {
        try { tick(); } catch (...) { ++failures_; }
        std::unique_lock<std::mutex> lock(sleep_mutex_);
        wake_.wait_for(lock, std::chrono::milliseconds(config_.policy.poll_ms),
                       [this] { return stopping_.load(std::memory_order_acquire) ||
                           wake_requested_.exchange(false, std::memory_order_acq_rel); });
    }
}
KvMemRepositoryGc::Stats KvMemRepositoryGc::stats() const {
    return {ticks_.load(), examined_.load(), retired_.load(), reclaimed_.load(),
        protected_.load(), deferred_.load(), failures_.load(), races_.load(), gaps_.load(), stale_.load(),
        shadow_.load(), shadow_reuses_.load(), worker_ns_.load(), max_tick_ns_.load(),
        sizeof(*this) + uint64_t(repository_->capacity()) * sizeof(Metadata) +
        uint64_t(history_.capacity()) * sizeof(History) + uint64_t(candidates_.capacity()) * sizeof(Candidate) +
        uint64_t(sorted_scores_.capacity()) * sizeof(Score) + uint64_t(score_seen_.capacity()) * sizeof(uint64_t),
        policy_fallback_retired_.load(), incomplete_backing_.load(), reader_conflicts_.load()};
}
} // namespace qw3
