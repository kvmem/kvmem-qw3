#include "backend.hpp"
#include "env_flags.hpp"
#include "kvmem_session.hpp"
#include "qwen_executor.hpp"
#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/tokenizer.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace qw3 {
namespace {

double wall_seconds() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

void apply_token_penalties(std::vector<float> &logits,
                           const std::unordered_map<uint32_t, uint32_t> &seen,
                           float presence_penalty,
                           float repetition_penalty) {
    const bool use_presence = presence_penalty != 0.0f;
    const bool use_repetition = repetition_penalty > 0.0f && repetition_penalty != 1.0f;
    if (!use_presence && !use_repetition) return;
    for (const auto &kv : seen) {
        const uint32_t token = kv.first;
        if (token >= logits.size()) continue;
        float &logit = logits[token];
        if (use_repetition) {
            logit = logit >= 0.0f ? logit / repetition_penalty
                                  : logit * repetition_penalty;
        }
        if (use_presence) {
            logit -= presence_penalty;
        }
    }
}

// Full-vocab probability vector that token sampling draws from: softmax(logits/temp)
// with top-k/top-p/min-p truncation applied and renormalized over the kept set.
// Entries outside the kept set are 0. Penalties are NOT applied here — callers
// penalize the logits first (matching apply_token_penalties usage). This is the
// single source of truth for the sampling distribution, shared by the non-MTP
// sampler (sample_token) and the MTP speculative accept test, so MTP under temp>0
// reproduces exactly the same distribution as plain decode.
std::vector<float> sampling_distribution(const std::vector<float> &logits,
                                         float temp, float top_p, int top_k,
                                         float min_p) {
    const int n = static_cast<int>(logits.size());
    std::vector<float> out(n > 0 ? static_cast<size_t>(n) : 0, 0.0f);
    if (n <= 0) return out;
    // softmax(logits / temp), numerically stabilized by max subtraction.
    const float inv_t = 1.0f / (temp > 0.0f ? temp : 1.0f);
    float maxv = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > maxv) maxv = logits[i];
    std::vector<std::pair<float, int>> probs(n);
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const float p = std::exp((logits[i] - maxv) * inv_t);
        probs[i] = {p, i};
        sum += p;
    }
    const float norm = static_cast<float>(1.0 / (sum > 0.0 ? sum : 1.0));
    for (auto &pr : probs) pr.first *= norm;
    const bool need_sort =
        (top_k > 0 && top_k < n) ||
        (top_p < 1.0f && top_p > 0.0f) ||
        (min_p > 0.0f);
    if (need_sort) {
        std::sort(probs.begin(), probs.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
    }
    // Truncation order mirrors the original sample_token: top_k, then min_p, then
    // nucleus (top_p). The list is sorted descending, so each filter keeps a prefix.
    size_t keep = probs.size();
    if (top_k > 0 && top_k < static_cast<int>(keep)) keep = static_cast<size_t>(top_k);
    if (min_p > 0.0f && keep > 0) {
        const float cutoff = probs.front().first * min_p;
        size_t k = 0;
        while (k < keep && probs[k].first >= cutoff) ++k;
        keep = k;
    }
    if (top_p < 1.0f && top_p > 0.0f) {
        double cum = 0.0;
        size_t k = keep;
        for (size_t i = 0; i < keep; ++i) {
            cum += probs[i].first;
            if (cum >= top_p) { k = i + 1; break; }
        }
        keep = k;
    }
    double kept_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) kept_sum += probs[i].first;
    if (kept_sum <= 0.0) {
        // Degenerate (e.g. min_p removed everything): collapse to the top token.
        if (!probs.empty()) out[probs.front().second] = 1.0f;
        return out;
    }
    const float kn = static_cast<float>(1.0 / kept_sum);
    for (size_t i = 0; i < keep; ++i) out[probs[i].second] = probs[i].first * kn;
    return out;
}

// Draw an index from a (full-vocab) probability vector. Nonzero entries are
// collected first so discrete_distribution stays cheap even over a 150K vocab.
int32_t sample_from(const std::vector<float> &probs, std::mt19937_64 &rng) {
    std::vector<double> w;
    std::vector<int> idx;
    for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
        if (probs[i] > 0.0f) { w.push_back(probs[i]); idx.push_back(i); }
    }
    if (w.empty()) return -1;
    std::discrete_distribution<int> dist(w.begin(), w.end());
    return idx[dist(rng)];
}

// Host-side token sampler over the full fp32 vocab logits. temp<=0 is greedy.
// Otherwise draws from sampling_distribution(). Kept on host because
// copy_last_logits() already round-trips logits; sampling adds no device work.
int32_t sample_token(const std::vector<float> &logits, float temp, float top_p,
                     int top_k, float min_p,
                     std::mt19937_64 &rng) {
    const int n = static_cast<int>(logits.size());
    if (n <= 0) return -1;
    if (temp <= 0.0f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < n; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    return sample_from(sampling_distribution(logits, temp, top_p, top_k, min_p), rng);
}

struct SpecAcceptResult {
    uint32_t accepted = 0;     // number of leading drafts accepted
    int32_t extra_token = -1;  // residual (on reject) or bonus (on full accept)
};

// Distribution-lossless speculative-sampling accept test for an argmax-fed draft
// chain. Because each draft was produced greedily, the proposal q_i is a point mass
// at draft_i, so the classic rejection test collapses to: accept draft_i with
// probability p_i(draft_i) where p_i = sampling_distribution(penalized target row i);
// on reject, emit a draw from the residual (p_i with draft_i removed, renormalized)
// and stop; if all drafts accept, emit a bonus draw from the final target row.
// Penalties are applied per row against the running committed prefix (seen + the
// drafts accepted so far). This is provably equal in distribution to drawing each
// committed token directly from the target distribution p — the same distribution
// the non-MTP sampler draws from — so MTP temp>0 output matches plain temp>0.
SpecAcceptResult speculative_accept_pointmass(
        const std::vector<std::vector<float>> &target_rows,  // size == drafts.size()+1
        const std::vector<uint32_t> &drafts,
        float temp, float top_p, int top_k, float min_p,
        float presence_penalty, float repetition_penalty,
        const std::unordered_map<uint32_t, uint32_t> &seen,
        std::mt19937_64 &rng) {
    SpecAcceptResult r;
    std::unordered_map<uint32_t, uint32_t> ctx = seen;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const size_t n = drafts.size();
    // Target distribution for a row. temp>0 is the truncated/renormalized
    // sampling distribution; temp<=0 collapses to a point mass at the argmax so
    // the accept test reproduces greedy decoding (now over the penalized logits),
    // letting penalties apply correctly even when sampling is off.
    auto target_dist = [&](const std::vector<float> &row) -> std::vector<float> {
        if (temp > 0.0f)
            return sampling_distribution(row, temp, top_p, top_k, min_p);
        std::vector<float> p(row.size(), 0.0f);
        if (!row.empty()) {
            int best = 0;
            float bv = row[0];
            for (int i = 1; i < static_cast<int>(row.size()); ++i)
                if (row[i] > bv) { bv = row[i]; best = i; }
            p[best] = 1.0f;
        }
        return p;
    };
    for (size_t i = 0; i < n && i < target_rows.size(); ++i) {
        std::vector<float> row = target_rows[i];  // copy; penalized in place
        apply_token_penalties(row, ctx, presence_penalty, repetition_penalty);
        std::vector<float> p = target_dist(row);
        const uint32_t d = drafts[i];
        const float accept_prob = (d < p.size()) ? p[d] : 0.0f;
        if (unit(rng) < static_cast<double>(accept_prob)) {
            ++ctx[d];
            ++r.accepted;
            continue;
        }
        // Reject: draw from the residual (p with the rejected draft removed).
        if (d < p.size()) p[d] = 0.0f;
        int32_t resid = sample_from(p, rng);
        r.extra_token = resid >= 0 ? resid : static_cast<int32_t>(d);
        return r;  // accepted == i
    }
    // All drafts accepted: bonus draw from the final (post-chain) target row.
    if (n < target_rows.size()) {
        std::vector<float> row = target_rows[n];
        apply_token_penalties(row, ctx, presence_penalty, repetition_penalty);
        std::vector<float> p = target_dist(row);
        r.extra_token = sample_from(p, rng);
    }
    return r;
}

// True iff a request needs its per-row target logits pulled to host on the MTP
// verify path: sampling (temp>0) OR active presence/repetition penalties. When
// false the greedy argmax-equality accept loop runs and is byte-identical to the
// pre-sampling MTP. Shared by the single-request and CB accept sites so both gate
// identically.
inline bool mtp_options_need_logits(float temperature,
                                    float presence_penalty,
                                    float repetition_penalty) {
    if (temperature > 0.0f) return true;
    if (presence_penalty != 0.0f) return true;
    if (repetition_penalty > 0.0f && repetition_penalty != 1.0f) return true;
    return false;
}

bool native_prefill_flashinfer_effective() {
    if (env_value_is_ci("QW3_PREFILL_ATTN", "flashinfer")) return true;
    const char *raw = std::getenv("QW3_PREFILL_ATTN");
    if (raw && *raw) return false;
#ifdef QW3_ENABLE_FLASHINFER
    return true;
#else
    return false;
#endif
}

bool native_decode_flashinfer_effective() {
    if (env_value_is_ci("QW3_DECODE_ATTN", "flashinfer")) return true;
    const char *raw = std::getenv("QW3_DECODE_ATTN");
    if (raw && *raw) return false;
#ifdef QW3_ENABLE_FLASHINFER
    return true;
#else
    return false;
#endif
}

bool prefill_trace_enabled() {
    return env_flag_enabled("QW3_PREFILL_TRACE");
}

bool decode_trace_enabled() {
    return env_flag_enabled("QW3_DECODE_TRACE");
}

bool mtp_verify_trace_enabled() {
    return env_flag_enabled("QW3_MTP_VERIFY_TRACE");
}

bool mtp_token_trace_enabled() {
    return env_flag_enabled("QW3_MTP_TOKEN_TRACE");
}

bool mtp_phase_sync_enabled() {
    return env_flag_enabled("QW3_MTP_PHASE_SYNC");
}

bool mtp_device_draft_chain_enabled() {
    return env_flag_enabled("QW3_MTP_DEVICE_DRAFT_CHAIN", true);
}

bool mtp_verify_sequential_enabled() {
    const char *raw = std::getenv("QW3_MTP_VERIFY");
    if (raw && *raw) {
        const std::string value = env_lower_ascii(raw);
        if (value == "sequential" || value == "serial") return true;
        if (value == "batched" || value == "batch") return false;
        if (env_disabled_value(value)) return false;
        throw std::runtime_error("invalid QW3_MTP_VERIFY: " + value);
    }
    return false;
}

bool mtp_flashinfer_verify_uses_non_target_path() {
    if (!native_decode_flashinfer_effective()) return false;
    if (env_value_is_ci("QW3_FLASHINFER_VERIFY_DECODE", "ported") ||
        env_value_is_ci("QW3_FLASHINFER_VERIFY_DECODE", "splitk") ||
        env_value_is_ci("QW3_MTP_VERIFY_ATTENTION", "ported") ||
        env_value_is_ci("QW3_MTP_VERIFY_ATTENTION", "splitk")) {
        return true;
    }
    if (env_uint32_or("QW3_FLASHINFER_VERIFY_DECODE_MAX_BATCH", 8) == 0 &&
        native_prefill_flashinfer_effective()) {
        return true;
    }
    return false;
}

enum class MtpTransactionalReplayMode {
    Off,
    Rejects,
    All,
};

MtpTransactionalReplayMode mtp_transactional_replay_mode() {
    const char *raw = std::getenv("QW3_MTP_TRANSACTIONAL_REPLAY");
    if (raw && *raw) {
        const std::string value = env_lower_ascii(raw);
        if (env_disabled_value(value)) return MtpTransactionalReplayMode::Off;
        if (value == "reject" || value == "rejects" || value == "rollback") {
            return MtpTransactionalReplayMode::Rejects;
        }
        return MtpTransactionalReplayMode::All;
    }
    return mtp_flashinfer_verify_uses_non_target_path()
        ? MtpTransactionalReplayMode::All
        : MtpTransactionalReplayMode::Off;
}

bool decode_as_batch_enabled() {
    return env_flag_enabled("QW3_DECODE_AS_BATCH");
}

uint32_t continuous_batching_max_active() {
    return std::max<uint32_t>(1, env_uint32_or("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", 2));
}

uint32_t continuous_batching_max_pending() {
    return std::max<uint32_t>(1, env_uint32_or("QW3_CONTINUOUS_BATCHING_MAX_PENDING", 128));
}

uint32_t continuous_batching_prefill_burst(uint32_t max_active) {
    return std::max<uint32_t>(
        1, env_uint32_or("QW3_CONTINUOUS_BATCHING_PREFILL_BURST", max_active));
}

uint32_t continuous_batching_active_prefill_burst() {
    return std::max<uint32_t>(
        1, env_uint32_or("QW3_CONTINUOUS_BATCHING_ACTIVE_PREFILL_BURST", 1));
}

uint32_t continuous_batching_admission_wait_us() {
    return env_uint32_or("QW3_CONTINUOUS_BATCHING_ADMISSION_WAIT_US", 1000);
}

uint64_t continuous_batching_max_total_tokens(uint32_t ctx_size, uint32_t max_active) {
    (void)max_active;
    const uint64_t default_budget = static_cast<uint64_t>(ctx_size);
    const uint64_t configured =
        env_uint64_or("QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS", default_budget);
    return configured == 0 ? std::numeric_limits<uint64_t>::max() : configured;
}

bool continuous_batching_trace_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_TRACE");
}

bool continuous_batching_timing_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_TIMING");
}

bool continuous_batching_body_batch_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_BODY_BATCH", true);
}

bool continuous_batching_lm_head_batch_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_LM_HEAD_BATCH", true);
}

bool continuous_batching_recurrent_batch_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_RECURRENT_BATCH", true);
}

bool continuous_batching_prefill_batch_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_PREFILL_BATCH", true);
}

bool continuous_batching_prefill_pack_recurrent_state_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_PREFILL_PACK_RECURRENT_STATE");
}

bool continuous_batching_ragged_prefill_executor_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_RAGGED_PREFILL_EXECUTOR",
                            true);
}

uint32_t continuous_batching_mtp_ragged_verify_min_tokens() {
    return std::max<uint32_t>(
        1, env_uint32_or("QW3_CONTINUOUS_BATCHING_MTP_RAGGED_VERIFY_MIN_TOKENS",
                         16));
}

bool continuous_batching_mtp_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING_MTP", true);
}

bool continuous_mtp_batched_draft_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_MTP_BATCHED_DRAFT", true);
}

bool continuous_mtp_layered_verify_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_MTP_LAYERED_VERIFY");
}

uint32_t continuous_batching_ragged_prefill_min_tokens() {
    return std::max<uint32_t>(
        1, env_uint32_or("QW3_CONTINUOUS_BATCHING_RAGGED_PREFILL_MIN_TOKENS",
                         512));
}

bool prefix_cache_enabled() {
    return env_flag_enabled("QW3_PREFIX_CACHE");
}

// Cap on total physical KV pages pinned by the prefix cache. 0 = unlimited
// (bounded only by the pool). Used to trigger LRU eviction.
uint32_t prefix_cache_max_pages() {
    return env_uint32_or("QW3_PREFIX_CACHE_MAX_PAGES", 0);
}

bool prefix_cache_trace_enabled() {
    return env_flag_enabled("QW3_PREFIX_CACHE_TRACE");
}

class GlobalKvPagePool final : public KvPhysicalPageAllocator {
public:
    GlobalKvPagePool(uint32_t total_pages, uint32_t page_size)
        : total_pages_(total_pages), page_size_(page_size) {
        free_pages_.reserve(total_pages_);
        for (uint32_t i = 0; i < total_pages_; ++i) {
            free_pages_.push_back(static_cast<int32_t>(total_pages_ - 1U - i));
        }
        page_pin_refcount_.assign(total_pages_, 0);
    }

    // Installed by the backend's prefix cache. On free-stack exhaustion the
    // pool invokes this to reclaim pinned (refcount==0 entry) cache pages back
    // onto the free stack. Must return the number of pages freed (0 = nothing
    // evictable). The callback runs WITHOUT the pool mutex held (it itself
    // calls unpin_pages/release_physical_pages, which take the lock).
    void set_evict_callback(std::function<uint32_t()> cb) {
        evict_cb_ = std::move(cb);
    }

    int32_t allocate_physical_page() override {
        for (;;) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!free_pages_.empty()) {
                    const int32_t page = free_pages_.back();
                    free_pages_.pop_back();
                    ++used_pages_;
                    return page;
                }
            }
            // Free stack empty: ask the prefix cache to evict the LRU entry
            // whose refcount is 0, then retry. A single eviction may reclaim
            // ZERO physical pages when the victim shares all its pages with a
            // longer cached entry (extension/multi-turn). The evict callback
            // returns the number of ENTRIES it evicted (progress), not pages,
            // so we keep evicting until a page actually frees up or there is
            // nothing left to evict. Each iteration drops one entry's pin, so
            // the entry that uniquely owns the shared region eventually frees.
            if (!evict_cb_ || evict_cb_() == 0) break;
        }
        std::lock_guard<std::mutex> lk(mu_);
        throw std::runtime_error(
            "global KV page pool exhausted: free=0 total=" +
            std::to_string(total_pages_) +
            " page_size=" + std::to_string(page_size_));
    }

    void release_physical_pages(const std::vector<int32_t> &pages) override {
        if (pages.empty()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            // Defense in depth: a page still pinned by a prefix-cache entry
            // must never re-enter the free stack (the owning executor's
            // `owned` flag should already prevent this).
            if (page_pin_refcount_[page] > 0) continue;
            free_pages_.push_back(page);
            if (used_pages_ > 0) --used_pages_;
        }
    }

    // Pin pages held by a prefix-cache entry so they can never be handed out
    // by allocate_physical_page or returned to the free stack while live.
    void pin_pages(const std::vector<int32_t> &pages) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            ++page_pin_refcount_[page];
        }
    }

    // Drop the cache's pin on pages (called just before releasing them at
    // eviction). Does NOT return them to the free stack; pair with
    // release_physical_pages.
    void unpin_pages(const std::vector<int32_t> &pages) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            if (page_pin_refcount_[page] > 0) --page_pin_refcount_[page];
        }
    }

    uint32_t free_pages() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<uint32_t>(free_pages_.size());
    }

    uint32_t used_pages() const override {
        std::lock_guard<std::mutex> lk(mu_);
        return used_pages_;
    }

    uint32_t total_pages() const override { return total_pages_; }
    uint32_t page_size() const { return page_size_; }

private:
    uint32_t total_pages_ = 0;
    uint32_t page_size_ = 0;
    mutable std::mutex mu_;
    std::vector<int32_t> free_pages_;
    std::vector<uint16_t> page_pin_refcount_;
    std::function<uint32_t()> evict_cb_;
    uint32_t used_pages_ = 0;
};

bool mtp_trace_enabled() {
    return env_flag_enabled("QW3_MTP_TRACE");
}

bool mtp_prefix_enabled(const EngineOptions &options) {
    return options.native_mtp_prefix || env_flag_enabled("QW3_MTP_PREFIX");
}

bool mtp_speculate_enabled(const EngineOptions &options) {
    return options.native_mtp_speculate || env_flag_enabled("QW3_MTP_SPECULATE");
}

bool mtp_skip_verify_logits_copy_enabled() {
    return env_flag_enabled("QW3_MTP_SKIP_VERIFY_LOGITS_COPY", true);
}

bool mtp_rebuild_accepted_prefix_enabled() {
    return env_flag_enabled("QW3_MTP_REBUILD_ACCEPTED_PREFIX", true);
}

bool mtp_single_token_replay_enabled() {
    return env_flag_enabled("QW3_MTP_SINGLE_TOKEN_REPLAY", true);
}

bool mtp_reuse_current_prefix_enabled() {
    return env_flag_enabled("QW3_MTP_REUSE_CURRENT_PREFIX", true);
}

bool mtp_prefix1_state_enabled() {
    return env_flag_enabled("QW3_MTP_PREFIX1_STATE", true);
}

uint32_t mtp_prefix_rebuild_batch_min_tokens() {
    return std::max<uint32_t>(1, env_uint32_or("QW3_MTP_PREFIX_REBUILD_BATCH_MIN", 1));
}

uint32_t mtp_state_checkpoint_count(uint32_t chain_len) {
    const char *raw = std::getenv("QW3_MTP_STATE_CHECKPOINTS");
    if (!raw || !*raw) {
        return mtp_prefix1_state_enabled() ? chain_len : 0;
    }
    const std::string value = env_lower_ascii(raw);
    if (env_disabled_value(value) || value == "off" || value == "none") {
        return 0;
    }
    if (value == "all") {
        return chain_len;
    }
    size_t pos = 0;
    const unsigned long parsed = std::stoul(value, &pos);
    if (pos != value.size() || parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid QW3_MTP_STATE_CHECKPOINTS: " + value);
    }
    return std::min<uint32_t>(static_cast<uint32_t>(parsed), chain_len);
}

uint32_t mtp_reject_budget(size_t prompt_tokens) {
    (void)prompt_tokens;
    constexpr uint32_t kDefaultBudget = std::numeric_limits<uint32_t>::max();
    const char *raw = std::getenv("QW3_MTP_REJECT_BUDGET");
    if (!raw || !*raw) return kDefaultBudget;
    const std::string value = env_lower_ascii(raw);
    if (env_disabled_value(value) || value == "off" || value == "none") {
        return kDefaultBudget;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (*end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) return kDefaultBudget;
    return static_cast<uint32_t>(parsed);
}

uint32_t mtp_prefix_max_prompt_tokens() {
    constexpr uint32_t kDefaultMaxPrompt = std::numeric_limits<uint32_t>::max();
    const char *raw = std::getenv("QW3_MTP_PREFIX_MAX_PROMPT");
    if (!raw || !*raw) return kDefaultMaxPrompt;
    const std::string value = env_lower_ascii(raw);
    if (env_disabled_value(value) || value == "off" || value == "none") {
        return kDefaultMaxPrompt;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (*end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) return kDefaultMaxPrompt;
    return static_cast<uint32_t>(parsed);
}

uint32_t mtp_trace_chain_len(const EngineOptions &options) {
    const uint32_t configured = options.native_mtp_chain > 0
        ? static_cast<uint32_t>(options.native_mtp_chain)
        : 1u;
    return std::max<uint32_t>(1, env_uint32_or("QW3_MTP_CHAIN", configured));
}

uint32_t mtp_safe_chain_max() {
    constexpr uint32_t kDefaultMaxChain = std::numeric_limits<uint32_t>::max();
    const char *raw = std::getenv("QW3_MTP_SAFE_MAX_CHAIN");
    if (!raw || !*raw) return kDefaultMaxChain;
    const std::string value = env_lower_ascii(raw);
    if (env_disabled_value(value) || value == "off" || value == "none") {
        return kDefaultMaxChain;
    }
    size_t pos = 0;
    const unsigned long parsed = std::stoul(value, &pos);
    if (pos != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid QW3_MTP_SAFE_MAX_CHAIN: " + value);
    }
    return static_cast<uint32_t>(parsed);
}

bool mtp_adaptive_policy_requested() {
    const char *raw = std::getenv("QW3_MTP_POLICY");
    if (!raw || !*raw) return false;
    const std::string value = env_lower_ascii(raw);
    if (env_disabled_value(value) || value == "fixed" || value == "static") {
        return false;
    }
    if (value == "adaptive" || value == "auto") {
        return true;
    }
    throw std::runtime_error("invalid QW3_MTP_POLICY: " + value);
}

bool mtp_policy_trace_enabled() {
    return env_flag_enabled("QW3_MTP_POLICY_TRACE");
}

uint32_t mtp_startup_demote_batches() {
    const char *raw = std::getenv("QW3_MTP_ADAPTIVE_STARTUP_DEMOTE_BATCHES");
    if (!raw || !*raw) return 0;
    const std::string value = env_lower_ascii(raw);
    if (env_disabled_value(value) || value == "off" || value == "none") {
        return 0;
    }
    size_t pos = 0;
    const unsigned long parsed = std::stoul(value, &pos);
    if (pos != value.size() ||
        parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid QW3_MTP_ADAPTIVE_STARTUP_DEMOTE_BATCHES: " +
                                 value);
    }
    return static_cast<uint32_t>(parsed);
}

float env_float_or(const char *name, float fallback) {
    const char *raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char *end = nullptr;
    const float value = std::strtof(raw, &end);
    if (end == raw || (end && *end != '\0') || !std::isfinite(value)) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
    return value;
}

uint32_t mtp_adaptive_prior_depth(size_t prompt_tokens, uint32_t max_depth) {
    uint32_t prior = 3;
    if (prompt_tokens <= 6144) {
        prior = 4;
    } else if (prompt_tokens <= 12288) {
        prior = 5;
    } else if (prompt_tokens <= 40960) {
        prior = 4;
    } else if (prompt_tokens <= 73728) {
        prior = 3;
    } else {
        prior = 3;
    }
    return std::min<uint32_t>(prior, max_depth);
}

struct MtpAdaptivePolicy {
    static constexpr uint32_t kMaxTrackedDepth = 8;
    static constexpr uint32_t kCostDepths = 5;

    bool enabled = false;
    bool trace = false;
    uint32_t min_depth = 1;
    uint32_t max_depth = 1;
    uint32_t update_interval = 16;
    uint32_t min_decision_batches = 64;
    uint32_t cooldown_batches = 8;
    uint32_t demote_windows = 2;
    uint32_t promote_windows = 1;
    uint32_t startup_demote_batches = 0;
    uint32_t initial_depth = 1;
    uint32_t depth = 1;
    uint32_t batches = 0;
    uint32_t cooldown = 0;
    uint32_t promotions = 0;
    uint32_t demotions = 0;
    uint32_t changes = 0;
    uint32_t bad_windows = 0;
    uint32_t good_windows = 0;
    uint32_t window_depth = 0;
    uint32_t window_batches = 0;
    std::array<uint32_t, kMaxTrackedDepth + 1> window_hist{};
    float last_benefit = 0.0f;
    float last_cost = 0.0f;
    float last_next_cost = 0.0f;
    float last_avg_committed = 0.0f;
    float last_full_rate = 0.0f;
    float demote_margin = 0.005f;
    float promote_margin = 0.005f;
    std::array<uint64_t, kMaxTrackedDepth + 1> depth_batches{};
    std::array<uint64_t, kMaxTrackedDepth + 1> depth_drafted{};
    std::array<uint64_t, kMaxTrackedDepth + 1> depth_accepted{};
    std::array<std::array<uint64_t, kMaxTrackedDepth + 1>, kMaxTrackedDepth + 1>
        depth_accept_hist{};

    void configure(bool active, uint32_t chain_len, size_t prompt_tokens) {
        enabled = active && mtp_adaptive_policy_requested() && chain_len > 1;
        trace = mtp_policy_trace_enabled();
        if (!enabled) {
            min_depth = 1;
            max_depth = std::max<uint32_t>(1, chain_len);
            initial_depth = max_depth;
            depth = max_depth;
            return;
        }

        max_depth = std::min<uint32_t>(
            chain_len,
            env_uint32_or("QW3_MTP_ADAPTIVE_MAX_CHAIN", chain_len));
        max_depth = std::min<uint32_t>(max_depth, kCostDepths);
        max_depth = std::min<uint32_t>(max_depth, kMaxTrackedDepth);
        max_depth = std::max<uint32_t>(1, max_depth);
        const uint32_t default_min_depth = max_depth >= 4 ? 4 : 1;
        min_depth = std::min<uint32_t>(
            max_depth,
            env_uint32_or("QW3_MTP_ADAPTIVE_MIN_CHAIN", default_min_depth));
        min_depth = std::max<uint32_t>(1, min_depth);
        update_interval = std::max<uint32_t>(
            1, env_uint32_or("QW3_MTP_ADAPTIVE_UPDATE_INTERVAL", 16));
        min_decision_batches = std::max<uint32_t>(
            1, env_uint32_or("QW3_MTP_ADAPTIVE_MIN_BATCHES", 64));
        cooldown_batches = env_uint32_or("QW3_MTP_ADAPTIVE_COOLDOWN", 8);
        demote_windows = std::max<uint32_t>(
            1, env_uint32_or("QW3_MTP_ADAPTIVE_DEMOTE_WINDOWS", 2));
        promote_windows = std::max<uint32_t>(
            1, env_uint32_or("QW3_MTP_ADAPTIVE_PROMOTE_WINDOWS", 1));
        startup_demote_batches = mtp_startup_demote_batches();
        demote_margin = std::max<float>(
            0.0f, env_float_or("QW3_MTP_ADAPTIVE_DEMOTE_MARGIN", 0.005f));
        promote_margin = std::max<float>(
            0.0f, env_float_or("QW3_MTP_ADAPTIVE_PROMOTE_MARGIN", 0.005f));
        initial_depth = std::max<uint32_t>(
            min_depth,
            std::min<uint32_t>(mtp_adaptive_prior_depth(prompt_tokens, max_depth),
                               max_depth));
        depth = initial_depth;
    }

    uint32_t draft_limit(uint32_t remaining_tokens, uint32_t fixed_chain_len) const {
        const uint32_t selected = enabled ? depth : fixed_chain_len;
        return std::max<uint32_t>(1, std::min<uint32_t>(selected, remaining_tokens));
    }

    static uint32_t cost_bin(size_t context_tokens) {
        if (context_tokens <= 6144) return 0;
        if (context_tokens <= 12288) return 1;
        if (context_tokens <= 24576) return 2;
        if (context_tokens <= 49152) return 3;
        if (context_tokens <= 98304) return 4;
        return 5;
    }

    static float round_cost_ms(size_t context_tokens, uint32_t depth) {
        static constexpr float kRoundCostMs[6][kCostDepths] = {
            // Empirical default-path round costs from 4K, 8K, 16K, 32K,
            // 64K, and 128K fixed-depth MTP sweeps. These costs are only
            // used for policy decisions, never for timing/reporting.
            {23.98f, 26.54f, 28.99f, 30.50f, 33.37f},
            {24.45f, 26.58f, 28.35f, 30.37f, 32.90f},
            {24.34f, 27.26f, 29.18f, 31.59f, 34.64f},
            {25.92f, 28.26f, 30.67f, 33.26f, 36.14f},
            {28.69f, 31.80f, 34.64f, 37.97f, 40.79f},
            {30.74f, 36.39f, 40.99f, 47.60f, 51.99f},
        };
        const uint32_t d = std::max<uint32_t>(
            1, std::min<uint32_t>(depth, kCostDepths));
        return kRoundCostMs[cost_bin(context_tokens)][d - 1];
    }

    static float marginal_cost_fraction(size_t context_tokens, uint32_t depth) {
        if (depth <= 1) return 1.0f;
        const float prev = round_cost_ms(context_tokens, depth - 1);
        const float current = round_cost_ms(context_tokens, depth);
        if (current <= 0.0f || current <= prev) return 0.0f;
        return (current - prev) / current;
    }

    void reset_window(uint32_t new_depth) {
        window_depth = new_depth;
        window_batches = 0;
        window_hist.fill(0);
    }

    uint64_t full_depth_batches(uint32_t current_depth) const {
        if (current_depth == 0 || current_depth > kMaxTrackedDepth) return 0;
        uint64_t batches_for_depth = 0;
        for (uint32_t i = 0; i <= current_depth; ++i) {
            batches_for_depth += depth_accept_hist[current_depth][i];
        }
        return batches_for_depth;
    }

    bool compute_depth_score(size_t context_tokens, uint32_t current_depth,
                             bool require_min_batches = true) {
        if (current_depth == 0 ||
            current_depth > kMaxTrackedDepth) {
            return false;
        }
        const uint64_t batches_for_depth = full_depth_batches(current_depth);
        if (batches_for_depth == 0 ||
            (require_min_batches && batches_for_depth < min_decision_batches)) {
            return false;
        }
        uint64_t committed = 0;
        for (uint32_t i = 0; i <= current_depth; ++i) {
            committed += static_cast<uint64_t>(i + 1) *
                         depth_accept_hist[current_depth][i];
        }
        last_avg_committed =
            static_cast<float>(committed) /
            static_cast<float>(batches_for_depth);
        last_full_rate =
            static_cast<float>(depth_accept_hist[current_depth][current_depth]) /
            static_cast<float>(batches_for_depth);
        last_benefit = last_avg_committed > 0.0f
            ? last_full_rate / last_avg_committed
            : 0.0f;
        last_cost = marginal_cost_fraction(context_tokens, current_depth);
        last_next_cost = current_depth < max_depth
            ? marginal_cost_fraction(context_tokens, current_depth + 1)
            : 0.0f;
        return true;
    }

    const char *update(uint32_t drafted, uint32_t accepted, size_t context_tokens) {
        if (!enabled || drafted == 0) return "disabled";

        const uint32_t tracked_depth = std::min<uint32_t>(depth, kMaxTrackedDepth);
        ++depth_batches[tracked_depth];
        depth_drafted[tracked_depth] += drafted;
        depth_accepted[tracked_depth] += accepted;

        if (window_depth != depth) {
            reset_window(depth);
        }

        ++batches;
        if (drafted != depth || accepted > drafted) {
            return "partial";
        }
        ++window_batches;
        ++window_hist[accepted];
        ++depth_accept_hist[tracked_depth][accepted];

        if (startup_demote_batches > 0 && batches <= startup_demote_batches &&
            depth == initial_depth && depth == max_depth && depth > min_depth &&
            full_depth_batches(depth) == startup_demote_batches &&
            compute_depth_score(context_tokens, depth, false) &&
            last_benefit + demote_margin < last_cost) {
            --depth;
            ++demotions;
            ++changes;
            bad_windows = 0;
            good_windows = 0;
            cooldown = cooldown_batches;
            reset_window(depth);
            return "startup_demote";
        }

        if (cooldown > 0) {
            --cooldown;
            return "cooldown";
        }
        const uint64_t batches_for_depth = full_depth_batches(depth);
        if (batches_for_depth < min_decision_batches ||
            batches_for_depth % update_interval != 0) {
            return "hold";
        }

        const bool have_score = compute_depth_score(context_tokens, depth);
        if (have_score && depth > min_depth &&
            last_benefit + demote_margin < last_cost) {
            ++bad_windows;
            if (bad_windows >= demote_windows) {
                --depth;
                ++demotions;
                ++changes;
                bad_windows = 0;
                good_windows = 0;
                cooldown = cooldown_batches;
                reset_window(depth);
                return "demote";
            }
            good_windows = 0;
            reset_window(depth);
            return "warn";
        }
        bad_windows = 0;
        if (have_score && depth < max_depth &&
            last_benefit > last_next_cost + promote_margin) {
            ++good_windows;
            if (good_windows >= promote_windows) {
                ++depth;
                ++promotions;
                ++changes;
                good_windows = 0;
                cooldown = cooldown_batches;
                reset_window(depth);
                return "promote";
            }
            reset_window(depth);
            return "probe";
        }
        good_windows = 0;
        reset_window(depth);
        return "hold";
    }
};

uint32_t decode_trace_top_n() {
    return env_uint32_or("QW3_DECODE_TRACE_TOP", 20);
}

struct TraceStats {
    uint64_t calls = 0;
    double total_us = 0.0;
    double max_us = 0.0;
};

std::string trace_group_name(const std::string &op) {
    constexpr const char *prefix = "layer.";
    if (op.rfind(prefix, 0) != 0) return op;
    const size_t second_dot = op.find('.', std::char_traits<char>::length(prefix));
    if (second_dot == std::string::npos || second_dot + 1 >= op.size()) return op;
    return std::string(prefix) + op.substr(second_dot + 1);
}

void accumulate_trace(std::unordered_map<std::string, TraceStats> &stats,
                      const NativeExecutorReport &report) {
    const size_t n = std::min(report.executed.size(), report.elapsed_us.size());
    for (size_t i = 0; i < n; ++i) {
        TraceStats &item = stats[trace_group_name(report.executed[i])];
        item.calls++;
        item.total_us += report.elapsed_us[i];
        item.max_us = std::max(item.max_us, report.elapsed_us[i]);
    }
}

std::string escape_text(const std::string &s) {
    std::ostringstream out;
    out << "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << "\"";
    return out.str();
}

// Streams one JSON object per forward step into a file. Each line records
// the (input token, decoded text, argmax token, top-k logits + token strings)
// so it can be diffed against a llama.cpp logit dump or used to localize
// where qw3 diverges from a reference.
class DumpStream {
public:
    DumpStream(const std::string &path, int top_k,
               const std::vector<uint32_t> &prompt_tokens,
               const QwenTokenizer &tok)
        : top_k_(top_k > 0 ? top_k : 1) {
        out_.open(path);
        if (!out_) throw std::runtime_error("failed to open dump-logits path: " + path);
        out_ << "{\"event\":\"prompt\",\"tokens\":[";
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (i) out_ << ",";
            out_ << prompt_tokens[i];
        }
        out_ << "],\"strings\":[";
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            if (i) out_ << ",";
            out_ << escape_text(tok.decode_one(static_cast<int32_t>(prompt_tokens[i])));
        }
        out_ << "]}\n";
    }

    void record(int step_idx, const char *phase, int32_t input_token,
                const QwenExecutor &exec, const QwenTokenizer &tok) {
        std::vector<float> logits;
        if (!exec.copy_last_logits(logits)) return;
        const int K = std::min<int>(top_k_, static_cast<int>(logits.size()));
        std::vector<int> idx(K, -1);
        std::vector<float> val(K, -INFINITY);
        for (size_t i = 0; i < logits.size(); ++i) {
            const float v = logits[i];
            int slot = -1;
            for (int j = 0; j < K; ++j) if (v > val[j]) { slot = j; break; }
            if (slot < 0) continue;
            for (int j = K - 1; j > slot; --j) {
                val[j] = val[j - 1];
                idx[j] = idx[j - 1];
            }
            val[slot] = v;
            idx[slot] = static_cast<int>(i);
        }
        out_ << "{\"event\":\"step\",\"step\":" << step_idx
             << ",\"phase\":\"" << phase << "\""
             << ",\"input_token\":" << input_token
             << ",\"input_text\":" << escape_text(tok.decode_one(input_token))
             << ",\"argmax_token\":" << idx[0]
             << ",\"argmax_logit\":" << val[0]
             << ",\"argmax_text\":" << escape_text(tok.decode_one(idx[0]))
             << ",\"top\":[";
        for (int j = 0; j < K; ++j) {
            if (j) out_ << ",";
            out_ << "{\"id\":" << idx[j]
                 << ",\"logit\":" << val[j]
                 << ",\"text\":" << escape_text(tok.decode_one(idx[j])) << "}";
        }
        out_ << "]}\n";
    }

private:
    std::ofstream out_;
    int top_k_ = 16;
};

class QwenNativeBackend final : public Backend {
public:
    ~QwenNativeBackend() override {
        stop_continuous_batch_worker();
    }

    std::string name() const override {
        return "qwen-native";
    }

    void load(const EngineOptions &options) override {
        if (options.model_path.empty()) {
            throw std::invalid_argument("qwen-native backend requires --model");
        }
        options_ = options;

        const double t0 = wall_seconds();
        model_ = std::make_unique<QwenNativeModel>(std::make_unique<GgufFile>(options.model_path));
        tokenizer_ = std::make_unique<QwenTokenizer>(model_->gguf());
        const double t_gguf = wall_seconds();

        // Device backend + weight uploads are now part of load(), not
        // generate(). Subsequent generate() calls reuse the same DeviceBackend
        // and the same on-GPU weight buffers.
        if (options_.native_kernels != "cuda") {
            // mock/cpu kernels are no longer wired here; we still let load()
            // complete so callers that just want to inspect the plan can do so.
            log("native load: gguf=" + fmt_seconds(t_gguf - t0) +
                " (skipped device init: native-kernels=" + options_.native_kernels + ")");
            return;
        }
        if (!cuda_device_backend_available()) {
            throw std::runtime_error("CUDA kernels were not built; configure with -DQW3_ENABLE_CUDA=ON");
        }
        const LinearBackend linear_backend = parse_linear_backend(options_.native_linear_backend);
        device_ = make_cuda_device_backend(linear_backend);
        if (!device_) throw std::runtime_error("CUDA device backend is unavailable");
        cb_host_tier_pool_ = std::make_unique<HostTierBufferPool>(*device_);

        DeviceStatus st = device_->begin();
        if (!st.ok) throw std::runtime_error(std::string("device begin failed: ") + st.message);

        const double t_begin = wall_seconds();
        weights_ = std::make_unique<QwenWeights>(*model_, *device_);
        st = device_->synchronize();
        if (!st.ok) throw std::runtime_error(std::string("weight upload sync failed: ") + st.message);
        const double t_weights = wall_seconds();

        const uint32_t ctx_size = options_.ctx_size > 0 ? static_cast<uint32_t>(options_.ctx_size) : 4096u;
        executor_ = std::make_unique<QwenExecutor>(*model_, *weights_, *device_, ctx_size);
        executor_->set_prefill_chunk_override(options_.prefill_chunk);
        configure_executor_kvmem(*executor_);
        executor_->reset_state();
        if (env_flag_enabled("QW3_CONTINUOUS_BATCHING")) {
            const uint32_t kv_page_size =
                std::max<uint32_t>(1, env_uint32_or("QW3_PAGED_KV_PAGE_SIZE", 16));
            const uint32_t per_executor_pages =
                (ctx_size + kv_page_size - 1) / kv_page_size;
            const uint32_t default_pool_pages = per_executor_pages;
            const uint32_t requested_pool_pages =
                env_uint32_or("QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES",
                              default_pool_pages);
            const uint32_t pool_pages = std::max<uint32_t>(1, requested_pool_pages);
            cb_kv_pool_ =
                std::make_unique<GlobalKvPagePool>(pool_pages, kv_page_size);
            allocate_continuous_kv_cache(pool_pages, kv_page_size);
            const uint32_t mtp_pool_pages = std::max<uint32_t>(
                1, env_uint32_or("QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES",
                                 pool_pages));
            cb_mtp_kv_pool_ =
                std::make_unique<GlobalKvPagePool>(mtp_pool_pages,
                                                   kv_page_size);
            allocate_continuous_mtp_kv_cache(mtp_pool_pages, kv_page_size);
        }

        st = device_->end();
        if (!st.ok) throw std::runtime_error(std::string("device end failed: ") + st.message);

        const double mib = static_cast<double>(weights_->total_bytes_uploaded()) / (1024.0 * 1024.0);
        std::ostringstream msg;
        msg << "native load: gguf=" << fmt_seconds(t_gguf - t0)
            << " device_init=" << fmt_seconds(t_begin - t_gguf)
            << " weights_upload=" << fmt_seconds(t_weights - t_begin)
            << " tensors=" << weights_->tensor_count()
            << " size=" << std::fixed << std::setprecision(1) << mib << " MiB"
            << " backend=" << linear_backend_name(linear_backend);
        log(msg.str());
    }

    // ---- kvmem growth-profiling harness ----------------------------------
    // Drives the persistent context-growth experiment: prefill an initial long
    // context, then keep appending document chunks across turns up to the
    // largest ladder target, sampling a short MTP decode probe at each point.
    // For each turn it prints the user-requested sequential micro-step
    // breakdown (selection / stage-in / stage-out / assemble / prefill /
    // decode) plus a final summary table with tier residency. Requires kvmem
    // enabled (the free run_kvmem_session forces it on with update-mode=step).
    int run_kvmem_session(const KvMemSessionConfig &cfg) {
        if (!model_ || !tokenizer_ || !device_ || !executor_) {
            throw std::runtime_error("kvmem-session: backend not fully loaded "
                                     "(need --native-kernels cuda)");
        }
        if (!executor_->kvmem_enabled()) {
            throw std::runtime_error("kvmem-session requires kvmem enabled");
        }
        if (cfg.ladder_tokens.empty()) {
            throw std::runtime_error("kvmem-session: empty ladder");
        }
        if (!QwenExecutor::kvmem_timing_enabled()) {
            log("kvmem-session: WARNING QW3_KVMEM_TIMING not set; the "
                "selection/stage/assemble breakdown will read as zero");
        }

        // Pre-tokenize a synthetic document pool large enough to cover the
        // largest ladder target. Each turn carves an exact token slice so the
        // running position lands precisely on each ladder point after prefill.
        const uint64_t max_target = cfg.ladder_tokens.back();
        std::vector<uint32_t> pool;
        build_session_corpus(pool, max_target + 2048);
        log("kvmem-session: corpus pool tokens=" + std::to_string(pool.size()) +
            " target_max=" + std::to_string(max_target));

        GenerationOptions gen;
        gen.max_tokens = std::max(1, cfg.decode_tokens);
        gen.temperature = 0.0f;
        gen.top_k = 0;
        gen.top_p = 1.0f;
        gen.ignore_eos = true;  // decode exactly decode_tokens (steady-state TBT)

        struct TurnRow {
            size_t turn = 0;
            uint64_t ctx_tokens = 0;
            uint64_t delta_tokens = 0;
            double sel_ms = 0, stage_in_ms = 0, stage_out_ms = 0, assemble_ms = 0;
            double asm_pages_ms = 0, asm_rerope_ms = 0, asm_kbar_ms = 0;
            uint32_t stage_in_blocks = 0, stage_out_blocks = 0;
            // kvmem work forced DURING prefill (bounded-pool offload), folded
            // into prefill_s. Surfaced so step5's gross wall isn't opaque.
            double inpre_ms = 0;
            uint32_t inpre_stage_in_blocks = 0, inpre_stage_out_blocks = 0;
            double prefill_s = 0, decode_s = 0, decode_tps = 0;
            int decoded = 0;
            double acceptance = 0;
            uint64_t kv_bytes = 0, gpu_used = 0, cpu_used = 0, nvme_used = 0;
            bool gpu_pool = false;
            uint64_t gpu_mib = 0, rss_mib = 0;
        };
        std::vector<TurnRow> rows;

        size_t pool_cursor = 0;
        for (size_t ti = 0; ti < cfg.ladder_tokens.size(); ++ti) {
            const uint64_t target = cfg.ladder_tokens[ti];
            const uint64_t cur_pos = executor_->position();
            if (target <= cur_pos) {
                log("kvmem-session: ladder[" + std::to_string(ti) +
                    "]=" + std::to_string(target) +
                    " already reached (pos=" + std::to_string(cur_pos) +
                    "), skipping");
                continue;
            }
            uint64_t delta = target - cur_pos;
            if (pool_cursor + delta > pool.size()) {
                delta = pool.size() - pool_cursor;  // clamp (corpus exhausted)
            }
            if (delta == 0) {
                log("kvmem-session: corpus exhausted at ladder[" +
                    std::to_string(ti) + "]");
                break;
            }
            std::vector<uint32_t> chunk(
                pool.begin() + static_cast<std::ptrdiff_t>(pool_cursor),
                pool.begin() + static_cast<std::ptrdiff_t>(pool_cursor + delta));
            pool_cursor += delta;

            const QwenExecutor::KvMemTimingSnapshot tbase =
                QwenExecutor::kvmem_timing_snapshot();

            MtpGenStats stats;
            const bool reset = (ti == 0);
            (void)generate_mtp(chunk, gen, TokenCallback{}, /*dump=*/nullptr,
                               /*spec_mtp=*/true, /*trace_mtp=*/false,
                               /*override_executor=*/nullptr,
                               /*manage_device_scope=*/true,
                               /*reset_session=*/reset, &stats);

            const QwenExecutor::KvMemTimingSnapshot tnow =
                QwenExecutor::kvmem_timing_snapshot();

            auto dms = [](uint64_t a, uint64_t b) {
                return static_cast<double>(a - b) / 1.0e6;  // ns -> ms
            };
            // Steps 1-4 (selection/stage-in/stage-out/assemble) are charged at
            // the POST-PREFILL decode-window reselect only. generate_mtp returns
            // the kvmem timing snapshot taken at the prefill->reselect boundary;
            // diffing the global counters (tnow, captured after the call -- and
            // decode does NOT reselect in step mode) against that boundary
            // isolates the boundary reselect from mid-prefill offload churn,
            // which stays inside prefill_s.
            const QwenExecutor::KvMemTimingSnapshot &base =
                stats.kvmem_boundary_valid ? stats.kvmem_at_boundary : tbase;
            TurnRow row;
            row.turn = ti;
            row.ctx_tokens = executor_->position();
            row.delta_tokens = delta;
            row.sel_ms = dms(tnow.retrieval_ns, base.retrieval_ns);
            row.stage_in_ms = dms(tnow.stage_in_ns, base.stage_in_ns);
            row.stage_out_ms = dms(tnow.stage_out_ns, base.stage_out_ns);
            row.assemble_ms = dms(tnow.assemble_ns, base.assemble_ns);
            row.asm_pages_ms = dms(tnow.assemble_pages_ns, base.assemble_pages_ns);
            row.asm_rerope_ms = dms(tnow.assemble_rerope_ns, base.assemble_rerope_ns);
            row.asm_kbar_ms = dms(tnow.assemble_kbar_ns, base.assemble_kbar_ns);
            row.stage_in_blocks = tnow.stage_in_blocks - base.stage_in_blocks;
            row.stage_out_blocks = tnow.stage_out_blocks - base.stage_out_blocks;
            // kvmem work forced during prefill = (turn start -> prefill end).
            if (stats.kvmem_boundary_valid) {
                const QwenExecutor::KvMemTimingSnapshot &b = stats.kvmem_at_boundary;
                row.inpre_ms = dms(b.retrieval_ns, tbase.retrieval_ns) +
                               dms(b.stage_in_ns, tbase.stage_in_ns) +
                               dms(b.stage_out_ns, tbase.stage_out_ns) +
                               dms(b.assemble_ns, tbase.assemble_ns);
                row.inpre_stage_in_blocks = b.stage_in_blocks - tbase.stage_in_blocks;
                row.inpre_stage_out_blocks = b.stage_out_blocks - tbase.stage_out_blocks;
            }
            row.prefill_s = stats.prefill_s;
            // decode_s is already the pure decode loop -- generate_mtp now
            // excludes the post-prefill reselect (reported via stats.reselect_s).
            row.decode_s = std::max(stats.decode_s, 1.0e-9);
            row.decoded = stats.decoded;
            row.decode_tps =
                row.decoded > 0 ? row.decoded / row.decode_s : 0.0;
            row.acceptance = stats.acceptance;

            const QwenExecutor::KvMemTierUsage tu = executor_->kvmem_tier_usage();
            row.kv_bytes = tu.total_blocks * tu.block_bytes;
            row.gpu_used = tu.gpu_used_bytes;
            row.cpu_used = tu.cpu_used_bytes;
            row.nvme_used = tu.nvme_used_bytes;
            row.gpu_pool = tu.gpu_pool;
            const uint64_t gpu_total = device_->total_device_bytes();
            const uint64_t gpu_free = device_->free_device_bytes();
            row.gpu_mib =
                (gpu_total > gpu_free ? gpu_total - gpu_free : 0) / (1024 * 1024);
            row.rss_mib = current_rss_mib();

            print_session_turn(row.turn, row.ctx_tokens, row.delta_tokens,
                               row.sel_ms, row.stage_in_ms, row.stage_out_ms,
                               row.assemble_ms, row.asm_pages_ms,
                               row.asm_rerope_ms, row.asm_kbar_ms,
                               row.stage_in_blocks, row.stage_out_blocks,
                               row.prefill_s, row.decode_s, row.decode_tps,
                               row.decoded, row.acceptance, row.gpu_mib,
                               row.rss_mib, row.inpre_ms,
                               row.inpre_stage_in_blocks,
                               row.inpre_stage_out_blocks);
            rows.push_back(row);
        }

        // Final summary table.
        std::ostringstream tbl;
        tbl << "\n=== kvmem-session SUMMARY (update_mode=step, MTP on) ===\n";
        tbl << "  turn      ctx    delta   sel_ms  in_ms  out_ms  asm_ms"
               "  prefill_s  pre_tok/s  decode_s  dec_tok/s  accept    KVgib  GPUgib  CPUgib  NVMEgib  pool  GPUmib  RSSmib\n";
        for (const TurnRow &r : rows) {
            tbl << "  " << std::setw(4) << r.turn
                << std::setw(9) << r.ctx_tokens
                << std::setw(9) << r.delta_tokens
                << std::fixed << std::setprecision(2)
                << std::setw(9) << r.sel_ms
                << std::setw(7) << r.stage_in_ms
                << std::setw(8) << r.stage_out_ms
                << std::setw(8) << r.assemble_ms
                << std::setw(11) << r.prefill_s
                << std::setprecision(1)
                << std::setw(11) << (r.delta_tokens / std::max(r.prefill_s, 1e-9))
                << std::setprecision(2)
                << std::setw(10) << r.decode_s
                << std::setw(11) << r.decode_tps
                << std::setprecision(4)
                << std::setw(8) << r.acceptance
                << std::setprecision(3)
                << std::setw(9) << (r.kv_bytes / (1024.0 * 1024.0 * 1024.0))
                << std::setw(8) << (r.gpu_used / (1024.0 * 1024.0 * 1024.0))
                << std::setw(8) << (r.cpu_used / (1024.0 * 1024.0 * 1024.0))
                << std::setw(9) << (r.nvme_used / (1024.0 * 1024.0 * 1024.0))
                << std::setw(6) << (r.gpu_pool ? 1 : 0)
                << std::setw(8) << r.gpu_mib
                << std::setw(8) << r.rss_mib
                << "\n";
        }
        std::cerr << tbl.str();
        return 0;
    }

    std::string generate(const std::string &prompt,
                         const GenerationOptions &options,
                         const TokenCallback &on_text) override {
        if (!model_) throw std::runtime_error("qwen-native backend is not loaded");
        if (options_.native_kernels != "cuda") {
            throw std::runtime_error("qwen-native now uses a device-resident executor; use --native-kernels cuda");
        }
        if (!device_ || !weights_ || !executor_) {
            throw std::runtime_error("qwen-native backend was not fully initialized in load()");
        }

        if (!tokenizer_) tokenizer_ = std::make_unique<QwenTokenizer>(model_->gguf());
        const std::vector<int32_t> ids = tokenizer_->encode(prompt);
        std::vector<uint32_t> prompt_tokens(ids.begin(), ids.end());
        GenerationOptions effective_options = options;
        const uint32_t ctx_size = options_.ctx_size > 0
            ? static_cast<uint32_t>(options_.ctx_size)
            : 4096u;
        if (prompt_tokens.size() > static_cast<size_t>(ctx_size)) {
            throw std::runtime_error("prompt exceeds KV context: prompt_tokens=" +
                                     std::to_string(prompt_tokens.size()) +
                                     " ctx=" + std::to_string(ctx_size));
        }
        const uint32_t max_emit_tokens =
            ctx_size - static_cast<uint32_t>(prompt_tokens.size()) + 1U;
        if (effective_options.max_tokens > static_cast<int>(max_emit_tokens)) {
            log("native generate: capping max_tokens from " +
                std::to_string(effective_options.max_tokens) + " to " +
                std::to_string(max_emit_tokens) +
                " to fit KV ctx=" + std::to_string(ctx_size));
            effective_options.max_tokens = static_cast<int>(max_emit_tokens);
        }

        if (options_.dump_tokens) {
            std::ostringstream out;
            out << "tokens=" << prompt_tokens.size() << "\n";
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                const std::string text = tokenizer_->decode_one(static_cast<int32_t>(prompt_tokens[i]));
                out << i << "\t" << prompt_tokens[i] << "\t" << escape_text(text) << "\n";
            }
            const std::string text = out.str();
            if (on_text) on_text(text);
            return text;
        }

        std::unique_ptr<DumpStream> dump;
        if (!options_.dump_logits_path.empty()) {
            dump = std::make_unique<DumpStream>(options_.dump_logits_path,
                                                options_.dump_logits_top_k,
                                                prompt_tokens, *tokenizer_);
        }

        if (!options_.native_heavy) {
            DeviceStatus st = device_->begin();
            if (!st.ok) throw std::runtime_error(st.message);
            NativeExecutorReport dry = executor_->dry_run_token(
                prompt_tokens.empty() ? 0u : prompt_tokens.front(), false);
            st = device_->end();
            if (!st.ok) throw std::runtime_error(st.message);
            std::ostringstream out;
            out << "executor_dry_run: " << (dry.ok ? "complete" : "blocked") << "\n";
            const std::string text = out.str();
            if (on_text) on_text(text);
            return text;
        }

        const bool spec_mtp = mtp_speculate_enabled(options_);
        const bool trace_mtp = options_.native_mtp_trace || mtp_trace_enabled();
        const bool active_mtp = trace_mtp || spec_mtp;

        const bool route_continuous =
            effective_options.continuous_batching &&
            (!active_mtp || continuous_batching_mtp_enabled());

        // kvmem × MTP on the continuous-batching path now runs through the
        // window-aware RAGGED verify route (build_continuous_mtp_verify_batch
        // substitutes the assembled window page table + logical positions for
        // every kvmem row; the post-kernel advance + snapshot/checkpoint
        // rollback keep window_query_pos_ in lockstep with position_). The only
        // remaining unsupported combination is the opt-in LAYERED verifier
        // (QW3_CONTINUOUS_MTP_LAYERED_VERIFY): its BatchedDecodeExecutor reads
        // each executor's live full-cache page table, not the window, so it
        // would silently verify against the wrong KV. Per the locked design
        // decision that stays a HARD ERROR — never a silent fallback or MTP
        // disable. (When --kvmem is set, configure_executor_kvmem applies to
        // every CB executor, so this global guard covers all rows; there is no
        // kvmem/non-kvmem mix within one CB worker.)
        if (options_.kvmem_enabled && active_mtp && route_continuous &&
            continuous_mtp_layered_verify_enabled()) {
            throw std::runtime_error(
                "kvmem (--kvmem) cannot be combined with the layered MTP "
                "verifier (QW3_CONTINUOUS_MTP_LAYERED_VERIFY) on the "
                "continuous-batching path: the layered decode executor is not "
                "kvmem-window-aware. Unset QW3_CONTINUOUS_MTP_LAYERED_VERIFY to "
                "use the window-aware ragged verify route, or drop --kvmem.");
        }

        if (route_continuous &&
            continuous_batch_request_supported(effective_options, dump.get())) {
            return generate_continuous_batched(prompt_tokens, effective_options, on_text);
        }

        // The continuous batching worker owns the shared CUDA backend while it
        // is running. Unsupported requests fall back to the original path only
        // after the worker drains, so scratch/stream state is never shared by
        // two host threads at once.
        stop_continuous_batch_worker();

        // Non-MTP greedy decode keeps qw3's native graph-capture / FlashInfer
        // / internal-chunking path byte-for-byte. MTP machinery is purely
        // additive: only the active_mtp branch differs.
        if (!active_mtp) {
            return generate_plain(prompt_tokens, effective_options, on_text, dump.get());
        }
        return generate_mtp(prompt_tokens, effective_options, on_text, dump.get(),
                            spec_mtp, trace_mtp);
    }

private:
    struct ContinuousBatchRequest {
        uint64_t id = 0;
        std::vector<uint32_t> prompt_tokens;
        GenerationOptions options;
        TokenCallback on_text;
        uint64_t reserved_tokens = 0;
        bool budget_released = false;
        bool spec_mtp = false;
        bool trace_mtp = false;
        bool active_mtp = false;

        std::mutex mu;
        std::condition_variable cv;
        bool done = false;
        std::string generated;
        std::string error;
    };

    struct ContinuousRequestKvState {
        uint32_t seq_len = 0;
        uint32_t ctx_size = 0;
        uint32_t page_size = 0;
        uint32_t logical_pages = 0;
        std::vector<int32_t> physical_pages;

        void update(const QwenExecutor::KvStateSnapshot &snapshot) {
            seq_len = snapshot.seq_len;
            ctx_size = snapshot.ctx_size;
            page_size = snapshot.page_size;
            logical_pages = snapshot.logical_pages;
            physical_pages = snapshot.physical_pages;
        }
    };

    // Per-request state for the thinking-token budget. Counts tokens emitted
    // inside an open <think> block; when the budget is reached the engine feeds
    // a short queued guidance line + the </think> close token so the model
    // proceeds straight to its answer. Disabled when budget == 0.
    struct ThinkingBudgetState {
        int budget = 0;          // max tokens inside <think>; 0 disables
        bool open = false;       // a <think> block is currently open
        int close_id = -1;       // tokenizer id of "</think>" (-1 = unavailable)
        int think_tokens = 0;    // tokens generated so far inside the block
        bool forced = false;     // currently feeding forced close tokens
        std::deque<uint32_t> forced_queue; // remaining forced tokens to feed

        bool active() const { return budget > 0 && close_id >= 0; }
    };

    struct ContinuousBatchActive {
        std::shared_ptr<ContinuousBatchRequest> req;
        std::unique_ptr<QwenExecutor> executor;
        ContinuousRequestKvState kv_state;
        std::unordered_map<uint32_t, uint32_t> seen_tokens;
        std::mt19937_64 rng;
        std::vector<float> logit_buf;
        ThinkingBudgetState budget;
        uint32_t next_token = 0;
        int decoded = 0;
        uint32_t prefill_offset = 0;
        uint64_t prefill_ops = 0;
        uint64_t decode_ops = 0;
        double prefill_s = 0.0;
        double decode_start = 0.0;
        // Prefix-cache bookkeeping (serve path). held_prefix_entries are entry
        // ids whose refcount this request holds (adopted on hit + committed on
        // miss); all are decremented at finish. prefix_commit_pending requests
        // a one-shot recurrent+KV snapshot when prefill_offset reaches
        // prefix_commit_len (a page-aligned boundary < prompt_len).
        std::vector<uint64_t> held_prefix_entries;
        bool prefix_commit_pending = false;
        uint32_t prefix_commit_len = 0;
        // kvmem reselect cadence (per-request, mirrors generate_plain's
        // bs_steps_since_reselect): counts committed decode tokens since the
        // last window reselection. Unused when --kvmem is off.
        int kvmem_steps_since_reselect = 0;
        // Position-delta cadence trackers for the MTP path, where multiple
        // tokens commit per verify step. Mirror the single-request
        // kvmem_advance_to lambda: register the position() delta since the last
        // call (so an accepted chain registers its whole jump at once) and
        // reselect on the interval boundary. Unused when --kvmem is off.
        uint32_t kvmem_registered_pos = 0;
        uint32_t kvmem_last_reselect_pos = 0;
        // KVMem component-timing baseline (env QW3_KVMEM_TIMING). Snapshot of
        // the process-global tier/selection accumulators at admit; the per-
        // request breakdown is the delta emitted at finish. Inert when timing
        // is off.
        QwenExecutor::KvMemTimingSnapshot kvmem_timing_baseline;
    };

    struct ContinuousDecodeBatch {
        std::vector<size_t> active_indices;
        std::vector<uint32_t> feed_tokens;
        std::vector<uint32_t> positions;
        std::vector<QwenExecutor::DecodeStateView> state_views;

        void clear() {
            active_indices.clear();
            feed_tokens.clear();
            positions.clear();
            state_views.clear();
        }

        size_t size() const { return active_indices.size(); }
    };

    struct ContinuousPrefillBatchEntry {
        size_t prefill_index = 0;
        uint64_t request_id = 0;
        uint32_t offset = 0;
        uint32_t total = 0;
        uint32_t chunk = 0;
        bool final_chunk = false;
    };

    struct ContinuousPrefillBatch {
        std::vector<ContinuousPrefillBatchEntry> entries;
        std::vector<int32_t> q_indptr;
        std::vector<int32_t> page_indptr;
        std::vector<int32_t> row_page_indptr;
        std::vector<int32_t> page_indices;
        std::vector<int32_t> logical_positions;
        std::vector<int32_t> last_page_len;
        std::vector<int32_t> seq_lens;
        std::vector<uint32_t> token_rows;
        uint64_t total_tokens = 0;
        uint32_t final_chunks = 0;
        uint32_t page_size = 0;
        uint32_t max_seq_len = 0;
        bool ragged_metadata_ready = false;
        bool ragged_device_metadata_ready = false;
        bool ragged_row_metadata_ready = false;
        bool collect_row_argmaxes = false;
        bool recurrent_state_ready = false;
        bool recurrent_state_packed = false;
        bool recurrent_state_unpacked = false;
        uint32_t recurrent_state_packed_layers = 0;

        void clear() {
            entries.clear();
            q_indptr.clear();
            page_indptr.clear();
            row_page_indptr.clear();
            page_indices.clear();
            logical_positions.clear();
            last_page_len.clear();
            seq_lens.clear();
            token_rows.clear();
            total_tokens = 0;
            final_chunks = 0;
            page_size = 0;
            max_seq_len = 0;
            ragged_metadata_ready = false;
            ragged_device_metadata_ready = false;
            ragged_row_metadata_ready = false;
            collect_row_argmaxes = false;
            recurrent_state_ready = false;
            recurrent_state_packed = false;
            recurrent_state_unpacked = false;
            recurrent_state_packed_layers = 0;
        }

        size_t size() const { return entries.size(); }
    };

    struct BatchedDecodeInput {
        const ContinuousDecodeBatch *batch = nullptr;
    };

    struct BatchedDecodeOutput {
        size_t active_index = 0;
        uint32_t feed_token = 0;
        NativeExecutorReport report;
        std::string error;

        bool ok() const { return error.empty() && report.ok; }
    };

    struct BatchedDecodeTiming {
        double total_s = 0.0;
        double prepare_s = 0.0;
        double metadata_s = 0.0;
        double embed_s = 0.0;
        double layers_s = 0.0;
        double recurrent_s = 0.0;
        double recurrent_state_s = 0.0;
        double attention_s = 0.0;
        double qkv_s = 0.0;
        double kv_append_s = 0.0;
        double attn_kernel_s = 0.0;
        double attn_output_s = 0.0;
        double ffn_s = 0.0;
        double final_s = 0.0;
        double lm_head_s = 0.0;
        double argmax_s = 0.0;
        double post_s = 0.0;
    };

    struct BatchedPrefillOutput {
        size_t prefill_index = 0;
        uint64_t request_id = 0;
        uint32_t offset = 0;
        uint32_t total = 0;
        uint32_t chunk = 0;
        bool final_chunk = false;
        double seconds = 0.0;
        NativeExecutorReport report;
        std::vector<DeviceArgmax> row_argmaxes;
        // Per-row full fp32 LM-head logits, populated only on the single-verifier
        // (forward_n_tokens) path when the request samples (temp>0); the MTP
        // speculative-sampling accept test needs the target distribution per row,
        // not just the argmax. Empty on greedy / ragged-verify rows.
        std::vector<std::vector<float>> row_logits;
        QwenExecutor::StateCheckpointSet checkpoints;
        std::string error;

        bool ok() const { return error.empty() && report.ok; }
    };

    struct BatchedPrefillTiming {
        double total_s = 0.0;
        double prepare_s = 0.0;
        double embed_s = 0.0;
        double layers_s = 0.0;
        double recurrent_s = 0.0;
        double attention_s = 0.0;
        double ffn_s = 0.0;
        double final_s = 0.0;
        double post_s = 0.0;
        double ragged_s = 0.0;
        double delegated_s = 0.0;
    };

    struct BatchedPrefillDeviceMetadata {
        const DeviceTensor *q_indptr = nullptr;
        const DeviceTensor *page_indptr = nullptr;
        const DeviceTensor *row_page_indptr = nullptr;
        const DeviceTensor *page_indices = nullptr;
        const DeviceTensor *logical_positions = nullptr;
        const DeviceTensor *last_page_len = nullptr;
        const DeviceTensor *seq_lens = nullptr;
    };

    class BatchedPrefillExecutor {
    public:
        BatchedPrefillExecutor(const QwenNativeModel &model,
                               const QwenWeights &weights,
                               DeviceBackend &backend)
            : model_(model), weights_(weights), backend_(backend) {}

        const std::string &last_mode() const { return last_mode_; }
        const BatchedPrefillTiming &last_timing() const { return last_timing_; }
        uint32_t last_kernel_batch() const { return last_kernel_batch_; }

        std::vector<BatchedPrefillOutput> prefill(
                std::vector<ContinuousBatchActive> &prefilling,
                const ContinuousPrefillBatch &batch,
                const BatchedPrefillDeviceMetadata &metadata) {
            std::vector<BatchedPrefillOutput> outputs;
            outputs.reserve(batch.size());
            last_timing_ = {};
            if (can_use_ragged_prefill(prefilling, batch, metadata)) {
                return prefill_ragged(prefilling, batch, metadata);
            }
            last_mode_ = "delegated";
            last_kernel_batch_ = 1;
            const double t0 = wall_seconds();

            for (const ContinuousPrefillBatchEntry &entry : batch.entries) {
                BatchedPrefillOutput out;
                out.prefill_index = entry.prefill_index;
                out.request_id = entry.request_id;
                out.offset = entry.offset;
                out.total = entry.total;
                out.chunk = entry.chunk;
                out.final_chunk = entry.final_chunk;
                const double step0 = wall_seconds();
                try {
                    if (entry.prefill_index >= prefilling.size()) {
                        throw std::runtime_error("prefill index out of range");
                    }
                    ContinuousBatchActive &a = prefilling[entry.prefill_index];
                    if (!a.req || !a.executor) {
                        throw std::runtime_error("prefill request unavailable");
                    }
                    const std::vector<uint32_t> &prompt = a.req->prompt_tokens;
                    if (entry.offset != a.prefill_offset ||
                        entry.offset + entry.chunk > prompt.size()) {
                        throw std::runtime_error("prefill batch entry is stale");
                    }
                    std::vector<uint32_t> chunk_tokens(
                        prompt.begin() +
                            static_cast<std::ptrdiff_t>(entry.offset),
                        prompt.begin() +
                            static_cast<std::ptrdiff_t>(entry.offset + entry.chunk));
                    out.report =
                        a.executor->forward_n_tokens(chunk_tokens,
                                                     entry.final_chunk);
                    if (!out.report.ok) {
                        throw std::runtime_error("prefill failed");
                    }
                    const double step1 = wall_seconds();
                    out.seconds = std::max(step1 - step0, 1e-9);
                    a.prefill_s += out.seconds;
                    a.prefill_ops += out.report.ops_executed;
                    a.prefill_offset += entry.chunk;
                    a.kv_state.update(a.executor->kv_state_snapshot());
                } catch (const std::exception &e) {
                    out.error = e.what();
                    out.seconds = std::max(wall_seconds() - step0, 0.0);
                }
                outputs.push_back(std::move(out));
            }

            const double t1 = wall_seconds();
            last_timing_.total_s = std::max(t1 - t0, 0.0);
            last_timing_.delegated_s = last_timing_.total_s;
            return outputs;
        }

    private:
        static bool request_needs_logits(const GenerationOptions &options) {
            return options.temperature > 0.0f ||
                   options.presence_penalty != 0.0f ||
                   (options.repetition_penalty > 0.0f &&
                    options.repetition_penalty != 1.0f);
        }

        bool can_use_ragged_prefill(
                const std::vector<ContinuousBatchActive> &prefilling,
                const ContinuousPrefillBatch &batch,
                const BatchedPrefillDeviceMetadata &metadata) const {
            if (!continuous_batching_ragged_prefill_executor_enabled()) return false;
            if (batch.size() < 2 || batch.total_tokens == 0) return false;
            if (!batch.collect_row_argmaxes && batch.total_tokens <
                continuous_batching_ragged_prefill_min_tokens()) {
                return false;
            }
            if (!batch.ragged_metadata_ready ||
                !batch.ragged_device_metadata_ready ||
                !batch.ragged_row_metadata_ready ||
                !batch.recurrent_state_ready) {
                return false;
            }
            if (!metadata.q_indptr || !metadata.page_indptr ||
                !metadata.row_page_indptr || !metadata.page_indices ||
                !metadata.logical_positions || !metadata.last_page_len ||
                !metadata.seq_lens) {
                return false;
            }
            const std::string kv_dtype =
                env_lower_ascii(env_value("QW3_KV_DTYPE"));
            if (!kv_dtype.empty() && kv_dtype != "fp16" && kv_dtype != "fp8") {
                return false;
            }
            for (const ContinuousPrefillBatchEntry &entry : batch.entries) {
                if (entry.prefill_index >= prefilling.size()) return false;
                const ContinuousBatchActive &a = prefilling[entry.prefill_index];
                if (!a.req || !a.executor) return false;
                if (request_needs_logits(a.req->options)) return false;
                if (entry.offset != a.prefill_offset) return false;
                if (entry.chunk == 0) return false;
                if (!batch.collect_row_argmaxes &&
                    entry.offset + entry.chunk > a.req->prompt_tokens.size()) {
                    return false;
                }
            }
            for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
                const QwenLayerWeights &layer = weights_.layer(il);
                if (!layer.recurrent && (!layer.attn_q || !layer.attn_k ||
                                         !layer.attn_v || !layer.attn_output)) {
                    return false;
                }
            }
            return true;
        }

        void ensure_ragged_scratch(uint32_t total_q, uint32_t final_rows,
                                   uint32_t batch_size) {
            const QwenConfig &cfg = model_.config();
            uint64_t max_ffn = 1;
            uint64_t max_q = 1;
            uint64_t max_k = 1;
            uint64_t max_v = 1;
            uint64_t max_recurrent_qkv = 1;
            uint64_t max_recurrent_value = 1;
            for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
                const QwenLayerWeights &layer = weights_.layer(il);
                max_ffn = std::max<uint64_t>(max_ffn, layer.ffn_dim);
                max_q = std::max<uint64_t>(max_q, layer.q_rows);
                max_k = std::max<uint64_t>(max_k, layer.k_rows);
                max_v = std::max<uint64_t>(max_v, layer.v_rows);
                max_recurrent_qkv =
                    std::max<uint64_t>(max_recurrent_qkv,
                                       layer.recurrent_qkv_dim);
                max_recurrent_value =
                    std::max<uint64_t>(max_recurrent_value,
                                       layer.recurrent_value_dim);
            }
            if (total_q > ragged_capacity_) {
                const uint64_t T = total_q;
                hidden_batch_ = backend_.scratch_f32(T * cfg.n_embd,
                                                     "cb_prefill_hidden_batch");
                norm_batch_ = backend_.scratch_f32(T * cfg.n_embd,
                                                   "cb_prefill_norm_batch");
                attn_out_batch_ = backend_.scratch_f32(T * cfg.n_embd,
                                                       "cb_prefill_attn_out_batch");
                ffn_gate_batch_ = backend_.scratch_f32(T * max_ffn,
                                                       "cb_prefill_ffn_gate_batch");
                ffn_up_batch_ = backend_.scratch_f32(T * max_ffn,
                                                     "cb_prefill_ffn_up_batch");
                ffn_mid_batch_ = backend_.scratch_f32(T * max_ffn,
                                                      "cb_prefill_ffn_mid_batch");
                ffn_out_batch_ = backend_.scratch_f32(T * cfg.n_embd,
                                                      "cb_prefill_ffn_out_batch");
                q_batch_ = backend_.scratch_f32(T * max_q,
                                                "cb_prefill_q_batch");
                k_batch_ = backend_.scratch_f32(T * max_k,
                                                "cb_prefill_k_batch");
                v_batch_ = backend_.scratch_f32(T * max_v,
                                                "cb_prefill_v_batch");
                mid_batch_ = backend_.scratch_f32(
                    T * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim,
                    "cb_prefill_mid_batch");
                recurrent_proj_batch_ = backend_.scratch_f32(
                    T * max_recurrent_qkv,
                    "cb_prefill_recurrent_proj_batch");
                recurrent_gate_batch_ = backend_.scratch_f32(
                    T * max_recurrent_value,
                    "cb_prefill_recurrent_gate_batch");
                recurrent_alpha_batch_ = backend_.scratch_f32(
                    T * cfg.num_v_heads(),
                    "cb_prefill_recurrent_alpha_batch");
                recurrent_beta_batch_ = backend_.scratch_f32(
                    T * cfg.num_v_heads(),
                    "cb_prefill_recurrent_beta_batch");
                recurrent_core_batch_ = backend_.scratch_f32(
                    T * max_recurrent_value,
                    "cb_prefill_recurrent_core_batch");
                recurrent_conv_out_batch_ = backend_.scratch_f32(
                    T * max_recurrent_qkv,
                    "cb_prefill_recurrent_conv_out_batch");
                ragged_capacity_ = total_q;
            }
            if (batch_size > ragged_state_capacity_) {
                const uint64_t B = batch_size;
                recurrent_state_batch_ = backend_.scratch_f32(
                    B * static_cast<uint64_t>(cfg.num_v_heads()) *
                        cfg.head_v_dim_ssm() * cfg.head_k_dim(),
                    "cb_prefill_recurrent_state_batch");
                recurrent_conv_state_batch_ = backend_.scratch_f32(
                    B * max_recurrent_qkv *
                        static_cast<uint64_t>(cfg.ssm_conv_kernel - 1),
                    "cb_prefill_recurrent_conv_state_batch");
                ragged_state_capacity_ = batch_size;
            }
            if (final_rows > final_capacity_) {
                const uint32_t vocab =
                    static_cast<uint32_t>(weights_.output().rows);
                final_hidden_batch_ = backend_.scratch_f32(
                    static_cast<uint64_t>(final_rows) * cfg.n_embd,
                    "cb_prefill_final_hidden_batch");
                final_norm_batch_ = backend_.scratch_f32(
                    static_cast<uint64_t>(final_rows) * cfg.n_embd,
                    "cb_prefill_final_norm_batch");
                final_logits_batch_ = backend_.scratch_f32(
                    static_cast<uint64_t>(final_rows) * vocab,
                    "cb_prefill_final_logits_batch");
                final_capacity_ = final_rows;
            }
        }

        static void require_ok(const DeviceStatus &st) {
            if (!st.ok) throw std::runtime_error(st.message);
        }

        static DeviceTensor *layer_k_cache(QwenExecutor::MutableDecodeStateView &view,
                                           uint32_t layer_index) {
            if (view.k_cache_external && layer_index < view.k_cache_external->size()) {
                return (*view.k_cache_external)[layer_index];
            }
            if (view.k_cache && layer_index < view.k_cache->size() &&
                (*view.k_cache)[layer_index]) {
                return (*view.k_cache)[layer_index].get();
            }
            return nullptr;
        }

        static DeviceTensor *layer_v_cache(QwenExecutor::MutableDecodeStateView &view,
                                           uint32_t layer_index) {
            if (view.v_cache_external && layer_index < view.v_cache_external->size()) {
                return (*view.v_cache_external)[layer_index];
            }
            if (view.v_cache && layer_index < view.v_cache->size() &&
                (*view.v_cache)[layer_index]) {
                return (*view.v_cache)[layer_index].get();
            }
            return nullptr;
        }

        std::vector<BatchedPrefillOutput> prefill_ragged(
                std::vector<ContinuousBatchActive> &prefilling,
                const ContinuousPrefillBatch &batch,
                const BatchedPrefillDeviceMetadata &metadata) {
            last_mode_ = "ragged_prefill";
            last_kernel_batch_ = static_cast<uint32_t>(batch.size());
            const double t0 = wall_seconds();
            std::vector<BatchedPrefillOutput> outputs;
            outputs.reserve(batch.size());
            for (const ContinuousPrefillBatchEntry &entry : batch.entries) {
                BatchedPrefillOutput out;
                out.prefill_index = entry.prefill_index;
                out.request_id = entry.request_id;
                out.offset = entry.offset;
                out.total = entry.total;
                out.chunk = entry.chunk;
                out.final_chunk = entry.final_chunk;
                outputs.push_back(std::move(out));
            }

            try {
                const double t_prepare0 = wall_seconds();
                const uint32_t bsz = static_cast<uint32_t>(batch.size());
                const uint32_t total_q = static_cast<uint32_t>(batch.total_tokens);
                const uint32_t final_rows = batch.collect_row_argmaxes
                    ? std::max<uint32_t>(total_q, 1)
                    : std::max<uint32_t>(batch.final_chunks, 1);
                const QwenConfig &cfg = model_.config();
                const uint32_t hidden = cfg.n_embd;
                const uint32_t vocab =
                    static_cast<uint32_t>(weights_.output().rows);
                const uint32_t standard_head_dim = cfg.head_dim;
                const uint32_t standard_n_heads = cfg.n_heads;
                const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
                const uint32_t q_head_stride = 2 * standard_head_dim;
                const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
                const uint32_t mid_stride = standard_n_heads * standard_head_dim;
                const float eps = cfg.rms_eps;
                const float scale =
                    1.0f / std::sqrt(static_cast<float>(standard_head_dim));
                ensure_ragged_scratch(total_q, final_rows, bsz);
                uint32_t ragged_checkpoint_count = 0;
                if (batch.collect_row_argmaxes) {
                    for (uint32_t row = 0; row < bsz; ++row) {
                        BatchedPrefillOutput &out = outputs[row];
                        ragged_checkpoint_count =
                            std::max<uint32_t>(ragged_checkpoint_count,
                                               out.chunk);
                        out.checkpoints.ready = false;
                        out.checkpoints.base_position = out.offset;
                        out.checkpoints.count = out.chunk;
                        out.checkpoints.h_stride = hidden;
                        // Capture the pre-advance window frame so that
                        // restore_state_checkpoint(index) can roll the kvmem
                        // window tail back per accepted row, mirroring how it
                        // rolls position_ back. window_query_pos() still equals
                        // the verify base here (advance happens post-kernel).
                        if (out.prefill_index < prefilling.size()) {
                            QwenExecutor *ex =
                                prefilling[out.prefill_index].executor.get();
                            out.checkpoints.kvmem_active = ex->kvmem_active();
                            out.checkpoints.window_base_query_pos =
                                ex->window_query_pos();
                            out.checkpoints.window_base_page_count =
                                ex->window_page_count();
                        }
                        if (out.checkpoints.recurrent_states.size() !=
                            weights_.n_layers()) {
                            out.checkpoints.recurrent_states.resize(
                                weights_.n_layers());
                        }
                        if (out.checkpoints.conv_states.size() !=
                            weights_.n_layers()) {
                            out.checkpoints.conv_states.resize(
                                weights_.n_layers());
                        }
                        if (out.checkpoints.recurrent_states_shared.size() !=
                            weights_.n_layers()) {
                            out.checkpoints.recurrent_states_shared.resize(
                                weights_.n_layers());
                        }
                        if (out.checkpoints.conv_states_shared.size() !=
                            weights_.n_layers()) {
                            out.checkpoints.conv_states_shared.resize(
                                weights_.n_layers());
                        }
                    }
                }

                require_ok(backend_.begin());
                const double t_prepare1 = wall_seconds();

                std::vector<uint64_t> rows_h(total_q, 0);
                for (uint32_t req = 0; req < bsz; ++req) {
                    const ContinuousPrefillBatchEntry &entry = batch.entries[req];
                    const uint32_t row_begin =
                        static_cast<uint32_t>(batch.q_indptr[req]);
                    for (uint32_t t = 0; t < entry.chunk; ++t) {
                        rows_h[row_begin + t] = !batch.token_rows.empty()
                            ? batch.token_rows[row_begin + t]
                            : prefilling[entry.prefill_index]
                                  .req->prompt_tokens[entry.offset + t];
                    }
                }
                require_ok(backend_.q8_0_get_rows_batch(
                    *hidden_batch_, weights_.token_embd(), rows_h.data(),
                    total_q));
                const double t_embed1 = wall_seconds();

                double recurrent_s = 0.0;
                double attention_s = 0.0;
                double ffn_s = 0.0;
                const double t_layers0 = wall_seconds();
                for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
                    const QwenLayerWeights &layer = weights_.layer(il);
                    if (layer.recurrent) {
                        const double t_recurrent0 = wall_seconds();
                        const uint32_t num_k_heads = cfg.num_k_heads();
                        const uint32_t num_v_heads = cfg.num_v_heads();
                        const uint32_t head_k_dim = cfg.head_k_dim();
                        const uint32_t head_v_dim = cfg.head_v_dim_ssm();
                        const uint32_t proj_stride =
                            static_cast<uint32_t>(layer.recurrent_qkv_dim);
                        const uint32_t gate_stride =
                            static_cast<uint32_t>(layer.recurrent_value_dim);
                        const uint32_t alpha_stride = num_v_heads;
                        const uint32_t beta_stride = num_v_heads;
                        const uint32_t core_stride =
                            static_cast<uint32_t>(layer.recurrent_value_dim);
                        const uint32_t state_stride =
                            num_v_heads * head_v_dim * head_k_dim;
                        const uint32_t conv_state_stride =
                            proj_stride * (cfg.ssm_conv_kernel - 1);
                        if (proj_stride == 0 || gate_stride == 0 ||
                            core_stride == 0 || state_stride == 0 ||
                            conv_state_stride == 0) {
                            throw std::runtime_error(
                                "ragged prefill recurrent layer shape unavailable");
                        }
                        require_ok(backend_.rms_norm_batch(
                            *norm_batch_, *hidden_batch_, *layer.attn_norm,
                            total_q, hidden, eps));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_proj_batch_, *layer.attn_qkv,
                            *norm_batch_, total_q, hidden, proj_stride));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_gate_batch_, *layer.attn_gate,
                            *norm_batch_, total_q, hidden, gate_stride));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_alpha_batch_, *layer.ssm_alpha,
                            *norm_batch_, total_q, hidden, alpha_stride));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_beta_batch_, *layer.ssm_beta,
                            *norm_batch_, total_q, hidden, beta_stride));
                        for (uint32_t row = 0; row < bsz; ++row) {
                            ContinuousBatchActive &a =
                                prefilling[outputs[row].prefill_index];
                            QwenExecutor::MutableDecodeStateView view =
                                a.executor->mutable_decode_state_view();
                            if (!view.recurrent_states || !view.conv_states ||
                                il >= view.recurrent_states->size() ||
                                il >= view.conv_states->size() ||
                                !(*view.recurrent_states)[il] ||
                                !(*view.conv_states)[il]) {
                                throw std::runtime_error(
                                    "ragged prefill recurrent state unavailable");
                            }
                            DeviceTensor &state = *(*view.recurrent_states)[il];
                            DeviceTensor &conv_state = *(*view.conv_states)[il];
                            if (state.count < state_stride ||
                                conv_state.count < conv_state_stride) {
                                throw std::runtime_error(
                                    "ragged prefill recurrent state too small");
                            }
                            require_ok(backend_.copy_d2d_into(
                                *recurrent_state_batch_,
                                static_cast<uint64_t>(row) * state_stride,
                                state, 0, state_stride));
                            require_ok(backend_.copy_d2d_into(
                                *recurrent_conv_state_batch_,
                                static_cast<uint64_t>(row) * conv_state_stride,
                                conv_state, 0, conv_state_stride));
                        }
                        std::shared_ptr<DeviceTensor> state_checkpoint_shared;
                        std::shared_ptr<DeviceTensor> conv_checkpoint_shared;
                        DeviceTensor *state_checkpoint_ptr = nullptr;
                        DeviceTensor *conv_checkpoint_ptr = nullptr;
                        if (batch.collect_row_argmaxes &&
                            ragged_checkpoint_count > 0) {
                            state_checkpoint_shared = backend_.scratch_f32(
                                static_cast<uint64_t>(bsz) *
                                    ragged_checkpoint_count * state_stride,
                                "mtp_ragged_checkpoint_state_batch");
                            conv_checkpoint_shared = backend_.scratch_f32(
                                static_cast<uint64_t>(bsz) *
                                    ragged_checkpoint_count *
                                    conv_state_stride,
                                "mtp_ragged_checkpoint_conv_batch");
                            state_checkpoint_ptr =
                                state_checkpoint_shared.get();
                            conv_checkpoint_ptr =
                                conv_checkpoint_shared.get();
                        }
                        require_ok(backend_.recurrent_batch_ragged(
                            *recurrent_core_batch_,
                            *recurrent_state_batch_,
                            *recurrent_conv_state_batch_,
                            *recurrent_conv_out_batch_,
                            *recurrent_proj_batch_,
                            *recurrent_gate_batch_,
                            *recurrent_alpha_batch_,
                            *recurrent_beta_batch_,
                            *metadata.q_indptr,
                            batch.q_indptr.data(),
                            *layer.ssm_conv1d,
                            *layer.ssm_a,
                            *layer.ssm_dt_bias,
                            *layer.ssm_norm,
                            bsz, total_q, num_k_heads, num_v_heads,
                            head_k_dim, head_v_dim, cfg.ssm_conv_kernel,
                            proj_stride, proj_stride, gate_stride,
                            alpha_stride, beta_stride, core_stride,
                            state_stride, conv_state_stride, eps,
                            state_checkpoint_ptr, conv_checkpoint_ptr,
                            ragged_checkpoint_count));
                        if (state_checkpoint_shared &&
                            conv_checkpoint_shared) {
                            for (uint32_t row = 0; row < bsz; ++row) {
                                BatchedPrefillOutput &out = outputs[row];
                                out.checkpoints.checkpoint_stride =
                                    ragged_checkpoint_count;
                                out.checkpoints.checkpoint_row = row;
                                out.checkpoints.recurrent_states_shared[il] =
                                    state_checkpoint_shared;
                                out.checkpoints.conv_states_shared[il] =
                                    conv_checkpoint_shared;
                            }
                        }
                        require_ok(backend_.q8_0_matmul_add(
                            *hidden_batch_, *hidden_batch_, *attn_out_batch_,
                            *layer.ssm_out, *recurrent_core_batch_,
                            total_q, core_stride, hidden));
                        for (uint32_t row = 0; row < bsz; ++row) {
                            ContinuousBatchActive &a =
                                prefilling[outputs[row].prefill_index];
                            QwenExecutor::MutableDecodeStateView view =
                                a.executor->mutable_decode_state_view();
                            DeviceTensor &state = *(*view.recurrent_states)[il];
                            DeviceTensor &conv_state = *(*view.conv_states)[il];
                            require_ok(backend_.copy_d2d_into(
                                state, 0, *recurrent_state_batch_,
                                static_cast<uint64_t>(row) * state_stride,
                                state_stride));
                            require_ok(backend_.copy_d2d_into(
                                conv_state, 0, *recurrent_conv_state_batch_,
                                static_cast<uint64_t>(row) * conv_state_stride,
                                conv_state_stride));
                        }
                        recurrent_s +=
                            std::max(wall_seconds() - t_recurrent0, 0.0);
                    } else {
                        const double t_attention0 = wall_seconds();
                        require_ok(backend_.rms_norm_batch(
                            *norm_batch_, *hidden_batch_, *layer.attn_norm,
                            total_q, hidden, eps));
                        DeviceTensor *qkv_outs[3] = {
                            q_batch_.get(), k_batch_.get(), v_batch_.get()
                        };
                        const DeviceWeight *qkv_ws[3] = {
                            layer.attn_q, layer.attn_k, layer.attn_v
                        };
                        const uint32_t qkv_strides[3] = {
                            static_cast<uint32_t>(layer.q_rows),
                            static_cast<uint32_t>(layer.k_rows),
                            static_cast<uint32_t>(layer.v_rows),
                        };
                        require_ok(backend_.q8_0_matmul_fanout(
                            qkv_outs, qkv_ws, qkv_strides, 3,
                            *norm_batch_, total_q, hidden));
                        require_ok(backend_.rmsnorm_per_head_batch(
                            *q_batch_, *layer.attn_q_norm, total_q,
                            static_cast<uint32_t>(layer.q_rows),
                            standard_n_heads, q_head_stride,
                            standard_head_dim, eps));
                        require_ok(backend_.rmsnorm_per_head_batch(
                            *k_batch_, *layer.attn_k_norm, total_q,
                            static_cast<uint32_t>(layer.k_rows),
                            standard_n_kv_heads, standard_head_dim,
                            standard_head_dim, eps));
                        require_ok(backend_.rope_partial_batch_positions(
                            *q_batch_, total_q,
                            static_cast<uint32_t>(layer.q_rows),
                            standard_n_heads, q_head_stride, cfg.rope_dim,
                            *metadata.logical_positions, cfg.rope_theta));
                        require_ok(backend_.rope_partial_batch_positions(
                            *k_batch_, total_q,
                            static_cast<uint32_t>(layer.k_rows),
                            standard_n_kv_heads, standard_head_dim,
                            cfg.rope_dim, *metadata.logical_positions,
                            cfg.rope_theta));

                        QwenExecutor::MutableDecodeStateView first_view =
                            prefilling[outputs[0].prefill_index]
                                .executor->mutable_decode_state_view();
                        DeviceTensor *k_cache = layer_k_cache(first_view, il);
                        DeviceTensor *v_cache = layer_v_cache(first_view, il);
                        if (!k_cache || !v_cache) {
                            throw std::runtime_error(
                                "ragged prefill KV cache unavailable");
                        }
                        require_ok(backend_.kv_append_batch_paged_ragged_device(
                            *k_cache, *k_batch_, *metadata.logical_positions,
                            per_pos, total_q,
                            static_cast<uint32_t>(layer.k_rows),
                            *metadata.page_indices,
                            *metadata.row_page_indptr,
                            first_view.kv_page_size));
                        require_ok(backend_.kv_append_batch_paged_ragged_device(
                            *v_cache, *v_batch_, *metadata.logical_positions,
                            per_pos, total_q,
                            static_cast<uint32_t>(layer.v_rows),
                            *metadata.page_indices,
                            *metadata.row_page_indptr,
                            first_view.kv_page_size));
                        require_ok(
                            backend_.attention_prefill_batch_paged_ragged_gated_device(
                                *mid_batch_, *q_batch_, q_head_stride,
                                *k_cache, *v_cache, *metadata.page_indices,
                                *metadata.page_indptr,
                                *metadata.last_page_len,
                                *metadata.q_indptr, batch.q_indptr.data(),
                                batch.page_indptr.data(), bsz, total_q,
                                first_view.kv_page_size, standard_n_heads,
                                standard_n_kv_heads, standard_head_dim,
                                static_cast<uint32_t>(layer.q_rows),
                                mid_stride, scale));
                        require_ok(backend_.q8_0_matmul(
                            *attn_out_batch_, *layer.attn_output,
                            *mid_batch_, total_q, mid_stride, hidden));
                        require_ok(backend_.add_n(
                            *hidden_batch_, *hidden_batch_, *attn_out_batch_,
                            static_cast<uint64_t>(total_q) * hidden));
                        attention_s +=
                            std::max(wall_seconds() - t_attention0, 0.0);
                    }

                    const double t_ffn0 = wall_seconds();
                    require_ok(backend_.rms_norm_batch(
                        *norm_batch_, *hidden_batch_, *layer.ffn_norm,
                        total_q, hidden, eps));
                    require_ok(backend_.q8_0_matmul(
                        *ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                        total_q, hidden,
                        static_cast<uint32_t>(layer.ffn_dim)));
                    require_ok(backend_.q8_0_matmul(
                        *ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                        total_q, hidden,
                        static_cast<uint32_t>(layer.ffn_dim)));
                    require_ok(backend_.silu_mul_n(
                        *ffn_mid_batch_, *ffn_gate_batch_, *ffn_up_batch_,
                        static_cast<uint64_t>(total_q) * layer.ffn_dim));
                    require_ok(backend_.q8_0_matmul(
                        *ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                        total_q, static_cast<uint32_t>(layer.ffn_dim),
                        hidden));
                    require_ok(backend_.add_n(
                        *hidden_batch_, *hidden_batch_, *ffn_out_batch_,
                        static_cast<uint64_t>(total_q) * hidden));
                    ffn_s += std::max(wall_seconds() - t_ffn0, 0.0);
                }
                const double t_layers1 = wall_seconds();

                uint32_t final_row = 0;
                std::vector<uint32_t> final_output_rows;
                final_output_rows.reserve(batch.final_chunks);
                for (uint32_t row = 0; row < bsz; ++row) {
                    const uint32_t last_q =
                        static_cast<uint32_t>(batch.q_indptr[row + 1] - 1);
                    ContinuousBatchActive &a =
                        prefilling[outputs[row].prefill_index];
                    QwenExecutor::MutableDecodeStateView view =
                        a.executor->mutable_decode_state_view();
                    require_ok(backend_.copy_d2d_into(
                        *view.hidden, 0, *hidden_batch_,
                        static_cast<uint64_t>(last_q) * hidden, hidden));
                    a.executor->advance_position(outputs[row].chunk);
                    if (batch.collect_row_argmaxes) {
                        BatchedPrefillOutput &out = outputs[row];
                        const uint32_t begin =
                            static_cast<uint32_t>(batch.q_indptr[row]);
                        out.checkpoints.h_shared =
                            std::shared_ptr<DeviceTensor>(
                                hidden_batch_.get(),
                                [](DeviceTensor *) {});
                        out.checkpoints.h_checkpoint_row = begin;
                        out.checkpoints.ready = true;
                    }
                    if (outputs[row].final_chunk) {
                        final_output_rows.push_back(row);
                        require_ok(backend_.copy_d2d_into(
                            *final_hidden_batch_,
                            static_cast<uint64_t>(final_row) * hidden,
                            *hidden_batch_,
                            static_cast<uint64_t>(last_q) * hidden,
                            hidden));
                        ++final_row;
                    }
                }
                std::vector<DeviceArgmax> argmaxes;
                if (batch.collect_row_argmaxes) {
                    require_ok(backend_.rms_norm_batch(
                        *final_norm_batch_, *hidden_batch_,
                        weights_.output_norm(), total_q, hidden, eps));
                    require_ok(backend_.q8_0_matmul(
                        *final_logits_batch_, weights_.output(),
                        *final_norm_batch_, total_q, hidden, vocab));
                    require_ok(backend_.argmax_batch(
                        *final_logits_batch_, total_q, vocab, argmaxes));
                } else if (final_row > 0) {
                    require_ok(backend_.rms_norm_batch(
                        *final_norm_batch_, *final_hidden_batch_,
                        weights_.output_norm(), final_row, hidden, eps));
                    require_ok(backend_.q8_0_matmul(
                        *final_logits_batch_, weights_.output(),
                        *final_norm_batch_, final_row, hidden, vocab));
                    require_ok(backend_.argmax_batch(
                        *final_logits_batch_, final_row, vocab, argmaxes));
                }
                require_ok(backend_.end());
                const double t_final1 = wall_seconds();

                for (uint32_t row = 0; row < bsz; ++row) {
                    ContinuousBatchActive &a =
                        prefilling[outputs[row].prefill_index];
                    outputs[row].seconds =
                        std::max((wall_seconds() - t0) /
                                     static_cast<double>(bsz),
                                 1e-9);
                    outputs[row].report.ok = true;
                    outputs[row].report.ops_executed =
                        static_cast<uint64_t>(weights_.n_layers());
                    a.prefill_s += outputs[row].seconds;
                    a.prefill_ops += outputs[row].report.ops_executed;
                    a.prefill_offset += outputs[row].chunk;
                    a.kv_state.update(a.executor->kv_state_snapshot());
                }
                if (batch.collect_row_argmaxes) {
                    for (uint32_t row = 0; row < bsz; ++row) {
                        const uint32_t begin =
                            static_cast<uint32_t>(batch.q_indptr[row]);
                        const uint32_t end =
                            static_cast<uint32_t>(batch.q_indptr[row + 1]);
                        BatchedPrefillOutput &out = outputs[row];
                        if (begin < end && end <= argmaxes.size()) {
                            out.row_argmaxes.assign(
                                argmaxes.begin() +
                                    static_cast<std::ptrdiff_t>(begin),
                                argmaxes.begin() +
                                    static_cast<std::ptrdiff_t>(end));
                            const DeviceArgmax &last = out.row_argmaxes.back();
                            out.report.argmax_token = last.token;
                            out.report.argmax_logit = last.logit;
                            out.report.argmax_text =
                                model_.gguf().token_text(
                                    static_cast<uint32_t>(last.token));
                        }
                    }
                } else {
                    for (uint32_t i = 0; i < final_output_rows.size() &&
                                         i < argmaxes.size(); ++i) {
                        BatchedPrefillOutput &out =
                            outputs[final_output_rows[i]];
                        out.report.argmax_token = argmaxes[i].token;
                        out.report.argmax_logit = argmaxes[i].logit;
                        out.report.argmax_text =
                            model_.gguf().token_text(
                                static_cast<uint32_t>(argmaxes[i].token));
                    }
                }
                const double t_post1 = wall_seconds();
                last_timing_.prepare_s =
                    std::max(t_prepare1 - t_prepare0, 0.0);
                last_timing_.embed_s =
                    std::max(t_embed1 - t_prepare1, 0.0);
                last_timing_.layers_s =
                    std::max(t_layers1 - t_layers0, 0.0);
                last_timing_.recurrent_s = recurrent_s;
                last_timing_.attention_s = attention_s;
                last_timing_.ffn_s = ffn_s;
                last_timing_.final_s =
                    std::max(t_final1 - t_layers1, 0.0);
                last_timing_.post_s =
                    std::max(t_post1 - t_final1, 0.0);
            } catch (const std::exception &e) {
                try { (void)backend_.end(); } catch (...) {}
                for (auto &out : outputs) {
                    if (out.error.empty()) out.error = e.what();
                    out.seconds = std::max(wall_seconds() - t0, 0.0);
                }
            }

            const double t1 = wall_seconds();
            last_timing_.total_s = std::max(t1 - t0, 0.0);
            last_timing_.ragged_s = last_timing_.total_s;
            last_timing_.delegated_s = 0.0;
            return outputs;
        }

        const QwenNativeModel &model_;
        const QwenWeights &weights_;
        DeviceBackend &backend_;
        uint32_t ragged_capacity_ = 0;
        uint32_t ragged_state_capacity_ = 0;
        uint32_t final_capacity_ = 0;
        std::unique_ptr<DeviceTensor> hidden_batch_;
        std::unique_ptr<DeviceTensor> norm_batch_;
        std::unique_ptr<DeviceTensor> attn_out_batch_;
        std::unique_ptr<DeviceTensor> ffn_gate_batch_;
        std::unique_ptr<DeviceTensor> ffn_up_batch_;
        std::unique_ptr<DeviceTensor> ffn_mid_batch_;
        std::unique_ptr<DeviceTensor> ffn_out_batch_;
        std::unique_ptr<DeviceTensor> q_batch_;
        std::unique_ptr<DeviceTensor> k_batch_;
        std::unique_ptr<DeviceTensor> v_batch_;
        std::unique_ptr<DeviceTensor> mid_batch_;
        std::unique_ptr<DeviceTensor> recurrent_proj_batch_;
        std::unique_ptr<DeviceTensor> recurrent_gate_batch_;
        std::unique_ptr<DeviceTensor> recurrent_alpha_batch_;
        std::unique_ptr<DeviceTensor> recurrent_beta_batch_;
        std::unique_ptr<DeviceTensor> recurrent_core_batch_;
        std::unique_ptr<DeviceTensor> recurrent_conv_out_batch_;
        std::unique_ptr<DeviceTensor> recurrent_state_batch_;
        std::unique_ptr<DeviceTensor> recurrent_conv_state_batch_;
        std::unique_ptr<DeviceTensor> final_hidden_batch_;
        std::unique_ptr<DeviceTensor> final_norm_batch_;
        std::unique_ptr<DeviceTensor> final_logits_batch_;
        std::string last_mode_ = "delegated";
        BatchedPrefillTiming last_timing_;
        uint32_t last_kernel_batch_ = 1;
    };

    class BatchedDecodeExecutor {
    public:
        BatchedDecodeExecutor(const QwenNativeModel &model,
                              const QwenWeights &weights,
                              DeviceBackend &backend)
            : model_(model), weights_(weights), backend_(backend) {}

        const std::string &last_mode() const { return last_mode_; }
        uint32_t last_kernel_batch() const { return last_kernel_batch_; }
        bool last_body_batch_ready() const { return last_body_batch_ready_; }
        bool last_ragged_metadata_ready() const { return last_ragged_metadata_ready_; }
        uint32_t last_ragged_metadata_pages() const { return last_ragged_metadata_pages_; }
        uint32_t last_ragged_metadata_max_seq_len() const {
            return last_ragged_metadata_max_seq_len_;
        }
        const BatchedDecodeTiming &last_timing() const { return last_timing_; }

        std::vector<BatchedDecodeOutput> decode(
                std::vector<ContinuousBatchActive> &active,
                const BatchedDecodeInput &input) {
            std::vector<BatchedDecodeOutput> outputs;
            if (input.batch == nullptr) return outputs;
            const ContinuousDecodeBatch &batch = *input.batch;
            reset_last_ragged_metadata();
            last_timing_ = {};
            if (continuous_batching_body_batch_enabled() &&
                can_use_body_batch(active, batch)) {
                return decode_body_batch(active, batch);
            }
            if (can_use_lm_head_batch(active, batch)) {
                return decode_lm_head_batch(active, batch);
            }
            last_mode_ = "delegated";
            last_kernel_batch_ = 1;
            outputs.reserve(batch.size());
            const double t0 = wall_seconds();
            for (size_t batch_i = 0; batch_i < batch.size(); ++batch_i) {
                BatchedDecodeOutput out;
                out.active_index = batch.active_indices[batch_i];
                out.feed_token = batch.feed_tokens[batch_i];
                if (out.active_index >= active.size()) {
                    out.error = "decode active index out of range";
                    outputs.push_back(std::move(out));
                    continue;
                }
                try {
                    out.report =
                        active[out.active_index].executor->forward_one_token(
                            out.feed_token);
                    if (!out.report.ok) out.error = "decode failed";
                } catch (const std::exception &e) {
                    out.error = e.what();
                }
                outputs.push_back(std::move(out));
            }
            last_timing_.total_s = std::max(wall_seconds() - t0, 0.0);
            return outputs;
        }

    private:
        static bool request_needs_logits(const GenerationOptions &options) {
            return options.temperature > 0.0f ||
                   options.presence_penalty != 0.0f ||
                   (options.repetition_penalty > 0.0f &&
                    options.repetition_penalty != 1.0f);
        }

        bool can_use_lm_head_batch(const std::vector<ContinuousBatchActive> &active,
                                   const ContinuousDecodeBatch &batch) const {
            if (!continuous_batching_lm_head_batch_enabled()) return false;
            if (batch.size() < 2) return false;
            for (size_t batch_i = 0; batch_i < batch.size(); ++batch_i) {
                const size_t active_index = batch.active_indices[batch_i];
                if (active_index >= active.size()) return false;
                const ContinuousBatchActive &a = active[active_index];
                if (!a.req || !a.executor) return false;
                if (request_needs_logits(a.req->options)) return false;
            }
            return true;
        }

        bool can_use_body_batch(const std::vector<ContinuousBatchActive> &active,
                                const ContinuousDecodeBatch &batch) const {
            if (!can_use_lm_head_batch(active, batch)) return false;
            // kvmem sequences must run the per-seq forward_one_token path: the
            // body-batch executor reads each request's live page table directly
            // and has no window/re-RoPE branch, so batching a kvmem seq would
            // attend over the full cache instead of the assembled window. The
            // delegated + lm_head paths call forward_one_token, which honors the
            // kvmem window. (Phase D adds a window-aware body batch.)
            for (size_t batch_i = 0; batch_i < batch.size(); ++batch_i) {
                const size_t active_index = batch.active_indices[batch_i];
                if (active_index >= active.size()) return false;
                const ContinuousBatchActive &a = active[active_index];
                if (a.executor && a.executor->kvmem_enabled()) return false;
            }
            const std::string kv_dtype = env_lower_ascii(env_value("QW3_KV_DTYPE"));
            if (!kv_dtype.empty() && kv_dtype != "fp16" && kv_dtype != "fp8") return false;
            for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
                const QwenLayerWeights &layer = weights_.layer(il);
                if (!layer.recurrent && (!layer.attn_q || !layer.attn_k ||
                                         !layer.attn_v || !layer.attn_output)) {
                    return false;
                }
            }
            return true;
        }

        void ensure_lm_head_scratch(uint32_t batch, uint32_t hidden, uint32_t vocab) {
            if (batch == 0 || batch <= lm_head_batch_capacity_) return;
            hidden_batch_ = backend_.scratch_f32(static_cast<uint64_t>(batch) * hidden,
                                                 "cb_decode_hidden_batch");
            norm_batch_ = backend_.scratch_f32(static_cast<uint64_t>(batch) * hidden,
                                               "cb_decode_norm_batch");
            logits_batch_ = backend_.scratch_f32(static_cast<uint64_t>(batch) * vocab,
                                                 "cb_decode_logits_batch");
            lm_head_batch_capacity_ = batch;
        }

        void ensure_body_scratch(uint32_t batch) {
            if (batch == 0 || batch <= body_batch_capacity_) return;
            const QwenConfig &cfg = model_.config();
            uint64_t max_ffn = 1;
            uint64_t max_q = 1;
            uint64_t max_k = 1;
            uint64_t max_v = 1;
            uint64_t max_recurrent_qkv = 1;
            uint64_t max_recurrent_value = 1;
            for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
                const QwenLayerWeights &layer = weights_.layer(il);
                max_ffn = std::max<uint64_t>(max_ffn, layer.ffn_dim);
                max_q = std::max<uint64_t>(max_q, layer.q_rows);
                max_k = std::max<uint64_t>(max_k, layer.k_rows);
                max_v = std::max<uint64_t>(max_v, layer.v_rows);
                max_recurrent_qkv =
                    std::max<uint64_t>(max_recurrent_qkv,
                                       layer.recurrent_qkv_dim);
                max_recurrent_value =
                    std::max<uint64_t>(max_recurrent_value,
                                       layer.recurrent_value_dim);
            }
            const uint64_t B = batch;
            hidden_batch_ = backend_.scratch_f32(B * cfg.n_embd,
                                                 "cb_body_hidden_batch");
            norm_batch_ = backend_.scratch_f32(B * cfg.n_embd,
                                               "cb_body_norm_batch");
            attn_out_batch_ = backend_.scratch_f32(B * cfg.n_embd,
                                                   "cb_body_attn_out_batch");
            ffn_gate_batch_ = backend_.scratch_f32(B * max_ffn,
                                                   "cb_body_ffn_gate_batch");
            ffn_up_batch_ = backend_.scratch_f32(B * max_ffn,
                                                 "cb_body_ffn_up_batch");
            ffn_mid_batch_ = backend_.scratch_f32(B * max_ffn,
                                                  "cb_body_ffn_mid_batch");
            ffn_out_batch_ = backend_.scratch_f32(B * cfg.n_embd,
                                                  "cb_body_ffn_out_batch");
            q_batch_ = backend_.scratch_f32(B * max_q, "cb_body_q_batch");
            k_batch_ = backend_.scratch_f32(B * max_k, "cb_body_k_batch");
            v_batch_ = backend_.scratch_f32(B * max_v, "cb_body_v_batch");
            mid_batch_ = backend_.scratch_f32(
                B * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim,
                "cb_body_mid_batch");
            recurrent_proj_batch_ = backend_.scratch_f32(
                B * max_recurrent_qkv, "cb_body_recurrent_proj_batch");
            recurrent_gate_batch_ = backend_.scratch_f32(
                B * max_recurrent_value, "cb_body_recurrent_gate_batch");
            recurrent_alpha_batch_ = backend_.scratch_f32(
                B * cfg.num_v_heads(), "cb_body_recurrent_alpha_batch");
            recurrent_beta_batch_ = backend_.scratch_f32(
                B * cfg.num_v_heads(), "cb_body_recurrent_beta_batch");
            recurrent_core_batch_ = backend_.scratch_f32(
                B * max_recurrent_value, "cb_body_recurrent_core_batch");
            recurrent_conv_out_batch_ = backend_.scratch_f32(
                B * max_recurrent_qkv, "cb_body_recurrent_conv_out_batch");
            recurrent_state_batch_ = backend_.scratch_f32(
                B * static_cast<uint64_t>(cfg.num_v_heads()) *
                    cfg.head_v_dim_ssm() * cfg.head_k_dim(),
                "cb_body_recurrent_state_batch");
            recurrent_conv_state_batch_ = backend_.scratch_f32(
                B * max_recurrent_qkv *
                    static_cast<uint64_t>(cfg.ssm_conv_kernel - 1),
                "cb_body_recurrent_conv_state_batch");
            body_batch_capacity_ = batch;
        }

        void ensure_ragged_metadata_scratch(uint32_t batch, uint32_t pages) {
            if (batch > ragged_metadata_batch_capacity_) {
                ragged_positions_i32_ =
                    backend_.tensor_i32(batch, "cb_decode_positions_i32");
                ragged_page_indptr_i32_ =
                    backend_.tensor_i32(static_cast<uint64_t>(batch) + 1,
                                        "cb_decode_page_indptr_i32");
                ragged_last_page_len_i32_ =
                    backend_.tensor_i32(batch, "cb_decode_last_page_len_i32");
                ragged_seq_lens_i32_ =
                    backend_.tensor_i32(batch, "cb_decode_seq_lens_i32");
                ragged_metadata_batch_capacity_ = batch;
            }
            if (pages > ragged_metadata_page_capacity_) {
                ragged_page_indices_i32_ =
                    backend_.tensor_i32(pages, "cb_decode_page_indices_i32");
                ragged_metadata_page_capacity_ = pages;
            }
        }

        static void require_ok(const DeviceStatus &st) {
            if (!st.ok) throw std::runtime_error(st.message);
        }

        void reset_last_ragged_metadata() {
            last_body_batch_ready_ = false;
            last_ragged_metadata_ready_ = false;
            last_ragged_metadata_pages_ = 0;
            last_ragged_metadata_max_seq_len_ = 0;
        }

        bool prepare_body_batch_inputs(std::vector<ContinuousBatchActive> &active,
                                       const ContinuousDecodeBatch &batch) {
            if (batch.size() < 2) return false;
            uint32_t page_size = 0;
            for (size_t batch_i = 0; batch_i < batch.size(); ++batch_i) {
                const size_t active_index = batch.active_indices[batch_i];
                if (active_index >= active.size()) return false;
                ContinuousBatchActive &a = active[active_index];
                if (!a.executor) return false;
                a.executor->prepare_decode_token_pages(1);
                QwenExecutor::MutableDecodeStateView view =
                    a.executor->mutable_decode_state_view();
                if (view.hidden == nullptr ||
                    view.kv_page_size == 0 ||
                    view.kv_page_indices_host == nullptr ||
                    view.kv_page_indices_device == nullptr ||
                    view.kv_page_count == 0) {
                    return false;
                }
                if (page_size == 0) {
                    page_size = view.kv_page_size;
                } else if (page_size != view.kv_page_size) {
                    return false;
                }
            }
            last_body_batch_ready_ = true;
            return true;
        }

        bool pack_ragged_metadata_after_body(
                const std::vector<ContinuousBatchActive> &active,
                const std::vector<BatchedDecodeOutput> &outputs) {
            last_ragged_metadata_ready_ = false;
            last_ragged_metadata_pages_ = 0;
            last_ragged_metadata_max_seq_len_ = 0;
            const uint32_t bsz = static_cast<uint32_t>(outputs.size());
            if (bsz == 0) return false;

            uint32_t page_size = 0;
            uint32_t total_pages = 0;
            uint32_t max_seq_len = 0;
            ragged_positions_h_.assign(bsz, 0);
            ragged_page_indptr_h_.assign(static_cast<size_t>(bsz) + 1, 0);
            ragged_last_page_len_h_.assign(bsz, 0);
            ragged_seq_lens_h_.assign(bsz, 0);
            ragged_page_indices_h_.clear();

            for (uint32_t row = 0; row < bsz; ++row) {
                const BatchedDecodeOutput &out = outputs[row];
                if (!out.error.empty() || out.active_index >= active.size()) {
                    return false;
                }
                const QwenExecutor::DecodeStateView view =
                    active[out.active_index].executor->decode_state_view();
                const uint32_t seq_len = view.position;
                if (seq_len == 0 || view.kv_page_size == 0 ||
                    view.kv_page_indices_host == nullptr) {
                    return false;
                }
                if (page_size == 0) {
                    page_size = view.kv_page_size;
                } else if (page_size != view.kv_page_size) {
                    return false;
                }
                const uint32_t pages =
                    (seq_len + page_size - 1) / page_size;
                if (pages == 0 || view.kv_page_count < pages) {
                    return false;
                }
                ragged_positions_h_[row] = static_cast<int32_t>(seq_len - 1);
                ragged_seq_lens_h_[row] = static_cast<int32_t>(seq_len);
                const uint32_t last_len = seq_len % page_size;
                ragged_last_page_len_h_[row] =
                    static_cast<int32_t>(last_len == 0 ? page_size : last_len);
                ragged_page_indptr_h_[row] =
                    static_cast<int32_t>(ragged_page_indices_h_.size());
                for (uint32_t p = 0; p < pages; ++p) {
                    ragged_page_indices_h_.push_back(view.kv_page_indices_host[p]);
                }
                total_pages += pages;
                max_seq_len = std::max(max_seq_len, seq_len);
            }
            ragged_page_indptr_h_[bsz] =
                static_cast<int32_t>(ragged_page_indices_h_.size());
            if (total_pages != ragged_page_indices_h_.size()) return false;

            ensure_ragged_metadata_scratch(bsz, total_pages);
            require_ok(backend_.copy_i32_from_host(
                *ragged_positions_i32_, 0, ragged_positions_h_.data(), bsz));
            require_ok(backend_.copy_i32_from_host(
                *ragged_page_indices_i32_, 0,
                ragged_page_indices_h_.data(), total_pages));
            require_ok(backend_.copy_i32_from_host(
                *ragged_page_indptr_i32_, 0,
                ragged_page_indptr_h_.data(), static_cast<uint64_t>(bsz) + 1));
            require_ok(backend_.copy_i32_from_host(
                *ragged_last_page_len_i32_, 0,
                ragged_last_page_len_h_.data(), bsz));
            require_ok(backend_.copy_i32_from_host(
                *ragged_seq_lens_i32_, 0, ragged_seq_lens_h_.data(), bsz));

            last_ragged_metadata_ready_ = true;
            last_ragged_metadata_pages_ = total_pages;
            last_ragged_metadata_max_seq_len_ = max_seq_len;
            return true;
        }

        bool pack_ragged_metadata_for_body(
                std::vector<ContinuousBatchActive> &active,
                const ContinuousDecodeBatch &batch) {
            last_ragged_metadata_ready_ = false;
            last_ragged_metadata_pages_ = 0;
            last_ragged_metadata_max_seq_len_ = 0;
            const uint32_t bsz = static_cast<uint32_t>(batch.size());
            if (bsz == 0) return false;

            uint32_t page_size = 0;
            uint32_t total_pages = 0;
            uint32_t max_seq_len = 0;
            ragged_positions_h_.assign(bsz, 0);
            ragged_page_indptr_h_.assign(static_cast<size_t>(bsz) + 1, 0);
            ragged_last_page_len_h_.assign(bsz, 0);
            ragged_seq_lens_h_.assign(bsz, 0);
            ragged_page_indices_h_.clear();

            for (uint32_t row = 0; row < bsz; ++row) {
                const size_t active_index = batch.active_indices[row];
                if (active_index >= active.size() || !active[active_index].executor) {
                    return false;
                }
                active[active_index].executor->prepare_decode_token_pages(1);
                QwenExecutor::MutableDecodeStateView view =
                    active[active_index].executor->mutable_decode_state_view();
                const uint32_t old_pos = view.position;
                const uint32_t seq_len = old_pos + 1;
                if (view.kv_page_size == 0 || view.kv_page_indices_host == nullptr) {
                    return false;
                }
                if (page_size == 0) {
                    page_size = view.kv_page_size;
                } else if (page_size != view.kv_page_size) {
                    return false;
                }
                const uint32_t pages = (seq_len + page_size - 1) / page_size;
                if (pages == 0 || view.kv_page_count < pages) return false;
                ragged_positions_h_[row] = static_cast<int32_t>(old_pos);
                ragged_seq_lens_h_[row] = static_cast<int32_t>(seq_len);
                const uint32_t last_len = seq_len % page_size;
                ragged_last_page_len_h_[row] =
                    static_cast<int32_t>(last_len == 0 ? page_size : last_len);
                ragged_page_indptr_h_[row] =
                    static_cast<int32_t>(ragged_page_indices_h_.size());
                for (uint32_t p = 0; p < pages; ++p) {
                    ragged_page_indices_h_.push_back(view.kv_page_indices_host[p]);
                }
                total_pages += pages;
                max_seq_len = std::max(max_seq_len, seq_len);
            }
            ragged_page_indptr_h_[bsz] =
                static_cast<int32_t>(ragged_page_indices_h_.size());
            if (total_pages != ragged_page_indices_h_.size()) return false;

            ensure_ragged_metadata_scratch(bsz, total_pages);
            require_ok(backend_.copy_i32_from_host(
                *ragged_positions_i32_, 0, ragged_positions_h_.data(), bsz));
            require_ok(backend_.copy_i32_from_host(
                *ragged_page_indices_i32_, 0,
                ragged_page_indices_h_.data(), total_pages));
            require_ok(backend_.copy_i32_from_host(
                *ragged_page_indptr_i32_, 0,
                ragged_page_indptr_h_.data(), static_cast<uint64_t>(bsz) + 1));
            require_ok(backend_.copy_i32_from_host(
                *ragged_last_page_len_i32_, 0,
                ragged_last_page_len_h_.data(), bsz));
            require_ok(backend_.copy_i32_from_host(
                *ragged_seq_lens_i32_, 0, ragged_seq_lens_h_.data(), bsz));

            last_ragged_metadata_ready_ = true;
            last_ragged_metadata_pages_ = total_pages;
            last_ragged_metadata_max_seq_len_ = max_seq_len;
            return true;
        }

        static DeviceTensor *layer_k_cache(QwenExecutor::MutableDecodeStateView &view,
                                           uint32_t layer_index) {
            if (view.k_cache_external && layer_index < view.k_cache_external->size()) {
                return (*view.k_cache_external)[layer_index];
            }
            if (view.k_cache && layer_index < view.k_cache->size() &&
                (*view.k_cache)[layer_index]) {
                return (*view.k_cache)[layer_index].get();
            }
            return nullptr;
        }

        static DeviceTensor *layer_v_cache(QwenExecutor::MutableDecodeStateView &view,
                                           uint32_t layer_index) {
            if (view.v_cache_external && layer_index < view.v_cache_external->size()) {
                return (*view.v_cache_external)[layer_index];
            }
            if (view.v_cache && layer_index < view.v_cache->size() &&
                (*view.v_cache)[layer_index]) {
                return (*view.v_cache)[layer_index].get();
            }
            return nullptr;
        }

        std::vector<BatchedDecodeOutput> decode_body_batch(
                std::vector<ContinuousBatchActive> &active,
                const ContinuousDecodeBatch &batch) {
            std::vector<BatchedDecodeOutput> outputs;
            outputs.reserve(batch.size());
            const double t_total0 = wall_seconds();
            last_mode_ = "body_batch_fp16";
            last_kernel_batch_ = static_cast<uint32_t>(batch.size());
            last_body_batch_ready_ = false;
            const uint32_t bsz = static_cast<uint32_t>(batch.size());
            const QwenConfig &cfg = model_.config();
            const uint32_t hidden = cfg.n_embd;
            const uint32_t vocab = static_cast<uint32_t>(weights_.output().rows);
            const uint32_t standard_head_dim = cfg.head_dim;
            const uint32_t standard_n_heads = cfg.n_heads;
            const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
            const uint32_t q_stride = 2 * standard_head_dim;
            const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
            const uint32_t mid_stride = standard_n_heads * standard_head_dim;
            const float eps = cfg.rms_eps;
            const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));

            try {
                const double t_prepare0 = wall_seconds();
                require_ok(backend_.begin());
                if (!prepare_body_batch_inputs(active, batch)) {
                    throw std::runtime_error("body batch inputs unavailable");
                }
                ensure_body_scratch(bsz);
                ensure_lm_head_scratch(bsz, hidden, vocab);
                const double t_prepare1 = wall_seconds();
                if (!pack_ragged_metadata_for_body(active, batch)) {
                    throw std::runtime_error("body batch ragged metadata unavailable");
                }
                const double t_metadata1 = wall_seconds();

                std::vector<uint64_t> rows_h(bsz, 0);
                for (uint32_t row = 0; row < bsz; ++row) {
                    rows_h[row] = batch.feed_tokens[row];
                    BatchedDecodeOutput out;
                    out.active_index = batch.active_indices[row];
                    out.feed_token = batch.feed_tokens[row];
                    outputs.push_back(std::move(out));
                }

                require_ok(backend_.q8_0_get_rows_batch(
                    *hidden_batch_, weights_.token_embd(), rows_h.data(), bsz));
                const double t_embed1 = wall_seconds();

                const double t_layers0 = wall_seconds();
                double recurrent_s = 0.0;
                double recurrent_state_s = 0.0;
                double attention_s = 0.0;
                double qkv_s = 0.0;
                double kv_append_s = 0.0;
                double attn_kernel_s = 0.0;
                double attn_output_s = 0.0;
                double ffn_s = 0.0;
                for (uint32_t il = 0; il < weights_.n_layers(); ++il) {
                    const QwenLayerWeights &layer = weights_.layer(il);
                    if (layer.recurrent) {
                        const double t_recurrent0 = wall_seconds();
                        if (!continuous_batching_recurrent_batch_enabled()) {
                            for (uint32_t row = 0; row < bsz; ++row) {
                                ContinuousBatchActive &a = active[outputs[row].active_index];
                                QwenExecutor::MutableDecodeStateView view =
                                    a.executor->mutable_decode_state_view();
                                require_ok(backend_.copy_d2d_into(
                                    *view.hidden, 0, *hidden_batch_,
                                    static_cast<uint64_t>(row) * hidden, hidden));
                            }
                            require_ok(backend_.end());
                            for (uint32_t row = 0; row < bsz; ++row) {
                                ContinuousBatchActive &a = active[outputs[row].active_index];
                                NativeExecutorReport r =
                                    a.executor->forward_recurrent_layer_from_current_hidden(il);
                                if (!r.ok) throw std::runtime_error("recurrent layer failed");
                                outputs[row].report.ops_executed += r.ops_executed;
                            }
                            require_ok(backend_.begin());
                            for (uint32_t row = 0; row < bsz; ++row) {
                                ContinuousBatchActive &a = active[outputs[row].active_index];
                                QwenExecutor::MutableDecodeStateView view =
                                    a.executor->mutable_decode_state_view();
                                require_ok(backend_.copy_d2d_into(
                                    *hidden_batch_, static_cast<uint64_t>(row) * hidden,
                                    *view.hidden, 0, hidden));
                            }
                            recurrent_s +=
                                std::max(wall_seconds() - t_recurrent0, 0.0);
                            continue;
                        }
                        const uint32_t num_k_heads = cfg.num_k_heads();
                        const uint32_t num_v_heads = cfg.num_v_heads();
                        const uint32_t head_k_dim = cfg.head_k_dim();
                        const uint32_t head_v_dim = cfg.head_v_dim_ssm();
                        const uint32_t proj_stride =
                            static_cast<uint32_t>(layer.recurrent_qkv_dim);
                        const uint32_t gate_stride =
                            static_cast<uint32_t>(layer.recurrent_value_dim);
                        const uint32_t alpha_stride = num_v_heads;
                        const uint32_t beta_stride = num_v_heads;
                        const uint32_t core_stride =
                            static_cast<uint32_t>(layer.recurrent_value_dim);
                        const uint32_t state_stride =
                            num_v_heads * head_v_dim * head_k_dim;
                        const uint32_t conv_state_stride =
                            proj_stride * (cfg.ssm_conv_kernel - 1);
                        if (proj_stride == 0 || gate_stride == 0 ||
                            core_stride == 0 || state_stride == 0 ||
                            conv_state_stride == 0) {
                            throw std::runtime_error("recurrent layer shape unavailable");
                        }
                        require_ok(backend_.rms_norm_batch(
                            *norm_batch_, *hidden_batch_, *layer.attn_norm,
                            bsz, hidden, eps));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_proj_batch_, *layer.attn_qkv,
                            *norm_batch_, bsz, hidden, proj_stride));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_gate_batch_, *layer.attn_gate,
                            *norm_batch_, bsz, hidden, gate_stride));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_alpha_batch_, *layer.ssm_alpha,
                            *norm_batch_, bsz, hidden, alpha_stride));
                        require_ok(backend_.q8_0_matmul(
                            *recurrent_beta_batch_, *layer.ssm_beta,
                            *norm_batch_, bsz, hidden, beta_stride));
                        const double t_state_pack0 = wall_seconds();
                        for (uint32_t row = 0; row < bsz; ++row) {
                            ContinuousBatchActive &a = active[outputs[row].active_index];
                            QwenExecutor::MutableDecodeStateView view =
                                a.executor->mutable_decode_state_view();
                            if (!view.recurrent_states || !view.conv_states ||
                                il >= view.recurrent_states->size() ||
                                il >= view.conv_states->size() ||
                                !(*view.recurrent_states)[il] ||
                                !(*view.conv_states)[il]) {
                                throw std::runtime_error("body batch recurrent state unavailable");
                            }
                            DeviceTensor &state = *(*view.recurrent_states)[il];
                            DeviceTensor &conv_state = *(*view.conv_states)[il];
                            if (state.count < state_stride ||
                                conv_state.count < conv_state_stride) {
                                throw std::runtime_error("body batch recurrent state too small");
                            }
                            require_ok(backend_.copy_d2d_into(
                                *recurrent_state_batch_,
                                static_cast<uint64_t>(row) * state_stride,
                                state, 0, state_stride));
                            require_ok(backend_.copy_d2d_into(
                                *recurrent_conv_state_batch_,
                                static_cast<uint64_t>(row) * conv_state_stride,
                                conv_state, 0, conv_state_stride));
                        }
                        recurrent_state_s +=
                            std::max(wall_seconds() - t_state_pack0, 0.0);
                        require_ok(backend_.recurrent_batch_independent(
                            *recurrent_core_batch_,
                            *recurrent_state_batch_,
                            *recurrent_conv_state_batch_,
                            *recurrent_conv_out_batch_,
                            *recurrent_proj_batch_,
                            *recurrent_gate_batch_,
                            *recurrent_alpha_batch_,
                            *recurrent_beta_batch_,
                            *layer.ssm_conv1d,
                            *layer.ssm_a,
                            *layer.ssm_dt_bias,
                            *layer.ssm_norm,
                            bsz, num_k_heads, num_v_heads,
                            head_k_dim, head_v_dim, cfg.ssm_conv_kernel,
                            proj_stride, proj_stride, gate_stride,
                            alpha_stride, beta_stride, core_stride,
                            state_stride, conv_state_stride, eps));
                        require_ok(backend_.q8_0_matmul_add(
                            *hidden_batch_, *hidden_batch_, *attn_out_batch_,
                            *layer.ssm_out, *recurrent_core_batch_,
                            bsz, core_stride, hidden));
                        const double t_state_unpack0 = wall_seconds();
                        for (uint32_t row = 0; row < bsz; ++row) {
                            ContinuousBatchActive &a = active[outputs[row].active_index];
                            QwenExecutor::MutableDecodeStateView view =
                                a.executor->mutable_decode_state_view();
                            DeviceTensor &state = *(*view.recurrent_states)[il];
                            DeviceTensor &conv_state = *(*view.conv_states)[il];
                            require_ok(backend_.copy_d2d_into(
                                state, 0, *recurrent_state_batch_,
                                static_cast<uint64_t>(row) * state_stride,
                                state_stride));
                            require_ok(backend_.copy_d2d_into(
                                conv_state, 0, *recurrent_conv_state_batch_,
                                static_cast<uint64_t>(row) * conv_state_stride,
                                conv_state_stride));
                            outputs[row].report.ops_executed += 1;
                        }
                        recurrent_state_s +=
                            std::max(wall_seconds() - t_state_unpack0, 0.0);
                        const double t_ffn0 = wall_seconds();
                        require_ok(backend_.rms_norm_batch(
                            *norm_batch_, *hidden_batch_, *layer.ffn_norm,
                            bsz, hidden, eps));
                        require_ok(backend_.q8_0_matmul(
                            *ffn_gate_batch_, *layer.ffn_gate,
                            *norm_batch_, bsz, hidden,
                            static_cast<uint32_t>(layer.ffn_dim)));
                        require_ok(backend_.q8_0_matmul(
                            *ffn_up_batch_, *layer.ffn_up,
                            *norm_batch_, bsz, hidden,
                            static_cast<uint32_t>(layer.ffn_dim)));
                        require_ok(backend_.silu_mul_n(
                            *ffn_mid_batch_, *ffn_gate_batch_, *ffn_up_batch_,
                            static_cast<uint64_t>(bsz) * layer.ffn_dim));
                        require_ok(backend_.q8_0_matmul(
                            *ffn_out_batch_, *layer.ffn_down,
                            *ffn_mid_batch_, bsz,
                            static_cast<uint32_t>(layer.ffn_dim), hidden));
                        require_ok(backend_.add_n(
                            *hidden_batch_, *hidden_batch_, *ffn_out_batch_,
                            static_cast<uint64_t>(bsz) * hidden));
                        ffn_s += std::max(wall_seconds() - t_ffn0, 0.0);
                        recurrent_s +=
                            std::max(wall_seconds() - t_recurrent0, 0.0);
                        continue;
                    }

                    const double t_attention0 = wall_seconds();
                    require_ok(backend_.rms_norm_batch(
                        *norm_batch_, *hidden_batch_, *layer.attn_norm,
                        bsz, hidden, eps));
                    DeviceTensor *qkv_outs[3] = {
                        q_batch_.get(), k_batch_.get(), v_batch_.get()
                    };
                    const DeviceWeight *qkv_ws[3] = {
                        layer.attn_q, layer.attn_k, layer.attn_v
                    };
                    const uint32_t qkv_strides[3] = {
                        static_cast<uint32_t>(layer.q_rows),
                        static_cast<uint32_t>(layer.k_rows),
                        static_cast<uint32_t>(layer.v_rows),
                    };
                    const double t_qkv0 = wall_seconds();
                    require_ok(backend_.q8_0_matmul_fanout(
                        qkv_outs, qkv_ws, qkv_strides, 3,
                        *norm_batch_, bsz, hidden));
                    require_ok(backend_.rmsnorm_per_head_batch(
                        *q_batch_, *layer.attn_q_norm, bsz,
                        static_cast<uint32_t>(layer.q_rows), standard_n_heads,
                        q_stride, standard_head_dim, eps));
                    require_ok(backend_.rmsnorm_per_head_batch(
                        *k_batch_, *layer.attn_k_norm, bsz,
                        static_cast<uint32_t>(layer.k_rows), standard_n_kv_heads,
                        standard_head_dim, standard_head_dim, eps));
                    require_ok(backend_.rope_partial_batch_positions(
                        *q_batch_, bsz, static_cast<uint32_t>(layer.q_rows),
                        standard_n_heads, q_stride, cfg.rope_dim,
                        *ragged_positions_i32_, cfg.rope_theta));
                    require_ok(backend_.rope_partial_batch_positions(
                        *k_batch_, bsz, static_cast<uint32_t>(layer.k_rows),
                        standard_n_kv_heads, standard_head_dim, cfg.rope_dim,
                        *ragged_positions_i32_, cfg.rope_theta));
                    qkv_s += std::max(wall_seconds() - t_qkv0, 0.0);

                    QwenExecutor::MutableDecodeStateView first_view =
                        active[outputs[0].active_index].executor->mutable_decode_state_view();
                    DeviceTensor *k_cache = layer_k_cache(first_view, il);
                    DeviceTensor *v_cache = layer_v_cache(first_view, il);
                    if (!k_cache || !v_cache) {
                        throw std::runtime_error("body batch KV cache unavailable");
                    }
                    const double t_kv_append0 = wall_seconds();
                    require_ok(backend_.kv_append_batch_paged_ragged_device(
                        *k_cache, *k_batch_, *ragged_positions_i32_, per_pos, bsz,
                        static_cast<uint32_t>(layer.k_rows), *ragged_page_indices_i32_,
                        *ragged_page_indptr_i32_, first_view.kv_page_size));
                    require_ok(backend_.kv_append_batch_paged_ragged_device(
                        *v_cache, *v_batch_, *ragged_positions_i32_, per_pos, bsz,
                        static_cast<uint32_t>(layer.v_rows), *ragged_page_indices_i32_,
                        *ragged_page_indptr_i32_, first_view.kv_page_size));
                    kv_append_s += std::max(wall_seconds() - t_kv_append0, 0.0);
                    const double t_attn_kernel0 = wall_seconds();
                    require_ok(backend_.attention_decode_batch_paged_gated_ragged_device(
                        *mid_batch_, *q_batch_, q_stride, *k_cache, *v_cache,
                        *ragged_page_indices_i32_, *ragged_page_indptr_i32_,
                        *ragged_last_page_len_i32_, *ragged_seq_lens_i32_,
                        ragged_page_indptr_h_.data(),
                        ragged_last_page_len_h_.data(),
                        ragged_seq_lens_h_.data(),
                        first_view.kv_page_size, standard_n_heads,
                        standard_n_kv_heads, standard_head_dim, bsz,
                        static_cast<uint32_t>(layer.q_rows), mid_stride, scale));
                    attn_kernel_s +=
                        std::max(wall_seconds() - t_attn_kernel0, 0.0);
                    const double t_attn_output0 = wall_seconds();
                    require_ok(backend_.q8_0_matmul(
                        *attn_out_batch_, *layer.attn_output, *mid_batch_,
                        bsz, mid_stride, hidden));
                    require_ok(backend_.add_n(
                        *hidden_batch_, *hidden_batch_, *attn_out_batch_,
                        static_cast<uint64_t>(bsz) * hidden));
                    attn_output_s +=
                        std::max(wall_seconds() - t_attn_output0, 0.0);
                    attention_s +=
                        std::max(wall_seconds() - t_attention0, 0.0);

                    const double t_ffn0 = wall_seconds();
                    require_ok(backend_.rms_norm_batch(
                        *norm_batch_, *hidden_batch_, *layer.ffn_norm,
                        bsz, hidden, eps));
                    require_ok(backend_.q8_0_matmul(
                        *ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                        bsz, hidden, static_cast<uint32_t>(layer.ffn_dim)));
                    require_ok(backend_.q8_0_matmul(
                        *ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                        bsz, hidden, static_cast<uint32_t>(layer.ffn_dim)));
                    require_ok(backend_.silu_mul_n(
                        *ffn_mid_batch_, *ffn_gate_batch_, *ffn_up_batch_,
                        static_cast<uint64_t>(bsz) * layer.ffn_dim));
                    require_ok(backend_.q8_0_matmul(
                        *ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                        bsz, static_cast<uint32_t>(layer.ffn_dim), hidden));
                    require_ok(backend_.add_n(
                        *hidden_batch_, *hidden_batch_, *ffn_out_batch_,
                        static_cast<uint64_t>(bsz) * hidden));
                    ffn_s += std::max(wall_seconds() - t_ffn0, 0.0);
                }
                const double t_layers1 = wall_seconds();

                for (uint32_t row = 0; row < bsz; ++row) {
                    ContinuousBatchActive &a = active[outputs[row].active_index];
                    QwenExecutor::MutableDecodeStateView view =
                        a.executor->mutable_decode_state_view();
                    require_ok(backend_.copy_d2d_into(
                        *view.hidden, 0, *hidden_batch_,
                        static_cast<uint64_t>(row) * hidden, hidden));
                    a.executor->advance_position(1);
                }
                require_ok(backend_.rms_norm_batch(
                    *norm_batch_, *hidden_batch_, weights_.output_norm(),
                    bsz, hidden, eps));
                const double t_lm_head0 = wall_seconds();
                require_ok(backend_.q8_0_matmul(
                    *logits_batch_, weights_.output(), *norm_batch_,
                    bsz, hidden, vocab));
                const double t_argmax0 = wall_seconds();
                std::vector<DeviceArgmax> argmaxes;
                require_ok(backend_.argmax_batch(*logits_batch_, bsz, vocab, argmaxes));
                require_ok(backend_.end());
                const double t_final1 = wall_seconds();

                for (uint32_t row = 0; row < bsz && row < argmaxes.size(); ++row) {
                    outputs[row].report.argmax_token = argmaxes[row].token;
                    outputs[row].report.argmax_logit = argmaxes[row].logit;
                    outputs[row].report.argmax_text =
                        model_.gguf().token_text(
                            static_cast<uint32_t>(argmaxes[row].token));
                    outputs[row].report.ok = true;
                    outputs[row].report.ops_executed += 1;
                }
                const double t_post1 = wall_seconds();
                last_timing_.prepare_s = std::max(t_prepare1 - t_prepare0, 0.0);
                last_timing_.metadata_s = std::max(t_metadata1 - t_prepare1, 0.0);
                last_timing_.embed_s = std::max(t_embed1 - t_metadata1, 0.0);
                last_timing_.layers_s = std::max(t_layers1 - t_layers0, 0.0);
                last_timing_.recurrent_s = recurrent_s;
                last_timing_.recurrent_state_s = recurrent_state_s;
                last_timing_.attention_s = attention_s;
                last_timing_.qkv_s = qkv_s;
                last_timing_.kv_append_s = kv_append_s;
                last_timing_.attn_kernel_s = attn_kernel_s;
                last_timing_.attn_output_s = attn_output_s;
                last_timing_.ffn_s = ffn_s;
                last_timing_.final_s = std::max(t_final1 - t_layers1, 0.0);
                last_timing_.lm_head_s = std::max(t_argmax0 - t_lm_head0, 0.0);
                last_timing_.argmax_s = std::max(t_final1 - t_argmax0, 0.0);
                last_timing_.post_s = std::max(t_post1 - t_final1, 0.0);
                last_timing_.total_s = std::max(t_post1 - t_total0, 0.0);
            } catch (const std::exception &e) {
                try { (void)backend_.end(); } catch (...) {}
                for (auto &out : outputs) {
                    if (out.error.empty()) out.error = e.what();
                }
                last_timing_.total_s = std::max(wall_seconds() - t_total0, 0.0);
            }
            return outputs;
        }

        std::vector<BatchedDecodeOutput> decode_lm_head_batch(
                std::vector<ContinuousBatchActive> &active,
                const ContinuousDecodeBatch &batch) {
            std::vector<BatchedDecodeOutput> outputs;
            outputs.reserve(batch.size());
            const double t_total0 = wall_seconds();
            last_mode_ = "lm_head_batch";
            last_kernel_batch_ = static_cast<uint32_t>(batch.size());
            const double t_prepare0 = wall_seconds();
            last_body_batch_ready_ = prepare_body_batch_inputs(active, batch);
            const double t_prepare1 = wall_seconds();

            uint32_t hidden = 0;
            const uint32_t vocab = static_cast<uint32_t>(weights_.output().rows);
            try {
                const double t_layers0 = wall_seconds();
                for (size_t batch_i = 0; batch_i < batch.size(); ++batch_i) {
                    BatchedDecodeOutput out;
                    out.active_index = batch.active_indices[batch_i];
                    out.feed_token = batch.feed_tokens[batch_i];
                    if (out.active_index >= active.size()) {
                        out.error = "decode active index out of range";
                        outputs.push_back(std::move(out));
                        continue;
                    }
                    ContinuousBatchActive &a = active[out.active_index];
                    try {
                        out.report = a.executor->forward_one_token(
                            out.feed_token, /*compute_logits=*/false);
                        if (!out.report.ok) out.error = "decode body failed";
                    } catch (const std::exception &e) {
                        out.error = e.what();
                    }
                    if (out.error.empty()) {
                        QwenExecutor::DecodeStateView view =
                            a.executor->decode_state_view();
                        if (view.hidden == nullptr || view.hidden->count == 0) {
                            out.error = "decode hidden state unavailable";
                        } else if (hidden == 0) {
                            hidden = static_cast<uint32_t>(view.hidden->count);
                        } else if (view.hidden->count != hidden) {
                            out.error = "decode hidden size mismatch";
                        }
                    }
                    outputs.push_back(std::move(out));
                }
                const double t_layers1 = wall_seconds();

                bool any_error = false;
                for (const auto &out : outputs) {
                    any_error = any_error || !out.error.empty();
                }
                if (any_error || hidden == 0) return outputs;

                const uint32_t bsz = static_cast<uint32_t>(batch.size());
                ensure_lm_head_scratch(bsz, hidden, vocab);
                require_ok(backend_.begin());
                const double t_metadata0 = wall_seconds();
                (void)pack_ragged_metadata_after_body(active, outputs);
                const double t_metadata1 = wall_seconds();
                for (uint32_t row = 0; row < bsz; ++row) {
                    ContinuousBatchActive &a = active[outputs[row].active_index];
                    QwenExecutor::DecodeStateView view =
                        a.executor->decode_state_view();
                    require_ok(backend_.copy_d2d_into(
                        *hidden_batch_, static_cast<uint64_t>(row) * hidden,
                        *view.hidden, 0, hidden));
                }
                const float eps = model_.config().rms_eps;
                require_ok(backend_.rms_norm_batch(
                    *norm_batch_, *hidden_batch_, weights_.output_norm(),
                    bsz, hidden, eps));
                const double t_lm_head0 = wall_seconds();
                require_ok(backend_.q8_0_matmul(
                    *logits_batch_, weights_.output(), *norm_batch_,
                    bsz, hidden, vocab));
                const double t_argmax0 = wall_seconds();
                std::vector<DeviceArgmax> argmaxes;
                require_ok(backend_.argmax_batch(*logits_batch_, bsz, vocab, argmaxes));
                require_ok(backend_.end());
                const double t_final1 = wall_seconds();
                for (uint32_t row = 0; row < bsz && row < argmaxes.size(); ++row) {
                    outputs[row].report.argmax_token = argmaxes[row].token;
                    outputs[row].report.argmax_logit = argmaxes[row].logit;
                    outputs[row].report.argmax_text =
                        model_.gguf().token_text(
                            static_cast<uint32_t>(argmaxes[row].token));
                    outputs[row].report.ok = true;
                    outputs[row].report.ops_executed += 3;
                }
                const double t_post1 = wall_seconds();
                last_timing_.prepare_s = std::max(t_prepare1 - t_prepare0, 0.0);
                last_timing_.metadata_s = std::max(t_metadata1 - t_metadata0, 0.0);
                last_timing_.layers_s = std::max(t_layers1 - t_layers0, 0.0);
                last_timing_.final_s = std::max(t_final1 - t_metadata1, 0.0);
                last_timing_.lm_head_s = std::max(t_argmax0 - t_lm_head0, 0.0);
                last_timing_.argmax_s = std::max(t_final1 - t_argmax0, 0.0);
                last_timing_.post_s = std::max(t_post1 - t_final1, 0.0);
                last_timing_.total_s = std::max(t_post1 - t_total0, 0.0);
            } catch (const std::exception &e) {
                try { (void)backend_.end(); } catch (...) {}
                for (auto &out : outputs) {
                    if (out.error.empty()) out.error = e.what();
                }
                last_timing_.total_s = std::max(wall_seconds() - t_total0, 0.0);
            }
            return outputs;
        }

        const QwenNativeModel &model_;
        const QwenWeights &weights_;
        DeviceBackend &backend_;
        uint32_t lm_head_batch_capacity_ = 0;
        uint32_t body_batch_capacity_ = 0;
        uint32_t ragged_metadata_batch_capacity_ = 0;
        uint32_t ragged_metadata_page_capacity_ = 0;
        BatchedDecodeTiming last_timing_;
        std::unique_ptr<DeviceTensor> hidden_batch_;
        std::unique_ptr<DeviceTensor> norm_batch_;
        std::unique_ptr<DeviceTensor> logits_batch_;
        std::unique_ptr<DeviceTensor> attn_out_batch_;
        std::unique_ptr<DeviceTensor> ffn_gate_batch_;
        std::unique_ptr<DeviceTensor> ffn_up_batch_;
        std::unique_ptr<DeviceTensor> ffn_mid_batch_;
        std::unique_ptr<DeviceTensor> ffn_out_batch_;
        std::unique_ptr<DeviceTensor> q_batch_;
        std::unique_ptr<DeviceTensor> k_batch_;
        std::unique_ptr<DeviceTensor> v_batch_;
        std::unique_ptr<DeviceTensor> mid_batch_;
        std::unique_ptr<DeviceTensor> recurrent_proj_batch_;
        std::unique_ptr<DeviceTensor> recurrent_gate_batch_;
        std::unique_ptr<DeviceTensor> recurrent_alpha_batch_;
        std::unique_ptr<DeviceTensor> recurrent_beta_batch_;
        std::unique_ptr<DeviceTensor> recurrent_core_batch_;
        std::unique_ptr<DeviceTensor> recurrent_conv_out_batch_;
        std::unique_ptr<DeviceTensor> recurrent_state_batch_;
        std::unique_ptr<DeviceTensor> recurrent_conv_state_batch_;
        std::unique_ptr<DeviceTensor> ragged_positions_i32_;
        std::unique_ptr<DeviceTensor> ragged_page_indices_i32_;
        std::unique_ptr<DeviceTensor> ragged_page_indptr_i32_;
        std::unique_ptr<DeviceTensor> ragged_last_page_len_i32_;
        std::unique_ptr<DeviceTensor> ragged_seq_lens_i32_;
        std::vector<int32_t> ragged_positions_h_;
        std::vector<int32_t> ragged_page_indices_h_;
        std::vector<int32_t> ragged_page_indptr_h_;
        std::vector<int32_t> ragged_last_page_len_h_;
        std::vector<int32_t> ragged_seq_lens_h_;
        std::string last_mode_ = "delegated";
        uint32_t last_kernel_batch_ = 1;
        bool last_body_batch_ready_ = false;
        bool last_ragged_metadata_ready_ = false;
        uint32_t last_ragged_metadata_pages_ = 0;
        uint32_t last_ragged_metadata_max_seq_len_ = 0;
    };

    static bool continuous_batch_request_supported(const GenerationOptions &options,
                                                   const DumpStream *dump) {
        return dump == nullptr && options.max_tokens >= 0;
    }

    void start_continuous_batch_worker() {
        std::lock_guard<std::mutex> lk(cb_mu_);
        if (cb_running_) return;
        // A previous worker may have exited on its own (e.g. an unrecoverable
        // error inside the loop) leaving cb_running_ == false but the thread
        // object still joinable. Reassigning a joinable std::thread calls
        // std::terminate, so join the dead worker before spawning a new one.
        if (cb_worker_.joinable()) cb_worker_.join();
        cb_stop_ = false;
        cb_running_ = true;
        cb_worker_ = std::thread([this]() { continuous_batch_worker_loop(); });
        log("native continuous_batching: enabled=true mode=batch_step_executor");
    }

    void stop_continuous_batch_worker() {
        {
            std::lock_guard<std::mutex> lk(cb_mu_);
            // If the worker self-exited (cb_running_ already false) the thread
            // object can still be joinable; fall through to join it so the
            // std::thread destructor never sees a joinable thread (which would
            // std::terminate). Only skip the stop-signal when still running.
            if (cb_running_) cb_stop_ = true;
        }
        cb_cv_.notify_all();
        if (cb_worker_.joinable()) cb_worker_.join();
        cb_running_ = false;
    }

    std::string generate_continuous_batched(const std::vector<uint32_t> &prompt_tokens,
                                            const GenerationOptions &options,
                                            const TokenCallback &on_text) {
        start_continuous_batch_worker();
        auto req = std::make_shared<ContinuousBatchRequest>();
        req->id = ++cb_request_counter_;
        req->prompt_tokens = prompt_tokens;
        req->options = options;
        req->on_text = on_text;
        req->spec_mtp = mtp_speculate_enabled(options_);
        req->trace_mtp = options_.native_mtp_trace || mtp_trace_enabled();
        req->active_mtp = req->spec_mtp || req->trace_mtp;
        // reserved_tokens is an admission-accounting figure only (it gates the
        // shared KV budget; it does NOT bound actual generation, which uses
        // options.max_tokens in the decode loop, nor actual KV pages, which the
        // page pool hands out on demand with eviction). Clients that omit
        // max_tokens get g.max_tokens defaulted to the whole remaining context
        // upstream, which would make a single request reserve the entire KV
        // pool and reject every concurrent request. Cap the per-request
        // reservation at the fair per-slot share of the budget (ctx /
        // max_active) so up to --max-active requests can always co-reside;
        // genuine physical pressure is still bounded by the page pool.
        {
            const uint32_t reserve_ctx = options_.ctx_size > 0
                ? static_cast<uint32_t>(options_.ctx_size)
                : 4096u;
            const uint32_t reserve_active =
                std::max(1u, continuous_batching_max_active());
            const uint64_t per_request_cap =
                std::max<uint64_t>(1, reserve_ctx / reserve_active);
            const uint64_t requested =
                static_cast<uint64_t>(prompt_tokens.size()) +
                static_cast<uint64_t>(std::max(0, options.max_tokens));
            req->reserved_tokens = std::min(requested, per_request_cap);
        }
        if (continuous_batching_trace_enabled()) {
            std::ostringstream msg;
            msg << "native continuous_request:"
                << " request=" << req->id
                << " prompt_tokens=" << prompt_tokens.size()
                << " max_tokens=" << options.max_tokens
                << " ignore_eos=" << (options.ignore_eos ? "true" : "false")
                << " active_mtp=" << (req->active_mtp ? "true" : "false")
                << " reserved_tokens=" << req->reserved_tokens;
            log(msg.str());
        }
        const uint32_t ctx_size = options_.ctx_size > 0
            ? static_cast<uint32_t>(options_.ctx_size)
            : 4096u;
        const uint32_t max_active = continuous_batching_max_active();
        const uint32_t max_pending = continuous_batching_max_pending();
        const uint64_t max_total_tokens =
            continuous_batching_max_total_tokens(ctx_size, max_active);
        {
            std::lock_guard<std::mutex> lk(cb_mu_);
            if (cb_pending_.size() >= max_pending) {
                throw std::runtime_error(
                    "continuous batching admission rejected: pending queue full (" +
                    std::to_string(cb_pending_.size()) + "/" +
                    std::to_string(max_pending) + ")");
            }
            if (req->reserved_tokens > max_total_tokens) {
                throw std::runtime_error(
                    "continuous batching admission rejected: request token reservation " +
                    std::to_string(req->reserved_tokens) +
                    " exceeds total token budget " +
                    std::to_string(max_total_tokens));
            }
            if (cb_reserved_tokens_ + req->reserved_tokens > max_total_tokens) {
                throw std::runtime_error(
                    "continuous batching admission rejected: total token budget exhausted " +
                    std::to_string(cb_reserved_tokens_) + "+" +
                    std::to_string(req->reserved_tokens) + ">" +
                    std::to_string(max_total_tokens));
            }
            cb_reserved_tokens_ += req->reserved_tokens;
            cb_pending_.push_back(req);
        }
        cb_cv_.notify_one();

        std::unique_lock<std::mutex> lk(req->mu);
        req->cv.wait(lk, [&]() { return req->done; });
        if (!req->error.empty()) throw std::runtime_error(req->error);
        return req->generated;
    }

    void complete_continuous_request(const std::shared_ptr<ContinuousBatchRequest> &req,
                                     std::string generated,
                                     std::string error = {}) {
        release_continuous_request_budget(req);
        {
            std::lock_guard<std::mutex> lk(req->mu);
            req->generated = std::move(generated);
            req->error = std::move(error);
            req->done = true;
        }
        req->cv.notify_one();
    }

    void release_continuous_request_budget(
            const std::shared_ptr<ContinuousBatchRequest> &req) {
        std::lock_guard<std::mutex> lk(cb_mu_);
        if (req->budget_released) return;
        req->budget_released = true;
        cb_reserved_tokens_ =
            req->reserved_tokens > cb_reserved_tokens_
                ? 0
                : cb_reserved_tokens_ - req->reserved_tokens;
    }

    void log_zero_decode_diagnostic(const char *path,
                                    const std::vector<uint32_t> &prompt_tokens,
                                    const NativeExecutorReport &step) {
        std::ostringstream msg;
        msg << "native zero_decode:"
            << " path=" << path
            << " prompt_tokens=" << prompt_tokens.size()
            << " argmax_token=" << step.argmax_token
            << " argmax_logit=" << std::fixed << std::setprecision(4)
            << step.argmax_logit;
        if (tokenizer_ && step.argmax_token >= 0) {
            msg << " argmax_text="
                << escape_text(tokenizer_->decode_one(step.argmax_token));
        }
        msg << " prompt_tail=[";
        const size_t begin = prompt_tokens.size() > 16 ? prompt_tokens.size() - 16 : 0;
        for (size_t i = begin; i < prompt_tokens.size(); ++i) {
            if (i != begin) msg << ",";
            msg << prompt_tokens[i];
        }
        msg << "]";
        log(msg.str());
    }

    void allocate_continuous_kv_cache(uint32_t pool_pages, uint32_t page_size) {
        const QwenConfig &cfg = model_->config();
        const uint64_t kv_per_pos =
            static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
        const uint64_t physical_slots =
            static_cast<uint64_t>(pool_pages) * page_size;
        const std::string kv_dtype = env_lower_ascii(env_value("QW3_KV_DTYPE"));
        const bool kv_use_fp32 = kv_dtype == "fp32";
        const bool kv_use_q8 = kv_dtype == "q8";
        const bool kv_use_fp8 = kv_dtype == "fp8";
        const bool kv_use_fp16 = !kv_use_fp32 && !kv_use_q8 && !kv_use_fp8;

        cb_k_cache_storage_.clear();
        cb_v_cache_storage_.clear();
        cb_k_cache_storage_.resize(weights_->n_layers());
        cb_v_cache_storage_.resize(weights_->n_layers());
        cb_kv_cache_view_.physical_slots = physical_slots;
        cb_kv_cache_view_.k_cache.assign(weights_->n_layers(), nullptr);
        cb_kv_cache_view_.v_cache.assign(weights_->n_layers(), nullptr);

        for (uint32_t il = 0; il < weights_->n_layers(); ++il) {
            if (!cfg.is_standard_attention_layer(il)) continue;
            const std::string klabel = "cb_k_cache_l" + std::to_string(il);
            const std::string vlabel = "cb_v_cache_l" + std::to_string(il);
            if (kv_use_q8) {
                cb_k_cache_storage_[il] = device_->tensor_q8_kv(
                    kv_per_pos * physical_slots, cfg.head_dim, klabel.c_str());
                cb_v_cache_storage_[il] = device_->tensor_q8_kv(
                    kv_per_pos * physical_slots, cfg.head_dim, vlabel.c_str());
            } else if (kv_use_fp8) {
                cb_k_cache_storage_[il] = device_->tensor_fp8_kv(
                    kv_per_pos * physical_slots, klabel.c_str());
                cb_v_cache_storage_[il] = device_->tensor_fp8_kv(
                    kv_per_pos * physical_slots, vlabel.c_str());
            } else if (kv_use_fp16) {
                cb_k_cache_storage_[il] = device_->tensor_f16(
                    kv_per_pos * physical_slots, klabel.c_str());
                cb_v_cache_storage_[il] = device_->tensor_f16(
                    kv_per_pos * physical_slots, vlabel.c_str());
            } else {
                cb_k_cache_storage_[il] = device_->tensor_f32(
                    kv_per_pos * physical_slots, klabel.c_str());
                cb_v_cache_storage_[il] = device_->tensor_f32(
                    kv_per_pos * physical_slots, vlabel.c_str());
            }
            cb_kv_cache_view_.k_cache[il] = cb_k_cache_storage_[il].get();
            cb_kv_cache_view_.v_cache[il] = cb_v_cache_storage_[il].get();
        }
        log("native continuous_batching: global KV cache pages=" +
            std::to_string(pool_pages) +
            " page_size=" + std::to_string(page_size) +
            " physical_slots=" + std::to_string(physical_slots));
    }

    void allocate_continuous_mtp_kv_cache(uint32_t pool_pages,
                                          uint32_t page_size) {
        const QwenConfig &cfg = model_->config();
        const uint64_t kv_per_pos =
            static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
        const uint64_t physical_slots =
            static_cast<uint64_t>(pool_pages) * page_size;
        const std::string kv_dtype = env_lower_ascii(env_value("QW3_KV_DTYPE"));
        const bool kv_use_fp32 = kv_dtype == "fp32";
        const bool kv_use_q8 = kv_dtype == "q8";
        const bool kv_use_fp8 = kv_dtype == "fp8";
        const bool kv_use_fp16 = !kv_use_fp32 && !kv_use_q8 && !kv_use_fp8;

        cb_mtp_k_cache_storage_.clear();
        cb_mtp_v_cache_storage_.clear();
        cb_mtp_k_cache_storage_.resize(1);
        cb_mtp_v_cache_storage_.resize(1);
        cb_mtp_kv_cache_view_.physical_slots = physical_slots;
        cb_mtp_kv_cache_view_.k_cache.assign(1, nullptr);
        cb_mtp_kv_cache_view_.v_cache.assign(1, nullptr);

        if (kv_use_q8) {
            cb_mtp_k_cache_storage_[0] = device_->tensor_q8_kv(
                kv_per_pos * physical_slots, cfg.head_dim,
                "cb_mtp_k_cache");
            cb_mtp_v_cache_storage_[0] = device_->tensor_q8_kv(
                kv_per_pos * physical_slots, cfg.head_dim,
                "cb_mtp_v_cache");
        } else if (kv_use_fp8) {
            cb_mtp_k_cache_storage_[0] = device_->tensor_fp8_kv(
                kv_per_pos * physical_slots, "cb_mtp_k_cache");
            cb_mtp_v_cache_storage_[0] = device_->tensor_fp8_kv(
                kv_per_pos * physical_slots, "cb_mtp_v_cache");
        } else if (kv_use_fp16) {
            cb_mtp_k_cache_storage_[0] = device_->tensor_f16(
                kv_per_pos * physical_slots, "cb_mtp_k_cache");
            cb_mtp_v_cache_storage_[0] = device_->tensor_f16(
                kv_per_pos * physical_slots, "cb_mtp_v_cache");
        } else {
            cb_mtp_k_cache_storage_[0] = device_->tensor_f32(
                kv_per_pos * physical_slots, "cb_mtp_k_cache");
            cb_mtp_v_cache_storage_[0] = device_->tensor_f32(
                kv_per_pos * physical_slots, "cb_mtp_v_cache");
        }
        cb_mtp_kv_cache_view_.k_cache[0] =
            cb_mtp_k_cache_storage_[0].get();
        cb_mtp_kv_cache_view_.v_cache[0] =
            cb_mtp_v_cache_storage_[0].get();
        log("native continuous_batching: global MTP KV cache pages=" +
            std::to_string(pool_pages) +
            " page_size=" + std::to_string(page_size) +
            " physical_slots=" + std::to_string(physical_slots));
    }

    void continuous_batch_worker_loop() {
        std::vector<ContinuousBatchActive> active;
        std::vector<ContinuousBatchActive> prefilling;
        try {
            DeviceStatus st = device_->begin();
            if (!st.ok) throw std::runtime_error(st.message);

            const uint32_t ctx_size = options_.ctx_size > 0
                ? static_cast<uint32_t>(options_.ctx_size)
                : 4096u;
            const int32_t eos = tokenizer_->eos_id();
            const uint32_t max_active = continuous_batching_max_active();
            const uint32_t prefill_burst =
                continuous_batching_prefill_burst(max_active);
            ContinuousDecodeBatch decode_batch;

            while (true) {
                std::deque<std::shared_ptr<ContinuousBatchRequest>> arrivals;
                {
                    std::unique_lock<std::mutex> lk(cb_mu_);
                    cb_cv_.wait(lk, [&]() {
                        return cb_stop_ || !cb_pending_.empty() ||
                               !active.empty() || !prefilling.empty();
                    });
                    if (cb_stop_ && cb_pending_.empty() && active.empty() &&
                        prefilling.empty()) {
                        break;
                    }
                    if (active.empty() && prefilling.empty() &&
                        !cb_pending_.empty()) {
                        const uint32_t wait_us =
                            continuous_batching_admission_wait_us();
                        if (wait_us > 0 && cb_pending_.size() < max_active) {
                            cb_cv_.wait_for(
                                lk, std::chrono::microseconds(wait_us), [&]() {
                                    return cb_stop_ ||
                                           cb_pending_.size() >= max_active;
                                });
                        }
                    }
                    arrivals.swap(cb_pending_);
                }

                while (!arrivals.empty() &&
                       active.size() + prefilling.size() < max_active) {
                    auto req = arrivals.front();
                    if (req && req->active_mtp) {
                        if (active.empty() && prefilling.empty()) {
                            std::vector<std::shared_ptr<ContinuousBatchRequest>>
                                mtp_reqs;
                            while (!arrivals.empty() &&
                                   mtp_reqs.size() < max_active &&
                                   arrivals.front() &&
                                   arrivals.front()->active_mtp) {
                                mtp_reqs.push_back(arrivals.front());
                                arrivals.pop_front();
                            }
                            if (!mtp_reqs.empty()) {
                                run_continuous_mtp_batch_requests(
                                    mtp_reqs, ctx_size, max_active);
                            }
                        }
                        break;
                    }
                    arrivals.pop_front();
                    try {
                        ContinuousBatchActive a;
                        initialize_continuous_active(a, req, ctx_size);
                        prefilling.push_back(std::move(a));
                    } catch (const std::exception &e) {
                        complete_continuous_request(req, {}, e.what());
                    }
                }
                if (!arrivals.empty()) {
                    std::lock_guard<std::mutex> lk(cb_mu_);
                    while (!arrivals.empty()) {
                        cb_pending_.push_front(arrivals.back());
                        arrivals.pop_back();
                    }
                }

                const bool had_active_decode = !active.empty();
                if (had_active_decode) {
                    build_continuous_decode_batch(active, decode_batch);
                    continuous_decode_batch_step(active, decode_batch, eos);
                }
                const uint32_t prefill_steps = std::min<uint32_t>(
                    static_cast<uint32_t>(prefilling.size()),
                    had_active_decode
                        ? continuous_batching_active_prefill_burst()
                        : prefill_burst);
                advance_continuous_prefill_batch(prefilling, active, eos,
                                                 prefill_steps);
            }

            st = device_->end();
            if (!st.ok) throw std::runtime_error(st.message);
            {
                std::lock_guard<std::mutex> lk(cb_mu_);
                cb_running_ = false;
                cb_stop_ = false;
            }
        } catch (const std::exception &e) {
            std::deque<std::shared_ptr<ContinuousBatchRequest>> pending;
            {
                std::lock_guard<std::mutex> lk(cb_mu_);
                pending.swap(cb_pending_);
                cb_running_ = false;
                cb_stop_ = false;
            }
            while (!pending.empty()) {
                complete_continuous_request(pending.front(), {}, e.what());
                pending.pop_front();
            }
            for (auto &a : active) {
                if (a.req) complete_continuous_request(a.req, {}, e.what());
            }
            for (auto &a : prefilling) {
                if (a.req) complete_continuous_request(a.req, {}, e.what());
            }
            log(std::string("native continuous_batching: worker_failed reason=\"") +
                e.what() + "\"");
        }
    }

    void run_continuous_mtp_request(
            const std::shared_ptr<ContinuousBatchRequest> &req,
            uint32_t ctx_size) {
        if (!req) return;
        try {
            if (continuous_batching_trace_enabled()) {
                std::ostringstream msg;
                msg << "native continuous_mtp:"
                    << " request=" << req->id
                    << " prompt_tokens=" << req->prompt_tokens.size()
                    << " max_tokens=" << req->options.max_tokens
                    << " spec=" << (req->spec_mtp ? "true" : "false")
                    << " trace=" << (req->trace_mtp ? "true" : "false")
                    << " mode=single_request_barrier";
                log(msg.str());
            }
            auto executor = std::make_unique<QwenExecutor>(
                *model_, *weights_, *device_, ctx_size,
                cb_kv_pool_.get(), &cb_kv_cache_view_,
                cb_mtp_kv_pool_.get(), &cb_mtp_kv_cache_view_);
            executor->set_prefill_chunk_override(options_.prefill_chunk);
            std::vector<uint32_t> prompt = req->prompt_tokens;
            std::string generated = generate_mtp(
                prompt, req->options, req->on_text, nullptr,
                req->spec_mtp, req->trace_mtp, executor.get(),
                /*manage_device_scope=*/false);
            complete_continuous_request(req, std::move(generated));
        } catch (const std::exception &e) {
            complete_continuous_request(req, {}, e.what());
        }
    }

    struct ContinuousMtpVerifyJob {
        size_t row = 0;
        uint32_t current = 0;
        uint32_t base_position = 0;
        std::vector<uint32_t> drafts;
        std::vector<uint32_t> verify_tokens;
        // snapshot/checkpoints live in the per-row MtpStats (persistent across
        // loop iterations) and are referenced by `row`, never owned here.
        bool layered_verified = false;
    };

    struct ContinuousMtpDraftStep {
        size_t row = 0;
        uint32_t input_token = 0;
        uint32_t cache_pos = 0;
        int32_t output_token = -1;
        float output_logit = 0.0f;
        NativeExecutorReport report;
    };

    void ensure_continuous_mtp_draft_scratch(uint32_t batch) {
        if (batch == 0 || batch <= cb_mtp_draft_capacity_) return;
        const QwenConfig &cfg = model_->config();
        const QwenMtpWeights *mtp = weights_->mtp();
        if (!mtp) throw std::runtime_error("MTP weights unavailable");
        const QwenLayerWeights &layer = mtp->layer;
        const uint64_t B = batch;
        cb_mtp_h_input_batch_ =
            device_->scratch_f32(B * cfg.n_embd, "cb_mtp_draft_h_input");
        cb_mtp_h_batch_ =
            device_->scratch_f32(B * cfg.n_embd, "cb_mtp_draft_h");
        cb_mtp_norm_batch_ =
            device_->scratch_f32(B * cfg.n_embd, "cb_mtp_draft_norm");
        cb_mtp_concat_batch_ = device_->scratch_f32(
            B * static_cast<uint64_t>(2) * cfg.n_embd,
            "cb_mtp_draft_concat");
        cb_mtp_q_batch_ = device_->scratch_f32(
            B * std::max<uint64_t>(layer.q_rows, 1),
            "cb_mtp_draft_q");
        cb_mtp_q_row_ =
            device_->scratch_f32(std::max<uint64_t>(layer.q_rows, 1),
                                 "cb_mtp_draft_q_row");
        cb_mtp_k_batch_ = device_->scratch_f32(
            B * std::max<uint64_t>(layer.k_rows, 1),
            "cb_mtp_draft_k");
        cb_mtp_v_batch_ = device_->scratch_f32(
            B * std::max<uint64_t>(layer.v_rows, 1),
            "cb_mtp_draft_v");
        cb_mtp_k_row_ =
            device_->scratch_f32(std::max<uint64_t>(layer.k_rows, 1),
                                 "cb_mtp_draft_k_row");
        cb_mtp_v_row_ =
            device_->scratch_f32(std::max<uint64_t>(layer.v_rows, 1),
                                 "cb_mtp_draft_v_row");
        cb_mtp_mid_batch_ = device_->scratch_f32(
            B * static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim,
            "cb_mtp_draft_mid");
        cb_mtp_mid_row_ = device_->scratch_f32(
            static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim,
            "cb_mtp_draft_mid_row");
        cb_mtp_ffn_gate_batch_ = device_->scratch_f32(
            B * std::max<uint64_t>(layer.ffn_dim, 1),
            "cb_mtp_draft_ffn_gate");
        cb_mtp_ffn_up_batch_ = device_->scratch_f32(
            B * std::max<uint64_t>(layer.ffn_dim, 1),
            "cb_mtp_draft_ffn_up");
        cb_mtp_ffn_mid_batch_ = device_->scratch_f32(
            B * std::max<uint64_t>(layer.ffn_dim, 1),
            "cb_mtp_draft_ffn_mid");
        cb_mtp_ffn_out_batch_ =
            device_->scratch_f32(B * cfg.n_embd, "cb_mtp_draft_ffn_out");
        cb_mtp_logits_batch_ = device_->scratch_f32(
            B * static_cast<uint64_t>(weights_->output().rows),
            "cb_mtp_draft_logits");
        cb_mtp_draft_capacity_ = batch;
    }

    bool run_continuous_mtp_batched_draft_step(
            std::vector<ContinuousBatchActive> &mtp_active,
            std::vector<ContinuousMtpDraftStep> &steps,
            bool first_depth) {
        if (steps.size() < 2) return false;
        auto trace_stage = [&](const char *stage) {
            if (!continuous_batching_trace_enabled()) return;
            std::ostringstream msg;
            msg << "native continuous_mtp_batched_draft_stage:"
                << " stage=" << stage
                << " batch=" << steps.size()
                << " first_depth=" << (first_depth ? "true" : "false");
            log(msg.str());
        };
        const NativePlanInfo &plan = model_->plan();
        const QwenMtpWeights *mtp = weights_->mtp();
        if (!plan.mtp_supported || !mtp || mtp->layer.recurrent) return false;
        if (!mtp->eh_proj || !mtp->embed_tokens || !mtp->enorm ||
            !mtp->hnorm || !mtp->shared_head_head || !mtp->shared_head_norm) {
            return false;
        }
        const uint32_t bsz = static_cast<uint32_t>(steps.size());
        const QwenConfig &cfg = model_->config();
        const QwenLayerWeights &layer = mtp->layer;
        const uint32_t hidden = cfg.n_embd;
        const uint32_t standard_head_dim = cfg.head_dim;
        const uint32_t standard_n_heads = cfg.n_heads;
        const uint32_t standard_n_kv_heads = cfg.n_kv_heads;
        const uint32_t q_stride = static_cast<uint32_t>(layer.q_rows);
        const uint32_t k_stride = static_cast<uint32_t>(layer.k_rows);
        const uint32_t v_stride = static_cast<uint32_t>(layer.v_rows);
        const uint32_t per_pos = standard_n_kv_heads * standard_head_dim;
        const uint32_t mid_stride = standard_n_heads * standard_head_dim;
        const uint32_t ffn_stride = static_cast<uint32_t>(layer.ffn_dim);
        const uint32_t vocab = static_cast<uint32_t>(weights_->output().rows);
        const float eps = cfg.rms_eps;
        const float scale = 1.0f / std::sqrt(static_cast<float>(standard_head_dim));
        if (q_stride == 0 || k_stride == 0 || v_stride == 0 ||
            ffn_stride == 0) {
            return false;
        }

        ensure_continuous_mtp_draft_scratch(bsz);
        ContinuousPrefillBatch metadata_batch;
        metadata_batch.collect_row_argmaxes = true;
        metadata_batch.page_size = 0;
        metadata_batch.q_indptr.push_back(0);
        metadata_batch.page_indptr.push_back(0);
        std::vector<uint64_t> rows_h(bsz, 0);

        for (uint32_t i = 0; i < bsz; ++i) {
            ContinuousMtpDraftStep &step = steps[i];
            if (step.row >= mtp_active.size()) return false;
            ContinuousBatchActive &a = mtp_active[step.row];
            if (!a.executor) return false;
            a.executor->prepare_mtp_prefix_pages(step.cache_pos, 1);
            QwenExecutor::MtpPrefixStateView view =
                a.executor->mtp_prefix_state_view();
            if (!view.ready || !view.k_cache || !view.v_cache ||
                !view.page_indices_host || view.page_size == 0 ||
                !view.draft_hidden || !view.current_hidden) {
                return false;
            }
            const uint32_t seq_len = step.cache_pos + 1;
            const uint32_t pages =
                (seq_len + view.page_size - 1) / view.page_size;
            if (pages == 0 || pages > view.page_count) return false;
            if (metadata_batch.page_size == 0) {
                metadata_batch.page_size = view.page_size;
            } else if (metadata_batch.page_size != view.page_size) {
                return false;
            }
            rows_h[i] = step.input_token;
            metadata_batch.logical_positions.push_back(
                static_cast<int32_t>(step.cache_pos));
            metadata_batch.row_page_indptr.push_back(
                static_cast<int32_t>(metadata_batch.page_indices.size()));
            for (uint32_t p = 0; p < pages; ++p) {
                metadata_batch.page_indices.push_back(view.page_indices_host[p]);
            }
            const uint32_t last_len = seq_len % view.page_size;
            metadata_batch.last_page_len.push_back(static_cast<int32_t>(
                last_len == 0 ? view.page_size : last_len));
            metadata_batch.seq_lens.push_back(static_cast<int32_t>(seq_len));
            metadata_batch.q_indptr.push_back(static_cast<int32_t>(i + 1));
            metadata_batch.page_indptr.push_back(
                static_cast<int32_t>(metadata_batch.page_indices.size()));
            metadata_batch.max_seq_len =
                std::max<uint32_t>(metadata_batch.max_seq_len, seq_len);
        }
        metadata_batch.total_tokens = bsz;
        metadata_batch.row_page_indptr.push_back(
            static_cast<int32_t>(metadata_batch.page_indices.size()));
        metadata_batch.ragged_metadata_ready =
            metadata_batch.page_size > 0 &&
            metadata_batch.q_indptr.size() == static_cast<size_t>(bsz) + 1 &&
            metadata_batch.page_indptr.size() == static_cast<size_t>(bsz) + 1 &&
            metadata_batch.logical_positions.size() == bsz &&
            metadata_batch.row_page_indptr.size() == static_cast<size_t>(bsz) + 1 &&
            metadata_batch.last_page_len.size() == bsz &&
            metadata_batch.seq_lens.size() == bsz &&
            !metadata_batch.page_indices.empty();
        metadata_batch.ragged_row_metadata_ready =
            metadata_batch.ragged_metadata_ready;
        // The fused KV-append + attention below require valid ragged metadata
        // (one global MTP KV tensor indexed per-row). If it didn't assemble,
        // fall back to the per-row draft path.
        if (!metadata_batch.ragged_metadata_ready) return false;

        try {
            trace_stage("begin");
            require_device_status(device_->begin());
            prepare_continuous_prefill_ragged_metadata_device(metadata_batch);
            if (!metadata_batch.ragged_device_metadata_ready) return false;
            trace_stage("pack_hidden");
            for (uint32_t i = 0; i < bsz; ++i) {
                ContinuousMtpDraftStep &step = steps[i];
                ContinuousBatchActive &a = mtp_active[step.row];
                QwenExecutor::MtpPrefixStateView view =
                    a.executor->mtp_prefix_state_view();
                DeviceTensor *src_h =
                    first_depth ? view.current_hidden : view.draft_hidden;
                if (!src_h) throw std::runtime_error("MTP draft hidden missing");
                require_device_status(device_->copy_d2d_into(
                    *cb_mtp_h_input_batch_,
                    static_cast<uint64_t>(i) * hidden,
                    *src_h, 0, hidden));
            }
            trace_stage("embed");
            require_device_status(device_->q8_0_get_rows_batch(
                *cb_mtp_norm_batch_, *mtp->embed_tokens, rows_h.data(), bsz));
            require_device_status(device_->rms_norm_batch(
                *cb_mtp_ffn_out_batch_, *cb_mtp_norm_batch_, *mtp->enorm,
                bsz, hidden, eps));
            require_device_status(device_->rms_norm_batch(
                *cb_mtp_h_batch_, *cb_mtp_h_input_batch_, *mtp->hnorm,
                bsz, hidden, eps));
            require_device_status(device_->pack_mtp_concat(
                *cb_mtp_concat_batch_, *cb_mtp_ffn_out_batch_,
                *cb_mtp_h_batch_, bsz, hidden, hidden,
                static_cast<uint32_t>(2 * hidden), hidden));
            require_device_status(device_->q8_0_matmul(
                *cb_mtp_h_batch_, *mtp->eh_proj, *cb_mtp_concat_batch_,
                bsz, static_cast<uint32_t>(2 * hidden), hidden));
            trace_stage("qkv");
            require_device_status(device_->rms_norm_batch(
                *cb_mtp_norm_batch_, *cb_mtp_h_batch_, *layer.attn_norm,
                bsz, hidden, eps));
            DeviceTensor *qkv_outs[3] = {
                cb_mtp_q_batch_.get(), cb_mtp_k_batch_.get(),
                cb_mtp_v_batch_.get()
            };
            const DeviceWeight *qkv_ws[3] = {
                layer.attn_q, layer.attn_k, layer.attn_v
            };
            const uint32_t qkv_strides[3] = {q_stride, k_stride, v_stride};
            require_device_status(device_->q8_0_matmul_fanout(
                qkv_outs, qkv_ws, qkv_strides, 3,
                *cb_mtp_norm_batch_, bsz, hidden));
            require_device_status(device_->rmsnorm_per_head_batch(
                *cb_mtp_q_batch_, *layer.attn_q_norm, bsz, q_stride,
                standard_n_heads, 2 * standard_head_dim,
                standard_head_dim, eps));
            require_device_status(device_->rmsnorm_per_head_batch(
                *cb_mtp_k_batch_, *layer.attn_k_norm, bsz, k_stride,
                standard_n_kv_heads, standard_head_dim,
                standard_head_dim, eps));
            std::vector<int32_t> rope_pos_h(bsz, 0);
            for (uint32_t i = 0; i < bsz; ++i) {
                rope_pos_h[i] = static_cast<int32_t>(steps[i].cache_pos);
            }
            if (bsz > cb_mtp_draft_positions_capacity_) {
                cb_mtp_draft_positions_i32_ =
                    device_->tensor_i32(bsz, "cb_mtp_draft_positions");
                cb_mtp_draft_positions_capacity_ = bsz;
            }
            require_device_status(device_->copy_i32_from_host(
                *cb_mtp_draft_positions_i32_, 0, rope_pos_h.data(), bsz));
            require_device_status(device_->rope_partial_batch_positions(
                *cb_mtp_q_batch_, bsz, q_stride, standard_n_heads,
                2 * standard_head_dim, cfg.rope_dim,
                *cb_mtp_draft_positions_i32_, cfg.rope_theta));
            require_device_status(device_->rope_partial_batch_positions(
                *cb_mtp_k_batch_, bsz, k_stride, standard_n_kv_heads,
                standard_head_dim, cfg.rope_dim,
                *cb_mtp_draft_positions_i32_, cfg.rope_theta));
            trace_stage("kv_append_k_before");
            // All rows share one global MTP KV tensor, so KV-append and
            // attention fuse into single ragged calls across the batch (mirrors
            // the plain decode body-batch path). cb_mtp_draft_positions_i32_
            // already holds per-row cache_pos; cb_prefill_* device metadata was
            // uploaded above from metadata_batch.
            QwenExecutor::MtpPrefixStateView view0 =
                mtp_active[steps[0].row].executor->mtp_prefix_state_view();
            require_device_status(device_->kv_append_batch_paged_ragged_device(
                *view0.k_cache, *cb_mtp_k_batch_, *cb_mtp_draft_positions_i32_,
                per_pos, bsz, k_stride,
                *cb_prefill_page_indices_i32_, *cb_prefill_page_indptr_i32_,
                view0.page_size));
            trace_stage("kv_append_k_after");
            trace_stage("kv_append_v_before");
            require_device_status(device_->kv_append_batch_paged_ragged_device(
                *view0.v_cache, *cb_mtp_v_batch_, *cb_mtp_draft_positions_i32_,
                per_pos, bsz, v_stride,
                *cb_prefill_page_indices_i32_, *cb_prefill_page_indptr_i32_,
                view0.page_size));
            trace_stage("kv_append_v_after");
            trace_stage("attention");
            require_device_status(
                device_->attention_decode_batch_paged_gated_ragged_device(
                    *cb_mtp_mid_batch_, *cb_mtp_q_batch_, 2 * standard_head_dim,
                    *view0.k_cache, *view0.v_cache,
                    *cb_prefill_page_indices_i32_, *cb_prefill_page_indptr_i32_,
                    *cb_prefill_last_page_len_i32_, *cb_prefill_seq_lens_i32_,
                    metadata_batch.page_indptr.data(),
                    metadata_batch.last_page_len.data(),
                    metadata_batch.seq_lens.data(),
                    view0.page_size, standard_n_heads, standard_n_kv_heads,
                    standard_head_dim, bsz, q_stride, mid_stride, scale));
            require_device_status(device_->q8_0_matmul_add(
                *cb_mtp_h_batch_, *cb_mtp_h_batch_, *cb_mtp_ffn_out_batch_,
                *layer.attn_output, *cb_mtp_mid_batch_, bsz, mid_stride,
                hidden));
            trace_stage("ffn");
            require_device_status(device_->rms_norm_batch(
                *cb_mtp_norm_batch_, *cb_mtp_h_batch_, *layer.ffn_norm,
                bsz, hidden, eps));
            DeviceTensor *ffn_outs[2] = {
                cb_mtp_ffn_gate_batch_.get(), cb_mtp_ffn_up_batch_.get()
            };
            const DeviceWeight *ffn_ws[2] = {layer.ffn_gate, layer.ffn_up};
            const uint32_t ffn_strides[2] = {ffn_stride, ffn_stride};
            require_device_status(device_->q8_0_matmul_fanout(
                ffn_outs, ffn_ws, ffn_strides, 2, *cb_mtp_norm_batch_,
                bsz, hidden));
            require_device_status(device_->silu_mul_n(
                *cb_mtp_ffn_mid_batch_, *cb_mtp_ffn_gate_batch_,
                *cb_mtp_ffn_up_batch_, static_cast<uint64_t>(bsz) * ffn_stride));
            require_device_status(device_->q8_0_matmul_add(
                *cb_mtp_h_batch_, *cb_mtp_h_batch_, *cb_mtp_ffn_out_batch_,
                *layer.ffn_down, *cb_mtp_ffn_mid_batch_, bsz, ffn_stride,
                hidden));
            require_device_status(device_->rms_norm_batch(
                *cb_mtp_norm_batch_, *cb_mtp_h_batch_, *mtp->shared_head_norm,
                bsz, hidden, eps));
            require_device_status(device_->q8_0_matmul(
                *cb_mtp_logits_batch_, *mtp->shared_head_head,
                *cb_mtp_norm_batch_, bsz, hidden, vocab));
            trace_stage("argmax");
            std::vector<DeviceArgmax> argmaxes;
            require_device_status(device_->argmax_batch(
                *cb_mtp_logits_batch_, bsz, vocab, argmaxes));
            trace_stage("writeback");
            for (uint32_t i = 0; i < bsz; ++i) {
                ContinuousMtpDraftStep &step = steps[i];
                ContinuousBatchActive &a = mtp_active[step.row];
                QwenExecutor::MtpPrefixStateView view =
                    a.executor->mtp_prefix_state_view();
                if (!view.draft_hidden) {
                    throw std::runtime_error("MTP draft hidden missing");
                }
                require_device_status(device_->copy_d2d_into(
                    *view.draft_hidden, 0, *cb_mtp_h_batch_,
                    static_cast<uint64_t>(i) * hidden, hidden));
                a.executor->set_mtp_prefix_len(step.cache_pos + 1);
                if (i < argmaxes.size()) {
                    step.output_token = argmaxes[i].token;
                    step.output_logit = argmaxes[i].logit;
                }
                step.report.ok = step.output_token >= 0;
                step.report.argmax_token = step.output_token;
                step.report.argmax_logit = step.output_logit;
                if (step.output_token >= 0) {
                    step.report.argmax_text = model_->gguf().token_text(
                        static_cast<uint32_t>(step.output_token));
                }
                step.report.ops_executed = 1;
            }
            require_device_status(device_->end());
            return true;
        } catch (...) {
            try { (void)device_->end(); } catch (...) {}
            throw;
        }
    }

    // True if any verify job's executor is currently running in the kvmem
    // window frame. Such rows must take the window-aware ragged verify route,
    // never the full-cache layered verifier.
    static bool mtp_jobs_have_kvmem_row(
            const std::vector<ContinuousBatchActive> &mtp_active,
            const std::vector<ContinuousMtpVerifyJob> &jobs) {
        for (const ContinuousMtpVerifyJob &job : jobs) {
            if (job.row < mtp_active.size() && mtp_active[job.row].executor &&
                mtp_active[job.row].executor->kvmem_active()) {
                return true;
            }
        }
        return false;
    }

    void build_continuous_mtp_verify_batch(
            std::vector<ContinuousBatchActive> &mtp_active,
            const std::vector<ContinuousMtpVerifyJob> &jobs,
            ContinuousPrefillBatch &batch) {
        batch.clear();
        batch.collect_row_argmaxes = true;
        if (jobs.empty()) return;
        batch.entries.reserve(jobs.size());
        batch.q_indptr.reserve(jobs.size() + 1);
        batch.page_indptr.reserve(jobs.size() + 1);
        batch.last_page_len.reserve(jobs.size());
        batch.seq_lens.reserve(jobs.size());
        batch.q_indptr.push_back(0);
        batch.page_indptr.push_back(0);
        bool all_recurrent_state_ready = true;
        for (uint32_t row = 0; row < jobs.size(); ++row) {
            const ContinuousMtpVerifyJob &job = jobs[row];
            if (job.row >= mtp_active.size()) continue;
            ContinuousBatchActive &a = mtp_active[job.row];
            if (!a.executor || job.verify_tokens.empty()) continue;
            const uint32_t chunk =
                static_cast<uint32_t>(job.verify_tokens.size());
            a.prefill_offset = job.base_position;
            a.executor->prepare_runtime_state();
            a.executor->prepare_kv_pages(job.base_position, chunk);
            // kvmem rows verify in the WINDOW frame: extend the assembled window
            // so the `chunk` verify tokens append at the window tail (true KV at
            // [base_position, base_position+chunk) was just allocated above), and
            // source the page table + logical positions from the window instead
            // of the full cache. Mirrors forward_n_tokens' chunk_bs branch so the
            // batched verify attends exactly what the single-request path would.
            const bool row_bs = a.executor->kvmem_active();
            if (row_bs) {
                a.executor->kvmem_extend_window_for_verify(chunk,
                                                           job.base_position);
            }
            QwenExecutor::DecodeStateView view =
                a.executor->decode_state_view();
            const uint32_t row_base =
                row_bs ? a.executor->window_query_pos() : job.base_position;
            const int32_t *row_page_src =
                row_bs ? a.executor->window_pages_host().data()
                       : view.kv_page_indices_host;
            const uint32_t row_page_avail =
                row_bs ? a.executor->window_page_count() : view.kv_page_count;

            ContinuousPrefillBatchEntry entry;
            entry.prefill_index = job.row;
            entry.request_id = a.req ? a.req->id : 0;
            entry.offset = job.base_position;
            entry.total = job.base_position + chunk;
            entry.chunk = chunk;
            entry.final_chunk = true;

            bool entry_recurrent_ready =
                view.recurrent_states != nullptr &&
                view.conv_states != nullptr &&
                view.recurrent_states->size() >= weights_->n_layers() &&
                view.conv_states->size() >= weights_->n_layers();
            if (entry_recurrent_ready) {
                for (uint32_t il = 0; il < weights_->n_layers(); ++il) {
                    const QwenLayerWeights &layer = weights_->layer(il);
                    if (!layer.recurrent) continue;
                    if (!(*view.recurrent_states)[il] ||
                        !(*view.conv_states)[il]) {
                        entry_recurrent_ready = false;
                        break;
                    }
                }
            }
            all_recurrent_state_ready =
                all_recurrent_state_ready && entry_recurrent_ready;

            bool entry_metadata_ready =
                view.kv_page_size > 0 &&
                row_page_src != nullptr &&
                row_page_avail > 0;
            if (entry_metadata_ready) {
                if (batch.page_size == 0) {
                    batch.page_size = view.kv_page_size;
                } else if (batch.page_size != view.kv_page_size) {
                    entry_metadata_ready = false;
                }
            }
            // Window frame for kvmem rows (row_base == window_query_pos), true
            // frame otherwise (row_base == base_position).
            const uint32_t seq_len = row_base + chunk;
            if (entry_metadata_ready) {
                const uint32_t pages =
                    (seq_len + view.kv_page_size - 1) / view.kv_page_size;
                if (pages == 0 || row_page_avail < pages) {
                    entry_metadata_ready = false;
                } else {
                    const int32_t request_page_begin =
                        static_cast<int32_t>(batch.page_indices.size());
                    for (uint32_t p = 0; p < pages; ++p) {
                        batch.page_indices.push_back(row_page_src[p]);
                    }
                    for (uint32_t t = 0; t < chunk; ++t) {
                        batch.logical_positions.push_back(
                            static_cast<int32_t>(row_base + t));
                        batch.row_page_indptr.push_back(request_page_begin);
                        batch.token_rows.push_back(job.verify_tokens[t]);
                    }
                    const uint32_t last_len = seq_len % view.kv_page_size;
                    batch.last_page_len.push_back(static_cast<int32_t>(
                        last_len == 0 ? view.kv_page_size : last_len));
                    batch.seq_lens.push_back(static_cast<int32_t>(seq_len));
                    batch.max_seq_len = std::max(batch.max_seq_len, seq_len);
                }
            }
            if (!entry_metadata_ready) {
                batch.clear();
                batch.collect_row_argmaxes = true;
                return;
            }
            batch.total_tokens += chunk;
            ++batch.final_chunks;
            batch.entries.push_back(entry);
            batch.q_indptr.push_back(static_cast<int32_t>(batch.total_tokens));
            batch.page_indptr.push_back(
                static_cast<int32_t>(batch.page_indices.size()));
        }
        if (!batch.row_page_indptr.empty()) {
            batch.row_page_indptr.push_back(
                static_cast<int32_t>(batch.page_indices.size()));
        }
        batch.recurrent_state_ready =
            batch.size() > 0 && all_recurrent_state_ready;
        batch.ragged_metadata_ready =
            batch.size() > 0 &&
            batch.q_indptr.size() == batch.size() + 1 &&
            batch.page_indptr.size() == batch.size() + 1 &&
            batch.logical_positions.size() == batch.total_tokens &&
            batch.token_rows.size() == batch.total_tokens &&
            batch.row_page_indptr.size() == batch.total_tokens + 1 &&
            batch.last_page_len.size() == batch.size() &&
            batch.seq_lens.size() == batch.size() &&
            !batch.page_indices.empty() &&
            batch.page_size > 0;
        batch.ragged_row_metadata_ready = batch.ragged_metadata_ready;
    }

    void run_continuous_mtp_batch_requests(
            const std::vector<std::shared_ptr<ContinuousBatchRequest>> &reqs,
            uint32_t ctx_size,
            uint32_t max_active) {
        if (reqs.empty()) return;
        const int32_t eos = tokenizer_->eos_id();
        const bool cb_mtp_phase_sync = mtp_phase_sync_enabled();
        auto cb_phase_time = [&]() -> double {
            if (cb_mtp_phase_sync) {
                DeviceStatus sync_st = device_->synchronize();
                if (!sync_st.ok) throw std::runtime_error(sync_st.message);
            }
            return wall_seconds();
        };
        const uint32_t requested_chain = mtp_trace_chain_len(options_);
        const uint32_t chain_len =
            std::min<uint32_t>(requested_chain, mtp_safe_chain_max());
        const uint32_t state_checkpoint_count =
            mtp_state_checkpoint_count(chain_len);
        const bool use_device_draft =
            mtp_device_draft_chain_enabled() && !mtp_verify_trace_enabled();
        struct MtpStats {
            uint64_t drafted = 0;
            uint64_t accepted = 0;
            uint64_t rejected = 0;
            uint64_t rollbacks = 0;
            uint64_t verify_batches = 0;
            uint64_t verify_tokens = 0;
            uint64_t decode_ops = 0;
            uint64_t prefill_ops = 0;
            uint64_t state_checkpoint_reused = 0;
            uint64_t prefix1_reused = 0;
            double prefill_s = 0.0;
            double draft_s = 0.0;
            double snapshot_s = 0.0;
            double verify_s = 0.0;
            double restore_s = 0.0;
            double replay_s = 0.0;
            double prefix_s = 0.0;
            double loop_wall_s = 0.0;
            double decode_start = 0.0;
            MtpAdaptivePolicy policy;
            // Persistent per-row verify state. Reused across loop iterations so
            // the ~37 device tensors (hidden + recurrent + conv) backing the
            // snapshot/checkpoints are allocated once per row, not once per
            // batch. The job below only references these by row; allocating a
            // fresh snapshot per batch costs ~74 synchronous cudaMalloc/cudaFree
            // pairs per step (~11 ms), which was the entire CB-vs-legacy MTP
            // regression. Legacy reuses a single snapshot the same way.
            QwenExecutor::StateSnapshot snapshot;
            QwenExecutor::StateCheckpointSet checkpoints;
        };
        std::vector<ContinuousBatchActive> mtp_active;
        std::vector<MtpStats> stats;
        mtp_active.reserve(reqs.size());
        stats.reserve(reqs.size());

        auto run_layered_verifier =
            [&](std::vector<ContinuousMtpVerifyJob> &jobs,
                std::vector<BatchedPrefillOutput> &outputs) {
                outputs.clear();
                outputs.resize(jobs.size());
                if (jobs.empty()) return false;
                if (!cb_decode_executor_) {
                    cb_decode_executor_ =
                        std::make_unique<BatchedDecodeExecutor>(
                            *model_, *weights_, *device_);
                }
                std::vector<bool> active_job(jobs.size(), true);
                uint32_t active_count = static_cast<uint32_t>(jobs.size());
                const uint32_t max_rows = static_cast<uint32_t>(
                    std::max<size_t>(1, jobs.front().verify_tokens.size()));
                for (uint32_t depth = 0; depth < max_rows && active_count > 0;
                     ++depth) {
                    ContinuousDecodeBatch decode_batch;
                    decode_batch.active_indices.reserve(active_count);
                    decode_batch.feed_tokens.reserve(active_count);
                    decode_batch.positions.reserve(active_count);
                    decode_batch.state_views.reserve(active_count);
                    std::vector<uint32_t> job_indices;
                    job_indices.reserve(active_count);
                    for (uint32_t j = 0; j < jobs.size(); ++j) {
                        if (!active_job[j] ||
                            depth >= jobs[j].verify_tokens.size()) {
                            continue;
                        }
                        const size_t row = jobs[j].row;
                        if (row >= mtp_active.size()) return false;
                        ContinuousBatchActive &a = mtp_active[row];
                        a.next_token = jobs[j].verify_tokens[depth];
                        QwenExecutor::DecodeStateView view =
                            a.executor->decode_state_view();
                        decode_batch.active_indices.push_back(row);
                        decode_batch.feed_tokens.push_back(a.next_token);
                        decode_batch.positions.push_back(view.position);
                        decode_batch.state_views.push_back(view);
                        job_indices.push_back(j);
                    }
                    if (decode_batch.size() == 0) break;
                    const double verify0 = cb_phase_time();
                    const std::vector<BatchedDecodeOutput> layer_outputs =
                        cb_decode_executor_->decode(
                            mtp_active, BatchedDecodeInput{&decode_batch});
                    const double verify_s =
                        std::max(cb_phase_time() - verify0, 0.0);
                    if (layer_outputs.size() != job_indices.size()) {
                        return false;
                    }
                    const double per_row_verify_s =
                        verify_s / static_cast<double>(job_indices.size());
                    for (uint32_t row_i = 0; row_i < layer_outputs.size();
                         ++row_i) {
                        const uint32_t j = job_indices[row_i];
                        ContinuousMtpVerifyJob &job = jobs[j];
                        BatchedPrefillOutput &out = outputs[j];
                        const BatchedDecodeOutput &layer_out =
                            layer_outputs[row_i];
                        MtpStats &s = stats[job.row];
                        s.verify_s += per_row_verify_s;
                        if (!layer_out.ok()) {
                            out.error = layer_out.error.empty()
                                ? "MTP layered verifier failed"
                                : layer_out.error;
                            return false;
                        }
                        if (out.row_argmaxes.empty()) {
                            out.prefill_index = job.row;
                            out.request_id =
                                mtp_active[job.row].req
                                    ? mtp_active[job.row].req->id
                                    : 0;
                            out.offset = job.base_position;
                            out.total = job.base_position +
                                        static_cast<uint32_t>(
                                            job.verify_tokens.size());
                            out.chunk =
                                static_cast<uint32_t>(
                                    job.verify_tokens.size());
                            out.final_chunk = true;
                            out.report.ok = true;
                        }
                        out.row_argmaxes.push_back(DeviceArgmax{
                            layer_out.report.argmax_token,
                            layer_out.report.argmax_logit
                        });
                        out.report.ops_executed +=
                            layer_out.report.ops_executed;
                        if (depth < job.drafts.size()) {
                            const int32_t target =
                                layer_out.report.argmax_token >= 0
                                    ? layer_out.report.argmax_token
                                    : eos;
                            if (target !=
                                static_cast<int32_t>(job.drafts[depth])) {
                                active_job[j] = false;
                                --active_count;
                            }
                        } else {
                            active_job[j] = false;
                            --active_count;
                        }
                    }
                }
                for (uint32_t j = 0; j < jobs.size(); ++j) {
                    if (outputs[j].row_argmaxes.empty()) return false;
                    outputs[j].report.ok = true;
                    const DeviceArgmax &last = outputs[j].row_argmaxes.back();
                    outputs[j].report.argmax_token = last.token;
                    outputs[j].report.argmax_logit = last.logit;
                    if (last.token >= 0) {
                        outputs[j].report.argmax_text =
                            model_->gguf().token_text(
                                static_cast<uint32_t>(last.token));
                    }
                }
                return true;
            };

        auto should_stop = [&](const ContinuousBatchActive &a,
                               uint32_t token) {
            return a.req && !a.req->options.ignore_eos &&
                   token == static_cast<uint32_t>(eos);
        };
        auto emit = [&](ContinuousBatchActive &a, uint32_t token) {
            if (!a.req || a.decoded >= a.req->options.max_tokens ||
                should_stop(a, token)) {
                return false;
            }
            const std::string piece =
                tokenizer_->decode_one(static_cast<int32_t>(token));
            a.req->generated += piece;
            if (a.req->on_text) a.req->on_text(piece);
            budget_observe(a.budget, token);
            // Track committed tokens so the next verify batch's penalties
            // (presence/repetition) see the running output. No-op for greedy
            // (penalties unused), so greedy MTP stays byte-identical.
            ++a.seen_tokens[token];
            ++a.decoded;
            return true;
        };

        auto admit_mtp_request =
            [&](const std::shared_ptr<ContinuousBatchRequest> &req) {
                if (!req || !req->spec_mtp || req->trace_mtp) {
                    run_continuous_mtp_request(req, ctx_size);
                    return;
                }
                ContinuousBatchActive a;
                initialize_continuous_active(a, req, ctx_size);
                if (continuous_batching_trace_enabled()) {
                    std::ostringstream msg;
                    msg << "native continuous_mtp:"
                        << " request=" << req->id
                        << " prompt_tokens=" << req->prompt_tokens.size()
                        << " max_tokens=" << req->options.max_tokens
                        << " spec=true trace=false"
                        << " mode=batched_verify";
                    log(msg.str());
                }
                constexpr uint32_t kMtpPrefillChunk = 4096;
                a.executor->set_prefill_chunk_override(
                    static_cast<int>(kMtpPrefillChunk));
                NativeExecutorReport step;
                uint64_t prefill_ops = 0;
                if (QwenExecutor::kvmem_timing_enabled()) {
                    a.kvmem_timing_baseline = QwenExecutor::kvmem_timing_snapshot();
                }
                const double prefill0 = wall_seconds();
                // Prefix-cache: start prefill at a.prefill_offset (set to the
                // reused prefix length by seed_from_shared_prefix in
                // initialize_continuous_active). Starting at 0 would re-append
                // the seeded tokens at position_=hit_len, corrupting the KV
                // layout (double-prefill). The seeded MAIN KV pages are
                // physically resident; only the MTP draft prefix (a SEPARATE KV
                // cache) was not seeded, and its hidden states are not
                // recoverable from the prefix cache. So for a seeded request we
                // skip MTP-prefix priming: the draft chain then bails
                // (position_ > mtp_prefix_len_==0) to a full forward_one_token
                // per token — correct output, just no speculation over the
                // reused prefix. Non-seeded requests prime as before.
                const bool seeded_prefix = a.prefill_offset > 0;
                const size_t prefill_start = std::min<size_t>(
                    a.prefill_offset, req->prompt_tokens.size());
                for (size_t offset = prefill_start;
                     offset < req->prompt_tokens.size();
                     offset += kMtpPrefillChunk) {
                    const size_t end = std::min(
                        req->prompt_tokens.size(),
                        offset + static_cast<size_t>(kMtpPrefillChunk));
                    std::vector<uint32_t> chunk(
                        req->prompt_tokens.begin() +
                            static_cast<std::ptrdiff_t>(offset),
                        req->prompt_tokens.begin() +
                            static_cast<std::ptrdiff_t>(end));
                    const bool need_logits = end == req->prompt_tokens.size();
                    step = a.executor->forward_n_tokens(chunk, need_logits);
                    if (!step.ok) throw std::runtime_error("MTP prefill failed");
                    prefill_ops += step.ops_executed;
                    if (!seeded_prefix) {
                        NativeExecutorReport prefix =
                            a.executor->prime_mtp_prefix_from_last_batch(
                                chunk, static_cast<uint32_t>(offset));
                        if (!prefix.ok) {
                            throw std::runtime_error(
                                "MTP prefix priming failed in batched lane");
                        }
                    }
                }
                // Seeding only ever reuses a strict prefix (hit_len <
                // prompt_len), so prefill_start < prompt_len and the loop ran at
                // least one chunk, leaving `step` holding the final logits.
                if (!step.ok) {
                    throw std::runtime_error(
                        "MTP prefill produced no logits (empty prompt?)");
                }
                // kvmem: build the working-set window now that the full prompt
                // KV is resident. register_append clamps to position_, so the
                // seeded region registers correctly. Under the default all-fit
                // budget this is identity selection => decode stays
                // byte-identical to plain MTP. No-op when --kvmem is off.
                kvmem_on_prefill_complete(a);
                // Phase split: everything accumulated up to here (prefill-chunk
                // offload reselects + the first post-prefill window assembly) is
                // the TTFT-phase kvmem cost. Emit it as phase=prefill, then re-
                // baseline so the finish emit captures only the decode-phase
                // reselect cadence. This is what isolates "why is kvmem_gpu TTFT
                // ~= plain" (tiny prefill assembly) from the decode penalty.
                if (QwenExecutor::kvmem_timing_enabled()) {
                    const std::string ptag =
                        "phase=prefill request=" + std::to_string(req->id);
                    QwenExecutor::kvmem_timing_emit_delta(
                        ptag.c_str(), a.kvmem_timing_baseline);
                    a.kvmem_timing_baseline =
                        QwenExecutor::kvmem_timing_snapshot();
                }
                const double prefill_s = std::max(wall_seconds() - prefill0, 1e-9);
                a.prefill_ops = prefill_ops;
                // First post-prefill token: sample from the last prefill row when
                // temp>0 (pick_continuous_next_token round-trips logits_ via
                // copy_last_logits + applies penalties); greedy returns the
                // argmax unchanged, so greedy MTP stays byte-identical.
                a.next_token = static_cast<uint32_t>(pick_continuous_next_token(
                    a, step.argmax_token >= 0 ? step.argmax_token
                                              : static_cast<int32_t>(eos)));
                a.decode_start = wall_seconds();
                if (req->options.max_tokens > 0 &&
                    !should_stop(a, a.next_token)) {
                    emit(a, a.next_token);
                }
                if (a.decoded >= req->options.max_tokens ||
                    should_stop(a, a.next_token)) {
                    finish_continuous_active(a);
                } else {
                    mtp_active.push_back(std::move(a));
                    stats.emplace_back();
                    stats.back().prefill_ops = prefill_ops;
                    stats.back().prefill_s = prefill_s;
                    stats.back().decode_start = wall_seconds();
                    stats.back().policy.configure(
                        true, chain_len, req->prompt_tokens.size());
                    if (stats.back().policy.enabled) {
                        const MtpAdaptivePolicy &policy = stats.back().policy;
                        std::ostringstream policy_msg;
                        policy_msg << "native mtp_policy_config:"
                                   << " min=" << policy.min_depth
                                   << " max=" << policy.max_depth
                                   << " initial=" << policy.initial_depth
                                   << " update_interval="
                                   << policy.update_interval
                                   << " min_decision_batches="
                                   << policy.min_decision_batches
                                   << " cooldown="
                                   << policy.cooldown_batches
                                   << " demote_windows="
                                   << policy.demote_windows
                                   << " promote_windows="
                                   << policy.promote_windows
                                   << " startup_demote_batches="
                                   << policy.startup_demote_batches
                                   << " demote_margin="
                                   << policy.demote_margin
                                   << " promote_margin="
                                   << policy.promote_margin
                                   << " trace="
                                   << (policy.trace ? "true" : "false");
                        log(policy_msg.str());
                    }
                }
            };

        try {
            for (const auto &req : reqs) {
                admit_mtp_request(req);
            }

            auto admit_pending_mtp_requests = [&]() {
                uint32_t admitted = 0;
                while (mtp_active.size() < max_active) {
                    std::shared_ptr<ContinuousBatchRequest> req;
                    {
                        std::lock_guard<std::mutex> lk(cb_mu_);
                        if (cb_pending_.empty() || !cb_pending_.front() ||
                            !cb_pending_.front()->active_mtp ||
                            !cb_pending_.front()->spec_mtp ||
                            cb_pending_.front()->trace_mtp) {
                            break;
                        }
                        req = cb_pending_.front();
                        cb_pending_.pop_front();
                    }
                    admit_mtp_request(req);
                    ++admitted;
                }
                if (admitted > 0 && continuous_batching_trace_enabled()) {
                    std::ostringstream msg;
                    msg << "native continuous_mtp_admit_pending:"
                        << " admitted=" << admitted
                        << " active=" << mtp_active.size()
                        << " max_active=" << max_active;
                    log(msg.str());
                }
                return admitted;
            };

            admit_pending_mtp_requests();

            if (!cb_prefill_executor_) {
                cb_prefill_executor_ =
                    std::make_unique<BatchedPrefillExecutor>(
                        *model_, *weights_, *device_);
            }
            if (continuous_batching_trace_enabled() && !mtp_active.empty()) {
                uint32_t draft_ready = 0;
                for (ContinuousBatchActive &a : mtp_active) {
                    if (!a.executor) continue;
                    QwenExecutor::MtpPrefixStateView view =
                        a.executor->mtp_prefix_state_view();
                    if (view.ready && view.prefix_len >= a.executor->position()) {
                        ++draft_ready;
                    }
                }
                std::ostringstream msg;
                msg << "native continuous_mtp_batched_draft:"
                    << " eligible=" << draft_ready
                    << " active=" << mtp_active.size()
                    << " enabled="
                    << (continuous_mtp_batched_draft_enabled()
                            ? "true" : "false");
                log(msg.str());
            }
            while (!mtp_active.empty()) {
                const double loop_iter0 = wall_seconds();
                admit_pending_mtp_requests();
                std::vector<ContinuousMtpVerifyJob> jobs;
                jobs.reserve(mtp_active.size());
                std::vector<size_t> draft_rows;
                draft_rows.reserve(mtp_active.size());
                for (size_t row = 0; row < mtp_active.size(); ++row) {
                    ContinuousBatchActive &a = mtp_active[row];
                    if (!a.req || a.decoded >= a.req->options.max_tokens ||
                        should_stop(a, a.next_token)) {
                        continue;
                    }
                    // Thinking-budget forced rows: feed the current token to
                    // advance the cache, then emit the queued guidance/</think>
                    // token instead of speculating. Resumes normal speculation
                    // once the close tag drains and the block is closed.
                    if (a.budget.active() &&
                        (budget_should_force(a.budget) ||
                         !a.budget.forced_queue.empty())) {
                        const uint32_t current = a.next_token;
                        NativeExecutorReport fstep =
                            a.executor->forward_one_token(current);
                        if (!fstep.ok) {
                            throw std::runtime_error(
                                "MTP thinking-budget decode failed");
                        }
                        stats[row].decode_ops += fstep.ops_executed;
                        a.executor->commit_mtp_prefix(a.executor->position());
                        const uint32_t argmax = fstep.argmax_token >= 0
                            ? static_cast<uint32_t>(fstep.argmax_token)
                            : static_cast<uint32_t>(eos);
                        a.next_token = budget_next_feed(a.budget, argmax);
                        emit(a, a.next_token);
                        continue;
                    }
                    draft_rows.push_back(row);
                }

                std::vector<ContinuousMtpVerifyJob> batched_jobs;
                if (continuous_mtp_batched_draft_enabled() &&
                    draft_rows.size() >= 2) {
                    batched_jobs.reserve(draft_rows.size());
                    for (size_t row : draft_rows) {
                        ContinuousBatchActive &a = mtp_active[row];
                        ContinuousMtpVerifyJob job;
                        job.row = row;
                        job.current = a.next_token;
                        job.base_position = a.executor->position();
                        // kvmem rows always snapshot, even under checkpoints:
                        // the ragged single-fallback restore (and the commit
                        // snapshot-replay path) is window-aware only via
                        // restore_state(snapshot), so the window base must be
                        // captured here regardless of state_checkpoint_count.
                        if (state_checkpoint_count == 0 ||
                            a.executor->kvmem_active()) {
                            const double snapshot0 = cb_phase_time();
                            a.executor->capture_state(stats[row].snapshot);
                            stats[row].snapshot_s +=
                                std::max(cb_phase_time() - snapshot0, 0.0);
                        }
                        job.verify_tokens.push_back(job.current);
                        batched_jobs.push_back(std::move(job));
                    }
                    bool batch_ok = true;
                    for (uint32_t depth = 0; depth < chain_len && batch_ok;
                         ++depth) {
                        std::vector<ContinuousMtpDraftStep> steps;
                        steps.reserve(batched_jobs.size());
                        for (ContinuousMtpVerifyJob &job : batched_jobs) {
                            ContinuousBatchActive &a = mtp_active[job.row];
                            const uint32_t remaining =
                                static_cast<uint32_t>(
                                    a.req->options.max_tokens - a.decoded);
                            const uint32_t draft_limit =
                                stats[job.row].policy.draft_limit(remaining,
                                                                   chain_len);
                            if (depth >= draft_limit) continue;
                            ContinuousMtpDraftStep step;
                            step.row = job.row;
                            step.input_token = depth == 0
                                ? job.current
                                : job.drafts.back();
                            step.cache_pos = job.base_position + depth;
                            steps.push_back(std::move(step));
                        }
                        if (steps.size() < 2) break;
                        const double draft0 = cb_phase_time();
                        batch_ok = run_continuous_mtp_batched_draft_step(
                            mtp_active, steps, depth == 0);
                        const double per_row_s =
                            std::max(cb_phase_time() - draft0, 0.0) /
                            static_cast<double>(steps.size());
                        if (!batch_ok) break;
                        for (const ContinuousMtpDraftStep &step : steps) {
                            stats[step.row].draft_s += per_row_s;
                            if (!step.report.ok || step.output_token < 0) {
                                continue;
                            }
                            ContinuousBatchActive &a = mtp_active[step.row];
                            const uint32_t token =
                                static_cast<uint32_t>(step.output_token);
                            if (should_stop(a, token)) continue;
                            auto it = std::find_if(
                                batched_jobs.begin(), batched_jobs.end(),
                                [&](const ContinuousMtpVerifyJob &job) {
                                    return job.row == step.row;
                                });
                            if (it == batched_jobs.end()) continue;
                            it->drafts.push_back(token);
                            it->verify_tokens.push_back(token);
                            stats[step.row].drafted += 1;
                        }
                    }
                    if (batch_ok) {
                        for (ContinuousMtpVerifyJob &job : batched_jobs) {
                            if (!job.drafts.empty()) {
                                jobs.push_back(std::move(job));
                            }
                        }
                    }
                }

                for (size_t row : draft_rows) {
                    auto already = std::find_if(
                        jobs.begin(), jobs.end(),
                        [&](const ContinuousMtpVerifyJob &job) {
                            return job.row == row;
                        });
                    if (already != jobs.end()) continue;
                    ContinuousBatchActive &a = mtp_active[row];
                    const uint32_t current = a.next_token;
                    const uint32_t remaining =
                        static_cast<uint32_t>(
                            a.req->options.max_tokens - a.decoded);
                    const uint32_t draft_limit =
                        stats[row].policy.draft_limit(remaining, chain_len);
                    const double draft0 = cb_phase_time();
                    std::vector<NativeExecutorReport> chain =
                        use_device_draft
                            ? a.executor->forward_mtp_draft_chain_with_prefix_device(
                                  current, draft_limit)
                            : a.executor->forward_mtp_draft_chain_with_prefix(
                                  current, draft_limit);
                    stats[row].draft_s +=
                        std::max(cb_phase_time() - draft0, 0.0);
                    ContinuousMtpVerifyJob job;
                    job.row = row;
                    job.current = current;
                    job.base_position = a.executor->position();
                    // kvmem rows always snapshot, even under checkpoints (see
                    // batched-draft path above): restore_state(snapshot) is the
                    // only window-aware rollback used by the ragged
                    // single-fallback and commit snapshot-replay paths.
                    if (state_checkpoint_count == 0 ||
                        a.executor->kvmem_active()) {
                        const double snapshot0 = cb_phase_time();
                        a.executor->capture_state(stats[row].snapshot);
                        stats[row].snapshot_s +=
                            std::max(cb_phase_time() - snapshot0, 0.0);
                    }
                    job.verify_tokens.push_back(current);
                    for (const NativeExecutorReport &draft : chain) {
                        if (!draft.ok || draft.argmax_token < 0) break;
                        const uint32_t token =
                            static_cast<uint32_t>(draft.argmax_token);
                        if (should_stop(a, token)) break;
                        job.drafts.push_back(token);
                        job.verify_tokens.push_back(token);
                        stats[row].drafted += 1;
                    }
                    if (!job.drafts.empty()) {
                        jobs.push_back(std::move(job));
                    } else {
                        NativeExecutorReport step =
                            a.executor->forward_one_token(current);
                        if (!step.ok) {
                            throw std::runtime_error("MTP fallback decode failed");
                        }
                        stats[row].decode_ops += step.ops_executed;
                        // Sample/pick from the just-computed logits before
                        // commit_mtp_prefix (which leaves logits_ untouched but we
                        // pick first to be safe); samples when temp>0, else argmax.
                        const int32_t fb = step.argmax_token >= 0
                            ? step.argmax_token : eos;
                        a.next_token = static_cast<uint32_t>(
                            pick_continuous_next_token(a, fb));
                        a.executor->commit_mtp_prefix(a.executor->position());
                        emit(a, a.next_token);
                    }
                }
                if (!jobs.empty()) {
                    std::vector<BatchedPrefillOutput> outputs;
                    outputs.reserve(jobs.size());
                    auto run_single_verifier =
                        [&](ContinuousMtpVerifyJob &job) {
                            BatchedPrefillOutput out;
                            out.prefill_index = job.row;
                            ContinuousBatchActive &ja = mtp_active[job.row];
                            out.request_id = ja.req ? ja.req->id : 0;
                            out.offset = job.base_position;
                            out.total = job.base_position +
                                        static_cast<uint32_t>(
                                            job.verify_tokens.size());
                            out.chunk =
                                static_cast<uint32_t>(
                                    job.verify_tokens.size());
                            out.final_chunk = true;
                            // When the request samples (temp>0) or applies
                            // penalties, pull every verify row's full distribution
                            // to host for the point-mass accept test; pure-greedy
                            // rows skip this and stay byte-identical.
                            const bool job_samples =
                                ja.req && mtp_options_need_logits(
                                    ja.req->options.temperature,
                                    ja.req->options.presence_penalty,
                                    ja.req->options.repetition_penalty);
                            out.report =
                                ja.executor->forward_n_tokens(
                                    job.verify_tokens, true,
                                    &out.row_argmaxes,
                                    state_checkpoint_count > 0
                                        ? &stats[job.row].checkpoints
                                        : nullptr,
                                    state_checkpoint_count,
                                    /*copy_last_logits=*/false,
                                    job_samples ? &out.row_logits : nullptr);
                            if (!out.report.ok) {
                                out.error = "MTP single verifier failed";
                            }
                            return out;
                        };
                    // Speculative SAMPLING (temp>0) or penalties need each verify
                    // row's full distribution on host, which only the single-
                    // verifier (forward_n_tokens) path transports. The ragged/
                    // layered batched verifiers report argmax only, so any batch
                    // with a logits-needing row is forced through the per-job
                    // single verifier. Pure-greedy batches keep their existing fast
                    // routing byte-identically. (Throughput-batched sampling is
                    // deferred.)
                    bool batch_needs_sampling = false;
                    for (const ContinuousMtpVerifyJob &job : jobs) {
                        if (job.row < mtp_active.size() &&
                            mtp_active[job.row].req &&
                            mtp_options_need_logits(
                                mtp_active[job.row].req->options.temperature,
                                mtp_active[job.row].req->options.presence_penalty,
                                mtp_active[job.row].req->options.repetition_penalty)) {
                            batch_needs_sampling = true;
                            break;
                        }
                    }
                    if (jobs.size() == 1 || batch_needs_sampling) {
                        for (ContinuousMtpVerifyJob &job : jobs) {
                            const double verify0 = cb_phase_time();
                            outputs.push_back(run_single_verifier(job));
                            if (job.row < stats.size()) {
                                stats[job.row].verify_s +=
                                    std::max(cb_phase_time() - verify0, 0.0);
                            }
                        }
                    } else if (continuous_mtp_layered_verify_enabled() &&
                               !mtp_jobs_have_kvmem_row(mtp_active, jobs) &&
                               run_layered_verifier(jobs, outputs)) {
                        // The layered verifier reads the live full-cache page
                        // table and is NOT kvmem-window-aware, so kvmem rows are
                        // forced to the window-aware ragged route below. (The
                        // request-entry guard already hard-errors kvmem +
                        // layered, so this is belt-and-suspenders.)
                        for (ContinuousMtpVerifyJob &job : jobs) {
                            job.layered_verified = true;
                        }
                    } else {
                        bool used_ragged_verify = false;
                        ContinuousPrefillBatch batch;
                        build_continuous_mtp_verify_batch(
                            mtp_active, jobs, batch);
                        const bool try_ragged_verify =
                            batch.ragged_metadata_ready &&
                            batch.total_tokens >=
                                continuous_batching_mtp_ragged_verify_min_tokens();
                        if (try_ragged_verify) {
                            prepare_continuous_prefill_ragged_metadata_device(
                                batch);
                            BatchedPrefillDeviceMetadata metadata;
                            metadata.q_indptr = cb_prefill_q_indptr_i32_.get();
                            metadata.page_indptr =
                                cb_prefill_page_indptr_i32_.get();
                            metadata.row_page_indptr =
                                cb_prefill_row_page_indptr_i32_.get();
                            metadata.page_indices =
                                cb_prefill_page_indices_i32_.get();
                            metadata.logical_positions =
                                cb_prefill_logical_positions_i32_.get();
                            metadata.last_page_len =
                                cb_prefill_last_page_len_i32_.get();
                            metadata.seq_lens = cb_prefill_seq_lens_i32_.get();
                            const double verify0 = cb_phase_time();
                            outputs = cb_prefill_executor_->prefill(
                                mtp_active, batch, metadata);
                            used_ragged_verify = true;
                            const double verify_s =
                                std::max(cb_phase_time() - verify0, 0.0);
                            for (const ContinuousMtpVerifyJob &job : jobs) {
                                if (job.row < stats.size()) {
                                    stats[job.row].verify_s +=
                                        verify_s /
                                        static_cast<double>(jobs.size());
                                }
                            }
                        }
                        bool need_single_fallback =
                            outputs.size() != jobs.size();
                        if (!need_single_fallback) {
                            for (const BatchedPrefillOutput &out : outputs) {
                                if (!out.ok()) {
                                    need_single_fallback = true;
                                    break;
                                }
                                if (state_checkpoint_count > 0 &&
                                    !out.checkpoints.ready) {
                                    need_single_fallback = true;
                                    break;
                                }
                            }
                        }
                        if (need_single_fallback) {
                            used_ragged_verify = false;
                            for (const ContinuousMtpVerifyJob &job : jobs) {
                                if (job.row < mtp_active.size() &&
                                    job.row < stats.size() &&
                                    stats[job.row].snapshot.ready) {
                                    mtp_active[job.row].executor->restore_state(
                                        stats[job.row].snapshot);
                                }
                            }
                            outputs.clear();
                            outputs.reserve(jobs.size());
                            for (ContinuousMtpVerifyJob &job : jobs) {
                                const double verify0 = cb_phase_time();
                                outputs.push_back(run_single_verifier(job));
                                if (job.row < stats.size()) {
                                    stats[job.row].verify_s +=
                                        std::max(cb_phase_time() - verify0,
                                                 0.0);
                                }
                            }
                        } else if (used_ragged_verify &&
                                   state_checkpoint_count > 0) {
                            for (uint32_t j = 0;
                                 j < jobs.size() && j < outputs.size(); ++j) {
                                if (jobs[j].row < stats.size()) {
                                    stats[jobs[j].row].checkpoints =
                                        std::move(outputs[j].checkpoints);
                                }
                            }
                        }
                        // Ragged verify appended `chunk` tokens at each kvmem
                        // row's window tail (physical KV written) but, unlike
                        // forward_n_tokens, did NOT self-advance
                        // window_query_pos_. Advance it now on the
                        // ragged-success path ONLY so the window stays in
                        // lockstep with position_ (which prefill_ragged advanced
                        // by chunk). The single-fallback path re-ran
                        // forward_n_tokens, which self-advances, so advancing
                        // here would double-count. The commit loop below rolls
                        // both position_ and the window back per accepted row.
                        if (used_ragged_verify) {
                            for (uint32_t j = 0;
                                 j < jobs.size() && j < outputs.size(); ++j) {
                                const ContinuousMtpVerifyJob &job = jobs[j];
                                if (job.row >= mtp_active.size()) continue;
                                ContinuousBatchActive &a = mtp_active[job.row];
                                if (a.executor && a.executor->kvmem_active()) {
                                    a.executor->kvmem_advance_window(
                                        static_cast<uint32_t>(
                                            job.verify_tokens.size()));
                                }
                            }
                        }
                    }
                    for (uint32_t j = 0; j < jobs.size(); ++j) {
                        const ContinuousMtpVerifyJob &job = jobs[j];
                        ContinuousBatchActive &a = mtp_active[job.row];
                        MtpStats &s = stats[job.row];
                        const bool row_count_ok = job.layered_verified
                            ? (!outputs[j].row_argmaxes.empty() &&
                               outputs[j].row_argmaxes.size() <=
                                   job.verify_tokens.size())
                            : (outputs[j].row_argmaxes.size() ==
                               job.verify_tokens.size());
                        if (j >= outputs.size() || !outputs[j].ok() ||
                            !row_count_ok) {
                            throw std::runtime_error(
                                "MTP batched verifier failed");
                        }
                        const auto &row_argmaxes = outputs[j].row_argmaxes;
                        s.verify_batches += 1;
                        s.verify_tokens += row_argmaxes.size();
                        s.decode_ops += outputs[j].report.ops_executed;
                        // temp>0 or penalties → point-mass accept test over the
                        // host target rows (temp<=0 collapses to greedy over the
                        // penalized logits inside the helper). Pure greedy (no
                        // penalties) → the original argmax-equality loop,
                        // byte-identical to greedy MTP.
                        const bool do_sample =
                            a.req && mtp_options_need_logits(
                                a.req->options.temperature,
                                a.req->options.presence_penalty,
                                a.req->options.repetition_penalty);
                        uint32_t accepted = 0;
                        int32_t target = eos;
                        if (do_sample) {
                            const GenerationOptions &o = a.req->options;
                            SpecAcceptResult sr = speculative_accept_pointmass(
                                outputs[j].row_logits, job.drafts,
                                o.temperature, o.top_p, o.top_k, o.min_p,
                                o.presence_penalty, o.repetition_penalty,
                                a.seen_tokens, a.rng);
                            accepted = sr.accepted;
                            target = sr.extra_token >= 0 ? sr.extra_token : eos;
                            s.accepted += accepted;
                            if (accepted < job.drafts.size()) ++s.rejected;
                        } else {
                            for (uint32_t i = 0; i < job.drafts.size() &&
                                                 i < row_argmaxes.size(); ++i) {
                                target = row_argmaxes[i].token >= 0
                                    ? row_argmaxes[i].token
                                    : eos;
                                if (target == static_cast<int32_t>(job.drafts[i])) {
                                    ++accepted;
                                    ++s.accepted;
                                } else {
                                    ++s.rejected;
                                    break;
                                }
                            }
                        }
                        const char *policy_action = s.policy.update(
                            static_cast<uint32_t>(job.drafts.size()),
                            accepted,
                            a.req
                                ? a.req->prompt_tokens.size() +
                                      static_cast<size_t>(a.decoded)
                                : 0);
                        if (s.policy.enabled && s.policy.trace) {
                            std::ostringstream policy_msg;
                            policy_msg << "native mtp_policy:"
                                       << " batch=" << s.verify_batches
                                       << " ctx="
                                       << (a.req
                                               ? a.req->prompt_tokens.size() +
                                                     static_cast<size_t>(
                                                         a.decoded)
                                               : 0)
                                       << " drafted=" << job.drafts.size()
                                       << " accepted=" << accepted
                                       << " depth=" << s.policy.depth
                                       << " action=" << policy_action
                                       << " window_batches="
                                       << s.policy.window_batches
                                       << " avg_committed=" << std::fixed
                                       << std::setprecision(4)
                                       << s.policy.last_avg_committed
                                       << " full_rate="
                                       << s.policy.last_full_rate
                                       << " benefit=" << s.policy.last_benefit
                                       << " cost=" << s.policy.last_cost
                                       << " next_cost="
                                       << s.policy.last_next_cost;
                            log(policy_msg.str());
                        }
                        const bool all_accepted =
                            accepted == job.drafts.size();
                        if (all_accepted) {
                            if (job.layered_verified) {
                                if (mtp_rebuild_accepted_prefix_enabled()) {
                                    const double prefix0 = cb_phase_time();
                                    a.executor
                                        ->commit_mtp_prefix_from_current_hidden(
                                            a.executor->position());
                                    s.prefix_s += std::max(
                                        cb_phase_time() - prefix0, 0.0);
                                } else {
                                    a.executor->commit_mtp_prefix(
                                        a.executor->position());
                                }
                            } else if (mtp_rebuild_accepted_prefix_enabled()) {
                                const double prefix0 = cb_phase_time();
                                NativeExecutorReport prefix =
                                    a.executor->prime_mtp_prefix_from_last_batch(
                                        job.verify_tokens, job.base_position,
                                        mtp_prefix_rebuild_batch_min_tokens());
                                s.prefix_s +=
                                    std::max(cb_phase_time() - prefix0, 0.0);
                                if (!prefix.ok) {
                                    const std::string reason =
                                        prefix.missing_kernels.empty()
                                            ? "unknown"
                                            : prefix.missing_kernels.front();
                                    throw std::runtime_error(
                                        "MTP batched verifier prefix rebuild failed: " +
                                        reason);
                                }
                                s.decode_ops += prefix.ops_executed;
                            } else {
                                a.executor->commit_mtp_prefix(
                                    a.executor->position());
                            }
                        } else if (job.layered_verified) {
                            if (mtp_rebuild_accepted_prefix_enabled()) {
                                const double prefix0 = cb_phase_time();
                                a.executor
                                    ->commit_mtp_prefix_from_current_hidden(
                                        a.executor->position());
                                s.prefix_s +=
                                    std::max(cb_phase_time() - prefix0, 0.0);
                            } else {
                                a.executor->commit_mtp_prefix(
                                    a.executor->position());
                            }
                            ++s.rollbacks;
                        } else {
                            const bool use_checkpoint_replay =
                                state_checkpoint_count > 0 &&
                                s.checkpoints.ready &&
                                accepted < s.checkpoints.count;
                            if (use_checkpoint_replay) {
                                const double restore0 = cb_phase_time();
                                a.executor->restore_state_checkpoint(
                                    s.checkpoints, accepted);
                                s.restore_s +=
                                    std::max(cb_phase_time() - restore0, 0.0);
                                ++s.state_checkpoint_reused;
                                if (accepted == 0) ++s.prefix1_reused;
                                // restore_state_checkpoint clamps the MTP prefix
                                // length down to the restored position. When
                                // accepted>=1 that leaves position_ ahead of the
                                // prefix, so the next batch's draft chain bails
                                // (position_ > mtp_prefix_len_) and falls back to
                                // a full target forward_one_token (~22 ms each).
                                // Re-extend the prefix from the just-restored
                                // hidden state, mirroring the legacy reject path.
                                if (mtp_rebuild_accepted_prefix_enabled()) {
                                    const double prefix0 = cb_phase_time();
                                    a.executor
                                        ->commit_mtp_prefix_from_current_hidden(
                                            a.executor->position());
                                    s.prefix_s +=
                                        std::max(cb_phase_time() - prefix0, 0.0);
                                } else {
                                    a.executor->commit_mtp_prefix(
                                        a.executor->position());
                                }
                            } else {
                                if (!s.snapshot.ready) {
                                    throw std::runtime_error(
                                        "MTP batched verifier replay requires a state snapshot");
                                }
                                const double restore0 = cb_phase_time();
                                a.executor->restore_state(s.snapshot);
                                s.restore_s +=
                                    std::max(cb_phase_time() - restore0, 0.0);
                                std::vector<uint32_t> replay;
                                replay.reserve(accepted + 1);
                                replay.push_back(job.current);
                                for (uint32_t i = 0; i < accepted; ++i) {
                                    replay.push_back(job.drafts[i]);
                                }
                                double prefix_seconds = 0.0;
                                uint64_t prefix_ops = 0;
                                const double replay0 = cb_phase_time();
                                NativeExecutorReport replay_report =
                                    a.executor->replay_tokens_with_mtp_prefix(
                                        replay, job.base_position,
                                        mtp_rebuild_accepted_prefix_enabled(),
                                        &prefix_seconds, &prefix_ops);
                                s.replay_s +=
                                    std::max(cb_phase_time() - replay0, 0.0);
                                s.prefix_s += prefix_seconds;
                                if (!replay_report.ok) {
                                    throw std::runtime_error(
                                        "MTP batched verifier replay failed");
                                }
                                s.decode_ops +=
                                    replay_report.ops_executed + prefix_ops;
                            }
                            ++s.rollbacks;
                        }
                        // kvmem cadence: position() now reflects this row's
                        // final committed KV length (the rollback block above
                        // settled it for every accept/reject branch, and my
                        // Task-3 ragged window advance is in lockstep with it).
                        // Register the newly-committed tokens with the block
                        // store and reselect the window on the interval
                        // boundary, mirroring the single-request kvmem_advance_to
                        // cadence. No-op when --kvmem is off; under the default
                        // all-fit budget this stays byte-identical to plain MTP.
                        kvmem_mtp_advance_to(a, a.executor->position());
                        for (uint32_t i = 0; i < accepted; ++i) {
                            if (!emit(a, job.drafts[i])) break;
                        }
                        if (a.decoded >= a.req->options.max_tokens) continue;
                        if (all_accepted && !do_sample) {
                            // Greedy: bonus token is the final row's argmax.
                            // (Sampling already drew the bonus into `target`
                            // inside speculative_accept_pointmass.)
                            if (job.drafts.size() >= row_argmaxes.size()) {
                                throw std::runtime_error(
                                    "MTP verifier missing final target row");
                            }
                            target = row_argmaxes[job.drafts.size()].token >= 0
                                ? row_argmaxes[job.drafts.size()].token
                                : eos;
                        }
                        a.next_token = static_cast<uint32_t>(target);
                        emit(a, a.next_token);
                    }
                }
                for (MtpStats &ls : stats) {
                    ls.loop_wall_s += std::max(wall_seconds() - loop_iter0, 0.0);
                }
                for (size_t i = mtp_active.size(); i > 0; --i) {
                    ContinuousBatchActive &a = mtp_active[i - 1];
                    if (!a.req || a.decoded >= a.req->options.max_tokens ||
                        should_stop(a, a.next_token)) {
                        const MtpStats &s = stats[i - 1];
                        const double decode_s =
                            std::max(wall_seconds() - s.decode_start, 1e-9);
                        std::ostringstream summary;
                        summary << "native mtp_spec_summary:"
                                << " enabled=true"
                                << " batches=" << s.verify_batches
                                << " drafted=" << s.drafted
                                << " accepted=" << s.accepted
                                << " rejected=" << s.rejected
                                << " rollbacks=" << s.rollbacks
                                << " adaptive="
                                << (s.policy.enabled ? "true" : "false")
                                << " promotions=" << s.policy.promotions
                                << " reject_budget=off fallback=false"
                                << " acceptance=" << std::fixed
                                << std::setprecision(4)
                                << (s.drafted > 0
                                        ? static_cast<double>(s.accepted) /
                                              static_cast<double>(s.drafted)
                                        : 0.0)
                                << " mtp_ops=0 prefix_tokens=0 prefix_ops=0"
                                << " prefix1_reuse=" << s.prefix1_reused
                                << " state_ckpt_reuse="
                                << s.state_checkpoint_reused
                                << " state_ckpt_count="
                                << state_checkpoint_count
                                << " batched_verify_batches="
                                << s.verify_batches
                                << " batched_verify_tokens="
                                << s.verify_tokens
                                << " draft_s=" << fmt_seconds(s.draft_s)
                                << " snapshot_s=" << fmt_seconds(s.snapshot_s)
                                << " verify_s=" << fmt_seconds(s.verify_s)
                                << " restore_s=" << fmt_seconds(s.restore_s)
                                << " replay_s=" << fmt_seconds(s.replay_s)
                                << " prefix_s=" << fmt_seconds(s.prefix_s);
                        summary << " loop_wall_s=" << fmt_seconds(s.loop_wall_s);
                        if (cb_mtp_phase_sync) summary << " phase_sync=true";
                        log(summary.str());
                        log("native generate: prompt_tokens=" +
                            std::to_string(a.req->prompt_tokens.size()) +
                            " prefill=" + fmt_seconds(s.prefill_s) +
                            " decoded=" +
                            std::to_string(a.decoded) +
                            " decode=" + fmt_seconds(decode_s) +
                            " prefill_ops=" +
                            std::to_string(s.prefill_ops) +
                            " decode_ops=" +
                            std::to_string(s.decode_ops));
                        if (a.executor && a.executor->kvmem_enabled()) {
                            const QwenExecutor::KvMemTierUsage tu =
                                a.executor->kvmem_tier_usage();
                            std::ostringstream tmsg;
                            tmsg << "[kvmem-tier-usage]"
                                 << " total_blocks=" << tu.total_blocks
                                 << " block_bytes=" << tu.block_bytes
                                 << " gpu_used=" << tu.gpu_used_bytes
                                 << " gpu_cap=" << tu.gpu_capacity_bytes
                                 << " gpu_pool=" << (tu.gpu_pool ? 1 : 0)
                                 << " cpu_used=" << tu.cpu_used_bytes
                                 << " cpu_cap=" << tu.cpu_capacity_bytes
                                 << " nvme_used=" << tu.nvme_used_bytes
                                 << " nvme_cap=" << tu.nvme_capacity_bytes;
                            log(tmsg.str());
                        }
                        finish_continuous_active(a);
                        mtp_active.erase(mtp_active.begin() +
                                         static_cast<std::ptrdiff_t>(i - 1));
                        stats.erase(stats.begin() +
                                    static_cast<std::ptrdiff_t>(i - 1));
                    }
                }
            }
        } catch (const std::exception &e) {
            for (auto &a : mtp_active) {
                if (a.req) complete_continuous_request(a.req, {}, e.what());
            }
            for (const auto &req : reqs) {
                if (req && !req->done) {
                    complete_continuous_request(req, {}, e.what());
                }
            }
        }
    }

    // Build a KvMemStoreConfig from CLI options and enable kvmem on an
    // executor. No-op when --kvmem is off, so the forward path stays byte-
    // identical to plain. Shared by the single-session (generate_plain) and
    // per-request continuous-batching executors — kvmem state is per-executor,
    // so each concurrent request maintains its own block table + window with no
    // cross-request interference.
    void configure_executor_kvmem(QwenExecutor &exec) const {
        if (!options_.kvmem_enabled) return;
        KvMemStoreConfig bs_cfg;
        bs_cfg.block_tokens =
            static_cast<uint32_t>(std::max(1, options_.kvmem_block_tokens));
        bs_cfg.select_budget =
            static_cast<uint32_t>(std::max(1, options_.kvmem_budget));
        bs_cfg.sink_blocks =
            static_cast<uint32_t>(std::max(0, options_.kvmem_sink_blocks));
        bs_cfg.recent_blocks =
            static_cast<uint32_t>(std::max(0, options_.kvmem_recent_blocks));
        bs_cfg.retrieval_blocks =
            static_cast<uint32_t>(std::max(0, options_.kvmem_retrieval_blocks));
        bs_cfg.profile_blocks =
            static_cast<uint32_t>(std::max(0, options_.kvmem_profile_blocks));
        bs_cfg.gpu_memory_ratio = options_.kvmem_gpu_memory_ratio;
        bs_cfg.gpu_high_watermark = options_.kvmem_gpu_high_watermark;
        bs_cfg.gpu_low_watermark = options_.kvmem_gpu_low_watermark;
        bs_cfg.cpu_tier_bytes = options_.kvmem_cpu_bytes;
        bs_cfg.nvme_tier_bytes = options_.kvmem_nvme_bytes;
        bs_cfg.nvme_tier_dir = options_.kvmem_nvme_dir;
        if (options_.kvmem_method == "h2o") {
            bs_cfg.select_method = KvMemMethod::H2O;
        } else if (options_.kvmem_method == "recency") {
            bs_cfg.select_method = KvMemMethod::Recency;
        } else {
            bs_cfg.select_method = KvMemMethod::Retrieval;
        }
        bs_cfg.select_policy = options_.kvmem_select_policy == "quota"
            ? KvMemSelectPolicy::Quota
            : KvMemSelectPolicy::TopK;
        bs_cfg.retrieval_method = options_.kvmem_retrieval_method == "content_mean"
            ? KvMemRetrievalMethod::ContentMean
            : KvMemRetrievalMethod::MeanAttention;
        bs_cfg.update_mode = options_.kvmem_update_mode == "step"
            ? KvMemUpdateMode::Step
            : KvMemUpdateMode::Interval;
        exec.set_kvmem_enabled(true);
        exec.configure_kvmem(bs_cfg);
    }

    // First kvmem working-set build for a CB request, at the prefill->active
    // transition. Mirrors generate_plain (6293-6297): register the whole
    // prompt as context blocks then assemble the first window. Under the
    // default all-fit budget this selects every block (identity) so decode
    // stays byte-identical to plain. No-op when this request has no kvmem.
    void kvmem_on_prefill_complete(ContinuousBatchActive &a) {
        if (!a.executor || !a.executor->kvmem_enabled()) return;
        a.executor->kvmem_register_append(
            static_cast<uint32_t>(a.req->prompt_tokens.size()));
        a.executor->kvmem_reselect();
        a.kvmem_steps_since_reselect = 0;
        a.kvmem_registered_pos = a.executor->position();
        a.kvmem_last_reselect_pos = a.executor->position();
    }

    // Per-row MTP cadence: register the position() delta since the last call
    // and reselect on the interval boundary. The MTP commit path advances
    // position() by the accepted-chain length (>1), so register the whole jump
    // in one shot rather than per token. Mirrors the single-request
    // kvmem_advance_to lambda (~6641). No-op when this request has no kvmem.
    void kvmem_mtp_advance_to(ContinuousBatchActive &a, uint32_t committed_pos) {
        if (!a.executor || !a.executor->kvmem_enabled()) return;
        if (committed_pos > a.kvmem_registered_pos) {
            a.executor->kvmem_register_append(committed_pos -
                                              a.kvmem_registered_pos);
            a.kvmem_registered_pos = committed_pos;
        }
        if (options_.kvmem_update_mode == "step") return;
        const uint32_t interval =
            static_cast<uint32_t>(std::max(1, options_.kvmem_interval));
        if (committed_pos >= a.kvmem_last_reselect_pos + interval) {
            a.executor->kvmem_reselect();
            a.kvmem_last_reselect_pos = committed_pos;
        }
    }

    // Per committed decode token: grow the context by one and reselect on the
    // interval boundary. Mirrors generate_plain (6362-6368). No-op when this
    // request has no kvmem.
    void kvmem_on_decode_step(ContinuousBatchActive &a) {
        if (!a.executor || !a.executor->kvmem_enabled()) return;
        a.executor->kvmem_register_append(1);
        if (options_.kvmem_update_mode == "step") return;
        const int interval = std::max(1, options_.kvmem_interval);
        if (++a.kvmem_steps_since_reselect >= interval) {
            a.executor->kvmem_reselect();
            a.kvmem_steps_since_reselect = 0;
        }
    }

    void initialize_continuous_active(
            ContinuousBatchActive &a,
            const std::shared_ptr<ContinuousBatchRequest> &req,
            uint32_t ctx_size) {
        a.req = req;
        a.executor = std::make_unique<QwenExecutor>(
            *model_, *weights_, *device_, ctx_size,
            cb_kv_pool_.get(), &cb_kv_cache_view_,
            cb_mtp_kv_pool_.get(), &cb_mtp_kv_cache_view_);
        a.executor->set_prefill_chunk_override(options_.prefill_chunk);
        // Borrow the pinned CPU-tier buffer from the shared pool so admit does
        // not pay a per-request cudaHostAlloc of the whole --kvmem-cpu-gb tier.
        a.executor->set_host_tier_pool(cb_host_tier_pool_.get());
        a.executor->reset_state();
        // Per-request kvmem: each CB executor gets its own block store + window
        // when --kvmem is set. The reselect cadence is driven below at the
        // prefill->active transition (first register + reselect) and per decode
        // step (continuous_decode_batch_step). No-op when --kvmem is off.
        configure_executor_kvmem(*a.executor);
        a.seen_tokens.reserve(req->prompt_tokens.size() +
                              static_cast<size_t>(req->options.max_tokens));
        for (uint32_t token : req->prompt_tokens) ++a.seen_tokens[token];
        a.rng.seed(req->options.seed);
        budget_init(a.budget, req->options);

        // ---- Prefix cache: seed on hit, schedule commit on miss ----------
        if (prefix_cache_enabled() && cb_kv_pool_ && !req->prompt_tokens.empty()) {
            prefix_cache_install_evict_cb();
            const uint32_t page_size = a.executor->kv_page_size_public();
            const uint32_t prompt_len =
                static_cast<uint32_t>(req->prompt_tokens.size());
            std::vector<int32_t> hit_pages;
            QwenExecutor::StateSnapshot hit_recur;
            uint32_t hit_len = 0;
            const uint64_t hit_id = prefix_cache_lookup(
                req->prompt_tokens, page_size, hit_pages, hit_recur, hit_len);
            if (hit_id != 0 && hit_len > 0 && hit_len < prompt_len) {
                try {
                    a.executor->seed_from_shared_prefix(hit_pages, hit_recur,
                                                        hit_len);
                    a.prefill_offset = hit_len;
                    a.held_prefix_entries.push_back(hit_id);
                    a.kv_state.update(a.executor->kv_state_snapshot());
                    if (prefix_cache_trace_enabled()) {
                        std::ostringstream m;
                        m << "prefix_cache hit id=" << hit_id
                          << " req=" << req->id
                          << " reused_tokens=" << hit_len
                          << " pages=" << hit_pages.size();
                        log(m.str());
                    }
                } catch (const std::exception &e) {
                    // Seeding failed: roll back to a clean cold prefill and
                    // drop the refcount we took in lookup.
                    a.executor->reset_state();
                    a.prefill_offset = 0;
                    ContinuousBatchActive tmp;
                    tmp.held_prefix_entries.push_back(hit_id);
                    prefix_cache_release(tmp);
                    if (prefix_cache_trace_enabled()) {
                        log(std::string("prefix_cache hit-seed failed, cold "
                                        "fallback: ") + e.what());
                    }
                }
            }
            // Schedule a commit of the longest page-aligned prefix strictly
            // shorter than the prompt, unless we already reuse one that long.
            // v1 commits a single prefix per prompt.
            if (page_size > 0 && prompt_len >= 2 * page_size) {
                uint32_t commit_len = (prompt_len / page_size) * page_size;
                if (commit_len >= prompt_len) commit_len -= page_size;
                if (commit_len > a.prefill_offset) {
                    a.prefix_commit_pending = true;
                    a.prefix_commit_len = commit_len;
                }
            }
        }
    }

    uint32_t continuous_prefill_chunk_tokens(uint32_t remaining) const {
        if (remaining == 0) return 0;
        if (options_.prefill_chunk == 0) return remaining;
        uint32_t chunk = options_.prefill_chunk > 0
            ? static_cast<uint32_t>(options_.prefill_chunk)
            : 2048u;
        chunk = std::max<uint32_t>(512u, chunk);
        chunk = std::min<uint32_t>(4096u, chunk);
        return std::min<uint32_t>(remaining, chunk);
    }

    void advance_continuous_prefill(std::vector<ContinuousBatchActive> &prefilling,
                                    std::vector<ContinuousBatchActive> &active,
                                    int32_t eos) {
        if (prefilling.empty()) return;
        ContinuousBatchActive &a = prefilling.front();
        try {
            const std::vector<uint32_t> &prompt = a.req->prompt_tokens;
            if (a.prefill_offset >= prompt.size() && !prompt.empty()) {
                throw std::runtime_error("continuous prefill has no prompt tokens");
            }
            const uint32_t remaining =
                static_cast<uint32_t>(prompt.size() - a.prefill_offset);
            uint32_t chunk = continuous_prefill_chunk_tokens(remaining);
            // Clamp the chunk so prefill lands exactly on the prefix-cache
            // commit boundary; capture_state then snapshots recurrent state at
            // precisely aligned_len. Without this the executor would overshoot
            // the boundary and the snapshot would be lossy.
            if (a.prefix_commit_pending &&
                a.prefill_offset < a.prefix_commit_len &&
                a.prefill_offset + chunk > a.prefix_commit_len) {
                chunk = a.prefix_commit_len - a.prefill_offset;
            }
            const bool final_chunk = prompt.empty() || chunk >= remaining;
            std::vector<uint32_t> chunk_tokens(
                prompt.begin() + static_cast<std::ptrdiff_t>(a.prefill_offset),
                prompt.begin() + static_cast<std::ptrdiff_t>(a.prefill_offset + chunk));

            const double t0 = wall_seconds();
            NativeExecutorReport step =
                a.executor->forward_n_tokens(chunk_tokens, final_chunk);
            if (!step.ok) throw std::runtime_error("prefill failed");
            const double t1 = wall_seconds();
            a.prefill_s += std::max(t1 - t0, 1e-9);
            a.prefill_ops += step.ops_executed;
            a.prefill_offset += chunk;
            a.kv_state.update(a.executor->kv_state_snapshot());

            // Commit the aligned prefix the moment prefill reaches its end.
            if (a.prefix_commit_pending &&
                a.prefill_offset == a.prefix_commit_len) {
                prefix_cache_commit(a, a.prefix_commit_len);
                a.prefix_commit_pending = false;
            }

            if (continuous_batching_trace_enabled()) {
                std::ostringstream msg;
                msg << "native continuous_prefill_chunk:"
                    << " request=" << a.req->id
                    << " offset=" << a.prefill_offset
                    << " total=" << prompt.size()
                    << " chunk=" << chunk
                    << " final=" << (final_chunk ? "true" : "false");
                log(msg.str());
            }

            if (!final_chunk) {
                if (prefilling.size() > 1) {
                    std::rotate(prefilling.begin(), prefilling.begin() + 1,
                                prefilling.end());
                }
                return;
            }

            a.decode_start = t1;
            // kvmem: first working-set build now that the full prompt is
            // prefilled (identity selection under the default budget).
            kvmem_on_prefill_complete(a);
            const int32_t seed = step.argmax_token >= 0 ? step.argmax_token : eos;
            a.next_token = budget_apply(
                a.budget,
                static_cast<uint32_t>(pick_continuous_next_token(a, seed)));

            if (a.req->options.max_tokens <= 0 ||
                (!a.req->options.ignore_eos &&
                 a.next_token == static_cast<uint32_t>(eos))) {
                if (a.req->options.max_tokens > 0) {
                    log_zero_decode_diagnostic("continuous",
                                               a.req->prompt_tokens,
                                               step);
                }
                complete_continuous_request(a.req, {});
                prefilling.erase(prefilling.begin());
                return;
            }

            emit_continuous_token(a, a.next_token);
            if (a.decoded >= a.req->options.max_tokens) {
                finish_continuous_active(a);
            } else {
                active.push_back(std::move(a));
            }
            prefilling.erase(prefilling.begin());
        } catch (const std::exception &e) {
            complete_continuous_request(a.req, {}, e.what());
            prefilling.erase(prefilling.begin());
        }
    }

    void build_continuous_prefill_batch(
            std::vector<ContinuousBatchActive> &prefilling,
            uint32_t max_chunks,
            ContinuousPrefillBatch &batch) {
        batch.clear();
        if (prefilling.empty() || max_chunks == 0) return;
        const uint32_t n =
            std::min<uint32_t>(max_chunks,
                               static_cast<uint32_t>(prefilling.size()));
        batch.entries.reserve(n);
        batch.q_indptr.reserve(static_cast<size_t>(n) + 1);
        batch.page_indptr.reserve(static_cast<size_t>(n) + 1);
        batch.last_page_len.reserve(n);
        batch.seq_lens.reserve(n);
        batch.q_indptr.push_back(0);
        batch.page_indptr.push_back(0);
        bool all_recurrent_state_ready = true;
        for (uint32_t i = 0; i < n; ++i) {
            ContinuousBatchActive &a = prefilling[i];
            if (!a.req) continue;
            const std::vector<uint32_t> &prompt = a.req->prompt_tokens;
            if (a.prefill_offset >= prompt.size() && !prompt.empty()) continue;
            const uint32_t remaining =
                a.prefill_offset < prompt.size()
                    ? static_cast<uint32_t>(prompt.size() - a.prefill_offset)
                    : 0u;
            ContinuousPrefillBatchEntry entry;
            entry.prefill_index = i;
            entry.request_id = a.req->id;
            entry.offset = a.prefill_offset;
            entry.total = static_cast<uint32_t>(prompt.size());
            entry.chunk = continuous_prefill_chunk_tokens(remaining);
            entry.final_chunk = prompt.empty() || entry.chunk >= remaining;
            if (entry.chunk > 0 && a.executor) {
                a.executor->prepare_runtime_state();
                a.executor->prepare_kv_pages(entry.offset, entry.chunk);
            }
            const QwenExecutor::DecodeStateView view =
                a.executor ? a.executor->decode_state_view()
                           : QwenExecutor::DecodeStateView{};
            bool entry_recurrent_ready =
                view.recurrent_states != nullptr &&
                view.conv_states != nullptr &&
                view.recurrent_states->size() >= weights_->n_layers() &&
                view.conv_states->size() >= weights_->n_layers();
            if (entry_recurrent_ready) {
                for (uint32_t il = 0; il < weights_->n_layers(); ++il) {
                    const QwenLayerWeights &layer = weights_->layer(il);
                    if (!layer.recurrent) continue;
                    if (!(*view.recurrent_states)[il] ||
                        !(*view.conv_states)[il]) {
                        entry_recurrent_ready = false;
                        break;
                    }
                }
            }
            all_recurrent_state_ready =
                all_recurrent_state_ready && entry_recurrent_ready;
            const uint32_t seq_len = entry.offset + entry.chunk;
            bool entry_metadata_ready =
                entry.chunk > 0 &&
                view.kv_page_size > 0 &&
                view.kv_page_indices_host != nullptr &&
                view.kv_page_count > 0;
            if (entry_metadata_ready) {
                if (batch.page_size == 0) {
                    batch.page_size = view.kv_page_size;
                } else if (batch.page_size != view.kv_page_size) {
                    entry_metadata_ready = false;
                }
            }
            if (entry_metadata_ready) {
                const uint32_t pages =
                    (seq_len + view.kv_page_size - 1) / view.kv_page_size;
                if (pages == 0 || view.kv_page_count < pages) {
                    entry_metadata_ready = false;
                } else {
                    const int32_t request_page_begin =
                        static_cast<int32_t>(batch.page_indices.size());
                    for (uint32_t p = 0; p < pages; ++p) {
                        batch.page_indices.push_back(view.kv_page_indices_host[p]);
                    }
                    for (uint32_t t = 0; t < entry.chunk; ++t) {
                        batch.logical_positions.push_back(
                            static_cast<int32_t>(entry.offset + t));
                        batch.row_page_indptr.push_back(request_page_begin);
                    }
                    const uint32_t last_len = seq_len % view.kv_page_size;
                    batch.last_page_len.push_back(static_cast<int32_t>(
                        last_len == 0 ? view.kv_page_size : last_len));
                    batch.seq_lens.push_back(static_cast<int32_t>(seq_len));
                    batch.max_seq_len = std::max(batch.max_seq_len, seq_len);
                }
            }
            if (!entry_metadata_ready) {
                batch.page_indices.clear();
                batch.page_indptr.clear();
                batch.row_page_indptr.clear();
                batch.logical_positions.clear();
                batch.last_page_len.clear();
                batch.seq_lens.clear();
                batch.page_size = 0;
                batch.max_seq_len = 0;
            }
            batch.total_tokens += entry.chunk;
            if (entry.final_chunk) ++batch.final_chunks;
            batch.entries.push_back(entry);
            batch.q_indptr.push_back(static_cast<int32_t>(batch.total_tokens));
            if (!batch.page_indptr.empty()) {
                batch.page_indptr.push_back(
                    static_cast<int32_t>(batch.page_indices.size()));
            }
        }
        batch.recurrent_state_ready =
            batch.size() > 0 && all_recurrent_state_ready;
        if (!batch.row_page_indptr.empty()) {
            batch.row_page_indptr.push_back(
                static_cast<int32_t>(batch.page_indices.size()));
        }
        batch.ragged_metadata_ready =
            batch.size() > 0 &&
            batch.q_indptr.size() == batch.size() + 1 &&
            batch.page_indptr.size() == batch.size() + 1 &&
            batch.logical_positions.size() == batch.total_tokens &&
            batch.row_page_indptr.size() == batch.total_tokens + 1 &&
            batch.last_page_len.size() == batch.size() &&
            batch.seq_lens.size() == batch.size() &&
            !batch.page_indices.empty() &&
            batch.page_size > 0;
        batch.ragged_row_metadata_ready = batch.ragged_metadata_ready;
    }

    void ensure_continuous_prefill_ragged_metadata_device(
            uint32_t batch_size,
            uint32_t pages,
            uint32_t total_q) {
        if (batch_size > cb_prefill_ragged_batch_capacity_) {
            cb_prefill_q_indptr_i32_ =
                device_->tensor_i32(static_cast<uint64_t>(batch_size) + 1,
                                    "cb_prefill_q_indptr_i32");
            cb_prefill_page_indptr_i32_ =
                device_->tensor_i32(static_cast<uint64_t>(batch_size) + 1,
                                    "cb_prefill_page_indptr_i32");
            cb_prefill_last_page_len_i32_ =
                device_->tensor_i32(batch_size,
                                    "cb_prefill_last_page_len_i32");
            cb_prefill_seq_lens_i32_ =
                device_->tensor_i32(batch_size, "cb_prefill_seq_lens_i32");
            cb_prefill_ragged_batch_capacity_ = batch_size;
        }
        if (pages > cb_prefill_ragged_page_capacity_) {
            cb_prefill_page_indices_i32_ =
                device_->tensor_i32(pages, "cb_prefill_page_indices_i32");
            cb_prefill_ragged_page_capacity_ = pages;
        }
        if (total_q > cb_prefill_ragged_row_capacity_) {
            cb_prefill_logical_positions_i32_ =
                device_->tensor_i32(total_q,
                                    "cb_prefill_logical_positions_i32");
            cb_prefill_row_page_indptr_i32_ =
                device_->tensor_i32(static_cast<uint64_t>(total_q) + 1,
                                    "cb_prefill_row_page_indptr_i32");
            cb_prefill_ragged_row_capacity_ = total_q;
        }
    }

    static void require_device_status(const DeviceStatus &st) {
        if (!st.ok) throw std::runtime_error(st.message);
    }

    void prepare_continuous_prefill_ragged_metadata_device(
            ContinuousPrefillBatch &batch) {
        batch.ragged_device_metadata_ready = false;
        if (!batch.ragged_metadata_ready || batch.size() == 0) return;
        const uint32_t bsz = static_cast<uint32_t>(batch.size());
        const uint32_t pages = static_cast<uint32_t>(batch.page_indices.size());
        const uint32_t total_q = static_cast<uint32_t>(batch.total_tokens);
        ensure_continuous_prefill_ragged_metadata_device(bsz, pages, total_q);
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_q_indptr_i32_, 0, batch.q_indptr.data(),
            static_cast<uint64_t>(bsz) + 1));
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_page_indptr_i32_, 0, batch.page_indptr.data(),
            static_cast<uint64_t>(bsz) + 1));
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_page_indices_i32_, 0, batch.page_indices.data(), pages));
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_logical_positions_i32_, 0,
            batch.logical_positions.data(), total_q));
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_row_page_indptr_i32_, 0,
            batch.row_page_indptr.data(), static_cast<uint64_t>(total_q) + 1));
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_last_page_len_i32_, 0,
            batch.last_page_len.data(), bsz));
        require_device_status(device_->copy_i32_from_host(
            *cb_prefill_seq_lens_i32_, 0, batch.seq_lens.data(), bsz));
        batch.ragged_device_metadata_ready = true;
    }

    void ensure_continuous_prefill_recurrent_state_scratch(
            uint32_t batch_size,
            uint64_t state_stride,
            uint64_t conv_state_stride) {
        const uint64_t state_count =
            static_cast<uint64_t>(batch_size) * state_stride;
        const uint64_t conv_count =
            static_cast<uint64_t>(batch_size) * conv_state_stride;
        if (state_count > cb_prefill_recurrent_state_capacity_) {
            cb_prefill_recurrent_state_batch_ =
                device_->scratch_f32(state_count,
                                     "cb_prefill_recurrent_state_batch");
            cb_prefill_recurrent_state_capacity_ = state_count;
        }
        if (conv_count > cb_prefill_conv_state_capacity_) {
            cb_prefill_conv_state_batch_ =
                device_->scratch_f32(conv_count,
                                     "cb_prefill_conv_state_batch");
            cb_prefill_conv_state_capacity_ = conv_count;
        }
    }

    void pack_continuous_prefill_recurrent_state_batch(
            std::vector<ContinuousBatchActive> &prefilling,
            ContinuousPrefillBatch &batch) {
        batch.recurrent_state_packed = false;
        batch.recurrent_state_unpacked = false;
        batch.recurrent_state_packed_layers = 0;
        if (!batch.recurrent_state_ready || batch.size() == 0) return;
        const uint32_t bsz = static_cast<uint32_t>(batch.size());
        uint32_t unpacked_layers = 0;
        for (uint32_t il = 0; il < weights_->n_layers(); ++il) {
            const QwenLayerWeights &layer = weights_->layer(il);
            if (!layer.recurrent) continue;
            uint64_t state_stride = 0;
            uint64_t conv_state_stride = 0;
            for (const ContinuousPrefillBatchEntry &entry : batch.entries) {
                if (entry.prefill_index >= prefilling.size()) return;
                QwenExecutor::DecodeStateView view =
                    prefilling[entry.prefill_index].executor->decode_state_view();
                if (!view.recurrent_states || !view.conv_states ||
                    il >= view.recurrent_states->size() ||
                    il >= view.conv_states->size() ||
                    !(*view.recurrent_states)[il] ||
                    !(*view.conv_states)[il]) {
                    return;
                }
                const uint64_t row_state =
                    (*view.recurrent_states)[il]->count;
                const uint64_t row_conv =
                    (*view.conv_states)[il]->count;
                if (state_stride == 0) {
                    state_stride = row_state;
                    conv_state_stride = row_conv;
                } else if (state_stride != row_state ||
                           conv_state_stride != row_conv) {
                    return;
                }
            }
            if (state_stride == 0 || conv_state_stride == 0) return;
            ensure_continuous_prefill_recurrent_state_scratch(
                bsz, state_stride, conv_state_stride);
            for (uint32_t row = 0; row < bsz; ++row) {
                const ContinuousPrefillBatchEntry &entry = batch.entries[row];
                QwenExecutor::DecodeStateView view =
                    prefilling[entry.prefill_index].executor->decode_state_view();
                require_device_status(device_->copy_d2d_into(
                    *cb_prefill_recurrent_state_batch_,
                    static_cast<uint64_t>(row) * state_stride,
                    *(*view.recurrent_states)[il], 0, state_stride));
                require_device_status(device_->copy_d2d_into(
                    *cb_prefill_conv_state_batch_,
                    static_cast<uint64_t>(row) * conv_state_stride,
                    *(*view.conv_states)[il], 0, conv_state_stride));
            }
            for (uint32_t row = 0; row < bsz; ++row) {
                const ContinuousPrefillBatchEntry &entry = batch.entries[row];
                QwenExecutor::DecodeStateView view =
                    prefilling[entry.prefill_index].executor->decode_state_view();
                require_device_status(device_->copy_d2d_into(
                    *(*view.recurrent_states)[il], 0,
                    *cb_prefill_recurrent_state_batch_,
                    static_cast<uint64_t>(row) * state_stride,
                    state_stride));
                require_device_status(device_->copy_d2d_into(
                    *(*view.conv_states)[il], 0,
                    *cb_prefill_conv_state_batch_,
                    static_cast<uint64_t>(row) * conv_state_stride,
                    conv_state_stride));
            }
            ++batch.recurrent_state_packed_layers;
            ++unpacked_layers;
        }
        batch.recurrent_state_packed =
            batch.recurrent_state_packed_layers > 0;
        batch.recurrent_state_unpacked =
            batch.recurrent_state_packed &&
            unpacked_layers == batch.recurrent_state_packed_layers;
    }

    void apply_continuous_prefill_batch_outputs(
            std::vector<ContinuousBatchActive> &prefilling,
            std::vector<ContinuousBatchActive> &active,
            const std::vector<BatchedPrefillOutput> &outputs,
            int32_t eos) {
        if (outputs.empty()) return;
        std::vector<uint8_t> state(prefilling.size(), 0);
        constexpr uint8_t kProcessedNonFinal = 1;
        constexpr uint8_t kRemove = 2;

        for (const BatchedPrefillOutput &out : outputs) {
            if (out.prefill_index >= prefilling.size()) continue;
            ContinuousBatchActive &a = prefilling[out.prefill_index];
            if (!out.error.empty()) {
                complete_continuous_request(a.req, {}, out.error);
                state[out.prefill_index] = kRemove;
                continue;
            }
            if (!out.ok()) {
                complete_continuous_request(a.req, {}, "prefill failed");
                state[out.prefill_index] = kRemove;
                continue;
            }

            if (continuous_batching_trace_enabled()) {
                std::ostringstream msg;
                msg << "native continuous_prefill_chunk:"
                    << " request=" << out.request_id
                    << " offset=" << (out.offset + out.chunk)
                    << " total=" << out.total
                    << " chunk=" << out.chunk
                    << " final=" << (out.final_chunk ? "true" : "false");
                log(msg.str());
            }

            if (!out.final_chunk) {
                state[out.prefill_index] = kProcessedNonFinal;
                continue;
            }

            a.decode_start = wall_seconds();
            // kvmem: first working-set build now that the full prompt is
            // prefilled (identity selection under the default budget).
            kvmem_on_prefill_complete(a);
            const int32_t seed =
                out.report.argmax_token >= 0 ? out.report.argmax_token : eos;
            a.next_token = budget_apply(
                a.budget,
                static_cast<uint32_t>(pick_continuous_next_token(a, seed)));

            if (a.req->options.max_tokens <= 0 ||
                (!a.req->options.ignore_eos &&
                 a.next_token == static_cast<uint32_t>(eos))) {
                if (a.req->options.max_tokens > 0) {
                    log_zero_decode_diagnostic("continuous",
                                               a.req->prompt_tokens,
                                               out.report);
                }
                complete_continuous_request(a.req, {});
                state[out.prefill_index] = kRemove;
                continue;
            }

            emit_continuous_token(a, a.next_token);
            if (a.decoded >= a.req->options.max_tokens) {
                finish_continuous_active(a);
            } else {
                active.push_back(std::move(a));
            }
            state[out.prefill_index] = kRemove;
        }

        std::vector<ContinuousBatchActive> remaining;
        std::vector<ContinuousBatchActive> processed_nonfinal;
        remaining.reserve(prefilling.size());
        processed_nonfinal.reserve(prefilling.size());
        for (size_t i = 0; i < prefilling.size(); ++i) {
            if (state[i] == kRemove) continue;
            if (state[i] == kProcessedNonFinal) {
                processed_nonfinal.push_back(std::move(prefilling[i]));
            } else {
                remaining.push_back(std::move(prefilling[i]));
            }
        }
        for (auto &a : processed_nonfinal) {
            remaining.push_back(std::move(a));
        }
        prefilling.swap(remaining);
    }

    void advance_continuous_prefill_batch(
            std::vector<ContinuousBatchActive> &prefilling,
            std::vector<ContinuousBatchActive> &active,
            int32_t eos,
            uint32_t max_chunks) {
        if (prefilling.empty() || max_chunks == 0) return;
        const bool use_batch_boundary =
            continuous_batching_prefill_batch_enabled() &&
            !prefix_cache_enabled() &&
            active.empty() && max_chunks > 1 && prefilling.size() > 1;
        if (!use_batch_boundary) {
            for (uint32_t step = 0;
                 step < max_chunks && !prefilling.empty();
                 ++step) {
                advance_continuous_prefill(prefilling, active, eos);
                if (!active.empty()) break;
            }
            return;
        }

        ContinuousPrefillBatch batch;
        build_continuous_prefill_batch(prefilling, max_chunks, batch);
        if (batch.size() == 0) return;
        prepare_continuous_prefill_ragged_metadata_device(batch);
        if (continuous_batching_prefill_pack_recurrent_state_enabled()) {
            pack_continuous_prefill_recurrent_state_batch(prefilling, batch);
        }

        if (continuous_batching_trace_enabled()) {
            std::ostringstream msg;
            msg << "native continuous_prefill_batch:"
                << " mode=delegated"
                << " chunks=" << batch.size()
                << " tokens=" << batch.total_tokens
                << " final_chunks=" << batch.final_chunks
                << " first_request=" << batch.entries.front().request_id
                << " first_offset=" << batch.entries.front().offset
                << " ragged_metadata_ready="
                << (batch.ragged_metadata_ready ? "true" : "false")
                << " ragged_device_metadata_ready="
                << (batch.ragged_device_metadata_ready ? "true" : "false")
                << " ragged_row_metadata_ready="
                << (batch.ragged_row_metadata_ready ? "true" : "false")
                << " recurrent_state_ready="
                << (batch.recurrent_state_ready ? "true" : "false")
                << " recurrent_state_packed="
                << (batch.recurrent_state_packed ? "true" : "false")
                << " recurrent_state_unpacked="
                << (batch.recurrent_state_unpacked ? "true" : "false")
                << " recurrent_state_packed_layers="
                << batch.recurrent_state_packed_layers
                << " ragged_pages=" << batch.page_indices.size()
                << " ragged_max_seq_len=" << batch.max_seq_len;
            log(msg.str());
        }

        if (!cb_prefill_executor_) {
            cb_prefill_executor_ =
                std::make_unique<BatchedPrefillExecutor>(
                    *model_, *weights_, *device_);
        }
        BatchedPrefillDeviceMetadata prefill_metadata;
        prefill_metadata.q_indptr = cb_prefill_q_indptr_i32_.get();
        prefill_metadata.page_indptr = cb_prefill_page_indptr_i32_.get();
        prefill_metadata.row_page_indptr = cb_prefill_row_page_indptr_i32_.get();
        prefill_metadata.page_indices = cb_prefill_page_indices_i32_.get();
        prefill_metadata.logical_positions =
            cb_prefill_logical_positions_i32_.get();
        prefill_metadata.last_page_len = cb_prefill_last_page_len_i32_.get();
        prefill_metadata.seq_lens = cb_prefill_seq_lens_i32_.get();
        const std::vector<BatchedPrefillOutput> outputs =
            cb_prefill_executor_->prefill(prefilling, batch, prefill_metadata);
        uint32_t executed_chunks = 0;
        uint64_t executed_tokens = 0;
        uint32_t completed_chunks = 0;
        for (const BatchedPrefillOutput &out : outputs) {
            ++executed_chunks;
            executed_tokens += out.chunk;
            if (out.final_chunk && out.ok()) ++completed_chunks;
        }
        apply_continuous_prefill_batch_outputs(prefilling, active, outputs, eos);

        if (continuous_batching_timing_enabled() ||
            continuous_batching_trace_enabled()) {
            const BatchedPrefillTiming &timing =
                cb_prefill_executor_->last_timing();
            std::ostringstream msg;
            msg << "native continuous_prefill_batch_done:"
                << " mode=" << cb_prefill_executor_->last_mode()
                << " kernel_batch="
                << cb_prefill_executor_->last_kernel_batch()
                << " chunks=" << executed_chunks
                << " tokens=" << executed_tokens
                << " completed_chunks=" << completed_chunks
                << " total=" << fmt_seconds(timing.total_s)
                << " prepare=" << fmt_seconds(timing.prepare_s)
                << " embed=" << fmt_seconds(timing.embed_s)
                << " layers=" << fmt_seconds(timing.layers_s)
                << " recurrent=" << fmt_seconds(timing.recurrent_s)
                << " attention=" << fmt_seconds(timing.attention_s)
                << " ffn=" << fmt_seconds(timing.ffn_s)
                << " final=" << fmt_seconds(timing.final_s)
                << " post=" << fmt_seconds(timing.post_s)
                << " ragged=" << fmt_seconds(timing.ragged_s)
                << " delegated=" << fmt_seconds(timing.delegated_s);
            log(msg.str());
        }
    }

    void build_continuous_decode_batch(const std::vector<ContinuousBatchActive> &active,
                                       ContinuousDecodeBatch &batch) {
        batch.clear();
        batch.active_indices.reserve(active.size());
        batch.feed_tokens.reserve(active.size());
        batch.positions.reserve(active.size());
        batch.state_views.reserve(active.size());
        for (size_t i = 0; i < active.size(); ++i) {
            QwenExecutor::DecodeStateView view = active[i].executor->decode_state_view();
            batch.active_indices.push_back(i);
            batch.feed_tokens.push_back(active[i].next_token);
            batch.positions.push_back(view.position);
            batch.state_views.push_back(view);
        }
    }

    static bool continuous_decode_batch_has_paged_kv(
            const ContinuousDecodeBatch &batch) {
        if (batch.size() == 0) return false;
        for (const auto &view : batch.state_views) {
            if (view.kv_page_size == 0 ||
                view.kv_page_count == 0 ||
                view.kv_page_indices_device == nullptr ||
                ((view.k_cache == nullptr || view.k_cache->empty()) &&
                 (view.k_cache_external == nullptr ||
                  view.k_cache_external->empty())) ||
                ((view.v_cache == nullptr || view.v_cache->empty()) &&
                 (view.v_cache_external == nullptr ||
                  view.v_cache_external->empty()))) {
                return false;
            }
        }
        return true;
    }

    void continuous_decode_batch_step(std::vector<ContinuousBatchActive> &active,
                                      const ContinuousDecodeBatch &batch,
                                      int32_t eos) {
        if (batch.size() == 0) return;
        if (continuous_batching_trace_enabled()) {
            std::ostringstream msg;
            msg << "native continuous_batch_step:"
                << " decode_executor=delegated"
                << " batch=" << batch.size()
                << " paged_kv_ready="
                << (continuous_decode_batch_has_paged_kv(batch) ? "true" : "false")
                << " total_batches=" << (cb_decode_batches_.load() + 1)
                << " total_tokens=" << (cb_decode_tokens_.load() + batch.size());
            log(msg.str());
        }
        ++cb_decode_batches_;
        cb_decode_tokens_ += batch.size();
        uint32_t prev_max = cb_decode_max_batch_.load();
        while (batch.size() > prev_max &&
               !cb_decode_max_batch_.compare_exchange_weak(
                   prev_max, static_cast<uint32_t>(batch.size()))) {}

        if (!cb_decode_executor_) {
            cb_decode_executor_ =
                std::make_unique<BatchedDecodeExecutor>(*model_, *weights_, *device_);
        }
        const std::vector<BatchedDecodeOutput> outputs =
            cb_decode_executor_->decode(active, BatchedDecodeInput{&batch});
        if (continuous_batching_timing_enabled()) {
            const BatchedDecodeTiming &timing = cb_decode_executor_->last_timing();
            std::ostringstream msg;
            msg << "native continuous_batch_timing:"
                << " mode=" << cb_decode_executor_->last_mode()
                << " batch=" << batch.size()
                << " kernel_batch=" << cb_decode_executor_->last_kernel_batch()
                << " total=" << fmt_seconds(timing.total_s)
                << " prepare=" << fmt_seconds(timing.prepare_s)
                << " metadata=" << fmt_seconds(timing.metadata_s)
                << " embed=" << fmt_seconds(timing.embed_s)
                << " layers=" << fmt_seconds(timing.layers_s)
                << " recurrent=" << fmt_seconds(timing.recurrent_s)
                << " recurrent_state="
                << fmt_seconds(timing.recurrent_state_s)
                << " attention=" << fmt_seconds(timing.attention_s)
                << " qkv=" << fmt_seconds(timing.qkv_s)
                << " kv_append=" << fmt_seconds(timing.kv_append_s)
                << " attn_kernel=" << fmt_seconds(timing.attn_kernel_s)
                << " attn_output=" << fmt_seconds(timing.attn_output_s)
                << " ffn=" << fmt_seconds(timing.ffn_s)
                << " final=" << fmt_seconds(timing.final_s)
                << " lm_head=" << fmt_seconds(timing.lm_head_s)
                << " argmax=" << fmt_seconds(timing.argmax_s)
                << " post=" << fmt_seconds(timing.post_s)
                << " ragged_pages="
                << cb_decode_executor_->last_ragged_metadata_pages()
                << " ragged_max_seq_len="
                << cb_decode_executor_->last_ragged_metadata_max_seq_len();
            log(msg.str());
        }
        if (continuous_batching_trace_enabled()) {
            std::ostringstream msg;
            msg << "native continuous_batch_executor:"
                << " mode=" << cb_decode_executor_->last_mode()
                << " scheduler_batch=" << batch.size()
                << " kernel_batch=" << cb_decode_executor_->last_kernel_batch()
                << " body_batch_ready="
                << (cb_decode_executor_->last_body_batch_ready()
                    ? "true" : "false")
                << " ragged_metadata_ready="
                << (cb_decode_executor_->last_ragged_metadata_ready()
                    ? "true" : "false")
                << " ragged_pages="
                << cb_decode_executor_->last_ragged_metadata_pages()
                << " ragged_max_seq_len="
                << cb_decode_executor_->last_ragged_metadata_max_seq_len();
            log(msg.str());
        }
        for (const BatchedDecodeOutput &out : outputs) {
            const size_t active_index = out.active_index;
            if (active_index >= active.size()) continue;
            ContinuousBatchActive &a = active[active_index];
            try {
                if (!out.error.empty()) throw std::runtime_error(out.error);
                const NativeExecutorReport &step = out.report;
                if (!out.ok()) throw std::runtime_error("decode failed");
                a.decode_ops += step.ops_executed;
                a.kv_state.update(a.executor->kv_state_snapshot());
                // kvmem: the token just decoded grew this request's context by
                // one. Register it and reselect on the interval boundary.
                kvmem_on_decode_step(a);
                const int32_t next = step.argmax_token >= 0 ? step.argmax_token : eos;
                a.next_token = budget_apply(
                    a.budget,
                    static_cast<uint32_t>(pick_continuous_next_token(a, next)));
                if (!a.req->options.ignore_eos &&
                    a.next_token == static_cast<uint32_t>(eos)) {
                    finish_continuous_active(a);
                    a.req.reset();
                    continue;
                }
                emit_continuous_token(a, a.next_token);
                if (a.decoded >= a.req->options.max_tokens) {
                    finish_continuous_active(a);
                    a.req.reset();
                    continue;
                }
            } catch (const std::exception &e) {
                complete_continuous_request(a.req, {}, e.what());
                a.req.reset();
            }
        }
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [](const ContinuousBatchActive &a) {
                                        return !a.req;
                                    }),
        active.end());
    }

    int32_t pick_continuous_next_token(ContinuousBatchActive &a,
                                       int32_t fallback_argmax) {
        const GenerationOptions &options = a.req->options;
        const bool do_sample = options.temperature > 0.0f;
        const bool need_logits =
            do_sample ||
            options.presence_penalty != 0.0f ||
            (options.repetition_penalty > 0.0f &&
             options.repetition_penalty != 1.0f);
        if (!need_logits) return fallback_argmax;
        if (!a.executor->copy_last_logits(a.logit_buf)) return fallback_argmax;
        apply_token_penalties(a.logit_buf, a.seen_tokens,
                              options.presence_penalty,
                              options.repetition_penalty);
        if (!do_sample) {
            int best = 0;
            float bv = a.logit_buf.empty()
                ? -std::numeric_limits<float>::infinity()
                : a.logit_buf[0];
            for (int i = 1; i < static_cast<int>(a.logit_buf.size()); ++i) {
                if (a.logit_buf[i] > bv) {
                    bv = a.logit_buf[i];
                    best = i;
                }
            }
            return a.logit_buf.empty() ? fallback_argmax : best;
        }
        const int32_t token = sample_token(a.logit_buf, options.temperature,
                                           options.top_p, options.top_k,
                                           options.min_p, a.rng);
        return token >= 0 ? token : fallback_argmax;
    }

    void emit_continuous_token(ContinuousBatchActive &a, uint32_t token) {
        const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(token));
        a.req->generated += piece;
        if (a.req->on_text) a.req->on_text(piece);
        ++a.seen_tokens[token];
        ++a.decoded;
    }

    // ---- Prefix cache methods (Phase 1) -----------------------------------

    void prefix_require_ok(const DeviceStatus &st) {
        if (!st.ok) throw std::runtime_error(st.message);
    }

    static uint64_t prefix_cache_hash(const uint32_t *tokens, size_t n) {
        // FNV-1a over the token id bytes. Only used to bucket; exact-token
        // comparison defeats collisions.
        uint64_t h = 1469598103934665603ULL;
        for (size_t i = 0; i < n; ++i) {
            uint32_t t = tokens[i];
            for (int b = 0; b < 4; ++b) {
                h ^= static_cast<uint64_t>(t & 0xFFu);
                h *= 1099511628211ULL;
                t >>= 8;
            }
        }
        return h;
    }

    void prefix_cache_install_evict_cb() {
        if (prefix_cache_evict_cb_installed_ || !cb_kv_pool_) return;
        cb_kv_pool_->set_evict_callback([this]() -> uint32_t {
            return prefix_cache_evict_lru(1);
        });
        prefix_cache_evict_cb_installed_ = true;
    }

    // Longest cached entry whose `tokens` is an exact prefix of `prompt` and
    // strictly shorter than it. On hit, bumps refcount + LRU and returns a
    // copy of the entry's pages/state via out-params; caller seeds the executor.
    // Returns the entry id on hit, 0 on miss.
    uint64_t prefix_cache_lookup(const std::vector<uint32_t> &prompt,
                                 uint32_t page_size,
                                 std::vector<int32_t> &out_pages,
                                 QwenExecutor::StateSnapshot &out_recur,
                                 uint32_t &out_aligned_len) {
        if (!prefix_cache_enabled() || prompt.size() < 2 * page_size) return 0;
        std::lock_guard<std::mutex> lk(prefix_cache_mu_);
        const uint32_t prompt_len = static_cast<uint32_t>(prompt.size());
        // Probe page-aligned prefix lengths from longest to shortest. The
        // longest aligned length strictly below prompt_len is the first probe.
        uint32_t probe = (prompt_len / page_size) * page_size;
        if (probe >= prompt_len) probe -= page_size;
        for (; probe >= page_size; probe -= page_size) {
            const uint64_t h = prefix_cache_hash(prompt.data(), probe);
            auto it = prefix_cache_.find(h);
            if (it == prefix_cache_.end()) continue;
            for (PrefixCacheEntry &e : it->second) {
                if (e.aligned_len != probe) continue;
                if (e.tokens.size() != probe) continue;
                if (!std::equal(e.tokens.begin(), e.tokens.end(),
                                prompt.begin())) {
                    continue;  // hash collision
                }
                // Hit. Pin already held by the entry; bump refcount + LRU.
                ++e.refcount;
                e.last_used_seq = ++prefix_cache_seq_;
                out_pages = e.kv_pages;
                out_recur = clone_state_snapshot(e.recur);
                out_aligned_len = e.aligned_len;
                return e.id;
            }
        }
        return 0;
    }

    // Deep-copy a StateSnapshot (device tensors) so the adopting executor gets
    // its own restorable copy and the cache entry's master copy is untouched.
    QwenExecutor::StateSnapshot clone_state_snapshot(
            const QwenExecutor::StateSnapshot &src) {
        QwenExecutor::StateSnapshot dst;
        dst.ready = src.ready;
        dst.position = src.position;
        dst.kv_logical_pages = src.kv_logical_pages;
        dst.mtp_prefix_len = src.mtp_prefix_len;
        if (src.h) {
            dst.h = device_->scratch_f32(src.h->count, "prefix_clone_h");
            prefix_require_ok(device_->copy_d2d(*dst.h, *src.h, 0, src.h->count));
        }
        dst.recurrent_states.resize(src.recurrent_states.size());
        dst.conv_states.resize(src.conv_states.size());
        for (size_t i = 0; i < src.recurrent_states.size(); ++i) {
            if (src.recurrent_states[i]) {
                dst.recurrent_states[i] = device_->scratch_f32(
                    src.recurrent_states[i]->count, "prefix_clone_recur");
                prefix_require_ok(device_->copy_d2d(*dst.recurrent_states[i],
                                             *src.recurrent_states[i], 0,
                                             src.recurrent_states[i]->count));
            }
        }
        for (size_t i = 0; i < src.conv_states.size(); ++i) {
            if (src.conv_states[i]) {
                dst.conv_states[i] = device_->scratch_f32(
                    src.conv_states[i]->count, "prefix_clone_conv");
                prefix_require_ok(device_->copy_d2d(*dst.conv_states[i],
                                             *src.conv_states[i], 0,
                                             src.conv_states[i]->count));
            }
        }
        return dst;
    }

    // Commit the page-aligned prefix [0..aligned_len) of a freshly-prefilled
    // request: snapshot recurrent state (already at aligned_len), pin the
    // prefix KV pages, mark them borrowed in the executor, and insert an entry.
    // The committing request itself holds a refcount (released at finish).
    void prefix_cache_commit(ContinuousBatchActive &a, uint32_t aligned_len) {
        if (!prefix_cache_enabled() || aligned_len == 0 || !a.executor) return;
        const uint32_t page_size = a.executor->kv_page_size_public();
        if (page_size == 0 || (aligned_len % page_size) != 0) return;
        const uint32_t prefix_pages = aligned_len / page_size;

        // Don't duplicate an entry that already covers this exact prefix.
        const std::vector<uint32_t> &prompt = a.req->prompt_tokens;
        const uint64_t h = prefix_cache_hash(prompt.data(), aligned_len);
        {
            std::lock_guard<std::mutex> lk(prefix_cache_mu_);
            auto it = prefix_cache_.find(h);
            if (it != prefix_cache_.end()) {
                for (PrefixCacheEntry &e : it->second) {
                    if (e.aligned_len == aligned_len &&
                        e.tokens.size() == aligned_len &&
                        std::equal(e.tokens.begin(), e.tokens.end(),
                                   prompt.begin())) {
                        ++e.refcount;  // creator holds it too
                        e.last_used_seq = ++prefix_cache_seq_;
                        a.held_prefix_entries.push_back(e.id);
                        return;
                    }
                }
            }
        }

        PrefixCacheEntry entry;
        entry.aligned_len = aligned_len;
        entry.tokens.assign(prompt.begin(),
                            prompt.begin() + static_cast<std::ptrdiff_t>(aligned_len));
        // Capture recurrent+conv state at exactly aligned_len. The executor's
        // position_ is aligned_len here (caller invokes at the boundary).
        a.executor->capture_state(entry.recur);
        // Mark prefix pages borrowed in the executor + collect their physical
        // ids; the executor keeps reading them but will not free them.
        entry.kv_pages = a.executor->mark_kv_prefix_shared(prefix_pages);
        if (entry.kv_pages.size() != prefix_pages) return;  // nothing to share
        if (cb_kv_pool_) cb_kv_pool_->pin_pages(entry.kv_pages);

        uint64_t eid = 0;
        {
            std::lock_guard<std::mutex> lk(prefix_cache_mu_);
            entry.id = prefix_cache_next_id_++;
            entry.refcount = 1;  // creator holds it until finish
            entry.last_used_seq = ++prefix_cache_seq_;
            prefix_cache_pinned_pages_ += prefix_pages;
            a.held_prefix_entries.push_back(entry.id);
            eid = entry.id;
            const uint32_t alen = entry.aligned_len;
            prefix_cache_[h].push_back(std::move(entry));
            if (prefix_cache_trace_enabled()) {
                std::ostringstream m;
                m << "prefix_cache commit id=" << eid
                  << " req=" << a.req->id
                  << " aligned_len=" << alen
                  << " pages=" << prefix_pages
                  << " pinned_pages=" << prefix_cache_pinned_pages_;
                log(m.str());
            }
        }
        (void) eid;
        // Enforce the page budget: evict LRU refcount==0 entries until we are
        // within QW3_PREFIX_CACHE_MAX_PAGES (0 = unlimited). The just-committed
        // entry has refcount 1 so it is never the victim here.
        const uint32_t budget = prefix_cache_max_pages();
        if (budget > 0) {
            for (int guard = 0; guard < 4096; ++guard) {
                uint32_t pinned;
                {
                    std::lock_guard<std::mutex> lk(prefix_cache_mu_);
                    pinned = prefix_cache_pinned_pages_;
                }
                if (pinned <= budget) break;
                if (prefix_cache_evict_lru(1) == 0) break;  // nothing evictable
            }
        }
    }

    // Drop refcounts this request holds. Pinned pages are NOT freed here; they
    // stay until the entry is evicted at refcount 0.
    void prefix_cache_release(ContinuousBatchActive &a) {
        if (a.held_prefix_entries.empty()) return;
        std::lock_guard<std::mutex> lk(prefix_cache_mu_);
        for (uint64_t eid : a.held_prefix_entries) {
            for (auto &kv : prefix_cache_) {
                for (PrefixCacheEntry &e : kv.second) {
                    if (e.id == eid && e.refcount > 0) { --e.refcount; break; }
                }
            }
        }
        a.held_prefix_entries.clear();
    }

    // Evict up to `want` LRU entries with refcount==0, unpin + free their pages
    // back to the pool. Returns the number of ENTRIES evicted (the progress
    // signal for allocate_physical_page's retry loop), NOT the page count: an
    // entry whose pages are all shared with a longer cached entry reclaims ZERO
    // physical pages on its own (its pages stay pinned by the other entry), yet
    // dropping it is still progress -- the next eviction can then free the
    // uniquely-owning entry. Called by the pool's evict callback (pool mutex
    // NOT held here).
    uint32_t prefix_cache_evict_lru(uint32_t want) {
        std::lock_guard<std::mutex> lk(prefix_cache_mu_);
        uint32_t evicted_entries = 0;
        for (uint32_t n = 0; n < want; ++n) {
            // Find the global LRU evictable entry.
            uint64_t best_seq = UINT64_MAX;
            uint64_t best_hash = 0;
            size_t best_idx = 0;
            bool found = false;
            for (auto &kv : prefix_cache_) {
                for (size_t i = 0; i < kv.second.size(); ++i) {
                    const PrefixCacheEntry &e = kv.second[i];
                    if (e.refcount != 0) continue;
                    if (e.last_used_seq < best_seq) {
                        best_seq = e.last_used_seq;
                        best_hash = kv.first;
                        best_idx = i;
                        found = true;
                    }
                }
            }
            if (!found) break;
            auto &bucket = prefix_cache_[best_hash];
            PrefixCacheEntry &victim = bucket[best_idx];
            if (cb_kv_pool_) {
                cb_kv_pool_->unpin_pages(victim.kv_pages);
                cb_kv_pool_->release_physical_pages(victim.kv_pages);
            }
            const uint32_t pages = static_cast<uint32_t>(victim.kv_pages.size());
            ++evicted_entries;
            if (prefix_cache_pinned_pages_ >= pages) {
                prefix_cache_pinned_pages_ -= pages;
            }
            if (prefix_cache_trace_enabled()) {
                std::ostringstream m;
                m << "prefix_cache evict id=" << victim.id
                  << " aligned_len=" << victim.aligned_len
                  << " pages=" << pages
                  << " pinned_pages=" << prefix_cache_pinned_pages_;
                log(m.str());
            }
            bucket.erase(bucket.begin() + static_cast<std::ptrdiff_t>(best_idx));
            if (bucket.empty()) prefix_cache_.erase(best_hash);
        }
        return evicted_entries;
    }

    void prefix_cache_clear() {
        std::lock_guard<std::mutex> lk(prefix_cache_mu_);
        for (auto &kv : prefix_cache_) {
            for (PrefixCacheEntry &e : kv.second) {
                if (cb_kv_pool_) {
                    cb_kv_pool_->unpin_pages(e.kv_pages);
                    cb_kv_pool_->release_physical_pages(e.kv_pages);
                }
            }
        }
        prefix_cache_.clear();
        prefix_cache_pinned_pages_ = 0;
    }

    void finish_continuous_active(ContinuousBatchActive &a) {
        // Release any prefix-cache entries this request held (adopted or
        // committed) before tearing down the executor: dropping the refcount
        // makes the entry evictable, and the executor dtor frees only its
        // private (owned) suffix pages — never the pinned shared prefix.
        prefix_cache_release(a);
        const double decode_s = std::max(wall_seconds() - a.decode_start, 1e-9);
        std::ostringstream msg;
        msg << "native continuous_batch:"
            << " request=" << a.req->id
            << " prompt_tokens=" << a.req->prompt_tokens.size()
            << " prefill=" << fmt_seconds(a.prefill_s);
        if (!a.req->prompt_tokens.empty()) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (a.req->prompt_tokens.size() / a.prefill_s) << " tok/s)";
        }
        msg << " decoded=" << a.decoded
            << " decode=" << fmt_seconds(decode_s);
        if (a.decoded > 0) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (a.decoded / decode_s) << " tok/s)";
        }
        msg << " prefill_ops=" << a.prefill_ops
            << " decode_ops=" << a.decode_ops
            << " kv_seq_len=" << a.kv_state.seq_len
            << " kv_page_size=" << a.kv_state.page_size
            << " kv_pages=" << a.kv_state.logical_pages
            << " kv_pool_used="
            << (cb_kv_pool_ ? cb_kv_pool_->used_pages() : 0)
            << " kv_pool_free="
            << (cb_kv_pool_ ? cb_kv_pool_->free_pages() : 0)
            << " batch_steps=" << cb_decode_batches_.load()
            << " batch_tokens=" << cb_decode_tokens_.load()
            << " max_batch=" << cb_decode_max_batch_.load();
        log(msg.str());
        if (QwenExecutor::kvmem_timing_enabled()) {
            std::string tag = "phase=decode request=" + std::to_string(a.req->id);
            QwenExecutor::kvmem_timing_emit_delta(tag.c_str(),
                                                  a.kvmem_timing_baseline);
        }
        complete_continuous_request(a.req, std::move(a.req->generated));
    }

    // ---- Thinking-token budget helpers ------------------------------------
    // Force the model out of a long <think> block once it has spent its token
    // budget. We count tokens emitted while a <think> block is open; when the
    // count reaches the budget we stop sampling and instead feed a short
    // queued guidance line followed by the </think> close tag. The model then
    // resumes free generation for its actual answer. This mirrors the in-loop
    // injection that vLLM/SGLang use for reasoning budgets.

    // Guidance line injected right before the forced </think>, matching the
    // phrasing Qwen's own budget reference uses so the model transitions
    // cleanly from reasoning to answer.
    static const char *thinking_budget_guidance() {
        return "\n\nConsidering the limited time by the user, I have to give the "
               "solution based on the thinking directly now.\n";
    }

    void budget_init(ThinkingBudgetState &state,
                     const GenerationOptions &options) const {
        state = ThinkingBudgetState{};
        state.budget = options.thinking_budget;
        state.open = options.thinking_open;
        if (state.budget > 0 && tokenizer_) {
            state.close_id = tokenizer_->token_id("</think>");
        }
    }

    // Build the forced-token sequence (guidance text + </think>) to inject when
    // the budget is hit. Returns empty when the budget is inactive.
    std::vector<uint32_t> budget_close_tokens(const ThinkingBudgetState &state) const {
        std::vector<uint32_t> out;
        if (!state.active() || !tokenizer_) return out;
        const std::vector<int32_t> guide =
            tokenizer_->encode(thinking_budget_guidance());
        out.reserve(guide.size() + 1);
        for (int32_t id : guide) {
            if (id >= 0 && id != state.close_id) out.push_back(static_cast<uint32_t>(id));
        }
        out.push_back(static_cast<uint32_t>(state.close_id));
        return out;
    }

    // Account for a token the model just committed. Detects a natural </think>
    // close (stops counting) and otherwise advances the in-think counter.
    void budget_observe(ThinkingBudgetState &state, uint32_t token) const {
        if (!state.active() || !state.open) return;
        if (static_cast<int>(token) == state.close_id) {
            state.open = false;
            state.forced = false;
            state.forced_queue.clear();
            return;
        }
        ++state.think_tokens;
    }

    // If the budget is exhausted while still inside <think>, enqueue the forced
    // guidance+close tokens. Returns the next token to FEED in place of the
    // model's own pick: the front of the forced queue, or `proposed` when no
    // override is active. The caller emits the returned token and feeds it back
    // into the model so the KV cache stays consistent.
    uint32_t budget_next_feed(ThinkingBudgetState &state, uint32_t proposed) const {
        if (!state.active()) return proposed;
        if (state.forced_queue.empty() && state.open && !state.forced &&
            state.think_tokens >= state.budget) {
            const std::vector<uint32_t> close = budget_close_tokens(state);
            for (uint32_t t : close) state.forced_queue.push_back(t);
            state.forced = !state.forced_queue.empty();
        }
        if (!state.forced_queue.empty()) {
            const uint32_t forced = state.forced_queue.front();
            state.forced_queue.pop_front();
            return forced;
        }
        return proposed;
    }

    // Single entry point for the non-speculative paths: override the proposed
    // token with the budget's forced token (if any), then account for the
    // chosen token. Returns the token to emit + feed.
    uint32_t budget_apply(ThinkingBudgetState &state, uint32_t proposed) const {
        const uint32_t chosen = budget_next_feed(state, proposed);
        budget_observe(state, chosen);
        return chosen;
    }

    // True when an open <think> block has spent its budget and the forced
    // close sequence has not yet been queued/drained. Used by speculative
    // paths to break out to a plain forced-feed loop at a round boundary.
    bool budget_should_force(const ThinkingBudgetState &state) const {
        return state.active() && state.open && !state.forced &&
               state.forced_queue.empty() && state.think_tokens >= state.budget;
    }

    // qw3's original non-MTP generate path. Unchanged behavior: internal
    // chunking + graph capture live inside the executor.
    std::string generate_plain(const std::vector<uint32_t> &prompt_tokens,
                               const GenerationOptions &options,
                               const TokenCallback &on_text,
                               DumpStream *dump) {
        DeviceStatus st = device_->begin();
        if (!st.ok) throw std::runtime_error(st.message);
        executor_->reset_state();

        // Query-conditioned KVMem (#82): mark the question token span BEFORE
        // prefill (mirrors generate_mtp). Inert unless kvmem is on and the span
        // is non-empty -> single-token / recency path unchanged.
        if (executor_->kvmem_enabled() &&
            options.kvmem_query_end > options.kvmem_query_begin) {
            executor_->kvmem_set_query_span(options.kvmem_query_begin,
                                            options.kvmem_query_end);
            std::ostringstream qmsg;
            qmsg << "native kvmem query-conditioned: span=["
                 << options.kvmem_query_begin << "," << options.kvmem_query_end
                 << ") tokens=" << (options.kvmem_query_end - options.kvmem_query_begin);
            log(qmsg.str());
        }

        const double t_prefill_start = wall_seconds();
        uint64_t prefill_ops = 0;
        NativeExecutorReport step;
        if (dump) {
            for (size_t pi = 0; pi < prompt_tokens.size(); ++pi) {
                step = executor_->forward_one_token(prompt_tokens[pi]);
                if (!step.ok) throw std::runtime_error("prefill failed");
                prefill_ops += step.ops_executed;
                dump->record(static_cast<int>(pi), "prefill",
                             static_cast<int32_t>(prompt_tokens[pi]),
                             *executor_, *tokenizer_);
            }
        } else {
            step = executor_->forward_n_tokens(prompt_tokens);
            if (!step.ok) throw std::runtime_error("prefill failed");
            prefill_ops = step.ops_executed;
        }
        const double t_prefill_end = wall_seconds();

        // Block-sparse: register the prefilled prompt as context blocks and
        // assemble the first working set. Under the default all-fit budget this
        // selects every block (identity), so the decode path stays byte-
        // identical to plain; once the context exceeds the budget the built-in
        // top-k starts dropping cold blocks.
        const bool bs_on = executor_->kvmem_enabled();
        int bs_steps_since_reselect = 0;
        const int bs_interval =
            std::max(1, options_.kvmem_interval);
        if (bs_on) {
            executor_->kvmem_register_append(
                static_cast<uint32_t>(prompt_tokens.size()));
            executor_->kvmem_reselect();
        }

        // Sampling setup. temp<=0 keeps the greedy argmax path (bit-identical
        // to before); temp>0 draws from copy_last_logits() via sample_token().
        const bool do_sample = options.temperature > 0.0f;
        std::mt19937_64 rng(options.seed);
        std::vector<float> logit_buf;
        std::unordered_map<uint32_t, uint32_t> seen_tokens;
        seen_tokens.reserve(prompt_tokens.size() + static_cast<size_t>(options.max_tokens));
        for (uint32_t token : prompt_tokens) ++seen_tokens[token];
        auto pick_next = [&](int32_t fallback_argmax) -> int32_t {
            const bool need_logits =
                do_sample ||
                options.presence_penalty != 0.0f ||
                (options.repetition_penalty > 0.0f &&
                 options.repetition_penalty != 1.0f);
            if (!need_logits) return fallback_argmax;
            if (!executor_->copy_last_logits(logit_buf)) return fallback_argmax;
            apply_token_penalties(logit_buf, seen_tokens,
                                  options.presence_penalty,
                                  options.repetition_penalty);
            if (!do_sample) {
                int best = 0;
                float bv = logit_buf.empty() ? -std::numeric_limits<float>::infinity()
                                             : logit_buf[0];
                for (int i = 1; i < static_cast<int>(logit_buf.size()); ++i) {
                    if (logit_buf[i] > bv) { bv = logit_buf[i]; best = i; }
                }
                return logit_buf.empty() ? fallback_argmax : best;
            }
            const int32_t t = sample_token(logit_buf, options.temperature,
                                           options.top_p, options.top_k,
                                           options.min_p, rng);
            return t >= 0 ? t : fallback_argmax;
        };

        std::string generated;
        const int32_t eos = tokenizer_->eos_id();
        ThinkingBudgetState budget;
        budget_init(budget, options);
        const int32_t seed_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
        uint32_t next_token = static_cast<uint32_t>(pick_next(seed_argmax));
        next_token = budget_apply(budget, next_token);
        uint64_t decode_ops = 0;
        int decoded = 0;
        const auto should_stop_eos = [&]() {
            return !options.ignore_eos && next_token == static_cast<uint32_t>(eos);
        };
        if (options.max_tokens > 0 && !should_stop_eos()) {
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(next_token));
            generated += piece;
            if (on_text) on_text(piece);
            ++seen_tokens[next_token];
            ++decoded;
        }
        for (int i = 0; i + 1 < options.max_tokens; ++i) {
            if (should_stop_eos()) break;
            const uint32_t feed = next_token;
            step = executor_->forward_one_token(feed);
            if (!step.ok) throw std::runtime_error("decode failed");
            decode_ops += step.ops_executed;
            // Block-sparse: the token just decoded grew the context by one.
            // Register it, and reselect the working set on the interval
            // boundary (the agent-step cadence is approximated here by a fixed
            // step interval; reselection re-bakes any moved block in place).
            if (bs_on) {
                executor_->kvmem_register_append(1);
                if (options_.kvmem_update_mode != "step" &&
                    ++bs_steps_since_reselect >= bs_interval) {
                    executor_->kvmem_reselect();
                    bs_steps_since_reselect = 0;
                }
            }
            const int32_t fallback = step.argmax_token >= 0 ? step.argmax_token : eos;
            if (dump) dump->record(static_cast<int>(prompt_tokens.size() + i),
                                   "decode", static_cast<int32_t>(feed),
                                   *executor_, *tokenizer_);
            const int32_t new_token = pick_next(fallback);
            next_token = budget_apply(budget, static_cast<uint32_t>(new_token));
            if (should_stop_eos()) break;
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(next_token));
            generated += piece;
            if (on_text) on_text(piece);
            ++seen_tokens[next_token];
            ++decoded;
        }
        const double t_decode_end = wall_seconds();

        if (decoded == 0 && options.max_tokens > 0) {
            log_zero_decode_diagnostic("plain", prompt_tokens, step);
        }

        st = device_->end();
        if (!st.ok) throw std::runtime_error(st.message);

        const double prefill_s = std::max(t_prefill_end - t_prefill_start, 1e-9);
        const double decode_s = std::max(t_decode_end - t_prefill_end, 1e-9);
        std::ostringstream msg;
        msg << "native generate:"
            << " prompt_tokens=" << prompt_tokens.size()
            << " prefill=" << fmt_seconds(prefill_s);
        if (!prompt_tokens.empty()) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (prompt_tokens.size() / prefill_s) << " tok/s)";
        }
        msg << " decoded=" << decoded
            << " decode=" << fmt_seconds(decode_s);
        if (decoded > 0) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (decoded / decode_s) << " tok/s)";
        }
        msg << " prefill_ops=" << prefill_ops << " decode_ops=" << decode_ops;
        log(msg.str());

        return generated;
    }

    // Per-call wall-clock + acceptance, threaded out for the kvmem-session
    // growth harness (run_kvmem_session). Filled just before return when a
    // non-null stats_out is passed; default callers ignore it.
    struct MtpGenStats {
        double prefill_s = 0.0;
        double decode_s = 0.0;   // PURE decode loop (post-prefill reselect excluded)
        double reselect_s = 0.0; // post-prefill (decode-window) reselect wall clock
        int decoded = 0;
        uint64_t prompt_tokens = 0;
        double acceptance = 0.0;
        // kvmem timing snapshot captured at the prefill->reselect boundary, so
        // the session harness can isolate the post-prefill (decode-window)
        // reselect breakdown from mid-prefill offload churn (which is folded
        // into prefill_s). Valid only when kvmem timing is enabled.
        QwenExecutor::KvMemTimingSnapshot kvmem_at_boundary;
        bool kvmem_boundary_valid = false;
    };

    // MTP draft/verify/speculate + adaptive-depth path. Ported from qw3_ly,
    // adapted to qw3's executor signatures + memory-safe prefix chunking.
    //
    // reset_session: when true (default) the executor state is wiped at entry,
    // which is the standalone single-request behavior every existing caller
    // relies on. The kvmem-session harness passes false to keep the executor's
    // KV cache + kvmem block store alive across turns so context grows
    // incrementally (the new turn's chunk is appended at the running position).
    std::string generate_mtp(std::vector<uint32_t> &prompt_tokens,
                             const GenerationOptions &options,
                             const TokenCallback &on_text,
                             DumpStream *dump,
                             bool spec_mtp,
                             bool trace_mtp,
                             QwenExecutor *override_executor = nullptr,
                             bool manage_device_scope = true,
                             bool reset_session = true,
                             MtpGenStats *stats_out = nullptr) {
        QwenExecutor *executor_ =
            override_executor != nullptr ? override_executor : this->executor_.get();
        if (!executor_) throw std::runtime_error("MTP executor unavailable");
        DeviceStatus st;
        if (manage_device_scope) {
            st = device_->begin();
            if (!st.ok) throw std::runtime_error(st.message);
        }
        if (reset_session) {
            executor_->reset_state();
        }

        // kvmem × MTP (Phase C). When kvmem is enabled the verify path must
        // attend over the assembled window, which only the per-token
        // forward_one_token path honors — so MTP verify is forced sequential
        // here (each verify token re-enters the window-aware forward_one_token
        // and appends K/V at the window tail). Rejection rollback uses the full
        // state snapshot + sequential replay, both window-aware; the window
        // page-table tail is truncated back by restore_state. The draft head
        // runs over its own MTP prefix KV at true positions (not the window) —
        // under the default identity (all-fit) budget the window equals the
        // true positions so this is byte-identical to plain MTP; under a sparse
        // budget the drafts are merely lower-acceptance guesses while the
        // window-aware verify stays authoritative. The kvmem cadence
        // (register_append per committed token + interval reselect) is driven
        // off position deltas in the loops below.
        const bool kvmem_on = executor_->kvmem_enabled();
        // Query-conditioned KVMem (#82): mark the question token span BEFORE
        // prefill so the executor captures the in-span Q rows during prefill and
        // ranks blocks by the multi-token mean at the boundary. Inert (single-token
        // / recency path unchanged) unless kvmem is on and the span is non-empty.
        if (kvmem_on && options.kvmem_query_end > options.kvmem_query_begin) {
            executor_->kvmem_set_query_span(options.kvmem_query_begin,
                                            options.kvmem_query_end);
            std::ostringstream qmsg;
            qmsg << "native kvmem query-conditioned: span=["
                 << options.kvmem_query_begin << "," << options.kvmem_query_end
                 << ") tokens=" << (options.kvmem_query_end - options.kvmem_query_begin);
            log(qmsg.str());
        }
        const int kvmem_interval = std::max(1, options_.kvmem_interval);
        uint32_t kvmem_last_reselect_pos = 0;
        // Register newly-committed tokens with the block store and reselect the
        // working set on the interval boundary. `committed_pos` is the
        // executor's post-commit position(); we register the delta since the
        // last call so accepted-chain commits (which advance position by >1)
        // register the whole jump in one shot.
        uint32_t kvmem_registered_pos = 0;
        auto kvmem_advance_to = [&](uint32_t committed_pos,
                                    bool defer_finish = false) -> bool {
            if (!kvmem_on) return false;
            if (committed_pos > kvmem_registered_pos) {
                executor_->kvmem_register_append(committed_pos -
                                                 kvmem_registered_pos);
                kvmem_registered_pos = committed_pos;
            }
            if (options_.kvmem_update_mode != "step" &&
                committed_pos >= kvmem_last_reselect_pos +
                                     static_cast<uint32_t>(kvmem_interval)) {
                if (defer_finish) {
                    executor_->kvmem_prepare_reselect();
                } else {
                    executor_->kvmem_reselect();
                }
                kvmem_last_reselect_pos = committed_pos;
                return defer_finish;
            }
            return false;
        };
        const uint32_t requested_mtp_chain_len = mtp_trace_chain_len(options_);
        const uint32_t safe_mtp_chain_max = spec_mtp
            ? mtp_safe_chain_max()
            : std::numeric_limits<uint32_t>::max();
        const uint32_t mtp_chain_len =
            std::min<uint32_t>(requested_mtp_chain_len, safe_mtp_chain_max);
        const uint32_t mtp_reject_limit = spec_mtp ? mtp_reject_budget(prompt_tokens.size()) : 0;
        bool use_mtp_prefix = spec_mtp || mtp_prefix_enabled(options_);
        if (spec_mtp && mtp_chain_len != requested_mtp_chain_len) {
            std::ostringstream chain_msg;
            chain_msg << "native mtp_spec_config:"
                      << " requested_chain=" << requested_mtp_chain_len
                      << " effective_chain=" << mtp_chain_len
                      << " safe_max=" << safe_mtp_chain_max
                      << " reason=\"QW3_MTP_SAFE_MAX_CHAIN correctness guard\"";
            log(chain_msg.str());
        }
        const uint32_t mtp_prefix_max_prompt = mtp_prefix_max_prompt_tokens();
        if (use_mtp_prefix &&
            prompt_tokens.size() > static_cast<size_t>(mtp_prefix_max_prompt)) {
            log("native mtp_prefix: ok=false reason=\"prompt exceeds QW3_MTP_PREFIX_MAX_PROMPT\"");
            use_mtp_prefix = false;
        }

        // MTP prefix priming reuses the last batch's hidden rows out of the
        // executor's h_batch_ scratch, which only retains the final internal
        // chunk. Drive chunk boundaries here at a memory-safe width and pin the
        // executor so each backend chunk is processed as a single batch; if a
        // chunk is ever re-split internally, prime_mtp_prefix_from_last_batch
        // returns not-ok and we disable the prefix path rather than corrupt it.
        constexpr uint32_t kMtpPrefillChunk = 4096;
        if (use_mtp_prefix) {
            executor_->set_prefill_chunk_override(static_cast<int>(kMtpPrefillChunk));
        }

        uint32_t mtp_prefix_tokens = 0;
        uint64_t mtp_prefix_ops = 0;
        auto prime_mtp_prefix = [&](const std::vector<uint32_t> &tokens,
                                    uint32_t base_position) {
            if (!use_mtp_prefix) return;
            NativeExecutorReport mtp = executor_->prime_mtp_prefix_from_last_batch(tokens,
                                                                                   base_position);
            if (!mtp.ok) {
                std::ostringstream msg;
                msg << "native mtp_prefix: ok=false"
                    << " base_position=" << base_position
                    << " tokens=" << tokens.size();
                if (!mtp.missing_kernels.empty()) {
                    msg << " reason=" << escape_text(mtp.missing_kernels.front());
                }
                log(msg.str());
                use_mtp_prefix = false;
                return;
            }
            mtp_prefix_tokens += static_cast<uint32_t>(tokens.size());
            mtp_prefix_ops += mtp.ops_executed;
        };

        const double t_prefill_start = wall_seconds();
        // Absolute base position for this prefill. Zero on a fresh session
        // (reset_session=true), but on a resumed kvmem-session turn the prompt
        // chunk appends at the running position, so the MTP draft prefix must
        // be primed at the absolute base (not the call-relative chunk offset).
        const uint32_t prefill_base = executor_->position();
        QwenExecutor::KvMemTimingSnapshot kvmem_tbase;
        if (kvmem_on && QwenExecutor::kvmem_timing_enabled()) {
            kvmem_tbase = QwenExecutor::kvmem_timing_snapshot();
        }
        uint64_t prefill_ops = 0;
        NativeExecutorReport step;
        uint32_t prefill_chunks = 0;
        uint32_t prefill_chunk_size = 0;
        const bool trace_prefill = prefill_trace_enabled();
        if (dump) {
            if (use_mtp_prefix) {
                log("native mtp_prefix: ok=false reason=\"dump logits path does not expose batch hidden rows\"");
                use_mtp_prefix = false;
            }
            for (size_t pi = 0; pi < prompt_tokens.size(); ++pi) {
                step = executor_->forward_one_token(prompt_tokens[pi]);
                if (!step.ok) throw std::runtime_error("prefill failed");
                prefill_ops += step.ops_executed;
                dump->record(static_cast<int>(pi), "prefill",
                             static_cast<int32_t>(prompt_tokens[pi]),
                             *executor_, *tokenizer_);
            }
        } else if (!use_mtp_prefix) {
            // No prefix priming needed: defer to qw3's internal-chunking prefill.
            const double t_chunk_start = wall_seconds();
            step = executor_->forward_n_tokens(prompt_tokens);
            if (!step.ok) throw std::runtime_error("prefill failed");
            const double t_chunk_end = wall_seconds();
            prefill_ops = step.ops_executed;
            prefill_chunks = 1;
            if (trace_prefill) {
                log_prefill_chunk(0, 0, prompt_tokens.size(), t_chunk_end - t_chunk_start);
            }
        } else {
            for (size_t offset = 0; offset < prompt_tokens.size(); offset += kMtpPrefillChunk) {
                const size_t end = std::min(prompt_tokens.size(),
                                            offset + static_cast<size_t>(kMtpPrefillChunk));
                std::vector<uint32_t> chunk(prompt_tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                                            prompt_tokens.begin() + static_cast<std::ptrdiff_t>(end));
                const double t_chunk_start = wall_seconds();
                const bool need_logits = end == prompt_tokens.size();
                step = executor_->forward_n_tokens(chunk, need_logits);
                if (!step.ok) throw std::runtime_error("prefill failed");
                prime_mtp_prefix(chunk, prefill_base + static_cast<uint32_t>(offset));
                const double t_chunk_end = wall_seconds();
                prefill_ops += step.ops_executed;
                if (trace_prefill) {
                    log_prefill_chunk(prefill_chunks, offset, chunk.size(),
                                      t_chunk_end - t_chunk_start);
                }
                ++prefill_chunks;
            }
            prefill_chunk_size = kMtpPrefillChunk;
        }
        const double t_prefill_end = wall_seconds();

        // Snapshot kvmem timing at the prefill->reselect boundary. The reselect
        // below is the post-prefill (decode-window) selection the session
        // harness charges as steps 1-4; any mid-prefill offload churn already
        // landed before this point and stays inside prefill_s.
        QwenExecutor::KvMemTimingSnapshot kvmem_at_prefill_end;
        bool kvmem_boundary_valid = false;
        if (kvmem_on && QwenExecutor::kvmem_timing_enabled()) {
            kvmem_at_prefill_end = QwenExecutor::kvmem_timing_snapshot();
            kvmem_boundary_valid = true;
        }

        // kvmem: register the prefilled prompt as context blocks and assemble
        // the first working set (mirrors generate_plain). Under the default
        // all-fit budget this is an identity selection so the window equals the
        // true cache and MTP verify stays byte-identical to plain MTP.
        if (kvmem_on) {
            executor_->kvmem_register_append(
                static_cast<uint32_t>(prompt_tokens.size()));
            executor_->kvmem_reselect();
            kvmem_registered_pos = executor_->position();
            kvmem_last_reselect_pos = executor_->position();
        }
        if (kvmem_on && QwenExecutor::kvmem_timing_enabled()) {
            QwenExecutor::kvmem_timing_emit_delta("phase=prefill request=mtp",
                                                  kvmem_tbase);
            kvmem_tbase = QwenExecutor::kvmem_timing_snapshot();
        }
        // Boundary between the post-prefill reselect and the decode loop. For
        // plain MTP (kvmem off) the reselect block above is skipped, so this
        // equals t_prefill_end and decode_s stays byte-identical to before.
        const double t_reselect_end = wall_seconds();

        std::string generated;
        const int32_t eos = tokenizer_->eos_id();
        ThinkingBudgetState budget;
        budget_init(budget, options);

        // Sampling setup for distribution-lossless MTP. temp<=0 with no penalties
        // keeps the exact greedy argmax path (byte-identical to pre-change). When
        // temp>0 we run the point-mass speculative-sampling accept test over the
        // target rows transported to host, reproducing the non-MTP sampler's
        // distribution; mtp_seen tracks committed tokens for presence/repetition
        // penalties, and mtp_rng is seeded per-request like generate_plain.
        const bool mtp_do_sample = options.temperature > 0.0f;
        const bool mtp_use_penalties =
            options.presence_penalty != 0.0f ||
            (options.repetition_penalty > 0.0f &&
             options.repetition_penalty != 1.0f);
        const bool mtp_need_logits_pick = mtp_do_sample || mtp_use_penalties;
        std::mt19937_64 mtp_rng(options.seed);
        std::vector<float> mtp_logit_buf;
        std::unordered_map<uint32_t, uint32_t> mtp_seen;
        if (mtp_need_logits_pick) {
            mtp_seen.reserve(prompt_tokens.size() +
                             static_cast<size_t>(options.max_tokens));
            for (uint32_t token : prompt_tokens) ++mtp_seen[token];
        }
        // Pick the next token from the executor's last logits row, applying
        // penalties + sampling when active, else returning the device argmax
        // fallback. Used for the prefill seed, the drafts-empty fallback, and the
        // plain-decode tail so a sampling request samples on every non-spec path.
        auto pick_from_last_logits = [&](int32_t fallback_argmax) -> uint32_t {
            if (!mtp_need_logits_pick) return static_cast<uint32_t>(fallback_argmax);
            if (!executor_->copy_last_logits(mtp_logit_buf))
                return static_cast<uint32_t>(fallback_argmax);
            apply_token_penalties(mtp_logit_buf, mtp_seen,
                                  options.presence_penalty,
                                  options.repetition_penalty);
            int32_t tok;
            if (mtp_do_sample) {
                tok = sample_token(mtp_logit_buf, options.temperature,
                                   options.top_p, options.top_k,
                                   options.min_p, mtp_rng);
            } else {
                int best = 0;
                float bv = mtp_logit_buf.empty()
                    ? -std::numeric_limits<float>::infinity()
                    : mtp_logit_buf[0];
                for (int i = 1; i < static_cast<int>(mtp_logit_buf.size()); ++i)
                    if (mtp_logit_buf[i] > bv) { bv = mtp_logit_buf[i]; best = i; }
                tok = mtp_logit_buf.empty() ? fallback_argmax : best;
            }
            return tok >= 0 ? static_cast<uint32_t>(tok)
                            : static_cast<uint32_t>(fallback_argmax);
        };

        uint32_t next_token = pick_from_last_logits(
            step.argmax_token >= 0 ? step.argmax_token : eos);
        uint64_t decode_ops = 0;
        std::unordered_map<std::string, TraceStats> decode_trace;
        uint64_t decode_trace_steps = 0;
        const bool trace_mtp_verify = mtp_verify_trace_enabled();
        const bool use_device_mtp_draft_chain =
            spec_mtp && mtp_device_draft_chain_enabled() && !trace_mtp_verify;
        const bool trace_mtp_tokens = mtp_token_trace_enabled();
        std::unordered_map<std::string, TraceStats> mtp_draft_trace;
        std::unordered_map<std::string, TraceStats> mtp_verify_trace;
        std::unordered_map<std::string, TraceStats> mtp_replay_trace;
        uint64_t mtp_draft_trace_steps = 0;
        uint64_t mtp_verify_trace_steps = 0;
        uint64_t mtp_replay_trace_steps = 0;
        const bool sync_mtp_phase_timing = mtp_phase_sync_enabled();
        auto mtp_phase_time = [&]() -> double {
            if (sync_mtp_phase_timing) {
                DeviceStatus sync_st = device_->synchronize();
                if (!sync_st.ok) throw std::runtime_error(sync_st.message);
            }
            return wall_seconds();
        };
        int decoded = 0;
        struct PendingMtpChain {
            int start_index = 0;
            int input_token = -1;
            std::vector<int> drafts;
        };
        std::vector<PendingMtpChain> mtp_pending_chains;
        std::vector<uint64_t> mtp_chain_verified(mtp_chain_len, 0);
        std::vector<uint64_t> mtp_chain_accepted(mtp_chain_len, 0);
        uint64_t mtp_chains = 0;
        uint64_t mtp_drafts = 0;
        uint64_t mtp_ops = 0;
        uint64_t mtp_spec_batches = 0;
        uint64_t mtp_spec_drafted = 0;
        uint64_t mtp_spec_accepted = 0;
        uint64_t mtp_spec_rejected = 0;
        uint64_t mtp_spec_rollbacks = 0;
        uint64_t mtp_spec_prefix1_reused = 0;
        uint64_t mtp_spec_state_checkpoint_reused = 0;
        uint64_t mtp_spec_batched_verify_batches = 0;
        uint64_t mtp_spec_batched_verify_tokens = 0;
        std::vector<uint64_t> mtp_accept_len_hist(mtp_chain_len + 1, 0);
        double mtp_spec_draft_s = 0.0;
        double mtp_spec_snapshot_s = 0.0;
        double mtp_spec_verify_s = 0.0;
        double mtp_spec_restore_s = 0.0;
        double mtp_spec_replay_s = 0.0;
        double mtp_spec_prefix_s = 0.0;
        double mtp_spec_plain_s = 0.0;
        bool mtp_spec_fallback = false;
        QwenExecutor::StateSnapshot mtp_spec_snapshot;
        QwenExecutor::StateCheckpointSet mtp_spec_state_checkpoints;
        const bool use_single_token_replay = mtp_single_token_replay_enabled();
        const bool reuse_current_mtp_prefix = mtp_reuse_current_prefix_enabled();
        // kvmem (D1/D1.1): the BATCHED verifier (forward_n_tokens) is window-aware
        // — it appends + attends in the assembled window frame (window_query_pos_
        // base + window page table) — and restore_state_checkpoint() now also
        // rolls the window tail back per accepted row (StateCheckpointSet carries
        // the window base). So kvmem uses the SAME fast batched verify + free
        // per-row checkpoint rollback as plain MTP; no path is disabled. The
        // draft head still runs over its own MTP prefix KV at true positions;
        // verify over the window stays authoritative, so a draft/window mismatch
        // only lowers acceptance, never correctness.
        const bool use_sequential_verifier = mtp_verify_sequential_enabled();
        const MtpTransactionalReplayMode transactional_replay_mode =
            !use_sequential_verifier
                ? mtp_transactional_replay_mode()
                : MtpTransactionalReplayMode::Off;
        const bool use_transactional_replay =
            transactional_replay_mode != MtpTransactionalReplayMode::Off;
        const bool use_transactional_replay_all =
            transactional_replay_mode == MtpTransactionalReplayMode::All;
        const uint32_t state_checkpoint_count =
            mtp_state_checkpoint_count(mtp_chain_len);

        auto rebuild_accepted_mtp_prefix = [&](const std::vector<uint32_t> &tokens,
                                               uint32_t base_position,
                                               bool finish_deferred_kvmem = false) {
            if (!mtp_rebuild_accepted_prefix_enabled()) {
                executor_->commit_mtp_prefix(executor_->position());
                if (finish_deferred_kvmem) executor_->kvmem_finish_reselect();
                return;
            }
            const double t_prefix_start = mtp_phase_time();
            NativeExecutorReport prefix =
                executor_->prime_mtp_prefix_from_last_batch(
                    tokens, base_position, mtp_prefix_rebuild_batch_min_tokens());
            mtp_spec_prefix_s += mtp_phase_time() - t_prefix_start;
            if (!prefix.ok) {
                std::string reason = prefix.missing_kernels.empty()
                    ? "unknown"
                    : prefix.missing_kernels.front();
                throw std::runtime_error("MTP accepted-prefix rebuild failed: " + reason);
            }
            mtp_ops += prefix.ops_executed;
            if (finish_deferred_kvmem) executor_->kvmem_finish_reselect();
        };
        auto rebuild_current_mtp_prefix = [&](uint32_t token, uint32_t base_position,
                                              bool finish_deferred_kvmem = false) {
            if (!mtp_rebuild_accepted_prefix_enabled()) {
                executor_->commit_mtp_prefix(executor_->position());
                if (finish_deferred_kvmem) executor_->kvmem_finish_reselect();
                return;
            }
            const double t_prefix_start = mtp_phase_time();
            NativeExecutorReport prefix =
                executor_->prime_mtp_prefix_from_current(token, base_position);
            mtp_spec_prefix_s += mtp_phase_time() - t_prefix_start;
            if (!prefix.ok) {
                std::string reason = prefix.missing_kernels.empty()
                    ? "unknown"
                    : prefix.missing_kernels.front();
                throw std::runtime_error("MTP current-prefix rebuild failed: " + reason);
            }
            mtp_ops += prefix.ops_executed;
            if (finish_deferred_kvmem) executor_->kvmem_finish_reselect();
        };
        auto replay_target_tokens_sequential = [&](const std::vector<uint32_t> &tokens,
                                                   uint32_t base_position,
                                                   bool rebuild_prefix) {
            double prefix_seconds = 0.0;
            uint64_t prefix_ops = 0;
            NativeExecutorReport replay_report =
                executor_->replay_tokens_with_mtp_prefix(tokens, base_position,
                                                         rebuild_prefix,
                                                         &prefix_seconds,
                                                         &prefix_ops);
            mtp_spec_prefix_s += prefix_seconds;
            mtp_ops += prefix_ops;
            return replay_report;
        };

        auto trace_mtp_chain = [&](uint32_t input_token, int target_index) {
            std::vector<NativeExecutorReport> chain = use_mtp_prefix
                ? executor_->forward_mtp_draft_chain_with_prefix(input_token, mtp_chain_len)
                : executor_->forward_mtp_draft_chain(input_token, mtp_chain_len);
            PendingMtpChain pending;
            pending.start_index = target_index;
            pending.input_token = static_cast<int>(input_token);
            uint32_t step_index = 0;
            for (const NativeExecutorReport &mtp : chain) {
                std::ostringstream mtp_msg;
                mtp_msg << "native mtp_draft:"
                        << " start_index=" << target_index
                        << " step=" << (step_index + 1)
                        << " input_token=" << input_token
                        << " prefix=" << (use_mtp_prefix ? "true" : "false")
                        << " ok=" << (mtp.ok ? "true" : "false");
                if (mtp.ok) {
                    mtp_msg << " draft_token=" << mtp.argmax_token
                            << " draft_logit=" << std::fixed << std::setprecision(4)
                            << mtp.argmax_logit
                            << " draft_text="
                            << escape_text(tokenizer_->decode_one(mtp.argmax_token))
                            << " ops=" << mtp.ops_executed;
                    pending.drafts.push_back(mtp.argmax_token);
                    input_token = static_cast<uint32_t>(mtp.argmax_token);
                    ++mtp_drafts;
                    mtp_ops += mtp.ops_executed;
                } else if (!mtp.missing_kernels.empty()) {
                    mtp_msg << " reason=" << escape_text(mtp.missing_kernels.front());
                }
                log(mtp_msg.str());
                if (!mtp.ok) break;
                ++step_index;
            }
            if (!pending.drafts.empty()) {
                ++mtp_chains;
                mtp_pending_chains.push_back(std::move(pending));
            }
        };

        auto verify_mtp_chains = [&](int target_index, int32_t target_token) {
            std::vector<PendingMtpChain> still_pending;
            still_pending.reserve(mtp_pending_chains.size());
            for (const PendingMtpChain &pending : mtp_pending_chains) {
                const int offset = target_index - pending.start_index;
                if (offset <= 0) {
                    still_pending.push_back(pending);
                    continue;
                }
                if (offset > static_cast<int>(pending.drafts.size())) {
                    continue;
                }
                const uint32_t slot = static_cast<uint32_t>(offset - 1);
                const int draft_token = pending.drafts[slot];
                const bool accepted = draft_token == target_token;
                ++mtp_chain_verified[slot];
                if (accepted) ++mtp_chain_accepted[slot];
                std::ostringstream verify_msg;
                verify_msg << "native mtp_verify:"
                           << " start_index=" << pending.start_index
                           << " step=" << offset
                           << " input_token=" << pending.input_token
                           << " draft_token=" << draft_token
                           << " target_token=" << target_token
                           << " accepted=" << (accepted ? "true" : "false")
                           << " target_text=" << escape_text(tokenizer_->decode_one(target_token));
                log(verify_msg.str());
                if (accepted && offset < static_cast<int>(pending.drafts.size())) {
                    still_pending.push_back(pending);
                }
            }
            mtp_pending_chains.swap(still_pending);
        };

        bool run_spec_mtp = spec_mtp && use_mtp_prefix && !dump;
        const bool mtp_spec_started = run_spec_mtp;
        MtpAdaptivePolicy mtp_policy;
        mtp_policy.configure(run_spec_mtp, mtp_chain_len, prompt_tokens.size());
        if (run_spec_mtp && mtp_policy.enabled) {
            std::ostringstream policy_msg;
            policy_msg << "native mtp_policy_config:"
                       << " enabled=true"
                       << " min=" << mtp_policy.min_depth
                       << " max=" << mtp_policy.max_depth
                       << " initial=" << mtp_policy.initial_depth
                       << " update_interval=" << mtp_policy.update_interval
                       << " min_decision_batches="
                       << mtp_policy.min_decision_batches
                       << " cooldown=" << mtp_policy.cooldown_batches
                       << " demote_windows=" << mtp_policy.demote_windows
                       << " promote_windows=" << mtp_policy.promote_windows
                       << " startup_demote_batches="
                       << mtp_policy.startup_demote_batches
                       << " demote_margin=" << std::fixed << std::setprecision(4)
                       << mtp_policy.demote_margin
                       << " promote_margin=" << mtp_policy.promote_margin
                       << " trace=" << (mtp_policy.trace ? "true" : "false");
            log(policy_msg.str());
        }
        if (run_spec_mtp && !use_sequential_verifier &&
            !std::getenv("QW3_MTP_VERIFY") &&
            native_decode_flashinfer_effective()) {
            log("native mtp_verify_config: mode=batched attention=batch_decode reason=\"FlashInfer BatchDecode verifier is the default small-batch path\" override=\"QW3_MTP_VERIFY=sequential\"");
        }
        if (run_spec_mtp && use_transactional_replay &&
            native_decode_flashinfer_effective()) {
            log(std::string("native mtp_transactional_replay: enabled=true mode=") +
                (use_transactional_replay_all ? "all" : "rejects") +
                " reason=\"commit verifier tokens through stable single-token state path\" override=\"QW3_MTP_TRANSACTIONAL_REPLAY=0\"");
        }
        if (spec_mtp && dump) {
            log("native mtp_speculate: ok=false reason=\"dump logits path is not supported\"");
        } else if (spec_mtp && !use_mtp_prefix) {
            log("native mtp_speculate: ok=false reason=\"MTP prefix cache is unavailable\"");
        }

        auto should_stop_mtp_eos = [&](uint32_t token) -> bool {
            return !options.ignore_eos && token == static_cast<uint32_t>(eos);
        };
        auto emit_generated_token = [&](uint32_t token) -> bool {
            if (decoded >= options.max_tokens || should_stop_mtp_eos(token)) return false;
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(token));
            generated += piece;
            if (on_text) on_text(piece);
            budget_observe(budget, token);
            // Track committed tokens for the next accept test's penalties (no-op
            // for greedy without penalties, keeping that path byte-identical).
            if (mtp_need_logits_pick) ++mtp_seen[token];
            ++decoded;
            return true;
        };

        if (options.max_tokens > 0 && !should_stop_mtp_eos(next_token)) {
            if (emit_generated_token(next_token) && trace_mtp && !run_spec_mtp) {
                trace_mtp_chain(next_token, decoded - 1);
            }
        }

        uint64_t plain_decode_forwards = 0;
        const bool decode_as_batch = decode_as_batch_enabled();
        auto run_plain_decode_remaining = [&]() {
            while (decoded < options.max_tokens && !should_stop_mtp_eos(next_token)) {
                const uint32_t feed = next_token;
                // kvmem requires the window-aware per-token path: the batched
                // forward_n_tokens attends over true positions, not the window.
                if (decode_as_batch && !kvmem_on) {
                    const std::vector<uint32_t> one_token{feed};
                    step = executor_->forward_n_tokens(one_token);
                } else {
                    step = executor_->forward_one_token(feed);
                }
                if (!step.ok) throw std::runtime_error("decode failed");
                decode_ops += step.ops_executed;
                // Sample (or greedily pick) before any kvmem reselect or draft
                // trace runs, so logits_ still holds this forward's output.
                const uint32_t sampled = pick_from_last_logits(
                    step.argmax_token >= 0 ? step.argmax_token : eos);
                kvmem_advance_to(executor_->position());
                if (decode_trace_enabled() && !step.elapsed_us.empty()) {
                    accumulate_trace(decode_trace, step);
                    ++decode_trace_steps;
                }
                const int32_t new_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
                if (trace_mtp) {
                    verify_mtp_chains(decoded, new_argmax);
                }
                if (dump) dump->record(static_cast<int>(prompt_tokens.size() + plain_decode_forwards),
                                       "decode", static_cast<int32_t>(feed),
                                       *executor_, *tokenizer_);
                ++plain_decode_forwards;
                next_token = budget_next_feed(budget, sampled);
                if (!emit_generated_token(next_token)) break;
                if (trace_mtp && decoded < options.max_tokens) {
                    trace_mtp_chain(next_token, decoded - 1);
                }
            }
        };

        if (run_spec_mtp) {
            while (run_spec_mtp &&
                   decoded < options.max_tokens &&
                   !should_stop_mtp_eos(next_token)) {
                // Once the thinking budget is exhausted, stop speculating and
                // fall through to the plain decode loop, which feeds the forced
                // guidance + </think> close tokens deterministically.
                if (budget_should_force(budget)) {
                    run_spec_mtp = false;
                    break;
                }
                const uint32_t current = next_token;
                const uint32_t remaining_tokens =
                    static_cast<uint32_t>(options.max_tokens - decoded);
                const uint32_t draft_limit =
                    mtp_policy.draft_limit(remaining_tokens, mtp_chain_len);
                const double t_draft_start = mtp_phase_time();
                std::vector<NativeExecutorReport> chain = use_device_mtp_draft_chain
                    ? executor_->forward_mtp_draft_chain_with_prefix_device(current, draft_limit)
                    : executor_->forward_mtp_draft_chain_with_prefix(current, draft_limit);
                mtp_spec_draft_s += mtp_phase_time() - t_draft_start;
                std::vector<uint32_t> drafts;
                uint32_t step_index = 0;
                for (const NativeExecutorReport &mtp : chain) {
                    if (trace_mtp_verify) {
                        accumulate_trace(mtp_draft_trace, mtp);
                        ++mtp_draft_trace_steps;
                    }
                    if (trace_mtp) {
                        std::ostringstream mtp_msg;
                        mtp_msg << "native mtp_spec_draft:"
                                << " step=" << (step_index + 1)
                                << " input_token=" << current
                                << " ok=" << (mtp.ok ? "true" : "false");
                        if (mtp.ok) {
                            mtp_msg << " draft_token=" << mtp.argmax_token
                                    << " draft_logit=" << std::fixed << std::setprecision(4)
                                    << mtp.argmax_logit
                                    << " draft_text="
                                    << escape_text(tokenizer_->decode_one(mtp.argmax_token))
                                    << " ops=" << mtp.ops_executed;
                        } else if (!mtp.missing_kernels.empty()) {
                            mtp_msg << " reason=" << escape_text(mtp.missing_kernels.front());
                        }
                        log(mtp_msg.str());
                    }
                    if (!mtp.ok || mtp.argmax_token < 0 ||
                        should_stop_mtp_eos(static_cast<uint32_t>(mtp.argmax_token))) {
                        break;
                    }
                    drafts.push_back(static_cast<uint32_t>(mtp.argmax_token));
                    ++mtp_drafts;
                    ++mtp_spec_drafted;
                    mtp_ops += mtp.ops_executed;
                    ++step_index;
                }

                if (drafts.empty()) {
                    step = executor_->forward_one_token(current);
                    if (!step.ok) throw std::runtime_error("decode failed");
                    decode_ops += step.ops_executed;
                    const int32_t new_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
                    // Sample/pick before commit_mtp_prefix so logits_ is intact.
                    next_token = pick_from_last_logits(new_argmax);
                    executor_->commit_mtp_prefix(executor_->position());
                    kvmem_advance_to(executor_->position());
                    if (!emit_generated_token(next_token)) break;
                    continue;
                }

                const bool checkpoints_cover_rejects =
                    !use_sequential_verifier &&
                    !use_transactional_replay &&
                    state_checkpoint_count >= drafts.size();
                bool captured_snapshot = false;
                uint32_t verify_base_position = executor_->position();
                if (!checkpoints_cover_rejects) {
                    const double t_snapshot_start = mtp_phase_time();
                    executor_->capture_state(mtp_spec_snapshot);
                    verify_base_position = mtp_spec_snapshot.position;
                    mtp_spec_snapshot_s += mtp_phase_time() - t_snapshot_start;
                    captured_snapshot = true;
                }
                std::vector<uint32_t> verify_tokens;
                verify_tokens.reserve(drafts.size() + 1);
                verify_tokens.push_back(current);
                verify_tokens.insert(verify_tokens.end(), drafts.begin(), drafts.end());
                std::vector<DeviceArgmax> row_argmaxes;
                // Per-row target distributions on host, populated only when the
                // request samples (temp>0). Same indexing as row_argmaxes: row i
                // gates draft i; the final row is the all-accept bonus source.
                std::vector<std::vector<float>> row_logits;
                const double t_verify_start = mtp_phase_time();
                if (use_sequential_verifier) {
                    step = NativeExecutorReport{};
                    step.ok = true;
                    row_argmaxes.reserve(verify_tokens.size());
                    if (mtp_need_logits_pick) row_logits.reserve(verify_tokens.size());
                    for (uint32_t token : verify_tokens) {
                        NativeExecutorReport verify_step = executor_->forward_one_token(token);
                        if (trace_mtp_verify) {
                            accumulate_trace(mtp_verify_trace, verify_step);
                            ++mtp_verify_trace_steps;
                        }
                        step.ops_executed += verify_step.ops_executed;
                        if (!verify_step.ok) {
                            step.ok = false;
                            step.missing_kernels = std::move(verify_step.missing_kernels);
                            break;
                        }
                        row_argmaxes.push_back(DeviceArgmax{
                            verify_step.argmax_token,
                            verify_step.argmax_logit
                        });
                        // Copy this row's logits before the next forward_one_token
                        // overwrites the executor's logits_ scratch.
                        if (mtp_need_logits_pick) {
                            row_logits.emplace_back();
                            executor_->copy_last_logits(row_logits.back());
                        }
                    }
                } else {
                    step = executor_->forward_n_tokens(
                        verify_tokens, true, &row_argmaxes,
                        state_checkpoint_count > 0 ? &mtp_spec_state_checkpoints : nullptr,
                        state_checkpoint_count,
                        /*copy_last_logits=*/!mtp_skip_verify_logits_copy_enabled(),
                        mtp_need_logits_pick ? &row_logits : nullptr);
                    ++mtp_spec_batched_verify_batches;
                    mtp_spec_batched_verify_tokens += verify_tokens.size();
                    if (trace_mtp_verify) {
                        accumulate_trace(mtp_verify_trace, step);
                        mtp_verify_trace_steps += verify_tokens.size();
                    }
                }
                mtp_spec_verify_s += mtp_phase_time() - t_verify_start;
                if (!step.ok || row_argmaxes.size() != verify_tokens.size()) {
                    throw std::runtime_error("MTP target verification failed");
                }
                decode_ops += step.ops_executed;
                ++mtp_spec_batches;

                uint32_t accepted = 0;
                int32_t target_token = eos;
                if (mtp_need_logits_pick) {
                    // Distribution-lossless point-mass speculative-sampling accept
                    // test. extra_token is the residual draw on reject or the bonus
                    // draw on full accept; both replace the greedy argmax target.
                    // temp<=0 with penalties collapses to greedy over the penalized
                    // logits inside the helper, so penalties apply there too.
                    SpecAcceptResult sr = speculative_accept_pointmass(
                        row_logits, drafts,
                        options.temperature, options.top_p, options.top_k,
                        options.min_p, options.presence_penalty,
                        options.repetition_penalty, mtp_seen, mtp_rng);
                    accepted = sr.accepted;
                    target_token = sr.extra_token >= 0 ? sr.extra_token : eos;
                    for (uint32_t i = 0; i < accepted; ++i) {
                        ++mtp_chain_verified[i];
                        ++mtp_spec_accepted;
                        ++mtp_chain_accepted[i];
                    }
                    if (accepted < drafts.size()) {
                        ++mtp_chain_verified[accepted];
                        ++mtp_spec_rejected;
                    }
                } else {
                    for (uint32_t i = 0; i < drafts.size(); ++i) {
                        target_token = row_argmaxes[i].token >= 0 ? row_argmaxes[i].token : eos;
                        ++mtp_chain_verified[i];
                        const bool ok = target_token == static_cast<int32_t>(drafts[i]);
                        if (ok) {
                            ++accepted;
                            ++mtp_spec_accepted;
                            ++mtp_chain_accepted[i];
                        } else {
                            ++mtp_spec_rejected;
                            break;
                        }
                    }
                }

                const bool all_accepted = accepted == drafts.size();
                if (accepted < mtp_accept_len_hist.size()) {
                    ++mtp_accept_len_hist[accepted];
                }
                if (trace_mtp_tokens) {
                    std::ostringstream tok_msg;
                    tok_msg << "native mtp_spec_tokens:"
                            << " decoded=" << decoded
                            << " pos=" << verify_base_position
                            << " current=" << current
                            << " drafts=";
                    for (size_t i = 0; i < drafts.size(); ++i) {
                        if (i) tok_msg << ",";
                        tok_msg << drafts[i];
                    }
                    tok_msg << " targets=";
                    for (size_t i = 0; i < row_argmaxes.size(); ++i) {
                        if (i) tok_msg << ",";
                        tok_msg << row_argmaxes[i].token;
                    }
                    tok_msg << " accepted=" << accepted
                            << " all=" << (all_accepted ? "true" : "false");
                    log(tok_msg.str());
                }
                const char *policy_action =
                    mtp_policy.update(static_cast<uint32_t>(drafts.size()),
                                      accepted,
                                      prompt_tokens.size() +
                                          static_cast<size_t>(decoded));
                if (mtp_policy.enabled && mtp_policy.trace) {
                    std::ostringstream policy_msg;
                    policy_msg << "native mtp_policy:"
                               << " batch=" << mtp_spec_batches
                               << " ctx=" << (prompt_tokens.size() +
                                               static_cast<size_t>(decoded))
                               << " drafted=" << drafts.size()
                               << " accepted=" << accepted
                               << " depth=" << mtp_policy.depth
                               << " action=" << policy_action
                               << " window_batches=" << mtp_policy.window_batches
                               << " avg_committed=" << std::fixed << std::setprecision(4)
                               << mtp_policy.last_avg_committed
                               << " full_rate=" << mtp_policy.last_full_rate
                               << " benefit=" << mtp_policy.last_benefit
                               << " cost=" << mtp_policy.last_cost
                               << " next_cost=" << mtp_policy.last_next_cost;
                    log(policy_msg.str());
                }
                if (all_accepted) {
                    bool kvmem_advanced_this_iter = false;
                    bool kvmem_finish_after_prefix = false;
                    if (use_transactional_replay_all) {
                        if (!captured_snapshot) {
                            throw std::runtime_error("MTP transactional replay requires a state snapshot");
                        }
                        const double t_restore_start = mtp_phase_time();
                        executor_->restore_state(mtp_spec_snapshot);
                        mtp_spec_restore_s += mtp_phase_time() - t_restore_start;
                        const double t_replay_start = mtp_phase_time();
                        step = replay_target_tokens_sequential(
                            verify_tokens, verify_base_position,
                            mtp_rebuild_accepted_prefix_enabled());
                        mtp_spec_replay_s += mtp_phase_time() - t_replay_start;
                        if (!step.ok) {
                            throw std::runtime_error("MTP transactional replay failed");
                        }
                        decode_ops += step.ops_executed;
                    } else {
                        kvmem_finish_after_prefix =
                            kvmem_advance_to(executor_->position(),
                                             /*defer_finish=*/true);
                        kvmem_advanced_this_iter = true;
                        rebuild_accepted_mtp_prefix(verify_tokens, verify_base_position,
                                                    kvmem_finish_after_prefix);
                    }
                    for (uint32_t i = 0; i < accepted; ++i) {
                        if (!emit_generated_token(drafts[i])) break;
                    }
                    if (decoded >= options.max_tokens) break;
                    // Greedy: the bonus token is the final row's argmax. The accept
                    // test already drew the bonus into target_token whenever logits
                    // were picked (sampling or penalties), so only recompute it on the
                    // pure-greedy (no-penalty) path.
                    if (!mtp_need_logits_pick) {
                        target_token = row_argmaxes[drafts.size()].token >= 0
                            ? row_argmaxes[drafts.size()].token
                            : eos;
                    }
                    next_token = static_cast<uint32_t>(target_token);
                    if (!emit_generated_token(next_token)) break;
                    if (!kvmem_advanced_this_iter) {
                        kvmem_advance_to(executor_->position());
                        kvmem_advanced_this_iter = true;
                    }
                } else {
                    bool kvmem_advanced_this_iter = false;
                    bool kvmem_finish_after_prefix = false;
                    ++mtp_spec_rollbacks;
                    std::vector<uint32_t> replay;
                    replay.reserve(accepted + 1);
                    replay.push_back(current);
                    for (uint32_t i = 0; i < accepted; ++i) {
                        replay.push_back(drafts[i]);
                    }
                    const bool single_token_replay =
                        use_single_token_replay && replay.size() == 1;
                    const bool use_checkpoint_replay =
                        !use_sequential_verifier &&
                        state_checkpoint_count > 0 &&
                        mtp_spec_state_checkpoints.ready &&
                        accepted < mtp_spec_state_checkpoints.count;
                    if (use_checkpoint_replay) {
                        const double t_restore_start = mtp_phase_time();
                        executor_->restore_state_checkpoint(mtp_spec_state_checkpoints,
                                                           accepted);
                        mtp_spec_restore_s += mtp_phase_time() - t_restore_start;
                        ++mtp_spec_state_checkpoint_reused;
                        if (accepted == 0) {
                            ++mtp_spec_prefix1_reused;
                        }
                        step = NativeExecutorReport{};
                        step.ok = true;
                    } else {
                        const double t_restore_start = mtp_phase_time();
                        if (!captured_snapshot) {
                            throw std::runtime_error("MTP rollback checkpoint unavailable without snapshot");
                        }
                        executor_->restore_state(mtp_spec_snapshot);
                        mtp_spec_restore_s += mtp_phase_time() - t_restore_start;
                        const double t_replay_start = mtp_phase_time();
                        if (use_sequential_verifier || use_transactional_replay) {
                            step = replay_target_tokens_sequential(
                                replay, verify_base_position,
                                use_transactional_replay &&
                                mtp_rebuild_accepted_prefix_enabled());
                        } else if (single_token_replay) {
                            step = executor_->forward_one_token(replay.front(),
                                                               /*compute_logits=*/false);
                        } else {
                            step = executor_->forward_n_tokens(replay, false);
                        }
                        mtp_spec_replay_s += mtp_phase_time() - t_replay_start;
                    }
                    if (trace_mtp_verify && !step.executed.empty()) {
                        accumulate_trace(mtp_replay_trace, step);
                        mtp_replay_trace_steps += replay.size();
                    }
                    if (!step.ok) throw std::runtime_error("MTP rollback replay failed");
                    decode_ops += step.ops_executed;
                    if (use_transactional_replay) {
                        // The transactional replay already rebuilt the MTP
                        // prefix token by token from the stable target hidden
                        // states.
                    } else if (single_token_replay &&
                        reuse_current_mtp_prefix &&
                        mtp_rebuild_accepted_prefix_enabled()) {
                        kvmem_finish_after_prefix =
                            kvmem_advance_to(executor_->position(),
                                             /*defer_finish=*/true);
                        kvmem_advanced_this_iter = true;
                        const double t_prefix_start = mtp_phase_time();
                        executor_->commit_mtp_prefix_from_current_hidden(executor_->position());
                        mtp_spec_prefix_s += mtp_phase_time() - t_prefix_start;
                        if (kvmem_finish_after_prefix) {
                            executor_->kvmem_finish_reselect();
                        }
                    } else if (single_token_replay) {
                        kvmem_finish_after_prefix =
                            kvmem_advance_to(executor_->position(),
                                             /*defer_finish=*/true);
                        kvmem_advanced_this_iter = true;
                        rebuild_current_mtp_prefix(replay.front(), verify_base_position,
                                                   kvmem_finish_after_prefix);
                    } else {
                        kvmem_finish_after_prefix =
                            kvmem_advance_to(executor_->position(),
                                             /*defer_finish=*/true);
                        kvmem_advanced_this_iter = true;
                        rebuild_accepted_mtp_prefix(replay, verify_base_position,
                                                    kvmem_finish_after_prefix);
                    }
                    for (uint32_t i = 0; i < accepted; ++i) {
                        if (!emit_generated_token(drafts[i])) break;
                    }
                    if (decoded >= options.max_tokens) break;
                    next_token = static_cast<uint32_t>(target_token);
                    if (!emit_generated_token(next_token)) break;
                    if (mtp_spec_rejected > mtp_reject_limit) {
                        mtp_spec_fallback = true;
                        run_spec_mtp = false;
                    }
                    if (!kvmem_advanced_this_iter) {
                        kvmem_advance_to(executor_->position());
                        kvmem_advanced_this_iter = true;
                    }
                }
                // kvmem cadence: the batched window-aware verify
                // (forward_n_tokens) advances position_ + the window tail
                // (window_query_pos_) in lockstep; on reject, either
                // restore_state_checkpoint (window-aware per-row rollback, D1.1)
                // or restore_state(snapshot) rolls BOTH back, and the replay
                // re-advances both by the accepted count. So position_ and the
                // window tail stay consistent for every committed token this
                // iteration. The branches above register the committed delta
                // exactly once. When a prefix rebuild follows, they may split
                // reselection into prepare/finish so tier H2D can overlap with
                // the MTP-prefix compute, then finish before the next target
                // attention uses the KVMem window.
            }
        }
        if (!run_spec_mtp) {
            const double t_plain_start = mtp_phase_time();
            run_plain_decode_remaining();
            if (spec_mtp) {
                mtp_spec_plain_s += mtp_phase_time() - t_plain_start;
            }
        }
        const double t_decode_end = wall_seconds();

        if (manage_device_scope) {
            st = device_->end();
            if (!st.ok) throw std::runtime_error(st.message);
        }

        const double prefill_s = std::max(t_prefill_end - t_prefill_start, 1e-9);
        // The post-prefill (decode-window) reselect sits between t_prefill_end
        // and the decode loop; isolate it so decode_s is the pure decode loop.
        const double reselect_s = std::max(t_reselect_end - t_prefill_end, 0.0);
        const double decode_s = std::max(t_decode_end - t_reselect_end, 1e-9);
        std::ostringstream msg;
        msg << "native generate:"
            << " prompt_tokens=" << prompt_tokens.size()
            << " prefill=" << fmt_seconds(prefill_s);
        if (!prompt_tokens.empty()) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (prompt_tokens.size() / prefill_s) << " tok/s)";
        }
        msg << " decoded=" << decoded
            << " decode=" << fmt_seconds(decode_s);
        if (decoded > 0) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (decoded / decode_s) << " tok/s)";
        }
        if (prefill_chunks > 1) {
            msg << " prefill_chunks=" << prefill_chunks
                << " prefill_chunk_size=" << prefill_chunk_size;
        }
        msg << " prefill_ops=" << prefill_ops << " decode_ops=" << decode_ops;
        log(msg.str());
        if (kvmem_on && QwenExecutor::kvmem_timing_enabled()) {
            QwenExecutor::kvmem_timing_emit_delta("phase=decode request=mtp",
                                                  kvmem_tbase);
        }
        if (kvmem_on) {
            const QwenExecutor::KvMemTierUsage tu = executor_->kvmem_tier_usage();
            std::ostringstream tmsg;
            tmsg << "[kvmem-tier-usage]"
                 << " total_blocks=" << tu.total_blocks
                 << " block_bytes=" << tu.block_bytes
                 << " gpu_used=" << tu.gpu_used_bytes
                 << " gpu_cap=" << tu.gpu_capacity_bytes
                 << " gpu_pool=" << (tu.gpu_pool ? 1 : 0)
                 << " cpu_used=" << tu.cpu_used_bytes
                 << " cpu_cap=" << tu.cpu_capacity_bytes
                 << " nvme_used=" << tu.nvme_used_bytes
                 << " nvme_cap=" << tu.nvme_capacity_bytes;
            log(tmsg.str());
        }
        if (spec_mtp) {
            std::ostringstream spec_summary;
            spec_summary << "native mtp_spec_summary:"
                         << " enabled=" << (mtp_spec_started ? "true" : "false")
                         << " batches=" << mtp_spec_batches
                         << " drafted=" << mtp_spec_drafted
                         << " accepted=" << mtp_spec_accepted
                         << " rejected=" << mtp_spec_rejected
                         << " rollbacks=" << mtp_spec_rollbacks
                         << " adaptive=" << (mtp_policy.enabled ? "true" : "false")
                         << " promotions=" << mtp_policy.promotions
                         << " reject_budget=";
            if (mtp_reject_limit == std::numeric_limits<uint32_t>::max()) {
                spec_summary << "off";
            } else {
                spec_summary << mtp_reject_limit;
            }
            spec_summary << " fallback=" << (mtp_spec_fallback ? "true" : "false")
                         << " acceptance=" << std::fixed << std::setprecision(4)
                         << (mtp_spec_drafted > 0
                                 ? static_cast<double>(mtp_spec_accepted) /
                                   static_cast<double>(mtp_spec_drafted)
                                 : 0.0)
                         << " mtp_ops=" << mtp_ops
                         << " prefix_tokens=" << mtp_prefix_tokens
                         << " prefix_ops=" << mtp_prefix_ops
                         << " prefix1_reuse=" << mtp_spec_prefix1_reused
                         << " state_ckpt_reuse=" << mtp_spec_state_checkpoint_reused
                         << " state_ckpt_count=" << state_checkpoint_count
                         << " batched_verify_batches="
                         << mtp_spec_batched_verify_batches
                         << " batched_verify_tokens="
                         << mtp_spec_batched_verify_tokens
                         << " draft_s=" << fmt_seconds(mtp_spec_draft_s)
                         << " snapshot_s=" << fmt_seconds(mtp_spec_snapshot_s)
                         << " verify_s=" << fmt_seconds(mtp_spec_verify_s)
                         << " restore_s=" << fmt_seconds(mtp_spec_restore_s)
                         << " replay_s=" << fmt_seconds(mtp_spec_replay_s)
                         << " plain_s=" << fmt_seconds(mtp_spec_plain_s)
                         << " prefix_s=" << fmt_seconds(mtp_spec_prefix_s);
            if (sync_mtp_phase_timing) {
                spec_summary << " phase_sync=true";
            }
            log(spec_summary.str());
            if (mtp_policy.enabled) {
                mtp_policy.compute_depth_score(prompt_tokens.size() + decoded,
                                               mtp_policy.depth,
                                               false);
                std::ostringstream policy_summary;
                policy_summary << "native mtp_policy_summary:"
                               << " enabled=true"
                               << " min=" << mtp_policy.min_depth
                               << " max=" << mtp_policy.max_depth
                               << " initial=" << mtp_policy.initial_depth
                               << " final=" << mtp_policy.depth
                               << " batches=" << mtp_policy.batches
                               << " changes=" << mtp_policy.changes
                               << " promotions=" << mtp_policy.promotions
                               << " demotions=" << mtp_policy.demotions
                               << " bad_windows=" << mtp_policy.bad_windows
                               << " good_windows=" << mtp_policy.good_windows
                               << " avg_committed=" << std::fixed << std::setprecision(4)
                               << mtp_policy.last_avg_committed
                               << " full_rate=" << mtp_policy.last_full_rate
                               << " benefit=" << mtp_policy.last_benefit
                               << " cost=" << mtp_policy.last_cost
                               << " next_cost=" << mtp_policy.last_next_cost;
                for (uint32_t i = 1; i <= mtp_policy.max_depth &&
                                     i <= MtpAdaptivePolicy::kMaxTrackedDepth; ++i) {
                    policy_summary << " d" << i << "_batches="
                                   << mtp_policy.depth_batches[i]
                                   << " d" << i << "_drafted="
                                   << mtp_policy.depth_drafted[i]
                                   << " d" << i << "_accepted="
                                   << mtp_policy.depth_accepted[i];
                }
                log(policy_summary.str());
            }
            std::ostringstream hist_summary;
            hist_summary << "native mtp_accept_hist:";
            for (uint32_t i = 0; i < mtp_accept_len_hist.size(); ++i) {
                hist_summary << " len" << i << "=" << mtp_accept_len_hist[i];
            }
            log(hist_summary.str());
            if (!trace_mtp) {
                for (uint32_t i = 0; i < mtp_chain_len; ++i) {
                    std::ostringstream offset_summary;
                    offset_summary << "native mtp_chain_offset:"
                                   << " step=" << (i + 1)
                                   << " verified=" << mtp_chain_verified[i]
                                   << " accepted=" << mtp_chain_accepted[i]
                                   << " acceptance=" << std::fixed << std::setprecision(4)
                                   << (mtp_chain_verified[i] > 0
                                           ? static_cast<double>(mtp_chain_accepted[i]) /
                                             static_cast<double>(mtp_chain_verified[i])
                                           : 0.0);
                    log(offset_summary.str());
                }
            }
        }
        if (trace_mtp) {
            const uint64_t mtp_verified = mtp_chain_verified.empty() ? 0 : mtp_chain_verified[0];
            const uint64_t mtp_accepted = mtp_chain_accepted.empty() ? 0 : mtp_chain_accepted[0];
            std::ostringstream mtp_summary;
            mtp_summary << "native mtp_summary:"
                        << " drafts=" << mtp_drafts
                        << " verified=" << mtp_verified
                        << " accepted=" << mtp_accepted
                        << " acceptance=" << std::fixed << std::setprecision(4)
                        << (mtp_verified > 0
                                ? static_cast<double>(mtp_accepted) / static_cast<double>(mtp_verified)
                                : 0.0)
                        << " mtp_ops=" << mtp_ops;
            log(mtp_summary.str());
            std::ostringstream chain_summary;
            chain_summary << "native mtp_chain_summary:"
                          << " max=" << mtp_chain_len
                          << " chains=" << mtp_chains
                          << " drafts=" << mtp_drafts
                          << " prefix=" << (use_mtp_prefix ? "true" : "false")
                          << " prefix_tokens=" << mtp_prefix_tokens
                          << " prefix_ops=" << mtp_prefix_ops
                          << " mtp_ops=" << mtp_ops;
            log(chain_summary.str());
            for (uint32_t i = 0; i < mtp_chain_len; ++i) {
                std::ostringstream offset_summary;
                offset_summary << "native mtp_chain_offset:"
                               << " step=" << (i + 1)
                               << " verified=" << mtp_chain_verified[i]
                               << " accepted=" << mtp_chain_accepted[i]
                               << " acceptance=" << std::fixed << std::setprecision(4)
                               << (mtp_chain_verified[i] > 0
                                       ? static_cast<double>(mtp_chain_accepted[i]) /
                                         static_cast<double>(mtp_chain_verified[i])
                                       : 0.0);
                log(offset_summary.str());
            }
        }
        if (trace_mtp_verify) {
            if (!mtp_draft_trace.empty()) {
                log_trace("mtp_draft_trace", mtp_draft_trace, mtp_draft_trace_steps);
            }
            if (!mtp_verify_trace.empty()) {
                log_trace("mtp_verify_trace", mtp_verify_trace, mtp_verify_trace_steps);
            }
            if (!mtp_replay_trace.empty()) {
                log_trace("mtp_replay_trace", mtp_replay_trace, mtp_replay_trace_steps);
            }
        }
        if (decode_trace_enabled() && !decode_trace.empty()) {
            log_decode_trace(decode_trace, decode_trace_steps);
        }

        if (stats_out) {
            stats_out->prefill_s = prefill_s;
            stats_out->decode_s = decode_s;
            stats_out->reselect_s = reselect_s;
            stats_out->decoded = decoded;
            stats_out->prompt_tokens = prompt_tokens.size();
            stats_out->acceptance =
                mtp_spec_drafted > 0
                    ? static_cast<double>(mtp_spec_accepted) /
                          static_cast<double>(mtp_spec_drafted)
                    : 0.0;
            stats_out->kvmem_at_boundary = kvmem_at_prefill_end;
            stats_out->kvmem_boundary_valid = kvmem_boundary_valid;
        }

        return generated;
    }

    static std::string fmt_seconds(double s) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3) << s << "s";
        return ss.str();
    }

    void log(const std::string &line) const {
        std::cerr << "[qw3] " << line << "\n";
    }

    // Build a synthetic document pool of at least `want_tokens` tokens. The
    // sentences are deterministic but varied (cycled vocab + a small LCG) so the
    // MTP draft head sees enough local structure to keep acceptance healthy
    // without the degenerate-repeat fp-atomic nondeterminism. Tokens are flat,
    // so each session turn can carve an exact-count slice for its delta.
    void build_session_corpus(std::vector<uint32_t> &pool,
                              uint64_t want_tokens) const {
        static const char *kNames[] = {
            "Atlas", "Borealis", "Cygnus", "Draco", "Equinox", "Fenrir",
            "Gemini", "Helios", "Icarus", "Juno", "Kepler", "Lyra",
            "Meridian", "Nimbus", "Orion", "Pegasus", "Quasar", "Rigel",
            "Solstice", "Titan", "Umbra", "Vega", "Wraith", "Xenon"};
        static const char *kNouns[] = {
            "module", "pipeline", "scheduler", "allocator", "kernel", "tensor",
            "gateway", "registry", "cache", "shard", "ledger", "planner",
            "executor", "router", "sentinel", "indexer", "compiler", "daemon"};
        static const char *kVerbs[] = {
            "processed", "validated", "compressed", "dispatched", "reconciled",
            "buffered", "serialized", "rebalanced", "checkpointed", "migrated"};
        const size_t nn = sizeof(kNames) / sizeof(kNames[0]);
        const size_t no = sizeof(kNouns) / sizeof(kNouns[0]);
        const size_t nv = sizeof(kVerbs) / sizeof(kVerbs[0]);
        pool.reserve(static_cast<size_t>(want_tokens) + 4096);
        uint64_t lcg = 0x9e3779b97f4a7c15ull;
        uint64_t doc = 0;
        std::string para;
        para.reserve(8192);
        while (pool.size() < want_tokens) {
            para.clear();
            // Batch ~32 sentences per encode call to amortize tokenizer cost.
            for (int s = 0; s < 32 && pool.size() < want_tokens; ++s) {
                lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
                const uint64_t r = lcg >> 17;
                para += "Record ";
                para += std::to_string(doc++);
                para += ": the ";
                para += kNouns[(r) % no];
                para += " ";
                para += kNames[(r >> 3) % nn];
                para += " ";
                para += kVerbs[(r >> 8) % nv];
                para += " ";
                para += std::to_string(static_cast<unsigned>((r >> 11) % 100000));
                para += " entries for region ";
                para += kNames[(r >> 21) % nn];
                para += " while the ";
                para += kNouns[(r >> 27) % no];
                para += " reported a latency of ";
                para += std::to_string(static_cast<unsigned>((r >> 33) % 4096));
                para += " microseconds.\n";
            }
            const std::vector<int32_t> ids = tokenizer_->encode(para, false);
            pool.reserve(pool.size() + ids.size());
            for (int32_t id : ids) {
                pool.push_back(static_cast<uint32_t>(id));
                if (pool.size() >= want_tokens) break;
            }
        }
    }

    static uint64_t current_rss_mib() {
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
        // ru_maxrss is in KiB on Linux.
        return static_cast<uint64_t>(ru.ru_maxrss) / 1024;
    }

    void print_session_turn(size_t turn, uint64_t ctx_tokens,
                            uint64_t delta_tokens, double sel_ms,
                            double stage_in_ms, double stage_out_ms,
                            double assemble_ms, double asm_pages_ms,
                            double asm_rerope_ms, double asm_kbar_ms,
                            uint32_t stage_in_blocks, uint32_t stage_out_blocks,
                            double prefill_s, double decode_s, double decode_tps,
                            int decoded, double acceptance, uint64_t gpu_mib,
                            uint64_t rss_mib, double inpre_ms = 0.0,
                            uint32_t inpre_stage_in_blocks = 0,
                            uint32_t inpre_stage_out_blocks = 0) const {
        std::ostringstream m;
        m << std::fixed;
        m << "\n[kvmem-session] turn=" << turn
          << " ctx=" << ctx_tokens << "tok (+" << delta_tokens << ")"
          << " GPU=" << gpu_mib << "MiB RSS=" << rss_mib << "MiB\n";
        m << std::setprecision(3);
        m << "  step1 selection (top-k retrieval)        = "
          << std::setw(10) << sel_ms << " ms\n";
        m << "  step2 stage-in   (CPU/NVMe -> GPU)        = "
          << std::setw(10) << stage_in_ms << " ms  (" << stage_in_blocks
          << " blk)\n";
        m << "  step3 stage-out  (GPU -> CPU/NVMe evict)  = "
          << std::setw(10) << stage_out_ms << " ms  (" << stage_out_blocks
          << " blk)\n";
        m << "  step4 assemble   (pages+re-RoPE+k-bar)    = "
          << std::setw(10) << assemble_ms << " ms  (pages=" << asm_pages_ms
          << " rerope=" << asm_rerope_ms << " kbar=" << asm_kbar_ms << ")\n";
        m << "  step5 prefill    (forward new chunk)      = "
          << std::setw(10) << (prefill_s * 1000.0) << " ms  ("
          << std::setprecision(1) << (delta_tokens / std::max(prefill_s, 1e-9))
          << " tok/s)\n";
        m << std::setprecision(3);
        // The bounded GPU pool can force kvmem stage-in/out mid-prefill; that
        // cost is INSIDE step5's wall above (not double-counted in steps 1-4,
        // which are the post-prefill decode-window reselect only).
        m << "    +-- of which kvmem in-prefill offload   = "
          << std::setw(10) << inpre_ms << " ms  (in=" << inpre_stage_in_blocks
          << " out=" << inpre_stage_out_blocks << " blk)\n";
        m << "  step6 decode     (MTP, " << decoded << " tok)            = "
          << std::setw(10) << (decode_s * 1000.0) << " ms  ("
          << std::setprecision(2) << decode_tps << " tok/s, accept="
          << std::setprecision(4) << acceptance << ")";
        log(m.str());
    }

    void log_prefill_chunk(uint32_t chunk_index, size_t offset, size_t tokens,
                           double seconds) const {
        std::ostringstream msg;
        msg << "prefill_chunk"
            << " index=" << chunk_index
            << " offset=" << offset
            << " tokens=" << tokens
            << " elapsed=" << fmt_seconds(seconds);
        if (tokens > 0) {
            msg << " (" << std::fixed << std::setprecision(2)
                << (static_cast<double>(tokens) / std::max(seconds, 1e-9))
                << " tok/s)";
        }
        log(msg.str());
    }

    void log_trace(const std::string &label,
                   const std::unordered_map<std::string, TraceStats> &stats,
                   uint64_t steps) const {
        std::vector<std::pair<std::string, TraceStats>> items(stats.begin(), stats.end());
        std::sort(items.begin(), items.end(),
                  [](const auto &a, const auto &b) {
                      return a.second.total_us > b.second.total_us;
                  });
        double total_us = 0.0;
        for (const auto &item : items) total_us += item.second.total_us;

        std::ostringstream header;
        header << label
               << " steps=" << steps
               << " groups=" << items.size()
               << " total=" << fmt_seconds(total_us / 1.0e6);
        log(header.str());

        const uint32_t top_n = std::min<uint32_t>(
            decode_trace_top_n(), static_cast<uint32_t>(items.size()));
        for (uint32_t i = 0; i < top_n; ++i) {
            const auto &[name, item] = items[i];
            const double share = total_us > 0.0 ? (100.0 * item.total_us / total_us) : 0.0;
            const double per_step_us = steps > 0 ? item.total_us / static_cast<double>(steps) : 0.0;
            const double avg_call_us = item.calls > 0 ? item.total_us / static_cast<double>(item.calls) : 0.0;
            std::ostringstream line;
            line << label
                 << " rank=" << (i + 1)
                 << " op=" << name
                 << " calls=" << item.calls
                 << " total_ms=" << std::fixed << std::setprecision(3) << (item.total_us / 1000.0)
                 << " per_step_ms=" << (per_step_us / 1000.0)
                 << " avg_call_us=" << avg_call_us
                 << " max_call_us=" << item.max_us
                 << " share=" << share << "%";
            log(line.str());
        }
    }

    void log_decode_trace(const std::unordered_map<std::string, TraceStats> &stats,
                          uint64_t steps) const {
        log_trace("decode_trace", stats, steps);
    }

    EngineOptions options_;
    std::unique_ptr<QwenNativeModel> model_;
    std::unique_ptr<DeviceBackend> device_;
    // Recycles the pinned kvmem CPU-tier buffer across the per-request CB
    // executors so it is cudaHostAlloc'd once, not per admit (an 8 GiB tier
    // otherwise added ~5 s to every request's TTFT). Declared right after
    // device_ so it is destroyed before it (frees buffers while CUDA is alive)
    // and after the executors that borrow from it.
    std::unique_ptr<HostTierBufferPool> cb_host_tier_pool_;
    std::unique_ptr<QwenWeights> weights_;
    std::unique_ptr<QwenExecutor> executor_;
    std::unique_ptr<QwenTokenizer> tokenizer_;
    std::unique_ptr<BatchedPrefillExecutor> cb_prefill_executor_;
    std::unique_ptr<BatchedDecodeExecutor> cb_decode_executor_;
    std::unique_ptr<GlobalKvPagePool> cb_kv_pool_;
    std::unique_ptr<GlobalKvPagePool> cb_mtp_kv_pool_;

    // ---- Prefix cache (Phase 1: lossless page-aligned prefix reuse) -------
    // One entry per committed, page-aligned prompt prefix. Pages are pinned in
    // cb_kv_pool_ while the entry lives; recur holds the recurrent+conv state
    // captured at exactly aligned_len so reuse is lossless on the hybrid model.
    struct PrefixCacheEntry {
        uint64_t id = 0;
        std::vector<uint32_t> tokens;     // exact prefix tokens (collision-safe)
        uint32_t aligned_len = 0;         // == tokens.size(), multiple of page_size
        std::vector<int32_t> kv_pages;    // pinned physical pages, logical 0..n
        QwenExecutor::StateSnapshot recur;// recurrent+conv state at aligned_len
        uint32_t refcount = 0;            // live requests reading these pages
        uint64_t last_used_seq = 0;       // LRU
    };
    std::unordered_map<uint64_t, std::vector<PrefixCacheEntry>> prefix_cache_;
    std::mutex prefix_cache_mu_;
    uint64_t prefix_cache_seq_ = 0;
    uint64_t prefix_cache_next_id_ = 1;
    uint32_t prefix_cache_pinned_pages_ = 0;
    bool prefix_cache_evict_cb_installed_ = false;
    std::vector<std::unique_ptr<DeviceTensor>> cb_k_cache_storage_;
    std::vector<std::unique_ptr<DeviceTensor>> cb_v_cache_storage_;
    std::vector<std::unique_ptr<DeviceTensor>> cb_mtp_k_cache_storage_;
    std::vector<std::unique_ptr<DeviceTensor>> cb_mtp_v_cache_storage_;
    QwenExecutor::KvCacheStorage cb_kv_cache_view_;
    QwenExecutor::KvCacheStorage cb_mtp_kv_cache_view_;
    uint32_t cb_mtp_draft_capacity_ = 0;
    std::unique_ptr<DeviceTensor> cb_mtp_h_input_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_h_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_norm_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_concat_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_q_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_q_row_;
    std::unique_ptr<DeviceTensor> cb_mtp_k_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_v_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_k_row_;
    std::unique_ptr<DeviceTensor> cb_mtp_v_row_;
    std::unique_ptr<DeviceTensor> cb_mtp_mid_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_mid_row_;
    std::unique_ptr<DeviceTensor> cb_mtp_ffn_gate_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_ffn_up_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_ffn_mid_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_ffn_out_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_logits_batch_;
    std::unique_ptr<DeviceTensor> cb_mtp_draft_positions_i32_;
    uint32_t cb_mtp_draft_positions_capacity_ = 0;
    uint32_t cb_prefill_ragged_batch_capacity_ = 0;
    uint32_t cb_prefill_ragged_page_capacity_ = 0;
    uint32_t cb_prefill_ragged_row_capacity_ = 0;
    std::unique_ptr<DeviceTensor> cb_prefill_q_indptr_i32_;
    std::unique_ptr<DeviceTensor> cb_prefill_page_indptr_i32_;
    std::unique_ptr<DeviceTensor> cb_prefill_row_page_indptr_i32_;
    std::unique_ptr<DeviceTensor> cb_prefill_page_indices_i32_;
    std::unique_ptr<DeviceTensor> cb_prefill_logical_positions_i32_;
    std::unique_ptr<DeviceTensor> cb_prefill_last_page_len_i32_;
    std::unique_ptr<DeviceTensor> cb_prefill_seq_lens_i32_;
    uint64_t cb_prefill_recurrent_state_capacity_ = 0;
    uint64_t cb_prefill_conv_state_capacity_ = 0;
    std::unique_ptr<DeviceTensor> cb_prefill_recurrent_state_batch_;
    std::unique_ptr<DeviceTensor> cb_prefill_conv_state_batch_;

    std::mutex cb_mu_;
    std::condition_variable cb_cv_;
    std::deque<std::shared_ptr<ContinuousBatchRequest>> cb_pending_;
    std::thread cb_worker_;
    bool cb_running_ = false;
    bool cb_stop_ = false;
    std::atomic<uint64_t> cb_request_counter_{0};
    uint64_t cb_reserved_tokens_ = 0;
    std::atomic<uint64_t> cb_decode_batches_{0};
    std::atomic<uint64_t> cb_decode_tokens_{0};
    std::atomic<uint32_t> cb_decode_max_batch_{0};
};

} // namespace

std::unique_ptr<Backend> make_qwen_native_backend() {
    return std::make_unique<QwenNativeBackend>();
}

int run_kvmem_session(EngineOptions engine, const KvMemSessionConfig &cfg) {
    // Force the single-request native MTP + kvmem path this harness profiles.
    engine.backend = BackendKind::QwenNative;
    engine.native_heavy = true;
    if (engine.native_kernels.empty()) engine.native_kernels = "cuda";
    if (engine.prefill_chunk < 0) engine.prefill_chunk = 2048;

    // kvmem is mandatory; reselect only at the prefill boundary (step mode),
    // never during decode -- exactly the configuration the user asked to
    // profile.
    engine.kvmem_enabled = true;
    engine.kvmem_update_mode = "step";

    // MTP on for realistic decode throughput.
    if (!engine.native_mtp_chain_set || engine.native_mtp_chain <= 0) {
        engine.native_mtp_chain = 4;
        engine.native_mtp_chain_set = true;
    }
    engine.native_mtp_speculate = true;

    // Env bridge (the backend still reads a few toggles from process env at
    // load + generate). QW3_KVMEM_TIMING populates the step-1..4 breakdown; its
    // extra device syncs only touch the prefill-boundary reselect (step mode
    // has no decode-time reselect), so steady-state decode throughput is
    // unperturbed.
    setenv("QW3_MTP_SPECULATE", "1", 1);
    setenv("QW3_MTP_POLICY",
           engine.mtp_policy.empty() ? "fixed" : engine.mtp_policy.c_str(), 1);
    setenv("QW3_KVMEM_TIMING", "1", 1);

    QwenNativeBackend backend;
    backend.load(engine);
    return backend.run_kvmem_session(cfg);
}

} // namespace qw3
