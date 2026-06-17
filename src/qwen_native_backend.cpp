#include "backend.hpp"
#include "env_flags.hpp"
#include "qwen_executor.hpp"
#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/tokenizer.hpp"

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

// Host-side token sampler over the full fp32 vocab logits. temp<=0 is greedy.
// Otherwise: apply top-k/top-p/min-p filters, renormalize, draw with rng.
// Kept on host because copy_last_logits() already round-trips logits for the
// dump-logits path; sampling adds no device work.
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
    // softmax(logits / temp), numerically stabilized by max subtraction.
    const float inv_t = 1.0f / temp;
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
    if (top_k > 0 && top_k < static_cast<int>(probs.size())) {
        probs.resize(static_cast<size_t>(top_k));
    }
    if (min_p > 0.0f && !probs.empty()) {
        const float cutoff = probs.front().first * min_p;
        probs.erase(std::remove_if(probs.begin(), probs.end(),
                                   [&](const auto &pr) { return pr.first < cutoff; }),
                    probs.end());
        if (probs.empty()) return -1;
    }
    // Nucleus: keep smallest descending-probability prefix with cumulative
    // probability >= top_p.
    if (top_p < 1.0f && top_p > 0.0f) {
        double cum = 0.0;
        size_t keep = probs.size();
        for (size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i].first;
            if (cum >= top_p) { keep = i + 1; break; }
        }
        probs.resize(keep);
    }
    std::vector<double> weights;
    weights.reserve(probs.size());
    for (const auto &pr : probs) weights.push_back(pr.first);
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    return probs[dist(rng)].second;
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

class GlobalKvPagePool final : public KvPhysicalPageAllocator {
public:
    GlobalKvPagePool(uint32_t total_pages, uint32_t page_size)
        : total_pages_(total_pages), page_size_(page_size) {
        free_pages_.reserve(total_pages_);
        for (uint32_t i = 0; i < total_pages_; ++i) {
            free_pages_.push_back(static_cast<int32_t>(total_pages_ - 1U - i));
        }
    }

    int32_t allocate_physical_page() override {
        std::lock_guard<std::mutex> lk(mu_);
        if (free_pages_.empty()) {
            throw std::runtime_error(
                "global KV page pool exhausted: free=0 total=" +
                std::to_string(total_pages_) +
                " page_size=" + std::to_string(page_size_));
        }
        const int32_t page = free_pages_.back();
        free_pages_.pop_back();
        ++used_pages_;
        return page;
    }

    void release_physical_pages(const std::vector<int32_t> &pages) override {
        if (pages.empty()) return;
        std::lock_guard<std::mutex> lk(mu_);
        for (int32_t page : pages) {
            if (page < 0 || static_cast<uint32_t>(page) >= total_pages_) {
                continue;
            }
            free_pages_.push_back(page);
            if (used_pages_ > 0) --used_pages_;
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
        cb_stop_ = false;
        cb_running_ = true;
        cb_worker_ = std::thread([this]() { continuous_batch_worker_loop(); });
        log("native continuous_batching: enabled=true mode=batch_step_executor");
    }

    void stop_continuous_batch_worker() {
        {
            std::lock_guard<std::mutex> lk(cb_mu_);
            if (!cb_running_) return;
            cb_stop_ = true;
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
        req->reserved_tokens =
            static_cast<uint64_t>(prompt_tokens.size()) +
            static_cast<uint64_t>(std::max(0, options.max_tokens));
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
        QwenExecutor::StateSnapshot snapshot;
        QwenExecutor::StateCheckpointSet checkpoints;
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
            QwenExecutor::DecodeStateView view =
                a.executor->decode_state_view();

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
                view.kv_page_indices_host != nullptr &&
                view.kv_page_count > 0;
            if (entry_metadata_ready) {
                if (batch.page_size == 0) {
                    batch.page_size = view.kv_page_size;
                } else if (batch.page_size != view.kv_page_size) {
                    entry_metadata_ready = false;
                }
            }
            const uint32_t seq_len = job.base_position + chunk;
            if (entry_metadata_ready) {
                const uint32_t pages =
                    (seq_len + view.kv_page_size - 1) / view.kv_page_size;
                if (pages == 0 || view.kv_page_count < pages) {
                    entry_metadata_ready = false;
                } else {
                    const int32_t request_page_begin =
                        static_cast<int32_t>(batch.page_indices.size());
                    for (uint32_t p = 0; p < pages; ++p) {
                        batch.page_indices.push_back(
                            view.kv_page_indices_host[p]);
                    }
                    for (uint32_t t = 0; t < chunk; ++t) {
                        batch.logical_positions.push_back(
                            static_cast<int32_t>(job.base_position + t));
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
            double decode_start = 0.0;
            MtpAdaptivePolicy policy;
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
                    const double verify0 = wall_seconds();
                    const std::vector<BatchedDecodeOutput> layer_outputs =
                        cb_decode_executor_->decode(
                            mtp_active, BatchedDecodeInput{&decode_batch});
                    const double verify_s =
                        std::max(wall_seconds() - verify0, 0.0);
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
                const double prefill0 = wall_seconds();
                for (size_t offset = 0; offset < req->prompt_tokens.size();
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
                    NativeExecutorReport prefix =
                        a.executor->prime_mtp_prefix_from_last_batch(
                            chunk, static_cast<uint32_t>(offset));
                    if (!prefix.ok) {
                        throw std::runtime_error(
                            "MTP prefix priming failed in batched lane");
                    }
                }
                const double prefill_s = std::max(wall_seconds() - prefill0, 1e-9);
                a.prefill_ops = prefill_ops;
                a.next_token = step.argmax_token >= 0
                    ? static_cast<uint32_t>(step.argmax_token)
                    : static_cast<uint32_t>(eos);
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
                        if (state_checkpoint_count == 0) {
                            const double snapshot0 = wall_seconds();
                            a.executor->capture_state(job.snapshot);
                            stats[row].snapshot_s +=
                                std::max(wall_seconds() - snapshot0, 0.0);
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
                        const double draft0 = wall_seconds();
                        batch_ok = run_continuous_mtp_batched_draft_step(
                            mtp_active, steps, depth == 0);
                        const double per_row_s =
                            std::max(wall_seconds() - draft0, 0.0) /
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
                    const double draft0 = wall_seconds();
                    std::vector<NativeExecutorReport> chain =
                        use_device_draft
                            ? a.executor->forward_mtp_draft_chain_with_prefix_device(
                                  current, draft_limit)
                            : a.executor->forward_mtp_draft_chain_with_prefix(
                                  current, draft_limit);
                    stats[row].draft_s +=
                        std::max(wall_seconds() - draft0, 0.0);
                    ContinuousMtpVerifyJob job;
                    job.row = row;
                    job.current = current;
                    job.base_position = a.executor->position();
                    if (state_checkpoint_count == 0) {
                        const double snapshot0 = wall_seconds();
                        a.executor->capture_state(job.snapshot);
                        stats[row].snapshot_s +=
                            std::max(wall_seconds() - snapshot0, 0.0);
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
                        a.executor->commit_mtp_prefix(a.executor->position());
                        a.next_token = step.argmax_token >= 0
                            ? static_cast<uint32_t>(step.argmax_token)
                            : static_cast<uint32_t>(eos);
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
                            out.report =
                                mtp_active[job.row].executor->forward_n_tokens(
                                    job.verify_tokens, true,
                                    &out.row_argmaxes,
                                    state_checkpoint_count > 0
                                        ? &job.checkpoints
                                        : nullptr,
                                    state_checkpoint_count,
                                    /*copy_last_logits=*/false);
                            if (!out.report.ok) {
                                out.error = "MTP single verifier failed";
                            }
                            return out;
                        };
                    if (jobs.size() == 1) {
                        ContinuousMtpVerifyJob &job = jobs.front();
                        const double verify0 = wall_seconds();
                        outputs.push_back(run_single_verifier(job));
                        if (job.row < stats.size()) {
                            stats[job.row].verify_s +=
                                std::max(wall_seconds() - verify0, 0.0);
                        }
                    } else if (continuous_mtp_layered_verify_enabled() &&
                               run_layered_verifier(jobs, outputs)) {
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
                            const double verify0 = wall_seconds();
                            outputs = cb_prefill_executor_->prefill(
                                mtp_active, batch, metadata);
                            used_ragged_verify = true;
                            const double verify_s =
                                std::max(wall_seconds() - verify0, 0.0);
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
                                if (job.snapshot.ready &&
                                    job.row < mtp_active.size()) {
                                    mtp_active[job.row].executor->restore_state(
                                        job.snapshot);
                                }
                            }
                            outputs.clear();
                            outputs.reserve(jobs.size());
                            for (ContinuousMtpVerifyJob &job : jobs) {
                                const double verify0 = wall_seconds();
                                outputs.push_back(run_single_verifier(job));
                                if (job.row < stats.size()) {
                                    stats[job.row].verify_s +=
                                        std::max(wall_seconds() - verify0,
                                                 0.0);
                                }
                            }
                        } else if (used_ragged_verify &&
                                   state_checkpoint_count > 0) {
                            for (uint32_t j = 0;
                                 j < jobs.size() && j < outputs.size(); ++j) {
                                jobs[j].checkpoints =
                                    std::move(outputs[j].checkpoints);
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
                        uint32_t accepted = 0;
                        int32_t target = eos;
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
                                    const double prefix0 = wall_seconds();
                                    a.executor
                                        ->commit_mtp_prefix_from_current_hidden(
                                            a.executor->position());
                                    s.prefix_s += std::max(
                                        wall_seconds() - prefix0, 0.0);
                                } else {
                                    a.executor->commit_mtp_prefix(
                                        a.executor->position());
                                }
                            } else if (mtp_rebuild_accepted_prefix_enabled()) {
                                const double prefix0 = wall_seconds();
                                NativeExecutorReport prefix =
                                    a.executor->prime_mtp_prefix_from_last_batch(
                                        job.verify_tokens, job.base_position,
                                        mtp_prefix_rebuild_batch_min_tokens());
                                s.prefix_s +=
                                    std::max(wall_seconds() - prefix0, 0.0);
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
                                const double prefix0 = wall_seconds();
                                a.executor
                                    ->commit_mtp_prefix_from_current_hidden(
                                        a.executor->position());
                                s.prefix_s +=
                                    std::max(wall_seconds() - prefix0, 0.0);
                            } else {
                                a.executor->commit_mtp_prefix(
                                    a.executor->position());
                            }
                            ++s.rollbacks;
                        } else {
                            const bool use_checkpoint_replay =
                                state_checkpoint_count > 0 &&
                                job.checkpoints.ready &&
                                accepted < job.checkpoints.count;
                            if (use_checkpoint_replay) {
                                const double restore0 = wall_seconds();
                                a.executor->restore_state_checkpoint(
                                    job.checkpoints, accepted);
                                s.restore_s +=
                                    std::max(wall_seconds() - restore0, 0.0);
                                ++s.state_checkpoint_reused;
                                if (accepted == 0) ++s.prefix1_reused;
                            } else {
                                if (!job.snapshot.ready) {
                                    throw std::runtime_error(
                                        "MTP batched verifier replay requires a state snapshot");
                                }
                                const double restore0 = wall_seconds();
                                a.executor->restore_state(job.snapshot);
                                s.restore_s +=
                                    std::max(wall_seconds() - restore0, 0.0);
                                std::vector<uint32_t> replay;
                                replay.reserve(accepted + 1);
                                replay.push_back(job.current);
                                for (uint32_t i = 0; i < accepted; ++i) {
                                    replay.push_back(job.drafts[i]);
                                }
                                double prefix_seconds = 0.0;
                                uint64_t prefix_ops = 0;
                                const double replay0 = wall_seconds();
                                NativeExecutorReport replay_report =
                                    a.executor->replay_tokens_with_mtp_prefix(
                                        replay, job.base_position,
                                        mtp_rebuild_accepted_prefix_enabled(),
                                        &prefix_seconds, &prefix_ops);
                                s.replay_s +=
                                    std::max(wall_seconds() - replay0, 0.0);
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
                        for (uint32_t i = 0; i < accepted; ++i) {
                            if (!emit(a, job.drafts[i])) break;
                        }
                        if (a.decoded >= a.req->options.max_tokens) continue;
                        if (all_accepted) {
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
                for (size_t i = mtp_active.size(); i > 0; --i) {
                    ContinuousBatchActive &a = mtp_active[i - 1];
                    if (!a.req || a.decoded >= a.req->options.max_tokens ||
                        should_stop(a, a.next_token)) {
                        const MtpStats s = stats[i - 1];
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
        a.executor->reset_state();
        a.seen_tokens.reserve(req->prompt_tokens.size() +
                              static_cast<size_t>(req->options.max_tokens));
        for (uint32_t token : req->prompt_tokens) ++a.seen_tokens[token];
        a.rng.seed(req->options.seed);
        budget_init(a.budget, req->options);
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
            const uint32_t chunk = continuous_prefill_chunk_tokens(remaining);
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

    void finish_continuous_active(ContinuousBatchActive &a) {
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

    // MTP draft/verify/speculate + adaptive-depth path. Ported from qw3_ly,
    // adapted to qw3's executor signatures + memory-safe prefix chunking.
    std::string generate_mtp(std::vector<uint32_t> &prompt_tokens,
                             const GenerationOptions &options,
                             const TokenCallback &on_text,
                             DumpStream *dump,
                             bool spec_mtp,
                             bool trace_mtp,
                             QwenExecutor *override_executor = nullptr,
                             bool manage_device_scope = true) {
        QwenExecutor *executor_ =
            override_executor != nullptr ? override_executor : this->executor_.get();
        if (!executor_) throw std::runtime_error("MTP executor unavailable");
        DeviceStatus st;
        if (manage_device_scope) {
            st = device_->begin();
            if (!st.ok) throw std::runtime_error(st.message);
        }
        executor_->reset_state();

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
                prime_mtp_prefix(chunk, static_cast<uint32_t>(offset));
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

        std::string generated;
        const int32_t eos = tokenizer_->eos_id();
        ThinkingBudgetState budget;
        budget_init(budget, options);
        uint32_t next_token = step.argmax_token >= 0 ? static_cast<uint32_t>(step.argmax_token)
                                                     : static_cast<uint32_t>(eos);
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
                                               uint32_t base_position) {
            if (!mtp_rebuild_accepted_prefix_enabled()) {
                executor_->commit_mtp_prefix(executor_->position());
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
        };
        auto rebuild_current_mtp_prefix = [&](uint32_t token, uint32_t base_position) {
            if (!mtp_rebuild_accepted_prefix_enabled()) {
                executor_->commit_mtp_prefix(executor_->position());
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
                if (decode_as_batch) {
                    const std::vector<uint32_t> one_token{feed};
                    step = executor_->forward_n_tokens(one_token);
                } else {
                    step = executor_->forward_one_token(feed);
                }
                if (!step.ok) throw std::runtime_error("decode failed");
                decode_ops += step.ops_executed;
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
                next_token = budget_next_feed(budget, static_cast<uint32_t>(new_argmax));
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
                    executor_->commit_mtp_prefix(executor_->position());
                    next_token = static_cast<uint32_t>(new_argmax);
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
                const double t_verify_start = mtp_phase_time();
                if (use_sequential_verifier) {
                    step = NativeExecutorReport{};
                    step.ok = true;
                    row_argmaxes.reserve(verify_tokens.size());
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
                    }
                } else {
                    step = executor_->forward_n_tokens(
                        verify_tokens, true, &row_argmaxes,
                        state_checkpoint_count > 0 ? &mtp_spec_state_checkpoints : nullptr,
                        state_checkpoint_count,
                        /*copy_last_logits=*/!mtp_skip_verify_logits_copy_enabled());
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
                        rebuild_accepted_mtp_prefix(verify_tokens, verify_base_position);
                    }
                    for (uint32_t i = 0; i < accepted; ++i) {
                        if (!emit_generated_token(drafts[i])) break;
                    }
                    if (decoded >= options.max_tokens) break;
                    target_token = row_argmaxes[drafts.size()].token >= 0
                        ? row_argmaxes[drafts.size()].token
                        : eos;
                    next_token = static_cast<uint32_t>(target_token);
                    if (!emit_generated_token(next_token)) break;
                } else {
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
                        const double t_prefix_start = mtp_phase_time();
                        executor_->commit_mtp_prefix_from_current_hidden(executor_->position());
                        mtp_spec_prefix_s += mtp_phase_time() - t_prefix_start;
                    } else if (single_token_replay) {
                        rebuild_current_mtp_prefix(replay.front(), verify_base_position);
                    } else {
                        rebuild_accepted_mtp_prefix(replay, verify_base_position);
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
                }
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
        if (prefill_chunks > 1) {
            msg << " prefill_chunks=" << prefill_chunks
                << " prefill_chunk_size=" << prefill_chunk_size;
        }
        msg << " prefill_ops=" << prefill_ops << " decode_ops=" << decode_ops;
        log(msg.str());
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
    std::unique_ptr<QwenWeights> weights_;
    std::unique_ptr<QwenExecutor> executor_;
    std::unique_ptr<QwenTokenizer> tokenizer_;
    std::unique_ptr<BatchedPrefillExecutor> cb_prefill_executor_;
    std::unique_ptr<BatchedDecodeExecutor> cb_decode_executor_;
    std::unique_ptr<GlobalKvPagePool> cb_kv_pool_;
    std::unique_ptr<GlobalKvPagePool> cb_mtp_kv_pool_;
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

} // namespace qw3
