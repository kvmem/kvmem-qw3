#include "backend.hpp"
#include "env_flags.hpp"
#include "qwen_executor.hpp"
#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw3 {
namespace {

double wall_seconds() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
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
        min_depth = std::min<uint32_t>(
            max_depth,
            env_uint32_or("QW3_MTP_ADAPTIVE_MIN_CHAIN", 1));
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

        // Non-MTP greedy decode keeps qw3's native graph-capture / FlashInfer
        // / internal-chunking path byte-for-byte. MTP machinery is purely
        // additive: only the active_mtp branch differs.
        if (!active_mtp) {
            return generate_plain(prompt_tokens, options, on_text, dump.get());
        }
        return generate_mtp(prompt_tokens, options, on_text, dump.get(),
                            spec_mtp, trace_mtp);
    }

private:
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

        std::string generated;
        const int32_t eos = tokenizer_->eos_id();
        uint32_t next_token = step.argmax_token >= 0 ? static_cast<uint32_t>(step.argmax_token)
                                                     : static_cast<uint32_t>(eos);
        uint64_t decode_ops = 0;
        int decoded = 0;
        if (options.max_tokens > 0 && next_token != static_cast<uint32_t>(eos)) {
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(next_token));
            generated += piece;
            if (on_text && !piece.empty()) on_text(piece);
            ++decoded;
        }
        for (int i = 0; i + 1 < options.max_tokens; ++i) {
            if (next_token == static_cast<uint32_t>(eos)) break;
            const uint32_t feed = next_token;
            step = executor_->forward_one_token(feed);
            if (!step.ok) throw std::runtime_error("decode failed");
            decode_ops += step.ops_executed;
            const int32_t new_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
            if (dump) dump->record(static_cast<int>(prompt_tokens.size() + i),
                                   "decode", static_cast<int32_t>(feed),
                                   *executor_, *tokenizer_);
            next_token = static_cast<uint32_t>(new_argmax);
            if (next_token == static_cast<uint32_t>(eos)) break;
            const std::string piece = tokenizer_->decode_one(new_argmax);
            generated += piece;
            if (on_text && !piece.empty()) on_text(piece);
            ++decoded;
        }
        const double t_decode_end = wall_seconds();

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
                             bool trace_mtp) {
        DeviceStatus st = device_->begin();
        if (!st.ok) throw std::runtime_error(st.message);
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
        uint32_t next_token = step.argmax_token >= 0 ? static_cast<uint32_t>(step.argmax_token)
                                                     : static_cast<uint32_t>(eos);
        uint64_t decode_ops = 0;
        std::unordered_map<std::string, TraceStats> decode_trace;
        uint64_t decode_trace_steps = 0;
        const bool trace_mtp_verify = mtp_verify_trace_enabled();
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

        auto emit_generated_token = [&](uint32_t token) -> bool {
            if (decoded >= options.max_tokens || token == static_cast<uint32_t>(eos)) return false;
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(token));
            generated += piece;
            if (on_text && !piece.empty()) on_text(piece);
            ++decoded;
            return true;
        };

        if (options.max_tokens > 0 && next_token != static_cast<uint32_t>(eos)) {
            if (emit_generated_token(next_token) && trace_mtp && !run_spec_mtp) {
                trace_mtp_chain(next_token, decoded - 1);
            }
        }

        uint64_t plain_decode_forwards = 0;
        const bool decode_as_batch = decode_as_batch_enabled();
        auto run_plain_decode_remaining = [&]() {
            while (decoded < options.max_tokens && next_token != static_cast<uint32_t>(eos)) {
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
                next_token = static_cast<uint32_t>(new_argmax);
                if (!emit_generated_token(next_token)) break;
                if (trace_mtp && decoded < options.max_tokens) {
                    trace_mtp_chain(next_token, decoded - 1);
                }
            }
        };

        if (run_spec_mtp) {
            while (run_spec_mtp &&
                   decoded < options.max_tokens &&
                   next_token != static_cast<uint32_t>(eos)) {
                const uint32_t current = next_token;
                const uint32_t remaining_tokens =
                    static_cast<uint32_t>(options.max_tokens - decoded);
                const uint32_t draft_limit =
                    mtp_policy.draft_limit(remaining_tokens, mtp_chain_len);
                const double t_draft_start = mtp_phase_time();
                std::vector<NativeExecutorReport> chain =
                    executor_->forward_mtp_draft_chain_with_prefix(current, draft_limit);
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
                        mtp.argmax_token == eos) {
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
};

} // namespace

std::unique_ptr<Backend> make_qwen_native_backend() {
    return std::make_unique<QwenNativeBackend>();
}

} // namespace qw3
