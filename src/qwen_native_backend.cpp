#include "backend.hpp"
#include "env_flags.hpp"
#include "kvmem_archive_cli.hpp"
#include "kvmem_session.hpp"
#include "qwen_executor.hpp"
#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/tokenizer.hpp"

#ifdef QW3_ENABLE_CUDA
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif

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

class ScopedKvMemSemanticBudget {
public:
    ScopedKvMemSemanticBudget(QwenExecutor *executor, uint32_t requested)
        : executor_(executor) {
        if (!executor_ || !executor_->kvmem_enabled()) {
            if (requested != 0) {
                throw std::invalid_argument(
                    "KVMem request semantic budget requires KVMem");
            }
            executor_ = nullptr;
            return;
        }
        previous_ = executor_->kvmem_set_runtime_select_budget(requested);
        active_ = true;
    }

    ScopedKvMemSemanticBudget(const ScopedKvMemSemanticBudget &) = delete;
    ScopedKvMemSemanticBudget &operator=(
        const ScopedKvMemSemanticBudget &) = delete;

    ~ScopedKvMemSemanticBudget() {
        if (!active_) return;
        try {
            (void)executor_->kvmem_set_runtime_select_budget(previous_);
        } catch (...) {
            // Both values were validated when installed. Destructors must not
            // turn an unrelated request exception into std::terminate.
        }
    }

private:
    QwenExecutor *executor_ = nullptr;
    uint32_t previous_ = 0;
    bool active_ = false;
};

std::string trim_archive_question_ascii_ws(const std::string &text) {
    const char *ws = " \t\n\r\f\v";
    const size_t begin = text.find_first_not_of(ws);
    if (begin == std::string::npos) return {};
    const size_t end = text.find_last_not_of(ws);
    return text.substr(begin, end - begin + 1);
}

std::string archive_json_escape(const std::string &text) {
    std::ostringstream out;
    for (unsigned char c : text) {
        switch (c) {
            case '\"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<unsigned>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

struct ArchiveTokenSpan {
    uint32_t begin = 0;
    uint32_t end = 0;
};

ArchiveTokenSpan archive_removed_text_token_span(
        const std::vector<int32_t> &full,
        const std::vector<int32_t> &removed,
        const char *label) {
    size_t begin = 0;
    const size_t common_limit = std::min(full.size(), removed.size());
    while (begin < common_limit && full[begin] == removed[begin]) ++begin;
    size_t suffix = 0;
    while (suffix < full.size() - begin &&
           suffix < removed.size() - begin &&
           full[full.size() - 1 - suffix] ==
               removed[removed.size() - 1 - suffix]) {
        ++suffix;
    }
    const size_t end = full.size() - suffix;
    if (end <= begin ||
        end > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            std::string("archive ") + label +
            " maps to an empty or oversized token span");
    }
    return {static_cast<uint32_t>(begin), static_cast<uint32_t>(end)};
}

std::string render_archive_question(
        const std::string &content,
        KvMemArchiveRunConfig::QuestionFormat format,
        bool *thinking_open = nullptr) {
    if (thinking_open) {
        *thinking_open = format ==
            KvMemArchiveRunConfig::QuestionFormat::QwenChatThinking;
    }
    if (format == KvMemArchiveRunConfig::QuestionFormat::Raw) {
        return content;
    }
    const bool user_continuation = format ==
        KvMemArchiveRunConfig::QuestionFormat::QwenUserContinuationNoThinking;
    const std::string prefix = user_continuation
        ? "\n"
        : "<|im_start|>user\n";
    const std::string suffix = format ==
            KvMemArchiveRunConfig::QuestionFormat::QwenChatThinking
        ? "<|im_end|>\n<|im_start|>assistant\n<think>\n"
        : "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    // Normal Chat Completions mirrors the server's ASCII trim. A continuation
    // is the remainder of an already-open canonical user message, so preserve
    // its leading/trailing bytes exactly.
    const std::string rendered_content = user_continuation
        ? content
        : trim_archive_question_ascii_ws(content);
    return prefix + rendered_content + suffix;
}

uint32_t kvmem_effective_replay_begin(const GenerationOptions &options) {
    return options.kvmem_replay_end > options.kvmem_replay_begin
        ? options.kvmem_replay_begin
        : options.kvmem_query_begin;
}

// Query replay only requires a fixed-stride, position-invariant Mean-K source
// index. Sub-block Mean-K has the same property: it stores a fixed number of
// equal sub-block means per physical block and differs only in boundary scoring.
// Per-token and DeltaNet retrieval still use incompatible capture/state layouts.
bool kvmem_query_replay_retrieval_supported(KvMemRetrievalMethod method) {
    return method == KvMemRetrievalMethod::MeanK ||
           method == KvMemRetrievalMethod::SubBlockMeanK;
}

// Query-conditioned KVMem has two independent lifecycle decisions:
//
//   prepare       capture Q and incrementally maintain the position-invariant
//                 source index for a future warm request;
//   select_active perform semantic selection/replay for the current request.
//
// A prefix-cached request just below the selection budget needs `prepare=true`
// even though selection must remain dense/identity. Its decode can cross the
// threshold, and the next strict-extension request then needs both artifacts to
// restore a post-query P/M checkpoint without rebuilding the complete history.
struct KvmemQueryLifecycle {
    bool requested = false;
    bool prepare = false;
    bool select_active = false;

    constexpr bool capture_active() const { return prepare || select_active; }
};

constexpr KvmemQueryLifecycle kvmem_query_lifecycle(
        bool valid_query_span, uint32_t logical_prompt_tokens,
        uint32_t selection_budget, bool reselection_enabled,
        bool warm_capture, bool source_index_supported) {
    KvmemQueryLifecycle out;
    out.requested = valid_query_span && selection_budget > 0 &&
        reselection_enabled;
    out.prepare = out.requested && warm_capture && source_index_supported;
    out.select_active = out.requested &&
        logical_prompt_tokens > selection_budget;
    return out;
}

static_assert(kvmem_query_lifecycle(
    true, 229282, 229376, true, true, true).prepare);
static_assert(!kvmem_query_lifecycle(
    true, 229282, 229376, true, true, true).select_active);
static_assert(kvmem_query_lifecycle(
    true, 229574, 229376, true, true, true).select_active);
static_assert(!kvmem_query_lifecycle(
    true, 229282, 229376, false, true, true).capture_active());

// Vector-position counterpart of the executor's RoPE range diagnostic. The
// continuous-batching paths feed ragged position arrays directly to CUDA, so
// inspect the host mirror immediately before each Q/K RoPE launch pair. One
// line summarizes one actual kernel pair without aborting or altering inputs.
void trace_rope_positions_if_out_of_range(const char *source,
                                          const std::vector<int32_t> &positions,
                                          uint32_t limit,
                                          int32_t layer,
                                          uint32_t kernel_uses = 1) {
    static const bool enabled = env_flag_enabled("QW3_ROPE_POSITION_TRACE");
    if (!enabled || positions.empty() || limit == 0) return;
    int32_t min_pos = std::numeric_limits<int32_t>::max();
    int32_t max_pos = std::numeric_limits<int32_t>::min();
    uint32_t out_of_range = 0;
    for (int32_t pos : positions) {
        min_pos = std::min(min_pos, pos);
        max_pos = std::max(max_pos, pos);
        if (pos < 0 || static_cast<uint32_t>(pos) >= limit) ++out_of_range;
    }
    if (out_of_range == 0) return;
    static std::atomic<uint64_t> sequence{0};
    const uint64_t seq = sequence.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[rope-position-oob] seq=cb-%llu source=%s min=%d max=%d "
                 "oob_rows=%u rows=%zu limit=%u layer=%d uses=%u\n",
                 static_cast<unsigned long long>(seq), source, min_pos, max_pos,
                 out_of_range, positions.size(), limit, layer, kernel_uses);
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
    // Softmax denominator over the full vocab, in one pass (no per-token storage).
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::exp((logits[i] - maxv) * inv_t);
    const float norm = static_cast<float>(1.0 / (sum > 0.0 ? sum : 1.0));
    auto prob_of = [&](int i) -> float {
        return std::exp((logits[i] - maxv) * inv_t) * norm;
    };
    const bool need_sort =
        (top_k > 0 && top_k < n) ||
        (top_p < 1.0f && top_p > 0.0f) ||
        (min_p > 0.0f);
    if (!need_sort) {
        // No truncation: emit the full-vocab softmax directly.
        for (int i = 0; i < n; ++i) out[i] = prob_of(i);
        return out;
    }
    // Truncation only needs the highest-probability tokens, so avoid the O(V log V)
    // full sort: isolate the top candidates with O(V) partial selection and sort
    // only that window. exp is monotonic, so ranking by logit equals ranking by
    // probability; the kept set + renormalization below is identical to sorting the
    // whole vocab. If a filter would keep the entire window, widen it and retry —
    // that bounds the answer exactly. Truncation order matches the original
    // sample_token: top_k, then min_p, then nucleus (top_p); each keeps a prefix.
    std::vector<int> idx(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) idx[i] = i;
    auto by_logit_desc = [&](int a, int b) { return logits[a] > logits[b]; };
    size_t cap = static_cast<size_t>(n);
    if (top_k > 0 && static_cast<size_t>(top_k) < cap) cap = static_cast<size_t>(top_k);
    size_t K = std::min<size_t>(cap, 128);
    if (K == 0) K = std::min<size_t>(cap, 1);
    size_t keep = 0;
    for (;;) {
        if (K > cap) K = cap;
        std::nth_element(idx.begin(), idx.begin() + K, idx.end(), by_logit_desc);
        std::sort(idx.begin(), idx.begin() + K, by_logit_desc);
        keep = K;
        if (top_k > 0 && static_cast<size_t>(top_k) < keep) keep = static_cast<size_t>(top_k);
        if (min_p > 0.0f && keep > 0) {
            const float cutoff = prob_of(idx[0]) * min_p;
            size_t k = 0;
            while (k < keep && prob_of(idx[k]) >= cutoff) ++k;
            keep = k;
        }
        if (top_p < 1.0f && top_p > 0.0f) {
            double cum = 0.0;
            size_t k = keep;
            for (size_t i = 0; i < keep; ++i) {
                cum += prob_of(idx[i]);
                if (cum >= top_p) { k = i + 1; break; }
            }
            keep = k;
        }
        // Exact once the kept set fits strictly inside the window (or the window
        // already spans the whole eligible vocab / top_k cap).
        if (keep < K || K >= cap) break;
        K = std::min<size_t>(cap, K * 2);
    }
    double kept_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) kept_sum += prob_of(idx[i]);
    if (kept_sum <= 0.0) {
        // Degenerate (e.g. min_p removed everything): collapse to the top token.
        out[idx[0]] = 1.0f;
        return out;
    }
    const float kn = static_cast<float>(1.0 / kept_sum);
    for (size_t i = 0; i < keep; ++i) out[idx[i]] = prob_of(idx[i]) * kn;
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

// kvmem single-request (non-CB) prefix cache. When on, the plain-route shared
// executor is kept warm across requests: if a new prompt begins with the entire
// token sequence of the previous request (prompt + generated response), the
// executor resumes at that end (via a captured StateSnapshot) and only the new
// suffix is prefilled. Default off -> reset_state() every request, byte-identical
// to today. Distinct from the CB-only QW3_PREFIX_CACHE above.
bool kvmem_prefix_cache_enabled() {
    return env_flag_enabled("QW3_KVMEM_PREFIX_CACHE");
}

bool kvmem_prefix_cache_trace_enabled() {
    return env_flag_enabled("QW3_KVMEM_PREFIX_CACHE_TRACE");
}

// Clean-query prefill (task #50). When on (and the turn is above budget with a
// query span), the query embedding used to SELECT context blocks is captured from
// the question tokens prefilled IN ISOLATION (PASS A), removing the recency bias
// the mid-prefill window otherwise bakes into it. Default off -> byte-identical.
bool kvmem_clean_query_enabled() {
    return env_flag_enabled("QW3_KVMEM_CLEAN_QUERY");
}

// Query replay is the default query-conditioned mean-K behavior: use the first
// pass to select blocks, then rewind to the query's aligned boundary and prefill
// that short suffix against the fixed final window. The environment override is
// retained only for legacy ablations; EngineOptions defaults this to true.
bool kvmem_recompute_query_enabled(bool configured_default) {
    return env_flag_enabled("QW3_KVMEM_RECOMPUTE_QUERY",
                            configured_default);
}

const char *kvmem_inline_refresh_name(KvMemInlineRefreshMode mode) {
    switch (mode) {
    case KvMemInlineRefreshMode::Off: return "off";
    case KvMemInlineRefreshMode::KvOnly: return "kv_only";
    case KvMemInlineRefreshMode::KvAndState: return "kv_and_state";
    }
    return "unknown";
}

class ScopedKvmemDisable {
public:
    explicit ScopedKvmemDisable(QwenExecutor *executor)
        : executor_(executor),
          was_enabled_(executor_ && executor_->kvmem_enabled()) {
        if (executor_) executor_->set_kvmem_enabled(false);
    }
    ~ScopedKvmemDisable() {
        if (executor_) executor_->set_kvmem_enabled(was_enabled_);
    }
    ScopedKvmemDisable(const ScopedKvmemDisable &) = delete;
    ScopedKvmemDisable &operator=(const ScopedKvmemDisable &) = delete;

private:
    QwenExecutor *executor_ = nullptr;
    bool was_enabled_ = false;
};

std::vector<uint32_t> kvmem_gather_selected_source_tokens(
    const std::vector<uint32_t> &prompt_tokens,
    const std::vector<KvMemBlock> &blocks,
    const std::vector<uint32_t> &selected_context) {
    std::vector<uint32_t> out;
    uint64_t total = 0;
    uint32_t previous_end = 0;
    bool first = true;
    for (uint32_t id : selected_context) {
        if (id >= blocks.size()) {
            throw std::runtime_error(
                "KVMem inline refresh selection contains an invalid block id");
        }
        const KvMemBlock &block = blocks[id];
        const uint64_t end = static_cast<uint64_t>(block.orig_pos_start) +
            block.n_tokens;
        if (end > prompt_tokens.size()) {
            throw std::runtime_error(
                "KVMem inline refresh block exceeds the prompt tokens");
        }
        if (!first && block.orig_pos_start < previous_end) {
            throw std::runtime_error(
                "KVMem inline refresh blocks are not in source order");
        }
        first = false;
        previous_end = static_cast<uint32_t>(end);
        total += block.n_tokens;
    }
    if (total > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(
            "KVMem inline refresh token count overflows size_t");
    }
    out.reserve(static_cast<size_t>(total));
    for (uint32_t id : selected_context) {
        const KvMemBlock &block = blocks[id];
        out.insert(
            out.end(),
            prompt_tokens.begin() +
                static_cast<std::ptrdiff_t>(block.orig_pos_start),
            prompt_tokens.begin() +
                static_cast<std::ptrdiff_t>(block.orig_pos_start +
                                            block.n_tokens));
    }
    return out;
}

// ARCHIVED (2026-07-23): helper code for the DeltaNet recurrent-state artifact
// experiment. The API entry and executor interchange methods are also compiled
// out; retain this block only as local source history.
#if 0
std::string kvmem_rebuilt_state_path(const std::string &key) {
    if (key.empty()) return {};
    const char *dir = std::getenv("QW3_KVMEM_REBUILT_STATE_DIR");
    if (!dir || !*dir) {
        throw std::runtime_error(
            "KVMem rebuilt-state request requires "
            "QW3_KVMEM_REBUILT_STATE_DIR");
    }
    std::string base(dir);
    while (!base.empty() && base.back() == '/') base.pop_back();
    if (base.empty()) {
        throw std::runtime_error(
            "QW3_KVMEM_REBUILT_STATE_DIR must not be the filesystem root");
    }
    return base + "/" + key + ".qw3-deltanet-state";
}

std::vector<uint32_t> kvmem_selected_source_tokens(
    const std::vector<uint32_t> &prompt_tokens,
    const std::vector<KvMemBlock> &blocks,
    const std::vector<uint32_t> &selected_context) {
    std::vector<uint32_t> out;
    uint64_t total = 0;
    uint32_t previous_end = 0;
    bool first = true;
    for (uint32_t id : selected_context) {
        if (id >= blocks.size()) {
            throw std::runtime_error(
                "KVMem rebuilt-state selection contains an invalid block id");
        }
        const KvMemBlock &block = blocks[id];
        const uint64_t end = static_cast<uint64_t>(block.orig_pos_start) +
            block.n_tokens;
        if (end > prompt_tokens.size()) {
            throw std::runtime_error(
                "KVMem rebuilt-state selected block exceeds the prompt tokens");
        }
        if (!first && block.orig_pos_start < previous_end) {
            throw std::runtime_error(
                "KVMem rebuilt-state selected blocks are not in source order");
        }
        first = false;
        previous_end = static_cast<uint32_t>(end);
        total += block.n_tokens;
    }
    if (total > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(
            "KVMem rebuilt-state selected token count overflows size_t");
    }
    out.reserve(static_cast<size_t>(total));
    for (uint32_t id : selected_context) {
        const KvMemBlock &block = blocks[id];
        out.insert(out.end(),
                   prompt_tokens.begin() + block.orig_pos_start,
                   prompt_tokens.begin() + block.orig_pos_start +
                       block.n_tokens);
    }
    return out;
}
#endif

// Query-only checkpoint/replay. The context is prefilled once, the final user
// query is run provisionally to capture its Q rows, then replayed from its
// boundary under the globally retrieved working set. Default off.
bool kvmem_query_replay_enabled() {
    return env_flag_enabled("QW3_KVMEM_QUERY_REPLAY");
}

std::vector<uint32_t> kvmem_selected_block_ids(const QwenExecutor *executor) {
    std::vector<uint32_t> ids;
    if (!executor || !executor->block_store()) return ids;
    for (const KvMemBlock &block : executor->block_store()->blocks()) {
        if (block.in_working_set) ids.push_back(block.block_id);
    }
    return ids;
}

// A pre-pressure identity window has no blocks marked `in_working_set`, even
// though every registered block is active. A restorable checkpoint must make
// that implicit identity state explicit; otherwise a frozen query's semantic
// selection would leak into the next branch when the saved vector is empty.
std::vector<uint32_t> kvmem_checkpoint_block_ids(
        const QwenExecutor *executor) {
    std::vector<uint32_t> ids = kvmem_selected_block_ids(executor);
    if (!ids.empty() || !executor || !executor->block_store()) return ids;
    const auto &blocks = executor->block_store()->blocks();
    ids.reserve(blocks.size());
    for (const KvMemBlock &block : blocks) ids.push_back(block.block_id);
    return ids;
}

uint32_t kvmem_selection_additions(const std::vector<uint32_t> &before,
                                   const std::vector<uint32_t> &after) {
    uint32_t additions = 0;
    size_t i = 0;
    size_t j = 0;
    while (j < after.size()) {
        while (i < before.size() && before[i] < after[j]) ++i;
        if (i == before.size() || before[i] != after[j]) ++additions;
        ++j;
    }
    return additions;
}

struct KvMemSemanticChunkStats {
    uint32_t chunks = 0;
    uint64_t provisional_tokens = 0;
    uint64_t replay_tokens = 0;
    double provisional_s = 0.0;
    double rollback_s = 0.0;
    double reselect_s = 0.0;
    double replay_s = 0.0;
};

// Run a history range through query-conditioned KVMem one physical prefill
// chunk at a time. The first pass captures only Q; all durable KV/raw-K/index
// state is produced by the replay after the semantic window is installed.
// `prefill` must execute exactly [begin,end), including MTP-prefix priming when
// MTP is enabled. Keeping that policy in the caller makes this orchestration
// identical for plain and MTP generation.
KvMemSemanticChunkStats kvmem_prefill_semantic_chunks(
        QwenExecutor *executor,
        uint32_t prompt_tokens,
        uint32_t begin,
        uint32_t end,
        uint32_t configured_start,
        uint32_t query_token_cap,
        const std::function<void(uint32_t, uint32_t, bool)> &prefill) {
    KvMemSemanticChunkStats stats;
    if (!executor || !executor->kvmem_enabled() || begin >= end) return stats;
    const KvMemStore *store = executor->block_store();
    if (!store) {
        throw std::runtime_error(
            "semantic-chunk prefill requires a KVMem block store");
    }
    if (!kvmem_query_replay_retrieval_supported(
            store->config().retrieval_method)) {
        throw std::runtime_error(
            "semantic-chunk prefill requires Mean-K/SubBlockMeanK retrieval");
    }
    const uint32_t bt = std::max<uint32_t>(1, store->config().block_tokens);
    const uint32_t gen_budget = store->config().gen_budget;
    uint32_t start = configured_start;
    if (start == 0) {
        start = store->config().prefill_budget != 0
            ? store->config().prefill_budget
            : store->select_budget_tokens();
    }
    start = ((start + bt - 1) / bt) * bt;
    if (begin % bt != 0 || end % bt != 0) {
        throw std::runtime_error(
            "semantic-chunk history boundaries must be block aligned");
    }

    uint32_t cursor = begin;
    if (cursor < std::min(start, end)) {
        const uint32_t dense_end = std::min(start, end);
        prefill(cursor, dense_end, /*compute_final_logits=*/false);
        executor->kvmem_register_append(dense_end - cursor);
        cursor = dense_end;
    }
    if (cursor >= end) return stats;

    QwenExecutor::StateSnapshot boundary;
    while (cursor < end) {
        const uint32_t remaining = end - cursor;
        uint32_t width = std::max<uint32_t>(
            bt, executor->effective_prefill_chunk_size(remaining));
        width = std::min(width, remaining);
        if (width < remaining) width -= width % bt;
        if (width == 0 || width % bt != 0) {
            throw std::runtime_error(
                "semantic-chunk prefill produced a non-aligned chunk");
        }
        if (gen_budget != 0 && width > gen_budget) {
            throw std::runtime_error(
                "semantic-chunk width exceeds the generation headroom");
        }
        const uint32_t chunk_begin = cursor;
        const uint32_t chunk_end = cursor + width;
        const uint32_t query_begin = query_token_cap == 0
            ? chunk_begin
            : std::max(chunk_begin, chunk_end - query_token_cap);

        // At the first pressure boundary every historical block still fits in
        // the selection budget, so query scoring cannot change the result.
        // Install that identity window and commit the chunk once; the ordinary
        // two-pass semantic path starts only after history has a real choice.
        if (store->block_count() <= store->budget_blocks()) {
            executor->kvmem_set_query_span(
                0, 0, prompt_tokens,
                /*index_tokens=*/chunk_end,
                /*preserve_content_index=*/true,
                /*capture_content_without_query=*/true);
            executor->capture_transient_state(boundary);
            const double select_start = wall_seconds();
            executor->kvmem_reselect_prefill_pressure();
            stats.reselect_s += wall_seconds() - select_start;
            const std::vector<uint32_t> selected =
                kvmem_selected_block_ids(executor);

            const double replay_start = wall_seconds();
            executor->kvmem_begin_query_replay(
                boundary, selected,
                /*reset_recurrent_state=*/false,
                /*preserve_selected_context=*/true);
            try {
                prefill(chunk_begin, chunk_end,
                        /*compute_final_logits=*/false);
            } catch (...) {
                executor->kvmem_end_query_replay();
                throw;
            }
            executor->kvmem_end_query_replay();
            executor->kvmem_prefill_writeback(chunk_end);
            stats.replay_s += wall_seconds() - replay_start;
            stats.replay_tokens += width;
            ++stats.chunks;
            cursor = chunk_end;

            if (std::getenv("QW3_KVMEM_TRACE")) {
                std::fprintf(
                    stderr,
                    "[kvmem-semantic-chunk] event=%u range=[%u,%u) "
                    "bootstrap=identity selected=%zu position=%u\n",
                    stats.chunks, chunk_begin, chunk_end, selected.size(),
                    executor->position());
            }
            continue;
        }

        // Reuse the fixed session index allocation and publish only historical
        // blocks at selection time. The provisional pass is query-only, so it
        // cannot append temporary Adaptive prototypes or tier writes.
        const double provisional_start = wall_seconds();
        executor->kvmem_set_provisional_prefill(true);
        executor->kvmem_set_query_span(
            query_begin, chunk_end, prompt_tokens,
            /*index_tokens=*/end,
            /*preserve_content_index=*/true);
        executor->capture_transient_state(boundary);
        (void)executor->kvmem_begin_provisional_adaptive_score();
        executor->kvmem_set_prefill_reselect_suppressed(true);
        try {
            prefill(chunk_begin, chunk_end,
                    /*compute_final_logits=*/false);
        } catch (...) {
            executor->kvmem_set_prefill_reselect_suppressed(false);
            executor->kvmem_set_provisional_prefill(false);
            throw;
        }
        executor->kvmem_set_prefill_reselect_suppressed(false);
        executor->kvmem_set_provisional_prefill(false);
        if (!executor->kvmem_stash_query(/*host_in_place=*/true)) {
            throw std::runtime_error(
                "semantic-chunk provisional pass did not capture its query");
        }
        // Stashing also drains the asynchronous query D2H capture. Include it
        // in the provisional phase so semantic-chunk accounting covers the
        // complete exposed critical path rather than leaving an unreported
        // gap between provisional forward and rollback.
        stats.provisional_s += wall_seconds() - provisional_start;
        stats.provisional_tokens += width;

        const double rollback_start = wall_seconds();
        executor->restore_state(boundary);
        // Provisional mode did not register blocks, capture raw K/index rows, or
        // start tier writes. restore_state() already drops its target/MTP pages
        // and recurrent suffix, so the general kvmem_truncate_to() would only
        // add an unnecessary global CPU/NVMe write drain here.
        executor->kvmem_set_query_span(
            query_begin, chunk_end, prompt_tokens,
            /*index_tokens=*/chunk_begin,
            /*preserve_content_index=*/true);
        executor->kvmem_restore_stashed_query();
        if (!executor->kvmem_publish_captured_prefix(chunk_begin)) {
            throw std::runtime_error(
                "semantic-chunk could not publish the historical index");
        }
        stats.rollback_s += wall_seconds() - rollback_start;

        // The provisional chunk was removed before selection. Consequently all
        // select-budget slots belong to historical context; the replayed chunk
        // uses the ordinary generation/headroom pages instead of diluting the
        // retrieved history.
        executor->kvmem_set_pin_from_block(0xffffffffu);
        const double select_start = wall_seconds();
        executor->kvmem_reselect();
        stats.reselect_s += wall_seconds() - select_start;
        const std::vector<uint32_t> selected =
            kvmem_selected_block_ids(executor);
        for (uint32_t id : selected) {
            if (id >= store->block_count() ||
                store->blocks()[id].orig_pos_start >= chunk_begin) {
                throw std::runtime_error(
                    "semantic-chunk selected a non-historical block");
            }
        }

        const double replay_start = wall_seconds();
        executor->kvmem_begin_query_replay(
            boundary, selected,
            /*reset_recurrent_state=*/false,
            /*preserve_selected_context=*/true);
        // The final pass appends the chunk's durable Adaptive/Mean-K rows while
        // no longer retaining its now-consumed query capture.
        executor->kvmem_set_query_span(
            0, 0, prompt_tokens,
            /*index_tokens=*/chunk_end,
            /*preserve_content_index=*/true,
            /*capture_content_without_query=*/true);
        try {
            prefill(chunk_begin, chunk_end,
                    /*compute_final_logits=*/false);
        } catch (...) {
            executor->kvmem_end_query_replay();
            throw;
        }
        executor->kvmem_end_query_replay();
        // Make the just-committed chunk durable immediately. Its asynchronous
        // D2H can overlap the next provisional target-model pass, so the next
        // selection normally only retires an already-started transfer instead
        // of exposing a synchronous pressure-point stage-out.
        executor->kvmem_prefill_writeback(chunk_end);
        stats.replay_s += wall_seconds() - replay_start;
        stats.replay_tokens += width;
        ++stats.chunks;
        cursor = chunk_end;

        if (std::getenv("QW3_KVMEM_TRACE")) {
            std::fprintf(
                stderr,
                "[kvmem-semantic-chunk] event=%u range=[%u,%u) "
                "query=[%u,%u) selected=%zu position=%u\n",
                stats.chunks, chunk_begin, chunk_end, query_begin, chunk_end,
                selected.size(), executor->position());
        }
    }
    return stats;
}

// Cap on total physical KV pages pinned by the prefix cache. 0 = unlimited
// (bounded only by the pool). Used to trigger LRU eviction.
uint32_t prefix_cache_max_pages() {
    return env_uint32_or("QW3_PREFIX_CACHE_MAX_PAGES", 0);
}

// Recurrent/conv and optional MTP-prefix state are independent device
// snapshots per cache entry even when a linear prompt chain shares nearly all
// physical KV pages. A page-only budget therefore cannot bound device memory:
// hundreds of progressively longer entries can pin the same pages while their
// state snapshots accumulate. Keep a separate LRU entry cap. The environment
// override accepts 0 for unlimited legacy behavior; the safe default leaves
// ample room for concurrent prompt branches without allowing an unbounded
// state-snapshot chain.
uint32_t prefix_cache_max_entries() {
    return env_uint32_or("QW3_PREFIX_CACHE_MAX_ENTRIES", 64);
}

// Leave this many complete KV pages behind the natural prompt boundary when
// publishing a shared prefix.  The default remains zero for existing users.
// A one-page guard is useful for independently rendered chat requests: BPE
// tokenization close to the user-content suffix can change when a question is
// appended, even if the byte prefix (including its separator) is identical.
// Publishing one page earlier keeps the cache strictly inside the stable
// token prefix while still reusing essentially the entire long context.
uint32_t prefix_cache_commit_guard_pages() {
    return std::min<uint32_t>(
        16u, env_uint32_or("QW3_PREFIX_CACHE_COMMIT_GUARD_PAGES", 0));
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

// Where to stage the resumable prompt-end (P) checkpoint for kvmem prefix
// reuse. The chat template rewrites the final few tokens each turn: the empty
// "<think>\n\n</think>\n\n" block the generation prompt injects for the LIVE
// turn is dropped once that turn becomes history, so next turn's
// longest-common-prefix D lands ~4 tokens below the raw prompt end (observed
// D == P-4). Capturing ckpt_P strictly below that rewrite makes the reuse fire:
//   - Long prompt (> block_tokens): the last block boundary below P; if that
//     boundary sits inside the tail-rewrite zone (P just above a boundary),
//     step back one block so D >= split still holds.
//   - Short prompt (<= block_tokens): there is no block boundary below P, so
//     stage a stable point kTailGuard tokens below P.
// Returns 0 when no useful split exists (nothing strictly between the resume
// base and P), in which case the caller captures at P as before.
uint32_t kvmem_prompt_checkpoint_split(uint32_t prompt_end,
                                       uint32_t prefill_base,
                                       uint32_t block_tokens) {
    constexpr uint32_t kTailGuard = 4;  // empty <think> block dropped from history
    if (prompt_end <= prefill_base) return 0;
    const uint32_t bt = std::max<uint32_t>(1, block_tokens);
    uint32_t split;
    if (prompt_end > bt) {
        split = ((prompt_end - 1) / bt) * bt;  // last block start < P
        // Step back a block while the boundary lands inside the tail rewrite.
        while (split > prefill_base && split + kTailGuard > prompt_end) {
            if (split < bt) { split = 0; break; }
            split -= bt;
        }
    } else {
        split = (prompt_end > kTailGuard) ? (prompt_end - kTailGuard) : 0;
    }
    if (split <= prefill_base) return 0;  // must beat the resume base to save work
    if (split >= prompt_end) return 0;
    return split;
}

// A query-conditioned warm prefix may only resume at a checkpoint from which
// the block-aligned query replay boundary can still be reached by forward
// prefill. Restoring a later checkpoint loses both the missing query rows and
// the recurrent/KV state needed to replay them; recurrent state is not
// reversible. Below-budget/no-query turns keep the ordinary LCP ceiling.
constexpr uint32_t kvmem_prefix_resume_ceiling(uint32_t common_prefix,
                                               bool query_conditioned,
                                               uint32_t query_begin,
                                               uint32_t block_tokens) {
    if (!query_conditioned) return common_prefix;
    const uint32_t bt = block_tokens == 0 ? 1 : block_tokens;
    const uint32_t replay_boundary = (query_begin / bt) * bt;
    return replay_boundary < common_prefix ? replay_boundary : common_prefix;
}

constexpr bool kvmem_prefix_checkpoint_reusable(uint32_t checkpoint,
                                                uint32_t prompt_tokens,
                                                uint32_t resume_ceiling) {
    return checkpoint <= resume_ceiling && checkpoint < prompt_tokens;
}

// A durable query snapshot makes a checkpoint *after* the historical query
// reusable as well.  The checkpoint must be at/after the complete query (never
// in its middle), remain inside the exact token LCP, and leave at least one new
// token to prefill.  Callers still require a tier-backed source index before
// enabling this path; this helper only expresses the positional invariant.
constexpr bool kvmem_post_query_checkpoint_reusable(
        uint32_t checkpoint, uint32_t prompt_tokens,
        uint32_t common_prefix, uint32_t query_end,
        bool query_snapshot_ready) {
    return query_snapshot_ready && checkpoint >= query_end &&
        checkpoint <= common_prefix && checkpoint < prompt_tokens;
}

// A sparse prompt checkpoint is useful to a later query-conditioned request
// only when both halves of the historical state survive: the tiered K/V blocks
// and the position-invariant source index. The current request's selection
// state is intentionally not an input here. A prefill-only history has no query
// yet, but it is exactly the
// request that must prepare a reusable source index for the next query.
constexpr bool kvmem_sparse_prompt_checkpoint_resumable(
        bool all_gpu_identity, bool has_tiers, bool source_index_ready,
        bool replay_compatible) {
    return all_gpu_identity ||
        (has_tiers && source_index_ready && replay_compatible);
}

static_assert(kvmem_prefix_resume_ceiling(4096, false, 3000, 32) == 4096);
static_assert(kvmem_prefix_resume_ceiling(4096, true, 3000, 32) == 2976);
static_assert(kvmem_prefix_resume_ceiling(2048, true, 3000, 32) == 2048);
static_assert(kvmem_prefix_resume_ceiling(4096, true, 17, 32) == 0);
static_assert(kvmem_prefix_checkpoint_reusable(2976, 4096, 2976));
static_assert(!kvmem_prefix_checkpoint_reusable(3008, 4096, 2976));
static_assert(!kvmem_prefix_checkpoint_reusable(4096, 4096, 4096));
static_assert(kvmem_post_query_checkpoint_reusable(
    3968, 4096, 4000, 3000, true));
static_assert(!kvmem_post_query_checkpoint_reusable(
    2976, 4096, 4000, 3000, true));
static_assert(!kvmem_post_query_checkpoint_reusable(
    3968, 4096, 3900, 3000, true));
static_assert(!kvmem_post_query_checkpoint_reusable(
    3968, 4096, 4000, 3000, false));
static_assert(kvmem_sparse_prompt_checkpoint_resumable(
    true, false, false, false));
static_assert(kvmem_sparse_prompt_checkpoint_resumable(
    false, true, true, true));
static_assert(!kvmem_sparse_prompt_checkpoint_resumable(
    false, true, false, true));
static_assert(!kvmem_sparse_prompt_checkpoint_resumable(
    false, true, true, false));

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

    // ---- Context archive (docs/kvmem_context_archive_design.md) ----

    // Fields that fix the archive's byte layout or its numerics. Deliberately
    // narrow: retrieval method, budgets, sub-block counts, index placement and
    // every other policy knob are excluded so one archive can serve many
    // different runtime configurations.
    KvMemArchiveLayout kvmem_archive_layout(uint32_t block_tokens,
                                            uint32_t raw_chunk_tokens) const {
        const QwenConfig &cfg = model_->config();
        KvMemArchiveLayout l;
        l.architecture = cfg.architecture;
        l.model_name = options_.model_path;
        const size_t slash = l.model_name.find_last_of('/');
        if (slash != std::string::npos) l.model_name = l.model_name.substr(slash + 1);
        std::ifstream probe(options_.model_path,
                            std::ios::binary | std::ios::ate);
        l.model_bytes = probe ? static_cast<uint64_t>(probe.tellg()) : 0;
        l.model_sha256 = kvmem_archive_model_sha256(options_.model_path);
        l.n_layers = cfg.n_layers;
        l.full_attention_interval = cfg.full_attention_interval;
        for (uint32_t i = 0; i < cfg.n_layers; ++i) {
            if (cfg.is_standard_attention_layer(i)) ++l.n_standard_layers;
        }
        l.n_kv_heads = cfg.n_kv_heads;
        l.head_dim = cfg.head_dim;
        l.head_v_dim = cfg.head_v_dim;
        l.rope_dim = cfg.rope_dim;
        l.rope_theta = cfg.rope_theta;
        const char *dtype = std::getenv("QW3_KV_DTYPE");
        l.kv_dtype = dtype ? dtype : "fp16";
        l.block_tokens = block_tokens;
        l.raw_chunk_tokens = raw_chunk_tokens;
        l.kv_page_size = std::max<uint32_t>(
            1, env_uint32_or("QW3_PAGED_KV_PAGE_SIZE", 16));
        l.immutable_source_k = options_.kvmem_immutable_source_k;
        l.raw_k_block_major = true;
        l.mtp_archived = options_.native_mtp_chain_set &&
                         options_.native_mtp_chain > 0;
        const uint64_t elem_bytes = l.kv_dtype == "fp8"
                                        ? 1
                                        : (l.kv_dtype == "fp32" ? 4 : 2);
        l.raw_k_row_bytes =
            static_cast<uint64_t>(l.n_kv_heads) * l.head_dim * elem_bytes;
        l.raw_chunk_bytes = static_cast<uint64_t>(l.n_standard_layers) *
                            raw_chunk_tokens * l.raw_k_row_bytes;
        l.mtp_chunk_bytes = l.mtp_archived
            ? static_cast<uint64_t>(raw_chunk_tokens) * l.raw_k_row_bytes
            : 0;
        const uint32_t archived_v_layers =
            l.n_standard_layers + (l.mtp_archived ? 1u : 0u);
        l.v_block_bytes = static_cast<uint64_t>(archived_v_layers) *
                          block_tokens * l.n_kv_heads * l.head_v_dim *
                          elem_bytes;
        return l;
    }

    // Provenance only. None of this constrains an attach; recording it is what
    // makes an A/B run self-describing after the fact.
    std::string kvmem_archive_policy_snapshot() const {
        std::ostringstream out;
        out << "budget=" << options_.kvmem_budget
            << " prefill_budget="
            << (options_.kvmem_prefill_budget > 0
                    ? options_.kvmem_prefill_budget
                    : options_.kvmem_budget)
            << " gen_budget=" << options_.kvmem_gen_budget
            << " interval=" << options_.kvmem_interval
            << " method=" << options_.kvmem_method
            << " retrieval=" << options_.kvmem_retrieval_method
            << " index_placement=" << options_.kvmem_index_placement
            << " subblocks=" << options_.kvmem_subblocks
            << " ctx=" << options_.ctx_size;
        return out.str();
    }

    void open_kvmem_archive(QwenExecutor &exec) {
        if (options_.kvmem_archive_dir.empty()) return;
        const bool build = options_.kvmem_archive_mode == "build";
        if (!build && options_.kvmem_archive_mode != "attach") {
            throw std::invalid_argument(
                "--kvmem-archive-mode must be build or attach");
        }
        const uint32_t block_tokens =
            static_cast<uint32_t>(std::max(1, options_.kvmem_block_tokens));
        uint32_t raw_chunk_tokens =
            env_uint32_or("QW3_KVMEM_RAW_K_CHUNK_TOKENS", 2048);
        raw_chunk_tokens = std::max(raw_chunk_tokens, block_tokens);
        raw_chunk_tokens =
            ((raw_chunk_tokens + block_tokens - 1) / block_tokens) *
            block_tokens;
        const KvMemArchiveLayout layout =
            kvmem_archive_layout(block_tokens, raw_chunk_tokens);
        // Early v1 archives physically included MTP raw-K/V because archive
        // commands always enable MTP, but their manifest described only the 16
        // main attention layers. Accept that one precisely-defined legacy shape
        // while all newly created v2 archives record the true physical stride.
        auto legacy_v1_layout = [&]() {
            KvMemArchiveLayout legacy = layout;
            legacy.model_sha256.clear();
            legacy.mtp_archived = false;
            legacy.mtp_chunk_bytes = 0;
            if (layout.mtp_archived) {
                const uint64_t mtp_v_bytes =
                    static_cast<uint64_t>(block_tokens) *
                    layout.n_kv_heads * layout.head_v_dim *
                    (layout.kv_dtype == "fp8"
                         ? 1ull
                         : (layout.kv_dtype == "fp32" ? 4ull : 2ull));
                legacy.v_block_bytes -= mtp_v_bytes;
            }
            return legacy;
        };
        auto is_legacy_v1 = [&](const KvMemArchiveManifest &manifest) {
            return manifest.format_version == 1 && layout.mtp_archived &&
                   !manifest.layout.mtp_archived &&
                   manifest.layout.explain_mismatch(
                       legacy_v1_layout()).empty();
        };
        if (build) {
            KvMemArchiveLayout build_layout = layout;
            const std::string manifest_path =
                options_.kvmem_archive_dir + "/" +
                kvmem_archive_files::kManifest;
            std::ifstream existing_manifest(manifest_path);
            if (existing_manifest) {
                const KvMemArchiveManifest existing =
                    KvMemArchive::read_manifest(options_.kvmem_archive_dir);
                if (is_legacy_v1(existing)) build_layout = existing.layout;
            }
            kvmem_archive_ = KvMemArchive::open_for_build(
                options_.kvmem_archive_dir, build_layout,
                kvmem_archive_policy_snapshot());
        } else {
            kvmem_archive_ =
                KvMemArchive::attach(options_.kvmem_archive_dir);
            const KvMemArchiveManifest &manifest =
                kvmem_archive_->manifest();
            const bool legacy_v1 = is_legacy_v1(manifest);
            KvMemArchiveLayout compatible_layout = layout;
            if (manifest.format_version < 3 &&
                manifest.layout.model_sha256.empty()) {
                compatible_layout.model_sha256.clear();
            }
            const std::string why = legacy_v1
                ? std::string()
                : manifest.layout.explain_mismatch(compatible_layout);
            if (!why.empty()) {
                throw std::runtime_error(
                    "KVMem archive layout does not match this engine: " + why);
            }
            if (legacy_v1) {
                auto file_bytes = [](const std::string &path) -> uint64_t {
                    std::ifstream in(path, std::ios::binary | std::ios::ate);
                    return in ? static_cast<uint64_t>(in.tellg()) : 0;
                };
                const uint64_t raw_record =
                    layout.raw_chunk_bytes + layout.mtp_chunk_bytes;
                const uint64_t raw_needed =
                    static_cast<uint64_t>(manifest.raw_chunks) * raw_record;
                const uint64_t v_needed =
                    static_cast<uint64_t>(manifest.total_blocks) *
                    layout.v_block_bytes;
                if (file_bytes(kvmem_archive_->raw_k_path()) < raw_needed ||
                    file_bytes(kvmem_archive_->v_path()) < v_needed) {
                    throw std::runtime_error(
                        "legacy v1 KVMem archive claims no MTP payload and its "
                        "arena sizes do not prove the physical MTP segments");
                }
                log("archive attach: accepted legacy v1 manifest after "
                    "validating physical MTP raw-K/V record strides");
            }
        }
        uint64_t ladder = options_.kvmem_archive_ladder_tokens;
        if (ladder == 0) ladder = static_cast<uint64_t>(raw_chunk_tokens) * 128;
        ladder = std::max<uint64_t>(
            raw_chunk_tokens,
            (ladder / raw_chunk_tokens) * raw_chunk_tokens);
        exec.kvmem_set_archive(kvmem_archive_.get(), build, ladder);
        kvmem_archive_ladder_tokens_ = ladder;
    }

    void load(const EngineOptions &options) override {
        if (options.model_path.empty()) {
            throw std::invalid_argument("qwen-native backend requires --model");
        }
        options_ = options;

        const double t0 = wall_seconds();
        model_ = std::make_unique<QwenNativeModel>(open_model_source(options.model_path));
        if (const GgufFile *gguf = model_->source().gguf()) {
            tokenizer_ = std::make_unique<QwenTokenizer>(*gguf);
        } else {
            tokenizer_ = std::make_unique<QwenTokenizer>(model_->source().model_directory());
        }
        const double t_gguf = wall_seconds();

        // Device backend + weight uploads are now part of load(), not
        // generate(). Subsequent generate() calls reuse the same DeviceBackend
        // and the same on-GPU weight buffers.
        if (options_.native_kernels != "cuda") {
            // mock/cpu kernels are no longer wired here; we still let load()
            // complete so callers that just want to inspect the plan can do so.
            log("native load: model=" + fmt_seconds(t_gguf - t0) +
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
        const bool load_mtp =
            options_.native_mtp_trace || mtp_trace_enabled() ||
            mtp_speculate_enabled(options_) ||
            (options_.native_mtp_chain_set &&
             options_.native_mtp_chain > 0);
        weights_ = std::make_unique<QwenWeights>(
            *model_, *device_, options_.cpu_embedding, load_mtp);
        st = device_->synchronize();
        if (!st.ok) throw std::runtime_error(std::string("weight upload sync failed: ") + st.message);
        const double t_weights = wall_seconds();

        const uint32_t ctx_size = options_.ctx_size > 0 ? static_cast<uint32_t>(options_.ctx_size) : 4096u;
        executor_ = std::make_unique<QwenExecutor>(*model_, *weights_, *device_, ctx_size);
        executor_->set_prefill_chunk_override(options_.prefill_chunk);
        // The archive owns the KV arenas, so it must exist before the tiers are
        // created.
        open_kvmem_archive(*executor_);
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
        const double host_mib =
            static_cast<double>(weights_->host_resident_bytes()) /
            (1024.0 * 1024.0);
        std::ostringstream msg;
        msg << "native load: gguf=" << fmt_seconds(t_gguf - t0)
            << " device_init=" << fmt_seconds(t_begin - t_gguf)
            << " weights_upload=" << fmt_seconds(t_weights - t_begin)
            << " tensors=" << weights_->tensor_count()
            << " size=" << std::fixed << std::setprecision(1) << mib << " MiB"
            << " host_resident=" << host_mib << " MiB"
            << " mtp_weights="
            << (weights_->mtp() ? "loaded"
                                : (load_mtp ? "unavailable" : "skipped"))
            << " backend=" << linear_backend_name(linear_backend);
        log(msg.str());
        if (executor_->kvmem_archive_attached() &&
            env_flag_enabled("QW3_KVMEM_ARCHIVE_SERVE")) {
            KvMemArchiveRunConfig init;
            init.tokens = options_.kvmem_archive_tokens;
            init.verbose = false;
            (void)run_kvmem_archive(init);
            log("archive serve: frozen base ready tokens=" +
                std::to_string(archive_query_base_prefix_tokens_) +
                " selected_blocks=" +
                std::to_string(archive_query_base_selection_.size()));
        }
    }

    // ---- kvmem growth-profiling harness ----------------------------------
    // Drives the persistent context-growth experiment: prefill an initial long
    // context, then keep appending document chunks across turns up to the
    // largest ladder target, sampling a short MTP decode probe at each point.
    // For each turn it prints the user-requested sequential micro-step
    // breakdown (selection / stage-in / stage-out / assemble / prefill /
    // decode) plus a final summary table with tier residency. Requires kvmem
    // enabled (the free run_kvmem_session forces it on with update-mode=step).
    int build_kvmem_archive(const KvMemArchiveBuildConfig &cfg) {
        if (!model_ || !tokenizer_ || !device_ || !executor_) {
            throw std::runtime_error(
                "archive build: backend not fully loaded "
                "(need --native-kernels cuda)");
        }
        if (!executor_->kvmem_archive_building()) {
            throw std::runtime_error(
                "archive build requires --kvmem-archive and "
                "--kvmem-archive-mode build");
        }

        std::vector<uint32_t> corpus;
        if (!cfg.token_input_path.empty()) {
            if (!cfg.input_path.empty()) {
                throw std::runtime_error(
                    "archive build accepts only one of --archive-input and "
                    "--archive-token-input");
            }
            std::ifstream in(cfg.token_input_path, std::ios::binary);
            if (!in) {
                throw std::runtime_error("cannot read archive token input: " +
                                         cfg.token_input_path);
            }
            char magic[8] = {};
            uint64_t count = 0;
            char token_model_sha256[64] = {};
            in.read(magic, sizeof(magic));
            in.read(reinterpret_cast<char *>(&count), sizeof(count));
            in.read(token_model_sha256, sizeof(token_model_sha256));
            static constexpr char expected[8] = {
                'Q', 'W', '3', 'T', 'O', 'K', '2', '\0'};
            if (!in || !std::equal(std::begin(magic), std::end(magic),
                                   std::begin(expected))) {
                throw std::runtime_error(
                    "invalid archive token input header: " +
                    cfg.token_input_path);
            }
            const std::string token_digest(
                token_model_sha256, sizeof(token_model_sha256));
            const std::string archive_digest =
                kvmem_archive_->manifest().layout.model_sha256;
            if (token_digest != archive_digest) {
                throw std::runtime_error(
                    "archive token input was produced by a different model: " +
                    cfg.token_input_path);
            }
            if (count > static_cast<uint64_t>(
                            std::numeric_limits<size_t>::max() /
                            sizeof(uint32_t))) {
                throw std::runtime_error("archive token input is too large");
            }
            corpus.resize(static_cast<size_t>(count));
            if (!corpus.empty()) {
                in.read(reinterpret_cast<char *>(corpus.data()),
                        static_cast<std::streamsize>(corpus.size() *
                                                     sizeof(corpus[0])));
            }
            if (!in) {
                throw std::runtime_error(
                    "truncated archive token input: " + cfg.token_input_path);
            }
            char trailing = 0;
            if (in.read(&trailing, 1)) {
                throw std::runtime_error(
                    "archive token input has trailing bytes: " +
                    cfg.token_input_path);
            }
        } else if (cfg.input_path.empty()) {
            build_session_corpus(corpus,
                                 cfg.tokens > 0 ? cfg.tokens : 65536);
        } else {
            std::ifstream in(cfg.input_path, std::ios::binary);
            if (!in) {
                throw std::runtime_error("cannot read archive input: " +
                                         cfg.input_path);
            }
            std::ostringstream text;
            text << in.rdbuf();
            const std::vector<int32_t> ids = tokenizer_->encode(text.str());
            corpus.assign(ids.begin(), ids.end());
        }
        uint64_t target = cfg.tokens > 0
                              ? std::min<uint64_t>(cfg.tokens, corpus.size())
                              : corpus.size();

        // Only a whole raw-K chunk is ever durable, so the archive's length is
        // rounded down to the chunk stride and the remainder is dropped.
        const uint32_t chunk_stride =
            kvmem_archive_->manifest().layout.raw_chunk_tokens;
        if (cfg.pad_final_chunk && target % chunk_stride != 0) {
            if (cfg.tokens > 0 && cfg.tokens < corpus.size()) {
                throw std::runtime_error(
                    "--archive-pad-final-chunk cannot be combined with a "
                    "shorter --archive-tokens prefix; omit --archive-tokens "
                    "to preserve and pad the complete input");
            }
            const uint64_t original_target = target;
            target = ((target + chunk_stride - 1) / chunk_stride) *
                     chunk_stride;
            const std::vector<int32_t> newline_ids = tokenizer_->encode("\n");
            if (newline_ids.empty()) {
                throw std::runtime_error(
                    "tokenizer produced no token for archive newline padding");
            }
            corpus.reserve(static_cast<size_t>(target));
            size_t pad_index = 0;
            while (corpus.size() < target) {
                corpus.push_back(static_cast<uint32_t>(
                    newline_ids[pad_index % newline_ids.size()]));
                ++pad_index;
            }
            log("archive build: padded corpus from " +
                std::to_string(original_target) + " to " +
                std::to_string(target) + " tokens with newline tokens");
        } else {
            target = (target / chunk_stride) * chunk_stride;
        }
        if (target == 0) {
            throw std::runtime_error(
                "archive build needs at least one full raw-K chunk (" +
                std::to_string(chunk_stride) + " tokens); input has " +
                std::to_string(corpus.size()));
        }

        // Resume where a previous run left off. Everything below the resume
        // point is already durable, including its recurrent state.
        uint64_t start = kvmem_archive_->resume_position();
        if (start > target) {
            throw std::runtime_error(
                "cannot resume KVMem archive build to a shorter target (" +
                std::to_string(target) + " tokens) than its durable ladder (" +
                std::to_string(start) + "); use a new archive directory");
        }
        if (start > 0) {
            // Refuse to splice a different corpus onto an existing durable
            // prefix. Compare in bounded chunks so 10M-token resumes do not
            // need a second full token copy.
            constexpr uint64_t kVerifyTokens = 1ull << 20;
            for (uint64_t off = 0; off < start; off += kVerifyTokens) {
                const uint64_t n = std::min(kVerifyTokens, start - off);
                const std::vector<uint32_t> archived =
                    kvmem_archive_->read_tokens(off, n);
                if (!std::equal(
                        archived.begin(), archived.end(),
                        corpus.begin() + static_cast<std::ptrdiff_t>(off))) {
                    throw std::runtime_error(
                        "cannot resume KVMem archive build: input token "
                        "prefix differs before ladder position " +
                        std::to_string(start));
                }
            }
            log("archive build: resuming at " + std::to_string(start) +
                " tokens");
            executor_->kvmem_attach_archive(start);
        }
        // A crash can happen after tokens.bin was appended but before the
        // ladder/manifest commit. Those bytes are not part of the resumable
        // prefix and must be discarded before appending the same suffix again.
        kvmem_archive_->truncate_tokens(start);

        const bool semantic_chunk_requested =
            cfg.prefill_window ==
            KvMemArchiveBuildConfig::PrefillWindow::SemanticChunk;
        if (semantic_chunk_requested && start != 0) {
            throw std::runtime_error(
                "semantic-chunk archive build cannot resume from an "
                "intermediate ladder; rebuild into a fresh archive");
        }
        const KvMemStore *initial_store = executor_->block_store();
        if (!initial_store) {
            throw std::runtime_error(
                "archive build lost its KVMem block store");
        }
        const bool semantic_chunk_build =
            semantic_chunk_requested &&
            target > initial_store->select_budget_tokens();

        GenerationOptions gen;
        gen.max_tokens = 0;  // ingest only; the archive never decodes
        gen.kvmem_reselect_mode = semantic_chunk_build
            ? KvMemReselectMode::Force : KvMemReselectMode::Off;
        gen.kvmem_query_begin = 0;
        gen.kvmem_query_end = 0;
        if (semantic_chunk_build) {
            if (target > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error(
                    "semantic-chunk archive build currently supports at most "
                    "2^32-1 tokens");
            }
            if (target <= chunk_stride) {
                throw std::runtime_error(
                    "semantic-chunk archive needs a non-empty history before "
                    "its final physical query chunk");
            }
            gen.kvmem_prefill_window_mode =
                KvMemPrefillWindowMode::SemanticChunk;
            gen.kvmem_prefill_semantic_start_tokens =
                cfg.semantic_start_tokens;
            gen.kvmem_prefill_semantic_query_tokens =
                cfg.semantic_query_tokens;
            gen.kvmem_query_begin = static_cast<uint32_t>(
                target - chunk_stride);
            gen.kvmem_query_end = static_cast<uint32_t>(target);
            // SemanticChunk is deliberately a single cold request. A named API
            // continuation would make the provisional/replay boundary depend
            // on a warm checkpoint and is rejected by the inference path.
            gen.kvmem_session_id.clear();
            kvmem_api_session_active_ = false;
            kvmem_api_session_id_.clear();
        } else {
            gen.kvmem_session_id = "kvmem-archive-build";
            kvmem_api_session_active_ = true;
            kvmem_api_session_id_ = gen.kvmem_session_id;
        }
        kvmem_api_boundary_ckpt_ = QwenExecutor::StateSnapshot{};
        kvmem_api_boundary_pos_ = 0;
        kvmem_api_tail_tokens_.clear();
        if (semantic_chunk_build) {
            kvmem_api_tokens_.clear();
        } else {
            kvmem_api_tokens_.assign(
                corpus.begin(),
                corpus.begin() + static_cast<std::ptrdiff_t>(start));
        }

        if (start > 0) {
            // A ladder restores model state and cold tier authority, but the
            // interrupted live builder also had a concrete bounded attention
            // window at this position. Recreate its deterministic pressure
            // selection without invoking the semantic/content scorer, then
            // establish the exact API turn boundary required by continuation.
            DeviceStatus scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            bool scope_open = true;
            try {
                const KvMemStore *store = executor_->block_store();
                if (!store) {
                    throw std::runtime_error(
                        "KVMem archive resume lost its block store");
                }
                const uint32_t bt =
                    kvmem_archive_->manifest().layout.block_tokens;
                executor_->kvmem_set_query_span(
                    0, 0, static_cast<uint32_t>(start),
                    /*index_tokens=*/0,
                    /*preserve_content_index=*/false,
                    /*capture_content_without_query=*/true);
                executor_->kvmem_build_index_from_raw_k(
                    static_cast<uint32_t>(start / bt));
                const std::vector<uint32_t> selected =
                    store->pick_prefill_pressure_blocks();
                (void)executor_->kvmem_set_selection(
                    selected, /*force_raw_refresh=*/true);
                executor_->capture_state(kvmem_api_boundary_ckpt_);
                kvmem_api_boundary_pos_ = static_cast<uint32_t>(start);
                DeviceStatus done = device_->end();
                scope_open = false;
                if (!done.ok) throw std::runtime_error(done.message);
            } catch (...) {
                if (scope_open) (void)device_->end();
                throw;
            }
        }

        const uint64_t ladder = std::max<uint64_t>(
            chunk_stride, kvmem_archive_ladder_tokens_);
        const double t_start = wall_seconds();
        uint64_t pos = start;
        bool first_turn = (start == 0);
        auto persist_turn = [&](const std::vector<uint32_t> &turn,
                                uint64_t expected_end,
                                bool require_exact_end) {
            DeviceStatus scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            try {
                const uint64_t sealable =
                    executor_->kvmem_archive_sealable_tokens();
                if (require_exact_end && sealable != expected_end) {
                    throw std::runtime_error(
                        "archive build durable position mismatch: expected " +
                        std::to_string(expected_end) + " got " +
                        std::to_string(sealable));
                }
                executor_->kvmem_archive_persist_through(sealable);
                // A durable ladder must define not only recurrent/hidden
                // tensors but also the canonical active attention frame used
                // by the next ingest turn. Normalize both uninterrupted and
                // resumed paths through the same raw-authority materialization.
                const KvMemStore *store = executor_->block_store();
                if (!store) {
                    throw std::runtime_error(
                        "KVMem archive build lost its block store");
                }
                const std::vector<uint32_t> checkpoint_selection =
                    store->pick_prefill_pressure_blocks();
                (void)executor_->kvmem_set_selection(
                    checkpoint_selection, /*force_raw_refresh=*/true);
                executor_->kvmem_archive_capture_ladder(sealable);
                kvmem_archive_->append_tokens(
                    turn.data(), static_cast<size_t>(turn.size()));
                const uint32_t bt =
                    kvmem_archive_->manifest().layout.block_tokens;
                kvmem_archive_->checkpoint_metadata(
                    sealable, static_cast<uint32_t>(sealable / bt),
                    static_cast<uint32_t>(sealable / chunk_stride));
                log("archive build: tokens=" + std::to_string(sealable) + "/" +
                    std::to_string(target) + " elapsed=" +
                    fmt_seconds(wall_seconds() - t_start));
            } catch (...) {
                (void)device_->end();
                throw;
            }
            DeviceStatus done = device_->end();
            if (!done.ok) throw std::runtime_error(done.message);
        };

        if (semantic_chunk_build) {
            std::vector<uint32_t> turn(
                corpus.begin(),
                corpus.begin() + static_cast<std::ptrdiff_t>(target));
            MtpGenStats stats;
            (void)generate_mtp(turn, gen, CancellableTokenCallback{},
                               /*dump=*/nullptr,
                               /*spec_mtp=*/true, /*trace_mtp=*/false,
                               /*override_executor=*/nullptr,
                               /*manage_device_scope=*/true,
                               /*reset_session=*/true, &stats);
            pos = target;
            persist_turn(turn, pos, /*require_exact_end=*/true);
        } else {
            while (pos < target) {
                const uint64_t next = std::min(target, pos + ladder);
                std::vector<uint32_t> turn(
                    corpus.begin() + static_cast<std::ptrdiff_t>(pos),
                    corpus.begin() + static_cast<std::ptrdiff_t>(next));
                MtpGenStats stats;
                (void)generate_mtp(turn, gen, CancellableTokenCallback{},
                                   /*dump=*/nullptr,
                                   /*spec_mtp=*/true, /*trace_mtp=*/false,
                                   /*override_executor=*/nullptr,
                                   /*manage_device_scope=*/true,
                                   /*reset_session=*/first_turn, &stats);
                first_turn = false;
                pos = next;
                persist_turn(turn, pos, /*require_exact_end=*/false);
            }
        }

        const uint32_t bt = kvmem_archive_->manifest().layout.block_tokens;
        kvmem_archive_->seal(target, static_cast<uint32_t>(target / bt),
                             static_cast<uint32_t>(target / chunk_stride));
        log("archive build: sealed tokens=" + std::to_string(target) +
            " blocks=" + std::to_string(target / bt) +
            " chunks=" + std::to_string(target / chunk_stride) +
            " ladder_points=" +
            std::to_string(kvmem_archive_->manifest().ladder.size()) +
            " total=" + fmt_seconds(wall_seconds() - t_start));
        return 0;
    }

    int run_kvmem_archive(const KvMemArchiveRunConfig &cfg) {
        if (!model_ || !tokenizer_ || !device_ || !executor_) {
            throw std::runtime_error(
                "archive query: backend not fully loaded "
                "(need --native-kernels cuda)");
        }
        if (!executor_->kvmem_archive_attached()) {
            throw std::runtime_error(
                "archive query requires --kvmem-archive and "
                "--kvmem-archive-mode attach");
        }
        archive_query_base_ready_ = false;
        archive_query_base_state_ = QwenExecutor::StateSnapshot{};
        archive_query_base_selection_.clear();
        archive_query_base_tokens_.clear();
        archive_query_base_pos_ = 0;
        archive_query_base_prefix_tokens_ = 0;
        const KvMemArchiveManifest &m = kvmem_archive_->manifest();
        uint64_t want = cfg.tokens == 0
                            ? m.total_tokens
                            : std::min<uint64_t>(cfg.tokens, m.total_tokens);
        // Questions append onto the prefix through the session path, which can
        // only resume from a block boundary, so truncation snaps down to one.
        const uint32_t block_tokens = std::max<uint32_t>(1, m.layout.block_tokens);
        want = (want / block_tokens) * block_tokens;
        if (want == 0) {
            throw std::runtime_error(
                "archive query: --archive-tokens must cover at least one "
                "block (" + std::to_string(block_tokens) + " tokens)");
        }

        const double t_attach = wall_seconds();
        DeviceStatus scope = device_->begin();
        if (!scope.ok) throw std::runtime_error(scope.message);
        uint64_t restored = 0;
        try {
            restored = executor_->kvmem_attach_archive(want);
        } catch (...) {
            (void)device_->end();
            throw;
        }
        DeviceStatus done = device_->end();
        if (!done.ok) throw std::runtime_error(done.message);
        const double attach_s = wall_seconds() - t_attach;

        // The residual between the nearest ladder point and the requested
        // length is replayed from the archived token stream, which is why
        // truncating to an arbitrary N stays cheap.
        std::vector<uint32_t> residual;
        if (want > restored) {
            residual = kvmem_archive_->read_tokens(restored, want - restored);
        }

        kvmem_api_session_active_ = true;
        kvmem_api_session_id_ = "kvmem-archive-query";
        kvmem_api_boundary_ckpt_ = QwenExecutor::StateSnapshot{};
        kvmem_api_boundary_pos_ = 0;
        kvmem_api_tail_tokens_.clear();
        kvmem_api_tokens_ = kvmem_archive_->read_tokens(0, restored);

        GenerationOptions gen;
        gen.max_tokens = 0;
        gen.kvmem_reselect_mode = KvMemReselectMode::Off;
        gen.kvmem_session_id = kvmem_api_session_id_;
        // The residual replay below is itself a session turn, so it needs the
        // same boundary invariant the question turns do: the ladder point is
        // chunk-aligned and therefore block-aligned, and the state we just
        // restored is exactly the state at it.
        if (restored > 0) {
            scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            try {
                executor_->capture_state(kvmem_api_boundary_ckpt_);
            } catch (...) {
                (void)device_->end();
                throw;
            }
            done = device_->end();
            if (!done.ok) throw std::runtime_error(done.message);
            kvmem_api_boundary_pos_ = static_cast<uint32_t>(restored);
        }

        // The retrieval index is policy, not payload: it is rebuilt from the
        // archived raw K, so this process is free to use a different retrieval
        // method or sub-block count than the build did. It has to happen before
        // the residual replay, because that replay is a session continuation and
        // extends the content index rather than creating one.
        const double t_index = wall_seconds();
        const uint32_t bt = m.layout.block_tokens;
        scope = device_->begin();
        if (!scope.ok) throw std::runtime_error(scope.message);
        try {
            executor_->kvmem_set_query_span(
                0, 0, static_cast<uint32_t>(restored), /*index_tokens=*/0,
                /*preserve_content_index=*/false,
                /*capture_content_without_query=*/true);
            executor_->kvmem_build_index_from_raw_k(
                static_cast<uint32_t>(restored / bt));
        } catch (...) {
            (void)device_->end();
            throw;
        }
        done = device_->end();
        if (!done.ok) throw std::runtime_error(done.message);
        double index_s = wall_seconds() - t_index;

        const double t_residual = wall_seconds();
        double residual_s = 0.0;
        if (!residual.empty()) {
            // The ladder restores recurrent/hidden state and cold KV authority,
            // not a live page window. A fresh build reaches the same boundary
            // with its deterministic sink+recent pressure context available to
            // the next chunk. Materialize that derived window before replaying
            // the residual; otherwise its first attention chunk would see an
            // all-absent page table and diverge despite byte-identical raw K/V.
            scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            try {
                executor_->kvmem_reselect_prefill_pressure();
            } catch (...) {
                (void)device_->end();
                throw;
            }
            done = device_->end();
            if (!done.ok) throw std::runtime_error(done.message);

            MtpGenStats stats;
            (void)generate_mtp(residual, gen, CancellableTokenCallback{},
                               /*dump=*/nullptr, /*spec_mtp=*/true,
                               /*trace_mtp=*/false,
                               /*override_executor=*/nullptr,
                               /*manage_device_scope=*/true,
                               /*reset_session=*/false, &stats);
            residual_s = wall_seconds() - t_residual;

            // The incremental capture used while replaying a residual merges a
            // boundary block and may be chunked differently from the dedicated
            // bulk builder. Canonicalize the complete [0,want) index from the
            // immutable raw-K authority so attach-to-N and a fresh build-N rank
            // identical bytes through exactly the same reduction path.
            const double t_canonical_index = wall_seconds();
            scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            try {
                executor_->kvmem_set_query_span(
                    0, 0, static_cast<uint32_t>(want), /*index_tokens=*/0,
                    /*preserve_content_index=*/false,
                    /*capture_content_without_query=*/true);
                executor_->kvmem_build_index_from_raw_k(
                    static_cast<uint32_t>(want / bt));
            } catch (...) {
                (void)device_->end();
                throw;
            }
            done = device_->end();
            if (!done.ok) throw std::runtime_error(done.message);
            index_s += wall_seconds() - t_canonical_index;
        }

        log("archive attach: requested=" + std::to_string(want) +
            " ladder=" + std::to_string(restored) +
            " residual=" + std::to_string(want - restored) +
            " attach=" + fmt_seconds(attach_s) +
            " replay=" + fmt_seconds(residual_s) +
            " index=" + fmt_seconds(index_s));

        // Every question starts from this same attached prefix. Capturing it
        // once and restoring between questions is what makes the comparison
        // between questions meaningful.
        QwenExecutor::StateSnapshot base;
        std::vector<uint32_t> base_selection;
        scope = device_->begin();
        if (!scope.ok) throw std::runtime_error(scope.message);
        try {
            // An attach with no residual has an all-cold page table, while a
            // residual leaves only its recently replayed pages resident. Neither
            // shape has an explicit working set, so it cannot be restored after
            // the first semantic branch. Normalize the prefix once to the same
            // sink+recent pressure window used by ordinary long-prefill execution;
            // this gives every question a concrete, bounded base selection.
            executor_->kvmem_reselect_prefill_pressure();
            executor_->capture_state(base);
            base_selection = kvmem_selected_block_ids(executor_.get());
            if (base_selection.empty()) {
                throw std::runtime_error(
                    "archive query failed to establish a bounded base window");
            }
        } catch (...) {
            (void)device_->end();
            throw;
        }
        done = device_->end();
        if (!done.ok) throw std::runtime_error(done.message);
        const uint32_t base_pos = executor_->position();
        const std::vector<uint32_t> base_tokens = kvmem_api_tokens_;
        archive_query_base_state_ = std::move(base);
        archive_query_base_selection_ = base_selection;
        archive_query_base_tokens_ = base_tokens;
        archive_query_base_pos_ = base_pos;
        archive_query_base_prefix_tokens_ = want;
        archive_query_base_ready_ = true;

        std::ofstream results;
        if (!cfg.results_path.empty()) {
            results.open(cfg.results_path, std::ios::out | std::ios::trunc);
            if (!results) {
                throw std::runtime_error("cannot create archive results file: " +
                                         cfg.results_path);
            }
        }

        for (size_t qi = 0; qi < cfg.questions.size(); ++qi) {
            if (qi > 0) {
                scope = device_->begin();
                if (!scope.ok) throw std::runtime_error(scope.message);
                try {
                    executor_->restore_state(archive_query_base_state_);
                    executor_->kvmem_truncate_to(base_pos);
                } catch (...) {
                    (void)device_->end();
                    throw;
                }
                done = device_->end();
                if (!done.ok) throw std::runtime_error(done.message);
                kvmem_api_tokens_ = base_tokens;
                kvmem_api_tail_tokens_.clear();
                kvmem_api_boundary_pos_ = base_pos;
            }

            // A previous branch may have re-baked retained working K in a
            // different semantic window. Reconstruct the archive-prefix
            // selection from its position-free raw-K authority before every
            // question (including q0), so all branches start from byte-equivalent
            // active K rather than accumulating inverse-RoPE roundoff or inheriting
            // a prior branch's page/window state. V is position-invariant and is
            // restored by ordinary tier materialization.
            if (!base_selection.empty()) {
                scope = device_->begin();
                if (!scope.ok) throw std::runtime_error(scope.message);
                try {
                    (void)executor_->kvmem_set_selection(
                        base_selection, /*force_raw_refresh=*/true);
                    // The session path consumes this checkpoint when replaying
                    // the query. Capture it only after the exact base window has
                    // been reconstructed, for q0 as well as later branches.
                    executor_->capture_state(kvmem_api_boundary_ckpt_);
                } catch (...) {
                    (void)device_->end();
                    throw;
                }
                done = device_->end();
                if (!done.ok) throw std::runtime_error(done.message);
            }

            // Match the HTTP server's request-boundary TTFT contract as
            // closely as the direct archive CLI permits: query rendering and
            // tokenization are part of final-query input processing and must
            // be charged before retrieval/replay/prefill/first-token decode.
            const double t_q = wall_seconds();
            const KvMemArchiveRunConfig::Question &question =
                cfg.questions[qi];
            bool thinking_open = false;
            const std::string rendered_question = render_archive_question(
                question.content, cfg.question_format, &thinking_open);
            const std::string replay_removed_question =
                render_archive_question({}, cfg.question_format);
            std::string score_removed_content;
            std::string score_span_text = question.content;
            if (question.has_query_content_span) {
                const size_t begin = static_cast<size_t>(
                    question.query_content_start);
                const size_t end = static_cast<size_t>(
                    question.query_content_end);
                score_span_text = question.content.substr(begin, end - begin);
                score_removed_content = question.content;
                score_removed_content.erase(begin, end - begin);
            }
            const std::string score_removed_question = render_archive_question(
                question.has_query_content_span ? score_removed_content
                                                : std::string{},
                cfg.question_format);

            const std::vector<int32_t> ids =
                tokenizer_->encode(rendered_question);
            std::vector<uint32_t> q(ids.begin(), ids.end());
            const ArchiveTokenSpan replay_span =
                archive_removed_text_token_span(
                    ids, tokenizer_->encode(replay_removed_question),
                    "replay content");
            const ArchiveTokenSpan score_span =
                archive_removed_text_token_span(
                    ids, tokenizer_->encode(score_removed_question),
                    "score query");
            if (score_span.begin < replay_span.begin ||
                score_span.end > replay_span.end) {
                throw std::runtime_error(
                    "archive score query token span is outside the complete "
                    "content replay span");
            }
            GenerationOptions qgen;
            qgen.max_tokens = std::max(1, cfg.max_tokens);
            qgen.temperature = cfg.temperature;
            qgen.top_p = cfg.top_p;
            qgen.top_k = cfg.top_k;
            qgen.thinking_open = thinking_open;
            qgen.thinking_budget = cfg.thinking_budget;
            qgen.kvmem_session_id = kvmem_api_session_id_;
            qgen.kvmem_reselect_mode = KvMemReselectMode::Force;
            qgen.kvmem_query_begin = base_pos + score_span.begin;
            qgen.kvmem_query_end = base_pos + score_span.end;
            qgen.kvmem_replay_begin = base_pos + replay_span.begin;
            qgen.kvmem_replay_end = base_pos + replay_span.end;
            if (cfg.query_attention_probe) {
                qgen.kvmem_query_attention_probe_tokens =
                    cfg.query_probe_tokens;
                qgen.kvmem_query_attention_score_tokens =
                    cfg.query_score_tokens;
            } else if (cfg.query_guided_query) {
                qgen.kvmem_query_guided_thinking_max_tokens =
                    cfg.query_probe_tokens;
                qgen.kvmem_query_guided_query_max_tokens =
                    cfg.query_score_tokens;
            }
            qgen.kvmem_trace_tag = "archive-q" + std::to_string(qi);

            std::cerr << "[archive-query-span] question_index=" << qi
                      << " score_span=[" << qgen.kvmem_query_begin << ","
                      << qgen.kvmem_query_end << ")"
                      << " score_tokens="
                      << (score_span.end - score_span.begin)
                      << " replay_span=[" << qgen.kvmem_replay_begin << ","
                      << qgen.kvmem_replay_end << ")"
                      << " replay_span_tokens="
                      << (replay_span.end - replay_span.begin)
                      << " replay_tail_tokens="
                      << (q.size() - replay_span.begin)
                      << " query_token_selector="
                      << (cfg.query_attention_probe
                              ? "attention-probe"
                              : cfg.query_guided_query
                                  ? "guided-query"
                              : "none")
                      << " score_span_text=\""
                      << archive_json_escape(score_span_text) << "\"\n";

            double first_token_s = -1.0;
            const CancellableTokenCallback first_token_callback =
                [&](const std::string &piece) {
                    if (first_token_s < 0.0 && !piece.empty()) {
                        first_token_s = wall_seconds() - t_q;
                    }
                    return true;
                };
            MtpGenStats stats;
            const std::string answer = generate_mtp(
                q, qgen, first_token_callback, /*dump=*/nullptr,
                /*spec_mtp=*/true, /*trace_mtp=*/false,
                /*override_executor=*/nullptr, /*manage_device_scope=*/true,
                /*reset_session=*/false, &stats);
            const double question_wall_s = wall_seconds() - t_q;
            if (first_token_s < 0.0 && !answer.empty()) {
                throw std::runtime_error(
                    "archive query produced output without a first-token "
                    "callback timestamp");
            }
            if (results) {
                results << "{\"question_index\":" << qi
                        << ",\"archive_tokens\":" << want
                        << ",\"prompt_tokens\":" << q.size()
                        << ",\"decoded_tokens\":" << stats.decoded
                        << ",\"wall_s\":" << std::setprecision(9)
                        << question_wall_s
                        << ",\"ttft_s\":";
                if (first_token_s >= 0.0) {
                    results << first_token_s;
                } else {
                    results << "null";
                }
                results
                        << ",\"prefill_s\":" << stats.prefill_s
                        << ",\"decode_s\":" << stats.decode_s
                        << ",\"score_tokens\":"
                        << (score_span.end - score_span.begin)
                        << ",\"retrieval_score_tokens\":"
                        << (stats.query_guided_query_tokens > 0
                                ? stats.query_guided_query_tokens
                                : stats.query_score_token_indices.empty()
                                    ? score_span.end - score_span.begin
                                : stats.query_score_token_indices.size())
                        << ",\"query_token_selector\":\""
                        << (cfg.query_attention_probe
                                ? "attention-probe"
                                : cfg.query_guided_query
                                    ? "guided-query"
                                : "none")
                        << "\",\"query_probe_requested_tokens\":"
                        << ((cfg.query_attention_probe ||
                             cfg.query_guided_query)
                                ? cfg.query_probe_tokens
                                : 0)
                        << ",\"query_probe_decoded_tokens\":"
                        << stats.query_attention_probe_decoded
                        << ",\"query_probe_attention_used\":"
                        << (stats.query_attention_probe_used
                                ? "true"
                                : "false")
                        << ",\"query_probe_s\":"
                        << stats.query_attention_probe_s
                        << ",\"guided_thinking_tokens\":"
                        << stats.query_guided_thinking_tokens
                        << ",\"guided_thinking_closed\":"
                        << (stats.query_guided_thinking_closed
                                ? "true" : "false")
                        << ",\"guided_query_tokens\":"
                        << stats.query_guided_query_tokens
                        << ",\"guided_query_s\":"
                        << stats.query_guided_query_s
                        << ",\"guided_query_text\":\""
                        << archive_json_escape(
                               stats.query_guided_query_text)
                        << "\""
                        << ",\"query_score_token_indices\":[";
                for (size_t i = 0;
                     i < stats.query_score_token_indices.size(); ++i) {
                    if (i) results << ',';
                    results << stats.query_score_token_indices[i];
                }
                results
                        << ']'
                        << ",\"score_span_text\":\""
                        << archive_json_escape(score_span_text)
                        << "\",\"replay_span_tokens\":"
                        << (replay_span.end - replay_span.begin)
                        << ",\"question\":\""
                        << archive_json_escape(question.content)
                        << "\",\"answer\":\""
                        << archive_json_escape(answer) << "\"}\n";
                results.flush();
            }
            if (cfg.verbose) {
                std::printf("\n[archive-q%zu] tokens=%llu prompt=%zu "
                            "ttft=%.3fs wall=%.3fs prefill=%.3fs decode=%.3fs "
                            "decoded=%d\n%s\n",
                            qi, static_cast<unsigned long long>(want),
                            q.size(), first_token_s, question_wall_s,
                            stats.prefill_s, stats.decode_s, stats.decoded,
                            answer.c_str());
                std::fflush(stdout);
            }
        }
        return 0;
    }

    std::string generate_kvmem_archive_request(
            const std::string &prompt,
            const GenerationOptions &options,
            const CancellableTokenCallback &on_text) {
        if (!executor_ || !executor_->kvmem_archive_attached()) {
            throw std::runtime_error(
                "archive-backed request requires an attached KVMem archive");
        }
        if (!options.kvmem_cache_save_id.empty() ||
            !options.kvmem_cache_load_id.empty() ||
            !options.kvmem_session_id.empty()) {
            throw std::invalid_argument(
                "archive-backed requests are frozen branches and cannot be "
                "combined with process-local cache or mutable session controls");
        }
        if (!options.kvmem_replay_query_spans.empty() ||
            !options.kvmem_retrieval_group_spans.empty()) {
            throw std::invalid_argument(
                "archive-backed requests v1 do not accept transcript replay "
                "or request-local historical retrieval groups");
        }
        if (!archive_query_base_ready_) {
            KvMemArchiveRunConfig init;
            init.tokens = options_.kvmem_archive_tokens;
            init.verbose = false;
            // An empty question list performs only attach/truncate, residual
            // replay, index rebuild and frozen-base capture. The model and
            // derived index then stay live for all HTTP requests.
            (void)run_kvmem_archive(init);
        }

        DeviceStatus scope = device_->begin();
        if (!scope.ok) throw std::runtime_error(scope.message);
        bool scope_open = true;
        try {
            executor_->restore_state(archive_query_base_state_);
            executor_->kvmem_truncate_to(archive_query_base_pos_);
            (void)executor_->kvmem_set_selection(
                archive_query_base_selection_,
                /*force_raw_refresh=*/true);
            executor_->capture_state(kvmem_api_boundary_ckpt_);
            DeviceStatus done = device_->end();
            scope_open = false;
            if (!done.ok) throw std::runtime_error(done.message);
        } catch (...) {
            if (scope_open) (void)device_->end();
            throw;
        }
        kvmem_api_session_active_ = true;
        kvmem_api_session_id_ = "kvmem-archive-serve";
        kvmem_api_boundary_pos_ = archive_query_base_pos_;
        kvmem_api_tail_tokens_.clear();
        kvmem_api_tokens_ = archive_query_base_tokens_;

        std::vector<uint32_t> q;
        if (!options.prompt_token_ids_override.empty()) {
            q = options.prompt_token_ids_override;
        } else {
            const std::vector<int32_t> ids = tokenizer_->encode(prompt);
            q.assign(ids.begin(), ids.end());
        }
        if (static_cast<uint64_t>(archive_query_base_pos_) + q.size() >=
            static_cast<uint64_t>(std::max(1, options_.ctx_size))) {
            throw std::invalid_argument(
                "archive prefix plus request prompt exceeds --ctx");
        }

        GenerationOptions qgen = options;
        qgen.kvmem_cache_save_id.clear();
        qgen.kvmem_cache_load_id.clear();
        qgen.kvmem_cache_load_mode = KvMemLocalCacheMode::None;
        qgen.kvmem_session_id = kvmem_api_session_id_;
        qgen.kvmem_reselect_mode = KvMemReselectMode::Force;
        qgen.kvmem_query_begin = archive_query_base_pos_;
        qgen.kvmem_query_end = archive_query_base_pos_ +
            static_cast<uint32_t>(q.size());

        MtpGenStats stats;
        return generate_mtp(
            q, qgen, on_text, /*dump=*/nullptr,
            /*spec_mtp=*/true, /*trace_mtp=*/false,
            /*override_executor=*/nullptr, /*manage_device_scope=*/true,
            /*reset_session=*/false, &stats);
    }

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
        if (cfg.repeat_queries < 0 ||
            (cfg.repeat_queries > 0 && cfg.query_tokens <= 0)) {
            throw std::runtime_error(
                "kvmem-session repeated queries require query_tokens > 0");
        }
        if (cfg.prefill_probe_tokens < 0 || cfg.prefill_probe_repeats < 0 ||
            (cfg.prefill_probe_repeats > 0 &&
             cfg.prefill_probe_tokens <= 0)) {
            throw std::runtime_error(
                "kvmem-session prefill probes require positive token and "
                "repeat counts");
        }
        if (cfg.repeat_mode != "frozen" &&
            cfg.repeat_mode != "sequential") {
            throw std::runtime_error(
                "kvmem-session repeat mode must be frozen|sequential");
        }
        if (!QwenExecutor::kvmem_timing_enabled()) {
            log("kvmem-session: WARNING QW3_KVMEM_TIMING not set; the "
                "selection/stage/assemble breakdown will read as zero");
        }

        // Pre-tokenize a synthetic document pool large enough to cover the
        // largest ladder target. Each turn carves an exact token slice so the
        // running position lands precisely on each ladder point after prefill.
        const uint64_t max_target = cfg.ladder_tokens.back();
        const uint64_t query_reserve = std::max<uint64_t>(
            2048,
            static_cast<uint64_t>(std::max(0, cfg.repeat_queries)) *
                static_cast<uint64_t>(std::max(1, cfg.query_tokens)));
        std::vector<uint32_t> pool;
        if (cfg.input_path.empty()) {
            build_session_corpus(pool, max_target);
        } else {
            std::ifstream in(cfg.input_path, std::ios::binary);
            if (!in) {
                throw std::runtime_error(
                    "kvmem-session cannot read input: " + cfg.input_path);
            }
            std::ostringstream text;
            text << in.rdbuf();
            const std::vector<int32_t> ids = tokenizer_->encode(text.str());
            pool.assign(ids.begin(), ids.end());
            if (pool.size() < max_target) {
                throw std::runtime_error(
                    "kvmem-session input token count " +
                    std::to_string(pool.size()) + " is shorter than target " +
                    std::to_string(max_target));
            }
        }
        const uint64_t branch_reserve = std::max<uint64_t>(
            query_reserve,
            static_cast<uint64_t>(std::max(0, cfg.prefill_probe_tokens)));
        std::vector<uint32_t> branch_pool;
        build_session_corpus(branch_pool, branch_reserve);
        log("kvmem-session: corpus pool tokens=" + std::to_string(pool.size()) +
            " target_max=" + std::to_string(max_target) +
            " source=" + (cfg.input_path.empty() ? "synthetic" : cfg.input_path) +
            " repeat_queries=" + std::to_string(cfg.repeat_queries) +
            " prefill_probe_tokens=" +
                std::to_string(cfg.prefill_probe_tokens) +
            " prefill_probe_repeats=" +
                std::to_string(cfg.prefill_probe_repeats) +
            " repeat_mode=" + cfg.repeat_mode);

        GenerationOptions gen;
        gen.max_tokens = std::max(1, cfg.decode_tokens);
        gen.temperature = cfg.temperature;
        gen.top_k = cfg.top_k;
        gen.top_p = cfg.top_p;
        gen.ignore_eos = true;  // decode exactly decode_tokens (steady-state TBT)
        if (cfg.query_tokens > 0) {
            gen.kvmem_session_id = "kvmem-session-profile";
            kvmem_api_session_active_ = true;
            kvmem_api_session_id_ = gen.kvmem_session_id;
            kvmem_api_boundary_ckpt_ = QwenExecutor::StateSnapshot{};
            kvmem_api_boundary_pos_ = 0;
            kvmem_api_tail_tokens_.clear();
            kvmem_api_tokens_.clear();
        }

        struct TurnRow {
            size_t turn = 0;
            uint64_t ctx_tokens = 0;
            uint64_t delta_tokens = 0;
            double sel_ms = 0, stage_in_ms = 0, stage_out_ms = 0, assemble_ms = 0;
            double asm_pages_ms = 0, asm_rerope_ms = 0;
            double asm_final_drain_ms = 0, asm_kbar_ms = 0;
            uint32_t stage_in_blocks = 0, stage_out_blocks = 0;
            // kvmem work forced DURING prefill (bounded-pool offload), folded
            // into prefill_s. Surfaced so step5's gross wall isn't opaque.
            double inpre_ms = 0;
            uint32_t inpre_stage_in_blocks = 0, inpre_stage_out_blocks = 0;
            double total_s = 0, setup_s = 0, prefill_s = 0;
            double postprefill_s = 0, decode_s = 0, finalize_s = 0;
            double semantic_s = 0, query_replay_s = 0, post_other_s = 0;
            double decode_tps = 0;
            int decoded = 0;
            double acceptance = 0;
            uint64_t kv_bytes = 0, gpu_used = 0, cpu_used = 0, nvme_used = 0;
            bool gpu_pool = false;
            uint64_t gpu_mib = 0, rss_mib = 0;
        };
        std::vector<TurnRow> rows;

        struct QueryRow {
            size_t turn = 0;
            int query = 0;
            std::string mode;
            uint32_t base_pos = 0;
            uint32_t query_begin = 0;
            uint32_t query_end = 0;
            uint32_t final_pos = 0;
            double capture_ms = 0;
            double restore_ms = 0;
            double total_s = 0;
            double semantic_s = 0;
            double query_replay_s = 0;
            double decode_s = 0;
            double first_token_ms = -1;
            double score_ms = 0;
            double stage_in_ms = 0;
            double stage_out_ms = 0;
            double assemble_ms = 0;
            uint32_t stage_in_blocks = 0;
            uint32_t stage_out_blocks = 0;
            int decoded = 0;
        };
        std::vector<QueryRow> query_rows;

        struct PrefillProbeRow {
            size_t turn = 0;
            int probe = 0;
            uint32_t base_pos = 0;
            uint32_t final_pos = 0;
            int tokens = 0;
            double restore_ms = 0;
            double prefill_s = 0;
            double total_s = 0;
        };
        std::vector<PrefillProbeRow> prefill_probe_rows;

        // An executor snapshot alone is not a complete KVMem branch point:
        // API-session query replay also owns a block-aligned recurrent checkpoint
        // plus the unaligned tail, and semantic selection changes the resident
        // working set. Keep all three so a frozen query starts from the same
        // logical history and the same selected GPU window.
        struct MilestoneState {
            uint32_t position = 0;
            QwenExecutor::StateSnapshot executor_state;
            QwenExecutor::StateSnapshot api_boundary_state;
            uint32_t api_boundary_pos = 0;
            std::vector<uint32_t> api_tail_tokens;
            std::vector<uint32_t> session_tokens;
            std::vector<uint32_t> selected_blocks;
        };
        auto capture_milestone = [&](MilestoneState &state) {
            const double start = wall_seconds();
            DeviceStatus scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            bool scope_open = true;
            try {
                state.position = executor_->position();
                executor_->capture_state(state.executor_state);
                state.api_boundary_pos = kvmem_api_boundary_pos_;
                state.api_tail_tokens = kvmem_api_tail_tokens_;
                state.session_tokens = kvmem_api_tokens_;
                state.selected_blocks =
                    kvmem_checkpoint_block_ids(executor_.get());
                state.api_boundary_state = kvmem_api_boundary_ckpt_.ready
                    ? clone_milestone_state_snapshot(
                          kvmem_api_boundary_ckpt_)
                    : QwenExecutor::StateSnapshot{};
                DeviceStatus done = device_->end();
                scope_open = false;
                if (!done.ok) throw std::runtime_error(done.message);
            } catch (...) {
                if (scope_open) (void)device_->end();
                throw;
            }
            return (wall_seconds() - start) * 1.0e3;
        };
        auto restore_milestone = [&](const MilestoneState &state) {
            const double start = wall_seconds();
            DeviceStatus scope = device_->begin();
            if (!scope.ok) throw std::runtime_error(scope.message);
            bool scope_open = true;
            try {
                executor_->restore_state(state.executor_state);
                executor_->kvmem_truncate_to(state.position);
                if (!state.selected_blocks.empty()) {
                    (void)executor_->kvmem_set_selection(
                        state.selected_blocks);
                }
                kvmem_api_boundary_pos_ = state.api_boundary_pos;
                kvmem_api_tail_tokens_ = state.api_tail_tokens;
                kvmem_api_tokens_ = state.session_tokens;
                kvmem_api_boundary_ckpt_ = state.api_boundary_state.ready
                    ? clone_milestone_state_snapshot(
                          state.api_boundary_state)
                    : QwenExecutor::StateSnapshot{};
                kvmem_api_session_active_ = true;
                kvmem_api_session_id_ = "kvmem-session-profile";
                DeviceStatus done = device_->end();
                scope_open = false;
                if (!done.ok) throw std::runtime_error(done.message);
            } catch (...) {
                if (scope_open) (void)device_->end();
                throw;
            }
            return (wall_seconds() - start) * 1.0e3;
        };

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
            const bool milestone_queries = cfg.repeat_queries > 0;
            if (cfg.query_tokens > 0 && !milestone_queries) {
                const uint64_t query_tokens = std::min<uint64_t>(
                    static_cast<uint64_t>(cfg.query_tokens), delta);
                gen.kvmem_query_begin =
                    static_cast<uint32_t>(target - query_tokens);
                gen.kvmem_query_end = static_cast<uint32_t>(target);
                gen.kvmem_reselect_mode = KvMemReselectMode::Force;
            } else {
                gen.kvmem_query_begin = 0;
                gen.kvmem_query_end = 0;
                // Milestone fan-out captures the exact context prefix before
                // any synthetic question or decode token is appended. History
                // ingest still performs ordinary pressure selection as the GPU
                // pool fills, but there is no final semantic reselect here.
                gen.kvmem_reselect_mode = milestone_queries
                    ? KvMemReselectMode::Off
                    : KvMemReselectMode::Auto;
            }
            gen.max_tokens = milestone_queries
                ? 0
                : std::max(1, cfg.decode_tokens);

            const QwenExecutor::KvMemTimingSnapshot tbase =
                QwenExecutor::kvmem_timing_snapshot();

            MtpGenStats stats;
            const bool reset = (ti == 0);
            (void)generate_mtp(chunk, gen, CancellableTokenCallback{},
                               /*dump=*/nullptr,
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
            row.asm_final_drain_ms = dms(tnow.assemble_final_drain_ns,
                                         base.assemble_final_drain_ns);
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
            row.total_s = stats.total_s;
            row.setup_s = stats.setup_s;
            row.prefill_s = stats.prefill_s;
            row.postprefill_s = stats.postprefill_s;
            row.finalize_s = stats.finalize_s;
            row.semantic_s = stats.semantic_reselect_s;
            row.query_replay_s = stats.query_replay_s;
            row.post_other_s = stats.post_other_s;
            // decode_s is already the pure decode loop -- generate_mtp now
            // excludes the post-prefill reselect (reported via stats.reselect_s).
            row.decode_s = std::max(stats.decode_s, 1.0e-9);
            row.decoded = stats.decoded;
            row.decode_tps =
                row.decoded > 0 ? row.decoded / row.decode_s : 0.0;
            row.acceptance = stats.acceptance;

            const QwenExecutor::KvMemTierUsage tu = executor_->kvmem_tier_usage();
            row.kv_bytes = tu.logical_kv_bytes;
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
                               row.asm_rerope_ms, row.asm_final_drain_ms,
                               row.asm_kbar_ms,
                               row.stage_in_blocks, row.stage_out_blocks,
                               row.prefill_s, row.decode_s, row.decode_tps,
                               row.decoded, row.acceptance, row.gpu_mib,
                               row.rss_mib, row.inpre_ms,
                               row.inpre_stage_in_blocks,
                               row.inpre_stage_out_blocks, row.total_s,
                               row.setup_s, row.postprefill_s,
                               row.finalize_s, row.semantic_s,
                               row.query_replay_s, row.post_other_s);
            rows.push_back(row);

            if (milestone_queries) {
                MilestoneState milestone;
                const double capture_ms = capture_milestone(milestone);
                if (milestone.position != target) {
                    throw std::runtime_error(
                        "kvmem-session milestone position does not match the "
                        "requested ladder target");
                }

                const QwenExecutor::KvMemIndexUsage iu =
                    executor_->kvmem_index_usage();
                std::ostringstream resource;
                resource << "[kvmem-session-resource] turn=" << ti
                         << " workspace_tokens=" << target
                         << " blocks=" << tu.total_blocks
                         << " logical_kv_bytes=" << tu.logical_kv_bytes
                         << " gpu_kv_used_bytes=" << tu.gpu_used_bytes
                         << " gpu_kv_capacity_bytes=" << tu.gpu_capacity_bytes
                         << " external_physical_bytes="
                         << tu.external_physical_bytes
                         << " persistent_authority_bytes="
                         << tu.persistent_authority_bytes
                         << " cpu_kv_bytes=" << tu.cpu_used_bytes
                         << " nvme_kv_bytes=" << tu.nvme_used_bytes
                         << " index_value_bytes=" << iu.value_bytes
                         << " index_metadata_bytes=" << iu.metadata_bytes
                         << " index_logical_bytes=" << iu.logical_bytes
                         << " index_capacity_bytes=" << iu.capacity_bytes
                         << " index_prototypes=" << iu.prototypes
                         << " indexed_blocks=" << iu.indexed_blocks
                         << " index_cpu=" << (iu.cpu_placement ? 1 : 0)
                         << " index_adaptive=" << (iu.adaptive ? 1 : 0);
                log(resource.str());

                if (cfg.prefill_probe_repeats > 0) {
                    std::vector<uint32_t> probe_chunk(
                        branch_pool.begin(),
                        branch_pool.begin() + cfg.prefill_probe_tokens);
                    for (int pi = 0; pi < cfg.prefill_probe_repeats; ++pi) {
                        const double restore_ms = restore_milestone(milestone);
                        GenerationOptions pgen = gen;
                        pgen.max_tokens = 0;
                        pgen.kvmem_query_begin = 0;
                        pgen.kvmem_query_end = 0;
                        pgen.kvmem_reselect_mode = KvMemReselectMode::Off;
                        MtpGenStats pstats;
                        (void)generate_mtp(
                            probe_chunk, pgen, CancellableTokenCallback{},
                            /*dump=*/nullptr,
                            /*spec_mtp=*/true, /*trace_mtp=*/false,
                            /*override_executor=*/nullptr,
                            /*manage_device_scope=*/true,
                            /*reset_session=*/false, &pstats);
                        PrefillProbeRow pr;
                        pr.turn = ti;
                        pr.probe = pi;
                        pr.base_pos = milestone.position;
                        pr.final_pos = executor_->position();
                        pr.tokens = cfg.prefill_probe_tokens;
                        pr.restore_ms = restore_ms;
                        pr.prefill_s = pstats.prefill_s;
                        pr.total_s = pstats.total_s;
                        prefill_probe_rows.push_back(pr);
                        std::ostringstream pmsg;
                        pmsg << std::fixed << std::setprecision(3)
                             << "[kvmem-session-prefill-probe] turn="
                             << pr.turn << " probe=" << pr.probe
                             << " base=" << pr.base_pos
                             << " final=" << pr.final_pos
                             << " tokens=" << pr.tokens
                             << " restore_ms=" << pr.restore_ms
                             << " prefill_ms=" << pr.prefill_s * 1.0e3
                             << " total_ms=" << pr.total_s * 1.0e3
                             << " prefill_tps="
                             << (pr.tokens /
                                 std::max(pr.prefill_s, 1.0e-9));
                        log(pmsg.str());
                    }
                    (void)restore_milestone(milestone);
                }

                const uint64_t query_slots = std::max<uint64_t>(
                    1, query_reserve /
                           static_cast<uint64_t>(cfg.query_tokens));
                for (int qi = 0; qi < cfg.repeat_queries; ++qi) {
                    double restore_ms = 0.0;
                    if (cfg.repeat_mode == "frozen" || qi == 0) {
                        restore_ms = restore_milestone(milestone);
                    }

                    const uint32_t query_begin = executor_->position();
                    // Reuse the exact same query set at every workspace point.
                    const uint64_t slot = static_cast<uint64_t>(qi) % query_slots;
                    const uint64_t qoff =
                        slot * static_cast<uint64_t>(cfg.query_tokens);
                    std::vector<uint32_t> query_chunk(
                        branch_pool.begin() + static_cast<std::ptrdiff_t>(qoff),
                        branch_pool.begin() + static_cast<std::ptrdiff_t>(
                            qoff + static_cast<uint64_t>(cfg.query_tokens)));

                    GenerationOptions qgen = gen;
                    qgen.max_tokens = std::max(1, cfg.decode_tokens);
                    qgen.kvmem_query_begin = query_begin;
                    qgen.kvmem_query_end =
                        query_begin + static_cast<uint32_t>(query_chunk.size());
                    qgen.kvmem_reselect_mode = KvMemReselectMode::Force;

                    const double query_start_s = wall_seconds();
                    const QwenExecutor::KvMemTimingSnapshot qt0 =
                        QwenExecutor::kvmem_timing_snapshot();
                    double first_token_ms = -1.0;
                    const CancellableTokenCallback first_token_callback =
                        [&](const std::string &piece) {
                            if (first_token_ms < 0.0 && !piece.empty()) {
                                first_token_ms =
                                    (wall_seconds() - query_start_s) * 1.0e3;
                            }
                            return true;
                        };
                    MtpGenStats qstats;
                    (void)generate_mtp(
                        query_chunk, qgen, first_token_callback,
                        /*dump=*/nullptr,
                        /*spec_mtp=*/true, /*trace_mtp=*/false,
                        /*override_executor=*/nullptr,
                        /*manage_device_scope=*/true,
                        /*reset_session=*/false, &qstats);
                    const QwenExecutor::KvMemTimingSnapshot qt1 =
                        QwenExecutor::kvmem_timing_snapshot();

                    QueryRow qr;
                    qr.turn = ti;
                    qr.query = qi;
                    qr.mode = cfg.repeat_mode;
                    qr.base_pos = milestone.position;
                    qr.query_begin = query_begin;
                    qr.query_end = qgen.kvmem_query_end;
                    qr.final_pos = executor_->position();
                    qr.capture_ms = qi == 0 ? capture_ms : 0.0;
                    qr.restore_ms = restore_ms;
                    qr.total_s = qstats.total_s;
                    qr.semantic_s = qstats.semantic_reselect_s;
                    qr.query_replay_s = qstats.query_replay_s;
                    qr.decode_s = qstats.decode_s;
                    qr.first_token_ms = first_token_ms;
                    qr.score_ms = dms(qt1.retrieval_ns, qt0.retrieval_ns);
                    qr.stage_in_ms = dms(qt1.stage_in_ns, qt0.stage_in_ns);
                    qr.stage_out_ms = dms(qt1.stage_out_ns, qt0.stage_out_ns);
                    qr.assemble_ms = dms(qt1.assemble_ns, qt0.assemble_ns);
                    qr.stage_in_blocks =
                        qt1.stage_in_blocks - qt0.stage_in_blocks;
                    qr.stage_out_blocks =
                        qt1.stage_out_blocks - qt0.stage_out_blocks;
                    qr.decoded = qstats.decoded;
                    query_rows.push_back(qr);

                    std::ostringstream qmsg;
                    qmsg << std::fixed << std::setprecision(3)
                         << "[kvmem-session-query] turn=" << qr.turn
                         << " query=" << qr.query
                         << " mode=" << qr.mode
                         << " base=" << qr.base_pos
                         << " span=[" << qr.query_begin << ","
                         << qr.query_end << ")"
                         << " final=" << qr.final_pos
                         << " capture_ms=" << qr.capture_ms
                         << " restore_ms=" << qr.restore_ms
                         << " total_ms=" << qr.total_s * 1.0e3
                         << " semantic_ms=" << qr.semantic_s * 1.0e3
                         << " replay_ms=" << qr.query_replay_s * 1.0e3
                         << " decode_ms=" << qr.decode_s * 1.0e3
                         << " first_token_ms=" << qr.first_token_ms
                         << " score_ms=" << qr.score_ms
                         << " stage_in_ms=" << qr.stage_in_ms
                         << " stage_out_ms=" << qr.stage_out_ms
                         << " assemble_ms=" << qr.assemble_ms
                         << " stage_in_blocks=" << qr.stage_in_blocks
                         << " stage_out_blocks=" << qr.stage_out_blocks
                         << " decoded=" << qr.decoded;
                    log(qmsg.str());
                }
                // Branch probes are diagnostics. Return to the exact milestone
                // so the next ladder point prefills only its true context delta.
                (void)restore_milestone(milestone);
            }
        }

        // Final summary table.
        std::ostringstream tbl;
        tbl << "\n=== kvmem-session SUMMARY (update_mode=step, MTP on) ===\n";
        tbl << "  turn      ctx    delta  total_s  setup_s  prefill_s"
               "  post_s  decode_s  final_s  sum_err_ms  pre_tok/s"
               "  dec_tok/s  accept    KVgib  GPUgib  CPUgib  NVMEgib"
               "  pool  GPUmib  RSSmib\n";
        for (const TurnRow &r : rows) {
            const double phase_sum =
                r.setup_s + r.prefill_s + r.postprefill_s +
                r.decode_s + r.finalize_s;
            tbl << "  " << std::setw(4) << r.turn
                << std::setw(9) << r.ctx_tokens
                << std::setw(9) << r.delta_tokens
                << std::fixed << std::setprecision(2)
                << std::setw(9) << r.total_s
                << std::setw(9) << r.setup_s
                << std::setw(11) << r.prefill_s
                << std::setw(8) << r.postprefill_s
                << std::setw(10) << r.decode_s
                << std::setw(9) << r.finalize_s
                << std::setprecision(3)
                << std::setw(12) << (r.total_s - phase_sum) * 1000.0
                << std::setprecision(1)
                << std::setw(11) << (r.delta_tokens / std::max(r.prefill_s, 1e-9))
                << std::setprecision(2)
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
        if (!query_rows.empty()) {
            std::ostringstream qtbl;
            qtbl << "\n=== kvmem-session REPEATED QUERY SUMMARY ===\n";
            qtbl << "  turn  query       mode       base      qbegin"
                    "        qend  restore_ms  total_ms  semantic_ms"
                    "  replay_ms  decode_ms  first_token_ms  score_ms  stagein_ms"
                    "  stageout_ms  assemble_ms  decoded\n";
            for (const QueryRow &q : query_rows) {
                qtbl << "  " << std::setw(4) << q.turn
                     << std::setw(7) << q.query
                     << std::setw(11) << q.mode
                     << std::setw(11) << q.base_pos
                     << std::setw(12) << q.query_begin
                     << std::setw(12) << q.query_end
                     << std::fixed << std::setprecision(3)
                     << std::setw(12) << q.restore_ms
                     << std::setw(10) << q.total_s * 1.0e3
                     << std::setw(13) << q.semantic_s * 1.0e3
                     << std::setw(11) << q.query_replay_s * 1.0e3
                     << std::setw(11) << q.decode_s * 1.0e3
                     << std::setw(16) << q.first_token_ms
                     << std::setw(10) << q.score_ms
                     << std::setw(12) << q.stage_in_ms
                     << std::setw(13) << q.stage_out_ms
                     << std::setw(13) << q.assemble_ms
                     << std::setw(9) << q.decoded << "\n";
            }
            std::cerr << qtbl.str();
        }
        return 0;
    }

    std::string generate(const std::string &prompt,
                         const GenerationOptions &options,
                         const CancellableTokenCallback &on_text) override {
        ScopedKvMemSemanticBudget budget_scope(
            executor_.get(), options.kvmem_semantic_budget);
        if (executor_ && executor_->kvmem_enabled() &&
            std::getenv("QW3_KVMEM_TRACE")) {
            const KvMemStore *store = executor_->block_store();
            log("native kvmem request semantic-budget configured=" +
                std::to_string(store->config().select_budget) +
                " requested=" +
                std::to_string(options.kvmem_semantic_budget) +
                " effective=" +
                std::to_string(store->select_budget_tokens()) +
                " prefill=" +
                std::to_string(store->config().prefill_budget));
        }
        if (executor_ && executor_->kvmem_archive_attached()) {
            return generate_kvmem_archive_request(prompt, options, on_text);
        }
        if (!options.kvmem_cache_save_id.empty() ||
            !options.kvmem_cache_load_id.empty()) {
            return generate_local_kvmem_cache(prompt, options, on_text);
        }
        invalidate_local_kvmem_caches("evicted");
        kvmem_api_session_active_ = false;
        kvmem_api_session_id_.clear();
        kvmem_api_boundary_ckpt_ = QwenExecutor::StateSnapshot{};
        kvmem_api_boundary_pos_ = 0;
        kvmem_api_tail_tokens_.clear();
        kvmem_api_tokens_.clear();
        return generate_internal(prompt, options, on_text,
                                 /*reset_session=*/true);
    }

    std::string generate_session(const std::string &prompt_fragment,
                                 const GenerationOptions &options,
                                 const TokenCallback &on_text,
                                 bool reset) override {
        ScopedKvMemSemanticBudget budget_scope(
            executor_.get(), options.kvmem_semantic_budget);
        // The legacy live-session API has no cache version/CAS field. Letting
        // it mutate a named checkpoint's shared executor lineage would leave
        // the registry at a stale version, so explicitly invalidate first.
        invalidate_local_kvmem_caches("evicted");
        if (!options_.kvmem_enabled) {
            throw std::runtime_error(
                "persistent session append requires KVMem");
        }
        // A persistent raw-token session only needs a scorer whose query
        // replay source index remains valid across the start/finish boundary.
        // Fixed/adaptive key-direction scorers use the same SubBlockMeanK
        // executor layout as sub-block-mean-k, so rejecting them here was an
        // API-layer restriction rather than an executor limitation.
        const bool session_retrieval_supported =
            options_.kvmem_retrieval_method == "mean-k" ||
            options_.kvmem_retrieval_method == "sub-block-mean-k" ||
            options_.kvmem_retrieval_method == "key-direction-fixed4" ||
            options_.kvmem_retrieval_method == "key-direction-adaptive";
        if (!options_.kvmem_query_conditioned ||
            !session_retrieval_supported) {
            throw std::runtime_error(
                "persistent KVMem sessions currently require "
                "--kvmem-query-conditioned with a replay-compatible "
                "Mean-K/SubBlockMeanK retrieval method");
        }
        if (options.kvmem_session_id.empty()) {
            throw std::invalid_argument(
                "persistent session append requires kvmem_session_id");
        }
        if (reset) {
            kvmem_api_session_active_ = true;
            kvmem_api_session_id_ = options.kvmem_session_id;
            kvmem_api_boundary_ckpt_ = QwenExecutor::StateSnapshot{};
            kvmem_api_boundary_pos_ = 0;
            kvmem_api_tail_tokens_.clear();
            kvmem_api_tokens_.clear();
        } else if (!kvmem_api_session_active_ ||
                   kvmem_api_session_id_ != options.kvmem_session_id) {
            throw std::runtime_error(
                "KVMem session append does not match the active session");
        }
        return generate_internal(
            prompt_fragment, options,
            on_text ? CancellableTokenCallback(
                          [on_text](const std::string &piece) {
                              on_text(piece);
                              return true;
                          })
                    : CancellableTokenCallback{},
            /*reset_session=*/reset);
    }

    KvMemLocalCacheInfo kvmem_local_cache_info(
            const std::string &id) override {
        auto it = kvmem_local_caches_.find(id);
        if (it == kvmem_local_caches_.end()) return {};
        expire_local_kvmem_cache(it->second);
        return local_kvmem_cache_info(it->second);
    }

    bool erase_kvmem_local_cache(const std::string &id) override {
        auto it = kvmem_local_caches_.find(id);
        if (it == kvmem_local_caches_.end()) return false;
        evict_local_kvmem_cache(it->second, "evicted");
        return true;
    }

    std::string generate_internal(const std::string &prompt,
                                  const GenerationOptions &options,
                                  const CancellableTokenCallback &on_text,
                                  bool reset_session) {
        if (!model_) throw std::runtime_error("qwen-native backend is not loaded");
        if (options_.native_kernels != "cuda") {
            throw std::runtime_error("qwen-native now uses a device-resident executor; use --native-kernels cuda");
        }
        if (!device_ || !weights_ || !executor_) {
            throw std::runtime_error("qwen-native backend was not fully initialized in load()");
        }

        if (!tokenizer_) {
            if (const GgufFile *gguf = model_->source().gguf()) {
                tokenizer_ = std::make_unique<QwenTokenizer>(*gguf);
            } else {
                tokenizer_ = std::make_unique<QwenTokenizer>(
                    model_->source().model_directory());
            }
        }
        std::vector<uint32_t> prompt_tokens;
        if (!options.prompt_token_ids_override.empty()) {
            prompt_tokens = options.prompt_token_ids_override;
        } else {
            const std::vector<int32_t> ids = tokenizer_->encode(prompt);
            prompt_tokens.assign(ids.begin(), ids.end());
        }
        GenerationOptions effective_options = options;
        const uint32_t append_base = reset_session
            ? 0u : static_cast<uint32_t>(executor_->position());
        if (!reset_session &&
            effective_options.kvmem_query_end >
                effective_options.kvmem_query_begin) {
            effective_options.kvmem_query_begin += append_base;
            effective_options.kvmem_query_end += append_base;
        }
        if (!reset_session &&
            effective_options.kvmem_context_end >
                effective_options.kvmem_context_begin) {
            effective_options.kvmem_context_begin += append_base;
            effective_options.kvmem_context_end += append_base;
        }
        const uint32_t ctx_size = options_.ctx_size > 0
            ? static_cast<uint32_t>(options_.ctx_size)
            : 4096u;
        const uint64_t total_prompt_tokens =
            static_cast<uint64_t>(append_base) + prompt_tokens.size();
        if (total_prompt_tokens > static_cast<uint64_t>(ctx_size)) {
            throw std::runtime_error("prompt exceeds KV context: prompt_tokens=" +
                                     std::to_string(total_prompt_tokens) +
                                     " ctx=" + std::to_string(ctx_size));
        }
        const uint32_t max_emit_tokens =
            ctx_size - static_cast<uint32_t>(total_prompt_tokens) + 1U;
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

        if (!effective_options.kvmem_session_id.empty() && !active_mtp) {
            throw std::runtime_error(
                "persistent KVMem session append currently requires native MTP");
        }

        const uint32_t mtp_route_rope_limit = model_->config().n_ctx_train;
        const uint64_t mtp_route_logical_end =
            static_cast<uint64_t>(prompt_tokens.size()) +
            static_cast<uint64_t>(std::max(0, effective_options.max_tokens));
        const bool mtp_route_positions_safe =
            !options_.kvmem_enabled || !active_mtp ||
            mtp_route_rope_limit == 0 ||
            mtp_route_logical_end <= mtp_route_rope_limit;
        const bool route_continuous = reset_session &&
            effective_options.continuous_batching &&
            (!active_mtp || continuous_batching_mtp_enabled()) &&
            mtp_route_positions_safe;

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
                            spec_mtp, trace_mtp,
                            /*override_executor=*/nullptr,
                            /*manage_device_scope=*/true,
                            reset_session);
    }

private:
    struct LocalKvMemCacheEntry {
        std::string id;
        std::string status = "ready";
        uint64_t version = 0;
        uint32_t position = 0;
        std::string fingerprint;
        int64_t created_at = 0;
        int64_t last_access_at = 0;
        int64_t expires_at = 0;
        QwenExecutor::StateSnapshot executor_state;
        QwenExecutor::StateSnapshot api_boundary_state;
        uint32_t api_boundary_pos = 0;
        std::vector<uint32_t> api_tail_tokens;
        std::vector<uint32_t> session_tokens;
        std::vector<uint32_t> selected_blocks;
        uint32_t total_blocks = 0;
        QwenExecutor::KvMemTierUsage tier_usage;
    };

    static int64_t local_cache_unix_now() {
        return std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    static bool valid_local_kvmem_cache_id(const std::string &id) {
        if (id.empty() || id.size() > 128) return false;
        return std::all_of(id.begin(), id.end(), [](unsigned char c) {
            return (c >= 'a' && c <= 'z') ||
                   (c >= 'A' && c <= 'Z') ||
                   (c >= '0' && c <= '9') ||
                   c == '-' || c == '_' || c == '.' || c == ':';
        });
    }

    static std::string local_cache_fingerprint(
            const std::vector<uint32_t> &tokens,
            const EngineOptions &options, uint32_t position) {
        // FNV-1a is an integrity/debug fingerprint, not an authentication
        // primitive. Exact cache lookup is always the explicit ID + version.
        uint64_t h = 1469598103934665603ULL;
        auto mix_byte = [&](uint8_t value) {
            h ^= static_cast<uint64_t>(value);
            h *= 1099511628211ULL;
        };
        auto mix_u64 = [&](uint64_t value) {
            for (int i = 0; i < 8; ++i) {
                mix_byte(static_cast<uint8_t>(value & 0xffu));
                value >>= 8;
            }
        };
        auto mix_string = [&](const std::string &value) {
            for (unsigned char c : value) mix_byte(c);
            mix_byte(0);
        };
        mix_string(options.model_path);
        mix_u64(static_cast<uint64_t>(options.ctx_size));
        mix_u64(static_cast<uint64_t>(options.kvmem_block_tokens));
        mix_u64(static_cast<uint64_t>(options.kvmem_budget));
        mix_u64(static_cast<uint64_t>(options.kvmem_prefill_budget));
        mix_u64(static_cast<uint64_t>(options.kvmem_gen_budget));
        mix_string(options.kvmem_retrieval_method);
        mix_string(options.kvmem_index_placement);
        mix_u64(position);
        for (uint32_t token : tokens) mix_u64(token);
        std::ostringstream out;
        out << "fnv1a64:" << std::hex << std::setfill('0')
            << std::setw(16) << h;
        return out.str();
    }

    static KvMemLocalCacheInfo local_kvmem_cache_info(
            const LocalKvMemCacheEntry &entry) {
        KvMemLocalCacheInfo info;
        info.found = true;
        info.id = entry.id;
        info.status = entry.status;
        info.version = entry.version;
        info.position = entry.position;
        info.fingerprint = entry.fingerprint;
        info.created_at = entry.created_at;
        info.last_access_at = entry.last_access_at;
        info.expires_at = entry.expires_at;
        info.selected_blocks =
            static_cast<uint32_t>(entry.selected_blocks.size());
        info.total_blocks = entry.total_blocks;
        info.gpu_bytes = entry.tier_usage.gpu_used_bytes;
        info.cpu_bytes = entry.tier_usage.cpu_used_bytes;
        info.nvme_bytes = entry.tier_usage.nvme_used_bytes;
        return info;
    }

    static void evict_local_kvmem_cache(LocalKvMemCacheEntry &entry,
                                        const char *status) {
        entry.status = status;
        entry.executor_state = QwenExecutor::StateSnapshot{};
        entry.api_boundary_state = QwenExecutor::StateSnapshot{};
        entry.api_tail_tokens.clear();
        entry.session_tokens.clear();
        entry.selected_blocks.clear();
    }

    static void expire_local_kvmem_cache(LocalKvMemCacheEntry &entry) {
        if (entry.status == "ready" && entry.expires_at > 0 &&
            local_cache_unix_now() >= entry.expires_at) {
            evict_local_kvmem_cache(entry, "expired");
        }
    }

    void invalidate_local_kvmem_caches(const char *status,
                                       const std::string &except_id = {}) {
        for (auto &[id, entry] : kvmem_local_caches_) {
            if (!except_id.empty() && id == except_id) continue;
            if (entry.status == "ready") {
                evict_local_kvmem_cache(entry, status);
            }
        }
    }

    LocalKvMemCacheEntry &require_local_kvmem_cache(
            const std::string &id, uint64_t expected_version,
            bool expected_version_set) {
        auto it = kvmem_local_caches_.find(id);
        if (it == kvmem_local_caches_.end()) {
            throw std::runtime_error(
                "KVMem local cache not found: " + id);
        }
        expire_local_kvmem_cache(it->second);
        if (it->second.status != "ready") {
            throw std::runtime_error(
                "KVMem local cache " + it->second.status + ": " + id);
        }
        if (expected_version_set &&
            it->second.version != expected_version) {
            throw std::runtime_error(
                "KVMem local cache version conflict: id=" + id +
                " expected=" + std::to_string(expected_version) +
                " actual=" + std::to_string(it->second.version));
        }
        it->second.last_access_at = local_cache_unix_now();
        return it->second;
    }

    void capture_local_kvmem_cache(const std::string &id, uint64_t version,
                                   uint64_t ttl_seconds,
                                   int64_t preserve_created_at = 0,
                                   int64_t preserve_expires_at = 0) {
        if (!executor_ || !executor_->kvmem_enabled()) {
            throw std::runtime_error(
                "KVMem local cache capture requires an active KVMem executor");
        }
        if (!kvmem_api_session_active_ ||
            kvmem_api_session_id_ != id) {
            throw std::runtime_error(
                "KVMem local cache capture lost its active session");
        }
        LocalKvMemCacheEntry next;
        next.id = id;
        next.status = "ready";
        next.version = version;
        next.position = executor_->position();
        const int64_t now = local_cache_unix_now();
        next.created_at = preserve_created_at > 0
            ? preserve_created_at : now;
        next.last_access_at = now;
        if (ttl_seconds > 0) {
            const uint64_t bounded = std::min<uint64_t>(
                ttl_seconds,
                static_cast<uint64_t>(
                    std::numeric_limits<int64_t>::max() - now));
            next.expires_at = now + static_cast<int64_t>(bounded);
        } else {
            next.expires_at = preserve_expires_at;
        }
        next.api_boundary_pos = kvmem_api_boundary_pos_;
        next.api_tail_tokens = kvmem_api_tail_tokens_;
        next.session_tokens = kvmem_api_tokens_;
        next.selected_blocks = kvmem_checkpoint_block_ids(executor_.get());
        next.total_blocks = executor_->block_store()
            ? executor_->block_store()->block_count() : 0;
        next.tier_usage = executor_->kvmem_tier_usage();
        next.fingerprint = local_cache_fingerprint(
            next.session_tokens, options_, next.position);

        DeviceStatus scope = device_->begin();
        if (!scope.ok) throw std::runtime_error(scope.message);
        bool scope_open = true;
        try {
            executor_->capture_state(next.executor_state);
            next.api_boundary_state = kvmem_api_boundary_ckpt_.ready
                ? clone_milestone_state_snapshot(kvmem_api_boundary_ckpt_)
                : QwenExecutor::StateSnapshot{};
            DeviceStatus done = device_->end();
            scope_open = false;
            if (!done.ok) throw std::runtime_error(done.message);
        } catch (...) {
            if (scope_open) (void)device_->end();
            throw;
        }
        invalidate_local_kvmem_caches("evicted", id);
        kvmem_local_caches_.insert_or_assign(id, std::move(next));
        log("native kvmem local-cache SAVE id=" + id +
            " version=" + std::to_string(version) +
            " position=" + std::to_string(executor_->position()));
    }

    void restore_local_kvmem_cache(LocalKvMemCacheEntry &entry) {
        DeviceStatus scope = device_->begin();
        if (!scope.ok) throw std::runtime_error(scope.message);
        bool scope_open = true;
        try {
            executor_->restore_state(entry.executor_state);
            executor_->kvmem_truncate_to(entry.position);
            if (!entry.selected_blocks.empty()) {
                (void)executor_->kvmem_set_selection(entry.selected_blocks);
            }
            kvmem_api_boundary_pos_ = entry.api_boundary_pos;
            kvmem_api_tail_tokens_ = entry.api_tail_tokens;
            kvmem_api_tokens_ = entry.session_tokens;
            kvmem_api_boundary_ckpt_ = entry.api_boundary_state.ready
                ? clone_milestone_state_snapshot(entry.api_boundary_state)
                : QwenExecutor::StateSnapshot{};
            kvmem_api_session_active_ = true;
            kvmem_api_session_id_ = entry.id;
            DeviceStatus done = device_->end();
            scope_open = false;
            if (!done.ok) throw std::runtime_error(done.message);
        } catch (...) {
            if (scope_open) (void)device_->end();
            throw;
        }
        log("native kvmem local-cache RESTORE id=" + entry.id +
            " version=" + std::to_string(entry.version) +
            " position=" + std::to_string(entry.position));
    }

    std::string generate_local_kvmem_cache(
            const std::string &prompt, const GenerationOptions &options,
            const CancellableTokenCallback &on_text) {
        if (!options_.kvmem_enabled) {
            throw std::runtime_error(
                "KVMem local cache requires --kvmem");
        }
        const bool save = !options.kvmem_cache_save_id.empty();
        const bool load = !options.kvmem_cache_load_id.empty();
        if (save == load) {
            throw std::invalid_argument(
                "KVMem local cache request requires exactly one save or load");
        }
        const std::string &request_id = save
            ? options.kvmem_cache_save_id : options.kvmem_cache_load_id;
        if (!valid_local_kvmem_cache_id(request_id)) {
            throw std::invalid_argument(
                "KVMem local cache ID must be 1..128 characters from "
                "[A-Za-z0-9_.:-]");
        }
        GenerationOptions effective = options;
        effective.kvmem_cache_save_id.clear();
        effective.kvmem_cache_load_id.clear();
        effective.kvmem_cache_load_mode = KvMemLocalCacheMode::None;

        if (save) {
            if (options.max_tokens != 0) {
                throw std::invalid_argument(
                    "KVMem local cache save currently requires max_tokens=0");
            }
            if (options.kvmem_cache_ttl_seconds > 31536000) {
                throw std::invalid_argument(
                    "KVMem local cache TTL must be at most 31536000 seconds");
            }
            invalidate_local_kvmem_caches("evicted");
            const std::string &id = options.kvmem_cache_save_id;
            kvmem_api_session_active_ = true;
            kvmem_api_session_id_ = id;
            kvmem_api_boundary_ckpt_ = QwenExecutor::StateSnapshot{};
            kvmem_api_boundary_pos_ = 0;
            kvmem_api_tail_tokens_.clear();
            kvmem_api_tokens_.clear();
            effective.kvmem_session_id = id;
            std::string generated = generate_internal(
                prompt, effective, on_text, /*reset_session=*/true);
            capture_local_kvmem_cache(
                id, /*version=*/1, options.kvmem_cache_ttl_seconds);
            return generated;
        }

        const std::string &id = options.kvmem_cache_load_id;
        if (options.kvmem_cache_load_mode == KvMemLocalCacheMode::None) {
            throw std::invalid_argument(
                "KVMem local cache load requires frozen or append mode");
        }
        if (options.kvmem_cache_load_mode == KvMemLocalCacheMode::Append) {
            if (options.max_tokens != 0) {
                throw std::invalid_argument(
                    "KVMem local cache append currently requires "
                    "max_tokens=0");
            }
            if (!options.kvmem_cache_expected_version_set) {
                throw std::invalid_argument(
                    "KVMem local cache append requires expected_version");
            }
        }
        LocalKvMemCacheEntry &entry = require_local_kvmem_cache(
            id, options.kvmem_cache_expected_version,
            options.kvmem_cache_expected_version_set);
        const uint64_t old_version = entry.version;
        const int64_t created_at = entry.created_at;
        const int64_t expires_at = entry.expires_at;
        restore_local_kvmem_cache(entry);
        effective.kvmem_session_id = id;
        try {
            std::string generated = generate_internal(
                prompt, effective, on_text, /*reset_session=*/false);
            if (options.kvmem_cache_load_mode ==
                KvMemLocalCacheMode::Append) {
                capture_local_kvmem_cache(
                    id, old_version + 1, /*ttl_seconds=*/0,
                    created_at, expires_at);
            } else {
                LocalKvMemCacheEntry &master = require_local_kvmem_cache(
                    id, old_version, /*expected_version_set=*/true);
                restore_local_kvmem_cache(master);
            }
            return generated;
        } catch (...) {
            auto it = kvmem_local_caches_.find(id);
            if (it != kvmem_local_caches_.end() &&
                it->second.status == "ready" &&
                it->second.version == old_version) {
                try {
                    restore_local_kvmem_cache(it->second);
                } catch (...) {
                    evict_local_kvmem_cache(it->second, "failed");
                }
            }
            throw;
        }
    }

    struct ContinuousBatchRequest {
        uint64_t id = 0;
        std::vector<uint32_t> prompt_tokens;
        GenerationOptions options;
        CancellableTokenCallback on_text;
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
        int eos_id = -1;         // tokenizer EOS id, used for recovery
        int think_tokens = 0;    // tokens generated so far inside the block
        bool forced = false;     // currently feeding forced close tokens
        bool recover_eos = false; // replace EOS while the think block is open
        std::deque<uint32_t> forced_queue; // remaining forced tokens to feed

        bool active() const { return budget > 0 && close_id >= 0; }
        bool can_recover_eos(uint32_t token) const {
            return recover_eos && open && close_id >= 0 &&
                   eos_id >= 0 && token == static_cast<uint32_t>(eos_id);
        }
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
        bool stream_cancelled = false;
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
                        DeviceTensor *projection_outs[4] = {
                            recurrent_proj_batch_.get(),
                            recurrent_gate_batch_.get(),
                            recurrent_alpha_batch_.get(),
                            recurrent_beta_batch_.get(),
                        };
                        const DeviceWeight *projection_weights[4] = {
                            layer.attn_qkv, layer.attn_gate,
                            layer.ssm_alpha, layer.ssm_beta,
                        };
                        const uint32_t projection_strides[4] = {
                            proj_stride, gate_stride,
                            alpha_stride, beta_stride,
                        };
                        require_ok(
                            backend_.rms_norm_q8_0_matmul_fanout(
                                *norm_batch_, projection_outs,
                                projection_weights, projection_strides, 4,
                                *hidden_batch_, *layer.attn_norm,
                                total_q, hidden, eps));
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
                        require_ok(
                            backend_.rms_norm_q8_0_matmul_fanout(
                                *norm_batch_, qkv_outs, qkv_ws,
                                qkv_strides, 3, *hidden_batch_,
                                *layer.attn_norm, total_q, hidden, eps));
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
                        trace_rope_positions_if_out_of_range(
                            "continuous_prefill.qk", batch.logical_positions,
                            cfg.n_ctx_train, static_cast<int32_t>(il),
                            /*kernel_uses=*/2);
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
                    const uint32_t ffn =
                        static_cast<uint32_t>(layer.ffn_dim);
                    DeviceStatus fused_ffn = backend_.nvfp4_ffn_prefill(
                        *ffn_out_batch_, *layer.ffn_gate, *layer.ffn_up,
                        *layer.ffn_down, *layer.ffn_norm, *hidden_batch_,
                        total_q, hidden, ffn, hidden, eps);
                    if (!fused_ffn.ok) {
                        require_ok(backend_.rms_norm_batch(
                            *norm_batch_, *hidden_batch_, *layer.ffn_norm,
                            total_q, hidden, eps));
                        DeviceStatus fused_swiglu =
                            backend_.q8_0_matmul_silu_mul(
                                *ffn_mid_batch_, *layer.ffn_gate,
                                *layer.ffn_up, *norm_batch_, total_q,
                                hidden, ffn);
                        if (!fused_swiglu.ok) {
                            require_ok(backend_.q8_0_matmul(
                                *ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                                total_q, hidden, ffn));
                            require_ok(backend_.q8_0_matmul(
                                *ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                                total_q, hidden, ffn));
                            require_ok(backend_.silu_mul_n(
                                *ffn_mid_batch_, *ffn_gate_batch_,
                                *ffn_up_batch_,
                                static_cast<uint64_t>(total_q) * ffn));
                        }
                        require_ok(backend_.q8_0_matmul(
                            *ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                            total_q, ffn, hidden));
                    }
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
                                model_.token_text(
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
                            model_.token_text(
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
                        DeviceTensor *projection_outs[4] = {
                            recurrent_proj_batch_.get(),
                            recurrent_gate_batch_.get(),
                            recurrent_alpha_batch_.get(),
                            recurrent_beta_batch_.get(),
                        };
                        const DeviceWeight *projection_weights[4] = {
                            layer.attn_qkv, layer.attn_gate,
                            layer.ssm_alpha, layer.ssm_beta,
                        };
                        const uint32_t projection_strides[4] = {
                            proj_stride, gate_stride,
                            alpha_stride, beta_stride,
                        };
                        require_ok(
                            backend_.rms_norm_q8_0_matmul_fanout(
                                *norm_batch_, projection_outs,
                                projection_weights, projection_strides, 4,
                                *hidden_batch_, *layer.attn_norm,
                                bsz, hidden, eps));
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
                        const uint32_t ffn =
                            static_cast<uint32_t>(layer.ffn_dim);
                        DeviceStatus fused_ffn = backend_.nvfp4_ffn_prefill(
                            *ffn_out_batch_, *layer.ffn_gate, *layer.ffn_up,
                            *layer.ffn_down, *layer.ffn_norm, *hidden_batch_,
                            bsz, hidden, ffn, hidden, eps);
                        if (!fused_ffn.ok) {
                            require_ok(backend_.rms_norm_batch(
                                *norm_batch_, *hidden_batch_, *layer.ffn_norm,
                                bsz, hidden, eps));
                            DeviceStatus fused_swiglu =
                                backend_.q8_0_matmul_silu_mul(
                                    *ffn_mid_batch_, *layer.ffn_gate,
                                    *layer.ffn_up, *norm_batch_, bsz,
                                    hidden, ffn);
                            if (!fused_swiglu.ok) {
                                require_ok(backend_.q8_0_matmul(
                                    *ffn_gate_batch_, *layer.ffn_gate,
                                    *norm_batch_, bsz, hidden, ffn));
                                require_ok(backend_.q8_0_matmul(
                                    *ffn_up_batch_, *layer.ffn_up,
                                    *norm_batch_, bsz, hidden, ffn));
                                require_ok(backend_.silu_mul_n(
                                    *ffn_mid_batch_, *ffn_gate_batch_,
                                    *ffn_up_batch_,
                                    static_cast<uint64_t>(bsz) * ffn));
                            }
                            require_ok(backend_.q8_0_matmul(
                                *ffn_out_batch_, *layer.ffn_down,
                                *ffn_mid_batch_, bsz, ffn, hidden));
                        }
                        require_ok(backend_.add_n(
                            *hidden_batch_, *hidden_batch_, *ffn_out_batch_,
                            static_cast<uint64_t>(bsz) * hidden));
                        ffn_s += std::max(wall_seconds() - t_ffn0, 0.0);
                        recurrent_s +=
                            std::max(wall_seconds() - t_recurrent0, 0.0);
                        continue;
                    }

                    const double t_attention0 = wall_seconds();
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
                    require_ok(
                        backend_.rms_norm_q8_0_matmul_fanout(
                            *norm_batch_, qkv_outs, qkv_ws,
                            qkv_strides, 3, *hidden_batch_,
                            *layer.attn_norm, bsz, hidden, eps));
                    require_ok(backend_.rmsnorm_per_head_batch(
                        *q_batch_, *layer.attn_q_norm, bsz,
                        static_cast<uint32_t>(layer.q_rows), standard_n_heads,
                        q_stride, standard_head_dim, eps));
                    require_ok(backend_.rmsnorm_per_head_batch(
                        *k_batch_, *layer.attn_k_norm, bsz,
                        static_cast<uint32_t>(layer.k_rows), standard_n_kv_heads,
                        standard_head_dim, standard_head_dim, eps));
                    trace_rope_positions_if_out_of_range(
                        "continuous_decode.qk", ragged_positions_h_,
                        cfg.n_ctx_train, static_cast<int32_t>(il),
                        /*kernel_uses=*/2);
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
                    const uint32_t ffn =
                        static_cast<uint32_t>(layer.ffn_dim);
                    DeviceStatus fused_ffn = backend_.nvfp4_ffn_prefill(
                        *ffn_out_batch_, *layer.ffn_gate, *layer.ffn_up,
                        *layer.ffn_down, *layer.ffn_norm, *hidden_batch_,
                        bsz, hidden, ffn, hidden, eps);
                    if (!fused_ffn.ok) {
                        require_ok(backend_.rms_norm_batch(
                            *norm_batch_, *hidden_batch_, *layer.ffn_norm,
                            bsz, hidden, eps));
                        DeviceStatus fused_swiglu =
                            backend_.q8_0_matmul_silu_mul(
                                *ffn_mid_batch_, *layer.ffn_gate,
                                *layer.ffn_up, *norm_batch_, bsz,
                                hidden, ffn);
                        if (!fused_swiglu.ok) {
                            require_ok(backend_.q8_0_matmul(
                                *ffn_gate_batch_, *layer.ffn_gate, *norm_batch_,
                                bsz, hidden, ffn));
                            require_ok(backend_.q8_0_matmul(
                                *ffn_up_batch_, *layer.ffn_up, *norm_batch_,
                                bsz, hidden, ffn));
                            require_ok(backend_.silu_mul_n(
                                *ffn_mid_batch_, *ffn_gate_batch_,
                                *ffn_up_batch_,
                                static_cast<uint64_t>(bsz) * ffn));
                        }
                        require_ok(backend_.q8_0_matmul(
                            *ffn_out_batch_, *layer.ffn_down, *ffn_mid_batch_,
                            bsz, ffn, hidden));
                    }
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
                        model_.token_text(
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
                        model_.token_text(
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
        return dump == nullptr && options.max_tokens >= 0 &&
               options.kvmem_semantic_budget == 0 &&
               options.kvmem_replay_query_spans.empty() &&
               options.kvmem_oracle_token_spans.empty() &&
               options.kvmem_inline_refresh ==
                   KvMemInlineRefreshMode::Off;
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
                                            const CancellableTokenCallback &on_text) {
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
            trace_rope_positions_if_out_of_range(
                "continuous_mtp_draft.qk", rope_pos_h, cfg.n_ctx_train,
                static_cast<int32_t>(cfg.n_layers),
                /*kernel_uses=*/2);
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
            DeviceStatus fused_ffn = device_->nvfp4_ffn_prefill(
                *cb_mtp_ffn_out_batch_, *layer.ffn_gate, *layer.ffn_up,
                *layer.ffn_down, *layer.ffn_norm, *cb_mtp_h_batch_,
                bsz, hidden, ffn_stride, hidden, eps);
            if (!fused_ffn.ok) {
                require_device_status(device_->rms_norm_batch(
                    *cb_mtp_norm_batch_, *cb_mtp_h_batch_,
                    *layer.ffn_norm, bsz, hidden, eps));
                DeviceStatus fused_swiglu = device_->q8_0_matmul_silu_mul(
                    *cb_mtp_ffn_mid_batch_, *layer.ffn_gate, *layer.ffn_up,
                    *cb_mtp_norm_batch_, bsz, hidden, ffn_stride);
                if (!fused_swiglu.ok) {
                    DeviceTensor *ffn_outs[2] = {
                        cb_mtp_ffn_gate_batch_.get(),
                        cb_mtp_ffn_up_batch_.get()
                    };
                    const DeviceWeight *ffn_ws[2] = {
                        layer.ffn_gate, layer.ffn_up
                    };
                    const uint32_t ffn_strides[2] = {
                        ffn_stride, ffn_stride
                    };
                    require_device_status(device_->q8_0_matmul_fanout(
                        ffn_outs, ffn_ws, ffn_strides, 2,
                        *cb_mtp_norm_batch_, bsz, hidden));
                    require_device_status(device_->silu_mul_n(
                        *cb_mtp_ffn_mid_batch_, *cb_mtp_ffn_gate_batch_,
                        *cb_mtp_ffn_up_batch_,
                        static_cast<uint64_t>(bsz) * ffn_stride));
                }
                require_device_status(device_->q8_0_matmul_add(
                    *cb_mtp_h_batch_, *cb_mtp_h_batch_,
                    *cb_mtp_ffn_out_batch_, *layer.ffn_down,
                    *cb_mtp_ffn_mid_batch_, bsz, ffn_stride, hidden));
            } else {
                require_device_status(device_->add_n(
                    *cb_mtp_h_batch_, *cb_mtp_h_batch_,
                    *cb_mtp_ffn_out_batch_,
                    static_cast<uint64_t>(bsz) * hidden));
            }
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
                    step.report.argmax_text = model_->token_text(
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
                            model_->token_text(
                                static_cast<uint32_t>(last.token));
                    }
                }
                return true;
            };

        auto should_stop = [&](const ContinuousBatchActive &a,
                               uint32_t token) {
            return a.req && !a.req->options.ignore_eos &&
                   token == static_cast<uint32_t>(eos) &&
                   !a.budget.can_recover_eos(token);
        };
        auto emit = [&](ContinuousBatchActive &a, uint32_t token) {
            recover_thinking_eos(a.budget, token);
            if (!a.req || a.decoded >= a.req->options.max_tokens ||
                should_stop(a, token)) {
                return false;
            }
            const std::string piece =
                tokenizer_->decode_one(static_cast<int32_t>(token));
            a.req->generated += piece;
            budget_observe(a.budget, token);
            // Track committed tokens so the next verify batch's penalties
            // (presence/repetition) see the running output. No-op for greedy
            // (penalties unused), so greedy MTP stays byte-identical.
            ++a.seen_tokens[token];
            ++a.decoded;
            if (a.req->on_text && !a.req->on_text(piece)) {
                a.stream_cancelled = true;
                return false;
            }
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
                a.executor->set_prefill_chunk_override(
                    options_.prefill_chunk);
                NativeExecutorReport step;
                uint64_t prefill_ops = 0;
                if (QwenExecutor::kvmem_timing_enabled()) {
                    a.kvmem_timing_baseline = QwenExecutor::kvmem_timing_snapshot();
                }
                const double prefill0 = wall_seconds();
                // Prefix-cache: start at the restored main+MTP page-aligned
                // boundary. Both page tables, recurrent/conv state, and
                // mtp_prefix_h are seeded by initialize_continuous_active, so
                // only the uncached suffix is prefetched and MTP-primed.
                const size_t prefill_start = std::min<size_t>(
                    a.prefill_offset, req->prompt_tokens.size());
                for (size_t offset = prefill_start;
                     offset < req->prompt_tokens.size();) {
                    const uint32_t remaining = static_cast<uint32_t>(
                        req->prompt_tokens.size() - offset);
                    const uint32_t width = std::max<uint32_t>(
                        1, a.executor->effective_prefill_chunk_size(remaining));
                    size_t end =
                        offset + std::min<size_t>(remaining, width);
                    // The MTP lane owns its prefill loop, so it must honor the
                    // same exact commit boundary as the plain lane. Landing on
                    // this boundary after both target and draft priming makes
                    // the cached state lossless for future MTP requests.
                    if (a.prefix_commit_pending &&
                        offset < a.prefix_commit_len &&
                        end > a.prefix_commit_len) {
                        end = a.prefix_commit_len;
                    }
                    std::vector<uint32_t> chunk(
                        req->prompt_tokens.begin() +
                            static_cast<std::ptrdiff_t>(offset),
                        req->prompt_tokens.begin() +
                            static_cast<std::ptrdiff_t>(end));
                    const bool need_logits = end == req->prompt_tokens.size();
                    const bool mtp_local =
                        a.executor->kvmem_mtp_local_positions();
                    if (mtp_local) {
                        a.executor->kvmem_set_defer_prefill_pressure(true);
                    }
                    try {
                        step = a.executor->forward_n_tokens(
                            chunk, need_logits);
                    } catch (...) {
                        if (mtp_local) {
                            a.executor->kvmem_set_defer_prefill_pressure(false);
                        }
                        throw;
                    }
                    if (!step.ok) {
                        if (mtp_local) {
                            a.executor->kvmem_set_defer_prefill_pressure(false);
                        }
                        throw std::runtime_error("MTP prefill failed");
                    }
                    prefill_ops += step.ops_executed;
                    NativeExecutorReport prefix;
                    try {
                        prefix = mtp_local
                            ? a.executor->prime_mtp_prefix_from_last_batch_at(
                                  chunk,
                                  a.executor->last_forward_logical_base(),
                                  a.executor->last_forward_rope_base())
                            : a.executor->prime_mtp_prefix_from_last_batch(
                                  chunk, static_cast<uint32_t>(offset));
                    } catch (...) {
                        if (mtp_local) {
                            a.executor->kvmem_set_defer_prefill_pressure(false);
                        }
                        throw;
                    }
                    if (!prefix.ok) {
                        throw std::runtime_error(
                            "MTP prefix priming failed in batched lane");
                    }
                    a.executor->kvmem_prefill_writeback(
                        a.executor->last_forward_logical_base() +
                        static_cast<uint32_t>(chunk.size()));
                    if (mtp_local) {
                        a.executor->kvmem_finish_deferred_prefill_pressure();
                    }
                    offset = end;
                    a.prefill_offset = static_cast<uint32_t>(offset);
                    if (a.prefix_commit_pending &&
                        a.prefill_offset == a.prefix_commit_len) {
                        prefix_cache_commit(a, a.prefix_commit_len);
                        a.prefix_commit_pending = false;
                    }
                }
                if (a.executor->kvmem_mtp_local_positions()) {
                    a.executor->kvmem_set_defer_prefill_pressure(false);
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
                a.prefill_s = prefill_s;
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
                if (a.stream_cancelled ||
                    a.decoded >= req->options.max_tokens ||
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
                    if (!a.req || a.stream_cancelled ||
                        a.decoded >= a.req->options.max_tokens ||
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
                                    a.executor->kvmem_mtp_local_positions()
                                    ? a.executor
                                          ->prime_mtp_prefix_from_last_batch_at(
                                              job.verify_tokens,
                                              job.base_position,
                                              a.executor
                                                  ->last_forward_rope_base(),
                                              mtp_prefix_rebuild_batch_min_tokens())
                                    : a.executor
                                          ->prime_mtp_prefix_from_last_batch(
                                              job.verify_tokens,
                                              job.base_position,
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
                        if (a.stream_cancelled ||
                            a.decoded >= a.req->options.max_tokens) continue;
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
                    if (!a.req || a.stream_cancelled ||
                        a.decoded >= a.req->options.max_tokens ||
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
                                 << " cpu_raw_k=" << tu.cpu_raw_k_bytes
                                 << " cpu_spill=" << tu.cpu_spill_bytes
                                 << " nvme_used=" << tu.nvme_used_bytes
                                 << " nvme_raw_k=" << tu.nvme_raw_k_bytes
                                 << " nvme_spill=" << tu.nvme_spill_bytes
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
        if (options_.kvmem_prefill_budget < 0) {
            throw std::runtime_error(
                "--kvmem-prefill-budget must be >= 0");
        }
        bs_cfg.prefill_budget = static_cast<uint32_t>(
            options_.kvmem_prefill_budget > 0
                ? options_.kvmem_prefill_budget
                : options_.kvmem_budget);
        if (bs_cfg.prefill_budget < bs_cfg.select_budget) {
            throw std::runtime_error(
                "--kvmem-prefill-budget must be >= --kvmem-budget");
        }
        if (bs_cfg.prefill_budget % bs_cfg.block_tokens != 0) {
            throw std::runtime_error(
                "--kvmem-prefill-budget must be divisible by "
                "--kvmem-block-tokens");
        }
        bs_cfg.gen_budget =
            static_cast<uint32_t>(std::max(1, options_.kvmem_gen_budget));
        const KvMemKeepAllocation keep =
            resolve_kvmem_keep_allocation(
                bs_cfg.block_tokens,
                bs_cfg.select_budget,
                options_.kvmem_sink_blocks,
                options_.kvmem_recent_blocks,
                options_.kvmem_sink_tokens,
                options_.kvmem_recent_tokens);
        bs_cfg.sink_blocks = keep.sink_blocks;
        bs_cfg.recent_blocks = keep.recent_blocks;
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
        bs_cfg.raw_k_nvme = options_.kvmem_raw_k_nvme;
        bs_cfg.immutable_source_k = options_.kvmem_immutable_source_k;
        bs_cfg.mtp_enabled =
            mtp_prefix_enabled(options_) || mtp_speculate_enabled(options_);
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
        bs_cfg.retrieval_method =
            options_.kvmem_retrieval_method == "per-token"
                ? KvMemRetrievalMethod::PerToken
                : (options_.kvmem_retrieval_method == "sub-block-mean-k" ||
                   options_.kvmem_retrieval_method ==
                       "key-direction-fixed4" ||
                   options_.kvmem_retrieval_method ==
                       "key-direction-adaptive"
                       ? KvMemRetrievalMethod::SubBlockMeanK
                       : (options_.kvmem_retrieval_method == "deltanet"
                              ? KvMemRetrievalMethod::DeltaNet
                              : KvMemRetrievalMethod::MeanK));
        bs_cfg.index_placement =
            options_.kvmem_index_placement == "cpu"
                ? KvMemIndexPlacement::CPU
                : KvMemIndexPlacement::GPU;
        bs_cfg.index_staging_bytes =
            static_cast<uint64_t>(
                std::max(1, options_.kvmem_index_staging_mb)) *
            1024ull * 1024ull;
        bs_cfg.numa_policy = options_.kvmem_numa_policy;
        bs_cfg.adaptive_score_mode =
            options_.kvmem_adaptive_score_mode == "layer-one-pass"
                ? KvMemAdaptiveScoreMode::LayerOnePass
                : options_.kvmem_adaptive_score_mode ==
                          "tiled-one-pass"
                      ? KvMemAdaptiveScoreMode::TiledOnePass
                      : options_.kvmem_adaptive_score_mode ==
                          "tiled-two-pass"
                      ? KvMemAdaptiveScoreMode::TiledTwoPass
                      : KvMemAdaptiveScoreMode::Auto;
        if (bs_cfg.index_placement == KvMemIndexPlacement::CPU &&
            (bs_cfg.retrieval_method == KvMemRetrievalMethod::PerToken ||
             bs_cfg.retrieval_method == KvMemRetrievalMethod::DeltaNet)) {
            throw std::runtime_error(
                "--kvmem-index-placement cpu currently supports "
                "mean-k, sub-block-mean-k, key-direction-fixed4, "
                "and key-direction-adaptive retrieval");
        }
        if (bs_cfg.index_placement == KvMemIndexPlacement::CPU &&
            !options_.kvmem_query_conditioned) {
            throw std::runtime_error(
                "--kvmem-index-placement cpu requires "
                "--kvmem-query-conditioned");
        }
        bs_cfg.deltanet_layers =
            static_cast<uint32_t>(std::max(0, options_.kvmem_deltanet_layers));
        bs_cfg.deltanet_layer_policy =
            options_.kvmem_deltanet_layer_policy == "late"
                ? KvMemDeltaNetLayerPolicy::Late
                : KvMemDeltaNetLayerPolicy::Even;
        bs_cfg.deltanet_mem_budget_bytes = static_cast<uint64_t>(
            std::max(0.0, options_.kvmem_deltanet_mem_budget_gb) *
            1024.0 * 1024.0 * 1024.0);
        bs_cfg.deltanet_decay = options_.kvmem_deltanet_decay;
        bs_cfg.deltanet_topk_q =
            static_cast<uint32_t>(std::max(1, options_.kvmem_deltanet_topk_q));
        bs_cfg.deltanet_topk_h =
            static_cast<uint32_t>(std::max(1, options_.kvmem_deltanet_topk_h));
        const bool key_direction_fixed4 =
            options_.kvmem_retrieval_method == "key-direction-fixed4";
        const bool key_direction_adaptive =
            options_.kvmem_retrieval_method == "key-direction-adaptive";
        bs_cfg.n_subblocks =
            (key_direction_fixed4 || key_direction_adaptive)
                ? (bs_cfg.block_tokens / 32u) * 4u
                : (bs_cfg.retrieval_method ==
                           KvMemRetrievalMethod::SubBlockMeanK
                       ? static_cast<uint32_t>(
                             std::max(1, options_.kvmem_subblocks))
                       : 1u);
        bs_cfg.prototype_mode =
            key_direction_fixed4
                ? KvMemPrototypeMode::KeyDirectionFixed4
                : key_direction_adaptive
                ? KvMemPrototypeMode::KeyDirectionAdaptive
                : KvMemPrototypeMode::Contiguous;
        bs_cfg.adaptive_gain_1to2 = options_.kvmem_adaptive_gain_1to2;
        bs_cfg.adaptive_gain_2to4 = options_.kvmem_adaptive_gain_2to4;
        if (!std::isfinite(bs_cfg.adaptive_gain_1to2) ||
            !std::isfinite(bs_cfg.adaptive_gain_2to4) ||
            bs_cfg.adaptive_gain_1to2 < 0.0 ||
            bs_cfg.adaptive_gain_1to2 > 1.0 ||
            bs_cfg.adaptive_gain_2to4 < 0.0 ||
            bs_cfg.adaptive_gain_2to4 > 1.0) {
            throw std::runtime_error(
                "adaptive Key-direction gain thresholds must be finite "
                "values in [0,1]");
        }
        bs_cfg.subblock_reduce =
            options_.kvmem_subblock_reduce == "sum"
                ? KvMemSubblockReduce::Sum
                : KvMemSubblockReduce::Max;
        if ((key_direction_fixed4 || key_direction_adaptive) &&
            (bs_cfg.block_tokens < 32 ||
             bs_cfg.block_tokens % 32 != 0 ||
             bs_cfg.subblock_reduce != KvMemSubblockReduce::Max)) {
            throw std::runtime_error(
                "key-direction prototype retrieval currently "
                "requires --kvmem-block-tokens to be a multiple of 32 and "
                "--kvmem-subblock-reduce max");
        }
        if (options_.kvmem_semantic_expansion == "round") {
            bs_cfg.semantic_expansion = KvMemSemanticExpansion::Round;
        } else if (options_.kvmem_semantic_expansion == "message") {
            bs_cfg.semantic_expansion = KvMemSemanticExpansion::Message;
        } else {
            bs_cfg.semantic_expansion = KvMemSemanticExpansion::None;
        }
        if (key_direction_adaptive &&
            bs_cfg.semantic_expansion != KvMemSemanticExpansion::None) {
            throw std::runtime_error(
                "key-direction-adaptive v1 does not yet support "
                "--kvmem-semantic-expansion");
        }
        bs_cfg.group_score_reduce =
            options_.kvmem_group_score_reduce == "length-normalized-mass"
                ? KvMemGroupScoreReduce::LengthNormalizedMass
                : KvMemGroupScoreReduce::Max;
        bs_cfg.group_length_norm_alpha =
            options_.kvmem_group_length_alpha;
        if (bs_cfg.group_length_norm_alpha < 0.0 ||
            bs_cfg.group_length_norm_alpha > 1.0) {
            throw std::runtime_error(
                "--kvmem-group-length-alpha must be in [0,1]");
        }
        if (bs_cfg.semantic_expansion == KvMemSemanticExpansion::None &&
            bs_cfg.group_score_reduce ==
                KvMemGroupScoreReduce::LengthNormalizedMass) {
            throw std::runtime_error(
                "--kvmem-group-score-reduce length-normalized-mass requires "
                "--kvmem-semantic-expansion round|message");
        }
        if (bs_cfg.semantic_expansion != KvMemSemanticExpansion::None &&
            bs_cfg.retrieval_method != KvMemRetrievalMethod::SubBlockMeanK) {
            throw std::runtime_error(
                "--kvmem-semantic-expansion requires "
                "--kvmem-retrieval-method sub-block-mean-k");
        }
        if (bs_cfg.semantic_expansion == KvMemSemanticExpansion::Message) {
            if (bs_cfg.block_tokens % bs_cfg.n_subblocks != 0 ||
                bs_cfg.block_tokens / bs_cfg.n_subblocks != 32) {
                throw std::runtime_error(
                    "--kvmem-semantic-expansion message requires 32-token "
                    "scoring slices: block_tokens / subblocks must equal 32");
            }
        }
        bs_cfg.update_mode = options_.kvmem_update_mode == "step"
            ? KvMemUpdateMode::Step
            : KvMemUpdateMode::Interval;
        bs_cfg.optimize_stage_out = options_.kvmem_opt_stage_out;
        bs_cfg.optimize_stage_in = options_.kvmem_opt_stage_in;
        bs_cfg.optimize_pack = options_.kvmem_opt_pack;
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
            std::vector<int32_t> hit_mtp_pages;
            QwenExecutor::StateSnapshot hit_recur;
            uint32_t hit_len = 0;
            const uint64_t hit_id = prefix_cache_lookup(
                req->prompt_tokens, page_size, req->spec_mtp,
                hit_pages, hit_mtp_pages, hit_recur, hit_len);
            if (hit_id != 0 && hit_len > 0 && hit_len < prompt_len) {
                try {
                    a.executor->seed_from_shared_prefix(
                        hit_pages, hit_mtp_pages, hit_recur, hit_len);
                    a.prefill_offset = hit_len;
                    a.held_prefix_entries.push_back(hit_id);
                    a.kv_state.update(a.executor->kv_state_snapshot());
                    if (prefix_cache_trace_enabled()) {
                        std::ostringstream m;
                        m << "prefix_cache hit id=" << hit_id
                          << " req=" << req->id
                          << " reused_tokens=" << hit_len
                          << " pages=" << hit_pages.size()
                          << " mtp_pages=" << hit_mtp_pages.size();
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
                const uint64_t guard_tokens =
                    static_cast<uint64_t>(prefix_cache_commit_guard_pages()) *
                    static_cast<uint64_t>(page_size);
                if (guard_tokens > 0) {
                    commit_len = guard_tokens < commit_len
                        ? commit_len - static_cast<uint32_t>(guard_tokens)
                        : 0u;
                }
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

            const bool emitted =
                emit_continuous_token(a, a.next_token);
            if (!emitted ||
                a.decoded >= a.req->options.max_tokens) {
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

            const bool emitted =
                emit_continuous_token(a, a.next_token);
            if (!emitted ||
                a.decoded >= a.req->options.max_tokens) {
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
                if (!emit_continuous_token(a, a.next_token) ||
                    a.decoded >= a.req->options.max_tokens) {
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

    bool emit_continuous_token(ContinuousBatchActive &a, uint32_t token) {
        const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(token));
        a.req->generated += piece;
        ++a.seen_tokens[token];
        ++a.decoded;
        if (a.req->on_text && !a.req->on_text(piece)) {
            a.stream_cancelled = true;
            return false;
        }
        return true;
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
        const auto evict_one = [this]() -> uint32_t {
            return prefix_cache_evict_lru(1);
        };
        cb_kv_pool_->set_evict_callback(evict_one);
        // MTP has a separate global page pool. Cached draft pages must be able
        // to trigger the same paired-entry eviction when that pool fills.
        if (cb_mtp_kv_pool_) cb_mtp_kv_pool_->set_evict_callback(evict_one);
        prefix_cache_evict_cb_installed_ = true;
    }

    // A chosen resume point: c = token position to resume at, prompt_ckpt selects
    // ckpt_prompt_ (P) vs ckpt_end_ (M). c==0 means "no reuse -> full reset".
    // Query-conditioned turns ordinarily constrain c to the aligned query
    // replay boundary. A later checkpoint is allowed only when the prior turn
    // also retained a complete, coordinate-matched query snapshot; that mode
    // replays solely the strict-extension suffix from c.
    struct KvmemReuse {
        uint32_t c = 0;
        bool prompt_ckpt = false;
        uint32_t common_prefix = 0;
        uint32_t resume_ceiling = 0;
        uint32_t replay_boundary = 0;
        bool query_limited = false;
        bool query_snapshot_reuse = false;
        bool checkpoint_after_query_boundary = false;
    };

    // Longest-common-prefix reuse pick. Computes D (the longest common token
    // prefix of the warm log and the new prompt) and returns the largest
    // checkpoint C in {M,P} with C <= D and C < prompt.size() (strict, so >=1
    // suffix token always re-prefills and re-seeds decode). On an above-budget
    // query-conditioned turn the ordinary ceiling is
    // B=align_down(query_begin, block_tokens). A durable query snapshot permits
    // C>=query_end instead: selection uses the retained Q and replay starts at C,
    // never in the middle of Q. Preference M over P: reusing at M skips
    // re-decoding as well as re-prefilling. Semantic selection is active only
    // above budget. Below budget the request remains dense/identity, while a
    // prefix-cached Query may still constrain reuse enough to keep its prepared
    // Q/source-index artifacts valid for a later crossing.
    KvmemReuse kvmem_prefix_reuse(const std::vector<uint32_t> &prompt,
                                  const GenerationOptions &options) {
#if 0  // Archived DeltaNet recurrent-state artifact requests.
        if (!options.kvmem_rebuilt_state_export_key.empty() ||
            !options.kvmem_rebuilt_state_import_key.empty() ||
            !options.kvmem_rebuilt_state_capture_key.empty() ||
            !options.kvmem_rebuilt_state_seed_key.empty()) return {};
#endif
        // Oracle selection is a cold, controlled representation experiment.
        // Do not let an earlier request's checkpoint/working set become another
        // variable in the direct-KV result.
        if (!options.kvmem_oracle_token_spans.empty()) return {};
        if (options.kvmem_inline_refresh !=
            KvMemInlineRefreshMode::Off) return {};
        if (!kvmem_prefix_cache_enabled()) return {};
        if (!executor_ || !executor_->kvmem_enabled()) return {};
        if (!kvmem_warm_valid_ || kvmem_warm_log_.empty()) return {};
        const uint32_t sel_budget = executor_->block_store()
            ? executor_->block_store()->select_budget_tokens() : 0;
        const bool qc_source_index_supported =
            executor_->block_store() &&
            kvmem_query_replay_retrieval_supported(
                executor_->block_store()->config().retrieval_method);
        const KvmemQueryLifecycle qc = kvmem_query_lifecycle(
            options.kvmem_query_end > options.kvmem_query_begin,
            static_cast<uint32_t>(prompt.size()), sel_budget,
            options.kvmem_reselect_mode != KvMemReselectMode::Off,
            /*warm_capture=*/true, qc_source_index_supported);
        // Preparation needs the same checkpoint safety properties as active
        // selection: a post-query checkpoint must restore Q, and its source
        // index must be incrementally extendable from that exact position.
        const bool qc_resume_guard = qc.capture_active();
        // Above-budget session reuse rebuilds only the suffix's slice of the
        // position-invariant mean-k content index (g_kbar_multi_, fixed-stride).
        // The per-token raw-key index (g_kraw_multi_, --kvmem-retrieval-method
        // per-token) is strided by the turn's TOTAL token count and can't be
        // fixed-stride at large --ctx, so above budget we refuse reuse in that mode
        // and fall back to a full cold prefill (correct, just unoptimized). Below
        // budget the index is unused, so the guard only bites above budget.
        const size_t lim = std::min(kvmem_warm_log_.size(), prompt.size());
        size_t D = 0;
        while (D < lim && kvmem_warm_log_[D] == prompt[D]) ++D;
        const uint32_t common_prefix = static_cast<uint32_t>(
            std::min<size_t>(D, std::numeric_limits<uint32_t>::max()));
        // A stashed query is valid only for the exact same token coordinates,
        // and only while the complete query remains inside the new prompt's
        // exact LCP.  This is the extra authority needed to resume after Q:
        // the historical source index supplies K, while the stash supplies the
        // otherwise-unrecoverable Q rows.
        const bool qc_query_snapshot_ready =
            qc_resume_guard && kvmem_warm_query_stashed_ &&
            kvmem_warm_query_begin_ == options.kvmem_query_begin &&
            kvmem_warm_query_end_ == options.kvmem_query_end &&
            options.kvmem_query_end <= common_prefix &&
            kvmem_recompute_query_enabled(options_.kvmem_recompute_query) &&
            options.kvmem_query_guided_thinking_max_tokens == 0 &&
            options.kvmem_query_guided_query_max_tokens == 0 &&
            !options.kvmem_query_guided_direct;
        const bool qc_pertoken_block =
            qc_resume_guard && executor_ && executor_->kvmem_qc_pertoken();
        const bool qc_end_checkpoint_block =
            qc_pertoken_block ||
            (qc_resume_guard && (!kvmem_warm_end_resumable_ ||
                           !kvmem_warm_end_source_index_ready_));
        const bool qc_prompt_checkpoint_block =
            qc_pertoken_block ||
            (qc_resume_guard && (!kvmem_warm_prompt_resumable_ ||
                           !kvmem_warm_source_index_ready_));
        const uint32_t bt = executor_->block_store()
            ? std::max<uint32_t>(
                  1, executor_->block_store()->config().block_tokens)
            : 1;
        const uint32_t replay_boundary = qc_resume_guard
            ? (options.kvmem_query_begin / bt) * bt
            : 0;
        const uint32_t query_boundary_ceiling = kvmem_prefix_resume_ceiling(
            common_prefix, qc_resume_guard, options.kvmem_query_begin, bt);
        const uint32_t resume_ceiling = qc_query_snapshot_ready
            ? common_prefix : query_boundary_ceiling;
        KvmemReuse result;
        result.common_prefix = common_prefix;
        result.resume_ceiling = resume_ceiling;
        result.replay_boundary = replay_boundary;
        result.query_limited = qc_resume_guard;
        const auto checkpoint_reusable =
            [&](uint32_t position) {
                return kvmem_prefix_checkpoint_reusable(
                           position, static_cast<uint32_t>(prompt.size()),
                           query_boundary_ceiling) ||
                    kvmem_post_query_checkpoint_reusable(
                           position, static_cast<uint32_t>(prompt.size()),
                           common_prefix, options.kvmem_query_end,
                           qc_query_snapshot_ready);
            };
        const auto checkpoint_crosses_query_boundary =
            [&](uint32_t position, bool resumable) {
                return qc_resume_guard && resumable &&
                    kvmem_prefix_checkpoint_reusable(
                        position, static_cast<uint32_t>(prompt.size()),
                        common_prefix) &&
                    position > query_boundary_ceiling &&
                    !kvmem_post_query_checkpoint_reusable(
                        position, static_cast<uint32_t>(prompt.size()),
                        common_prefix, options.kvmem_query_end,
                        qc_query_snapshot_ready);
            };
        result.checkpoint_after_query_boundary =
            checkpoint_crosses_query_boundary(
                kvmem_warm_end_pos_, !qc_end_checkpoint_block) ||
            checkpoint_crosses_query_boundary(
                kvmem_warm_prompt_pos_,
                !qc_prompt_checkpoint_block);
        // M (end-of-turn resume) is offered whenever its source index can be
        // extended from that exact position (except per-token QC above budget):
        // the resident [0,D) blocks stay canonically positioned
        // across the warm session, so above budget we prefill only the trailing new
        // question instead of the whole history (server-side session continuation).
        // Below budget behavior is byte-identical (M offered exactly as before).
        if (!qc_end_checkpoint_block &&
            checkpoint_reusable(kvmem_warm_end_pos_)) {
            result.c = kvmem_warm_end_pos_;
            result.prompt_ckpt = false;
            result.query_snapshot_reuse =
                qc_resume_guard && result.c > query_boundary_ceiling;
            return result;
        }
        // P (prompt-end resume) is offered at a BLOCK BOUNDARY below the prompt
        // end. Below budget that requires every block GPU-resident at identity
        // position (kvmem_all_gpu_identity, folded into
        // kvmem_warm_prompt_resumable_). Above budget the chat template drops the
        // per-turn empty <think> block from history, so D lands a few tokens below
        // M every turn and M never fires; the block-boundary P checkpoint sits
        // below that per-turn rewrite (B <= D), so offering it here is what turns
        // "prefill only the new question" on for real chat clients. Resumability
        // above budget is keyed on tiers keeping [0,B) recoverable + the mean-k
        // index being position-invariant (captured at capture time); the per-token
        // index can't be fixed-stride, so qc_pertoken_block still refuses.
        if (!qc_prompt_checkpoint_block &&
            checkpoint_reusable(kvmem_warm_prompt_pos_)) {
            result.c = kvmem_warm_prompt_pos_;
            result.prompt_ckpt = true;
            result.query_snapshot_reuse =
                qc_resume_guard && result.c > query_boundary_ceiling;
            return result;
        }
        // Miss despite a valid warm log: log why so the miss pattern is
        // diagnosable. D<P means divergence landed below the prompt-end
        // checkpoint (tokenization boundary or upstream prefix instability);
        // D far below both means an unrelated request overwrote the warm slot.
        if (kvmem_prefix_cache_trace_enabled() ||
            std::getenv("QW3_KVMEM_TRACE")) {
            std::ostringstream mmsg;
            mmsg << "kvmem prefix-cache MISS (why): D=" << D
                 << " ceiling=" << resume_ceiling
                 << " query_limited=" << (qc_resume_guard ? 1 : 0)
                 << " query_snapshot="
                 << (qc_query_snapshot_ready ? 1 : 0)
                 << " replay_boundary=" << replay_boundary
                 << " P=" << kvmem_warm_prompt_pos_
                 << " M=" << kvmem_warm_end_pos_
                 << " P_resumable=" << (kvmem_warm_prompt_resumable_ ? 1 : 0)
                 << " P_source_index="
                 << (kvmem_warm_source_index_ready_ ? 1 : 0)
                 << " M_resumable=" << (kvmem_warm_end_resumable_ ? 1 : 0)
                 << " M_source_index="
                 << (kvmem_warm_end_source_index_ready_ ? 1 : 0)
                 << " warm_log=" << kvmem_warm_log_.size()
                 << " prompt=" << prompt.size()
                 << " reason="
                 << (result.checkpoint_after_query_boundary
                         ? "checkpoint_after_query_boundary"
                         : "no_reusable_checkpoint");
            log(mmsg.str());
        }
        return result;
    }

    // True when the live block store is cleanly resumable at a prompt-end
    // checkpoint: every block is GPU-resident at its true position (identity
    // re-RoPE, baked_pos == orig_pos_start). Gated on actual runtime residency,
    // not on whether tiers are *configured* — below budget a tiered store keeps
    // everything GPU-resident with identity bakes, so ckpt_P is safe there.
    // Evaluated at ckpt_prompt_ capture time. Once anything spills (offload to
    // CPU/NVMe or a sparse budget re-RoPEs a block to a window slot) the scan
    // returns false -> ckpt_prompt_ not offered, and reuse falls back to
    // ckpt_end_ only.
    bool kvmem_all_gpu_identity() const {
        const KvMemStore *bs = executor_->block_store();
        if (!bs) return false;
        for (const auto &b : bs->blocks()) {
            if (b.tier != KvTier::GPU ||
                b.baked_pos != static_cast<int64_t>(b.orig_pos_start)) {
                return false;
            }
        }
        return true;
    }

    // Decide whether a prompt checkpoint contains enough durable state to be
    // restored before a future semantic query.  Dense/all-fit checkpoints need
    // no source index until they cross the budget, while sparse checkpoints
    // require tier backing plus the fixed-stride Mean-K family index captured
    // during the history request itself.
    bool kvmem_prompt_checkpoint_resumable(
            bool source_index_ready) const {
        const KvMemStore *bs = executor_ ? executor_->block_store() : nullptr;
        if (!bs) return false;
        return kvmem_sparse_prompt_checkpoint_resumable(
            kvmem_all_gpu_identity(), executor_->kvmem_has_tiers(),
            source_index_ready,
            kvmem_query_replay_retrieval_supported(
                bs->config().retrieval_method));
    }

    // Persist the complete de-RoPE'd query alongside the warm P/M checkpoints.
    // StateSnapshot intentionally owns only model-compute state; the query is a
    // scorer input with a backend-specific host/GPU layout, so QwenExecutor's
    // existing clean-query stash is the durable authority.  Metadata is stored
    // here and checked against the next request's exact LCP before restoration.
    bool kvmem_capture_warm_query_snapshot(
            QwenExecutor &executor, const GenerationOptions &options,
            bool eligible) {
        kvmem_warm_query_stashed_ = false;
        kvmem_warm_query_begin_ = 0;
        kvmem_warm_query_end_ = 0;
        if (!eligible ||
            options.kvmem_query_end <= options.kvmem_query_begin ||
            !executor.kvmem_query_capture_complete()) {
            return false;
        }
        try {
            if (!executor.kvmem_stash_query()) return false;
        } catch (const std::exception &e) {
            if (kvmem_prefix_cache_trace_enabled() ||
                std::getenv("QW3_KVMEM_TRACE")) {
                log(std::string("kvmem prefix-cache query snapshot skipped: ") +
                    e.what());
            }
            return false;
        }
        kvmem_warm_query_stashed_ = true;
        kvmem_warm_query_begin_ = options.kvmem_query_begin;
        kvmem_warm_query_end_ = options.kvmem_query_end;
        return true;
    }

    uint64_t prefix_cache_lookup(
            const std::vector<uint32_t> &prompt,
            uint32_t page_size,
            bool require_mtp,
            std::vector<int32_t> &out_pages,
            std::vector<int32_t> &out_mtp_pages,
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
                const uint32_t prefix_pages = probe / page_size;
                if (require_mtp &&
                    (e.mtp_kv_pages.size() != prefix_pages ||
                     e.recur.mtp_kv_logical_pages != prefix_pages ||
                     e.recur.mtp_prefix_len < probe ||
                     !e.recur.mtp_prefix_h)) {
                    continue;
                }
                // Hit. Pin already held by the entry; bump refcount + LRU.
                ++e.refcount;
                e.last_used_seq = ++prefix_cache_seq_;
                out_pages = e.kv_pages;
                out_mtp_pages = e.mtp_kv_pages;
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
        dst.mtp_kv_logical_pages = src.mtp_kv_logical_pages;
        dst.mtp_prefix_len = src.mtp_prefix_len;
        if (src.h) {
            dst.h = device_->scratch_like(
                *src.h, src.h->count, "prefix_clone_h");
            prefix_require_ok(device_->copy_d2d(*dst.h, *src.h, 0, src.h->count));
        }
        if (src.mtp_prefix_h) {
            dst.mtp_prefix_h = device_->scratch_like(
                *src.mtp_prefix_h, src.mtp_prefix_h->count,
                "prefix_clone_mtp_h");
            prefix_require_ok(device_->copy_d2d(
                *dst.mtp_prefix_h, *src.mtp_prefix_h, 0,
                src.mtp_prefix_h->count));
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

    // Complete clone used by the in-process KVMem milestone harness. The
    // continuous-batching prefix cache above clones the dense model state,
    // including MTP prefix hidden state, while its main/MTP KV pages are pinned
    // separately. A milestone branch additionally preserves registration and
    // sparse-window metadata.
    QwenExecutor::StateSnapshot clone_milestone_state_snapshot(
            const QwenExecutor::StateSnapshot &src) {
        QwenExecutor::StateSnapshot dst = clone_state_snapshot(src);
        dst.kvmem_registered_pos = src.kvmem_registered_pos;
        dst.kvmem_active = src.kvmem_active;
        dst.window_query_pos = src.window_query_pos;
        dst.window_page_count = src.window_page_count;
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
        const bool include_mtp = a.req && a.req->spec_mtp;

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
                                   prompt.begin()) &&
                        (!include_mtp ||
                         (e.mtp_kv_pages.size() == prefix_pages &&
                          e.recur.mtp_kv_logical_pages == prefix_pages &&
                          e.recur.mtp_prefix_len >= aligned_len &&
                          e.recur.mtp_prefix_h))) {
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
        if (!a.executor->mark_prefix_shared(
                prefix_pages, include_mtp,
                entry.kv_pages, entry.mtp_kv_pages)) {
            return;
        }
        if (cb_kv_pool_) cb_kv_pool_->pin_pages(entry.kv_pages);
        if (cb_mtp_kv_pool_ && !entry.mtp_kv_pages.empty()) {
            cb_mtp_kv_pool_->pin_pages(entry.mtp_kv_pages);
        }

        uint64_t eid = 0;
        {
            std::lock_guard<std::mutex> lk(prefix_cache_mu_);
            entry.id = prefix_cache_next_id_++;
            entry.refcount = 1;  // creator holds it until finish
            entry.last_used_seq = ++prefix_cache_seq_;
            prefix_cache_pinned_pages_ += prefix_pages;
            ++prefix_cache_entry_count_;
            a.held_prefix_entries.push_back(entry.id);
            eid = entry.id;
            const uint32_t alen = entry.aligned_len;
            const size_t mtp_page_count = entry.mtp_kv_pages.size();
            prefix_cache_[h].push_back(std::move(entry));
            if (prefix_cache_trace_enabled()) {
                std::ostringstream m;
                m << "prefix_cache commit id=" << eid
                  << " req=" << a.req->id
                  << " aligned_len=" << alen
                  << " pages=" << prefix_pages
                  << " mtp_pages=" << mtp_page_count
                  << " pinned_pages=" << prefix_cache_pinned_pages_
                  << " entries=" << prefix_cache_entry_count_;
                log(m.str());
            }
        }
        (void) eid;
        // The just-committed entry has refcount 1, so it cannot evict itself.
        prefix_cache_enforce_budgets();
    }

    // Enforce both physical-page and state-snapshot budgets. This is called
    // after commit and again after releasing request references: concurrent
    // requests may temporarily make every over-budget entry non-evictable,
    // but the release pass reclaims them as soon as they become idle.
    void prefix_cache_enforce_budgets() {
        const uint32_t page_budget = prefix_cache_max_pages();
        const uint32_t entry_budget = prefix_cache_max_entries();
        if (page_budget == 0 && entry_budget == 0) return;
        for (int guard = 0; guard < 4096; ++guard) {
            uint32_t pinned = 0;
            uint32_t entries = 0;
            {
                std::lock_guard<std::mutex> lk(prefix_cache_mu_);
                pinned = prefix_cache_pinned_pages_;
                entries = prefix_cache_entry_count_;
            }
            const bool pages_ok = page_budget == 0 || pinned <= page_budget;
            const bool entries_ok =
                entry_budget == 0 || entries <= entry_budget;
            if (pages_ok && entries_ok) break;
            if (prefix_cache_evict_lru(1) == 0) break;
        }
    }

    // Drop refcounts this request holds. Pinned pages are NOT freed here; they
    // stay until the entry is evicted at refcount 0.
    void prefix_cache_release(ContinuousBatchActive &a) {
        if (a.held_prefix_entries.empty()) return;
        {
            std::lock_guard<std::mutex> lk(prefix_cache_mu_);
            for (uint64_t eid : a.held_prefix_entries) {
                for (auto &kv : prefix_cache_) {
                    for (PrefixCacheEntry &e : kv.second) {
                        if (e.id == eid && e.refcount > 0) {
                            --e.refcount;
                            break;
                        }
                    }
                }
            }
            a.held_prefix_entries.clear();
        }
        prefix_cache_enforce_budgets();
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
            if (cb_mtp_kv_pool_ && !victim.mtp_kv_pages.empty()) {
                cb_mtp_kv_pool_->unpin_pages(victim.mtp_kv_pages);
                cb_mtp_kv_pool_->release_physical_pages(
                    victim.mtp_kv_pages);
            }
            const uint32_t pages = static_cast<uint32_t>(victim.kv_pages.size());
            ++evicted_entries;
            if (prefix_cache_entry_count_ > 0) {
                --prefix_cache_entry_count_;
            }
            if (prefix_cache_pinned_pages_ >= pages) {
                prefix_cache_pinned_pages_ -= pages;
            }
            if (prefix_cache_trace_enabled()) {
                std::ostringstream m;
                m << "prefix_cache evict id=" << victim.id
                  << " aligned_len=" << victim.aligned_len
                  << " pages=" << pages
                  << " mtp_pages=" << victim.mtp_kv_pages.size()
                  << " pinned_pages=" << prefix_cache_pinned_pages_
                  << " entries=" << prefix_cache_entry_count_;
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
                if (cb_mtp_kv_pool_ && !e.mtp_kv_pages.empty()) {
                    cb_mtp_kv_pool_->unpin_pages(e.mtp_kv_pages);
                    cb_mtp_kv_pool_->release_physical_pages(e.mtp_kv_pages);
                }
            }
        }
        prefix_cache_.clear();
        prefix_cache_pinned_pages_ = 0;
        prefix_cache_entry_count_ = 0;
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
        if (a.stream_cancelled) msg << " cancelled=true";
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
        state.recover_eos = options.recover_thinking_eos && !options.ignore_eos;
        if (tokenizer_) {
            state.eos_id = tokenizer_->eos_id();
        }
        if ((state.budget > 0 || state.recover_eos) && tokenizer_) {
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
        if (!state.open || state.close_id < 0) return;
        if (static_cast<int>(token) == state.close_id) {
            state.open = false;
            state.forced = false;
            state.forced_queue.clear();
            return;
        }
        if (!state.active()) return;
        ++state.think_tokens;
    }

    // If the budget is exhausted while still inside <think>, enqueue the forced
    // guidance+close tokens. Returns the next token to FEED in place of the
    // model's own pick: the front of the forced queue, or `proposed` when no
    // override is active. The caller emits the returned token and feeds it back
    // into the model so the KV cache stays consistent.
    uint32_t budget_next_feed(ThinkingBudgetState &state, uint32_t proposed) const {
        if (state.can_recover_eos(proposed)) {
            state.open = false;
            state.forced = false;
            state.forced_queue.clear();
            return static_cast<uint32_t>(state.close_id);
        }
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

    void recover_thinking_eos(ThinkingBudgetState &state, uint32_t &token) const {
        if (!state.can_recover_eos(token)) return;
        token = static_cast<uint32_t>(state.close_id);
        state.open = false;
        state.forced = false;
        state.forced_queue.clear();
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
                               const CancellableTokenCallback &on_text,
                               DumpStream *dump) {
        const bool semantic_chunk =
            options.kvmem_prefill_window_mode ==
            KvMemPrefillWindowMode::SemanticChunk;
        if (!options.kvmem_replay_query_spans.empty()) {
            throw std::runtime_error(
                "KVMem transcript replay currently requires native MTP");
        }
        const bool inline_refresh =
            options.kvmem_inline_refresh !=
            KvMemInlineRefreshMode::Off;
        if (inline_refresh &&
            (dump != nullptr || !options.kvmem_session_id.empty())) {
            throw std::runtime_error(
                "KVMem inline refresh requires a standalone one-shot "
                "request without logit dumping");
        }
        std::unique_ptr<ScopedKvmemDisable> inline_refresh_guard;
#if 0  // Archived DeltaNet recurrent-state artifact experiment.
        const bool rebuilt_state_export =
            !options.kvmem_rebuilt_state_export_key.empty();
        const bool rebuilt_state_import =
            !options.kvmem_rebuilt_state_import_key.empty();
        const bool rebuilt_state_capture =
            !options.kvmem_rebuilt_state_capture_key.empty();
        const bool rebuilt_state_seed =
            !options.kvmem_rebuilt_state_seed_key.empty();
        if (rebuilt_state_export && options.max_tokens != 0) {
            throw std::runtime_error(
                "KVMem rebuilt-state export requires max_tokens=0");
        }
#endif
        DeviceStatus st = device_->begin();
        if (!st.ok) throw std::runtime_error(st.message);

        // kvmem prefix cache (QW3_KVMEM_PREFIX_CACHE): resume the warm executor at
        // the longest reusable checkpoint (turn-end M or prompt-end P) that is a
        // token prefix of the new prompt, and prefill only the trailing suffix.
        // Otherwise fall back to the byte-identical reset+full-prefill path and
        // (if the flag is on) drop the now-stale warm checkpoints.
        const KvmemReuse ru = semantic_chunk
            ? KvmemReuse{}
            : kvmem_prefix_reuse(prompt_tokens, options);
        const uint32_t reuse_m = ru.c;
        const bool warm_reuse = reuse_m > 0;
        if (warm_reuse) {
            executor_->restore_state(ru.prompt_ckpt ? kvmem_warm_ckpt_prompt_
                                                    : kvmem_warm_ckpt_end_);
            // restore_state rewound position/KV/recurrent/window to C but not the
            // block store (still at the prior turn's end M). Rewind it to C.
            executor_->kvmem_truncate_to(reuse_m);
            if (kvmem_prefix_cache_trace_enabled()) {
                std::ostringstream tmsg;
                tmsg << "kvmem prefix-cache HIT (plain): reuse=" << reuse_m
                     << " ckpt=" << (ru.prompt_ckpt ? "P" : "M")
                     << " D=" << ru.common_prefix
                     << " ceiling=" << ru.resume_ceiling
                     << " replay_boundary=" << ru.replay_boundary
                     << " query_limited=" << (ru.query_limited ? 1 : 0)
                     << " query_snapshot="
                     << (ru.query_snapshot_reuse ? 1 : 0)
                     << " prompt=" << prompt_tokens.size()
                     << " suffix=" << (prompt_tokens.size() - reuse_m);
                log(tmsg.str());
            }
        } else {
            executor_->reset_state();
            if (kvmem_prefix_cache_enabled()) kvmem_warm_valid_ = false;
        }
#if 0  // Archived DeltaNet recurrent-state seed.
        if (rebuilt_state_seed) {
            if (warm_reuse) {
                throw std::runtime_error(
                    "KVMem rebuilt-state seed requires a cold full prefill");
            }
            // A fresh server has not executed its first token yet, so the
            // per-layer recurrent/conv tensors may still be lazily unallocated.
            // Materialize them before validating the captured 64-layer state.
            executor_->prepare_runtime_state();
            executor_->kvmem_import_recurrent_state(
                kvmem_rebuilt_state_path(
                    options.kvmem_rebuilt_state_seed_key),
                prompt_tokens);
            log("native kvmem rebuilt-state seed (plain): key=" +
                options.kvmem_rebuilt_state_seed_key +
                " identity_tokens=" + std::to_string(prompt_tokens.size()));
        }
#endif
        const bool warm_capture =
            !semantic_chunk && !inline_refresh &&
            kvmem_prefix_cache_enabled() &&
            executor_->kvmem_enabled();

        // "Operating dense" predicate: below budget the store keeps every block
        // GPU-resident in identity order, so selection/QC are no-ops and the
        // request behaves exactly like the dense (budget==ctx) config. Above
        // budget the tiered/sparse/QC machinery actually engages.
        const uint32_t kvmem_sel_budget =
            (executor_->kvmem_enabled() && executor_->block_store())
                ? executor_->block_store()->select_budget_tokens() : 0;
        const bool kvmem_below_budget =
            kvmem_sel_budget > 0 && prompt_tokens.size() <= kvmem_sel_budget;
        (void)kvmem_below_budget;
        const bool qc_source_index_supported =
            executor_->block_store() &&
            kvmem_query_replay_retrieval_supported(
                executor_->block_store()->config().retrieval_method);
        const KvmemQueryLifecycle qc = kvmem_query_lifecycle(
            options.kvmem_query_end > options.kvmem_query_begin,
            static_cast<uint32_t>(prompt_tokens.size()), kvmem_sel_budget,
            options.kvmem_reselect_mode != KvMemReselectMode::Off,
            warm_capture, qc_source_index_supported);
        const bool qc_select_active = qc.select_active;
        const bool qc_prepare = qc.prepare;
        const bool qc_capture_active = qc.capture_active();
        const bool warm_history_index_capture =
            warm_capture && !qc_capture_active && kvmem_sel_budget > 0 &&
            prompt_tokens.size() > kvmem_sel_budget &&
            qc_source_index_supported;
        const bool warm_source_index_ready =
            (qc_capture_active && qc_source_index_supported) ||
            warm_history_index_capture;

        if (semantic_chunk) {
            if (!executor_->kvmem_enabled() || !qc_select_active ||
                dump != nullptr ||
                inline_refresh || !options.kvmem_session_id.empty() ||
                !executor_->block_store() ||
                !kvmem_query_replay_retrieval_supported(
                    executor_->block_store()->config().retrieval_method)) {
                throw std::runtime_error(
                    "semantic-chunk prefill requires a fresh, above-budget, "
                    "query-conditioned Mean-K/SubBlockMeanK request");
            }
        }

        if (executor_->kvmem_enabled()) {
            executor_->kvmem_set_keep_selected_prefill(
                options.kvmem_prefill_window_mode ==
                KvMemPrefillWindowMode::KeepSelected);
            executor_->kvmem_set_trace_metadata(
                options.kvmem_trace_tag,
                options.kvmem_context_begin,
                options.kvmem_context_end,
                prompt_tokens);
            std::vector<std::pair<uint32_t, uint32_t>> retrieval_groups;
            retrieval_groups.reserve(
                options.kvmem_retrieval_group_spans.size());
            for (const auto &span :
                 options.kvmem_retrieval_group_spans) {
                retrieval_groups.emplace_back(span.begin, span.end);
            }
            executor_->kvmem_set_retrieval_group_spans(
                retrieval_groups);
            std::vector<std::pair<uint32_t, uint32_t>> mandatory_spans;
            mandatory_spans.reserve(
                options.kvmem_pinned_token_spans.size());
            for (const auto &span : options.kvmem_pinned_token_spans) {
                mandatory_spans.emplace_back(span.begin, span.end);
            }
            executor_->kvmem_set_mandatory_token_spans(mandatory_spans);
        }
        const uint32_t query_replay_base = executor_->position();
        const bool query_replay =
            !semantic_chunk && kvmem_query_replay_enabled() &&
            executor_->kvmem_enabled() &&
            qc_select_active && dump == nullptr &&
            options.kvmem_query_begin > 0 &&
            options.kvmem_query_begin >= query_replay_base &&
            options.kvmem_query_end <= prompt_tokens.size() &&
            prompt_tokens.size() - options.kvmem_query_begin <=
                executor_->block_store()->config().gen_budget &&
            executor_->block_store()->config().retrieval_method !=
                KvMemRetrievalMethod::DeltaNet;
        if (!semantic_chunk && kvmem_query_replay_enabled() && !query_replay) {
            std::ostringstream rmsg;
            rmsg << "native kvmem query-replay skipped (plain): base="
                 << query_replay_base << " span=[" << options.kvmem_query_begin
                 << "," << options.kvmem_query_end << ") prompt="
                 << prompt_tokens.size() << " gen_budget="
                 << (executor_->block_store()
                          ? executor_->block_store()->config().gen_budget
                          : 0);
            log(rmsg.str());
        }

        const bool query_attention_probe_requested =
            options.kvmem_query_attention_probe_tokens != 0 ||
            options.kvmem_query_attention_score_tokens != 0;
        const bool query_guided_query_requested =
            options.kvmem_query_guided_thinking_max_tokens != 0 ||
            options.kvmem_query_guided_query_max_tokens != 0 ||
            options.kvmem_query_guided_direct;
        if (query_attention_probe_requested &&
            (options.kvmem_query_attention_probe_tokens == 0 ||
             options.kvmem_query_attention_score_tokens == 0)) {
            throw std::invalid_argument(
                "KVMem query attention probe requires both a positive probe "
                "length and a positive score-token budget");
        }
        if (query_guided_query_requested) {
            throw std::invalid_argument(
                "KVMem guided query currently requires the MTP archive "
                "query path");
        }
        const bool query_attention_probe = query_attention_probe_requested;
        executor_->kvmem_set_query_score_budget_hint(
            query_attention_probe
                ? options.kvmem_query_attention_score_tokens
                : 0);
        // Capture is deliberately broader than selection. A prefix-cached
        // below-budget request prepares Q + the fixed-stride source index, but
        // remains dense/identity; only qc_select_active may select/replay.
        if (executor_->kvmem_enabled() && qc_capture_active) {
            executor_->kvmem_set_pin_from_block(0xffffffffu);
            executor_->kvmem_set_query_span(options.kvmem_query_begin,
                                            options.kvmem_query_end,
                                            static_cast<uint32_t>(prompt_tokens.size()),
                                            /*index_tokens=*/0,
                                            /*preserve_content_index=*/warm_reuse);
            if (ru.query_snapshot_reuse) {
                executor_->kvmem_restore_stashed_query();
            }
            std::ostringstream qmsg;
            qmsg << "native kvmem query-conditioned: mode="
                 << (qc_select_active ? "select" : "prepare")
                 << " span=["
                 << options.kvmem_query_begin << "," << options.kvmem_query_end
                 << ") tokens=" << (options.kvmem_query_end - options.kvmem_query_begin);
            log(qmsg.str());
        } else if (executor_->kvmem_enabled()) {
            // Below budget (or no span): clear any residual span/g_query_multi
            // so kvmem_prepare_reselect keeps the single-token identity path.
            executor_->kvmem_set_query_span(
                0, 0, static_cast<uint32_t>(prompt_tokens.size()),
                /*index_tokens=*/0,
                /*preserve_content_index=*/false,
                /*capture_content_without_query=*/
                    warm_history_index_capture);
        }

        const double t_prefill_start = wall_seconds();
#ifdef QW3_ENABLE_CUDA
        const bool profile_cuda_prefill =
            env_flag_enabled("QW3_CUDA_PROFILE_PREFILL");
        if (profile_cuda_prefill) {
            st = device_->synchronize();
            if (!st.ok) throw std::runtime_error(st.message);
            const cudaError_t profile_st = cudaProfilerStart();
            if (profile_st != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaProfilerStart failed: ") +
                    cudaGetErrorString(profile_st));
            }
        }
#endif
        uint64_t prefill_ops = 0;
        NativeExecutorReport step;
        // On warm reuse the [0,reuse_m) prefix is already resident (restored
        // above); prefill only the trailing suffix. reuse_m<prompt.size() is
        // guaranteed by the strict-extend predicate, so >=1 token still forwards
        // and step carries a fresh decode seed.
        const size_t prefill_begin = warm_reuse ? reuse_m : 0;
        // Above-budget continuation (P or M checkpoint): rebuild the prefill
        // working window before the first suffix token. Cold/all-fit requests
        // are no-ops here and enter pressure mode only when the pool fills.
        executor_->kvmem_prepare_prefill_window(
            static_cast<uint32_t>(prompt_tokens.size() - prefill_begin));
        // kvmem prefix cache: choose where to stage the prompt-end (P) checkpoint.
        // Capturing at the last BLOCK boundary strictly below P leaves a resume
        // point below the chat-template reformat that rewrites the final few
        // tokens each turn (observed D == P-4: the empty <think> block the
        // generation prompt injects for the live turn is dropped once that turn
        // becomes history). Dense/untiered OR below-budget tiered keep [0,B)
        // GPU-resident with identity bakes (the below-budget ckpt_P resumability
        // precondition). Above budget with tiers the same block boundary is the
        // resume point that makes session continuation actually fire: [0,B) stays
        // recoverable across tiers and the mean-k content index is
        // position-invariant, so the boundary snapshot is resumable there too.
        // The per-token index can't be fixed-stride, so above-budget per-token
        // still declines (captured as a non-resumable P below).
        const bool qc_pertoken_here = executor_->kvmem_qc_pertoken();
        const bool recompute_query =
            executor_->kvmem_enabled() && qc_select_active &&
            (semantic_chunk ||
             kvmem_recompute_query_enabled(options_.kvmem_recompute_query)) &&
            dump == nullptr && executor_->block_store() &&
            kvmem_query_replay_retrieval_supported(
                executor_->block_store()->config().retrieval_method);
        if (query_attention_probe && !recompute_query) {
            throw std::invalid_argument(
                "KVMem query attention probe requires the query-recompute/"
                "replay path with Mean-K or SubBlockMeanK retrieval");
        }
        if (inline_refresh && !recompute_query) {
            throw std::runtime_error(
                "KVMem inline refresh requires an above-budget, "
                "query-conditioned mean-k request with query replay enabled");
        }
#if 0  // Archived DeltaNet recurrent-state import/capture validation.
        if ((rebuilt_state_import || rebuilt_state_capture) &&
            !recompute_query) {
            throw std::runtime_error(
                "KVMem rebuilt-state import/capture requires an above-budget "
                "query-conditioned mean-k request with query replay enabled");
        }
#endif
        QwenExecutor::StateSnapshot query_replay_ckpt;
        uint32_t query_replay_begin = 0;
        if (recompute_query) {
            const uint32_t bt = std::max<uint32_t>(
                1, executor_->block_store()->config().block_tokens);
            query_replay_begin = ru.query_snapshot_reuse
                ? reuse_m
                : (options.kvmem_query_begin / bt) * bt;
            if (query_replay_begin == 0) {
                throw std::runtime_error(
                    "KVMem query replay requires a non-zero aligned boundary");
            }
        }
        uint32_t warm_prompt_pos = 0;
        bool warm_prompt_resumable = false;
        uint32_t ckpt_split = 0;
        bool do_boundary_capture = false;
        bool warm_checkpoint_staged = false;
        if (warm_capture && executor_->kvmem_enabled() && !qc_pertoken_here) {
            const uint32_t bt = executor_->block_store()
                                    ? executor_->block_store()->config().block_tokens
                                    : 256;
            const uint32_t prompt_end = static_cast<uint32_t>(prompt_tokens.size());
            const uint32_t split = kvmem_prompt_checkpoint_split(
                prompt_end, static_cast<uint32_t>(prefill_begin), bt);
            if (split > 0) {
                ckpt_split = split;
                do_boundary_capture = true;
            }
        }
        // Prefill the global range [gbegin, gend) of prompt_tokens. Factored out so
        // the block-boundary checkpoint can split the prompt prefill at `split`.
        auto do_prefill_range = [&](size_t gbegin, size_t gend,
                                    bool compute_final_logits = true) {
            if (gbegin >= gend) return;
            std::vector<uint32_t> seg(prompt_tokens.begin() + static_cast<std::ptrdiff_t>(gbegin),
                                      prompt_tokens.begin() + static_cast<std::ptrdiff_t>(gend));
            step = executor_->forward_n_tokens(
                seg, options.max_tokens > 0 && compute_final_logits &&
                         gend == prompt_tokens.size());
            if (!step.ok) throw std::runtime_error("prefill failed");
            prefill_ops += step.ops_executed;
        };
        auto do_query_prefill_range = [&](size_t gbegin, size_t gend,
                                          bool compute_final_logits) {
            executor_->kvmem_set_prefill_reselect_suppressed(true);
            try {
                do_prefill_range(gbegin, gend, compute_final_logits);
            } catch (...) {
                executor_->kvmem_set_prefill_reselect_suppressed(false);
                throw;
            }
            executor_->kvmem_set_prefill_reselect_suppressed(false);
        };
        bool query_replay_applied = false;
        KvMemSemanticChunkStats semantic_chunk_stats;
        if (dump) {
            for (size_t pi = prefill_begin; pi < prompt_tokens.size(); ++pi) {
                step = executor_->forward_one_token(prompt_tokens[pi]);
                if (!step.ok) throw std::runtime_error("prefill failed");
                prefill_ops += step.ops_executed;
                dump->record(static_cast<int>(pi), "prefill",
                             static_cast<int32_t>(prompt_tokens[pi]),
                             *executor_, *tokenizer_);
            }
        } else if (recompute_query) {
            if (query_replay_begin < prefill_begin) {
                throw std::runtime_error(
                    "KVMem warm checkpoint is after the query replay boundary");
            }
            if (semantic_chunk) {
                if (prefill_begin != 0 || do_boundary_capture) {
                    throw std::runtime_error(
                        "semantic-chunk prefill requires a cold prompt without "
                        "a warm prefix checkpoint");
                }
                semantic_chunk_stats = kvmem_prefill_semantic_chunks(
                    executor_.get(),
                    static_cast<uint32_t>(prompt_tokens.size()),
                    /*begin=*/0, query_replay_begin,
                    options.kvmem_prefill_semantic_start_tokens,
                    options.kvmem_prefill_semantic_query_tokens,
                    [&](uint32_t begin, uint32_t end, bool need_logits) {
                        do_prefill_range(begin, end, need_logits);
                    });
                // History chunks temporarily replace the query span. Restore
                // the real answer-producing user query before its first pass.
                executor_->kvmem_set_query_span(
                    options.kvmem_query_begin, options.kvmem_query_end,
                    static_cast<uint32_t>(prompt_tokens.size()),
                    /*index_tokens=*/query_replay_begin,
                    /*preserve_content_index=*/true);
                executor_->capture_state(query_replay_ckpt);
                executor_->kvmem_start_query_prefetch(
                    static_cast<uint32_t>(
                        prompt_tokens.size() - query_replay_begin));
                do_prefill_range(query_replay_begin, prompt_tokens.size());
            } else {
            // P and B are independent boundaries. If P is no later than B,
            // stage the warm checkpoint on the first pass before capturing the
            // replay snapshot. If P is after B, its first-pass state will be
            // overwritten by query replay, so it is captured during replay below.
            if (do_boundary_capture && ckpt_split <= query_replay_begin) {
                do_prefill_range(prefill_begin, ckpt_split);
                executor_->kvmem_register_append(
                    ckpt_split - static_cast<uint32_t>(prefill_begin));
                executor_->kvmem_reselect_prefill_pressure();
                kvmem_warm_valid_ = false;
                executor_->capture_state(kvmem_warm_ckpt_prompt_);
                warm_prompt_pos = ckpt_split;
                warm_prompt_resumable =
                    kvmem_prompt_checkpoint_resumable(
                        warm_source_index_ready);
                warm_checkpoint_staged = true;
                do_prefill_range(ckpt_split, query_replay_begin);
            } else {
                do_prefill_range(prefill_begin, query_replay_begin);
            }
            executor_->capture_state(query_replay_ckpt);
            executor_->kvmem_start_query_prefetch(
                static_cast<uint32_t>(
                    prompt_tokens.size() - query_replay_begin));
            do_prefill_range(query_replay_begin, prompt_tokens.size());
            }
        } else if (query_replay) {
            const uint32_t qb = options.kvmem_query_begin;
            const uint32_t qe = options.kvmem_query_end;
            const uint32_t bt = executor_->block_store()->config().block_tokens;
            const double checkpoint_start = wall_seconds();

            // Establish the pre-query state and its current (fallback/recency)
            // working set. This is the only state the replay rewinds to.
            do_prefill_range(prefill_begin, qb, /*compute_final_logits=*/false);
            executor_->kvmem_register_append(qb - static_cast<uint32_t>(prefill_begin));
            executor_->kvmem_reselect();
            const std::vector<uint32_t> selected_before =
                kvmem_selected_block_ids(executor_.get());
            QwenExecutor::StateSnapshot query_checkpoint;
            executor_->capture_state(query_checkpoint);
            const double checkpoint_end = wall_seconds();

            // Provisional query: capture Q under the current working set. Its
            // target/DeltaNet state and logits are discarded if capture succeeds.
            const double provisional_start = wall_seconds();
            do_query_prefill_range(qb, prompt_tokens.size(),
                                   /*compute_final_logits=*/true);
            const double provisional_end = wall_seconds();
            if (executor_->kvmem_stash_query()) {
                const double restore_start = wall_seconds();
                executor_->restore_state(query_checkpoint);
                executor_->kvmem_truncate_to(qb);
                // Keep the query in full-prompt coordinates, but publish only the
                // historical [0,qb) content index for the boundary selection.
                executor_->kvmem_set_query_span(
                    qb, qe, static_cast<uint32_t>(prompt_tokens.size()), qb);
                executor_->kvmem_restore_stashed_query();
                executor_->kvmem_set_pin_from_block(qb / std::max<uint32_t>(bt, 1));
                const bool index_ready = executor_->kvmem_query_selection_ready();
                const QwenExecutor::KvMemTimingSnapshot select_t0 =
                    QwenExecutor::kvmem_timing_snapshot();
                const double select_start = wall_seconds();
                executor_->kvmem_reselect();
                const double select_end = wall_seconds();
                const QwenExecutor::KvMemTimingSnapshot select_t1 =
                    QwenExecutor::kvmem_timing_snapshot();
                const std::vector<uint32_t> selected_after =
                    kvmem_selected_block_ids(executor_.get());
                const uint32_t additions =
                    kvmem_selection_additions(selected_before, selected_after);

                // The final replay must capture a fresh Q for later interval
                // reselects; otherwise the provisional Q would remain latched.
                executor_->kvmem_reset_query_capture();
                const double replay_start = wall_seconds();
                do_query_prefill_range(qb, prompt_tokens.size(),
                                       /*compute_final_logits=*/true);
                executor_->kvmem_register_append(
                    static_cast<uint32_t>(prompt_tokens.size()) - qb);
                const double replay_end = wall_seconds();
                query_replay_applied = true;

                std::ostringstream rmsg;
                rmsg << std::fixed << std::setprecision(3)
                     << "native kvmem query-replay (plain): span=[" << qb << ","
                     << qe << ") tail=" << (prompt_tokens.size() - qb)
                     << " checkpoint_ms=" << (checkpoint_end - checkpoint_start) * 1e3
                     << " provisional_ms=" << (provisional_end - provisional_start) * 1e3
                     << " restore_ms=" << (select_start - restore_start) * 1e3
                     << " select_ms=" << (select_end - select_start) * 1e3
                     << " replay_ms=" << (replay_end - replay_start) * 1e3
                     << " selected=" << selected_after.size()
                     << " replaced=" << additions
                     << " replace_rate="
                     << (selected_after.empty()
                             ? 0.0
                             : static_cast<double>(additions) / selected_after.size())
                     << " stage_in="
                     << (select_t1.stage_in_blocks - select_t0.stage_in_blocks)
                     << " stage_out="
                     << (select_t1.stage_out_blocks - select_t0.stage_out_blocks)
                     << " index_ready=" << (index_ready ? 1 : 0)
                     << " final_pos=" << executor_->position();
                log(rmsg.str());
            } else {
                log("native kvmem query-replay (plain): provisional query capture "
                    "was incomplete; keeping the single-pass result");
            }
        } else if (do_boundary_capture) {
            // First segment [prefill_begin, split): advance recurrent state to B.
            do_prefill_range(prefill_begin, ckpt_split,
                             /*compute_final_logits=*/false);
            // Register + reselect so the store/window describe exactly `split`
            // tokens, then stage ckpt_P at the block boundary (recurrent state is
            // captured here, physically at B).
            executor_->kvmem_register_append(
                ckpt_split - static_cast<uint32_t>(prefill_begin));
            executor_->kvmem_reselect_prefill_pressure();
            kvmem_warm_valid_ = false;  // invalid until end-capture re-validates
            executor_->capture_state(kvmem_warm_ckpt_prompt_);
            warm_prompt_pos = ckpt_split;
            warm_checkpoint_staged = true;
            // Below budget: the strict all-GPU-identity precondition (every block
            // resident at its true position). Above budget: identity never holds
            // (blocks spill / re-RoPE to window slots), but the boundary is still
            // resumable when tiers keep [0,B) recoverable and the mean-k index is
            // position-invariant. reselect on resume rebuilds the (stale-snapshot)
            // window from those preserved blocks. per-token is already excluded by
            // qc_pertoken_here above.
            warm_prompt_resumable =
                kvmem_prompt_checkpoint_resumable(
                    warm_source_index_ready);
            // Second segment [split, P): finish the prompt.
            do_prefill_range(ckpt_split, prompt_tokens.size(),
                             /*compute_final_logits=*/true);
        } else {
            do_prefill_range(prefill_begin, prompt_tokens.size(),
                             /*compute_final_logits=*/true);
        }
        const double t_prefill_end = wall_seconds();
        if (semantic_chunk_stats.chunks != 0) {
            std::ostringstream smsg;
            smsg << std::fixed << std::setprecision(3)
                 << "native kvmem semantic-chunk (plain): chunks="
                 << semantic_chunk_stats.chunks
                 << " provisional_tokens="
                 << semantic_chunk_stats.provisional_tokens
                 << " replay_tokens=" << semantic_chunk_stats.replay_tokens
                 << " provisional_ms="
                 << semantic_chunk_stats.provisional_s * 1e3
                 << " rollback_ms=" << semantic_chunk_stats.rollback_s * 1e3
                 << " reselect_ms=" << semantic_chunk_stats.reselect_s * 1e3
                 << " replay_ms=" << semantic_chunk_stats.replay_s * 1e3;
            log(smsg.str());
        }
#ifdef QW3_ENABLE_CUDA
        if (profile_cuda_prefill) {
            st = device_->synchronize();
            if (!st.ok) throw std::runtime_error(st.message);
            const cudaError_t profile_st = cudaProfilerStop();
            if (profile_st != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaProfilerStop failed: ") +
                    cudaGetErrorString(profile_st));
            }
        }
#endif

        // Block-sparse: register the prefilled prompt as context blocks and
        // assemble the first working set. Under the default all-fit budget this
        // selects every block (identity), so the decode path stays byte-
        // identical to plain; once the context exceeds the budget the built-in
        // top-k starts dropping cold blocks.
        const bool bs_on = executor_->kvmem_enabled();
        int bs_steps_since_reselect = 0;
        const int bs_interval =
            std::max(1, options_.kvmem_interval);
        // Diagnostics-only control: the oracle must alter exactly one thing —
        // the answer-producing semantic selection.  Installing it before
        // prefill would also force those blocks into pressure windows and change
        // how later historical K/V is constructed.
        auto kvmem_final_query_reselect = [&]() {
            if (options.kvmem_oracle_token_spans.empty()) {
                executor_->kvmem_reselect();
                return;
            }
            std::vector<std::pair<uint32_t, uint32_t>> oracle_spans;
            oracle_spans.reserve(options.kvmem_oracle_token_spans.size());
            for (const auto &span : options.kvmem_oracle_token_spans) {
                oracle_spans.emplace_back(span.begin, span.end);
            }
            executor_->kvmem_set_oracle_token_spans(
                oracle_spans, options.kvmem_oracle_only);
            executor_->kvmem_reselect();
            executor_->kvmem_set_oracle_token_spans({});
        };
        if (bs_on) {
            if (!query_replay_applied) {
            // Register the tokens not yet in the store so it lands at prompt.size().
            // Already registered before this point: reuse_m (warm restore) plus the
            // first segment when a block-boundary ckpt_P split ran.
            const uint32_t already =
                warm_checkpoint_staged
                    ? ckpt_split
                    : static_cast<uint32_t>(prefill_begin);
            const uint32_t reg_n =
                static_cast<uint32_t>(prompt_tokens.size()) - already;
            executor_->kvmem_register_append(reg_n);
            if (options.kvmem_reselect_mode == KvMemReselectMode::Off) {
                if (options.kvmem_prefill_window_mode !=
                    KvMemPrefillWindowMode::KeepSelected) {
                    executor_->kvmem_reselect_prefill_pressure();
                }
            } else {
                if (recompute_query) {
                    if (!executor_->kvmem_query_capture_complete()) {
                        throw std::runtime_error(
                            "KVMem query replay first pass captured " +
                            std::to_string(
                                executor_->kvmem_query_captured_tokens()) +
                            "/" +
                            std::to_string(
                                executor_->kvmem_query_expected_tokens()) +
                            " query tokens");
                    }
                    uint32_t query_attention_probe_decoded = 0;
                    bool query_attention_probe_used = false;
                    double post_query_attention_probe_s = 0.0;
                    std::vector<uint32_t> query_score_token_indices;
                    const uint32_t query_tokens =
                        options.kvmem_query_end - options.kvmem_query_begin;
                    const uint32_t keep_query_tokens = std::min<uint32_t>(
                        query_tokens,
                        options.kvmem_query_attention_score_tokens);
                    if (query_attention_probe &&
                        keep_query_tokens < query_tokens) {
                        const double t_probe_start = wall_seconds();
                        std::vector<float> attention_mass;
                        QwenExecutor::StateSnapshot probe_boundary;
                        try {
                            if (executor_->kvmem_begin_query_attention_probe(
                                    options.kvmem_query_begin,
                                    options.kvmem_query_end)) {
                                executor_->capture_transient_state(
                                    probe_boundary);
                                int32_t probe_token = step.argmax_token;
                                for (uint32_t i = 0;
                                     i < options
                                             .kvmem_query_attention_probe_tokens &&
                                     probe_token >= 0;
                                     ++i) {
                                    NativeExecutorReport probe_step =
                                        executor_->forward_one_token(
                                            static_cast<uint32_t>(probe_token));
                                    if (!probe_step.ok) break;
                                    ++query_attention_probe_decoded;
                                    probe_token = probe_step.argmax_token;
                                }
                                attention_mass = executor_
                                    ->kvmem_end_query_attention_probe();
                                executor_->restore_state(probe_boundary);
                            }
                        } catch (...) {
                            executor_->kvmem_cancel_query_attention_probe();
                            if (probe_boundary.ready) {
                                executor_->restore_state(probe_boundary);
                            }
                            throw;
                        }

                        if (attention_mass.size() == query_tokens) {
                            double positive_mass = 0.0;
                            std::vector<uint32_t> ranked(query_tokens);
                            for (uint32_t i = 0; i < query_tokens; ++i) {
                                ranked[i] = i;
                                if (std::isfinite(attention_mass[i]) &&
                                    attention_mass[i] > 0.0f) {
                                    positive_mass += attention_mass[i];
                                } else {
                                    attention_mass[i] =
                                        -std::numeric_limits<float>::infinity();
                                }
                            }
                            if (positive_mass > 0.0) {
                                std::partial_sort(
                                    ranked.begin(),
                                    ranked.begin() + keep_query_tokens,
                                    ranked.end(),
                                    [&](uint32_t a, uint32_t b) {
                                        if (attention_mass[a] !=
                                            attention_mass[b]) {
                                            return attention_mass[a] >
                                                   attention_mass[b];
                                        }
                                        return a < b;
                                    });
                                ranked.resize(keep_query_tokens);
                                std::sort(ranked.begin(), ranked.end());
                                query_score_token_indices = std::move(ranked);
                                query_attention_probe_used = true;
                            }
                        }

                        // Unsupported attention dtype/kernel or an incomplete
                        // probe falls back to deterministic full-span coverage.
                        // The sparse score budget remains enforced, avoiding the
                        // long-query scorer allocation that motivated the probe.
                        if (query_score_token_indices.empty()) {
                            query_score_token_indices.reserve(
                                keep_query_tokens);
                            for (uint32_t i = 0; i < keep_query_tokens; ++i) {
                                query_score_token_indices.push_back(
                                    static_cast<uint32_t>(
                                        ((2ull * i + 1ull) * query_tokens) /
                                        (2ull * keep_query_tokens)));
                            }
                        }
                        executor_->kvmem_set_query_score_token_indices(
                            query_score_token_indices);
                        post_query_attention_probe_s +=
                            wall_seconds() - t_probe_start;

                        std::ostringstream pmsg;
                        pmsg << "native kvmem query attention probe: requested="
                             << options.kvmem_query_attention_probe_tokens
                             << " decoded=" << query_attention_probe_decoded
                             << " query_tokens=" << query_tokens
                             << " score_tokens="
                             << query_score_token_indices.size()
                             << " selector="
                             << (query_attention_probe_used
                                     ? "attention-topk"
                                     : "uniform-fallback")
                             << " indices=[";
                        for (size_t i = 0;
                             i < query_score_token_indices.size(); ++i) {
                            if (i) pmsg << ',';
                            pmsg << query_score_token_indices[i];
                        }
                        pmsg << ']';
                        log(pmsg.str());
                    }
                    const uint32_t bt = std::max<uint32_t>(
                        1, executor_->block_store()->config().block_tokens);
                    executor_->kvmem_set_pin_from_block(
                        query_replay_begin / bt);
                }
                // `qc_prepare` below budget exists only to make the next warm
                // threshold-crossing request resumable. Do not turn that
                // metadata capture into an eager semantic selection.
                if (!qc_prepare || qc_select_active) {
                    kvmem_final_query_reselect();
                }
            }
            if (recompute_query &&
                options.kvmem_reselect_mode != KvMemReselectMode::Off) {
                std::vector<uint32_t> selected_context;
                bool suffix_fully_selected = true;
                const auto &blocks = executor_->block_store()->blocks();
                for (const KvMemBlock &b : blocks) {
                    if (b.orig_pos_start < query_replay_begin) {
                        if (b.in_working_set) {
                            selected_context.push_back(b.block_id);
                        }
                    } else if (!b.in_working_set) {
                        suffix_fully_selected = false;
                    }
                }
                if (!suffix_fully_selected) {
                    throw std::runtime_error(
                        "KVMem query replay requires every replay-suffix block "
                        "to be present in the final selection");
                }
                if (inline_refresh) {
                    std::vector<uint32_t> refreshed_tokens =
                        kvmem_gather_selected_source_tokens(
                            prompt_tokens, blocks, selected_context);
                    const size_t selected_prefix_tokens =
                        refreshed_tokens.size();
                    refreshed_tokens.insert(
                        refreshed_tokens.end(),
                        prompt_tokens.begin() +
                            static_cast<std::ptrdiff_t>(query_replay_begin),
                        prompt_tokens.end());
                    if (refreshed_tokens.empty() ||
                        refreshed_tokens.size() >
                            executor_->kv_ctx_size()) {
                        throw std::runtime_error(
                            "KVMem inline refresh compact prompt is empty or "
                            "exceeds the executor context");
                    }

                    // From here through decode, run the compact cache as a
                    // normal dense cache. The guard restores the configured
                    // KVMem switch on every return/exception, while reset_state
                    // discards only this request's now-unneeded million-token
                    // store.
                    inline_refresh_guard =
                        std::make_unique<ScopedKvmemDisable>(executor_.get());
                    executor_->reset_state();
                    if (options.kvmem_inline_refresh ==
                        KvMemInlineRefreshMode::KvOnly) {
                        if (selected_prefix_tokens == 0) {
                            throw std::runtime_error(
                                "KVMem inline KV-only refresh selected no "
                                "historical context tokens");
                        }
                        std::vector<uint32_t> prefix(
                            refreshed_tokens.begin(),
                            refreshed_tokens.begin() +
                                static_cast<std::ptrdiff_t>(
                                    selected_prefix_tokens));
                        NativeExecutorReport prefix_step =
                            executor_->forward_n_tokens(prefix, false);
                        if (!prefix_step.ok) {
                            throw std::runtime_error(
                                "KVMem inline refresh prefix prefill failed");
                        }
                        prefill_ops += prefix_step.ops_executed;
                        executor_->restore_recurrent_state(
                            query_replay_ckpt);
                        std::vector<uint32_t> suffix(
                            refreshed_tokens.begin() +
                                static_cast<std::ptrdiff_t>(
                                    selected_prefix_tokens),
                            refreshed_tokens.end());
                        step = executor_->forward_n_tokens(
                            suffix, options.max_tokens > 0);
                        if (!step.ok) {
                            throw std::runtime_error(
                                "KVMem inline refresh query prefill failed");
                        }
                        prefill_ops += step.ops_executed;
                    } else {
                        step = executor_->forward_n_tokens(
                            refreshed_tokens, options.max_tokens > 0);
                        if (!step.ok) {
                            throw std::runtime_error(
                                "KVMem inline refresh compact prefill failed");
                        }
                        prefill_ops += step.ops_executed;
                    }
                    log("native kvmem inline refresh (plain): mode=" +
                        std::string(kvmem_inline_refresh_name(
                            options.kvmem_inline_refresh)) +
                        " source_prompt_tokens=" +
                        std::to_string(prompt_tokens.size()) +
                        " selected_prefix_tokens=" +
                        std::to_string(selected_prefix_tokens) +
                        " query_suffix_tokens=" +
                        std::to_string(prompt_tokens.size() -
                                       query_replay_begin) +
                        " compact_prompt_tokens=" +
                        std::to_string(refreshed_tokens.size()) +
                        " fixed_context_blocks=" +
                        std::to_string(selected_context.size()));
                } else {
                    executor_->kvmem_begin_query_replay(
                        query_replay_ckpt, selected_context,
                        /*reset_recurrent_state=*/false,
                        /*preserve_selected_context=*/true,
                        /*preserve_query_capture=*/
                            ru.query_snapshot_reuse);
#if 0  // Archived DeltaNet recurrent-state capture/import.
                if (rebuilt_state_import || rebuilt_state_capture) {
                    const std::vector<uint32_t> selected_source_tokens =
                        kvmem_selected_source_tokens(
                            prompt_tokens,
                            executor_->block_store()->blocks(),
                            selected_context);
                    if (rebuilt_state_capture) {
                        executor_->kvmem_export_recurrent_state(
                            kvmem_rebuilt_state_path(
                                options.kvmem_rebuilt_state_capture_key),
                            selected_source_tokens);
                        log("native kvmem rebuilt-state capture (plain): key=" +
                            options.kvmem_rebuilt_state_capture_key +
                            " identity_tokens=" +
                            std::to_string(selected_source_tokens.size()) +
                            " fixed_context_blocks=" +
                            std::to_string(selected_context.size()));
                    } else {
                        executor_->kvmem_import_recurrent_state(
                            kvmem_rebuilt_state_path(
                                options.kvmem_rebuilt_state_import_key),
                            selected_source_tokens);
                        log("native kvmem rebuilt-state import (plain): key=" +
                            options.kvmem_rebuilt_state_import_key +
                            " source_tokens=" +
                            std::to_string(selected_source_tokens.size()) +
                            " fixed_context_blocks=" +
                            std::to_string(selected_context.size()));
                    }
                }
#endif
                if (do_boundary_capture && !warm_checkpoint_staged &&
                    ckpt_split > query_replay_begin) {
                    do_prefill_range(query_replay_begin, ckpt_split);
                    // Query replay registers its replacement suffix incrementally.
                    // Capture P from the final selected-context state, not from
                    // the first pressure-window pass that has just been discarded.
                    executor_->kvmem_register_append(
                        ckpt_split - query_replay_begin);
                    kvmem_warm_valid_ = false;
                    executor_->capture_state(kvmem_warm_ckpt_prompt_);
                    warm_prompt_pos = ckpt_split;
                    warm_prompt_resumable =
                        kvmem_prompt_checkpoint_resumable(
                            warm_source_index_ready);
                    warm_checkpoint_staged = true;
                    do_prefill_range(ckpt_split, prompt_tokens.size());
                } else {
                    do_prefill_range(query_replay_begin,
                                     prompt_tokens.size());
                }
                executor_->kvmem_end_query_replay();
                if (!executor_->kvmem_query_capture_complete()) {
                    throw std::runtime_error(
                        "KVMem query replay final pass captured " +
                        std::to_string(
                            executor_->kvmem_query_captured_tokens()) +
                        "/" +
                        std::to_string(
                            executor_->kvmem_query_expected_tokens()) +
                        " query tokens");
                }
                log("native kvmem query replay (plain): boundary=" +
                    std::to_string(query_replay_begin) +
                    " suffix_tokens=" +
                    std::to_string(prompt_tokens.size() -
                                   query_replay_begin) +
                    " warm_reuse=" +
                    std::to_string(warm_reuse ? 1 : 0) +
                    " warm_checkpoint=" +
                    std::string(warm_reuse
                                    ? (ru.prompt_ckpt ? "P" : "M")
                                    : "none") +
                    " fixed_context_blocks=" +
                    std::to_string(selected_context.size()));
                }
            }
        }
        }

        // kvmem prefix cache: fallback prompt-end (P) checkpoint. When the
        // block-boundary split above already staged ckpt_P at B, skip this;
        // otherwise (tiny prompt, sparse/tiered, or ckpt_M reuse past the last
        // boundary) capture at P here, AFTER register+reselect so the snapshot's
        // block store / window / registered_pos all describe exactly P tokens.
        if (warm_capture && !warm_checkpoint_staged) {
            kvmem_warm_valid_ = false;  // invalid until end-capture re-validates
            executor_->capture_state(kvmem_warm_ckpt_prompt_);
            warm_prompt_pos = static_cast<uint32_t>(executor_->position());
            warm_prompt_resumable =
                kvmem_prompt_checkpoint_resumable(
                    warm_source_index_ready);
        }

#if 0  // Archived DeltaNet recurrent-state export.
        if (rebuilt_state_export) {
            if (warm_reuse) {
                throw std::runtime_error(
                    "KVMem rebuilt-state export requires a cold full prefill");
            }
            if (bs_on && kvmem_sel_budget > 0 &&
                prompt_tokens.size() > kvmem_sel_budget) {
                throw std::runtime_error(
                    "KVMem rebuilt-state export source exceeds the dense "
                    "KVMem selection budget");
            }
            executor_->kvmem_export_recurrent_state(
                kvmem_rebuilt_state_path(
                    options.kvmem_rebuilt_state_export_key),
                prompt_tokens);
            log("native kvmem rebuilt-state export (plain): key=" +
                options.kvmem_rebuilt_state_export_key +
                " source_tokens=" + std::to_string(prompt_tokens.size()));
        }
#endif

        // A zero completion budget is a real prefill-only transaction. Persist
        // the exact prompt-end state, but never initialize sampling, choose a
        // next token, enter decode capture, or append generated KV.
        if (options.max_tokens == 0) {
            if (warm_capture) {
                kvmem_warm_log_ = prompt_tokens;
                const size_t pos = executor_->position();
                if (pos <= kvmem_warm_log_.size()) {
                    kvmem_warm_log_.resize(pos);
                    executor_->capture_state(kvmem_warm_ckpt_end_);
                    const bool prompt_source_index_ready =
                        warm_source_index_ready &&
                        executor_->kvmem_content_index_resume_compatible(
                            warm_prompt_pos);
                    const bool end_source_index_ready =
                        warm_source_index_ready &&
                        executor_->kvmem_content_index_resume_compatible(
                            static_cast<uint32_t>(pos));
                    const bool query_snapshot_ready =
                        kvmem_capture_warm_query_snapshot(
                            *executor_, options,
                            qc_prepare && prompt_source_index_ready);
                    kvmem_warm_end_pos_ = static_cast<uint32_t>(pos);
                    kvmem_warm_prompt_pos_ = warm_prompt_pos;
                    kvmem_warm_prompt_resumable_ = warm_prompt_resumable;
                    kvmem_warm_source_index_ready_ =
                        prompt_source_index_ready;
                    kvmem_warm_end_resumable_ =
                        kvmem_prompt_checkpoint_resumable(
                            end_source_index_ready);
                    kvmem_warm_end_source_index_ready_ =
                        end_source_index_ready;
                    kvmem_warm_valid_ = true;
                    if (kvmem_prefix_cache_trace_enabled()) {
                        std::ostringstream cmsg;
                        cmsg << "kvmem prefix-cache CAPTURE "
                             << "(plain prefill-only): warm_log="
                             << kvmem_warm_log_.size() << " pos=" << pos
                             << " P=" << warm_prompt_pos
                             << " P_resumable="
                             << (warm_prompt_resumable ? 1 : 0)
                             << " P_source_index="
                             << (prompt_source_index_ready ? 1 : 0)
                             << " M_resumable="
                             << (kvmem_warm_end_resumable_ ? 1 : 0)
                             << " M_source_index="
                             << (end_source_index_ready ? 1 : 0)
                             << " query_snapshot="
                             << (query_snapshot_ready ? 1 : 0);
                        log(cmsg.str());
                    }
                } else {
                    kvmem_warm_valid_ = false;
                }
            }
            st = device_->end();
            if (!st.ok) throw std::runtime_error(st.message);
            const double prefill_s =
                std::max(t_prefill_end - t_prefill_start, 1e-9);
            std::ostringstream msg;
            msg << "native prefill-only: prompt_tokens="
                << prompt_tokens.size() << " prefill="
                << fmt_seconds(prefill_s) << " prefill_ops=" << prefill_ops;
            if (warm_reuse) {
                msg << " kvmem_reuse=" << reuse_m
                    << " prefilled=" << (prompt_tokens.size() - reuse_m);
            }
            log(msg.str());
            return {};
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
        // Committed decode token ids, accumulated only when we intend to keep a
        // warm checkpoint. Appended in commit order so kvmem_warm_log_ stays
        // consistent with the executor KV / kvmem register count.
        std::vector<uint32_t> gen_tokens;
        const int32_t eos = tokenizer_->eos_id();
        ThinkingBudgetState budget;
        budget_init(budget, options);
        const int32_t seed_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
        uint32_t next_token = static_cast<uint32_t>(pick_next(seed_argmax));
        next_token = budget_apply(budget, next_token);
        uint64_t decode_ops = 0;
        std::unordered_map<std::string, TraceStats> decode_trace;
        uint64_t decode_trace_steps = 0;
        int decoded = 0;
        bool stream_cancelled = false;
        const auto should_stop_eos = [&]() {
            return !options.ignore_eos &&
                   next_token == static_cast<uint32_t>(eos) &&
                   !budget.can_recover_eos(next_token);
        };
        if (options.max_tokens > 0 && !should_stop_eos()) {
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(next_token));
            generated += piece;
            ++seen_tokens[next_token];
            if (warm_capture) gen_tokens.push_back(next_token);
            ++decoded;
            if (on_text && !on_text(piece)) stream_cancelled = true;
        }
        // Above-budget QC: index generated tokens as decode produces them so the
        // next turn's preserved [0,D) slices cover the response too (Gap A). No-op
        // unless kvmem + above budget + mean-k (guards inside begin()).
        executor_->kvmem_decode_capture_begin();
#ifdef QW3_ENABLE_CUDA
        const bool profile_cuda_decode =
            env_flag_enabled("QW3_CUDA_PROFILE_DECODE");
        if (profile_cuda_decode) {
            st = device_->synchronize();
            if (!st.ok) throw std::runtime_error(st.message);
            const cudaError_t profile_st = cudaProfilerStart();
            if (profile_st != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaProfilerStart failed: ") +
                    cudaGetErrorString(profile_st));
            }
        }
#endif
        for (int i = 0;
             i + 1 < options.max_tokens && !stream_cancelled;
             ++i) {
            if (should_stop_eos()) break;
            const uint32_t feed = next_token;
            step = executor_->forward_one_token(feed);
            if (!step.ok) throw std::runtime_error("decode failed");
            decode_ops += step.ops_executed;
            if (decode_trace_enabled() && !step.elapsed_us.empty()) {
                accumulate_trace(decode_trace, step);
                ++decode_trace_steps;
            }
            // Block-sparse: the token just decoded grew the context by one.
            // Register it, and reselect the working set on the interval
            // boundary (the agent-step cadence is approximated here by a fixed
            // step interval; reselection re-bakes any moved block in place).
            if (bs_on) {
                executor_->kvmem_register_append(1);
                // Only reselect once the store has actually spilled past budget;
                // below budget selection is identity (a no-op) so skip it and
                // "use all the context". No-op in step mode regardless.
                if (options_.kvmem_update_mode != "step" &&
                    executor_->block_store()->block_count() >
                        executor_->block_store()->budget_blocks() &&
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
            ++seen_tokens[next_token];
            if (warm_capture) gen_tokens.push_back(next_token);
            ++decoded;
            if (on_text && !on_text(piece)) stream_cancelled = true;
        }
#ifdef QW3_ENABLE_CUDA
        if (profile_cuda_decode) {
            st = device_->synchronize();
            if (!st.ok) throw std::runtime_error(st.message);
            const cudaError_t profile_st = cudaProfilerStop();
            if (profile_st != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaProfilerStop failed: ") +
                    cudaGetErrorString(profile_st));
            }
        }
#endif
        // Flush any trailing partial block's mean into the content index.
        executor_->kvmem_decode_capture_finalize();
        const double t_decode_end = wall_seconds();

        if (decoded == 0 && options.max_tokens > 0) {
            log_zero_decode_diagnostic("plain", prompt_tokens, step);
        }

        // kvmem prefix cache: record this request's end as the warm resume
        // point for the next request. The last emitted token is never forwarded
        // (its KV is not in the store), so resize the log down to the executor
        // position -> it matches KV pages + kvmem register count exactly. A
        // position() beyond the log can't happen on the plain path, but guard it
        // (invalidate) rather than capture an inconsistent checkpoint. Captured
        // inside the device scope since capture_state issues device copies.
        if (warm_capture && !stream_cancelled) {
            kvmem_warm_log_ = prompt_tokens;
            kvmem_warm_log_.insert(kvmem_warm_log_.end(),
                                   gen_tokens.begin(), gen_tokens.end());
            const size_t pos = executor_->position();
            if (pos <= kvmem_warm_log_.size()) {
                kvmem_warm_log_.resize(pos);
                executor_->capture_state(kvmem_warm_ckpt_end_);
                const bool prompt_source_index_ready =
                    warm_source_index_ready &&
                    executor_->kvmem_content_index_resume_compatible(
                        warm_prompt_pos);
                const bool end_source_index_ready =
                    warm_source_index_ready &&
                    executor_->kvmem_content_index_resume_compatible(
                        static_cast<uint32_t>(pos));
                const bool query_snapshot_ready =
                    kvmem_capture_warm_query_snapshot(
                        *executor_, options,
                        qc_prepare && prompt_source_index_ready);
                // Commit the staged prompt-end (P) checkpoint alongside the
                // turn-end (M) one; flip all warm fields together so a mid-decode
                // throw leaves the prior turn's warm state intact.
                kvmem_warm_end_pos_ = static_cast<uint32_t>(pos);
                kvmem_warm_prompt_pos_ = warm_prompt_pos;
                kvmem_warm_prompt_resumable_ = warm_prompt_resumable;
                kvmem_warm_source_index_ready_ =
                    prompt_source_index_ready;
                kvmem_warm_end_resumable_ =
                    kvmem_prompt_checkpoint_resumable(
                        end_source_index_ready);
                kvmem_warm_end_source_index_ready_ =
                    end_source_index_ready;
                kvmem_warm_valid_ = true;
                if (kvmem_prefix_cache_trace_enabled()) {
                    std::ostringstream cmsg;
                    cmsg << "kvmem prefix-cache CAPTURE (plain): warm_log="
                         << kvmem_warm_log_.size() << " pos=" << pos
                         << " P=" << warm_prompt_pos
                         << " P_resumable=" << (warm_prompt_resumable ? 1 : 0)
                         << " P_source_index="
                         << (prompt_source_index_ready ? 1 : 0)
                         << " M_resumable="
                         << (kvmem_warm_end_resumable_ ? 1 : 0)
                         << " M_source_index="
                         << (end_source_index_ready ? 1 : 0)
                         << " query_snapshot="
                         << (query_snapshot_ready ? 1 : 0)
                         << " decoded=" << decoded;
                    log(cmsg.str());
                }
            } else {
                kvmem_warm_valid_ = false;
            }
        } else if (stream_cancelled) {
            kvmem_warm_valid_ = false;
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
        if (warm_reuse) {
            msg << " kvmem_reuse=" << reuse_m
                << " prefilled=" << (prompt_tokens.size() - reuse_m);
        }
        if (stream_cancelled) msg << " cancelled=true";
        log(msg.str());
        if (decode_trace_enabled() && !decode_trace.empty()) {
            log_decode_trace(decode_trace, decode_trace_steps);
        }

        return generated;
    }

    // Per-call wall-clock + acceptance, threaded out for the kvmem-session
    // growth harness (run_kvmem_session). Filled just before return when a
    // non-null stats_out is passed; default callers ignore it.
    struct MtpGenStats {
        double total_s = 0.0;
        double setup_s = 0.0;
        double prefill_s = 0.0;
        double postprefill_s = 0.0;
        double decode_s = 0.0;   // PURE decode loop (post-prefill reselect excluded)
        double finalize_s = 0.0;
        double reselect_s = 0.0; // post-prefill (decode-window) reselect wall clock
        double semantic_reselect_s = 0.0;
        double query_replay_s = 0.0;
        double query_attention_probe_s = 0.0;
        double query_guided_query_s = 0.0;
        double post_other_s = 0.0;
        int decoded = 0;
        uint64_t prompt_tokens = 0;
        double acceptance = 0.0;
        uint32_t query_attention_probe_decoded = 0;
        bool query_attention_probe_used = false;
        std::vector<uint32_t> query_score_token_indices;
        uint32_t query_guided_thinking_tokens = 0;
        uint32_t query_guided_query_tokens = 0;
        bool query_guided_thinking_closed = false;
        std::string query_guided_query_text;
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
                             const CancellableTokenCallback &on_text,
                             DumpStream *dump,
                             bool spec_mtp,
                             bool trace_mtp,
                             QwenExecutor *override_executor = nullptr,
                             bool manage_device_scope = true,
                             bool reset_session = true,
                             MtpGenStats *stats_out = nullptr) {
        const double t_native_start = wall_seconds();
        QwenExecutor *executor_ =
            override_executor != nullptr ? override_executor : this->executor_.get();
        if (!executor_) throw std::runtime_error("MTP executor unavailable");
        // kvmem prefix cache (QW3_KVMEM_PREFIX_CACHE): only the plain-route shared
        // executor (override_executor==nullptr) on a fresh-session request
        // (reset_session) is eligible. run_kvmem_session (reset_session=false) and
        // the CB per-request executors are excluded, so those paths are untouched.
        const bool transcript_replay_requested =
            !options.kvmem_replay_query_spans.empty();
        const bool semantic_chunk =
            options.kvmem_prefill_window_mode ==
            KvMemPrefillWindowMode::SemanticChunk;
        if (semantic_chunk && transcript_replay_requested) {
            throw std::runtime_error(
                "semantic-chunk prefill and transcript replay are mutually "
                "exclusive");
        }
        const bool inline_refresh =
            options.kvmem_inline_refresh !=
            KvMemInlineRefreshMode::Off;
        if (inline_refresh &&
            (override_executor != nullptr || !reset_session ||
             transcript_replay_requested ||
             !options.kvmem_session_id.empty() || dump != nullptr)) {
            throw std::runtime_error(
                "KVMem inline refresh requires a standalone one-shot MTP "
                "request without logit dumping");
        }
        std::unique_ptr<ScopedKvmemDisable> inline_refresh_guard;
#if 0  // Archived DeltaNet recurrent-state artifact experiment.
        const bool rebuilt_state_export =
            !options.kvmem_rebuilt_state_export_key.empty();
        const bool rebuilt_state_import =
            !options.kvmem_rebuilt_state_import_key.empty();
        const bool rebuilt_state_capture =
            !options.kvmem_rebuilt_state_capture_key.empty();
        const bool rebuilt_state_seed =
            !options.kvmem_rebuilt_state_seed_key.empty();
        if (rebuilt_state_export && options.max_tokens != 0) {
            throw std::runtime_error(
                "KVMem rebuilt-state export requires max_tokens=0");
        }
        if ((rebuilt_state_export || rebuilt_state_import ||
             rebuilt_state_capture || rebuilt_state_seed) &&
            (override_executor != nullptr || !reset_session ||
             transcript_replay_requested ||
             !options.kvmem_session_id.empty())) {
            throw std::runtime_error(
                "KVMem rebuilt-state diagnostics require a standalone "
                "one-shot request");
        }
#endif
        const bool api_session = !options.kvmem_session_id.empty() &&
            override_executor == nullptr;
        const KvmemReuse kvmem_ru =
            (reset_session && override_executor == nullptr &&
             !transcript_replay_requested && !semantic_chunk && !api_session)
                ? kvmem_prefix_reuse(prompt_tokens, options)
                : KvmemReuse{};
        const uint32_t kvmem_reuse_m = kvmem_ru.c;
        const bool kvmem_warm_reuse = kvmem_reuse_m > 0;
        const bool kvmem_warm_capture =
            reset_session && override_executor == nullptr &&
            !transcript_replay_requested && !semantic_chunk && !api_session &&
            !inline_refresh &&
            kvmem_prefix_cache_enabled() &&
            executor_->kvmem_enabled();
        DeviceStatus st;
        if (manage_device_scope) {
            st = device_->begin();
            if (!st.ok) throw std::runtime_error(st.message);
        }
        if (kvmem_warm_reuse) {
            // Resume at the chosen checkpoint (M=turn-end or P=prompt-end), then
            // rewind the block store / tier slots / stale selection index to
            // exactly reuse_m tokens (restore_state rewinds position + KV pages +
            // recurrent state + window, but NOT the block store). Suffix-only
            // prefill below then re-prefills [reuse_m, prompt.end()).
            executor_->restore_state(kvmem_ru.prompt_ckpt ? kvmem_warm_ckpt_prompt_
                                                          : kvmem_warm_ckpt_end_);
            executor_->kvmem_truncate_to(kvmem_reuse_m);
            if (kvmem_prefix_cache_trace_enabled()) {
                std::ostringstream tmsg;
                tmsg << "kvmem prefix-cache HIT (mtp): reuse=" << kvmem_reuse_m
                     << " ckpt=" << (kvmem_ru.prompt_ckpt ? "P" : "M")
                     << " D=" << kvmem_ru.common_prefix
                     << " ceiling=" << kvmem_ru.resume_ceiling
                     << " replay_boundary=" << kvmem_ru.replay_boundary
                     << " query_limited="
                     << (kvmem_ru.query_limited ? 1 : 0)
                     << " query_snapshot="
                     << (kvmem_ru.query_snapshot_reuse ? 1 : 0)
                     << " prompt=" << prompt_tokens.size()
                     << " suffix=" << (prompt_tokens.size() - kvmem_reuse_m);
                log(tmsg.str());
            }
        } else if (reset_session) {
            executor_->reset_state();
            if (kvmem_warm_capture) kvmem_warm_valid_ = false;
        }
#if 0  // Archived DeltaNet recurrent-state seed.
        if (rebuilt_state_seed) {
            if (kvmem_warm_reuse) {
                throw std::runtime_error(
                    "KVMem rebuilt-state seed requires a cold full prefill");
            }
            // See generate_plain: seed import can be the first operation on a
            // freshly loaded executor, before forward() has allocated states.
            executor_->prepare_runtime_state();
            executor_->kvmem_import_recurrent_state(
                kvmem_rebuilt_state_path(
                    options.kvmem_rebuilt_state_seed_key),
                prompt_tokens);
            log("native kvmem rebuilt-state seed (mtp): key=" +
                options.kvmem_rebuilt_state_seed_key +
                " identity_tokens=" + std::to_string(prompt_tokens.size()));
        }
#endif

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
        if (kvmem_on) {
            executor_->kvmem_set_keep_selected_prefill(
                options.kvmem_prefill_window_mode ==
                KvMemPrefillWindowMode::KeepSelected);
        }
        // "Operating dense" predicate (see generate_plain): below budget the
        // store stays GPU-resident in identity order, so selection/QC are no-ops
        // and this turn behaves like the dense config.
        const uint32_t kvmem_sel_budget =
            (kvmem_on && executor_->block_store())
                ? executor_->block_store()->select_budget_tokens() : 0;
        const uint32_t logical_prompt_tokens = reset_session
            ? static_cast<uint32_t>(prompt_tokens.size())
            : static_cast<uint32_t>(executor_->position() +
                                    prompt_tokens.size());
        // Logical token positions identify durable pages and may be much larger
        // than the trained context. MTP local-position mode keeps that identity
        // separate from RoPE: prefix rows use the exact compact frame produced
        // by the target-model chunk, and selected MTP K is rebuilt from raw K at
        // the current window slots. The legacy path still guards the logical
        // end because it bakes MTP K at absolute positions.
        const uint32_t mtp_rope_limit = model_->config().n_ctx_train;
        const uint64_t mtp_logical_end =
            static_cast<uint64_t>(logical_prompt_tokens) +
            static_cast<uint64_t>(std::max(0, options.max_tokens));
        const bool mtp_local_positions =
            kvmem_on && executor_->kvmem_mtp_local_positions();
        // FULLCTX_QUERY deliberately postpones pressure selection and can keep
        // the target in the logical frame past n_ctx_train. In that diagnostic
        // mode compact MTP positions are unavailable until selection, so retain
        // the legacy guard for an over-limit prompt.
        const bool mtp_compact_prefill_safe =
            mtp_local_positions &&
            (!env_flag_enabled("QW3_KVMEM_FULLCTX_QUERY") ||
             mtp_rope_limit == 0 ||
             logical_prompt_tokens <= mtp_rope_limit);
        const uint64_t mtp_compact_end =
            static_cast<uint64_t>(kvmem_sel_budget) +
            static_cast<uint64_t>(std::max(0, options.max_tokens));
        const bool mtp_prefix_positions_safe =
            !kvmem_on || mtp_rope_limit == 0 ||
            (mtp_compact_prefill_safe
                 ? mtp_compact_end <= mtp_rope_limit
                 : mtp_logical_end <= mtp_rope_limit);
        const uint32_t api_append_base = api_session
            ? static_cast<uint32_t>(executor_->position()) : 0;
        const bool kvmem_below_budget =
            kvmem_sel_budget > 0 &&
            logical_prompt_tokens <= kvmem_sel_budget;
        (void)kvmem_below_budget;
        const bool qc_source_index_supported =
            executor_->block_store() &&
            kvmem_query_replay_retrieval_supported(
                executor_->block_store()->config().retrieval_method);
        const KvmemQueryLifecycle qc = kvmem_query_lifecycle(
            options.kvmem_query_end > options.kvmem_query_begin,
            logical_prompt_tokens, kvmem_sel_budget,
            options.kvmem_reselect_mode != KvMemReselectMode::Off,
            kvmem_warm_capture, qc_source_index_supported);
        const bool qc_select_active = qc.select_active;
        const bool qc_prepare = qc.prepare;
        const bool qc_capture_active = qc.capture_active();
        const bool kvmem_warm_history_index_capture =
            kvmem_warm_capture && !qc_capture_active &&
            kvmem_sel_budget > 0 &&
            logical_prompt_tokens > kvmem_sel_budget &&
            qc_source_index_supported;
        const bool kvmem_warm_source_index_ready =
            (qc_capture_active && qc_source_index_supported) ||
            kvmem_warm_history_index_capture;
        if (semantic_chunk) {
            if (!kvmem_on || !qc_select_active || !reset_session ||
                override_executor != nullptr || api_session || dump != nullptr ||
                inline_refresh || !executor_->block_store() ||
                !kvmem_query_replay_retrieval_supported(
                    executor_->block_store()->config().retrieval_method)) {
                throw std::runtime_error(
                    "semantic-chunk prefill requires a fresh, above-budget, "
                    "query-conditioned Mean-K/SubBlockMeanK MTP request");
            }
            if (!mtp_local_positions) {
                throw std::runtime_error(
                    "semantic-chunk MTP prefill requires local MTP positions");
            }
        }
        const bool transcript_replay =
            transcript_replay_requested && kvmem_on && qc_select_active &&
            reset_session && !kvmem_warm_reuse && !kvmem_warm_capture &&
            dump == nullptr && executor_->block_store() &&
            executor_->block_store()->config().retrieval_method ==
                KvMemRetrievalMethod::MeanK;
        if (transcript_replay_requested && !transcript_replay) {
            throw std::runtime_error(
                "KVMem transcript replay requires a fresh, above-budget, "
                "query-conditioned mean-k MTP request");
        }
        if (transcript_replay) {
            uint32_t previous_end = 0;
            for (size_t i = 0; i < options.kvmem_replay_query_spans.size(); ++i) {
                const auto &span = options.kvmem_replay_query_spans[i];
                if (span.end <= span.begin || span.end > prompt_tokens.size() ||
                    (i > 0 && span.begin < previous_end)) {
                    throw std::runtime_error(
                        "KVMem transcript replay query spans must be sorted, "
                        "non-overlapping, and inside the prompt");
                }
                previous_end = span.end;
            }
            const auto &last = options.kvmem_replay_query_spans.back();
            if (last.begin != options.kvmem_query_begin ||
                last.end != options.kvmem_query_end) {
                throw std::runtime_error(
                    "KVMem transcript replay final span does not match the "
                    "answer-producing query");
            }
        }
        const bool transcript_session_local =
            transcript_replay &&
            env_flag_enabled("QW3_KVMEM_TRANSCRIPT_SESSION_LOCAL");
        if (transcript_session_local) {
            if (options.kvmem_replay_session_starts.empty()) {
                throw std::runtime_error(
                    "KVMem session-local transcript construction requires "
                    "kvmem_session_start message metadata");
            }
            uint32_t previous_start = 0;
            for (size_t i = 0;
                 i < options.kvmem_replay_session_starts.size(); ++i) {
                const uint32_t start =
                    options.kvmem_replay_session_starts[i];
                if (start >= prompt_tokens.size() ||
                    (i > 0 && start < previous_start)) {
                    throw std::runtime_error(
                        "KVMem transcript session starts must be sorted and "
                        "inside the prompt");
                }
                previous_start = start;
            }
        }
        if (kvmem_on) {
            executor_->kvmem_set_trace_metadata(
                options.kvmem_trace_tag,
                options.kvmem_context_begin,
                options.kvmem_context_end,
                prompt_tokens);
            std::vector<std::pair<uint32_t, uint32_t>> retrieval_groups;
            retrieval_groups.reserve(
                options.kvmem_retrieval_group_spans.size());
            for (const auto &span :
                 options.kvmem_retrieval_group_spans) {
                retrieval_groups.emplace_back(span.begin, span.end);
            }
            executor_->kvmem_set_retrieval_group_spans(
                retrieval_groups);
            std::vector<std::pair<uint32_t, uint32_t>> mandatory_spans;
            mandatory_spans.reserve(
                options.kvmem_pinned_token_spans.size());
            for (const auto &span : options.kvmem_pinned_token_spans) {
                mandatory_spans.emplace_back(span.begin, span.end);
            }
            executor_->kvmem_set_mandatory_token_spans(mandatory_spans);
        }
        const uint32_t query_replay_base = executor_->position();
        const bool query_replay =
            !semantic_chunk && kvmem_query_replay_enabled() && kvmem_on &&
            qc_select_active &&
            reset_session && override_executor == nullptr && dump == nullptr &&
            options.kvmem_query_begin > 0 &&
            options.kvmem_query_begin >= query_replay_base &&
            options.kvmem_query_end <= prompt_tokens.size() &&
            prompt_tokens.size() - options.kvmem_query_begin <=
                executor_->block_store()->config().gen_budget &&
            executor_->block_store()->config().retrieval_method !=
                KvMemRetrievalMethod::DeltaNet;
        if (!semantic_chunk && kvmem_query_replay_enabled() && reset_session &&
            override_executor == nullptr && !query_replay) {
            std::ostringstream rmsg;
            rmsg << "native kvmem query-replay skipped (mtp): base="
                 << query_replay_base << " span=[" << options.kvmem_query_begin
                 << "," << options.kvmem_query_end << ") prompt="
                 << prompt_tokens.size() << " gen_budget="
                 << (executor_->block_store()
                          ? executor_->block_store()->config().gen_budget
                          : 0);
            log(rmsg.str());
        }
        // Clean-query prefill (task #50). Engage only above budget with a span, when
        // the flag is on, and NOT on a warm-reuse / prefix-cache-capture turn (those
        // own the prefill split) nor the logit-dump path. warm_reuse/warm_capture are
        // resolved above; boundary capture (a subset of warm_capture) is therefore
        // excluded too. When off, everything below is byte-identical to today.
        const bool clean_query =
            kvmem_on && qc_select_active && kvmem_clean_query_enabled() &&
            !transcript_replay && !query_replay &&
            !kvmem_warm_reuse && !kvmem_warm_capture && dump == nullptr;
        const bool recompute_query =
            kvmem_on && qc_select_active &&
            (semantic_chunk ||
             kvmem_recompute_query_enabled(options_.kvmem_recompute_query)) &&
            !transcript_replay && !query_replay && !clean_query &&
            dump == nullptr && executor_->block_store() &&
            kvmem_query_replay_retrieval_supported(
                executor_->block_store()->config().retrieval_method);
        const bool query_attention_probe_requested =
            options.kvmem_query_attention_probe_tokens != 0 ||
            options.kvmem_query_attention_score_tokens != 0;
        const bool query_guided_query_requested =
            options.kvmem_query_guided_thinking_max_tokens != 0 ||
            options.kvmem_query_guided_query_max_tokens != 0 ||
            options.kvmem_query_guided_direct;
        if (query_attention_probe_requested &&
            (options.kvmem_query_attention_probe_tokens == 0 ||
             options.kvmem_query_attention_score_tokens == 0)) {
            throw std::invalid_argument(
                "KVMem query attention probe requires both a positive probe "
                "length and a positive score-token budget");
        }
        if (query_attention_probe_requested && !recompute_query) {
            throw std::invalid_argument(
                "KVMem query attention probe requires the query-recompute/"
                "replay path with Mean-K or SubBlockMeanK retrieval");
        }
        if (query_guided_query_requested &&
            options.kvmem_query_guided_query_max_tokens == 0) {
            throw std::invalid_argument(
                "KVMem guided query requires a positive compact-query "
                "token budget");
        }
        if (query_guided_query_requested &&
            !options.kvmem_query_guided_direct &&
            options.kvmem_query_guided_thinking_max_tokens == 0) {
            throw std::invalid_argument(
                "KVMem reasoning-guided query requires a positive private-"
                "thinking limit");
        }
        if (options.kvmem_query_guided_direct &&
            options.kvmem_query_guided_thinking_max_tokens != 0) {
            throw std::invalid_argument(
                "KVMem direct guided query requires a zero private-thinking "
                "limit");
        }
        if (query_attention_probe_requested &&
            query_guided_query_requested) {
            throw std::invalid_argument(
                "KVMem attention-probe and guided-query selectors are "
                "independent and cannot be enabled together");
        }
        if (query_guided_query_requested && !recompute_query) {
            throw std::invalid_argument(
                "KVMem guided query requires the query-recompute/replay "
                "path with Mean-K or SubBlockMeanK retrieval");
        }
        const bool query_attention_probe =
            query_attention_probe_requested && recompute_query;
        const bool query_guided_query =
            query_guided_query_requested && recompute_query;
        const bool query_guided_direct =
            query_guided_query && options.kvmem_query_guided_direct;
        executor_->kvmem_set_query_score_budget_hint(
            query_attention_probe
                ? options.kvmem_query_attention_score_tokens
                : query_guided_query
                    ? options.kvmem_query_guided_query_max_tokens
                : 0);
        if (inline_refresh && !recompute_query) {
            throw std::runtime_error(
                "KVMem inline refresh requires an above-budget, "
                "query-conditioned mean-k request with query replay enabled");
        }
#if 0  // Archived DeltaNet recurrent-state import/capture validation.
        if ((rebuilt_state_import || rebuilt_state_capture) &&
            !recompute_query) {
            throw std::runtime_error(
                "KVMem rebuilt-state import/capture requires an above-budget "
                "query-conditioned mean-k request with query replay enabled");
        }
#endif
        const uint32_t api_bt = kvmem_on && executor_->block_store()
            ? std::max<uint32_t>(
                  1, executor_->block_store()->config().block_tokens)
            : 1;
        const uint32_t api_sequence_base = api_session
            ? (reset_session ? 0 : kvmem_api_boundary_pos_) : 0;
        std::vector<uint32_t> api_sequence_tokens;
        uint32_t api_final_boundary = 0;
        if (api_session) {
            if (!reset_session) {
                if ((kvmem_api_boundary_pos_ > 0 &&
                     !kvmem_api_boundary_ckpt_.ready) ||
                    kvmem_api_boundary_pos_ +
                            kvmem_api_tail_tokens_.size() !=
                        api_append_base) {
                    throw std::runtime_error(
                        "KVMem session boundary checkpoint/tail is not "
                        "consistent with the live executor position");
                }
                api_sequence_tokens = kvmem_api_tail_tokens_;
            }
            api_sequence_tokens.insert(api_sequence_tokens.end(),
                                       prompt_tokens.begin(),
                                       prompt_tokens.end());
            api_final_boundary =
                (logical_prompt_tokens / api_bt) * api_bt;
        }
        if (clean_query) {
            // PASS A: capture the query from the question tokens ALONE. The question
            // prefills at positions 0..S attending only over itself + sink, so the
            // de-RoPE'd (content-frame) query the executor captures is free of the
            // recency window that contaminates it in the normal single-pass prefill.
            // Stash it (survives reset_state), then reset so PASS B starts cold.
            const uint32_t qb = options.kvmem_query_begin;
            const uint32_t qe = options.kvmem_query_end;
            std::vector<uint32_t> qtoks(
                prompt_tokens.begin() + static_cast<std::ptrdiff_t>(qb),
                prompt_tokens.begin() + static_cast<std::ptrdiff_t>(qe));
            executor_->reset_state();
            executor_->kvmem_set_query_span(0,
                                            static_cast<uint32_t>(qtoks.size()),
                                            static_cast<uint32_t>(qtoks.size()));
            NativeExecutorReport pa =
                executor_->forward_n_tokens(qtoks, /*need_logits=*/false);
            if (!pa.ok) throw std::runtime_error("clean-query PASS A prefill failed");
            executor_->kvmem_stash_clean_query();
            executor_->reset_state();
            log("native kvmem clean-query PASS A: captured isolated query over "
                + std::to_string(qtoks.size()) + " question tokens");
        }
        // Oracle spans are intentionally NOT installed here.  Real-history
        // prefill, including every pressure selection, must remain identical to
        // the non-oracle run.  They are scoped around the single final-query
        // semantic selection below.
        auto kvmem_final_query_reselect = [&]() {
            if (options.kvmem_oracle_token_spans.empty()) {
                executor_->kvmem_reselect();
                return;
            }
            std::vector<std::pair<uint32_t, uint32_t>> oracle_spans;
            oracle_spans.reserve(options.kvmem_oracle_token_spans.size());
            for (const auto &span : options.kvmem_oracle_token_spans) {
                oracle_spans.emplace_back(span.begin, span.end);
            }
            executor_->kvmem_set_oracle_token_spans(
                oracle_spans, options.kvmem_oracle_only);
            executor_->kvmem_reselect();
            executor_->kvmem_set_oracle_token_spans({});
        };
        // Query capture and semantic selection have separate lifecycles. A
        // prefix-cached request below budget captures Q and the source index for
        // a future threshold crossing, while the current request remains in the
        // dense/identity path.
        if (kvmem_on) {
            executor_->kvmem_set_pin_from_block(0xffffffffu);
        }
        if (kvmem_on && api_session) {
            if (qc_select_active) {
                executor_->kvmem_set_query_span(
                    options.kvmem_query_begin, options.kvmem_query_end,
                    logical_prompt_tokens,
                    /*index_tokens=*/0,
                    /*preserve_content_index=*/!reset_session);
                std::ostringstream qmsg;
                qmsg << "native kvmem session query-conditioned: span=["
                     << options.kvmem_query_begin << ","
                     << options.kvmem_query_end << ") tokens="
                     << (options.kvmem_query_end -
                         options.kvmem_query_begin)
                     << " logical_prompt_tokens=" << logical_prompt_tokens;
                log(qmsg.str());
            } else {
                // Build and extend the position-invariant mean-K source index
                // during teacher-forced history ingest, before any query exists.
                executor_->kvmem_set_query_span(
                    0, 0, logical_prompt_tokens,
                    /*index_tokens=*/0,
                    /*preserve_content_index=*/!reset_session,
                    /*capture_content_without_query=*/true);
            }
        } else if (kvmem_on && qc_capture_active) {
            const uint32_t initial_qb =
                transcript_replay && !transcript_session_local
                ? options.kvmem_replay_query_spans.front().begin
                : options.kvmem_query_begin;
            const uint32_t initial_qe =
                transcript_replay && !transcript_session_local
                ? options.kvmem_replay_query_spans.front().end
                : options.kvmem_query_end;
            executor_->kvmem_set_query_span(
                initial_qb, initial_qe, logical_prompt_tokens,
                /*index_tokens=*/0,
                /*preserve_content_index=*/kvmem_warm_reuse);
            if (kvmem_ru.query_snapshot_reuse) {
                executor_->kvmem_restore_stashed_query();
            }
            std::ostringstream qmsg;
            qmsg << "native kvmem query-conditioned: mode="
                 << (qc_select_active ? "select" : "prepare")
                 << " span=["
                 << initial_qb << "," << initial_qe
                 << ") tokens=" << (initial_qe - initial_qb);
            if (transcript_replay) {
                qmsg << " transcript_events="
                     << options.kvmem_replay_query_spans.size()
                     << " session_local="
                     << (transcript_session_local ? 1 : 0);
            }
            log(qmsg.str());
        } else if (kvmem_on) {
            executor_->kvmem_set_query_span(
                0, 0, logical_prompt_tokens,
                /*index_tokens=*/0,
                /*preserve_content_index=*/false,
                /*capture_content_without_query=*/
                    kvmem_warm_history_index_capture);
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
            // Only reselect once the store has actually spilled past budget;
            // below budget selection is identity, so skip it and use all the
            // context. No-op in step mode regardless.
            if (options_.kvmem_update_mode != "step" &&
                executor_->block_store()->block_count() >
                    executor_->block_store()->budget_blocks() &&
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
        bool use_mtp_prefix =
            (spec_mtp || mtp_prefix_enabled(options_)) &&
            mtp_prefix_positions_safe;
        if (mtp_local_positions && !mtp_prefix_positions_safe) {
            throw std::runtime_error(
                "KVMem local-position MTP cannot construct an in-range compact "
                "prefix for this request; disable QW3_KVMEM_FULLCTX_QUERY or "
                "reduce the KVMem/generation budgets");
        }
        if (!mtp_prefix_positions_safe &&
            (spec_mtp || trace_mtp || mtp_prefix_enabled(options_))) {
            log("native mtp_position_guard: enabled=true logical_prompt_tokens=" +
                std::to_string(logical_prompt_tokens) + " limit=" +
                std::to_string(mtp_rope_limit) + " projected_logical_end=" +
                std::to_string(mtp_logical_end) + " projected_compact_end=" +
                std::to_string(mtp_compact_end) + " local_positions=" +
                std::to_string(mtp_local_positions ? 1 : 0) +
                " compact_prefill_safe=" +
                std::to_string(mtp_compact_prefill_safe ? 1 : 0) +
                " action=disable_mtp_prefix_and_speculation");
        } else if (mtp_compact_prefill_safe &&
                   (spec_mtp || trace_mtp ||
                    mtp_prefix_enabled(options_))) {
            log("native mtp_position_guard: enabled=true logical_prompt_tokens=" +
                std::to_string(logical_prompt_tokens) + " limit=" +
                std::to_string(mtp_rope_limit) + " projected_logical_end=" +
                std::to_string(mtp_logical_end) + " projected_compact_end=" +
                std::to_string(mtp_compact_end) +
                " local_positions=1 action=use_compact_window_positions");
        }
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
            if (mtp_local_positions) {
                log("native mtp_prefix: ignoring QW3_MTP_PREFIX_MAX_PROMPT "
                    "because local-position KVMem requires a complete MTP "
                    "prefix/V history");
            } else {
                log("native mtp_prefix: ok=false reason=\"prompt exceeds "
                    "QW3_MTP_PREFIX_MAX_PROMPT\"");
                use_mtp_prefix = false;
            }
        }

        // MTP prefix priming reuses the last batch's hidden rows out of the
        // executor's h_batch_ scratch, which only retains the final internal
        // chunk. Drive chunk boundaries here at exactly the width
        // forward_n_tokens would use internally (effective_prefill_chunk_size),
        // so each backend chunk is processed as a single batch and
        // prime_mtp_prefix_from_last_batch sees the whole chunk's hidden rows.
        // Keep the configured prefill chunk as the base. Above budget the
        // bounded GPU pool may still cap the width, which the priming loop
        // honors instead of forcing a re-split that would disable the path.
        if (use_mtp_prefix) {
            executor_->set_prefill_chunk_override(options_.prefill_chunk);
        }

        uint32_t mtp_prefix_tokens = 0;
        uint64_t mtp_prefix_ops = 0;
        auto prime_mtp_prefix = [&](const std::vector<uint32_t> &tokens,
                                    uint32_t base_position) {
            if (!use_mtp_prefix) return;
            // The provisional semantic pass only captures target-model Q and
            // is rolled back immediately. Building draft-prefix K/V here would
            // double MTP prefill work without affecting selection or output;
            // the durable replay below primes MTP once under the chosen window.
            if (executor_->kvmem_provisional_prefill()) {
                if (mtp_local_positions) {
                    executor_->kvmem_set_defer_prefill_pressure(false);
                }
                return;
            }
            NativeExecutorReport mtp;
            try {
                if (mtp_local_positions) {
                    const uint32_t rows = executor_->last_forward_rows();
                    const uint32_t logical_base =
                        executor_->last_forward_logical_base();
                    const uint32_t rope_base =
                        executor_->last_forward_rope_base();
                    if (rows != tokens.size() ||
                        logical_base != base_position) {
                        throw std::runtime_error(
                            "MTP compact prefix lost the target chunk position "
                            "metadata");
                    }
                    mtp = executor_->prime_mtp_prefix_from_last_batch_at(
                        tokens, logical_base, rope_base);
                } else {
                    mtp = executor_->prime_mtp_prefix_from_last_batch(
                        tokens, base_position);
                }
            } catch (...) {
                if (mtp_local_positions) {
                    executor_->kvmem_set_defer_prefill_pressure(false);
                }
                throw;
            }
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
                if (mtp_local_positions) {
                    executor_->kvmem_set_defer_prefill_pressure(false);
                }
                return;
            }
            // The target rows and MTP K/V for this chunk are now both durable.
            // Start SSD write-through before pressure selection: the new D2H
            // remains non-destructive and can overlap the next MAIN prefill.
            executor_->kvmem_prefill_writeback(
                base_position +
                static_cast<uint32_t>(tokens.size()));
            // Pressure selection may safely spill pages only after this point.
            if (mtp_local_positions) {
                executor_->kvmem_finish_deferred_prefill_pressure();
            }
            mtp_prefix_tokens += static_cast<uint32_t>(tokens.size());
            mtp_prefix_ops += mtp.ops_executed;
        };

        double post_semantic_reselect_s = 0.0;
        double post_query_replay_s = 0.0;
        double post_query_attention_probe_s = 0.0;
        double post_query_guided_query_s = 0.0;
        uint32_t query_attention_probe_decoded = 0;
        bool query_attention_probe_used = false;
        std::vector<uint32_t> query_score_token_indices;
        uint32_t query_guided_thinking_tokens = 0;
        uint32_t query_guided_query_tokens = 0;
        bool query_guided_thinking_closed = false;
        bool query_guided_fallback_original = false;
        std::string query_guided_query_text;
        std::string query_guided_fallback_reason;
        const double t_prefill_start = wall_seconds();
        const uint32_t prefill_base =
            static_cast<uint32_t>(executor_->position());
        // MTP priming below reads executor_->position() immediately before each
        // main-model chunk. This is the absolute append base on cold prefill,
        // warm reuse, block-aligned replay, and shifted refresh replay alike.
        std::vector<uint32_t> kvmem_suffix;
        if (kvmem_warm_reuse) {
            kvmem_suffix.assign(prompt_tokens.begin() + kvmem_reuse_m,
                                prompt_tokens.end());
        }
        const std::vector<uint32_t> &prefill_tokens =
            kvmem_warm_reuse ? kvmem_suffix : prompt_tokens;
        const size_t kvmem_prefill_begin = kvmem_warm_reuse ? kvmem_reuse_m : 0;
        // Common prefill entry for checkpoint reuse and persistent
        // reset_session=false growth: discard the prior semantic working set
        // before any new above-budget token is evaluated.
        executor_->kvmem_prepare_prefill_window(
            static_cast<uint32_t>(prefill_tokens.size()));
        // kvmem prefix cache: choose where to stage the prompt-end (P) checkpoint.
        // Capturing at the last BLOCK boundary strictly below P (rather than at P
        // itself) leaves a resume point below the chat-template reformat that
        // rewrites the final few tokens each turn (observed D == P-4: the empty
        // <think> block injected for the live turn is dropped once it is history),
        // so next turn's longest-common-prefix D >= B and the reuse actually
        // fires. Dense/untiered OR a below-budget tiered turn keep [0,B)
        // GPU-resident with identity bakes (the below-budget ckpt_P precondition);
        // above budget with tiers the same boundary is resumable because [0,B)
        // stays recoverable across tiers and the mean-k index is
        // position-invariant. per-token above budget can't be fixed-stride, so it
        // is excluded here and captured as a non-resumable P below.
        const bool kvmem_qc_pertoken_here = executor_->kvmem_qc_pertoken();
        uint32_t kvmem_warm_prompt_pos = 0;
        bool kvmem_warm_prompt_resumable = false;
        uint32_t kvmem_ckpt_split = 0;
        bool kvmem_do_boundary_capture = false;
        bool kvmem_warm_checkpoint_staged = false;
        QwenExecutor::StateSnapshot kvmem_query_replay_ckpt;
        uint32_t kvmem_query_replay_begin = 0;
        if (recompute_query) {
            const uint32_t bt = std::max<uint32_t>(
                1, executor_->block_store()->config().block_tokens);
            kvmem_query_replay_begin = kvmem_ru.query_snapshot_reuse
                ? kvmem_reuse_m
                : (kvmem_effective_replay_begin(options) / bt) * bt;
            if (kvmem_query_replay_begin == 0) {
                throw std::runtime_error(
                    "KVMem query replay requires a non-zero aligned boundary");
            }
            if (api_session &&
                kvmem_query_replay_begin < api_append_base) {
                if (!kvmem_api_boundary_ckpt_.ready ||
                    kvmem_api_boundary_pos_ !=
                        kvmem_query_replay_begin) {
                    throw std::runtime_error(
                        "KVMem session query replay cannot recover the "
                        "preceding partial block boundary");
                }
                // State at B was captured while ingesting the previous
                // request. The replay suffix below prepends the saved B..P tail
                // before the new query fragment.
                kvmem_query_replay_ckpt =
                    std::move(kvmem_api_boundary_ckpt_);
            }
        }
        if (kvmem_warm_capture && kvmem_on && !transcript_replay &&
            !query_replay && !kvmem_qc_pertoken_here) {
            const uint32_t bt = executor_->block_store()
                                    ? executor_->block_store()->config().block_tokens
                                    : 256;
            const uint32_t prompt_end = static_cast<uint32_t>(prompt_tokens.size());
            const uint32_t split = kvmem_prompt_checkpoint_split(
                prompt_end, static_cast<uint32_t>(kvmem_prefill_begin), bt);
            if (split > 0) {
                kvmem_ckpt_split = split;
                kvmem_do_boundary_capture = true;
            }
        }
        QwenExecutor::KvMemTimingSnapshot kvmem_tbase;
        if (kvmem_on && QwenExecutor::kvmem_timing_enabled()) {
            kvmem_tbase = QwenExecutor::kvmem_timing_snapshot();
        }
        uint64_t prefill_ops = 0;
        NativeExecutorReport step;
        uint32_t prefill_chunks = 0;
        uint32_t prefill_chunk_size = 0;
        const bool trace_prefill = prefill_trace_enabled();
        // Prefill the local range [lbegin, lend) of prefill_tokens. Factored out
        // so the block-boundary checkpoint can split the prompt prefill at `split`
        // (capture recurrent state there) and resume the tail without duplicating
        // the two prefill flavors (internal-chunking vs MTP-prefix priming).
        auto do_prefill_vector_raw = [&](const std::vector<uint32_t> &tokens,
                                         size_t lbegin, size_t lend,
                                         bool compute_final_logits = true) {
            if (lbegin >= lend) return;
            if (!use_mtp_prefix) {
                std::vector<uint32_t> seg(
                    tokens.begin() + static_cast<std::ptrdiff_t>(lbegin),
                    tokens.begin() + static_cast<std::ptrdiff_t>(lend));
                const double t_chunk_start = wall_seconds();
                step = executor_->forward_n_tokens(
                    seg, options.max_tokens > 0 && compute_final_logits &&
                             lend == tokens.size());
                if (!step.ok) throw std::runtime_error("prefill failed");
                const double t_chunk_end = wall_seconds();
                prefill_ops += step.ops_executed;
                if (trace_prefill) {
                    log_prefill_chunk(prefill_chunks, lbegin, seg.size(),
                                      t_chunk_end - t_chunk_start);
                }
                ++prefill_chunks;
            } else {
                for (size_t offset = lbegin; offset < lend;) {
                    const uint32_t remaining = static_cast<uint32_t>(lend - offset);
                    // Width forward_n_tokens will actually run as one internal
                    // chunk: the configured override below budget, pool-capped
                    // above budget. Matching it keeps prime_mtp_prefix's batch
                    // equal to the chunk it was primed for.
                    const uint32_t width = std::max<uint32_t>(
                        1, executor_->effective_prefill_chunk_size(remaining));
                    const size_t end =
                        offset + std::min<size_t>(remaining, width);
                    std::vector<uint32_t> chunk(
                        tokens.begin() + static_cast<std::ptrdiff_t>(offset),
                        tokens.begin() + static_cast<std::ptrdiff_t>(end));
                    const double t_chunk_start = wall_seconds();
                    const bool need_logits =
                        options.max_tokens > 0 && compute_final_logits &&
                        end == tokens.size();
                    // Use the executor's live position, not the original prompt
                    // offset. They are equal on ordinary prefill and ordinary
                    // block-aligned replay, while an ephemeral selected-session
                    // refresh deliberately shifts the final query to the right.
                    const uint32_t chunk_base = executor_->position();
                    if (mtp_local_positions) {
                        executor_->kvmem_set_defer_prefill_pressure(true);
                    }
                    try {
                        step = executor_->forward_n_tokens(chunk, need_logits);
                    } catch (...) {
                        if (mtp_local_positions) {
                            executor_->kvmem_set_defer_prefill_pressure(false);
                        }
                        throw;
                    }
                    if (!step.ok) {
                        if (mtp_local_positions) {
                            executor_->kvmem_set_defer_prefill_pressure(false);
                        }
                        throw std::runtime_error("prefill failed");
                    }
                    prime_mtp_prefix(chunk, chunk_base);
                    const double t_chunk_end = wall_seconds();
                    prefill_ops += step.ops_executed;
                    if (trace_prefill) {
                        log_prefill_chunk(prefill_chunks, offset, chunk.size(),
                                          t_chunk_end - t_chunk_start);
                    }
                    ++prefill_chunks;
                    prefill_chunk_size = static_cast<uint32_t>(chunk.size());
                    offset = end;
                }
            }
        };
        // Capture the last block-aligned state of an API session while the
        // corresponding tokens are actually being prefilled. Query replay may
        // call this a second time and intentionally overwrite the first-pass
        // snapshot with the state built against the selected semantic window.
        auto do_prefill_vector = [&](const std::vector<uint32_t> &tokens,
                                     uint32_t absolute_base,
                                     size_t lbegin, size_t lend,
                                     bool compute_final_logits = true) {
            if (lbegin >= lend) return;
            const uint64_t abs_begin =
                static_cast<uint64_t>(absolute_base) + lbegin;
            const uint64_t abs_end =
                static_cast<uint64_t>(absolute_base) + lend;
            if (api_session && api_final_boundary > abs_begin &&
                api_final_boundary <= abs_end) {
                const size_t split = static_cast<size_t>(
                    api_final_boundary - absolute_base);
                do_prefill_vector_raw(
                    tokens, lbegin, split, /*compute_final_logits=*/false);
                executor_->capture_state(kvmem_api_boundary_ckpt_);
                kvmem_api_boundary_pos_ = api_final_boundary;
                do_prefill_vector_raw(
                    tokens, split, lend, compute_final_logits);
            } else {
                do_prefill_vector_raw(
                    tokens, lbegin, lend, compute_final_logits);
            }
        };
        const uint32_t prefill_absolute_base = api_session
            ? api_append_base
            : static_cast<uint32_t>(kvmem_prefill_begin);
        auto do_prefill_range = [&](size_t lbegin, size_t lend,
                                    bool compute_final_logits = true) {
            do_prefill_vector(prefill_tokens, prefill_absolute_base,
                              lbegin, lend, compute_final_logits);
        };
        auto do_query_prefill_range = [&](size_t lbegin, size_t lend,
                                          bool compute_final_logits) {
            executor_->kvmem_set_prefill_reselect_suppressed(true);
            try {
                do_prefill_range(lbegin, lend, compute_final_logits);
            } catch (...) {
                executor_->kvmem_set_prefill_reselect_suppressed(false);
                throw;
            }
            executor_->kvmem_set_prefill_reselect_suppressed(false);
        };
        bool query_replay_applied = false;
        KvMemSemanticChunkStats semantic_chunk_stats;
        if (dump) {
            if (use_mtp_prefix) {
                log("native mtp_prefix: ok=false reason=\"dump logits path does not expose batch hidden rows\"");
                use_mtp_prefix = false;
            }
            for (size_t pi = kvmem_prefill_begin; pi < prompt_tokens.size(); ++pi) {
                step = executor_->forward_one_token(prompt_tokens[pi]);
                if (!step.ok) throw std::runtime_error("prefill failed");
                prefill_ops += step.ops_executed;
                dump->record(static_cast<int>(pi), "prefill",
                             static_cast<int32_t>(prompt_tokens[pi]),
                             *executor_, *tokenizer_);
            }
        } else if (query_replay) {
            const uint32_t qb = options.kvmem_query_begin;
            const uint32_t qe = options.kvmem_query_end;
            const uint32_t bt = executor_->block_store()->config().block_tokens;
            const size_t query_local = qb - prefill_base;
            const double checkpoint_start = wall_seconds();

            do_prefill_range(0, query_local, /*compute_final_logits=*/false);
            executor_->kvmem_register_append(qb - prefill_base);
            executor_->kvmem_reselect();
            const std::vector<uint32_t> selected_before =
                kvmem_selected_block_ids(executor_);
            QwenExecutor::StateSnapshot query_checkpoint;
            executor_->capture_state(query_checkpoint);
            const double checkpoint_end = wall_seconds();

            const double provisional_start = wall_seconds();
            do_query_prefill_range(query_local, prefill_tokens.size(),
                                   /*compute_final_logits=*/true);
            const double provisional_end = wall_seconds();
            if (executor_->kvmem_stash_query()) {
                const double restore_start = wall_seconds();
                executor_->restore_state(query_checkpoint);
                executor_->kvmem_truncate_to(qb);
                executor_->kvmem_set_query_span(
                    qb, qe, static_cast<uint32_t>(prompt_tokens.size()), qb);
                executor_->kvmem_restore_stashed_query();
                executor_->kvmem_set_pin_from_block(qb / std::max<uint32_t>(bt, 1));
                const bool index_ready = executor_->kvmem_query_selection_ready();
                const QwenExecutor::KvMemTimingSnapshot select_t0 =
                    QwenExecutor::kvmem_timing_snapshot();
                const double select_start = wall_seconds();
                executor_->kvmem_reselect();
                const double select_end = wall_seconds();
                const QwenExecutor::KvMemTimingSnapshot select_t1 =
                    QwenExecutor::kvmem_timing_snapshot();
                const std::vector<uint32_t> selected_after =
                    kvmem_selected_block_ids(executor_);
                const uint32_t additions =
                    kvmem_selection_additions(selected_before, selected_after);

                executor_->kvmem_reset_query_capture();
                const double replay_start = wall_seconds();
                do_query_prefill_range(query_local, prefill_tokens.size(),
                                       /*compute_final_logits=*/true);
                executor_->kvmem_register_append(
                    static_cast<uint32_t>(prompt_tokens.size()) - qb);
                const double replay_end = wall_seconds();
                query_replay_applied = true;

                std::ostringstream rmsg;
                rmsg << std::fixed << std::setprecision(3)
                     << "native kvmem query-replay (mtp): span=[" << qb << ","
                     << qe << ") tail=" << (prompt_tokens.size() - qb)
                     << " checkpoint_ms=" << (checkpoint_end - checkpoint_start) * 1e3
                     << " provisional_ms=" << (provisional_end - provisional_start) * 1e3
                     << " restore_ms=" << (select_start - restore_start) * 1e3
                     << " select_ms=" << (select_end - select_start) * 1e3
                     << " replay_ms=" << (replay_end - replay_start) * 1e3
                     << " selected=" << selected_after.size()
                     << " replaced=" << additions
                     << " replace_rate="
                     << (selected_after.empty()
                             ? 0.0
                             : static_cast<double>(additions) / selected_after.size())
                     << " stage_in="
                     << (select_t1.stage_in_blocks - select_t0.stage_in_blocks)
                     << " stage_out="
                     << (select_t1.stage_out_blocks - select_t0.stage_out_blocks)
                     << " index_ready=" << (index_ready ? 1 : 0)
                     << " final_pos=" << executor_->position();
                log(rmsg.str());
            } else {
                log("native kvmem query-replay (mtp): provisional query capture "
                    "was incomplete; keeping the single-pass result");
            }
        } else if (kvmem_do_boundary_capture && !recompute_query) {
            const size_t split_local =
                kvmem_ckpt_split - static_cast<uint32_t>(kvmem_prefill_begin);
            // First segment [prefill_base, split): advance recurrent state to B.
            do_prefill_range(0, split_local,
                             /*compute_final_logits=*/false);
            // Register + reselect so the store/window describe exactly `split`
            // tokens, then stage ckpt_P at the block boundary. The recurrent state
            // is captured here, physically at B — it cannot be rewound from a later
            // snapshot, which is why the prefill is split.
            executor_->kvmem_register_append(static_cast<uint32_t>(split_local));
            executor_->kvmem_reselect_prefill_pressure();
            kvmem_warm_valid_ = false;  // invalid until end-capture re-validates
            executor_->capture_state(kvmem_warm_ckpt_prompt_);
            kvmem_warm_prompt_pos = kvmem_ckpt_split;
            kvmem_warm_checkpoint_staged = true;
            // Below budget: strict all-GPU-identity. Above budget: identity never
            // holds, but the boundary is resumable when tiers keep [0,B)
            // recoverable (mean-k, position-invariant; per-token excluded above).
            kvmem_warm_prompt_resumable =
                kvmem_prompt_checkpoint_resumable(
                    kvmem_warm_source_index_ready);
            // Second segment [split, P): finish the prompt.
            do_prefill_range(split_local, prefill_tokens.size(),
                             /*compute_final_logits=*/true);
        } else if (transcript_replay) {
            // Experimental role-preserving replay. Dense/pressure-prefill until
            // the first post-threshold user query; thereafter each recorded user
            // query captures Q, semantically reselects a 224K working context,
            // rewinds to the query's block boundary, and re-prefills the query
            // against that new context. The bytes after an INTERMEDIATE query are
            // the trace's recorded assistant response/tool traffic and are
            // teacher-forced by the next do_prefill_range call. No sampler is
            // entered until this entire loop and the final assistant header end.
            const uint32_t bt = std::max<uint32_t>(
                1, executor_->block_store()->config().block_tokens);
            // Optional A/B: compute every first-pass user query against the same
            // deterministic sink+recent bootstrap window. Without this, event N's
            // query representation is conditioned on the semantic working set
            // selected by event N-1, so identical query text can rank different
            // history blocks solely because an unrelated prior question changed
            // the live window. The semantic selection and the block-aligned query
            // replay below are unchanged. Keep this opt-in until the frozen
            // LongMemEval-M comparison establishes that it is beneficial.
            const bool transcript_query_bootstrap =
                env_flag_enabled("QW3_KVMEM_TRANSCRIPT_QUERY_BOOTSTRAP");
            // Stable memory ingest: use semantic selection to answer/replay the
            // arriving query, but rebuild that short query into the deterministic
            // sink+recent pressure window before teacher-forcing its historical
            // response. This keeps long-term source KV construction independent
            // of the unrelated semantic window selected by the previous turn,
            // without re-prefilling the 224K history.
            const bool transcript_stable_ingest =
                env_flag_enabled("QW3_KVMEM_TRANSCRIPT_STABLE_INGEST");
            // Equivalent fast path for benchmark transcript ingestion. An
            // intermediate semantic replay is immediately truncated and
            // overwritten by the stable pressure replay before any recorded
            // response is teacher-forced, so it cannot affect durable K/V or
            // recurrent state. Skip that dead semantic assemble/replay and keep
            // the already-prefilled pressure-window query. The final event still
            // performs the ordinary mean-K selection and semantic replay. A rare
            // next-query boundary that falls inside this event keeps the slow
            // path because it needs a replay-time checkpoint inside the block.
            const bool transcript_stable_fast =
                transcript_stable_ingest && env_flag_enabled(
                    "QW3_KVMEM_TRANSCRIPT_STABLE_FAST");
            // Diagnostic isolation for hybrid models: keep the exact same
            // selected normal-attention K/V, but start the final query replay
            // from an empty recurrent/conv state instead of the chronological
            // checkpoint. This distinguishes cached-KV construction error from
            // stale/global DeltaNet-state conditioning. Intermediate turns keep
            // their normal state so the source-cache construction experiment is
            // otherwise unchanged.
            // Disabled: transcript replay must preserve the model's chronological
            // recurrent/conv state.  Clearing it here changes the model semantics
            // and only appears to work when the selected source text is replayed
            // to build a replacement state.  That O(selected_tokens) rebuild is a
            // diagnostic, not a viable KVMem runtime policy.
            const bool transcript_reset_final_recurrent = false;
            // Stronger construction A/B: every durable user-turn ingest starts
            // a fresh recurrent/conv segment. Normal-attention still reads the
            // fixed KVMem window, so historical KV reuse is preserved; only the
            // compact DeltaNet carry is prevented from leaking arbitrary prior
            // windows into the newly stored source K/V. With stable-ingest, the
            // discarded semantic replay is left untouched and only the durable
            // pressure replay is reset. The final answer query is reset as well.
            const bool transcript_reset_each_recurrent = false;
            size_t cursor = 0;
            if (transcript_session_local) {
                // Build a reusable memory representation per independent
                // transcript session. Session starts are rounded DOWN to the
                // block boundary: the at-most-(bt-1) preceding tokens become a
                // small overlap prefix, which preserves the invariant that one
                // physical block has one attention/RoPE construction frame.
                // At each boundary normal attention keeps only the fixed sink
                // while recurrent/conv state is cleared. The session then grows
                // naturally from its own tokens. No historical selected window
                // is re-prefilled; each main token is computed once apart from
                // the ordinary final-query replay.
                const auto &final_event =
                    options.kvmem_replay_query_spans.back();
                std::vector<uint32_t> session_boundaries;
                session_boundaries.reserve(
                    options.kvmem_replay_session_starts.size());
                for (uint32_t start :
                     options.kvmem_replay_session_starts) {
                    if (start >= final_event.begin) break;
                    const uint32_t boundary = (start / bt) * bt;
                    if (boundary > 0) {
                        session_boundaries.push_back(boundary);
                    }
                }
                std::sort(session_boundaries.begin(),
                          session_boundaries.end());
                session_boundaries.erase(
                    std::unique(session_boundaries.begin(),
                                session_boundaries.end()),
                    session_boundaries.end());

                uint32_t session_no = 0;
                for (uint32_t boundary : session_boundaries) {
                    if (boundary < cursor) {
                        throw std::runtime_error(
                            "KVMem session-local boundary moved backwards");
                    }
                    do_prefill_range(cursor, boundary);
                    if (kvmem_registered_pos < boundary) {
                        executor_->kvmem_register_append(
                            boundary - kvmem_registered_pos);
                        kvmem_registered_pos = boundary;
                    }
                    QwenExecutor::StateSnapshot session_boundary;
                    executor_->capture_state(session_boundary);

                    const auto &blocks = executor_->block_store()->blocks();
                    const uint32_t sink_count = std::min<uint32_t>(
                        executor_->block_store()->config().sink_blocks,
                        static_cast<uint32_t>(blocks.size()));
                    std::vector<uint32_t> sink_context;
                    sink_context.reserve(sink_count);
                    for (uint32_t id = 0; id < sink_count; ++id) {
                        if (blocks[id].orig_pos_start < boundary) {
                            sink_context.push_back(id);
                        }
                    }
                    executor_->kvmem_begin_query_replay(
                        session_boundary, sink_context,
                        /*reset_recurrent_state=*/false);
                    executor_->kvmem_end_query_replay();
                    kvmem_registered_pos = boundary;
                    cursor = boundary;
                    ++session_no;
                    if (std::getenv("QW3_KVMEM_TRACE")) {
                        log("native kvmem session-local boundary=" +
                            std::to_string(boundary) + " session=" +
                            std::to_string(session_no) + "/" +
                            std::to_string(session_boundaries.size()) +
                            " sink_blocks=" +
                            std::to_string(sink_context.size()));
                    }
                }

                const uint32_t boundary =
                    (final_event.begin / bt) * bt;
                if (boundary == 0 || boundary < cursor) {
                    throw std::runtime_error(
                        "KVMem session-local final query has an invalid "
                        "block boundary");
                }
                do_prefill_range(cursor, boundary);
                if (kvmem_registered_pos < boundary) {
                    executor_->kvmem_register_append(
                        boundary - kvmem_registered_pos);
                    kvmem_registered_pos = boundary;
                }
                QwenExecutor::StateSnapshot final_boundary;
                executor_->capture_state(final_boundary);
                do_prefill_range(boundary, final_event.end);
                executor_->kvmem_register_append(
                    final_event.end - kvmem_registered_pos);
                kvmem_registered_pos = final_event.end;
                if (!executor_->kvmem_publish_captured_prefix(boundary)) {
                    throw std::runtime_error(
                        "KVMem session-local construction could not publish "
                        "the final mean-K index/query");
                }

                executor_->kvmem_set_pin_from_block(boundary / bt);
                executor_->kvmem_set_trace_reselect_event(0, 1);
                executor_->kvmem_reselect();
                std::vector<uint32_t> selected_context;
                bool suffix_fully_selected = true;
                const auto &blocks = executor_->block_store()->blocks();
                for (const KvMemBlock &b : blocks) {
                    if (b.orig_pos_start < boundary) {
                        if (b.in_working_set) {
                            selected_context.push_back(b.block_id);
                        }
                    } else if (!b.in_working_set) {
                        suffix_fully_selected = false;
                    }
                }
                if (!suffix_fully_selected) {
                    throw std::runtime_error(
                        "KVMem session-local selection dropped part of the "
                        "final query suffix");
                }

                // Hybrid-state refresh: cached normal-attention K/V remains the
                // large context, while a small number of lexically relevant,
                // complete source sessions is teacher-forced once to build a
                // query-compatible DeltaNet/recurrent summary. This is bounded
                // by an explicit token budget; cached+fresh context stays
                // inside select_budget and the full gen reserve remains free.
                const uint32_t refresh_budget = env_uint32_or(
                    "QW3_KVMEM_TRANSCRIPT_REFRESH_TOKENS", 0);
                std::vector<uint32_t> refresh_tokens;
                std::vector<std::pair<uint32_t, uint32_t>> refresh_ranges;
                uint32_t refresh_session_count = 0;
                const bool refresh_from_mean_k_blocks =
                    refresh_budget > 0 && env_flag_enabled(
                        "QW3_KVMEM_TRANSCRIPT_REFRESH_BLOCKS");
                if (refresh_budget > 0) {
                    if (refresh_from_mean_k_blocks) {
                        std::vector<uint32_t> ranked_blocks;
                        ranked_blocks.reserve(selected_context.size());
                        const uint32_t sink_blocks =
                            executor_->block_store()->config().sink_blocks;
                        for (uint32_t id : selected_context) {
                            const KvMemBlock &block = blocks[id];
                            if (id < sink_blocks ||
                                block.orig_pos_start >= boundary ||
                                block.n_tokens != bt) {
                                continue;
                            }
                            ranked_blocks.push_back(id);
                        }
                        std::sort(
                            ranked_blocks.begin(), ranked_blocks.end(),
                            [&](uint32_t a, uint32_t b) {
                                const double as = blocks[a].retrieval_score;
                                const double bs = blocks[b].retrieval_score;
                                if (as != bs) return as > bs;
                                return a > b;
                            });
                        const uint32_t refresh_block_cap = refresh_budget / bt;
                        const uint32_t default_core_tokens =
                            std::min<uint32_t>(refresh_budget, 32768);
                        const uint32_t core_tokens = env_uint32_or(
                            "QW3_KVMEM_TRANSCRIPT_REFRESH_CORE_TOKENS",
                            default_core_tokens);
                        const uint32_t core_block_cap = std::min<uint32_t>(
                            refresh_block_cap,
                            std::max<uint32_t>(1, core_tokens / bt));
                        const uint32_t kept_core = std::min<uint32_t>(
                            core_block_cap,
                            static_cast<uint32_t>(ranked_blocks.size()));
                        std::vector<uint32_t> chosen_blocks(
                            ranked_blocks.begin(),
                            ranked_blocks.begin() + kept_core);
                        uint32_t session_added = 0;
                        uint32_t session_blocks_added = 0;
                        uint32_t neighbor_added = 0;
                        if (chosen_blocks.size() < refresh_block_cap) {
                            // Preserve the complete 32K retrieval core, then
                            // spend only additional refresh capacity on local
                            // continuity. This differs from replacing seeds by
                            // neighbors: low-ranked-but-essential facts in the
                            // core can never be displaced.
                            const uint32_t neighbor_radius = env_uint32_or(
                                "QW3_KVMEM_TRANSCRIPT_REFRESH_NEIGHBOR_BLOCKS",
                                2);
                            std::vector<uint8_t> chosen(blocks.size(), 0);
                            std::vector<uint8_t> candidate(blocks.size(), 0);
                            std::vector<uint32_t> core_rank(
                                blocks.size(), UINT32_MAX);
                            for (uint32_t seed_index = 0;
                                 seed_index < kept_core; ++seed_index) {
                                const uint32_t seed = ranked_blocks[seed_index];
                                chosen[seed] = 1;
                                core_rank[seed] = seed_index;
                            }
                            if (env_flag_enabled(
                                    "QW3_KVMEM_TRANSCRIPT_"
                                    "REFRESH_SESSION_EXPAND")) {
                                struct BlockSessionCandidate {
                                    uint32_t begin = 0;
                                    uint32_t end = 0;
                                    uint32_t best_core_rank = UINT32_MAX;
                                    double score_mass = 0.0;
                                    std::vector<uint32_t> missing_blocks;
                                };
                                std::vector<BlockSessionCandidate> sessions;
                                sessions.reserve(
                                    options.kvmem_replay_session_starts.size());
                                for (size_t si = 0;
                                     si < options
                                              .kvmem_replay_session_starts
                                              .size();
                                     ++si) {
                                    const uint32_t begin =
                                        options.kvmem_replay_session_starts[si];
                                    if (begin >= final_event.begin) break;
                                    const uint32_t end = std::min<uint32_t>(
                                        si + 1 < options
                                                     .kvmem_replay_session_starts
                                                     .size()
                                            ? options
                                                  .kvmem_replay_session_starts
                                                      [si + 1]
                                            : final_event.begin,
                                        final_event.begin);
                                    if (end > begin) {
                                        sessions.push_back(
                                            {begin, end, UINT32_MAX, 0.0, {}});
                                    }
                                }
                                size_t session_index = 0;
                                for (uint32_t id = sink_blocks;
                                     id < blocks.size() &&
                                     session_index < sessions.size();
                                     ++id) {
                                    const KvMemBlock &block = blocks[id];
                                    if (block.orig_pos_start >= boundary) break;
                                    while (session_index < sessions.size() &&
                                           block.orig_pos_start >=
                                               sessions[session_index].end) {
                                        ++session_index;
                                    }
                                    if (session_index == sessions.size()) break;
                                    if (block.n_tokens != bt) continue;
                                    // A 32-token block may straddle an exact
                                    // message/session boundary. Associate it
                                    // with both overlapping sessions so that
                                    // expanding the later session cannot lose
                                    // its role marker or first few words.
                                    for (size_t overlap = session_index;
                                         overlap < sessions.size() &&
                                         sessions[overlap].begin <
                                             block.orig_pos_end();
                                         ++overlap) {
                                        BlockSessionCandidate &session =
                                            sessions[overlap];
                                        if (block.orig_pos_end() <=
                                            session.begin) {
                                            continue;
                                        }
                                        session.score_mass +=
                                            block.retrieval_score;
                                        if (core_rank[id] != UINT32_MAX) {
                                            session.best_core_rank = std::min(
                                                session.best_core_rank,
                                                core_rank[id]);
                                        } else {
                                            session.missing_blocks.push_back(id);
                                        }
                                    }
                                }
                                sessions.erase(
                                    std::remove_if(
                                        sessions.begin(), sessions.end(),
                                        [](const BlockSessionCandidate &s) {
                                            return s.best_core_rank == UINT32_MAX ||
                                                   s.missing_blocks.empty();
                                        }),
                                    sessions.end());
                                std::sort(
                                    sessions.begin(), sessions.end(),
                                    [](const BlockSessionCandidate &a,
                                       const BlockSessionCandidate &b) {
                                        if (a.best_core_rank !=
                                            b.best_core_rank) {
                                            return a.best_core_rank <
                                                   b.best_core_rank;
                                        }
                                        if (a.score_mass != b.score_mass) {
                                            return a.score_mass > b.score_mass;
                                        }
                                        return a.begin > b.begin;
                                    });
                                for (const BlockSessionCandidate &session :
                                     sessions) {
                                    const size_t remaining =
                                        refresh_block_cap - chosen_blocks.size();
                                    const size_t needed = std::count_if(
                                        session.missing_blocks.begin(),
                                        session.missing_blocks.end(),
                                        [&](uint32_t id) { return !chosen[id]; });
                                    if (needed > remaining) {
                                        continue;
                                    }
                                    uint32_t added = 0;
                                    for (const uint32_t id :
                                         session.missing_blocks) {
                                        if (chosen[id]) continue;
                                        chosen[id] = 1;
                                        chosen_blocks.push_back(id);
                                        ++added;
                                    }
                                    if (added > 0) {
                                        ++session_added;
                                        session_blocks_added += added;
                                        if (std::getenv("QW3_KVMEM_TRACE")) {
                                            std::ostringstream smsg;
                                            smsg << "native kvmem "
                                                    "refresh-expand-session "
                                                 << "begin=" << session.begin
                                                 << " end=" << session.end
                                                 << " best_rank="
                                                 << (session.best_core_rank + 1)
                                                 << " score_mass="
                                                 << session.score_mass
                                                 << " added_blocks=" << added;
                                            log(smsg.str());
                                        }
                                    }
                                    if (chosen_blocks.size() ==
                                        refresh_block_cap) {
                                        break;
                                    }
                                }
                            }
                            // Complete-session expansion handles long answers
                            // such as lists. Use the remaining capacity for
                            // shorter block-level gaps around the retrieval
                            // core.
                            if (chosen_blocks.size() < refresh_block_cap) {
                                for (uint32_t seed_index = 0;
                                     seed_index < kept_core; ++seed_index) {
                                    const uint32_t seed =
                                        ranked_blocks[seed_index];
                                    for (uint32_t distance = 1;
                                         distance <= neighbor_radius;
                                         ++distance) {
                                        const uint64_t right =
                                            static_cast<uint64_t>(seed) +
                                            distance;
                                        const uint32_t neighbors[2] = {
                                            seed >= distance ? seed - distance
                                                             : UINT32_MAX,
                                            right < blocks.size()
                                                ? static_cast<uint32_t>(right)
                                                : UINT32_MAX};
                                        for (const uint32_t id : neighbors) {
                                            if (id == UINT32_MAX || chosen[id] ||
                                                candidate[id]) {
                                                continue;
                                            }
                                            const KvMemBlock &block = blocks[id];
                                            if (id < sink_blocks ||
                                                block.orig_pos_start >= boundary ||
                                                block.n_tokens != bt) {
                                                continue;
                                            }
                                            candidate[id] = 1;
                                        }
                                    }
                                }
                            }
                            std::vector<uint32_t> neighbor_candidates;
                            for (uint32_t id = 0; id < candidate.size(); ++id) {
                                if (candidate[id]) {
                                    neighbor_candidates.push_back(id);
                                }
                            }
                            // Allocate the limited continuity budget globally.
                            // Filling a one-block hole between two core blocks
                            // joins two fragments and is more useful than fully
                            // expanding only the very highest-ranked seeds. For
                            // ties, follow the best nearby seed's retrieval rank.
                            auto adjacent_core_count = [&](uint32_t id) {
                                uint32_t count = 0;
                                if (id > 0 && chosen[id - 1]) ++count;
                                if (id + 1 < chosen.size() && chosen[id + 1]) {
                                    ++count;
                                }
                                return count;
                            };
                            auto best_nearby_core_rank = [&](uint32_t id) {
                                uint32_t best = UINT32_MAX;
                                const uint32_t begin =
                                    id > neighbor_radius ? id - neighbor_radius
                                                         : 0;
                                const uint64_t end64 =
                                    static_cast<uint64_t>(id) +
                                    neighbor_radius + 1;
                                const uint32_t end = static_cast<uint32_t>(
                                    std::min<uint64_t>(blocks.size(), end64));
                                for (uint32_t nearby = begin;
                                     nearby < end; ++nearby) {
                                    best = std::min(best, core_rank[nearby]);
                                }
                                return best;
                            };
                            std::sort(
                                neighbor_candidates.begin(),
                                neighbor_candidates.end(),
                                [&](uint32_t a, uint32_t b) {
                                    const uint32_t a_adj =
                                        adjacent_core_count(a);
                                    const uint32_t b_adj =
                                        adjacent_core_count(b);
                                    if (a_adj != b_adj) return a_adj > b_adj;
                                    const uint32_t a_rank =
                                        best_nearby_core_rank(a);
                                    const uint32_t b_rank =
                                        best_nearby_core_rank(b);
                                    if (a_rank != b_rank) return a_rank < b_rank;
                                    const double as =
                                        blocks[a].retrieval_score;
                                    const double bs =
                                        blocks[b].retrieval_score;
                                    if (as != bs) return as > bs;
                                    return a > b;
                                });
                            for (const uint32_t id : neighbor_candidates) {
                                if (chosen_blocks.size() == refresh_block_cap) {
                                    break;
                                }
                                chosen[id] = 1;
                                chosen_blocks.push_back(id);
                                ++neighbor_added;
                            }
                            // A tiny context or radius=0 can leave capacity.
                            // Fill it by the original ranking without changing
                            // the guaranteed core or the neighbors already kept.
                            for (uint32_t id : ranked_blocks) {
                                if (chosen_blocks.size() == refresh_block_cap)
                                    break;
                                if (chosen[id]) continue;
                                chosen[id] = 1;
                                chosen_blocks.push_back(id);
                            }
                        }
                        ranked_blocks = std::move(chosen_blocks);
                        std::sort(ranked_blocks.begin(), ranked_blocks.end());
                        refresh_tokens.reserve(
                            static_cast<size_t>(ranked_blocks.size()) * bt);
                        for (uint32_t id : ranked_blocks) {
                            const KvMemBlock &block = blocks[id];
                            const uint32_t rb = block.orig_pos_start;
                            const uint32_t re = block.orig_pos_end();
                            if (!refresh_ranges.empty() &&
                                refresh_ranges.back().second == rb) {
                                refresh_ranges.back().second = re;
                            } else {
                                refresh_ranges.emplace_back(rb, re);
                            }
                            refresh_tokens.insert(
                                refresh_tokens.end(),
                                prompt_tokens.begin() + rb,
                                prompt_tokens.begin() + re);
                        }
                        refresh_session_count = static_cast<uint32_t>(
                            refresh_ranges.size());
                        if (std::getenv("QW3_KVMEM_TRACE")) {
                            log("native kvmem refresh-blocks blocks=" +
                                std::to_string(ranked_blocks.size()) +
                                " core_blocks=" +
                                std::to_string(kept_core) +
                                " expanded_sessions=" +
                                std::to_string(session_added) +
                                " session_blocks=" +
                                std::to_string(session_blocks_added) +
                                " neighbor_blocks=" +
                                std::to_string(neighbor_added) +
                                " runs=" +
                                std::to_string(refresh_ranges.size()) +
                                " tokens=" +
                                std::to_string(refresh_tokens.size()) +
                                " source=mean-k");
                        }
                    } else {
                    struct RefreshSession {
                        uint32_t begin = 0;
                        uint32_t end = 0;
                        double score = 0.0;
                        double bm25_score = 0.0;
                        double mean_k_score = 0.0;
                    };
                    std::unordered_map<uint32_t, uint32_t> query_tf;
                    for (uint32_t pos = final_event.begin;
                         pos < final_event.end; ++pos) {
                        const uint32_t tok = prompt_tokens[pos];
                        // Qwen chat/control tokens are framing, not retrieval
                        // terms. Ordinary tokens occupy the lower ID range.
                        if (tok < 248000) ++query_tf[tok];
                    }
                    std::vector<RefreshSession> candidates;
                    candidates.reserve(
                        options.kvmem_replay_session_starts.size());
                    for (size_t si = 0;
                         si < options.kvmem_replay_session_starts.size(); ++si) {
                        const uint32_t sb =
                            options.kvmem_replay_session_starts[si];
                        if (sb >= final_event.begin) break;
                        const uint32_t se = std::min<uint32_t>(
                            (si + 1 <
                                     options.kvmem_replay_session_starts.size())
                                ? options.kvmem_replay_session_starts[si + 1]
                                : final_event.begin,
                            final_event.begin);
                        if (se > sb) {
                            candidates.push_back({sb, se, 0.0, 0.0, 0.0});
                        }
                    }
                    std::unordered_map<uint32_t, uint32_t> session_df;
                    for (const RefreshSession &s : candidates) {
                        std::unordered_map<uint32_t, bool> seen;
                        for (uint32_t pos = s.begin; pos < s.end; ++pos) {
                            const uint32_t tok = prompt_tokens[pos];
                            if (query_tf.find(tok) != query_tf.end()) {
                                seen[tok] = true;
                            }
                        }
                        for (const auto &kv : seen) ++session_df[kv.first];
                    }
                    double avg_len = 0.0;
                    for (const RefreshSession &s : candidates) {
                        avg_len += static_cast<double>(s.end - s.begin);
                    }
                    if (!candidates.empty()) avg_len /= candidates.size();
                    constexpr double k1 = 1.2;
                    constexpr double b = 0.75;
                    const double ndoc = static_cast<double>(candidates.size());
                    for (RefreshSession &s : candidates) {
                        std::unordered_map<uint32_t, uint32_t> tf;
                        for (uint32_t pos = s.begin; pos < s.end; ++pos) {
                            const uint32_t tok = prompt_tokens[pos];
                            if (query_tf.find(tok) != query_tf.end()) ++tf[tok];
                        }
                        const double length_norm =
                            avg_len > 0.0
                                ? static_cast<double>(s.end - s.begin) / avg_len
                                : 1.0;
                        for (const auto &qkv : query_tf) {
                            const auto fit = tf.find(qkv.first);
                            const auto dit = session_df.find(qkv.first);
                            if (fit == tf.end() || dit == session_df.end()) continue;
                            const uint32_t df = dit->second;
                            // Remove generic instruction/date/chat tokens that
                            // occur in more than 20% of historical sessions.
                            if (df * 5 > candidates.size()) continue;
                            const double idf = std::log(
                                (ndoc - static_cast<double>(df) + 0.5) /
                                    (static_cast<double>(df) + 0.5) +
                                1.0);
                            const double freq = static_cast<double>(fit->second);
                            const double tf_norm =
                                freq * (k1 + 1.0) /
                                (freq + k1 * (1.0 - b + b * length_norm));
                            s.bm25_score += idf * tf_norm *
                                            std::min<uint32_t>(qkv.second, 2u);
                        }
                    }
                    // Keep the original BM25 refresh selector by default.  The
                    // opt-in mean-K variant projects the model's existing block
                    // retrieval distribution onto source sessions instead of
                    // introducing a second, potentially divergent retriever.
                    // Summed softmax mass rewards several independently useful
                    // blocks in the same session.
                    const bool refresh_use_mean_k = env_flag_enabled(
                        "QW3_KVMEM_TRANSCRIPT_REFRESH_MEANK");
                    if (refresh_use_mean_k) {
                        for (RefreshSession &s : candidates) {
                            for (const KvMemBlock &block : blocks) {
                                if (block.orig_pos_start >= s.end) break;
                                if (block.orig_pos_end() <= s.begin) continue;
                                s.mean_k_score += block.retrieval_score;
                            }
                        }
                    }
                    for (RefreshSession &s : candidates) {
                        s.score = refresh_use_mean_k
                                      ? s.mean_k_score
                                      : s.bm25_score;
                    }
                    std::sort(candidates.begin(), candidates.end(),
                              [](const RefreshSession &a,
                                 const RefreshSession &b) {
                                  if (a.score != b.score)
                                      return a.score > b.score;
                                  return a.begin > b.begin;
                              });
                    std::vector<RefreshSession> chosen;
                    uint32_t used = 0;
                    for (const RefreshSession &s : candidates) {
                        if (s.score <= 0.0) break;
                        const uint32_t n = s.end - s.begin;
                        if (n > refresh_budget -
                                    std::min(refresh_budget, used)) {
                            continue;
                        }
                        chosen.push_back(s);
                        used += n;
                        if (used >= refresh_budget) break;
                    }
                    // The working-set swap below must occur between complete
                    // source blocks. Otherwise the last refreshed source block
                    // would be partly baked in the sink-only frame and partly
                    // appended after the final cached+fresh assembly. Trim at
                    // most bt-1 tail tokens from the weakest sufficiently long
                    // chosen session to preserve that one-frame-per-block
                    // invariant without exceeding the refresh budget.
                    const uint32_t align_trim = used % bt;
                    if (align_trim > 0) {
                        auto trim_it = chosen.end();
                        for (auto it = chosen.begin(); it != chosen.end(); ++it) {
                            if (it->end - it->begin <= align_trim) continue;
                            if (trim_it == chosen.end() ||
                                it->score < trim_it->score) {
                                trim_it = it;
                            }
                        }
                        if (trim_it == chosen.end()) {
                            throw std::runtime_error(
                                "KVMem refresh cannot align selected sessions "
                                "to the block size");
                        }
                        trim_it->end -= align_trim;
                        used -= align_trim;
                        if (std::getenv("QW3_KVMEM_TRACE")) {
                            log("native kvmem refresh-align trimmed_tokens=" +
                                std::to_string(align_trim) +
                                " session_begin=" +
                                std::to_string(trim_it->begin) +
                                " aligned_total=" + std::to_string(used));
                        }
                    }
                    const bool relevance_order = env_flag_enabled(
                        "QW3_KVMEM_TRANSCRIPT_REFRESH_RELEVANCE_ORDER");
                    if (relevance_order) {
                        // Recurrent layers preferentially retain later edits.
                        // Put the strongest evidence closest to the final query.
                        std::sort(chosen.begin(), chosen.end(),
                                  [](const RefreshSession &a,
                                     const RefreshSession &b) {
                                      if (a.score != b.score)
                                          return a.score < b.score;
                                      return a.begin < b.begin;
                                  });
                    } else {
                        std::sort(chosen.begin(), chosen.end(),
                                  [](const RefreshSession &a,
                                     const RefreshSession &b) {
                                      return a.begin < b.begin;
                                  });
                    }
                    refresh_tokens.reserve(used);
                    refresh_ranges.reserve(chosen.size());
                    for (const RefreshSession &s : chosen) {
                        refresh_ranges.emplace_back(s.begin, s.end);
                        refresh_tokens.insert(
                            refresh_tokens.end(),
                            prompt_tokens.begin() + s.begin,
                            prompt_tokens.begin() + s.end);
                        if (std::getenv("QW3_KVMEM_TRACE")) {
                            std::ostringstream rmsg;
                            rmsg << "native kvmem refresh-session begin="
                                 << s.begin << " end=" << s.end
                                 << " tokens=" << (s.end - s.begin)
                                 << " score=" << s.score
                                 << " bm25=" << s.bm25_score
                                 << " mean_k=" << s.mean_k_score
                                 << " source="
                                 << (refresh_use_mean_k ? "mean-k" : "bm25")
                                 << " order="
                                 << (relevance_order ? "relevance" : "source");
                            log(rmsg.str());
                        }
                    }
                    refresh_session_count =
                        static_cast<uint32_t>(chosen.size());
                    }
                }

                if (!refresh_tokens.empty()) {
                    if (refresh_tokens.size() % bt != 0) {
                        throw std::runtime_error(
                            "KVMem refresh replay must end on a block boundary");
                    }
                    const uint32_t budget_blocks =
                        executor_->block_store()->budget_blocks();
                    const uint32_t refresh_blocks =
                        (static_cast<uint32_t>(refresh_tokens.size()) + bt - 1) /
                        bt;
                    const uint32_t query_reserve = 4;
                    uint32_t cached_cap =
                        budget_blocks > refresh_blocks + query_reserve
                            ? budget_blocks - refresh_blocks - query_reserve
                            : 0;
                    // Fresh block replay builds one internally consistent
                    // hybrid-model context. Mixing another large set of old,
                    // locally constructed K/V back into that view reintroduces
                    // the representation conflict the replay removes. Default
                    // block refresh to the stable sink only; experiments may
                    // explicitly opt into more cached evidence.
                    const uint32_t default_cached_tokens =
                        executor_->block_store()->config().sink_blocks * bt;
                    const uint32_t cached_token_cap = env_uint32_or(
                        "QW3_KVMEM_TRANSCRIPT_REFRESH_CACHED_TOKENS",
                        default_cached_tokens);
                    if (cached_token_cap > 0) {
                        cached_cap = std::min<uint32_t>(
                            cached_cap,
                            std::max<uint32_t>(
                                1, (cached_token_cap + bt - 1) / bt));
                    }
                    if (cached_cap == 0) {
                        throw std::runtime_error(
                            "KVMem refresh budget leaves no cached context");
                    }
                    // A refreshed source session must appear exactly once in
                    // the final attention window. Keeping its old cached blocks
                    // as well would duplicate the text with two incompatible
                    // hidden-state constructions (old local KV plus fresh KV),
                    // which can make the query attend to conflicting values.
                    const size_t selected_before_dedup =
                        selected_context.size();
                    selected_context.erase(
                        std::remove_if(
                            selected_context.begin(), selected_context.end(),
                            [&](uint32_t id) {
                                const KvMemBlock &block = blocks[id];
                                for (const auto &range : refresh_ranges) {
                                    if (block.orig_pos_start < range.second &&
                                        block.orig_pos_end() > range.first) {
                                        return true;
                                    }
                                }
                                return false;
                            }),
                        selected_context.end());
                    if (std::getenv("QW3_KVMEM_TRACE")) {
                        log("native kvmem refresh-dedup cached_removed=" +
                            std::to_string(selected_before_dedup -
                                           selected_context.size()));
                    }
                    std::sort(
                        selected_context.begin(), selected_context.end(),
                        [&](uint32_t a, uint32_t b) {
                            const bool a_sink = a < executor_->block_store()
                                                         ->config().sink_blocks;
                            const bool b_sink = b < executor_->block_store()
                                                         ->config().sink_blocks;
                            if (a_sink != b_sink) return a_sink;
                            const double as = blocks[a].retrieval_score;
                            const double bs = blocks[b].retrieval_score;
                            if (as != bs) return as > bs;
                            return a > b;
                        });
                    if (selected_context.size() > cached_cap) {
                        selected_context.resize(cached_cap);
                    }
                    std::sort(selected_context.begin(),
                              selected_context.end());
                }
                std::vector<uint32_t> refresh_seed_context;
                if (!refresh_tokens.empty()) {
                    const uint32_t sink_blocks = std::min<uint32_t>(
                        executor_->block_store()->config().sink_blocks,
                        executor_->block_store()->block_count());
                    refresh_seed_context.reserve(sink_blocks);
                    for (uint32_t id = 0; id < sink_blocks; ++id) {
                        if (executor_->block_store()->blocks()[id]
                                .orig_pos_start < boundary) {
                            refresh_seed_context.push_back(id);
                        }
                    }
                }
                // Build refreshed evidence against only the stable sink.  It
                // must not attend the old sparse working set: doing so would
                // bake the very stale/heterogeneous context that refresh is
                // intended to repair into its new K/V and recurrent state.
                executor_->kvmem_begin_query_replay(
                    final_boundary,
                    refresh_tokens.empty() ? selected_context
                                           : refresh_seed_context,
                    /*reset_recurrent_state=*/false);
                kvmem_registered_pos = boundary;
                uint32_t shifted_query_begin = final_event.begin;
                uint32_t shifted_query_end = final_event.end;
                if (!refresh_tokens.empty()) {
                    shifted_query_begin +=
                        static_cast<uint32_t>(refresh_tokens.size());
                    shifted_query_end +=
                        static_cast<uint32_t>(refresh_tokens.size());
                    executor_->kvmem_set_query_span(
                        shifted_query_begin, shifted_query_end,
                        static_cast<uint32_t>(prompt_tokens.size() +
                                              refresh_tokens.size()),
                        /*preserve_content_index=*/true);
                    for (size_t off = 0; off < refresh_tokens.size();) {
                        const uint32_t remaining = static_cast<uint32_t>(
                            refresh_tokens.size() - off);
                        const uint32_t width = std::max<uint32_t>(
                            1, executor_->effective_prefill_chunk_size(remaining));
                        const size_t end = off +
                            std::min<size_t>(remaining, width);
                        std::vector<uint32_t> chunk(
                            refresh_tokens.begin() +
                                static_cast<std::ptrdiff_t>(off),
                            refresh_tokens.begin() +
                                static_cast<std::ptrdiff_t>(end));
                        const uint32_t chunk_base = executor_->position();
                        if (mtp_local_positions) {
                            executor_->kvmem_set_defer_prefill_pressure(true);
                        }
                        try {
                            step = executor_->forward_n_tokens(
                                chunk, /*need_logits=*/false);
                        } catch (...) {
                            if (mtp_local_positions) {
                                executor_->kvmem_set_defer_prefill_pressure(
                                    false);
                            }
                            throw;
                        }
                        if (!step.ok) {
                            if (mtp_local_positions) {
                                executor_->kvmem_set_defer_prefill_pressure(
                                    false);
                            }
                            throw std::runtime_error(
                                "KVMem selected-session refresh prefill failed");
                        }
                        prime_mtp_prefix(chunk, chunk_base);
                        prefill_ops += step.ops_executed;
                        off = end;
                    }
                    // Publish the newly-built refresh blocks, then assemble the
                    // final attention view from old cached evidence plus those
                    // fresh blocks. kvmem_set_selection changes only the normal-
                    // attention page view and deliberately preserves the fresh
                    // DeltaNet/conv state constructed above.
                    executor_->kvmem_register_append(
                        static_cast<uint32_t>(refresh_tokens.size()));
                    kvmem_registered_pos = executor_->position();
                    std::vector<uint32_t> combined_context = selected_context;
                    const uint32_t refresh_end =
                        boundary + static_cast<uint32_t>(refresh_tokens.size());
                    const auto &refreshed_blocks =
                        executor_->block_store()->blocks();
                    uint32_t fresh_block_count = 0;
                    for (const KvMemBlock &block : refreshed_blocks) {
                        if (block.orig_pos_start < boundary) continue;
                        if (block.orig_pos_start >= refresh_end) break;
                        combined_context.push_back(block.block_id);
                        ++fresh_block_count;
                    }
                    std::sort(combined_context.begin(),
                              combined_context.end());
                    combined_context.erase(
                        std::unique(combined_context.begin(),
                                    combined_context.end()),
                        combined_context.end());
                    if (combined_context.size() >
                        executor_->block_store()->budget_blocks()) {
                        throw std::runtime_error(
                            "KVMem cached+fresh refresh exceeds select budget");
                    }
                    executor_->kvmem_set_selection(combined_context);
                    if (std::getenv("QW3_KVMEM_TRACE")) {
                        log("native kvmem refresh-assemble cached_blocks=" +
                            std::to_string(selected_context.size()) +
                            " fresh_blocks=" +
                            std::to_string(fresh_block_count) +
                            " final_blocks=" +
                            std::to_string(combined_context.size()));
                    }
                }
                // The session-local branch normally carries the recurrent
                // state produced by the compact refresh into the final query.
                // Honor the existing diagnostic isolation flag here as well:
                // retain the freshly rebuilt normal-attention K/V window, but
                // remove order-sensitive DeltaNet/conv carry from the fragment
                // replay before evaluating the exact question.
                if (transcript_reset_final_recurrent) {
                    executor_->kvmem_reset_recurrent_state();
                }
                do_prefill_range(boundary, final_event.end);
                executor_->kvmem_end_query_replay();
                executor_->kvmem_set_pin_from_block(
                    ((boundary + static_cast<uint32_t>(refresh_tokens.size())) /
                     bt));
                kvmem_registered_pos = executor_->position();
                cursor = final_event.end;

                std::ostringstream emsg;
                emsg << "native kvmem session-local final query=["
                     << final_event.begin << "," << final_event.end << ")"
                     << " boundary=" << boundary
                     << " sessions=" << session_boundaries.size()
                     << " selected_context_blocks="
                     << selected_context.size()
                     << " refresh_sessions=" << refresh_session_count
                     << " refresh_tokens=" << refresh_tokens.size()
                     << " shifted_query=[" << shifted_query_begin << ","
                     << shifted_query_end << ")";
                log(emsg.str());
            } else {
            QwenExecutor::StateSnapshot replay_boundary;
            bool replay_boundary_prepared = false;
            for (size_t ei = 0;
                 ei < options.kvmem_replay_query_spans.size(); ++ei) {
                const auto &event = options.kvmem_replay_query_spans[ei];
                const uint32_t boundary = (event.begin / bt) * bt;
                if (boundary == 0) {
                    throw std::runtime_error(
                        "KVMem transcript replay requires a non-zero query "
                        "boundary");
                }

                // Everything between events (including the recorded assistant
                // response) is ordinary teacher-forced prefill under the window
                // selected by the previous event.
                if (replay_boundary_prepared) {
                    if (!replay_boundary.ready ||
                        replay_boundary.position != boundary || cursor < boundary) {
                        throw std::runtime_error(
                            "KVMem transcript replay prepared the wrong shared-"
                            "block boundary checkpoint");
                    }
                } else {
                    if (boundary < cursor) {
                        throw std::runtime_error(
                            "KVMem transcript replay missed an overlapping "
                            "query-block checkpoint");
                    }
                    do_prefill_range(cursor, boundary);
                    executor_->capture_state(replay_boundary);
                }
                if (transcript_query_bootstrap || transcript_stable_ingest) {
                    // Make all teacher-forced bytes before the query visible to
                    // the block store before assembling the bootstrap. A prepared
                    // shared-block checkpoint may leave registered_pos past the
                    // aligned boundary; those bytes are real preceding transcript
                    // tokens and already belong in the recent tail.
                    if (kvmem_registered_pos < boundary) {
                        executor_->kvmem_register_append(
                            boundary - kvmem_registered_pos);
                        kvmem_registered_pos = boundary;
                    }
                    // If this query shares a block with the preceding event,
                    // replay_boundary was captured while that whole block was
                    // being rebuilt in the preceding stable pressure window.
                    // Keep that exact window for the short first-pass suffix:
                    // changing windows before completing the physical block
                    // would bake its old/new rows in two different RoPE frames.
                    // The semantic replay below rolls back to the aligned block
                    // boundary and rewrites the whole block, after which the
                    // stable replay may safely install the new pressure window.
                    if (!(transcript_stable_ingest &&
                          replay_boundary_prepared)) {
                        executor_->kvmem_reselect_prefill_pressure();
                    } else if (std::getenv("QW3_KVMEM_TRACE")) {
                        log("native kvmem transcript replay: retain stable "
                            "window across shared partial block boundary=" +
                            std::to_string(boundary));
                    }
                }
                if (transcript_stable_fast &&
                    transcript_reset_each_recurrent) {
                    // The query has not been prefetched yet: start its durable
                    // local segment now, while keeping the pressure-selected
                    // normal-attention window assembled above.
                    executor_->kvmem_reset_recurrent_state();
                }
                // With a prepared checkpoint, [boundary,cursor) is the tail of
                // the previous query/response sharing this 32-token block. It
                // has already been prefilled once under the current window; the
                // first pass for this event therefore begins at cursor. The
                // later block-aligned replay intentionally recomputes that tiny
                // overlap together with the new query.
                do_prefill_range(std::max<size_t>(cursor, boundary), event.end);

                executor_->kvmem_register_append(
                    event.end - kvmem_registered_pos);
                kvmem_registered_pos = event.end;
                if (!executor_->kvmem_publish_captured_prefix(boundary)) {
                    throw std::runtime_error(
                        "KVMem transcript replay could not publish a complete "
                        "mean-K prefix/query index at a user boundary");
                }

                bool next_boundary_inside_event = false;
                if (ei + 1 < options.kvmem_replay_query_spans.size()) {
                    const uint32_t nb =
                        (options.kvmem_replay_query_spans[ei + 1].begin / bt) * bt;
                    next_boundary_inside_event = nb <= event.end;
                }
                if (transcript_stable_fast &&
                    ei + 1 < options.kvmem_replay_query_spans.size() &&
                    !next_boundary_inside_event) {
                    // The first pass above was already computed in the stable
                    // pressure window selected at this event's boundary. Keep it
                    // as the durable source suffix and let the next iteration
                    // teacher-force the recorded response in the same window.
                    // No semantic selection is observable before that suffix is
                    // replaced, so skipping it is state/KV-equivalent.
                    kvmem_registered_pos = event.end;
                    cursor = event.end;
                    replay_boundary_prepared = false;

                    std::ostringstream emsg;
                    emsg << "native kvmem transcript replay event=" << (ei + 1)
                         << "/" << options.kvmem_replay_query_spans.size()
                         << " query=[" << event.begin << "," << event.end << ")"
                         << " boundary=" << boundary
                         << " selected_context_blocks=0"
                         << " query_bootstrap=sink+recent"
                         << " stable_ingest=1 stable_fast=1"
                         << " next=teacher_forced_prefill";
                    log(emsg.str());

                    const auto &next = options.kvmem_replay_query_spans[ei + 1];
                    executor_->kvmem_set_query_span(
                        next.begin, next.end,
                        static_cast<uint32_t>(prompt_tokens.size()),
                        /*index_tokens=*/0,
                        /*preserve_content_index=*/true);
                    continue;
                }
                executor_->kvmem_set_pin_from_block(boundary / bt);
                executor_->kvmem_set_trace_reselect_event(
                    static_cast<uint32_t>(ei),
                    static_cast<uint32_t>(options.kvmem_replay_query_spans.size()));
                executor_->kvmem_reselect();

                std::vector<uint32_t> selected_context;
                bool suffix_fully_selected = true;
                const auto &blocks = executor_->block_store()->blocks();
                for (const KvMemBlock &b : blocks) {
                    if (b.orig_pos_start < boundary) {
                        if (b.in_working_set) {
                            selected_context.push_back(b.block_id);
                        }
                    } else if (!b.in_working_set) {
                        suffix_fully_selected = false;
                    }
                }
                if (!suffix_fully_selected) {
                    throw std::runtime_error(
                        "KVMem transcript replay selection dropped part of "
                        "the query suffix");
                }

                executor_->kvmem_begin_query_replay(
                    replay_boundary, selected_context,
                    (transcript_reset_final_recurrent ||
                     transcript_reset_each_recurrent) &&
                        ei + 1 == options.kvmem_replay_query_spans.size(),
                    /*preserve_selected_context=*/true);
                // If the next user query begins in the same 32-token block,
                // save its boundary state WHILE replaying this event under the
                // newly selected context. This avoids dropping the first query
                // tokens or retaining a checkpoint from the previous window.
                bool prepared_next_boundary = false;
                uint32_t next_boundary = 0;
                QwenExecutor::StateSnapshot next_boundary_snapshot;
                if (ei + 1 < options.kvmem_replay_query_spans.size()) {
                    next_boundary =
                        (options.kvmem_replay_query_spans[ei + 1].begin / bt) * bt;
                    if (next_boundary >= boundary && next_boundary <= event.end) {
                        do_prefill_range(boundary, next_boundary);
                        executor_->capture_state(next_boundary_snapshot);
                        do_prefill_range(next_boundary, event.end);
                        prepared_next_boundary = true;
                    }
                }
                if (!prepared_next_boundary) {
                    do_prefill_range(boundary, event.end);
                }
                executor_->kvmem_end_query_replay();

                if (transcript_stable_ingest &&
                    ei + 1 < options.kvmem_replay_query_spans.size()) {
                    // The semantic replay above is the answer-time state. For
                    // durable memory construction, discard only this short query
                    // suffix, restore its aligned boundary, and rebuild it under
                    // deterministic pressure context. The recorded response that
                    // follows at the top of the next iteration is consequently
                    // constructed under the same stable window. Historical KV is
                    // reused unchanged; work is O(query tokens), not O(224K).
                    std::vector<uint32_t> pressure_context =
                        executor_->block_store()->pick_prefill_pressure_blocks();
                    pressure_context.erase(
                        std::remove_if(
                            pressure_context.begin(), pressure_context.end(),
                            [&](uint32_t id) {
                                const auto &bs = executor_->block_store()->blocks();
                                return id >= bs.size() ||
                                    bs[id].orig_pos_start >= boundary;
                            }),
                        pressure_context.end());
                    const uint32_t context_block_count = boundary / bt;
                    const uint32_t pressure_target = std::min<uint32_t>(
                        executor_->block_store()->budget_blocks(),
                        context_block_count);
                    std::vector<bool> pressure_kept(context_block_count, false);
                    for (uint32_t id : pressure_context) {
                        if (id < context_block_count) pressure_kept[id] = true;
                    }
                    for (uint32_t id = context_block_count;
                         id-- > 0 && pressure_context.size() < pressure_target;) {
                        if (!pressure_kept[id]) {
                            pressure_kept[id] = true;
                            pressure_context.push_back(id);
                        }
                    }
                    std::sort(pressure_context.begin(), pressure_context.end());
                    executor_->kvmem_begin_query_replay(replay_boundary,
                                                         pressure_context,
                                                         transcript_reset_each_recurrent);
                    if (prepared_next_boundary) {
                        do_prefill_range(boundary, next_boundary);
                        executor_->capture_state(next_boundary_snapshot);
                        do_prefill_range(next_boundary, event.end);
                    } else {
                        do_prefill_range(boundary, event.end);
                    }
                    executor_->kvmem_end_query_replay();
                }
                if (prepared_next_boundary) {
                    replay_boundary = std::move(next_boundary_snapshot);
                }
                kvmem_registered_pos = event.end;
                cursor = event.end;
                replay_boundary_prepared = prepared_next_boundary;

                std::ostringstream emsg;
                emsg << "native kvmem transcript replay event=" << (ei + 1)
                     << "/" << options.kvmem_replay_query_spans.size()
                     << " query=[" << event.begin << "," << event.end << ")"
                     << " boundary=" << boundary
                     << " selected_context_blocks="
                     << selected_context.size()
                     << " query_bootstrap="
                     << ((transcript_query_bootstrap || transcript_stable_ingest)
                             ? "sink+recent" : "previous")
                     << " stable_ingest="
                     << (transcript_stable_ingest ? 1 : 0)
                     << " stable_fast="
                     << (transcript_stable_fast ? 1 : 0)
                     << " next=teacher_forced_prefill";
                log(emsg.str());

                if (ei + 1 < options.kvmem_replay_query_spans.size()) {
                    const auto &next = options.kvmem_replay_query_spans[ei + 1];
                    executor_->kvmem_set_query_span(
                        next.begin, next.end,
                        static_cast<uint32_t>(prompt_tokens.size()),
                        /*index_tokens=*/0,
                        /*preserve_content_index=*/true);
                }
            }
            }
            // Closing user framing + the final assistant/thinking header are
            // prefilled once against the final selected context. Decode starts
            // only after this point.
            do_prefill_range(cursor, prefill_tokens.size());
        } else if (clean_query) {
            // PASS B (task #50). Split the prefill at the question boundary qb.
            const uint32_t bt = executor_->block_store()
                                    ? executor_->block_store()->config().block_tokens
                                    : 256;
            const uint32_t qb = options.kvmem_query_begin;
            // [0, qb): context-only prefill. The query span is [qb,qe), so this
            // builds g_kbar_multi_ for every context block but captures NO query
            // rows (none lie in [qb,qe) yet) — g_query_multi_ready_ stays false, so
            // any automatic pool-fill offload here uses the deterministic
            // prefill-pressure sink+tail window (staging only; the
            // position-invariant content index is what final selection reads).
            do_prefill_range(0, qb, /*compute_final_logits=*/false);
            // Register context blocks, restore the clean query stashed in PASS A,
            // pin the question + generated tail (survives every reselect), then
            // reselect: the CLEAN query scores the context and assembles the decode
            // window (kvmem_active_ / window_query_pos_).
            executor_->kvmem_register_append(qb);
            executor_->kvmem_restore_clean_query();
            executor_->kvmem_set_pin_from_block(qb / bt);
            kvmem_final_query_reselect();
            // [qb, P): prefill the question into the active window (attends over the
            // selected context at bounded window positions). This recaptures a
            // recency-contaminated query into g_query_multi_ — overwritten by the
            // restore in the post-prefill block before any decode-time reselect.
            do_prefill_range(qb, prefill_tokens.size(),
                             /*compute_final_logits=*/true);
        } else if (recompute_query) {
            // Preserve the ordinary single-pass retrieval query: first prefill
            // through the block boundary immediately before/containing Q, save
            // the hybrid recurrent state there, then finish the prompt exactly
            // as usual. The post-prefill selection below still uses this first
            // pass's pressure-window query.
            if (semantic_chunk) {
                if (prefill_absolute_base != 0 || kvmem_prefill_begin != 0 ||
                    kvmem_do_boundary_capture) {
                    throw std::runtime_error(
                        "semantic-chunk MTP prefill requires a cold prompt "
                        "without a warm prefix checkpoint");
                }
                semantic_chunk_stats = kvmem_prefill_semantic_chunks(
                    executor_, logical_prompt_tokens,
                    /*begin=*/0, kvmem_query_replay_begin,
                    options.kvmem_prefill_semantic_start_tokens,
                    options.kvmem_prefill_semantic_query_tokens,
                    [&](uint32_t begin, uint32_t end, bool need_logits) {
                        do_prefill_range(begin, end, need_logits);
                    });
                // Switch back from the per-chunk retrieval spans to the final
                // user question while preserving the accumulated history index.
                executor_->kvmem_set_query_span(
                    options.kvmem_query_begin, options.kvmem_query_end,
                    logical_prompt_tokens,
                    /*index_tokens=*/kvmem_query_replay_begin,
                    /*preserve_content_index=*/true);
                executor_->capture_state(kvmem_query_replay_ckpt);
                executor_->kvmem_start_query_prefetch(
                    static_cast<uint32_t>(
                        prefill_tokens.size() - kvmem_query_replay_begin));
                do_prefill_range(kvmem_query_replay_begin,
                                 prefill_tokens.size());
            } else if (kvmem_query_replay_begin >= prefill_absolute_base) {
                const size_t replay_local = static_cast<size_t>(
                    kvmem_query_replay_begin - prefill_absolute_base);
                if (replay_local > prefill_tokens.size()) {
                    throw std::runtime_error(
                        "KVMem query replay boundary is outside the appended "
                        "prompt fragment");
                }
                if (kvmem_do_boundary_capture &&
                    kvmem_ckpt_split <= kvmem_query_replay_begin) {
                    const size_t checkpoint_local = static_cast<size_t>(
                        kvmem_ckpt_split - prefill_absolute_base);
                    do_prefill_range(0, checkpoint_local);
                    executor_->kvmem_register_append(
                        kvmem_ckpt_split - prefill_absolute_base);
                    executor_->kvmem_reselect_prefill_pressure();
                    kvmem_warm_valid_ = false;
                    executor_->capture_state(kvmem_warm_ckpt_prompt_);
                    kvmem_warm_prompt_pos = kvmem_ckpt_split;
                    kvmem_warm_prompt_resumable =
                        kvmem_prompt_checkpoint_resumable(
                            kvmem_warm_source_index_ready);
                    kvmem_warm_checkpoint_staged = true;
                    do_prefill_range(checkpoint_local, replay_local);
                } else {
                    do_prefill_range(0, replay_local);
                }
                executor_->capture_state(kvmem_query_replay_ckpt);
                executor_->kvmem_start_query_prefetch(
                    static_cast<uint32_t>(
                        prefill_tokens.size() - replay_local));
                do_prefill_range(replay_local,
                                 prefill_tokens.size());
            } else {
                // The aligned boundary is in the previous request's trailing
                // partial block. Its snapshot was restored from the API
                // session checkpoint above; first-pass only the new fragment.
                // That fragment is still useful overlap time for advisory
                // stage-in, even though no new replay snapshot is needed here.
                executor_->kvmem_start_query_prefetch(
                    static_cast<uint32_t>(prefill_tokens.size()));
                do_prefill_range(0, prefill_tokens.size());
            }
        } else {
            do_prefill_range(0, prefill_tokens.size(),
                             /*compute_final_logits=*/true);
        }
        if (mtp_local_positions) {
            executor_->kvmem_set_defer_prefill_pressure(false);
        }
        const double t_prefill_end = wall_seconds();
        if (semantic_chunk_stats.chunks != 0) {
            std::ostringstream smsg;
            smsg << std::fixed << std::setprecision(3)
                 << "native kvmem semantic-chunk (mtp): chunks="
                 << semantic_chunk_stats.chunks
                 << " provisional_tokens="
                 << semantic_chunk_stats.provisional_tokens
                 << " replay_tokens=" << semantic_chunk_stats.replay_tokens
                 << " provisional_ms="
                 << semantic_chunk_stats.provisional_s * 1e3
                 << " rollback_ms=" << semantic_chunk_stats.rollback_s * 1e3
                 << " reselect_ms=" << semantic_chunk_stats.reselect_s * 1e3
                 << " replay_ms=" << semantic_chunk_stats.replay_s * 1e3;
            log(smsg.str());
        }

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
        if (kvmem_on && (transcript_replay || query_replay_applied)) {
            // Every event already performed its semantic re-selection. Register
            // only the final chat-template suffix; an additional selection here
            // would move the boundary and violate the requested lifecycle.
            if (transcript_replay) {
                const uint32_t final_position = executor_->position();
                if (final_position > kvmem_registered_pos) {
                    executor_->kvmem_register_append(
                        final_position - kvmem_registered_pos);
                }
            }
            kvmem_registered_pos = executor_->position();
            kvmem_last_reselect_pos = executor_->position();
        } else if (kvmem_on && clean_query) {
            // PASS B tail: the context blocks were registered + selected mid-prefill
            // (the reselect at qb IS the decode window; the question is its live
            // tail). Register only the question blocks, then restore the clean query
            // one more time to overwrite the recency-contaminated capture from the
            // [qb,P) re-prefill, so decode-time interval reselects keep scoring with
            // the clean query. No extra reselect — the window is already assembled.
            const uint32_t qb = options.kvmem_query_begin;
            executor_->kvmem_register_append(
                static_cast<uint32_t>(prompt_tokens.size()) - qb);
            executor_->kvmem_restore_clean_query();
            kvmem_registered_pos = executor_->position();
            kvmem_last_reselect_pos = executor_->position();
        } else if (kvmem_on) {
            // Register the tokens not yet in the store so it lands at prompt.size().
            // Already registered before this point: reuse_m (warm restore) plus the
            // first segment when a block-boundary ckpt_P split ran.
            const uint32_t already =
                kvmem_warm_checkpoint_staged
                    ? kvmem_ckpt_split
                    : static_cast<uint32_t>(kvmem_prefill_begin);
            const uint32_t reg_n =
                static_cast<uint32_t>(prompt_tokens.size()) - already;
            executor_->kvmem_register_append(reg_n);
            if (options.kvmem_reselect_mode == KvMemReselectMode::Off) {
                // A persistent prefill-only ingest must not collapse from the
                // physical 224K+32K pool to the 224K selection budget merely
                // because an API fragment ended. The mid-prefill page-pressure
                // hook selects sink+recent when the pool is actually close to
                // full; a later pressure-mode append also rebuilds the window
                // before ingest. Keep one-shot behavior unchanged.
                if (!api_session &&
                    options.kvmem_prefill_window_mode !=
                    KvMemPrefillWindowMode::KeepSelected) {
                    executor_->kvmem_reselect_prefill_pressure();
                }
            } else {
                // Optional reversible decode probe runs immediately before the
                // final semantic selection; the complete replay suffix is still
                // processed afterward under the selected context.
                if (recompute_query) {
                    if (!executor_->kvmem_query_capture_complete()) {
                        throw std::runtime_error(
                            "KVMem query replay first pass captured " +
                            std::to_string(
                                executor_->kvmem_query_captured_tokens()) +
                            "/" +
                            std::to_string(
                                executor_->kvmem_query_expected_tokens()) +
                            " query tokens");
                    }
                    if (query_guided_query) {
                        const double t_probe_start = wall_seconds();
                        QwenExecutor::StateSnapshot probe_boundary;
                        try {
                            if (!executor_->kvmem_begin_guided_query_probe(
                                    options
                                        .kvmem_query_guided_query_max_tokens)) {
                                throw std::runtime_error(
                                    "KVMem guided-query capture is unavailable "
                                    "for the active backend/configuration");
                            }
                            executor_->capture_transient_state(probe_boundary);

                            // This private turn is never registered or exposed.
                            // It asks for one semantic retrieval query rather
                            // than a bag of keywords; all of it is rolled back
                            // after its Q rows have been copied aside.
                            std::string guided_query_prompt;
                            std::string direct_source_query;
                            if (options.thinking_open) {
                                guided_query_prompt = "</think>\n";
                            }
                            if (query_guided_direct) {
                                std::vector<int32_t> source_query_tokens;
                                source_query_tokens.reserve(
                                    options.kvmem_query_end -
                                    options.kvmem_query_begin);
                                for (uint32_t pos = options.kvmem_query_begin;
                                     pos < options.kvmem_query_end &&
                                     pos < prompt_tokens.size();
                                     ++pos) {
                                    source_query_tokens.push_back(
                                        static_cast<int32_t>(
                                            prompt_tokens[pos]));
                                }
                                direct_source_query =
                                    tokenizer_->decode(source_query_tokens);
                                // Keep untrusted query text from introducing a
                                // structural ChatML boundary into the private turn.
                                size_t special = 0;
                                while ((special = direct_source_query.find(
                                            "<|", special)) !=
                                       std::string::npos) {
                                    direct_source_query.replace(
                                        special, 2, "< |");
                                    special += 3;
                                }
                                guided_query_prompt +=
                                    "<|im_end|>\n<|im_start|>user\n"
                                    "RETRIEVAL QUERY REWRITE TASK\n\n"
                                    "Convert only the SOURCE REQUEST below into "
                                    "one memory-search request.\n\n"
                                    "SOURCE REQUEST BEGIN\n" +
                                    direct_source_query +
                                    "\nSOURCE REQUEST END\n\n"
                                    "Your output must state what evidence should be "
                                    "found in conversation history. Copy all "
                                    "important names, dates, round numbers, "
                                    "relationships, and constraints from the source. "
                                    "Do not solve, calculate, guess, or state the "
                                    "answer. Do not call a tool. Do not use quotation "
                                    "marks, tags, XML, JSON, labels, explanations, "
                                    "or keyword lists. Output exactly one grammatical "
                                    "question or command on one line, ending with "
                                    "normal sentence punctuation. Nothing else."
                                    "<|im_end|>\n<|im_start|>assistant\n"
                                    "<think>\n\n</think>\n\nRetrieval query: ";
                            } else {
                                guided_query_prompt +=
                                    "<|im_end|>\n<|im_start|>user\n"
                                    "Before answering the preceding real user "
                                    "request, prepare a retrieval query for long-"
                                    "term memory. Think briefly about exactly what "
                                    "evidence must be found. Do not answer the real "
                                    "request. After finishing that private "
                                    "reasoning, output exactly one compact, "
                                    "standalone retrieval query on a single line. "
                                    "Write it as a grammatical natural-language "
                                    "question or declarative sentence that preserves "
                                    "the complete information need. Never output a "
                                    "keyword list, search-engine terms, quoted "
                                    "fragment, label, or answer. Preserve important "
                                    "entities, relationships, dates, and constraints; "
                                    "remove examples, filler, and answer-format "
                                    "instructions."
                                    "<|im_end|>\n<|im_start|>assistant\n<think>\n";
                            }
                            const std::vector<int32_t> private_prompt =
                                tokenizer_->encode(guided_query_prompt);
                            NativeExecutorReport guided_step;
                            for (int32_t forced : private_prompt) {
                                if (forced < 0) continue;
                                executor_->kvmem_set_guided_query_capture(false);
                                guided_step = executor_->forward_one_token(
                                    static_cast<uint32_t>(forced));
                                if (!guided_step.ok) {
                                    throw std::runtime_error(
                                        "KVMem guided-query private prompt "
                                        "decode failed");
                                }
                            }
                            int32_t proposed = guided_step.argmax_token;
                            const int32_t close_think =
                                tokenizer_->token_id("</think>");
                            const int32_t im_end =
                                tokenizer_->token_id("<|im_end|>");
                            const int32_t eos = tokenizer_->eos_id();

                            if (query_guided_direct) {
                                query_guided_thinking_closed = true;
                            } else {
                                for (uint32_t i = 0;
                                     i < options
                                             .kvmem_query_guided_thinking_max_tokens &&
                                     proposed >= 0;
                                     ++i) {
                                    if (proposed == eos || proposed == im_end) break;
                                    executor_->kvmem_set_guided_query_capture(false);
                                    guided_step = executor_->forward_one_token(
                                        static_cast<uint32_t>(proposed));
                                    if (!guided_step.ok) break;
                                    if (proposed == close_think) {
                                        query_guided_thinking_closed = true;
                                        proposed = guided_step.argmax_token;
                                        break;
                                    }
                                    ++query_guided_thinking_tokens;
                                    proposed = guided_step.argmax_token;
                                }

                                // Natural close gives a content-dependent thinking
                                // length. The cap is only a safety bound; on hitting
                                // it, inject the close tag so no reasoning Q leaks
                                // into the retrieval query tensor.
                                if (!query_guided_thinking_closed) {
                                    if (close_think < 0) {
                                        throw std::runtime_error(
                                            "model tokenizer has no </think> token");
                                    }
                                    executor_->kvmem_set_guided_query_capture(false);
                                    guided_step = executor_->forward_one_token(
                                        static_cast<uint32_t>(close_think));
                                    if (!guided_step.ok) {
                                        throw std::runtime_error(
                                            "KVMem guided-query forced think close "
                                            "failed");
                                    }
                                    query_guided_thinking_closed = true;
                                    proposed = guided_step.argmax_token;
                                }
                            }

                            bool query_started = false;
                            uint32_t uncaptured_prefix_tokens = 0;
                            const uint32_t prefix_guard = 16;
                            while (proposed >= 0 &&
                                   query_guided_query_tokens <
                                       options
                                           .kvmem_query_guided_query_max_tokens) {
                                if (proposed == eos || proposed == im_end ||
                                    proposed == close_think) {
                                    break;
                                }
                                const std::string piece =
                                    tokenizer_->decode_one(proposed);
                                const bool whitespace_only =
                                    !piece.empty() && std::all_of(
                                        piece.begin(), piece.end(),
                                        [](unsigned char ch) {
                                            return std::isspace(ch) != 0;
                                        });
                                if (query_started && whitespace_only &&
                                    piece.find('\n') != std::string::npos) {
                                    break;
                                }
                                std::string trimmed_piece = piece;
                                while (!trimmed_piece.empty() && std::isspace(
                                           static_cast<unsigned char>(
                                               trimmed_piece.back()))) {
                                    trimmed_piece.pop_back();
                                }
                                size_t trimmed_first = 0;
                                while (trimmed_first < trimmed_piece.size() &&
                                       std::isspace(static_cast<unsigned char>(
                                           trimmed_piece[trimmed_first]))) {
                                    ++trimmed_first;
                                }
                                trimmed_piece.erase(0, trimmed_first);
                                const bool direct_leading_wrapper =
                                    query_guided_direct && !query_started &&
                                    (trimmed_piece == "\"" ||
                                     trimmed_piece == "'" ||
                                     trimmed_piece == "`" ||
                                     trimmed_piece == "“" ||
                                     trimmed_piece == "‘");
                                const bool content_bearing =
                                    !piece.empty() && !whitespace_only &&
                                    !direct_leading_wrapper;
                                // Ignore only leading separators. Once the
                                // query starts, retain every token in that
                                // single line, including internal whitespace,
                                // so its Q sequence represents the compact
                                // query rather than a keyword bag.
                                const bool capture =
                                    query_started || content_bearing;
                                if (!capture && !query_started &&
                                    ++uncaptured_prefix_tokens > prefix_guard) {
                                    break;
                                }
                                executor_->kvmem_set_guided_query_capture(
                                    capture);
                                guided_step = executor_->forward_one_token(
                                    static_cast<uint32_t>(proposed));
                                if (!guided_step.ok) break;
                                if (capture) {
                                    query_started = true;
                                    ++query_guided_query_tokens;
                                    query_guided_query_text += piece;
                                }
                                const bool line_complete =
                                    query_started &&
                                    piece.find('\n') != std::string::npos;
                                // A direct rewrite is contractually one question.
                                // Stop as soon as that question ends, before the
                                // model can emit a closing quote or explanation.
                                const bool question_complete =
                                    query_guided_direct && capture &&
                                    (piece.find('?') != std::string::npos ||
                                     piece.find("？") != std::string::npos);
                                proposed = guided_step.argmax_token;
                                if (line_complete || question_complete) break;
                            }
                            executor_->kvmem_set_guided_query_capture(false);
                            if (query_guided_direct) {
                                std::string normalized = query_guided_query_text;
                                while (!normalized.empty() && std::isspace(
                                           static_cast<unsigned char>(
                                               normalized.back()))) {
                                    normalized.pop_back();
                                }
                                size_t first = 0;
                                while (first < normalized.size() && std::isspace(
                                           static_cast<unsigned char>(
                                               normalized[first]))) {
                                    ++first;
                                }
                                normalized.erase(0, first);
                                std::string lower = normalized;
                                std::transform(
                                    lower.begin(), lower.end(), lower.begin(),
                                    [](unsigned char ch) {
                                        return static_cast<char>(std::tolower(ch));
                                    });
                                const bool question =
                                    normalized.find('?') != std::string::npos ||
                                    normalized.find("？") != std::string::npos;
                                const bool terminal_ascii =
                                    !normalized.empty() &&
                                    (normalized.back() == '.' ||
                                     normalized.back() == '!');
                                const bool terminal_cjk =
                                    normalized.size() >= 3 &&
                                    (normalized.compare(
                                         normalized.size() - 3, 3, "。") == 0 ||
                                     normalized.compare(
                                         normalized.size() - 3, 3, "！") == 0);
                                const bool single_line_request =
                                    !normalized.empty() &&
                                    normalized.find('\n') == std::string::npos &&
                                    normalized.find('\r') == std::string::npos &&
                                    (question || terminal_ascii || terminal_cjk);
                                const bool structured_answer =
                                    lower.find("<answer") != std::string::npos ||
                                    lower.find("tool_call") != std::string::npos ||
                                    lower.find("<tool") != std::string::npos ||
                                    lower.find("```") != std::string::npos ||
                                    (!normalized.empty() &&
                                     (normalized.front() == '{' ||
                                      normalized.front() == '['));
                                if (!single_line_request || structured_answer) {
                                    // The original query Q was already captured
                                    // before this reversible probe. Cancelling here
                                    // leaves it intact, so malformed rewrites can
                                    // safely fall back without a second query replay.
                                    query_guided_fallback_original = true;
                                    query_guided_fallback_reason = structured_answer
                                        ? "structured-output"
                                        : normalized.empty()
                                            ? "empty-output"
                                            : "incomplete-or-multiline-output";
                                    executor_->kvmem_cancel_guided_query_probe();
                                    executor_->restore_state(probe_boundary);
                                    query_guided_query_tokens =
                                        executor_->kvmem_query_captured_tokens();
                                    query_guided_query_text = direct_source_query;
                                    if (query_guided_query_tokens == 0) {
                                        throw std::runtime_error(
                                            "KVMem guided-query fallback found no "
                                            "original query Q rows");
                                    }
                                }
                            }
                            if (!query_guided_fallback_original) {
                                const uint32_t published =
                                    executor_->kvmem_end_guided_query_probe();
                                executor_->restore_state(probe_boundary);
                                if (published == 0 ||
                                    published != query_guided_query_tokens) {
                                    throw std::runtime_error(
                                        "KVMem guided query produced no usable "
                                        "compact-query Q rows");
                                }
                            }
                        } catch (...) {
                            executor_->kvmem_cancel_guided_query_probe();
                            if (probe_boundary.ready) {
                                executor_->restore_state(probe_boundary);
                            }
                            throw;
                        }
                        post_query_guided_query_s +=
                            wall_seconds() - t_probe_start;
                        std::ostringstream gmsg;
                        gmsg << "native kvmem guided query: mode="
                             << (query_guided_fallback_original
                                     ? "fallback-original"
                                     : query_guided_direct ? "direct"
                                                           : "reasoning")
                             << " thinking_tokens="
                             << query_guided_thinking_tokens
                             << " thinking_closed="
                             << (query_guided_thinking_closed ? 1 : 0)
                             << " query_tokens="
                             << query_guided_query_tokens
                             << " query="
                             << escape_text(query_guided_query_text);
                        if (query_guided_fallback_original) {
                            gmsg << " reason="
                                 << query_guided_fallback_reason;
                        }
                        log(gmsg.str());
                    }
                    const uint32_t query_tokens =
                        options.kvmem_query_end - options.kvmem_query_begin;
                    const uint32_t keep_query_tokens = std::min<uint32_t>(
                        query_tokens,
                        options.kvmem_query_attention_score_tokens);
                    if (query_attention_probe &&
                        keep_query_tokens < query_tokens) {
                        const double t_probe_start = wall_seconds();
                        std::vector<float> attention_mass;
                        QwenExecutor::StateSnapshot probe_boundary;
                        try {
                            if (executor_->kvmem_begin_query_attention_probe(
                                    options.kvmem_query_begin,
                                    options.kvmem_query_end)) {
                                executor_->capture_transient_state(
                                    probe_boundary);
                                int32_t probe_token = step.argmax_token;
                                for (uint32_t i = 0;
                                     i < options
                                             .kvmem_query_attention_probe_tokens &&
                                     probe_token >= 0;
                                     ++i) {
                                    NativeExecutorReport probe_step =
                                        executor_->forward_one_token(
                                            static_cast<uint32_t>(probe_token));
                                    if (!probe_step.ok) break;
                                    ++query_attention_probe_decoded;
                                    probe_token = probe_step.argmax_token;
                                }
                                attention_mass = executor_
                                    ->kvmem_end_query_attention_probe();
                                executor_->restore_state(probe_boundary);
                            }
                        } catch (...) {
                            executor_->kvmem_cancel_query_attention_probe();
                            if (probe_boundary.ready) {
                                executor_->restore_state(probe_boundary);
                            }
                            throw;
                        }

                        if (attention_mass.size() == query_tokens) {
                            double positive_mass = 0.0;
                            std::vector<uint32_t> ranked(query_tokens);
                            for (uint32_t i = 0; i < query_tokens; ++i) {
                                ranked[i] = i;
                                if (std::isfinite(attention_mass[i]) &&
                                    attention_mass[i] > 0.0f) {
                                    positive_mass += attention_mass[i];
                                } else {
                                    attention_mass[i] =
                                        -std::numeric_limits<float>::infinity();
                                }
                            }
                            if (positive_mass > 0.0) {
                                std::partial_sort(
                                    ranked.begin(),
                                    ranked.begin() + keep_query_tokens,
                                    ranked.end(),
                                    [&](uint32_t a, uint32_t b) {
                                        if (attention_mass[a] !=
                                            attention_mass[b]) {
                                            return attention_mass[a] >
                                                   attention_mass[b];
                                        }
                                        return a < b;
                                    });
                                ranked.resize(keep_query_tokens);
                                std::sort(ranked.begin(), ranked.end());
                                query_score_token_indices = std::move(ranked);
                                query_attention_probe_used = true;
                            }
                        }

                        // Unsupported attention dtype/kernel or an incomplete
                        // probe falls back to deterministic full-span coverage.
                        // The sparse score budget remains enforced, avoiding the
                        // long-query scorer allocation that motivated the probe.
                        if (query_score_token_indices.empty()) {
                            query_score_token_indices.reserve(
                                keep_query_tokens);
                            for (uint32_t i = 0; i < keep_query_tokens; ++i) {
                                query_score_token_indices.push_back(
                                    static_cast<uint32_t>(
                                        ((2ull * i + 1ull) * query_tokens) /
                                        (2ull * keep_query_tokens)));
                            }
                        }
                        executor_->kvmem_set_query_score_token_indices(
                            query_score_token_indices);
                        post_query_attention_probe_s +=
                            wall_seconds() - t_probe_start;

                        std::ostringstream pmsg;
                        pmsg << "native kvmem query attention probe: requested="
                             << options.kvmem_query_attention_probe_tokens
                             << " decoded=" << query_attention_probe_decoded
                             << " query_tokens=" << query_tokens
                             << " score_tokens="
                             << query_score_token_indices.size()
                             << " selector="
                             << (query_attention_probe_used
                                     ? "attention-topk"
                                     : "uniform-fallback")
                             << " indices=[";
                        for (size_t i = 0;
                             i < query_score_token_indices.size(); ++i) {
                            if (i) pmsg << ',';
                            pmsg << query_score_token_indices[i];
                        }
                        pmsg << ']';
                        log(pmsg.str());
                    }
                    const uint32_t bt = std::max<uint32_t>(
                        1, executor_->block_store()->config().block_tokens);
                    executor_->kvmem_set_pin_from_block(
                        kvmem_query_replay_begin / bt);
                }
                const double t_semantic_start = wall_seconds();
                // Below-budget preparation must not trigger semantic scoring;
                // its only purpose is to make the next warm crossing resumable.
                if (!qc_prepare || qc_select_active) {
                    kvmem_final_query_reselect();
                }
                post_semantic_reselect_s +=
                    wall_seconds() - t_semantic_start;
            }
            if (recompute_query &&
                options.kvmem_reselect_mode != KvMemReselectMode::Off) {
                const double t_query_replay_start = wall_seconds();
                // Freeze the exact semantic context selected above. Every block
                // in the short replay suffix must already be selected; otherwise
                // adding it during replay would change the retrieval result and
                // cease to be a controlled query-state experiment.
                std::vector<uint32_t> selected_context;
                bool suffix_fully_selected = true;
                const auto &blocks = executor_->block_store()->blocks();
                for (const KvMemBlock &b : blocks) {
                    if (b.orig_pos_start < kvmem_query_replay_begin) {
                        if (b.in_working_set) {
                            selected_context.push_back(b.block_id);
                        }
                    } else if (!b.in_working_set) {
                        suffix_fully_selected = false;
                    }
                }
                if (!suffix_fully_selected) {
                    throw std::runtime_error(
                        "KVMem query replay requires every replay-suffix block "
                        "to be present in the final selection");
                }
                if (inline_refresh) {
                    std::vector<uint32_t> refreshed_tokens =
                        kvmem_gather_selected_source_tokens(
                            prompt_tokens, blocks, selected_context);
                    const size_t selected_prefix_tokens =
                        refreshed_tokens.size();
                    refreshed_tokens.insert(
                        refreshed_tokens.end(),
                        prompt_tokens.begin() +
                            static_cast<std::ptrdiff_t>(
                                kvmem_query_replay_begin),
                        prompt_tokens.end());
                    if (refreshed_tokens.empty() ||
                        refreshed_tokens.size() >
                            executor_->kv_ctx_size()) {
                        throw std::runtime_error(
                            "KVMem inline refresh compact prompt is empty or "
                            "exceeds the executor context");
                    }

                    inline_refresh_guard =
                        std::make_unique<ScopedKvmemDisable>(executor_);
                    executor_->reset_state();
                    if (options.kvmem_inline_refresh ==
                        KvMemInlineRefreshMode::KvOnly) {
                        if (selected_prefix_tokens == 0) {
                            throw std::runtime_error(
                                "KVMem inline KV-only refresh selected no "
                                "historical context tokens");
                        }
                        do_prefill_vector(
                            refreshed_tokens, 0, 0,
                            selected_prefix_tokens);
                        executor_->restore_recurrent_state(
                            kvmem_query_replay_ckpt);
                        do_prefill_vector(
                            refreshed_tokens, 0,
                            selected_prefix_tokens,
                            refreshed_tokens.size());
                    } else {
                        do_prefill_vector(
                            refreshed_tokens, 0, 0,
                            refreshed_tokens.size());
                    }
                    log("native kvmem inline refresh (mtp): mode=" +
                        std::string(kvmem_inline_refresh_name(
                            options.kvmem_inline_refresh)) +
                        " source_prompt_tokens=" +
                        std::to_string(prompt_tokens.size()) +
                        " selected_prefix_tokens=" +
                        std::to_string(selected_prefix_tokens) +
                        " query_suffix_tokens=" +
                        std::to_string(prompt_tokens.size() -
                                       kvmem_query_replay_begin) +
                        " compact_prompt_tokens=" +
                        std::to_string(refreshed_tokens.size()) +
                        " fixed_context_blocks=" +
                        std::to_string(selected_context.size()));
                } else {
                    executor_->kvmem_begin_query_replay(
                        kvmem_query_replay_ckpt, selected_context,
                        /*reset_recurrent_state=*/false,
                        /*preserve_selected_context=*/true,
                        /*preserve_query_capture=*/
                            query_guided_query ||
                            kvmem_ru.query_snapshot_reuse);
#if 0  // Archived DeltaNet recurrent-state capture/import.
                if (rebuilt_state_import || rebuilt_state_capture) {
                    const std::vector<uint32_t> selected_source_tokens =
                        kvmem_selected_source_tokens(
                            prompt_tokens,
                            executor_->block_store()->blocks(),
                            selected_context);
                    if (rebuilt_state_capture) {
                        executor_->kvmem_export_recurrent_state(
                            kvmem_rebuilt_state_path(
                                options.kvmem_rebuilt_state_capture_key),
                            selected_source_tokens);
                        log("native kvmem rebuilt-state capture (mtp): key=" +
                            options.kvmem_rebuilt_state_capture_key +
                            " identity_tokens=" +
                            std::to_string(selected_source_tokens.size()) +
                            " fixed_context_blocks=" +
                            std::to_string(selected_context.size()));
                    } else {
                        executor_->kvmem_import_recurrent_state(
                            kvmem_rebuilt_state_path(
                                options.kvmem_rebuilt_state_import_key),
                            selected_source_tokens);
                        log("native kvmem rebuilt-state import (mtp): key=" +
                            options.kvmem_rebuilt_state_import_key +
                            " source_tokens=" +
                            std::to_string(selected_source_tokens.size()) +
                            " fixed_context_blocks=" +
                            std::to_string(selected_context.size()));
                    }
                }
#endif
                std::vector<uint32_t> replay_tokens;
                if (api_session) {
                    if (kvmem_query_replay_begin < api_sequence_base ||
                        kvmem_query_replay_begin > logical_prompt_tokens) {
                        throw std::runtime_error(
                            "KVMem session replay boundary is outside the "
                            "checkpoint tail plus appended fragment");
                    }
                    const size_t replay_offset = static_cast<size_t>(
                        kvmem_query_replay_begin - api_sequence_base);
                    replay_tokens.assign(
                        api_sequence_tokens.begin() +
                            static_cast<std::ptrdiff_t>(replay_offset),
                        api_sequence_tokens.end());
                } else {
                    replay_tokens.assign(
                        prompt_tokens.begin() + static_cast<std::ptrdiff_t>(
                            kvmem_query_replay_begin),
                        prompt_tokens.end());
                }
                if (kvmem_do_boundary_capture &&
                    !kvmem_warm_checkpoint_staged &&
                    kvmem_ckpt_split > kvmem_query_replay_begin) {
                    const size_t checkpoint_local = static_cast<size_t>(
                        kvmem_ckpt_split - kvmem_query_replay_begin);
                    if (checkpoint_local >= replay_tokens.size()) {
                        throw std::runtime_error(
                            "KVMem warm prompt checkpoint is outside the query "
                            "replay suffix");
                    }
                    do_prefill_vector(replay_tokens,
                                      kvmem_query_replay_begin, 0,
                                      checkpoint_local);
                    executor_->kvmem_register_append(
                        kvmem_ckpt_split - kvmem_query_replay_begin);
                    kvmem_warm_valid_ = false;
                    executor_->capture_state(kvmem_warm_ckpt_prompt_);
                    kvmem_warm_prompt_pos = kvmem_ckpt_split;
                    kvmem_warm_prompt_resumable =
                        kvmem_prompt_checkpoint_resumable(
                            kvmem_warm_source_index_ready);
                    kvmem_warm_checkpoint_staged = true;
                    do_prefill_vector(replay_tokens,
                                      kvmem_query_replay_begin,
                                      checkpoint_local,
                                      replay_tokens.size());
                } else {
                    do_prefill_vector(replay_tokens,
                                      kvmem_query_replay_begin, 0,
                                      replay_tokens.size());
                }
                executor_->kvmem_end_query_replay();
                const bool query_capture_complete = query_guided_query
                    ? executor_->kvmem_query_captured_tokens() ==
                          query_guided_query_tokens
                    : executor_->kvmem_query_capture_complete();
                if (!query_capture_complete) {
                    throw std::runtime_error(
                        "KVMem query replay final pass captured " +
                        std::to_string(
                            executor_->kvmem_query_captured_tokens()) +
                        "/" +
                        std::to_string(
                            executor_->kvmem_query_expected_tokens()) +
                        " query tokens");
                }
                if (api_session &&
                    api_final_boundary == kvmem_query_replay_begin &&
                    !kvmem_api_boundary_ckpt_.ready) {
                    // No complete new block was crossed during replay, so the
                    // old aligned state is still the correct future boundary.
                    kvmem_api_boundary_ckpt_ =
                        std::move(kvmem_query_replay_ckpt);
                    kvmem_api_boundary_pos_ = api_final_boundary;
                }
                log("native kvmem query replay: boundary=" +
                    std::to_string(kvmem_query_replay_begin) +
                    " suffix_tokens=" +
                    std::to_string(replay_tokens.size()) +
                    " warm_reuse=" +
                    std::to_string(kvmem_warm_reuse ? 1 : 0) +
                    " warm_checkpoint=" +
                    std::string(kvmem_warm_reuse
                                    ? (kvmem_ru.prompt_ckpt ? "P" : "M")
                                    : "none") +
                    " fixed_context_blocks=" +
                    std::to_string(selected_context.size()));
                }
                post_query_replay_s +=
                    wall_seconds() - t_query_replay_start;
            }
            kvmem_registered_pos = executor_->position();
            kvmem_last_reselect_pos = executor_->position();
        }
        if (api_session) {
            if (executor_->position() != logical_prompt_tokens) {
                throw std::runtime_error(
                    "KVMem session prefill ended at an unexpected logical "
                    "position");
            }
            if (api_final_boundary > 0 &&
                (!kvmem_api_boundary_ckpt_.ready ||
                 kvmem_api_boundary_pos_ != api_final_boundary)) {
                throw std::runtime_error(
                    "KVMem session failed to capture its final aligned "
                    "checkpoint");
            }
            const size_t tail_offset = static_cast<size_t>(
                api_final_boundary - api_sequence_base);
            if (tail_offset > api_sequence_tokens.size()) {
                throw std::runtime_error(
                    "KVMem session tail offset exceeds the ingested token "
                    "sequence");
            }
            kvmem_api_tail_tokens_.assign(
                api_sequence_tokens.begin() +
                    static_cast<std::ptrdiff_t>(tail_offset),
                api_sequence_tokens.end());
            if (reset_session) {
                kvmem_api_tokens_ = prompt_tokens;
            } else {
                kvmem_api_tokens_.insert(kvmem_api_tokens_.end(),
                                         prompt_tokens.begin(),
                                         prompt_tokens.end());
            }
            if (std::getenv("QW3_KVMEM_TRACE")) {
                log("native kvmem api-session checkpoint: id=" +
                    options.kvmem_session_id + " boundary=" +
                    std::to_string(kvmem_api_boundary_pos_) + " tail=" +
                    std::to_string(kvmem_api_tail_tokens_.size()) +
                    " logical_end=" +
                    std::to_string(logical_prompt_tokens));
            }
        }
        // kvmem prefix cache: fallback prompt-end (P) checkpoint. When the
        // block-boundary split above already staged ckpt_P at B, skip this;
        // otherwise (tiny prompt, sparse/tiered, or ckpt_M reuse past the last
        // boundary) capture at P here, AFTER register+reselect so the snapshot
        // describes exactly P tokens (block store + window + registered_pos).
        if (kvmem_warm_capture && !kvmem_warm_checkpoint_staged) {
            kvmem_warm_valid_ = false;  // invalid until end-capture re-validates
            executor_->capture_state(kvmem_warm_ckpt_prompt_);
            kvmem_warm_prompt_pos = static_cast<uint32_t>(executor_->position());
            kvmem_warm_prompt_resumable =
                kvmem_prompt_checkpoint_resumable(
                    kvmem_warm_source_index_ready);
        }
#if 0  // Archived DeltaNet recurrent-state export.
        if (rebuilt_state_export) {
            if (kvmem_warm_reuse) {
                throw std::runtime_error(
                    "KVMem rebuilt-state export requires a cold full prefill");
            }
            if (kvmem_on && kvmem_sel_budget > 0 &&
                logical_prompt_tokens > kvmem_sel_budget) {
                throw std::runtime_error(
                    "KVMem rebuilt-state export source exceeds the dense "
                    "KVMem selection budget");
            }
            executor_->kvmem_export_recurrent_state(
                kvmem_rebuilt_state_path(
                    options.kvmem_rebuilt_state_export_key),
                prompt_tokens);
            log("native kvmem rebuilt-state export (mtp): key=" +
                options.kvmem_rebuilt_state_export_key +
                " source_tokens=" + std::to_string(prompt_tokens.size()));
        }
#endif
        if (kvmem_on && QwenExecutor::kvmem_timing_enabled()) {
            QwenExecutor::kvmem_timing_emit_delta("phase=prefill request=mtp",
                                                  kvmem_tbase);
            kvmem_tbase = QwenExecutor::kvmem_timing_snapshot();
        }
        // Boundary between the post-prefill reselect and the decode loop. For
        // plain MTP (kvmem off) the reselect block above is skipped, so this
        // equals t_prefill_end and decode_s stays byte-identical to before.
        const double t_reselect_end = wall_seconds();
        auto emit_native_accounting = [&](double t_decode_done,
                                          double t_native_done) {
            const double setup_s =
                std::max(t_prefill_start - t_native_start, 0.0);
            const double prefill_s_direct =
                std::max(t_prefill_end - t_prefill_start, 0.0);
            const double postprefill_s =
                std::max(t_reselect_end - t_prefill_end, 0.0);
            const double decode_s_direct =
                std::max(t_decode_done - t_reselect_end, 0.0);
            const double finalize_s =
                std::max(t_native_done - t_decode_done, 0.0);
            const double accounted_s =
                setup_s + prefill_s_direct + postprefill_s +
                decode_s_direct + finalize_s;
            const double total_s =
                std::max(t_native_done - t_native_start, 0.0);
            const double post_other_s = std::max(
                postprefill_s - post_semantic_reselect_s -
                    post_query_replay_s - post_query_attention_probe_s -
                    post_query_guided_query_s,
                0.0);
            if (stats_out) {
                stats_out->total_s = total_s;
                stats_out->setup_s = setup_s;
                stats_out->postprefill_s = postprefill_s;
                stats_out->finalize_s = finalize_s;
                stats_out->semantic_reselect_s =
                    post_semantic_reselect_s;
                stats_out->query_replay_s = post_query_replay_s;
                stats_out->query_attention_probe_s =
                    post_query_attention_probe_s;
                stats_out->query_guided_query_s =
                    post_query_guided_query_s;
                stats_out->post_other_s = post_other_s;
                stats_out->query_attention_probe_decoded =
                    query_attention_probe_decoded;
                stats_out->query_attention_probe_used =
                    query_attention_probe_used;
                stats_out->query_score_token_indices =
                    query_score_token_indices;
                stats_out->query_guided_thinking_tokens =
                    query_guided_thinking_tokens;
                stats_out->query_guided_query_tokens =
                    query_guided_query_tokens;
                stats_out->query_guided_thinking_closed =
                    query_guided_thinking_closed;
                stats_out->query_guided_query_text =
                    query_guided_query_text;
            }
            std::ostringstream amsg;
            // Keep enough printed precision that consumers can independently
            // verify total == sum(phases) without a millisecond-rounding
            // discrepancy. The intervals themselves share exact adjacent
            // boundaries; six decimal places here preserves nanosecond-scale
            // accounting after conversion to milliseconds.
            amsg << std::fixed << std::setprecision(6)
                 << "[qw3-native-accounting]"
                 << " trace_tag="
                 << (options.kvmem_trace_tag.empty()
                         ? "-"
                         : options.kvmem_trace_tag)
                 << " total_ms=" << total_s * 1000.0
                 << " setup_ms=" << setup_s * 1000.0
                 << " prefill_ms=" << prefill_s_direct * 1000.0
                 << " postprefill_ms=" << postprefill_s * 1000.0
                 << " decode_ms=" << decode_s_direct * 1000.0
                 << " finalize_ms=" << finalize_s * 1000.0
                 << " sum_ms=" << accounted_s * 1000.0
                 << " error_ms=" << (total_s - accounted_s) * 1000.0
                 << " post_semantic_ms="
                 << post_semantic_reselect_s * 1000.0
                 << " post_query_replay_ms="
                 << post_query_replay_s * 1000.0
                 << " post_query_probe_ms="
                 << post_query_attention_probe_s * 1000.0
                 << " post_guided_query_ms="
                 << post_query_guided_query_s * 1000.0
                 << " post_other_ms=" << post_other_s * 1000.0;
            log(amsg.str());
        };

        // Complete the state transaction here for max_tokens=0. In particular,
        // do not initialize the sampler, draft/verify machinery, or decode KV.
        if (options.max_tokens == 0) {
            if (kvmem_warm_capture) {
                kvmem_warm_log_ = prompt_tokens;
                const size_t pos = executor_->position();
                if (pos <= kvmem_warm_log_.size()) {
                    kvmem_warm_log_.resize(pos);
                    executor_->capture_state(kvmem_warm_ckpt_end_);
                    const bool prompt_source_index_ready =
                        kvmem_warm_source_index_ready &&
                        executor_->kvmem_content_index_resume_compatible(
                            kvmem_warm_prompt_pos);
                    const bool end_source_index_ready =
                        kvmem_warm_source_index_ready &&
                        executor_->kvmem_content_index_resume_compatible(
                            static_cast<uint32_t>(pos));
                    const bool query_snapshot_ready =
                        kvmem_capture_warm_query_snapshot(
                            *executor_, options,
                            qc_prepare && prompt_source_index_ready &&
                                !query_guided_query);
                    kvmem_warm_end_pos_ = static_cast<uint32_t>(pos);
                    kvmem_warm_prompt_pos_ = kvmem_warm_prompt_pos;
                    kvmem_warm_prompt_resumable_ =
                        kvmem_warm_prompt_resumable;
                    kvmem_warm_source_index_ready_ =
                        prompt_source_index_ready;
                    kvmem_warm_end_resumable_ =
                        kvmem_prompt_checkpoint_resumable(
                            end_source_index_ready);
                    kvmem_warm_end_source_index_ready_ =
                        end_source_index_ready;
                    kvmem_warm_valid_ = true;
                    if (kvmem_prefix_cache_trace_enabled()) {
                        std::ostringstream cmsg;
                        cmsg << "kvmem prefix-cache CAPTURE "
                             << "(mtp prefill-only): warm_log="
                             << kvmem_warm_log_.size() << " pos=" << pos
                             << " P=" << kvmem_warm_prompt_pos
                             << " P_resumable="
                             << (kvmem_warm_prompt_resumable ? 1 : 0)
                             << " P_source_index="
                             << (prompt_source_index_ready ? 1 : 0)
                             << " M_resumable="
                             << (kvmem_warm_end_resumable_ ? 1 : 0)
                             << " M_source_index="
                             << (end_source_index_ready ? 1 : 0)
                             << " query_snapshot="
                             << (query_snapshot_ready ? 1 : 0);
                        log(cmsg.str());
                    }
                } else {
                    kvmem_warm_valid_ = false;
                }
            }
            if (manage_device_scope) {
                st = device_->end();
                if (!st.ok) throw std::runtime_error(st.message);
            }
            const double t_native_end = wall_seconds();
            emit_native_accounting(t_reselect_end, t_native_end);
            const double prefill_s =
                std::max(t_prefill_end - t_prefill_start, 1e-9);
            const double reselect_s =
                std::max(t_reselect_end - t_prefill_end, 0.0);
            std::ostringstream msg;
            msg << "native prefill-only: prompt_tokens="
                << prompt_tokens.size() << " prefill="
                << fmt_seconds(prefill_s) << " reselect="
                << fmt_seconds(reselect_s) << " prefill_ops=" << prefill_ops;
            if (kvmem_warm_reuse) {
                msg << " kvmem_reuse=" << kvmem_reuse_m
                    << " prefilled="
                    << (prompt_tokens.size() - kvmem_reuse_m);
            }
            log(msg.str());
            if (stats_out) {
                stats_out->prefill_s = prefill_s;
                stats_out->decode_s = 0.0;
                stats_out->reselect_s = reselect_s;
                stats_out->decoded = 0;
                stats_out->prompt_tokens = prompt_tokens.size();
                stats_out->acceptance = 0.0;
                stats_out->kvmem_at_boundary = kvmem_at_prefill_end;
                stats_out->kvmem_boundary_valid = kvmem_boundary_valid;
            }
            return {};
        }

        std::string generated;
        // Committed decode token ids for the kvmem warm checkpoint; appended in
        // emit order (== commit order) only when a warm capture is intended.
        std::vector<uint32_t> gen_tokens;
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
        bool stream_cancelled = false;
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
            if (!mtp_rebuild_accepted_prefix_enabled() &&
                !mtp_local_positions) {
                executor_->commit_mtp_prefix(executor_->position());
                if (finish_deferred_kvmem) executor_->kvmem_finish_reselect();
                return;
            }
            const double t_prefix_start = mtp_phase_time();
            NativeExecutorReport prefix;
            if (mtp_local_positions) {
                if (executor_->last_forward_rows() < tokens.size() ||
                    executor_->last_forward_logical_base() != base_position) {
                    throw std::runtime_error(
                        "MTP accepted-prefix rebuild lost compact target "
                        "position metadata");
                }
                prefix = executor_->prime_mtp_prefix_from_last_batch_at(
                    tokens, base_position,
                    executor_->last_forward_rope_base(),
                    mtp_prefix_rebuild_batch_min_tokens());
            } else {
                prefix = executor_->prime_mtp_prefix_from_last_batch(
                    tokens, base_position,
                    mtp_prefix_rebuild_batch_min_tokens());
            }
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
            if (!mtp_rebuild_accepted_prefix_enabled() &&
                !mtp_local_positions) {
                executor_->commit_mtp_prefix(executor_->position());
                if (finish_deferred_kvmem) executor_->kvmem_finish_reselect();
                return;
            }
            const double t_prefix_start = mtp_phase_time();
            NativeExecutorReport prefix = mtp_local_positions
                ? executor_->prime_mtp_prefix_from_current_at(
                      token, base_position,
                      executor_->last_committed_rope_position())
                : executor_->prime_mtp_prefix_from_current(
                      token, base_position);
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
            if (!mtp_prefix_positions_safe) return;
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
            return !options.ignore_eos && token == static_cast<uint32_t>(eos) &&
                   !budget.can_recover_eos(token);
        };
        auto emit_generated_token = [&](uint32_t &token) -> bool {
            recover_thinking_eos(budget, token);
            if (decoded >= options.max_tokens || should_stop_mtp_eos(token)) return false;
            const std::string piece = tokenizer_->decode_one(static_cast<int32_t>(token));
            generated += piece;
            budget_observe(budget, token);
            // Track committed tokens for the next accept test's penalties (no-op
            // for greedy without penalties, keeping that path byte-identical).
            if (mtp_need_logits_pick) ++mtp_seen[token];
            if (kvmem_warm_capture) gen_tokens.push_back(token);
            ++decoded;
            if (on_text && !on_text(piece)) {
                stream_cancelled = true;
                return false;
            }
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
            while (!stream_cancelled &&
                   decoded < options.max_tokens &&
                   !should_stop_mtp_eos(next_token)) {
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
                   !stream_cancelled &&
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
                        should_stop_mtp_eos(static_cast<uint32_t>(mtp.argmax_token)) ||
                        budget.can_recover_eos(static_cast<uint32_t>(mtp.argmax_token))) {
                        break;
                    }
                    drafts.push_back(static_cast<uint32_t>(mtp.argmax_token));
                    ++mtp_drafts;
                    ++mtp_spec_drafted;
                    mtp_ops += mtp.ops_executed;
                    ++step_index;
                }

                if (drafts.empty()) {
                    const uint32_t current_base = executor_->position();
                    step = executor_->forward_one_token(current);
                    if (!step.ok) throw std::runtime_error("decode failed");
                    decode_ops += step.ops_executed;
                    const int32_t new_argmax = step.argmax_token >= 0 ? step.argmax_token : eos;
                    // Sample/pick before commit_mtp_prefix so logits_ is intact.
                    next_token = pick_from_last_logits(new_argmax);
                    if (mtp_local_positions) {
                        const bool finish_after_prefix =
                            kvmem_advance_to(executor_->position(),
                                             /*defer_finish=*/true);
                        rebuild_current_mtp_prefix(
                            current, current_base, finish_after_prefix);
                    } else {
                        executor_->commit_mtp_prefix(executor_->position());
                        kvmem_advance_to(executor_->position());
                    }
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
                            mtp_rebuild_accepted_prefix_enabled() ||
                                mtp_local_positions);
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
                    if (stream_cancelled ||
                        decoded >= options.max_tokens) break;
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
                                (mtp_rebuild_accepted_prefix_enabled() ||
                                 mtp_local_positions));
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
                        mtp_rebuild_accepted_prefix_enabled() &&
                        !mtp_local_positions) {
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
                    if (stream_cancelled ||
                        decoded >= options.max_tokens) break;
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
        if (!run_spec_mtp && !stream_cancelled) {
            const double t_plain_start = mtp_phase_time();
            run_plain_decode_remaining();
            if (spec_mtp) {
                mtp_spec_plain_s += mtp_phase_time() - t_plain_start;
            }
        }
        const double t_decode_end = wall_seconds();

        // kvmem prefix cache: record this request's end as the warm resume point.
        // warm_log = prompt + emitted tokens, resized down to the executor
        // position (the last emitted token is never forwarded, so its KV is not
        // resident). position() beyond the log (e.g. an accepted token committed
        // to KV but not emitted because max_tokens was hit) invalidates the warm
        // state instead of capturing an inconsistent checkpoint. Captured inside
        // the device scope since capture_state issues device copies.
        if (kvmem_warm_capture && !stream_cancelled) {
            kvmem_warm_log_ = prompt_tokens;
            kvmem_warm_log_.insert(kvmem_warm_log_.end(),
                                   gen_tokens.begin(), gen_tokens.end());
            const size_t pos = executor_->position();
            if (pos <= kvmem_warm_log_.size()) {
                kvmem_warm_log_.resize(pos);
                executor_->capture_state(kvmem_warm_ckpt_end_);
                const bool prompt_source_index_ready =
                    kvmem_warm_source_index_ready &&
                    executor_->kvmem_content_index_resume_compatible(
                        kvmem_warm_prompt_pos);
                const bool end_source_index_ready =
                    kvmem_warm_source_index_ready &&
                    executor_->kvmem_content_index_resume_compatible(
                        static_cast<uint32_t>(pos));
                const bool query_snapshot_ready =
                    kvmem_capture_warm_query_snapshot(
                        *executor_, options,
                        qc_prepare && prompt_source_index_ready &&
                            !query_guided_query);
                // Commit the staged prompt-end (P) checkpoint alongside the
                // turn-end (M) one; flip all warm fields together so a mid-decode
                // throw leaves the prior turn's warm state intact.
                kvmem_warm_end_pos_ = static_cast<uint32_t>(pos);
                kvmem_warm_prompt_pos_ = kvmem_warm_prompt_pos;
                kvmem_warm_prompt_resumable_ = kvmem_warm_prompt_resumable;
                kvmem_warm_source_index_ready_ =
                    prompt_source_index_ready;
                kvmem_warm_end_resumable_ =
                    kvmem_prompt_checkpoint_resumable(
                        end_source_index_ready);
                kvmem_warm_end_source_index_ready_ =
                    end_source_index_ready;
                kvmem_warm_valid_ = true;
                if (kvmem_prefix_cache_trace_enabled()) {
                    std::ostringstream cmsg;
                    cmsg << "kvmem prefix-cache CAPTURE (mtp): warm_log="
                         << kvmem_warm_log_.size() << " pos=" << pos
                         << " P=" << kvmem_warm_prompt_pos
                         << " P_resumable=" << (kvmem_warm_prompt_resumable ? 1 : 0)
                         << " P_source_index="
                         << (prompt_source_index_ready ? 1 : 0)
                         << " M_resumable="
                         << (kvmem_warm_end_resumable_ ? 1 : 0)
                         << " M_source_index="
                         << (end_source_index_ready ? 1 : 0)
                         << " query_snapshot="
                         << (query_snapshot_ready ? 1 : 0)
                         << " decoded=" << decoded;
                    log(cmsg.str());
                }
            } else {
                kvmem_warm_valid_ = false;
            }
        } else if (stream_cancelled) {
            kvmem_warm_valid_ = false;
        }

        if (manage_device_scope) {
            st = device_->end();
            if (!st.ok) throw std::runtime_error(st.message);
        }
        const double t_native_end = wall_seconds();
        emit_native_accounting(t_decode_end, t_native_end);

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
        if (kvmem_warm_reuse) {
            msg << " kvmem_reuse=" << kvmem_reuse_m
                << " prefilled=" << (prompt_tokens.size() - kvmem_reuse_m);
        }
        if (stream_cancelled) msg << " cancelled=true";
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
                 << " cpu_raw_k=" << tu.cpu_raw_k_bytes
                 << " cpu_spill=" << tu.cpu_spill_bytes
                 << " nvme_used=" << tu.nvme_used_bytes
                 << " nvme_raw_k=" << tu.nvme_raw_k_bytes
                 << " nvme_spill=" << tu.nvme_spill_bytes
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
                            double asm_rerope_ms, double asm_final_drain_ms,
                            double asm_kbar_ms,
                            uint32_t stage_in_blocks, uint32_t stage_out_blocks,
                            double prefill_s, double decode_s, double decode_tps,
                            int decoded, double acceptance, uint64_t gpu_mib,
                            uint64_t rss_mib, double inpre_ms = 0.0,
                            uint32_t inpre_stage_in_blocks = 0,
                            uint32_t inpre_stage_out_blocks = 0,
                            double total_s = 0.0, double setup_s = 0.0,
                            double postprefill_s = 0.0,
                            double finalize_s = 0.0,
                            double semantic_s = 0.0,
                            double query_replay_s = 0.0,
                            double post_other_s = 0.0) const {
        std::ostringstream m;
        m << std::fixed;
        m << "\n[kvmem-session] turn=" << turn
          << " ctx=" << ctx_tokens << "tok (+" << delta_tokens << ")"
          << " GPU=" << gpu_mib << "MiB RSS=" << rss_mib << "MiB\n";
        // The report is machine parsed. Six decimal places ensure the printed
        // components retain the same additive conservation as the underlying
        // adjacent wall-clock intervals.
        m << std::setprecision(6);
        const double phase_sum_s =
            setup_s + prefill_s + postprefill_s + decode_s + finalize_s;
        m << "  total native execution                   = "
          << std::setw(10) << total_s * 1000.0 << " ms\n";
        m << "    phase1 setup                           = "
          << std::setw(10) << setup_s * 1000.0 << " ms\n";
        m << "    phase2 prefill (forward new chunk)     = "
          << std::setw(10) << prefill_s * 1000.0 << " ms  ("
          << std::setprecision(1)
          << (delta_tokens / std::max(prefill_s, 1e-9))
          << " tok/s)\n";
        m << std::setprecision(6);
        m << "    phase3 post-prefill/reselection        = "
          << std::setw(10) << postprefill_s * 1000.0 << " ms\n";
        m << "      semantic selection                   = "
          << std::setw(10) << semantic_s * 1000.0 << " ms\n";
        m << "      query replay                         = "
          << std::setw(10) << query_replay_s * 1000.0 << " ms\n";
        m << "      other                                = "
          << std::setw(10) << post_other_s * 1000.0 << " ms\n";
        m << "    phase4 decode (MTP, " << decoded << " tok)             = "
          << std::setw(10) << decode_s * 1000.0 << " ms  ("
          << std::setprecision(2) << decode_tps << " tok/s, accept="
          << std::setprecision(4) << acceptance << ")\n";
        m << std::setprecision(6);
        m << "    phase5 finalize                        = "
          << std::setw(10) << finalize_s * 1000.0 << " ms\n";
        m << "    accounting error                       = "
          << std::setw(10) << (total_s - phase_sum_s) * 1000.0
          << " ms\n";
        m << "  nested KVMem diagnostics (not additive; I/O may overlap)\n";
        m << "    selection (top-k retrieval)            = "
          << std::setw(10) << sel_ms << " ms\n";
        m << "    stage-in   (CPU/NVMe -> GPU)           = "
          << std::setw(10) << stage_in_ms << " ms  (" << stage_in_blocks
          << " blk)\n";
        m << "    stage-out  (GPU -> CPU/NVMe evict)     = "
          << std::setw(10) << stage_out_ms << " ms  (" << stage_out_blocks
          << " blk)\n";
        m << "    assemble   (submit phases + final drain) = "
          << std::setw(10) << assemble_ms << " ms  (pages=" << asm_pages_ms
          << " rerope=" << asm_rerope_ms
          << " final_drain=" << asm_final_drain_ms
          << " kbar=" << asm_kbar_ms << ")\n";
        m << std::setprecision(6);
        // The bounded GPU pool can force kvmem stage-in/out mid-prefill; that
        // cost is INSIDE step5's wall above (not double-counted in steps 1-4,
        // which are the post-prefill decode-window reselect only).
        m << "    +-- of which kvmem in-prefill offload   = "
          << std::setw(10) << inpre_ms << " ms  (in=" << inpre_stage_in_blocks
          << " out=" << inpre_stage_out_blocks << " blk)\n";
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
    // Declared before executor_ so it is destroyed after it: the executor holds
    // a borrowed pointer to the archive for as long as it can write.
    std::unique_ptr<KvMemArchive> kvmem_archive_;
    uint64_t kvmem_archive_ladder_tokens_ = 0;
    // Frozen base shared by the dedicated archive-backed Serve path. Payload
    // authority remains in the immutable archive; these are only the live
    // recurrent/window checkpoint and host token/selection metadata.
    QwenExecutor::StateSnapshot archive_query_base_state_;
    std::vector<uint32_t> archive_query_base_selection_;
    std::vector<uint32_t> archive_query_base_tokens_;
    uint32_t archive_query_base_pos_ = 0;
    uint64_t archive_query_base_prefix_tokens_ = 0;
    bool archive_query_base_ready_ = false;
    std::unique_ptr<QwenExecutor> executor_;
    std::unique_ptr<QwenTokenizer> tokenizer_;
    std::unique_ptr<BatchedPrefillExecutor> cb_prefill_executor_;
    std::unique_ptr<BatchedDecodeExecutor> cb_decode_executor_;
    std::unique_ptr<GlobalKvPagePool> cb_kv_pool_;
    std::unique_ptr<GlobalKvPagePool> cb_mtp_kv_pool_;

    // ---- Prefix cache: lossless page-aligned prefix reuse -----------------
    // One entry per committed prompt prefix. Main pages and optional matching
    // MTP draft pages are pinned in their separate global pools; recur holds
    // recurrent/conv plus MTP-prefix hidden state at exactly aligned_len.
    struct PrefixCacheEntry {
        uint64_t id = 0;
        std::vector<uint32_t> tokens;     // exact prefix tokens (collision-safe)
        uint32_t aligned_len = 0;         // == tokens.size(), multiple of page_size
        std::vector<int32_t> kv_pages;    // pinned physical pages, logical 0..n
        std::vector<int32_t> mtp_kv_pages;// matching pinned MTP draft pages
        QwenExecutor::StateSnapshot recur;// recurrent/conv/MTP state at boundary
        uint32_t refcount = 0;            // live requests reading these pages
        uint64_t last_used_seq = 0;       // LRU
    };
    std::unordered_map<uint64_t, std::vector<PrefixCacheEntry>> prefix_cache_;
    std::mutex prefix_cache_mu_;
    uint64_t prefix_cache_seq_ = 0;
    uint64_t prefix_cache_next_id_ = 1;
    uint32_t prefix_cache_pinned_pages_ = 0;
    uint32_t prefix_cache_entry_count_ = 0;
    bool prefix_cache_evict_cb_installed_ = false;

    // ---- kvmem single-request prefix cache (QW3_KVMEM_PREFIX_CACHE) --------
    // Keeps the plain-route shared executor_ warm across requests. kvmem_warm_log_
    // is the full token sequence of the last request (prompt + committed decode
    // tokens). Two resume checkpoints of that request are kept (a small ladder):
    //   kvmem_warm_ckpt_end_    -- capture_state at turn end (position M, after
    //                              decode). Reuse here needs the whole warm log
    //                              to be a token prefix of the new prompt.
    //   kvmem_warm_ckpt_prompt_ -- capture_state at prompt end (position P, after
    //                              the post-prefill reselect, before decode). Lets
    //                              a new prompt that diverges INSIDE the prior
    //                              response region [P,M) still resume at P and
    //                              re-prefill only [P,end) instead of the whole
    //                              accumulated prompt. Only offered when the prior
    //                              request stayed dense/GPU-resident, or when a
    //                              tier-backed sparse history captured a complete
    //                              position-invariant source index.
    // On a new request we pick the largest checkpoint C in {M,P} with C <= D (the
    // longest common token prefix) and C < prompt.size(), restore it, rewind the
    // block store to C (kvmem_truncate_to), and prefill [C,end).
    std::vector<uint32_t> kvmem_warm_log_;
    QwenExecutor::StateSnapshot kvmem_warm_ckpt_end_;
    QwenExecutor::StateSnapshot kvmem_warm_ckpt_prompt_;
    uint32_t kvmem_warm_prompt_pos_ = 0;   // P (position of ckpt_prompt_)
    uint32_t kvmem_warm_end_pos_ = 0;      // M (position of ckpt_end_)
    bool kvmem_warm_prompt_resumable_ = false;  // ckpt_prompt_ safe to resume
    bool kvmem_warm_source_index_ready_ = false; // historical scorer rows durable
    bool kvmem_warm_end_resumable_ = false; // ckpt_end_ tier/window state durable
    bool kvmem_warm_end_source_index_ready_ = false; // scorer covers [0,M)
    bool kvmem_warm_query_stashed_ = false; // complete Q rows for post-query P/M
    uint32_t kvmem_warm_query_begin_ = 0;
    uint32_t kvmem_warm_query_end_ = 0;
    bool kvmem_warm_valid_ = false;

    // Generic API-level persistent session. The first implementation is
    // intentionally single-active-session because executor_ itself is a single
    // live KV/recurrent state; the id prevents accidental cross-sample appends.
    bool kvmem_api_session_active_ = false;
    std::string kvmem_api_session_id_;
    // Last block-aligned model/recurrent state plus the at-most-(block-1)
    // teacher-forced tokens after it. Together they let the next request replay
    // a query whose first token shares the preceding partial block, without
    // rebuilding the historical selected context.
    QwenExecutor::StateSnapshot kvmem_api_boundary_ckpt_;
    uint32_t kvmem_api_boundary_pos_ = 0;
    std::vector<uint32_t> kvmem_api_tail_tokens_;
    // Canonical teacher-forced token history for named local-cache integrity
    // metadata. This is host-only (4 bytes/token) and does not duplicate KV.
    std::vector<uint32_t> kvmem_api_tokens_;

    // Phase-1 request-level local checkpoint registry. Only one executor
    // lineage can be ready at a time; a cold unrelated request evicts ready
    // entries before reset_state clears their shared pool/tier authority.
    std::unordered_map<std::string, LocalKvMemCacheEntry>
        kvmem_local_caches_;

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

namespace {

// Both archive commands need the same immutable-K + fp8 + SSD-authority shape;
// they differ only in whether the arenas are writable.
void apply_archive_engine_defaults(EngineOptions &engine, const char *mode) {
    engine.backend = BackendKind::QwenNative;
    engine.native_heavy = true;
    if (engine.native_kernels.empty()) engine.native_kernels = "cuda";
    if (engine.prefill_chunk < 0) engine.prefill_chunk = 2048;
    engine.kvmem_enabled = true;
    engine.kvmem_immutable_source_k = true;
    engine.kvmem_raw_k_nvme = true;
    engine.kvmem_archive_mode = mode;
    if (!engine.native_mtp_chain_set || engine.native_mtp_chain <= 0) {
        engine.native_mtp_chain = 4;
        engine.native_mtp_chain_set = true;
    }
    engine.native_mtp_speculate = true;
    setenv("QW3_MTP_SPECULATE", "1", 1);
    setenv("QW3_MTP_POLICY",
           engine.mtp_policy.empty() ? "fixed" : engine.mtp_policy.c_str(), 1);
    // The archive format is fp8-only: 10M tokens is 305 GiB at fp8 and 610 GiB
    // at fp16, and immutable-K is already mandatory for fp8.
    setenv("QW3_KV_DTYPE", "fp8", 1);
}

} // namespace

int run_kvmem_archive_build(EngineOptions engine,
                            const KvMemArchiveBuildConfig &cfg) {
    if (engine.kvmem_archive_dir.empty()) {
        throw std::invalid_argument("archive build requires --kvmem-archive");
    }
    apply_archive_engine_defaults(engine, "build");
    engine.kvmem_update_mode = "step";
    if (cfg.ladder_tokens > 0) {
        engine.kvmem_archive_ladder_tokens = cfg.ladder_tokens;
    }
    QwenNativeBackend backend;
    backend.load(engine);
    return backend.build_kvmem_archive(cfg);
}

int run_kvmem_archive_query(EngineOptions engine,
                            const KvMemArchiveRunConfig &cfg) {
    if (engine.kvmem_archive_dir.empty()) {
        throw std::invalid_argument("archive query requires --kvmem-archive");
    }
    apply_archive_engine_defaults(engine, "attach");
    engine.kvmem_query_conditioned = true;
    engine.kvmem_update_mode = "step";
    QwenNativeBackend backend;
    backend.load(engine);
    return backend.run_kvmem_archive(cfg);
}

int run_kvmem_archive_info(const std::string &dir) {
    const KvMemArchiveManifest m = KvMemArchive::read_manifest(dir);
    const KvMemArchiveLayout &l = m.layout;
    std::printf("archive:        %s\n", dir.c_str());
    std::printf("format_version: %u\n", m.format_version);
    std::printf("sealed:         %s\n", m.sealed ? "yes" : "no");
    std::printf("tokens:         %llu\n",
                static_cast<unsigned long long>(m.total_tokens));
    std::printf("blocks:         %u\n", m.total_blocks);
    std::printf("raw_chunks:     %u\n", m.raw_chunks);
    std::printf("ladder_points:  %zu\n", m.ladder.size());
    if (!m.ladder.empty()) {
        std::printf("ladder_stride:  %llu\n",
                    static_cast<unsigned long long>(
                        m.ladder.size() > 1 ? m.ladder[1] - m.ladder[0]
                                            : m.ladder[0]));
    }
    std::printf("model:          %s (%llu bytes)\n", l.model_name.c_str(),
                static_cast<unsigned long long>(l.model_bytes));
    std::printf("model_sha256:   %s\n",
                l.model_sha256.empty() ? "(legacy-unavailable)"
                                       : l.model_sha256.c_str());
    std::printf("kv_dtype:       %s\n", l.kv_dtype.c_str());
    std::printf("block_tokens:   %u\n", l.block_tokens);
    std::printf("chunk_tokens:   %u\n", l.raw_chunk_tokens);
    std::printf("std_layers:     %u of %u\n", l.n_standard_layers, l.n_layers);
    std::printf("raw_k_bytes:    %llu / chunk\n",
                static_cast<unsigned long long>(l.raw_chunk_bytes));
    std::printf("v_bytes:        %llu / block\n",
                static_cast<unsigned long long>(l.v_block_bytes));
    std::printf("layout_key:     %s\n", l.key().c_str());
    std::printf("policy_at_build: %s\n", m.policy_snapshot.c_str());
    return 0;
}

} // namespace qw3
