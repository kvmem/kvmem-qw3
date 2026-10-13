#include "qwen_executor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace qw3 {
namespace {
void check_action(DeviceStatus s) {
    if (!s.ok) throw std::runtime_error(s.message ? s.message : "action attention failed");
}

}

void QwenExecutor::kvmem_action_configure(KvMemStoreConfig &cfg) {
    const auto &attention = cfg.attention;
    attention.validate();
    action_enabled_ = attention.enabled;
    action_running_ = false;
    action_pressure_ = attention.pressure;
    prefill_attention_enabled_ = attention.prefill;
    prefill_attention_half_life_ = attention.prefill_half_life;
    prefill_attention_interval_ = attention.prefill_interval;
    prefill_attention_stride_ = attention.prefill_stride;
    prefill_attention_scores_.clear();
    prefill_attention_end_ = prefill_attention_samples_ = 0;
    action_capture_prefill_ = false; action_capture_rows_ = 0;
    cfg.action_fraction = action_enabled_ ? attention.fraction : 0;
    cfg.action_combine = attention.combine;
    if (!action_enabled_) return;
    action_parser_.scope = detail::ActionAttentionParser::parse_scope(attention.scope);
    if (cfg.select_method != KvMemMethod::Retrieval ||
        cfg.semantic_expansion != KvMemSemanticExpansion::None)
        throw std::invalid_argument("action attention currently requires retrieval with block selection");
    action_decay_ = attention.decay;
    action_parser_.interval = attention.interval;
    std::vector<int32_t> standard;
    for (uint32_t il = 0; il < weights_.n_layers(); ++il)
        if (model_.config().is_standard_attention_layer(il)) standard.push_back(il);
    action_layers_ = detail::resolve_attention_layers(attention.layers, standard);
    std::string layer_ids;
    for (int32_t id : action_layers_) layer_ids += (layer_ids.empty() ? "" : ",") + std::to_string(id);
    std::fprintf(stderr, "[action-attention] enabled layers=%s layer_policy=%s aggregation=equal-mean interval=%u fraction=%.3f decay=%.3f scope=%s combine=%s\n",
        layer_ids.c_str(), attention.layers.c_str(), action_parser_.interval, cfg.action_fraction, action_decay_, action_parser_.scope_name(), attention.combine.c_str());
    std::fprintf(stderr, "[action-pressure] enabled=%d resident_only=1 accepted_round_overlay=1\n",
        action_pressure_ ? 1 : 0);
    if (prefill_attention_enabled_)
        std::fprintf(stderr, "[prefill-attention] enabled=1 query_stride=%u interval_tokens=%u force_near_capacity=1 half_life_tokens=%.0f normalized_exposure=1 policy=no-output-fallback semantic=1\n",
                     prefill_attention_stride_, prefill_attention_interval_, prefill_attention_half_life_);
}

KvMemStore::PressureActionScores QwenExecutor::kvmem_action_pressure_scores() const {
    KvMemStore::PressureActionScores scores;
    if (!action_pressure_ || !block_store_ || kvmem_query_probe_guard_active() ||
        kvmem_query_replay_active_ || kvmem_provisional_prefill_ || kvmem_blend_active_)
        return scores;
    std::vector<uint32_t> resident;
    uint32_t prefix = 0;
    if (kvmem_active_) {
        if (block_store_->has_repository()) {
            if (bs_window_repository_handles_.size() != bs_window_blocks_)
                throw std::logic_error("pressure action window has missing repository identities");
            for (uint32_t w = 0; w < bs_window_blocks_; ++w)
                if (block_store_->blocks().at(bs_window_block_ids_[w]).repository_generation !=
                    bs_window_repository_handles_[w].generation)
                    throw std::logic_error("pressure action window identity changed");
        }
        resident.assign(bs_window_block_ids_.begin(), bs_window_block_ids_.begin() + bs_window_blocks_);
        if (bs_window_blocks_)
            prefix = bs_win_base_host_[bs_window_blocks_ - 1] + bs_blk_tokens_host_[bs_window_blocks_ - 1];
    }
    // Inner prefill chunks register accepted rows before forward_n_tokens
    // advances position_. Include that committed tail in the next pressure check.
    const uint64_t committed_position = std::max(position_, kvmem_registered_pos_);
    const uint64_t physical = kvmem_active_ ? window_query_pos_ : committed_position;
    if (physical < prefix || physical - prefix > committed_position)
        throw std::logic_error("invalid pressure action window origin");
    const uint64_t tail_begin = committed_position - (physical - prefix);
    const uint64_t end = std::min(committed_position, block_store_->history_end());
    if (tail_begin < end) {
        if (block_store_->has_repository()) {
            const auto tail = block_store_->history_span_blocks(tail_begin, end, false);
            resident.insert(resident.end(), tail.begin(), tail.end());
        } else {
            const uint32_t bt = block_store_->config().block_tokens;
            for (uint64_t id = tail_begin / bt; id < (end + bt - 1) / bt; ++id)
                resident.push_back(static_cast<uint32_t>(id));
        }
    }
    std::sort(resident.begin(), resident.end());
    resident.erase(std::unique(resident.begin(), resident.end()), resident.end());
    const bool overlay = action_running_ && action_round_samples_ > 0;
    scores.reserve(resident.size());
    for (uint32_t id : resident) {
        if (!block_store_->is_live(id)) continue;
        const auto &b = block_store_->blocks().at(id);
        double score = b.action_score;
        if (overlay) {
            const uint64_t key = block_store_->has_repository() ? b.history_pos_start : b.orig_pos_start;
            const auto it = action_round_mass_.find(key);
            score = action_decay_ * score + (it == action_round_mass_.end()
                ? 0.0 : it->second / action_round_samples_);
        }
        scores.emplace_back(id, score);
    }
    // Prefill is a separate signal. Use it only when this resident window has
    // no positive output-attention evidence; never mix differently scaled scores
    // or let a long input dilute the committed action history.
    if (prefill_attention_enabled_ &&
        std::none_of(scores.begin(), scores.end(), [](const auto &s) { return s.second > 0; })) {
        uint32_t positive = 0;
        for (auto &[id, score] : scores) {
            const auto &b = block_store_->blocks().at(id);
            const uint64_t key = block_store_->has_repository() ? b.history_pos_start : b.orig_pos_start;
            const auto it = prefill_attention_scores_.find(key);
            if (it != prefill_attention_scores_.end()) score = it->second.mean();
            positive += score > 0;
        }
        if (positive && std::getenv("QW3_KVMEM_TRACE"))
            std::fprintf(stderr, "[prefill-attention-pressure] positive=%u resident=%zu samples=%llu\n",
                         positive, scores.size(), static_cast<unsigned long long>(prefill_attention_samples_));
    }
    return scores;
}

QwenExecutor::AttentionHistorySnapshot
QwenExecutor::kvmem_capture_attention_history() const {
    if (action_running_)
        throw std::logic_error("attention cache capture requires a completed request");
    AttentionHistorySnapshot snapshot;
    snapshot.prefill_end = prefill_attention_end_;
    snapshot.prefill_samples = prefill_attention_samples_;
    snapshot.prefill = prefill_attention_scores_;
    snapshot.window_blocks = bs_window_block_ids_;
    snapshot.window_bases = bs_win_base_host_;
    snapshot.window_counts = bs_blk_tokens_host_;
    snapshot.window_handles = bs_window_repository_handles_;
    if (block_store_) {
        snapshot.output.reserve(block_store_->block_count());
        for (const auto &block : block_store_->blocks())
            snapshot.output.push_back(block.action_score);
    }
    return snapshot;
}

void QwenExecutor::kvmem_restore_cached_window_identity(
        const AttentionHistorySnapshot &snapshot) {
    if (snapshot.window_blocks.size() != snapshot.window_bases.size() ||
        snapshot.window_blocks.size() != snapshot.window_counts.size())
        throw std::logic_error("cached sparse window has inconsistent identities");
    // restore_state restores the cached window coordinate frame. Truncation
    // must validate that SAME frame's identities, not the last query's sparse
    // selection (which can contain blocks beyond the frozen prefix).
    bs_score_ready_ = false;
    bs_window_block_ids_ = snapshot.window_blocks;
    bs_win_base_host_ = snapshot.window_bases;
    bs_blk_tokens_host_ = snapshot.window_counts;
    bs_window_repository_handles_ = snapshot.window_handles;
    bs_window_blocks_ = static_cast<uint32_t>(snapshot.window_blocks.size());
}

void QwenExecutor::kvmem_restore_attention_history(
        const AttentionHistorySnapshot &snapshot) {
    if (snapshot.prefill_end > position_ || !block_store_ ||
        snapshot.output.size() != block_store_->block_count())
        throw std::logic_error("attention cache does not match restored KV history");
    check_action(backend_.attention_capture_set(nullptr, 0, 0, 0));
    prefill_attention_scores_ = snapshot.prefill;
    prefill_attention_end_ = snapshot.prefill_end;
    prefill_attention_samples_ = snapshot.prefill_samples;
    block_store_->update_action_scores(snapshot.output, 0);
    action_running_ = action_captured_ = action_capture_prefill_ = false;
    action_capture_rows_ = action_capture_count_ = action_captured_layers_ = 0;
    action_round_samples_ = 0;
    action_round_mass_.clear();
    action_parser_.committed = {};
    action_parser_.candidate.clear();
    action_parser_.mask = 0;
    action_parser_.accepted = 0;
    if (action_enabled_) {
        double output_sum = 0;
        for (double score : snapshot.output) output_sum += score;
        std::fprintf(stderr, "[attention-cache] restored_end=%llu prefill_query_samples=%llu prefill_blocks=%zu output_score_sum=%.17g\n",
            static_cast<unsigned long long>(snapshot.prefill_end),
            static_cast<unsigned long long>(snapshot.prefill_samples),
            snapshot.prefill.size(), output_sum);
    }
}

void QwenExecutor::kvmem_action_begin(int32_t open, int32_t close,
        int32_t think_open, int32_t think_close, bool thinking) {
    if (!action_enabled_) return;
    if (think_open < 0 || think_close < 0)
        throw std::invalid_argument("action attention requires think special tokens");
    if (action_parser_.scope == detail::ActionAttentionParser::Scope::ToolCall && (open < 0 || close < 0))
        throw std::invalid_argument("tool_call attention scope requires tool_call special tokens");
    check_action(backend_.attention_capture_set(nullptr, 0, 0, 0));
    action_parser_.open = open; action_parser_.close = close;
    action_parser_.think_open = think_open; action_parser_.think_close = think_close;
    action_parser_.committed = {thinking, false, 0};
    action_parser_.candidate.clear(); action_parser_.mask = 0;
    action_running_ = true; action_captured_ = false;
    action_capture_prefill_ = false; action_capture_rows_ = 0;
    action_round_samples_ = 0; action_round_mass_.clear();
}

void QwenExecutor::kvmem_action_prepare(const uint32_t *tokens, uint32_t count, uint64_t base, bool prefill) {
    if (kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_) return;
    action_capture_prefill_ = prefill_attention_enabled_ && prefill && !action_running_ &&
        kvmem_enabled_ && kvmem_gpu_page_pool_ && count &&
        base + count > kvmem_registered_pos_ && base + count > prefill_attention_end_;
    if (action_capture_prefill_) {
        // Pages for this chunk are already allocated. Sample before headroom is
        // exhausted even if the regular interval is not due, so newly ingested
        // evidence has scores at the next capacity recovery.
        const uint32_t pages = (count + kv_pages_.page_size - 1) / kv_pages_.page_size;
        uint32_t free = kvmem_gpu_page_pool_->free_pages();
        if (kvmem_mtp_gpu_page_pool_) {
            const uint32_t draft_free = kvmem_mtp_gpu_page_pool_->free_pages();
            free = std::min(free, draft_free > pages ? draft_free - pages : 0u);
        }
        action_capture_prefill_ = !prefill_attention_end_ ||
            base + count - prefill_attention_end_ >= prefill_attention_interval_ || free <= pages + 1;
    }
    action_capture_rows_ = 0;
    action_captured_ = false;
    action_captured_layers_ = 0;
    if (!action_running_ && !action_capture_prefill_) return;
    action_pending_base_ = base;
    action_capture_count_ = count;
    action_capture_first_row_ = action_capture_prefill_ ? count - 1 : 0;
    action_capture_row_step_ = 1;
    if (action_capture_prefill_) {
        action_capture_row_step_ = prefill_attention_stride_ ? prefill_attention_stride_ : count;
        const uint32_t samples = 1 + (count - 1) / action_capture_row_step_;
        if (samples > 64)
            throw std::invalid_argument("prefill attention exceeds 64 query samples per chunk; increase --kvmem-prefill-attention-stride or reduce --prefill-chunk");
        action_capture_first_row_ = count - 1 - (samples - 1) * action_capture_row_step_;
        action_capture_rows_ = samples == 64 ? ~uint64_t(0) : (uint64_t(1) << samples) - 1;
    }
    else {
        action_parser_.prepare(tokens, count);
        action_capture_rows_ = action_parser_.mask;
    }
    action_mass_host_.clear();
    if (!action_capture_rows_) return;
    const uint32_t samples = __builtin_popcountll(action_capture_rows_);
    const uint32_t window_base = kvmem_active_ ? window_query_pos_ : static_cast<uint32_t>(base);
    action_stride_ = window_base + count;
    action_starts_.clear(); action_counts_.clear(); action_history_.clear();
    // Freeze logical identities at capture time, independently of packed slots.
    if (kvmem_active_) for (uint32_t w = 0; w < bs_window_blocks_; ++w) {
        const auto &b = block_store_->blocks().at(bs_window_block_ids_[w]);
        if (block_store_->has_repository() &&
            (bs_window_repository_handles_.size() != bs_window_blocks_ ||
             bs_window_repository_handles_[w].generation != b.repository_generation))
            throw std::logic_error("action attention window identity changed");
        action_starts_.push_back(bs_win_base_host_[w]);
        action_counts_.push_back(bs_blk_tokens_host_[w]);
        action_history_.push_back(block_store_->has_repository() ? b.history_pos_start : b.orig_pos_start);
    }
    uint32_t cursor = action_starts_.empty() ? 0 : action_starts_.back() + action_counts_.back();
    const uint32_t bt = block_store_->config().block_tokens;
    // Newly appended Read results also need credit BEFORE the next selection.
    // Map the dense suffix to original history, including a partial first block.
    // With no selected window this maps the entire dense prompt the same way.
    if (uint64_t(base) + cursor < window_base)
        throw std::logic_error("invalid action attention suffix origin");
    uint64_t history = base + cursor - window_base;
    while (cursor < action_stride_) {
        const uint32_t n = std::min<uint32_t>(bt - history % bt, action_stride_ - cursor);
        action_starts_.push_back(cursor); action_counts_.push_back(n);
        action_history_.push_back(history - history % bt);
        cursor += n; history += n;
    }
    action_blocks_ = static_cast<uint32_t>(action_history_.size());
    if (action_blocks_ > action_meta_capacity_) {
        action_meta_capacity_ = ((action_blocks_ + 255) / 256) * 256;
        action_starts_dev_ = backend_.tensor_i32(action_meta_capacity_, "action block starts");
        action_counts_dev_ = backend_.tensor_i32(action_meta_capacity_, "action block lengths");
    }
    check_action(backend_.copy_i32_from_host(*action_starts_dev_, 0, action_starts_.data(), action_blocks_));
    check_action(backend_.copy_i32_from_host(*action_counts_dev_, 0, action_counts_.data(), action_blocks_));
    const uint64_t size = uint64_t(samples) * model_.config().n_heads * action_stride_;
    if (size > action_logits_capacity_) {
        action_logits_capacity_ = ((size + 262143) / 262144) * 262144;
        action_logits_ = backend_.tensor_f32(action_logits_capacity_, "sampled action logits");
    }
    const uint64_t mass_size = uint64_t(samples) * (action_blocks_ + 1);
    if (mass_size > action_mass_capacity_) {
        action_mass_capacity_ = ((mass_size + 4095) / 4096) * 4096;
        action_mass_ = backend_.tensor_f32(action_mass_capacity_, "sampled action block mass");
    }
}

void QwenExecutor::kvmem_action_layer_begin(uint32_t layer, uint32_t base, uint32_t count) {
    if ((!action_running_ && !action_capture_prefill_) || !std::binary_search(action_layers_.begin(), action_layers_.end(), int32_t(layer)) ||
        !action_capture_rows_ || !action_blocks_ ||
        kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_) return;
    if (base + count != action_stride_)
        throw std::logic_error("action attention requires an unsplit target verifier transaction");
    check_action(backend_.attention_capture_set(action_logits_.get(), action_capture_rows_,
                                               action_stride_, model_.config().n_heads, action_capture_first_row_, action_capture_row_step_));
    action_captured_ = true;
}

void QwenExecutor::kvmem_action_layer_end(uint32_t layer, uint32_t base) {
    if ((!action_running_ && !action_capture_prefill_) || !action_captured_ || !std::binary_search(action_layers_.begin(), action_layers_.end(), int32_t(layer)) ||
        kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_) return;
    check_action(backend_.attention_capture_set(nullptr, 0, 0, 0));
    check_action(backend_.attention_capture_reduce(*action_mass_, *action_logits_,
        action_capture_rows_, action_stride_, model_.config().n_heads, base + action_capture_first_row_,
        action_blocks_, *action_starts_dev_, *action_counts_dev_, 1.0f / action_layers_.size(),
        action_captured_layers_ != 0, action_capture_row_step_));
    ++action_captured_layers_;
}

void QwenExecutor::kvmem_action_commit(uint64_t target) {
    if ((!action_running_ && !action_capture_prefill_) || kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_ ||
        target <= action_pending_base_) return;
    const uint64_t count = target - action_pending_base_;
    if (action_capture_prefill_ && (target <= prefill_attention_end_ || count < action_capture_count_)) return;
    if (count > action_capture_count_)
        throw std::logic_error("action attention acceptance exceeds target verifier rows");
    const uint64_t accepted = action_capture_prefill_ ? action_capture_rows_ : action_parser_.commit(static_cast<uint32_t>(count));
    if (!accepted || !action_captured_) return;
    if (action_captured_layers_ != action_layers_.size())
        throw std::logic_error("incomplete multi-layer attention capture");
    const uint32_t width = action_blocks_ + 1;
    if (action_mass_host_.empty()) {
        action_mass_host_.resize(uint64_t(__builtin_popcountll(action_capture_rows_)) * width);
        check_action(backend_.copy_to_host(*action_mass_, action_mass_host_.data(), 0, action_mass_host_.size()));
    }
    std::unordered_map<uint64_t, detail::PrefillAttentionScore> next;
    if (action_capture_prefill_) {
        // One update/decay per accepted chunk, independent of sampling density.
        const double decay = std::exp2(-double(target - prefill_attention_end_) / prefill_attention_half_life_);
        next.reserve(action_blocks_);
        for (uint64_t key : action_history_) {
            auto [it, inserted] = next.try_emplace(key);
            if (!inserted) continue;
            const auto old = prefill_attention_scores_.find(key);
            if (old != prefill_attention_scores_.end()) it->second = old->second;
            it->second.mass *= decay; it->second.exposure *= decay;
        }
    }
    for (uint32_t row = 0; row < 64; ++row) if ((accepted >> row) & 1) {
        const uint32_t slot = __builtin_popcountll(action_capture_rows_ & ((uint64_t(1) << row)-1));
        double sum = 0;
        for (uint32_t b = 0; b < width; ++b) {
            const float x = action_mass_host_[slot * width + b];
            if (!std::isfinite(x) || x < 0) throw std::runtime_error("invalid captured action attention mass");
            sum += x;
        }
        if (std::abs(sum - 1) > 0.005) throw std::runtime_error("action attention mass does not sum to one");
        if (action_capture_prefill_) {
            const uint32_t visible = action_stride_ - action_capture_count_ +
                action_capture_first_row_ + row * action_capture_row_step_ + 1;
            const double inv = 1.0 / __builtin_popcountll(accepted);
            std::unordered_set<uint64_t> exposed;
            for (uint32_t w = 0; w < action_blocks_; ++w) {
                if (uint32_t(action_starts_[w]) >= visible) continue;
                const uint64_t key = action_history_[w];
                // A partial selected block and its appended suffix can share
                // one historical identity; combine mass before counting exposure.
                auto &score = next.at(key);
                score.mass += inv * action_mass_host_[slot * width + w];
                if (exposed.insert(key).second) score.exposure += inv;
            }
            ++prefill_attention_samples_;
            continue;
        }
        for (uint32_t w = 0; w < action_blocks_; ++w) {
            const float x = action_mass_host_[slot * width + w];
            if (x > 0) action_round_mass_[action_history_[w]] += x;
        }
        ++action_round_samples_;
    }
    if (action_capture_prefill_) {
        prefill_attention_scores_.swap(next); // bound state to the observed window
        prefill_attention_end_ = target;
        if (std::getenv("QW3_KVMEM_TRACE"))
            std::fprintf(stderr, "[prefill-attention] end=%llu first_row=%u row_step=%u query_samples=%u layers=%zu blocks=%zu samples=%llu\n",
                static_cast<unsigned long long>(target), action_capture_first_row_, action_capture_row_step_,
                __builtin_popcountll(accepted), action_layers_.size(), prefill_attention_scores_.size(),
                static_cast<unsigned long long>(prefill_attention_samples_));
    }
}

void QwenExecutor::kvmem_action_end() {
    if (!action_running_) return;
    action_running_ = false;
    action_captured_ = false;
    action_capture_rows_ = 0;
    check_action(backend_.attention_capture_set(nullptr, 0, 0, 0));
    if (action_round_samples_ && block_store_) {
        std::vector<double> mean(block_store_->block_count(), 0);
        for (const auto &b : block_store_->blocks()) {
            const uint64_t key = block_store_->has_repository() ? b.history_pos_start : b.orig_pos_start;
            const auto it = action_round_mass_.find(key);
            if (it != action_round_mass_.end()) mean[b.block_id] = it->second / action_round_samples_;
        }
        block_store_->update_action_scores(mean, action_decay_);
    }
    // Keep action_tokens for existing telemetry consumers; scope identifies
    // whether this counts tool-call content or all non-thinking output.
    std::fprintf(stderr, "[action-attention] committed_samples=%llu action_tokens=%llu historical_blocks=%zu scope=%s\n",
        static_cast<unsigned long long>(action_round_samples_),
        static_cast<unsigned long long>(action_parser_.committed.tokens), action_round_mass_.size(),
        action_parser_.scope_name());
    action_parser_.candidate.clear(); action_parser_.mask = 0;
}
}
