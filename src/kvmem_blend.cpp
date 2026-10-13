#include "qwen_executor.hpp"
#include "kvmem_blend_policy.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

namespace qw3 {
namespace {
void check_blend_status(DeviceStatus status) {
    if (!status.ok) throw std::runtime_error(status.message);
}
}

void QwenExecutor::kvmem_blend_capture_tokens(
        uint64_t base, const uint32_t *tokens, uint32_t count) {
    if (!kvmem_enabled_ || kvmem_blend_ratio_ == 0 || !block_store_ ||
        kvmem_blend_active_ || kvmem_provisional_prefill_ ||
        kvmem_query_probe_guard_active()) return;
    const uint32_t bt = block_store_->config().block_tokens;
    if (count == 0 && block_store_->has_repository()) {
        std::unordered_set<uint64_t> live;
        for (const auto &block : block_store_->blocks())
            if (block.n_tokens) live.insert(block.history_pos_start);
        for (auto it = kvmem_blend_tokens_.begin(); it != kvmem_blend_tokens_.end();) {
            if (!live.count(it->first)) it = kvmem_blend_tokens_.erase(it);
            else ++it;
        }
        return;
    }
    for (uint32_t offset = 0; offset < count;) {
        const uint64_t pos = base + offset;
        const uint32_t row = pos % bt;
        const uint32_t n = std::min(bt - row, count - offset);
        auto &block = kvmem_blend_tokens_[pos - row];
        if (block.size() < row + n) block.resize(row + n, UINT32_MAX);
        std::copy_n(tokens + offset, n, block.begin() + row);
        offset += n;
    }
}

void QwenExecutor::kvmem_blend_restore() {
    for (const auto &saved : kvmem_blend_backup_) {
        for (const auto &piece : saved.pieces) {
            check_blend_status(backend_.copy_d2d_into(
                k_cache(saved.layer), piece.offset, *saved.k, piece.packed, piece.count));
            check_blend_status(backend_.copy_d2d_into(
                v_cache(saved.layer), piece.offset, *saved.v, piece.packed, piece.count));
        }
    }
    kvmem_blend_backup_.clear();
}

void QwenExecutor::kvmem_blend_selected() {
    kvmem_blend_pending_ = false;
    if (kvmem_blend_ratio_ == 0 || !kvmem_active_ || !block_store_ ||
        !kvmem_query_replay_active_ || bs_window_block_ids_.empty()) return;
    double alpha = 1;
    if (const char *value = std::getenv("QW3_KVMEM_BLEND_ALPHA")) {
        char *end = nullptr;
        alpha = std::strtod(value, &end);
        if (end == value || *end || !std::isfinite(alpha) || alpha < 0 || alpha > 1)
            throw std::invalid_argument("QW3_KVMEM_BLEND_ALPHA must be in [0,1]");
    }
    // M-RoPE input replacements and quantized block layouts need a separate
    // sparse-input contract; fail explicitly instead of silently recomputing
    // image markers or treating Q8 block bytes as scalar elements.
    if (!input_mrope_positions_.empty())
        throw std::runtime_error("KVMem blend currently requires text-only inputs");
    std::vector<uint32_t> layers;
    for (uint32_t layer = 0; layer < weights_.n_layers(); ++layer) {
        if (weights_.layer(layer).recurrent) continue;
        layers.push_back(layer);
        if (k_cache(layer).dtype == DeviceTensorDType::Q8_KV)
            throw std::runtime_error("KVMem blend supports FP16/BF16/FP32/FP8 KV, not Q8_KV");
    }
    const auto start = std::chrono::steady_clock::now();
    const bool bounded = block_store_->has_repository();
    std::vector<detail::KvMemBlendSpan> candidates;
    std::unordered_set<uint64_t> live;
    for (const auto &block : block_store_->blocks()) {
        if (block.n_tokens != 0)
            live.insert(bounded ? block.history_pos_start : block.orig_pos_start);
    }
    // Token retention follows repository liveness, including slot reuse.
    for (auto it = kvmem_blend_tokens_.begin(); it != kvmem_blend_tokens_.end();) {
        if (!live.count(it->first)) it = kvmem_blend_tokens_.erase(it);
        else ++it;
    }
    uint32_t missing = 0;
    bool changed_prefix = false;
    for (size_t i = 0; i < bs_window_block_ids_.size(); ++i) {
        const auto &block = block_store_->blocks().at(bs_window_block_ids_[i]);
        const uint64_t history = bounded ? block.history_pos_start : block.orig_pos_start;
        const uint32_t window = bs_win_base_host_.at(i);
        changed_prefix |= history != window;
        // An unchanged prefix already has correct causal KV. Query/suffix rows
        // are recomputed by normal replay, never charged to the blend budget.
        if (!changed_prefix || history >= position_ || block.n_tokens > position_ - history)
            continue;
        const auto found = kvmem_blend_tokens_.find(history);
        if (found == kvmem_blend_tokens_.end() || found->second.size() < block.n_tokens ||
            std::find(found->second.begin(), found->second.begin() + block.n_tokens,
                      UINT32_MAX) != found->second.begin() + block.n_tokens) {
            missing += block.n_tokens;
            continue;
        }
        candidates.push_back({history, window, block.n_tokens, block.retrieval_score});
    }
    auto selected = detail::kvmem_blend_plan(
        std::move(candidates), window_query_pos_, kvmem_blend_ratio_, kvmem_blend_max_tokens_);
    if (selected.empty()) {
        if (missing) std::fprintf(stderr, "[kvmem-blend] tokens=0 missing_source_tokens=%u\n", missing);
        return;
    }
    kvmem_blend_restore();
    // Retire any earlier D2H reads before temporarily patching their source.
    kvmem_finish_proactive_d2h(true);
    kvmem_reap_pending_writes(true);
    const uint32_t page_size = kv_pages_.page_size;
    const uint32_t per_pos = model_.config().n_kv_heads * model_.config().head_dim;
    uint32_t refreshed = 0;
    std::vector<KvMemBlendBackup::Piece> pieces;
    // Back up only edited physical rows, with the original dtype (no roundoff).
    // Allocate all backups before the first mutation for exception safety.
    for (const auto &span : selected) {
        for (uint32_t off = 0; off < span.tokens;) {
            const uint32_t local = span.window + off;
            const uint32_t n = std::min(page_size - local % page_size, span.tokens - off);
            const int32_t page = window_pages_host_.at(local / page_size);
            if (page < 0) throw std::runtime_error("KVMem blend encountered an absent page");
            const uint64_t offset = (uint64_t(page) * page_size + local % page_size) * per_pos;
            pieces.push_back({offset, uint64_t(refreshed + off) * per_pos, uint64_t(n) * per_pos});
            off += n;
        }
        refreshed += span.tokens;
    }
    // One compact backup per layer/dtype, rather than an allocation per page.
    const auto allocate = [&](const DeviceTensor &source, const char *label) {
        const uint64_t count = uint64_t(refreshed) * per_pos;
        switch (source.dtype) {
        case DeviceTensorDType::FP8_E4M3: return backend_.tensor_fp8_kv(count, label);
        case DeviceTensorDType::F16: return backend_.tensor_f16(count, label);
        default: return backend_.scratch_like(source, count, label);
        }
    };
    for (uint32_t layer : layers) {
        KvMemBlendBackup saved;
        saved.layer = layer;
        saved.pieces = pieces;
        saved.k = allocate(k_cache(layer), "blend_source_k");
        saved.v = allocate(v_cache(layer), "blend_source_v");
        if (!saved.k || !saved.v) throw std::runtime_error("KVMem blend backup allocation failed");
        for (const auto &piece : pieces) {
            check_blend_status(backend_.copy_d2d_into(*saved.k, piece.packed, k_cache(layer), piece.offset, piece.count));
            check_blend_status(backend_.copy_d2d_into(*saved.v, piece.packed, v_cache(layer), piece.offset, piece.count));
        }
        kvmem_blend_backup_.push_back(std::move(saved));
    }
    std::unique_ptr<DeviceTensor> mix_rows;
    std::vector<int32_t> physical_rows;
    if (alpha != 1) {
        physical_rows.reserve(refreshed);
        for (const auto &piece : pieces) {
            for (uint64_t offset = 0; offset < piece.count; offset += per_pos) {
                const uint64_t row = (piece.offset + offset) / per_pos;
                if (row > INT32_MAX) throw std::runtime_error("KVMem blend row index exceeds I32");
                physical_rows.push_back(static_cast<int32_t>(row));
            }
        }
        mix_rows = backend_.tensor_i32(refreshed, "blend_mix_rows");
        if (!mix_rows) throw std::runtime_error("KVMem blend row allocation failed");
        check_blend_status(backend_.copy_i32_from_host(*mix_rows, 0, physical_rows.data(), refreshed));
    }
    StateSnapshot boundary;
    capture_transient_state(boundary);
    const uint64_t saved_logical = last_forward_logical_base_;
    const uint32_t saved_rope = last_forward_rope_base_, saved_rows = last_forward_rows_;
    const bool saved_prefix_ready = g_provisional_prefix_replay_ready_;
    g_provisional_prefix_replay_ready_ = false;
    const auto restore_compute = [&] {
        kvmem_blend_active_ = false;
        position_ = boundary.position;
        window_query_pos_ = boundary.window_query_pos;
        last_forward_logical_base_ = saved_logical;
        last_forward_rope_base_ = saved_rope;
        last_forward_rows_ = saved_rows;
        g_provisional_prefix_replay_ready_ = saved_prefix_ready;
        restore_recurrent_state(boundary);
        if (boundary.h) check_blend_status(backend_.copy_d2d(*h_, *boundary.h, 0, h_->count));
    };
    uint32_t runs = 0;
    try {
        // Hybrid approximation: recurrent layers see the selected text in
        // chronological order. Starting at zero avoids leaking the later query
        // state into historical tokens. Restore the exact query-boundary state
        // afterwards. This is not full-prefill equivalence for DeltaNet.
        for (auto &state : recurrent_states_) if (state) check_blend_status(backend_.zero_tensor(*state));
        for (auto &state : conv_states_) if (state) check_blend_status(backend_.zero_tensor(*state));
        kvmem_blend_active_ = true;
        for (size_t i = 0; i < selected.size();) {
            const auto first = selected[i];
            std::vector<uint32_t> tokens;
            uint32_t end = first.window;
            do {
                const auto &span = selected[i];
                const auto &source = kvmem_blend_tokens_.at(span.history);
                tokens.insert(tokens.end(), source.begin(), source.begin() + span.tokens);
                end += span.tokens;
                ++i;
            } while (i < selected.size() && selected[i].window == end &&
                     selected[i].history == first.history + tokens.size());
            position_ = first.history;
            window_query_pos_ = first.window;
            if (!forward_n_tokens(tokens, false).ok)
                throw std::runtime_error("KVMem blend prefill failed");
            ++runs;
        }
        // Mix only after every original recomputation has finished. Earlier
        // segments therefore retain the original alpha=1 recompute semantics.
        // The backup is already in the assembled window's RoPE coordinates.
        if (mix_rows) {
            for (const auto &saved : kvmem_blend_backup_)
                check_blend_status(backend_.mix_kv_rows(
                    k_cache(saved.layer), v_cache(saved.layer), *saved.k, *saved.v,
                    *mix_rows, refreshed, per_pos, static_cast<float>(alpha)));
            check_blend_status(backend_.synchronize());
        }
        restore_compute();
    } catch (...) {
        restore_compute();
        kvmem_blend_restore();
        throw;
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr,
        "[kvmem-blend] tokens=%u window_tokens=%u blocks=%zu runs=%u "
        "ratio=%.6f max_tokens=%u missing_source_tokens=%u ms=%.3f alpha=%.6f "
        "recurrent=selected-text query_state=restored source_kv=preserved\n",
        refreshed, window_query_pos_, selected.size(), runs, kvmem_blend_ratio_,
        kvmem_blend_max_tokens_, missing, ms, alpha);
}
}  // namespace qw3
