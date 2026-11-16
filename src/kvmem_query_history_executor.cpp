#include "qwen_executor.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace qw3 {

void QwenExecutor::kvmem_query_history_begin(detail::KvMemQueryHistory &history,
        uint64_t input_end, uint64_t verified_prefix, double decay) {
    if (!kvmem_enabled_ || !block_store_ || !block_store_->config().query_history)
        return;
    if (kvmem_query_history_)
        throw std::logic_error("KVMem query history already has an open input request");
    history.begin(input_end, verified_prefix, decay);
    kvmem_query_history_ = &history;
    kvmem_query_history_complete_ = false;
    kvmem_query_history_blocks_.clear();
    kvmem_query_history_selection_.clear();
}

void QwenExecutor::kvmem_query_history_abort() noexcept {
    if (kvmem_query_history_) kvmem_query_history_->abort();
    kvmem_query_history_ = nullptr;
    kvmem_query_history_complete_ = false;
    kvmem_query_history_blocks_.clear();
    kvmem_query_history_selection_.clear();
}

void QwenExecutor::kvmem_query_history_observe(const std::vector<double> &scores,
                                             uint32_t scored_blocks) {
    if (!kvmem_query_history_ || !kvmem_query_history_->open() ||
        kvmem_query_probe_guard_active() || kvmem_provisional_prefill_ ||
        kvmem_blend_active_ || !kvmem_query_capture_complete())
        return;
    if (scores.size() != block_store_->block_count())
        throw std::logic_error("KVMem query history score domain does not match blocks");

    detail::KvMemQueryHistory::Scores raw;
    std::vector<QueryHistoryBlockIdentity> identities(scores.size());
    const bool repository = block_store_->has_repository();
    const auto &frame = kvmem_repository_score_frame_;
    if (repository && frame.enabled.size() != scores.size())
        throw std::logic_error("KVMem query history has no leased score frame");
    // Physical slot identities are needed only between publication and selection.
    // The persistent history is keyed by logical positions in a verified input.
    for (const auto &block : block_store_->blocks()) {
        if (!block_store_->is_live(block.block_id)) continue;
        const uint64_t start = repository ? block.history_pos_start : block.orig_pos_start;
        identities[block.block_id] = {start, block.repository_generation, block.n_tokens, true};
        if (block.block_id >= scored_blocks ||
            (repository && !frame.enabled[block.block_id])) continue;
        // A trailing block that includes unaccepted/private/header-only suffix
        // bytes must not teach the next input about those bytes.
        if (start >= kvmem_query_history_->input_end() ||
            block.n_tokens > kvmem_query_history_->input_end() - start) continue;
        raw.emplace(start, scores[block.block_id]);
    }
    if (repository) {
        for (const auto &lease : frame.leases) {
            const auto handle = lease.handle();
            if (handle.slot >= identities.size() ||
                !identities[handle.slot].live ||
                identities[handle.slot].generation != handle.generation)
                throw std::logic_error("KVMem query history score frame was recycled");
        }
    }
    kvmem_query_history_->preview(raw);
    kvmem_query_history_blocks_ = std::move(identities);
    kvmem_query_history_complete_ = true;
}

const KvMemStore::QuerySelectionScores *QwenExecutor::kvmem_query_history_selection() {
    if (!kvmem_query_history_ || !kvmem_query_history_complete_ ||
        kvmem_query_history_->scores().empty() || kvmem_query_probe_guard_active() ||
        kvmem_provisional_prefill_ || kvmem_blend_active_)
        return nullptr;
    const auto &scores = kvmem_query_history_->scores();
    kvmem_query_history_selection_.assign(block_store_->block_count(), 0);
    for (const auto &block : block_store_->blocks()) {
        if (!block_store_->is_live(block.block_id) ||
            block.block_id >= kvmem_query_history_blocks_.size()) continue;
        const auto &identity = kvmem_query_history_blocks_[block.block_id];
        if (!identity.live) continue;
        const uint64_t start = block_store_->has_repository()
            ? block.history_pos_start : block.orig_pos_start;
        if (start != identity.start || block.repository_generation != identity.generation ||
            block.n_tokens != identity.tokens)
            throw std::logic_error("KVMem query history selection identity changed after scoring");
        const auto found = scores.find(start);
        if (found != scores.end()) kvmem_query_history_selection_[block.block_id] = found->second;
    }
    return &kvmem_query_history_selection_;
}

void QwenExecutor::kvmem_query_history_finalize_input() {
    if (!kvmem_query_history_) return;
    if (kvmem_query_probe_guard_active() || kvmem_query_replay_active_ ||
        kvmem_provisional_prefill_ || kvmem_blend_active_ || kvmem_pending_reselect_)
        throw std::logic_error("KVMem query history cannot commit a provisional input");
    if (!kvmem_query_history_complete_ && kvmem_query_end_ > kvmem_query_begin_) {
        if (!kvmem_query_capture_complete())
            throw std::runtime_error("KVMem query history input ended with incomplete Query capture");
        // Below budget Query/index capture is prepared but ordinary selection
        // may not score. Learn from that input without installing a new window.
        std::string failure;
        if (!kvmem_retrieval_score_mean_softmax(-1, &failure) ||
            !kvmem_query_history_complete_)
            throw std::runtime_error("KVMem query history input scoring failed: " + failure);
        kvmem_repository_score_frame_ = {};
    }
    if (!kvmem_query_history_complete_) kvmem_query_history_->preview({});
    const bool observed = kvmem_query_history_->has_observation();
    kvmem_query_history_->commit();
    if (std::getenv("QW3_KVMEM_TRACE"))
        std::fprintf(stderr, "[query-history] committed_round=%llu input_end=%llu observed=%d historical_blocks=%zu normalization=positive-l1\n",
            static_cast<unsigned long long>(kvmem_query_history_->round_count()),
            static_cast<unsigned long long>(kvmem_query_history_->input_end()), observed ? 1 : 0,
            kvmem_query_history_->committed_scores().size());
    // Decode-time guided queries retain their existing instantaneous ranking.
    // Neither verifier replay nor a refresh can add another input observation.
    kvmem_query_history_ = nullptr;
    kvmem_query_history_complete_ = false;
    kvmem_query_history_blocks_.clear();
    kvmem_query_history_selection_.clear();
}

} // namespace qw3
