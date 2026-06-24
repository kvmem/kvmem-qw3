#include "qw3/kvmem_store.hpp"

#include <algorithm>

namespace qw3 {

void KvMemStore::register_append(uint32_t n_new_tokens) {
    if (n_new_tokens == 0) return;
    const uint32_t bt = cfg_.block_tokens;

    // Extend the trailing partial block first, then spill into new blocks.
    uint32_t remaining = n_new_tokens;
    if (!blocks_.empty()) {
        KvMemBlock &last = blocks_.back();
        if (last.n_tokens < bt) {
            const uint32_t room = bt - last.n_tokens;
            const uint32_t take = std::min(room, remaining);
            last.n_tokens += take;
            remaining -= take;
            total_tokens_ += take;
        }
    }
    while (remaining > 0) {
        KvMemBlock b;
        b.block_id = static_cast<uint32_t>(blocks_.size());
        b.orig_pos_start = total_tokens_;
        b.n_tokens = std::min(bt, remaining);
        b.baked_pos = static_cast<int64_t>(b.orig_pos_start);  // baked at true pos after prefill
        blocks_.push_back(b);
        total_tokens_ += b.n_tokens;
        remaining -= b.n_tokens;
    }
}

void KvMemStore::accumulate_attn(const std::vector<double> &scores) {
    const uint32_t n = std::min<uint32_t>(static_cast<uint32_t>(scores.size()),
                                          block_count());
    for (uint32_t i = 0; i < n; ++i) {
        blocks_[i].profile_score += scores[i];
        blocks_[i].attn_score = blocks_[i].profile_score;
    }
}

void KvMemStore::set_attn_scores(const std::vector<double> &scores) {
    set_retrieval_scores(scores);
}

void KvMemStore::set_retrieval_scores(const std::vector<double> &scores) {
    const uint32_t n = block_count();
    for (uint32_t i = 0; i < n; ++i) {
        blocks_[i].retrieval_score = (i < scores.size()) ? scores[i] : 0.0;
        blocks_[i].attn_score = blocks_[i].retrieval_score;
    }
}

void KvMemStore::set_block_tier(uint32_t block_id, KvTier tier,
                                int32_t cpu_slot, int32_t nvme_slot) {
    if (block_id >= block_count()) return;
    KvMemBlock &b = blocks_[block_id];
    b.tier = tier;
    if (tier == KvTier::GPU) {
        b.cpu_slot = -1;
        b.nvme_slot = -1;
    } else {
        b.gpu_slot = -1;
        b.cpu_slot = cpu_slot;
        b.nvme_slot = nvme_slot;
    }
}

std::vector<uint32_t> KvMemStore::pick_topk_blocks() const {
    const uint32_t n = block_count();
    std::vector<uint32_t> selected;
    if (n == 0) return selected;

    const uint32_t budget = budget_blocks();
    if (budget == 0 || n <= budget) {
        // Everything fits: select all in order.
        selected.reserve(n);
        for (uint32_t i = 0; i < n; ++i) selected.push_back(i);
        return selected;
    }

    // Always-keep windows: first `sink_blocks` and last `recent_blocks`.
    const uint32_t sink = std::min(cfg_.sink_blocks, n);
    uint32_t recent = cfg_.recent_blocks;
    if (recent == 0) {
        // Derive a default recent window: a quarter of the budget, at least 1.
        recent = std::max<uint32_t>(1, budget / 4);
    }
    recent = std::min(recent, n);

    std::vector<bool> kept(n, false);
    uint32_t kept_count = 0;
    auto keep = [&](uint32_t id) {
        if (id < n && !kept[id]) { kept[id] = true; ++kept_count; }
    };
    for (uint32_t i = 0; i < sink && kept_count < budget; ++i) keep(i);
    for (uint32_t i = 0; i < recent && kept_count < budget; ++i) {
        keep(n - 1 - i);
    }

    auto take_top = [&](uint32_t quota, auto score_fn) {
        if (kept_count >= budget || quota == 0) return;
        std::vector<uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!kept[i]) candidates.push_back(i);
        }
        if (candidates.empty()) return;
        const uint32_t need = std::min<uint32_t>(
            std::min<uint32_t>(quota, budget - kept_count),
            static_cast<uint32_t>(candidates.size()));
        auto better = [&](uint32_t a, uint32_t b) {
            const double sa = score_fn(blocks_[a]);
            const double sb = score_fn(blocks_[b]);
            if (sa != sb) return sa > sb;
            return a > b;
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need,
                             candidates.end(), better);
            candidates.resize(need);
        }
        for (uint32_t id : candidates) keep(id);
    };

    if (cfg_.select_policy == KvMemSelectPolicy::Quota && kept_count < budget) {
        uint32_t remaining = budget - kept_count;
        uint32_t retrieval_quota = cfg_.retrieval_blocks;
        uint32_t profile_quota = cfg_.profile_blocks;
        if (retrieval_quota == 0 && profile_quota == 0) {
            retrieval_quota = (remaining * 2) / 3;
            profile_quota = remaining - retrieval_quota;
        } else if (retrieval_quota == 0) {
            retrieval_quota = remaining > profile_quota ? remaining - profile_quota : 0;
        } else if (profile_quota == 0) {
            profile_quota = remaining > retrieval_quota ? remaining - retrieval_quota : 0;
        }

        take_top(retrieval_quota, [](const KvMemBlock &b) {
            return b.retrieval_score;
        });
        take_top(profile_quota, [](const KvMemBlock &b) {
            return b.profile_score;
        });
        // Fill any leftover quota with the configured method's combined score
        // so rounding, overlap, or zero explicit quotas still use the budget.
        if (kept_count < budget) {
            take_top(budget - kept_count, [](const KvMemBlock &b) {
                return b.attn_score;
            });
        }
        selected.reserve(kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (kept[i]) selected.push_back(i);
        }
        return selected;
    }

    // Fill the rest with the highest cumulative-attention middle blocks.
    if (kept_count < budget) {
        std::vector<uint32_t> candidates;
        candidates.reserve(n - kept_count);
        for (uint32_t i = 0; i < n; ++i) {
            if (!kept[i]) candidates.push_back(i);
        }
        const uint32_t need = budget - kept_count;
        // Partial sort by attn_score desc; tie-break by recency (higher id) so
        // selection is deterministic.
        auto better = [&](uint32_t a, uint32_t b) {
            if (blocks_[a].attn_score != blocks_[b].attn_score) {
                return blocks_[a].attn_score > blocks_[b].attn_score;
            }
            return a > b;
        };
        if (need < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + need,
                             candidates.end(), better);
            candidates.resize(need);
        }
        for (uint32_t id : candidates) keep(id);
    }

    selected.reserve(kept_count);
    for (uint32_t i = 0; i < n; ++i) {
        if (kept[i]) selected.push_back(i);
    }
    return selected;
}

KvMemPlan KvMemStore::set_selection(std::vector<uint32_t> selected_ids) {
    // Sort + dedupe so window order is deterministic (ascending block_id =
    // original chronological order: sink first ... recent last).
    std::sort(selected_ids.begin(), selected_ids.end());
    selected_ids.erase(std::unique(selected_ids.begin(), selected_ids.end()),
                       selected_ids.end());
    // Drop out-of-range IDs defensively (external selector could be stale).
    selected_ids.erase(
        std::remove_if(selected_ids.begin(), selected_ids.end(),
                       [&](uint32_t id) { return id >= block_count(); }),
        selected_ids.end());

    KvMemPlan plan;

    std::vector<bool> now_selected(block_count(), false);
    for (uint32_t id : selected_ids) now_selected[id] = true;

    // Stage-out: any GPU-resident block that is not selected. On the first
    // post-prefill selection, many cold blocks were never in the prior working
    // set, but they still occupy GPU KV pages and must be eligible for offload.
    // Their cache pages keep whatever bake they currently hold (baked_pos
    // unchanged); a future re-selection will de-rotate from there.
    for (auto &b : blocks_) {
        if (b.tier == KvTier::GPU && !now_selected[b.block_id]) {
            plan.stage_out.push_back(b.block_id);
            b.in_working_set = false;
        }
    }

    // Pack selected blocks contiguously into the window in ascending order.
    uint32_t window_pos = 0;
    for (uint32_t id : selected_ids) {
        KvMemBlock &b = blocks_[id];
        if (!b.in_working_set) plan.stage_in.push_back(id);

        KvMemRemap rm;
        rm.block_id = id;
        rm.n_tokens = b.n_tokens;
        rm.from_base = static_cast<int32_t>(b.baked_pos);   // de-rotate source
        rm.to_base = static_cast<int32_t>(window_pos);      // new window slot
        rm.skip = (b.baked_pos == static_cast<int64_t>(window_pos));
        plan.remaps.push_back(rm);

        b.in_working_set = true;
        b.baked_pos = static_cast<int64_t>(window_pos);
        window_pos += b.n_tokens;
    }
    plan.total_window_tokens = window_pos;
    return plan;
}

void KvMemStore::clear_working_set() {
    for (auto &b : blocks_) {
        b.in_working_set = false;
    }
}

} // namespace qw3
