#pragma once

#include "kvmem_refresh_policy.hpp"
#include "qw3/qw3.hpp"

#include <cstdint>
#include <algorithm>
#include <vector>

namespace qw3::detail {

struct KvmemRequestPlanInput {
    bool enabled = false;
    bool reselect = false;
    bool private_query = false;
    bool keep_selected = false;
    uint64_t logical_prompt_tokens = 0;
    uint64_t request_prompt_tokens = 0;
    uint64_t context_limit_tokens = 0;
    uint64_t selection_budget_tokens = 0;
    uint64_t generation_budget_tokens = 0;
    uint64_t refresh_trigger_tokens = 0;
    uint32_t requested_output_tokens = 0;
    uint32_t block_tokens = 0;
    uint32_t raw_mandatory_blocks = 0;
    uint32_t mandatory_blocks = 0;
    uint32_t raw_live_suffix_blocks = 0;
    uint32_t live_suffix_blocks = 0;
    uint32_t soft_retrievable_blocks = 0;
    uint32_t sink_blocks = 0;
    uint32_t retrieval_reserve_blocks = 0;
    bool capacity_fitted = false;
    bool pressure_ingest = false;
};

struct KvmemMandatoryBlockSpan {
    uint32_t begin = 0;
    uint32_t end = 0;
    uint8_t priority = 0;
};

struct KvmemMandatoryFitInput {
    uint32_t prompt_blocks = 0;
    uint32_t budget_blocks = 0;
    uint32_t sink_blocks = 0;
    uint32_t recent_blocks = 0;
    std::vector<KvmemMandatoryBlockSpan> hard_spans;
    uint32_t live_begin = 0;
    uint32_t live_end = 0;
};

struct KvmemMandatoryFitResult {
    std::vector<uint32_t> hard_blocks;
    uint32_t raw_mandatory_blocks = 0;
    uint32_t raw_live_blocks = 0;
    uint32_t fitted_live_blocks = 0;
    uint32_t soft_retrievable_blocks = 0;
    uint32_t retrieval_reserve_blocks = 0;
    uint32_t replay_begin_block = 0;
    bool capacity_fitted = false;
};

// Fit protocol continuity into a finite active window without dropping source
// history. A long live tool body is never an all-or-nothing pin: its envelope
// head and a bounded contiguous tail remain hard, while the complete middle is
// left in the ordinary content index for query-conditioned retrieval.
inline KvmemMandatoryFitResult kvmem_fit_mandatory_blocks(
        const KvmemMandatoryFitInput &in) {
    KvmemMandatoryFitResult out;
    if (in.prompt_blocks == 0 || in.budget_blocks == 0) return out;
    const uint32_t prompt_blocks = in.prompt_blocks;
    const uint32_t live_begin = std::min(in.live_begin, prompt_blocks);
    const uint32_t live_end = std::min(in.live_end, prompt_blocks);
    const bool has_live = live_end > live_begin;
    out.raw_live_blocks = has_live ? live_end - live_begin : 0;

    std::vector<uint8_t> raw(prompt_blocks, 0);
    const auto mark_range = [&](std::vector<uint8_t> &marks,
                                uint32_t begin, uint32_t end) {
        begin = std::min(begin, prompt_blocks);
        end = std::min(end, prompt_blocks);
        for (uint32_t id = begin; id < end; ++id) marks[id] = 1;
    };
    mark_range(raw, 0, std::min(in.sink_blocks, prompt_blocks));
    for (const auto &span : in.hard_spans) {
        mark_range(raw, span.begin, span.end);
    }
    if (has_live) mark_range(raw, live_begin, live_end);
    out.raw_mandatory_blocks = static_cast<uint32_t>(std::count(
        raw.begin(), raw.end(), static_cast<uint8_t>(1)));
    if (out.raw_mandatory_blocks <= in.budget_blocks) {
        out.hard_blocks.reserve(out.raw_mandatory_blocks);
        for (uint32_t id = 0; id < prompt_blocks; ++id) {
            if (raw[id]) out.hard_blocks.push_back(id);
        }
        out.fitted_live_blocks = out.raw_live_blocks;
        out.replay_begin_block = has_live ? live_begin : live_end;
        return out;
    }

    out.capacity_fitted = true;
    std::vector<uint8_t> hard(prompt_blocks, 0);
    const uint32_t sink = std::min({
        in.sink_blocks, prompt_blocks, in.budget_blocks});
    // Preserve some semantic capacity whenever the fixed sink does not already
    // consume the complete window. This is a retrieval reserve, not discarded
    // context: all non-hard blocks remain candidates in the full content index.
    const uint32_t max_reserve = in.budget_blocks / 8;
    const uint32_t reserve = max_reserve;
    const uint32_t hard_limit = in.budget_blocks - reserve;
    out.retrieval_reserve_blocks = reserve;
    uint32_t hard_count = 0;
    auto add_block = [&](uint32_t id) {
        if (id >= prompt_blocks || hard[id] || hard_count >= hard_limit) {
            return false;
        }
        hard[id] = 1;
        ++hard_count;
        return true;
    };
    // Live protocol edges take precedence over the body. Keep up to 16 head
    // blocks (512 tokens at the production block size) for the call/result
    // envelope and a bounded contiguous recent tail, so replay never
    // reintroduces the complete oversized result after selection.
    if (has_live) {
        const uint32_t live_head = std::min<uint32_t>(
            16, out.raw_live_blocks);
        const uint32_t desired_tail = std::min(
            out.raw_live_blocks,
            std::max<uint32_t>(8, in.recent_blocks));
        const uint32_t tail_begin = live_end - desired_tail;
        // The newest complete protocol suffix is the minimum useful recovery
        // unit, so allocate it before an accidentally oversized sink setting.
        for (uint32_t id = tail_begin;
             id < live_end && hard_count < hard_limit; ++id) {
            add_block(id);
        }
        for (uint32_t i = 0; i < live_head; ++i) {
            add_block(live_begin + i);
        }
    }
    for (uint32_t id = 0; id < sink && hard_count < hard_limit; ++id) {
        add_block(id);
    }

    std::vector<KvmemMandatoryBlockSpan> spans = in.hard_spans;
    std::stable_sort(spans.begin(), spans.end(),
        [](const auto &a, const auto &b) {
            if (a.priority != b.priority) return a.priority < b.priority;
            if (a.begin != b.begin) return a.begin < b.begin;
            return a.end < b.end;
        });
    // First retain the structural edges of every bounded semantic lifetime.
    for (const auto &span : spans) {
        const uint32_t begin = std::min(span.begin, prompt_blocks);
        const uint32_t end = std::min(span.end, prompt_blocks);
        if (end <= begin) continue;
        const uint32_t anchors = std::min<uint32_t>(4, end - begin);
        for (uint32_t i = 0; i < anchors; ++i) add_block(begin + i);
        for (uint32_t i = 0; i < anchors; ++i) add_block(end - 1 - i);
    }
    // Round-robin within each priority tier prevents one very large system or
    // policy span from starving the current query and other protocol regions.
    for (uint32_t priority = 0; priority <= 3 && hard_count < hard_limit;
         ++priority) {
        std::vector<uint32_t> cursors(spans.size(), 0);
        bool progress = true;
        while (progress && hard_count < hard_limit) {
            progress = false;
            for (size_t i = 0; i < spans.size() && hard_count < hard_limit;
                 ++i) {
                if (spans[i].priority != priority) continue;
                const uint32_t begin = std::min(spans[i].begin, prompt_blocks);
                const uint32_t end = std::min(spans[i].end, prompt_blocks);
                while (begin + cursors[i] < end &&
                       hard[begin + cursors[i]]) {
                    ++cursors[i];
                }
                if (begin + cursors[i] >= end) continue;
                progress = add_block(begin + cursors[i]) || progress;
                ++cursors[i];
            }
        }
    }

    out.hard_blocks.reserve(hard_count);
    for (uint32_t id = 0; id < prompt_blocks; ++id) {
        if (hard[id]) out.hard_blocks.push_back(id);
    }
    if (has_live) {
        for (uint32_t id = live_begin; id < live_end; ++id) {
            if (hard[id]) ++out.fitted_live_blocks;
        }
        uint32_t replay = live_end;
        while (replay > live_begin && hard[replay - 1]) --replay;
        out.replay_begin_block = replay;
    } else {
        out.replay_begin_block = live_end;
    }
    out.soft_retrievable_blocks =
        out.raw_mandatory_blocks > hard_count
            ? out.raw_mandatory_blocks - hard_count : 0;
    return out;
}

inline const char *kvmem_request_plan_action_name(
        KvMemRequestPlanAction action) {
    switch (action) {
    case KvMemRequestPlanAction::Disabled: return "disabled";
    case KvMemRequestPlanAction::DenseAppend: return "dense-append";
    case KvMemRequestPlanAction::PressurePrefill: return "pressure-prefill";
    case KvMemRequestPlanAction::KeepSelectedAppend:
        return "keep-selected-append";
    case KvMemRequestPlanAction::ResumeMThenAppend:
        return "resume-m-then-append";
    case KvMemRequestPlanAction::ResumePThenReplayTail:
        return "resume-p-then-replay-tail";
    case KvMemRequestPlanAction::ReselectWithUserQuery:
        return "reselect-with-user-query";
    case KvMemRequestPlanAction::GeneratePrivateQueryThenReselect:
        return "generate-private-query-then-reselect";
    case KvMemRequestPlanAction::RejectBeforeExecution:
        return "reject-before-execution";
    }
    return "unknown";
}

inline const char *kvmem_request_plan_reject_name(
        KvMemRequestPlanRejectReason reason) {
    switch (reason) {
    case KvMemRequestPlanRejectReason::None: return "none";
    case KvMemRequestPlanRejectReason::ContextLimit: return "context-limit";
    case KvMemRequestPlanRejectReason::InvalidBudget: return "invalid-budget";
    case KvMemRequestPlanRejectReason::CapacityFitFailure:
        return "capacity-fit-failure";
    }
    return "unknown";
}

// Pure, pre-execution admission. This function deliberately knows nothing
// about mutable executor state; checkpoint choice is finalized below.
inline KvMemRequestPlan kvmem_draft_request_plan(
        const KvmemRequestPlanInput &in) {
    KvMemRequestPlan out;
    out.enabled = in.enabled;
    out.logical_prompt_tokens = in.logical_prompt_tokens;
    out.request_prompt_tokens = in.request_prompt_tokens;
    out.context_limit_tokens = in.context_limit_tokens;
    out.selection_budget_tokens = in.selection_budget_tokens;
    out.generation_budget_tokens = in.generation_budget_tokens;
    out.refresh_trigger_tokens = in.refresh_trigger_tokens;
    out.requested_output_tokens = in.requested_output_tokens;
    out.block_tokens = in.block_tokens;
    out.raw_mandatory_blocks = in.raw_mandatory_blocks;
    out.mandatory_blocks = in.mandatory_blocks;
    out.raw_live_suffix_blocks = in.raw_live_suffix_blocks;
    out.live_suffix_blocks = in.live_suffix_blocks;
    out.soft_retrievable_blocks = in.soft_retrievable_blocks;
    out.sink_blocks = in.sink_blocks;
    out.retrieval_reserve_blocks = in.retrieval_reserve_blocks;
    out.capacity_fitted = in.capacity_fitted;
    out.pressure_ingest = in.pressure_ingest;
    if (!in.enabled) return out;

    if (in.selection_budget_tokens == 0 || in.block_tokens == 0 ||
        in.selection_budget_tokens < in.block_tokens) {
        out.action = KvMemRequestPlanAction::RejectBeforeExecution;
        out.reject_reason = KvMemRequestPlanRejectReason::InvalidBudget;
    } else if (in.context_limit_tokens > 0 &&
               in.logical_prompt_tokens >= in.context_limit_tokens) {
        out.action = KvMemRequestPlanAction::RejectBeforeExecution;
        out.reject_reason = KvMemRequestPlanRejectReason::ContextLimit;
    } else {
        const uint64_t budget_blocks =
            in.selection_budget_tokens / in.block_tokens;
        if (in.mandatory_blocks > budget_blocks) {
            out.action = KvMemRequestPlanAction::RejectBeforeExecution;
            out.reject_reason =
                KvMemRequestPlanRejectReason::CapacityFitFailure;
        }
    }
    if (out.action == KvMemRequestPlanAction::RejectBeforeExecution) {
        out.semantic_action = out.action;
        out.finalized = true;
        return out;
    }

    out.above_selection_budget =
        in.logical_prompt_tokens > in.selection_budget_tokens;
    if (!out.above_selection_budget) {
        out.action = KvMemRequestPlanAction::DenseAppend;
    } else if (in.reselect) {
        out.action = in.private_query
            ? KvMemRequestPlanAction::GeneratePrivateQueryThenReselect
            : KvMemRequestPlanAction::ReselectWithUserQuery;
    } else if (in.keep_selected) {
        out.action = KvMemRequestPlanAction::KeepSelectedAppend;
    } else {
        out.action = KvMemRequestPlanAction::PressurePrefill;
    }
    out.semantic_action = out.action;
    return out;
}

struct KvmemRequestCheckpointInput {
    bool warm_state_available = false;
    uint32_t resume_position = 0;
    bool prompt_checkpoint = false;
    uint32_t common_prefix = 0;
};

// Finalize the same plan before device.begin(), restore_state(), reset_state(),
// or any pool mutation. A keep-selected request must never silently become a
// cold full prefill: that is exactly the Happy DOM failure which exhausted the
// A+B page reserve after a short, unaligned post-refresh tool turn.
inline KvMemRequestPlan kvmem_finalize_request_plan(
        KvMemRequestPlan plan,
        const KvmemRequestCheckpointInput &checkpoint) {
    if (!plan.enabled || plan.finalized ||
        plan.action == KvMemRequestPlanAction::RejectBeforeExecution) {
        return plan;
    }
    plan.resume_position = checkpoint.resume_position;
    plan.common_prefix = checkpoint.common_prefix;
    const bool checkpoint_consistent = checkpoint.warm_state_available &&
        checkpoint.resume_position > 0 &&
        checkpoint.resume_position <= plan.request_prompt_tokens &&
        checkpoint.resume_position <= checkpoint.common_prefix;
    plan.checkpoint_available = checkpoint_consistent;
    if (!checkpoint_consistent) {
        plan.resume_position = 0;
    } else {
        plan.incremental_suffix_tokens =
            plan.request_prompt_tokens - checkpoint.resume_position;
    }
    const bool suffix_exhausts_reserve = checkpoint_consistent &&
        plan.above_selection_budget && plan.generation_budget_tokens > 0 &&
        plan.incremental_suffix_tokens > plan.generation_budget_tokens;
    if (suffix_exhausts_reserve) {
        // This is a final safety net, independent of the server's semantic
        // trigger bookkeeping. Never append more than B tokens to a restored
        // A-token epoch. Preserve an already-planned private/user query; a
        // stable continuation falls back to query-independent pressure ingest.
        plan.pressure_ingest = true;
        if (plan.action == KvMemRequestPlanAction::KeepSelectedAppend) {
            plan.action = KvMemRequestPlanAction::PressurePrefill;
        }
    } else if (checkpoint_consistent) {
        if (plan.action == KvMemRequestPlanAction::KeepSelectedAppend ||
            plan.action == KvMemRequestPlanAction::DenseAppend) {
            plan.action = checkpoint.prompt_checkpoint
                ? KvMemRequestPlanAction::ResumePThenReplayTail
                : KvMemRequestPlanAction::ResumeMThenAppend;
        }
    } else if (plan.action ==
                   KvMemRequestPlanAction::GeneratePrivateQueryThenReselect) {
        // A private query normally observes the still-live A+B epoch. If no
        // P/M checkpoint can restore that epoch, a cold keep-selected prefill
        // is impossible. Pressure-ingest the complete source, then generate
        // and apply the same private query against the bounded window.
        plan.pressure_ingest = true;
    } else if (plan.action == KvMemRequestPlanAction::KeepSelectedAppend) {
        // A checkpoint hole is a storage/replay event, not a semantic query.
        // Rebuild a bounded query-independent pressure window; never reuse the
        // stale query that happened to create the previous selection.
        plan.action = KvMemRequestPlanAction::PressurePrefill;
        plan.pressure_ingest = true;
    }
    plan.finalized = true;
    return plan;
}

}  // namespace qw3::detail
