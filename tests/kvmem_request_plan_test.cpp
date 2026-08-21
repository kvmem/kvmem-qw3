#include "kvmem_request_plan.hpp"

#include <cstdlib>
#include <iostream>

namespace {
void require(bool ok, const char *message) {
    if (ok) return;
    std::cerr << "kvmem_request_plan_test: " << message << "\n";
    std::exit(1);
}
}

int main() {
    using namespace qw3;
    using namespace qw3::detail;
    KvmemRequestPlanInput happy;
    happy.enabled = true;
    happy.keep_selected = true;
    happy.logical_prompt_tokens = 193294;
    happy.request_prompt_tokens = 193294;
    happy.context_limit_tokens = 1000000;
    happy.selection_budget_tokens = 131072;
    happy.generation_budget_tokens = 65536;
    happy.refresh_trigger_tokens = 28672;
    happy.requested_output_tokens = 65536;
    happy.block_tokens = 32;
    happy.raw_mandatory_blocks = 173;
    happy.mandatory_blocks = 173;
    happy.raw_live_suffix_blocks = 8;
    happy.live_suffix_blocks = 8;
    happy.sink_blocks = 64;
    auto draft = kvmem_draft_request_plan(happy);
    require(draft.action == KvMemRequestPlanAction::KeepSelectedAppend,
            "Happy DOM continuation was not admitted as keep-selected");
    auto resume_m = kvmem_finalize_request_plan(
        draft, {true, 193242, false, 193242});
    require(resume_m.action == KvMemRequestPlanAction::ResumeMThenAppend &&
                resume_m.resume_position == 193242 &&
                resume_m.checkpoint_available &&
                resume_m.incremental_suffix_tokens == 52,
            "unaligned exact M was not selected");
    auto resume_p = kvmem_finalize_request_plan(
        draft, {true, 192192, true, 193242});
    require(resume_p.action ==
                KvMemRequestPlanAction::ResumePThenReplayTail,
            "P fallback was not selected");
    auto recover = kvmem_finalize_request_plan(
        draft, {true, 0, false, 193242});
    require(recover.action == KvMemRequestPlanAction::PressurePrefill &&
                recover.pressure_ingest,
            "checkpoint hole did not select safe pressure rebuild");
    auto evicted_recover = kvmem_finalize_request_plan(
        draft, {false, 0, false, 0});
    require(evicted_recover.action ==
                KvMemRequestPlanAction::PressurePrefill &&
                evicted_recover.pressure_ingest,
            "evicted warm slot did not select safe pressure rebuild");
    auto inconsistent_recover = kvmem_finalize_request_plan(
        draft, {true, 193295, false, 193294});
    require(inconsistent_recover.action ==
                KvMemRequestPlanAction::PressurePrefill &&
                inconsistent_recover.pressure_ingest &&
                !inconsistent_recover.checkpoint_available &&
                inconsistent_recover.resume_position == 0,
            "out-of-range checkpoint was trusted by the request plan");

    KvmemRequestPlanInput large_suffix = happy;
    large_suffix.logical_prompt_tokens = 270000;
    large_suffix.request_prompt_tokens = 270000;
    auto large_suffix_draft = kvmem_draft_request_plan(large_suffix);
    auto large_suffix_final = kvmem_finalize_request_plan(
        large_suffix_draft, {true, 193242, false, 193242});
    require(large_suffix_final.action ==
                KvMemRequestPlanAction::PressurePrefill &&
                large_suffix_final.pressure_ingest &&
                large_suffix_final.incremental_suffix_tokens == 76758,
            "oversized checkpoint suffix bypassed the generation reserve");

    const auto fit_70k = kvmem_fit_mandatory_blocks(
        KvmemMandatoryFitInput{
            2200, 2048, 32, 164,
            {{0, 12, 0}, {2188, 2196, 0}},
            3, 2196});
    require(fit_70k.capacity_fitted &&
                fit_70k.raw_live_blocks == 2193 &&
                fit_70k.hard_blocks.size() < 2048 &&
                fit_70k.fitted_live_blocks < fit_70k.raw_live_blocks &&
                fit_70k.soft_retrievable_blocks > 0 &&
                fit_70k.retrieval_reserve_blocks == 256 &&
                fit_70k.replay_begin_block > 2000,
            "70K live tool result was not converted to bounded anchors");
    const auto fit_128k_plus = kvmem_fit_mandatory_blocks(
        KvmemMandatoryFitInput{
            5200, 2048, 32, 164,
            {{0, 20, 0}, {5180, 5190, 0}},
            100, 5190});
    require(fit_128k_plus.capacity_fitted &&
                fit_128k_plus.raw_live_blocks == 5090 &&
                fit_128k_plus.hard_blocks.size() <= 1792 &&
                fit_128k_plus.replay_begin_block >= 5000,
            "128K+ live tool result exhausted the active window");
    const auto fit_oversized_sink = kvmem_fit_mandatory_blocks(
        KvmemMandatoryFitInput{
            3000, 64, 1000, 8, {{0, 1000, 0}}, 2900, 3000});
    require(fit_oversized_sink.capacity_fitted &&
                fit_oversized_sink.hard_blocks.size() <= 56 &&
                fit_oversized_sink.replay_begin_block >= 2992,
            "oversized sink starved the live protocol tail");

    happy.raw_mandatory_blocks = 4097;
    happy.mandatory_blocks = 240;
    happy.raw_live_suffix_blocks = 3900;
    happy.live_suffix_blocks = 164;
    happy.soft_retrievable_blocks = 3857;
    happy.capacity_fitted = true;
    auto fitted = kvmem_draft_request_plan(happy);
    require(fitted.action == KvMemRequestPlanAction::KeepSelectedAppend &&
                fitted.capacity_fitted &&
                fitted.soft_retrievable_blocks == 3857,
            "fitted oversized spans were rejected instead of retrieved");
    happy.mandatory_blocks = 4097;
    auto invalid_fit = kvmem_draft_request_plan(happy);
    require(invalid_fit.action ==
                KvMemRequestPlanAction::RejectBeforeExecution &&
                invalid_fit.reject_reason ==
                    KvMemRequestPlanRejectReason::CapacityFitFailure,
            "invalid fitted plan crossed the active budget");
    happy.raw_mandatory_blocks = 173;
    happy.mandatory_blocks = 173;
    happy.raw_live_suffix_blocks = 8;
    happy.live_suffix_blocks = 8;
    happy.soft_retrievable_blocks = 0;
    happy.capacity_fitted = false;
    happy.keep_selected = false;
    happy.reselect = true;
    auto user_query = kvmem_draft_request_plan(happy);
    require(user_query.action ==
                KvMemRequestPlanAction::ReselectWithUserQuery,
            "new user query did not plan semantic reselection");
    happy.private_query = true;
    happy.pressure_ingest = true;
    auto private_query = kvmem_draft_request_plan(happy);
    require(private_query.action ==
                KvMemRequestPlanAction::GeneratePrivateQueryThenReselect &&
                private_query.pressure_ingest,
            "A+T refresh did not plan a private query");
    private_query.pressure_ingest = false;
    const auto private_checkpoint_hole = kvmem_finalize_request_plan(
        private_query, {false, 0, false, 3});
    require(private_checkpoint_hole.action ==
                KvMemRequestPlanAction::GeneratePrivateQueryThenReselect &&
                private_checkpoint_hole.pressure_ingest,
            "private query checkpoint hole attempted cold keep-selected prefill");
    private_query.request_prompt_tokens = 270000;
    private_query.logical_prompt_tokens = 270000;
    private_query.pressure_ingest = false;
    const auto private_large_suffix = kvmem_finalize_request_plan(
        private_query, {true, 193242, false, 193242});
    require(private_large_suffix.action ==
                KvMemRequestPlanAction::GeneratePrivateQueryThenReselect &&
                private_large_suffix.pressure_ingest,
            "oversized private-query suffix lost semantic reselection");

    std::cout << "kvmem_request_plan_test: PASS\n";
    return 0;
}
