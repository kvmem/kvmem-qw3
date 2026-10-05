#include "kvmem_refresh_policy.hpp"
#include "kvmem_request_plan.hpp"

#include <cstdlib>
#include <iostream>

namespace {
int checks = 0;
void require(bool value, const char *message) {
    ++checks;
    if (value) return;
    std::cerr << message << '\n';
    std::exit(1);
}
}

int main() {
    using namespace qw3;
    using namespace qw3::detail;
    // Structural values from the first above-budget saved desktop request.
    // No task contents, scoring answers, GPU execution, or model output.
    constexpr uint64_t A = 131072, B = 65536, T = 28672;
    KvmemHarnessRefreshInput input{
        true, true, false, true, false, false, 181074, A, 0,
        kvmem_cross_turn_refresh_decision(181074, 117861, 0, 0)};
    auto frozen = kvmem_harness_refresh_decision(input, false);
    require(!frozen.reselect && !frozen.private_query && !frozen.headroom_grace,
            "guided-off stable desktop task must match the observed gate");

    // The both mode supplies T to boundary decisions and authorizes a fresh
    // private query. Merely setting recompute-query cannot change this gate.
    input.trigger_tokens = T;
    input.cross_turn = kvmem_cross_turn_refresh_decision(181074, 117861, 0, T);
    auto candidate = kvmem_harness_refresh_decision(input, true);
    require(candidate.reselect && candidate.private_query,
            "both mode did not refresh the already-grown desktop task");
    require(candidate.reason == KvmemHarnessRefreshReason::InitialHeadroom,
            "first retrieval invented a prior refresh anchor");

    input.prompt_tokens = A + T - 1;
    auto before = kvmem_harness_refresh_decision(input, true);
    require(!before.reselect && before.initial_headroom_grace
                && before.middecode_tokens_until_refresh == 1,
            "both mode refreshed before the fixed threshold");
    input.prompt_tokens = A + T;
    require(kvmem_harness_refresh_decision(input, true).private_query,
            "both mode skipped the exact initial threshold");

    input.prompt_tokens = 183917;
    input.selection_started = true;
    input.cross_turn = kvmem_cross_turn_refresh_decision(183917, 181074, 181074, T);
    auto short_turn = kvmem_harness_refresh_decision(input, true);
    require(!short_turn.reselect && short_turn.headroom_grace,
            "short desktop callback unnecessarily regenerated a query");
    input.prompt_tokens = 216282;
    input.cross_turn = kvmem_cross_turn_refresh_decision(216282, 204267, 181074, T);
    auto grown = kvmem_harness_refresh_decision(input, true);
    require(grown.reselect && grown.private_query
                && grown.reason == KvmemHarnessRefreshReason::CrossTurnGrowth,
            "accumulated desktop growth did not refresh current intent");
    require(!kvmem_harness_refresh_decision(input, false).reselect,
            "disabled guided mode silently changed the frozen policy");

    input.same_user_query = false;
    auto changed_goal = kvmem_harness_refresh_decision(input, false);
    require(changed_goal.reselect && !changed_goal.private_query
                && changed_goal.reason == KvmemHarnessRefreshReason::NewUserQuery,
            "a genuinely new user goal must establish its own query");
    input.same_user_query = true;
    input.force_reselect = true;
    auto forced = kvmem_harness_refresh_decision(input, false);
    require(forced.reselect && !forced.private_query
                && forced.reason == KvmemHarnessRefreshReason::ExplicitForce,
            "force diagnostics changed their explicit query semantics");

    KvmemRequestPlanInput plan_input;
    plan_input.enabled = true;
    plan_input.keep_selected = true;
    plan_input.logical_prompt_tokens = plan_input.request_prompt_tokens = 181074;
    plan_input.context_limit_tokens = 524288;
    plan_input.selection_budget_tokens = A;
    plan_input.generation_budget_tokens = B;
    plan_input.refresh_trigger_tokens = T;
    plan_input.requested_output_tokens = 16384;
    plan_input.block_tokens = 128;
    plan_input.raw_mandatory_blocks = plan_input.mandatory_blocks = 599;
    plan_input.raw_live_suffix_blocks = plan_input.live_suffix_blocks = 495;
    auto old_plan = kvmem_finalize_request_plan(kvmem_draft_request_plan(plan_input), {});
    require(old_plan.action == KvMemRequestPlanAction::PressurePrefill
                && old_plan.pressure_ingest,
            "cold stable desktop request must match observed pressure fallback");
    plan_input.reselect = candidate.reselect;
    plan_input.private_query = candidate.private_query;
    auto new_plan = kvmem_finalize_request_plan(kvmem_draft_request_plan(plan_input), {});
    require(new_plan.action == KvMemRequestPlanAction::GeneratePrivateQueryThenReselect
                && new_plan.pressure_ingest,
            "cold private refresh lost its query during bounded source ingest");
    require(new_plan.selection_budget_tokens == A && new_plan.generation_budget_tokens == B
                && new_plan.requested_output_tokens == 16384,
            "candidate policy silently enlarged the frozen resource budgets");
    require(kvmem_private_refresh_requires_pressure(true, false, 0, A + B, A, B),
            "oversized first callback attempted an impossible dense epoch");
    std::cout << "{\"passed\":true,\"checks\":" << checks
              << ",\"model_calls\":0,\"device_inputs\":0,"
              << "\"scope\":\"CPU policy using saved desktop metadata\"}\n";
}
