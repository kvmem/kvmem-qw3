#include "kvmem_refresh_policy.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "kvmem_refresh_policy_test: " << message << "\n";
    std::exit(1);
}

}  // namespace

int main() {
    using namespace qw3::detail;
    require(!kvmem_middecode_refresh_due(24575, 0, 24576, 0, 2),
            "refresh fired before threshold");
    require(kvmem_middecode_refresh_due(24576, 0, 24576, 0, 2),
            "refresh did not fire at threshold");
    require(!kvmem_middecode_refresh_due(49152, 24576, 24576, 2, 2),
            "refresh exceeded request cap");
    require(kvmem_middecode_trigger_for_epoch(1770, 61440, 0) == 1770,
            "initial headroom epoch lost its remaining-distance trigger");
    require(kvmem_middecode_trigger_for_epoch(1770, 61440, 1) == 61440,
            "post-refresh epoch reused the initial remaining distance");
    require(kvmem_middecode_trigger_for_epoch(1770, 0, 1) == 1770,
            "zero steady trigger changed legacy direct-caller behavior");
    require(kvmem_mtp_compact_output_horizon(
                188416, 65473, 65536, 61377, 2) == 65536,
            "multi-epoch request used its request-wide output cap as one "
            "MTP compact epoch");
    require(kvmem_mtp_compact_output_horizon(
                32768, 65473, 65536, 61377, 2) == 32768,
            "short multi-epoch request expanded its MTP compact horizon");
    require(kvmem_mtp_compact_output_horizon(
                188416, 65473, 65536, 0, 2) == 188416,
            "request without a refresh trigger lost the conservative MTP "
            "position guard");
    require(kvmem_mtp_compact_output_horizon(
                188416, 65473, 65536, 61377, 0) == 188416,
            "request without refresh capacity lost the conservative MTP "
            "position guard");
    require(kvmem_middecode_refresh_due(
                1798, 0,
                kvmem_middecode_trigger_for_epoch(1770, 61440, 0), 0, 2),
            "remaining headroom did not authorize the first refresh");
    require(!kvmem_middecode_refresh_due(
                3586, 1798,
                kvmem_middecode_trigger_for_epoch(1770, 61440, 1), 1, 2),
            "second refresh repeated after the initial 1770-token distance");
    require(!kvmem_checkpoint_pressure_selection_allowed(true),
            "prefix checkpoint bypassed the A+B keep-selected grace epoch");
    require(kvmem_checkpoint_pressure_selection_allowed(false),
            "ordinary sparse prefix checkpoint lost pressure normalization");
    require(kvmem_refresh_prefill_keeps_epoch(true, false),
            "headroom grace did not preserve its A+B epoch");
    require(kvmem_refresh_prefill_keeps_epoch(false, true),
            "private query lost the A+B epoch before semantic selection");
    require(kvmem_prefill_chunk_pages(31, 32) == 1,
            "partial prefill tail did not round to one physical page");
    require(kvmem_prefill_chunk_fits(55, 31, 32),
            "194847-token prompt tail was falsely rejected at A+B boundary");
    require(!kvmem_prefill_chunk_fits(55, 2048, 32),
            "full 2K prefill chunk bypassed physical capacity admission");
    require(kvmem_prefill_chunk_fits(64, 2048, 32),
            "exact-fit prefill chunk was rejected before allocation");
    require(!kvmem_private_refresh_requires_pressure(
                true, true, 28672, 0, 65536, 32768),
            "ordinary A+T private refresh discarded a live epoch");
    require(kvmem_private_refresh_requires_pressure(
                true, true, 70000, 0, 65536, 32768),
            "large tool callback tried to append beyond A+B");
    require(kvmem_private_refresh_requires_pressure(
                true, false, 0, 140000, 65536, 32768),
            "initial oversized callback tried to retain an impossible epoch");
    require(!kvmem_refresh_prefill_keeps_epoch(false, false),
            "ordinary semantic request incorrectly retained a stale epoch");
    require(kvmem_guided_query_piece_terminates("Find Foo::bar?"),
            "ASCII question was not a query boundary");
    require(kvmem_guided_query_piece_terminates("查找失败的测试。\n"),
            "CJK sentence was not a query boundary");
    require(!kvmem_guided_query_piece_terminates("src/foo.rs"),
            "file-name suffix was treated as a query boundary");
    require(kvmem_guided_query_complete(63, 64, true),
            "naturally terminated query was rejected");
    require(!kvmem_guided_query_complete(64, 64, false),
            "unterminated token-cap prefix was published");
    require(!kvmem_guided_query_complete(0, 64, true),
            "empty query was published");
    const auto no_anchor = kvmem_cross_turn_refresh_decision(
        235098, 0, 0, 24576);
    require(!no_anchor.due && !no_anchor.reset,
            "ingest-only warm turn invented a refresh anchor");
    const auto below_cross_turn = kvmem_cross_turn_refresh_decision(
        256000, 250000, 236000, 24576);
    require(!below_cross_turn.due &&
                below_cross_turn.delta_tokens == 20000,
            "short tool turns crossed the threshold early");
    const auto due_cross_turn = kvmem_cross_turn_refresh_decision(
        260600, 256000, 236000, 24576);
    require(due_cross_turn.due &&
                due_cross_turn.delta_tokens == 24600,
            "accumulated tool turns did not trigger semantic refresh");
    const auto new_session = kvmem_cross_turn_refresh_decision(
        232000, 260600, 236000, 24576);
    require(new_session.reset && !new_session.due,
            "shorter replacement trajectory reused stale refresh state");
    const auto template_tail_rewrite = kvmem_cross_turn_refresh_decision(
        259994, 260000, 236000, 24576);
    require(!template_tail_rewrite.reset &&
                !template_tail_rewrite.due,
            "small chat-template tail rewrite reset the trajectory");
    require(kvmem_prompt_trajectory_reset(230000, 260000),
            "material compaction did not reset the trajectory");
    require(kvmem_harness_turn_gate(
                false, true, false, no_anchor).reselect,
            "new direct-harness task did not establish a semantic epoch");
    require(!kvmem_harness_turn_gate(
                false, false, false, no_anchor).reselect,
            "ingest-only new task ignored explicit reselection off");
    require(!kvmem_harness_turn_gate(
                true, true, false, below_cross_turn).reselect,
            "default-auto tool continuation reselected on every request");
    const auto due_gate = kvmem_harness_turn_gate(
        true, false, false, due_cross_turn);
    require(due_gate.reselect && due_gate.cross_turn_refresh,
            "off tool continuation did not promote at accumulated threshold");
    const KvmemHarnessRefreshInput first_crossing{
        true, true, false, true, false, false,
        65537, 65536, 28672, no_anchor};
    const auto first_grace =
        kvmem_harness_refresh_decision(first_crossing);
    require(!first_grace.reselect &&
                first_grace.headroom_grace &&
                first_grace.initial_headroom_grace &&
                first_grace.middecode_tokens_until_refresh == 28671,
            "first budget crossing did not preserve A+B headroom");
    KvmemHarnessRefreshInput just_before = first_crossing;
    just_before.prompt_tokens = 94207;
    const auto before_initial =
        kvmem_harness_refresh_decision(just_before);
    require(!before_initial.reselect &&
                before_initial.middecode_tokens_until_refresh == 1,
            "initial refresh fired before A+T");
    KvmemHarnessRefreshInput initial_due = first_crossing;
    initial_due.prompt_tokens = 94208;
    const auto at_initial = kvmem_harness_refresh_decision(initial_due);
    require(at_initial.reselect && at_initial.private_query &&
                at_initial.reason ==
                    KvmemHarnessRefreshReason::InitialHeadroom,
            "initial refresh did not fire at A+T");
    KvmemHarnessRefreshInput new_user = first_crossing;
    new_user.same_user_query = false;
    const auto new_user_decision =
        kvmem_harness_refresh_decision(new_user);
    require(new_user_decision.reselect &&
                !new_user_decision.private_query &&
                new_user_decision.reason ==
                    KvmemHarnessRefreshReason::NewUserQuery,
            "new user query above A did not refresh immediately");
    KvmemHarnessRefreshInput stable_selected = first_crossing;
    stable_selected.selection_started = true;
    stable_selected.cross_turn = kvmem_cross_turn_refresh_decision(
        264000, 263000, 236000, 28672);
    const auto selected_grace =
        kvmem_harness_refresh_decision(stable_selected);
    require(!selected_grace.reselect &&
                selected_grace.headroom_grace &&
                !selected_grace.initial_headroom_grace &&
                selected_grace.middecode_tokens_until_refresh == 672,
            "selected stable tool continuation reselected below T");
    stable_selected.cross_turn = kvmem_cross_turn_refresh_decision(
        264672, 264000, 236000, 28672);
    const auto selected_due =
        kvmem_harness_refresh_decision(stable_selected);
    require(selected_due.reselect && selected_due.private_query &&
                selected_due.reason ==
                    KvmemHarnessRefreshReason::CrossTurnGrowth,
            "selected trajectory did not refresh after T growth");
    KvmemHarnessRefreshInput cold_above = first_crossing;
    cold_above.known_conversation = false;
    const auto cold_decision =
        kvmem_harness_refresh_decision(cold_above);
    require(cold_decision.reselect &&
                cold_decision.reason ==
                    KvmemHarnessRefreshReason::ColdAboveBudget,
            "cold above-budget trajectory did not establish a safe epoch");
    require(!kvmem_middecode_refresh_text_safe("analysis complete"),
            "mid-sentence text was treated as a refresh boundary");
    require(kvmem_middecode_refresh_text_safe("analysis complete."),
            "sentence boundary was not a safe refresh point");
    require(!kvmem_middecode_refresh_text_safe("analysis <tool_"),
            "partial tool marker was treated as safe");
    require(!kvmem_middecode_refresh_text_safe(
                "<tool_call><function=read>"),
            "open tool call was treated as safe");
    require(!kvmem_middecode_refresh_text_safe("```json\n{"),
            "open code fence was treated as safe");
    require(kvmem_middecode_refresh_text_safe(
                "<tool_call><function=read></function></tool_call>"),
            "closed tool call was not a safe point");
    require(!kvmem_middecode_emergency_refresh_due(32255, 32768),
            "emergency refresh fired before the 512-token guard");
    require(kvmem_middecode_emergency_refresh_due(32256, 32768),
            "emergency refresh did not fire at the reserve guard");
    require(kvmem_middecode_emergency_refresh_due(32768, 32768),
            "emergency refresh did not fire at the reserve limit");
    require(kvmem_middecode_refresh_recent_tokens(
                65536, 32256, true) == 32256,
            "emergency refresh did not retain the complete open epoch");
    require(kvmem_middecode_refresh_recent_tokens(
                65536, 4000, true) == 8192,
            "short emergency epoch lost the ordinary recent tail");
    require(kvmem_middecode_refresh_recent_tokens(
                65536, 32256, false) == 8192,
            "safe refresh unexpectedly pinned the complete epoch");
    require(kvmem_middecode_recent_tokens(65536) == 8192,
            "64K budget did not keep an 8K generated tail");
    require(kvmem_middecode_pin_from_block(100000, 65536, 128) ==
                (100000 - 8192) / 128,
            "recent tail block boundary is incorrect");
    require(kvmem_bounded_tail_pin_from_block(
                547, 512, 64, {0, 1, 2, 172, 173}) == 483,
            "bounded tail did not retain its full desired suffix");
    std::vector<uint32_t> nearly_full;
    for (uint32_t id = 0; id < 508; ++id) nearly_full.push_back(id);
    require(kvmem_bounded_tail_pin_from_block(
                547, 512, 64, nearly_full) == 543,
            "bounded tail did not shrink to the remaining four blocks");
    require(kvmem_bounded_tail_pin_from_block(
                547, 512, 64, std::vector<uint32_t>(513, 0)) == 483,
            "duplicate mandatory ids were not de-duplicated");
    std::cout << "kvmem_refresh_policy_test: PASS\n";
    return 0;
}
