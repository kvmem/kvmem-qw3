#include "harness_semantics.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using qw3::detail::HarnessByteSpan;
using qw3::detail::HarnessKind;
using qw3::detail::HarnessProtocol;
using qw3::detail::HarnessRenderedMessageSpan;
using qw3::detail::HarnessRequestSignals;
using qw3::detail::HarnessSemanticPlan;
using qw3::detail::HarnessSpanReason;
using qw3::detail::classify_harness;
using qw3::detail::derive_harness_semantic_plan;
using qw3::detail::harness_message_is_meta_only;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "harness_semantics_test: " << message << "\n";
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

HarnessRenderedMessageSpan append_message(
        std::string &prompt, size_t index, const std::string &role,
        const std::string &content) {
    const size_t segment_begin = prompt.size();
    prompt += "<" + role + ">";
    const size_t content_begin = prompt.size();
    prompt += content;
    const size_t content_end = prompt.size();
    prompt += "</" + role + ">";
    return HarnessRenderedMessageSpan{
        index, role, segment_begin, prompt.size(), content_begin, content_end};
}

size_t count_reason(const HarnessSemanticPlan &plan,
                    HarnessSpanReason reason) {
    size_t count = 0;
    for (const HarnessByteSpan &span : plan.spans) {
        if (span.reason == reason) ++count;
    }
    return count;
}

void test_harness_classification() {
    HarnessRequestSignals signals;
    signals.protocol = HarnessProtocol::AnthropicMessages;
    require(classify_harness(signals).kind == HarnessKind::ClaudeCode,
            "Anthropic route was not classified as Claude Code");

    signals = {};
    signals.user_agent = "opencode/1.18.18 runtime/bun";
    require(classify_harness(signals).kind == HarnessKind::OpenCode,
            "OpenCode User-Agent was not recognized");

    signals = {};
    signals.has_opencode_header = true;
    require(classify_harness(signals).kind == HarnessKind::OpenCode,
            "explicit OpenCode header was not recognized");

    signals = {};
    signals.user_agent =
        "deepseek-harness/0.9.0 (+https://github.com/deepseek-ai/deepseek-harness)";
    signals.compact = true;
    const auto dsh = classify_harness(signals);
    require(dsh.kind == HarnessKind::DeepSeekHarness && dsh.compact,
            "DeepSeek Harness attribution/compact signal was not preserved");

    signals = {};
    signals.has_deepseek_harness_header = true;
    require(classify_harness(signals).kind == HarnessKind::DeepSeekHarness,
            "native DeepSeek Harness header was not recognized");

    signals = {};
    signals.has_tools = true;
    signals.has_leading_control_message = true;
    require(classify_harness(signals).kind == HarnessKind::GenericToolClient,
            "tool-bearing compatible client did not use the generic fallback");

    signals = {};
    signals.has_tools = true;
    require(classify_harness(signals).kind == HarnessKind::None,
            "plain tool request without control content was over-classified");
}

void test_meta_message_detection() {
    const std::string turn =
        "<system-reminder>Answer without a thinking block.</system-reminder>";
    require(harness_message_is_meta_only(HarnessKind::ClaudeCode, turn),
            "Claude turn reminder was not recognized as meta-only");
    require(!harness_message_is_meta_only(
                HarnessKind::GenericToolClient, turn),
            "unknown generic reminder was treated as trusted control");

    const std::string policy =
        "<system-reminder>\nInstructions from: packages/app/AGENTS.md\n"
        "Run the focused tests.\n</system-reminder>";
    require(harness_message_is_meta_only(
                HarnessKind::GenericToolClient, policy),
            "known persistent policy was not recognized for generic fallback");
    require(!harness_message_is_meta_only(
                HarnessKind::DeepSeekHarness, "actual task\n" + policy),
            "mixed real user content was incorrectly classified as meta-only");
}

const HarnessByteSpan *find_reason(const HarnessSemanticPlan &plan,
                                   HarnessSpanReason reason) {
    for (const HarnessByteSpan &span : plan.spans) {
        if (span.reason == reason) return &span;
    }
    return nullptr;
}

void test_bounded_query_and_live_tool_suffix() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "fix the parser"));
    spans.push_back(append_message(prompt, 1, "assistant", "calling read"));
    spans.push_back(append_message(prompt, 2, "tool", "old tool output"));
    spans.push_back(append_message(prompt, 3, "assistant", "calling test"));
    spans.push_back(append_message(prompt, 4, "tool", "latest test output"));

    for (HarnessKind kind : {HarnessKind::ClaudeCode,
                             HarnessKind::OpenCode,
                             HarnessKind::DeepSeekHarness,
                             HarnessKind::GenericToolClient}) {
        const HarnessSemanticPlan plan = derive_harness_semantic_plan(
            kind, prompt, control_end, spans);
        require(count_reason(plan, HarnessSpanReason::SystemControl) == 1 &&
                    count_reason(plan, HarnessSpanReason::CurrentQuery) == 1,
                "control/query exact spans were not retained");
        require(count_reason(plan, HarnessSpanReason::RootTask) == 0 &&
                    plan.root_task_message_index == 0,
                "one-turn root task was duplicated");
        require(count_reason(plan, HarnessSpanReason::LiveToolTrajectory) == 0,
                "live suffix leaked into exact mandatory spans");
        require(plan.current_query_message_index == 0,
                "real task was not identified as score query");
        require(plan.live_suffix_message_begin_index == 3,
                "latest tool transaction did not start at its assistant call");
        require(plan.live_suffix_span.has_value() &&
                    plan.live_suffix_span->begin == spans[3].segment_begin &&
                    plan.live_suffix_span->end == prompt.size() &&
                    plan.history_end == spans[3].segment_begin,
                "bounded live suffix/history boundary is incorrect");
        require(spans[1].segment_begin < plan.history_end &&
                    spans[2].segment_end <= plan.history_end,
                "completed tool transaction was not returned to history");
    }
}

void test_claude_user_role_tool_response_and_parallel_results() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "implement feature"));
    spans.push_back(append_message(prompt, 1, "assistant", "parallel calls"));
    spans.push_back(append_message(
        prompt, 2, "user", "<tool_response>first</tool_response>"));
    spans.push_back(append_message(
        prompt, 3, "user", "<tool_response>second</tool_response>"));
    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::ClaudeCode, prompt, control_end, spans);
    require(plan.current_query_message_index == 0,
            "Claude user-role tool result displaced the task query");
    require(plan.live_suffix_message_begin_index == 1 &&
                plan.live_suffix_span->begin == spans[1].segment_begin,
            "parallel Claude tool results did not retain their assistant call");
}

void test_new_user_query_is_its_own_live_suffix() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "first task"));
    spans.push_back(append_message(prompt, 1, "assistant", "done"));
    spans.push_back(append_message(prompt, 2, "user", "new task"));
    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::OpenCode, prompt, control_end, spans);
    require(plan.current_query_message_index == 2 &&
                plan.live_suffix_message_begin_index == 2,
            "latest real user query was not selected");
    require(plan.root_task_message_index == 0 &&
                count_reason(plan, HarnessSpanReason::RootTask) == 1,
            "stable root task was not retained with a later instruction");
    const HarnessByteSpan *root = find_reason(
        plan, HarnessSpanReason::RootTask);
    require(root != nullptr && root->begin == spans[0].segment_begin &&
                root->end == spans[0].segment_end,
            "root task span was not byte-exact");
    require(plan.live_suffix_span->begin == spans[2].segment_begin,
            "new query live suffix started before the query");
}

void test_tool_and_meta_turns_do_not_displace_root_or_current_task() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "exact root criteria"));
    spans.push_back(append_message(prompt, 1, "assistant", "call"));
    spans.push_back(append_message(
        prompt, 2, "user", "<tool_response>result</tool_response>"));
    spans.push_back(append_message(
        prompt, 3, "user",
        "<system-reminder>Answer without a thinking block.</system-reminder>"));
    spans.push_back(append_message(prompt, 4, "user", "finish the task"));
    spans.push_back(append_message(prompt, 5, "assistant", "call again"));
    spans.push_back(append_message(prompt, 6, "tool", "latest result"));

    for (HarnessKind kind : {HarnessKind::ClaudeCode,
                             HarnessKind::OpenCode,
                             HarnessKind::DeepSeekHarness}) {
        const HarnessSemanticPlan plan = derive_harness_semantic_plan(
            kind, prompt, control_end, spans);
        require(plan.root_task_message_index == 0 &&
                    plan.current_query_message_index == 4,
                "tool/meta turn displaced root or current task");
        require(count_reason(plan, HarnessSpanReason::RootTask) == 1 &&
                    count_reason(plan, HarnessSpanReason::CurrentQuery) == 1,
                "root/current lifetimes were not independently retained");
        require(plan.live_suffix_message_begin_index == 5,
                "latest tool suffix boundary changed after root pinning");
    }
}

void test_latest_policy_copy_is_exact_mandatory() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "inspect package"));
    spans.push_back(append_message(
        prompt, 1, "tool",
        "<system-reminder>\nInstructions from: pkg/AGENTS.md\nold\n"
        "</system-reminder>"));
    spans.push_back(append_message(
        prompt, 2, "tool",
        "<system-reminder>\nUpdated instructions from: pkg/AGENTS.md\nnew\n"
        "</system-reminder>"));
    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::OpenCode, prompt, control_end, spans);
    require(count_reason(plan, HarnessSpanReason::ProjectPolicy) == 1 &&
                plan.project_policy_frame_count == 1,
            "latest keyed policy did not supersede its stale copy");
    const HarnessByteSpan *policy = find_reason(
        plan, HarnessSpanReason::ProjectPolicy);
    require(policy != nullptr &&
                prompt.substr(policy->begin, policy->end - policy->begin)
                    .find("new") != std::string::npos,
            "stale policy copy remained mandatory");
}

void test_missing_control_prefix_does_not_invent_system_pin() {
    std::string prompt = "<user>run the tool</user>";
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(
        prompt, 1, "tool",
        "<system-reminder>Instructions from: AGENTS.md\npolicy"
        "</system-reminder>"));

    const HarnessSemanticPlan missing = derive_harness_semantic_plan(
        HarnessKind::OpenCode, prompt, 0, spans);
    require(count_reason(missing, HarnessSpanReason::SystemControl) == 0 &&
                count_reason(missing, HarnessSpanReason::ProjectPolicy) == 1,
            "missing control boundary invented system pin or lost exact policy");
    const HarnessSemanticPlan invalid = derive_harness_semantic_plan(
        HarnessKind::OpenCode, prompt, prompt.size() + 1, spans);
    require(count_reason(invalid, HarnessSpanReason::SystemControl) == 0,
            "out-of-range control boundary was accepted");
    require(derive_harness_semantic_plan(
                HarnessKind::None, prompt, prompt.size(), spans).spans.empty(),
            "unidentified client unexpectedly gained a mandatory span");
}

} // namespace

int main() {
    test_harness_classification();
    test_meta_message_detection();
    test_bounded_query_and_live_tool_suffix();
    test_claude_user_role_tool_response_and_parallel_results();
    test_new_user_query_is_its_own_live_suffix();
    test_tool_and_meta_turns_do_not_displace_root_or_current_task();
    test_latest_policy_copy_is_exact_mandatory();
    test_missing_control_prefix_does_not_invent_system_pin();
    std::cout << "harness_semantics_test: PASS\n";
    return 0;
}
