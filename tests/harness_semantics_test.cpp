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

void test_dsh_baseline_after_real_query() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "fix the parser"));
    spans.push_back(append_message(
        prompt, 1, "user",
        "<system-reminder>\n"
        "The following workspace instructions may be relevant to your work.\n\n"
        "Instructions from: AGENTS.md\n\nAlways run tests.\n"
        "</system-reminder>"));

    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::DeepSeekHarness, prompt, control_end, spans);
    require(plan.current_query_message_index.has_value() &&
                *plan.current_query_message_index == 0,
            "DSH workspace baseline displaced the real current query");
    require(count_reason(plan, HarnessSpanReason::SystemControl) == 1,
            "control prefix was not pinned");
    require(count_reason(plan, HarnessSpanReason::CurrentQuery) == 1,
            "real DSH query was not pinned");
    require(count_reason(plan, HarnessSpanReason::LiveToolTrajectory) == 1,
            "post-query DSH baseline was not in the live trajectory");
    require(count_reason(plan, HarnessSpanReason::ProjectPolicy) == 1 &&
                plan.project_policy_frame_count == 1,
            "DSH workspace baseline was not retained as project policy");
}

void test_opencode_nested_policy_latest_wins() {
    std::string prompt = "CONTROL\n";
    const size_t control_end = prompt.size();
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "inspect package"));
    spans.push_back(append_message(prompt, 1, "assistant", "calling read"));
    spans.push_back(append_message(
        prompt, 2, "tool",
        "file v1\n<system-reminder>\n"
        "Instructions from: packages/app/AGENTS.md\nold policy\n"
        "</system-reminder>"));
    spans.push_back(append_message(prompt, 3, "user", "read it again"));
    spans.push_back(append_message(prompt, 4, "assistant", "calling read"));
    spans.push_back(append_message(
        prompt, 5, "tool",
        "file v2\n<system-reminder>\n"
        "Instructions from: packages/app/AGENTS.md\nnew policy\n"
        "</system-reminder>"));
    spans.push_back(append_message(prompt, 6, "user", "now edit it"));

    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::OpenCode, prompt, control_end, spans);
    require(plan.current_query_message_index.has_value() &&
                *plan.current_query_message_index == 6,
            "latest OpenCode task was not selected");
    require(count_reason(plan, HarnessSpanReason::LiveToolTrajectory) == 0,
            "completed historical tool chain was retained as live");
    require(count_reason(plan, HarnessSpanReason::ProjectPolicy) == 1,
            "latest nested OpenCode policy did not supersede the old copy");
    const HarnessByteSpan *policy = nullptr;
    for (const HarnessByteSpan &span : plan.spans) {
        if (span.reason == HarnessSpanReason::ProjectPolicy) policy = &span;
    }
    require(policy != nullptr &&
                prompt.substr(policy->begin, policy->end - policy->begin)
                    .find("new policy") != std::string::npos,
            "the stale OpenCode policy won latest-path selection");
}

void test_claude_mixed_query_and_reminder() {
    std::string prompt;
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(
        prompt, 0, "user",
        "implement the change\n"
        "<system-reminder>\nInstructions from: CLAUDE.md\n"
        "Use the project style.\n</system-reminder>"));
    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::ClaudeCode, prompt, 0, spans);
    require(plan.current_query_message_index.has_value() &&
                *plan.current_query_message_index == 0,
            "Claude query with an appended reminder was discarded");
    require(count_reason(plan, HarnessSpanReason::CurrentQuery) == 1 &&
                count_reason(plan, HarnessSpanReason::ProjectPolicy) == 1,
            "Claude mixed query did not receive both lifetimes");
}

void test_control_prefix_without_messages() {
    const std::string prompt = "TOOLS AND SYSTEM";
    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::OpenCode, prompt, prompt.size(), {});
    require(count_reason(plan, HarnessSpanReason::SystemControl) == 1,
            "message-free control prefix was not pinned");
    require(!plan.current_query_message_index.has_value(),
            "message-free prompt invented a current query");
}

void test_user_role_tool_response_is_not_a_query() {
    std::string prompt;
    std::vector<HarnessRenderedMessageSpan> spans;
    spans.push_back(append_message(prompt, 0, "user", "run the tool"));
    spans.push_back(append_message(prompt, 1, "assistant", "tool call"));
    // The production renderer records assistant segment bounds but not a
    // request-content subrange because generated thinking/tool XML is mixed in.
    spans.back().content_begin = 0;
    spans.back().content_end = 0;
    spans.push_back(append_message(
        prompt, 2, "user",
        "<tool_response>tool output</tool_response>"));
    const HarnessSemanticPlan plan = derive_harness_semantic_plan(
        HarnessKind::GenericToolClient, prompt, 0, spans);
    require(plan.current_query_message_index.has_value() &&
                *plan.current_query_message_index == 0,
            "user-role tool response displaced the real query");
    require(count_reason(plan, HarnessSpanReason::LiveToolTrajectory) == 2,
            "user-role tool response chain was not kept live");
}

} // namespace

int main() {
    test_harness_classification();
    test_meta_message_detection();
    test_dsh_baseline_after_real_query();
    test_opencode_nested_policy_latest_wins();
    test_claude_mixed_query_and_reminder();
    test_control_prefix_without_messages();
    test_user_role_tool_response_is_not_a_query();
    std::cout << "harness_semantics_test: PASS\n";
    return 0;
}
