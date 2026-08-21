#include "harness_semantics.hpp"

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <utility>

namespace qw3 {
namespace detail {
namespace {

constexpr std::string_view kReminderOpen = "<system-reminder>";
constexpr std::string_view kReminderClose = "</system-reminder>";

std::string ascii_lower(std::string_view value) {
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool contains_ascii_case_insensitive(std::string_view value,
                                     std::string_view needle) {
    if (needle.empty()) return true;
    return ascii_lower(value).find(ascii_lower(needle)) != std::string::npos;
}

bool ascii_space_only(std::string_view value) {
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
}

std::string_view trim_ascii(std::string_view value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return value;
}

struct ReminderFrame {
    size_t begin = 0;
    size_t end = 0;
    bool persistent_policy = false;
    bool complete_baseline = false;
    std::vector<std::string> policy_keys;
};

std::vector<std::string> instruction_keys(std::string_view body) {
    static constexpr std::string_view prefixes[] = {
        "Instructions from:",
        "Additional instructions from:",
        "Updated instructions from:",
        "Instructions removed:",
    };
    std::vector<std::string> keys;
    size_t cursor = 0;
    while (cursor <= body.size()) {
        const size_t newline = body.find('\n', cursor);
        const size_t line_end = newline == std::string_view::npos
            ? body.size() : newline;
        const std::string_view line = trim_ascii(
            body.substr(cursor, line_end - cursor));
        for (std::string_view prefix : prefixes) {
            if (line.size() < prefix.size() ||
                line.compare(0, prefix.size(), prefix) != 0) {
                continue;
            }
            const std::string_view key = trim_ascii(line.substr(prefix.size()));
            if (!key.empty()) keys.emplace_back(key);
            break;
        }
        if (newline == std::string_view::npos) break;
        cursor = newline + 1;
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

ReminderFrame classify_frame(size_t begin, size_t end,
                             std::string_view body) {
    ReminderFrame frame;
    frame.begin = begin;
    frame.end = end;
    frame.policy_keys = instruction_keys(body);
    const std::string lower = ascii_lower(body);
    frame.complete_baseline =
        lower.find("complete workspace instruction baseline") !=
            std::string::npos ||
        lower.find("the following workspace instructions may be relevant") !=
            std::string::npos;
    frame.persistent_policy = frame.complete_baseline ||
        !frame.policy_keys.empty() ||
        lower.find("agents.md") != std::string::npos ||
        lower.find("claude.md") != std::string::npos ||
        lower.find("claudemd") != std::string::npos;
    return frame;
}

std::vector<ReminderFrame> reminder_frames(std::string_view content) {
    std::vector<ReminderFrame> frames;
    size_t cursor = 0;
    while (cursor < content.size()) {
        const size_t begin = content.find(kReminderOpen, cursor);
        if (begin == std::string_view::npos) break;
        const size_t body_begin = begin + kReminderOpen.size();
        const size_t close = content.find(kReminderClose, body_begin);
        if (close == std::string_view::npos) break;
        const size_t end = close + kReminderClose.size();
        frames.push_back(classify_frame(
            begin, end, content.substr(body_begin, close - body_begin)));
        cursor = end;
    }
    return frames;
}

bool frame_is_control(HarnessKind kind, const ReminderFrame &frame) {
    if (kind == HarnessKind::None) return false;
    if (kind == HarnessKind::GenericToolClient) {
        return frame.persistent_policy;
    }
    return true;
}

bool looks_like_tool_response(std::string_view content) {
    content = trim_ascii(content);
    constexpr std::string_view open = "<tool_response>";
    constexpr std::string_view close = "</tool_response>";
    return content.size() >= open.size() + close.size() &&
        content.compare(0, open.size(), open) == 0 &&
        content.compare(content.size() - close.size(), close.size(), close) == 0;
}

bool valid_segment_bounds(std::string_view prompt,
                          const HarnessRenderedMessageSpan &span) {
    return span.segment_begin < span.segment_end &&
        span.segment_end <= prompt.size();
}

bool valid_content_bounds(std::string_view prompt,
                          const HarnessRenderedMessageSpan &span) {
    return valid_segment_bounds(prompt, span) &&
        span.content_begin <= span.content_end &&
        span.content_end <= prompt.size() &&
        span.content_begin >= span.segment_begin &&
        span.content_end <= span.segment_end;
}

bool is_tool_result_message(std::string_view prompt,
                            const HarnessRenderedMessageSpan &span) {
    if (!valid_segment_bounds(prompt, span)) return false;
    if (span.role == "tool") return true;
    if (span.role != "user" || !valid_content_bounds(prompt, span)) {
        return false;
    }
    return looks_like_tool_response(prompt.substr(
        span.content_begin, span.content_end - span.content_begin));
}

bool is_meta_only_message(HarnessKind kind, std::string_view prompt,
                          const HarnessRenderedMessageSpan &span) {
    if (span.role != "user" || !valid_content_bounds(prompt, span)) {
        return false;
    }
    return harness_message_is_meta_only(kind, prompt.substr(
        span.content_begin, span.content_end - span.content_begin));
}

struct PolicyFrameRef {
    size_t message_order = 0;
    size_t prompt_begin = 0;
    size_t prompt_end = 0;
    bool complete_baseline = false;
    std::vector<std::string> keys;
};

} // namespace

HarnessRequestContext classify_harness(
        const HarnessRequestSignals &signals) {
    HarnessRequestContext out;
    out.compact = signals.compact;
    if (signals.protocol == HarnessProtocol::AnthropicMessages) {
        out.kind = HarnessKind::ClaudeCode;
        return out;
    }
    if (signals.has_deepseek_harness_header ||
        contains_ascii_case_insensitive(
            signals.user_agent, "deepseek-harness/")) {
        out.kind = HarnessKind::DeepSeekHarness;
        return out;
    }
    if (signals.has_opencode_header ||
        contains_ascii_case_insensitive(signals.user_agent, "opencode/")) {
        out.kind = HarnessKind::OpenCode;
        return out;
    }
    if (signals.has_tools && signals.has_leading_control_message) {
        out.kind = HarnessKind::GenericToolClient;
    }
    return out;
}

const char *harness_kind_name(HarnessKind kind) {
    switch (kind) {
    case HarnessKind::ClaudeCode: return "claude-code";
    case HarnessKind::OpenCode: return "opencode";
    case HarnessKind::DeepSeekHarness: return "deepseek-harness";
    case HarnessKind::GenericToolClient: return "generic-tool-client";
    case HarnessKind::None: return "none";
    }
    return "none";
}

bool harness_message_is_meta_only(HarnessKind kind,
                                  std::string_view content) {
    const std::vector<ReminderFrame> frames = reminder_frames(content);
    if (frames.empty()) return false;
    size_t cursor = 0;
    for (const ReminderFrame &frame : frames) {
        if (!frame_is_control(kind, frame) || frame.begin < cursor ||
            !ascii_space_only(content.substr(cursor, frame.begin - cursor))) {
            return false;
        }
        cursor = frame.end;
    }
    return ascii_space_only(content.substr(cursor));
}

HarnessSemanticPlan derive_harness_semantic_plan(
        HarnessKind kind,
        std::string_view prompt,
        size_t control_prefix_end,
        const std::vector<HarnessRenderedMessageSpan> &messages) {
    HarnessSemanticPlan plan;
    if (kind == HarnessKind::None) return plan;

    // The deterministic renderer prefix is stable across the complete agent
    // run and contains tool schemas/protocol plus system/developer controls.
    if (control_prefix_end > 0 && control_prefix_end <= prompt.size()) {
        plan.spans.push_back(HarnessByteSpan{
            0, control_prefix_end, HarnessSpanReason::SystemControl});
    }

    // Keep only the newest applicable copy of each workspace policy. A complete
    // baseline supersedes every earlier baseline; later keyed updates supersede
    // only the matching path. This makes policy cost stable rather than growing
    // once per tool result.
    std::vector<PolicyFrameRef> policies;
    for (size_t order = 0; order < messages.size(); ++order) {
        const HarnessRenderedMessageSpan &span = messages[order];
        if (!valid_content_bounds(prompt, span) ||
            (span.role != "user" && span.role != "tool")) {
            continue;
        }
        const std::string_view content = prompt.substr(
            span.content_begin, span.content_end - span.content_begin);
        for (const ReminderFrame &frame : reminder_frames(content)) {
            if (!frame.persistent_policy) continue;
            policies.push_back(PolicyFrameRef{
                order,
                span.content_begin + frame.begin,
                span.content_begin + frame.end,
                frame.complete_baseline,
                frame.policy_keys});
        }
    }

    size_t first_relevant_policy = 0;
    std::optional<size_t> latest_baseline;
    for (size_t i = 0; i < policies.size(); ++i) {
        if (policies[i].complete_baseline) latest_baseline = i;
    }
    if (latest_baseline.has_value()) first_relevant_policy = *latest_baseline;
    std::unordered_map<std::string, size_t> latest_by_key;
    for (size_t i = first_relevant_policy; i < policies.size(); ++i) {
        for (const std::string &key : policies[i].keys) {
            latest_by_key[key] = i;
        }
    }
    for (size_t i = first_relevant_policy; i < policies.size(); ++i) {
        const PolicyFrameRef &frame = policies[i];
        bool keep = latest_baseline.has_value() && i == *latest_baseline;
        if (frame.keys.empty() && !frame.complete_baseline) keep = true;
        for (const std::string &key : frame.keys) {
            const auto it = latest_by_key.find(key);
            if (it != latest_by_key.end() && it->second == i) keep = true;
        }
        if (!keep) continue;
        plan.spans.push_back(HarnessByteSpan{
            frame.prompt_begin, frame.prompt_end,
            HarnessSpanReason::ProjectPolicy});
        ++plan.project_policy_frame_count;
    }

    // Identify the first durable task as well as the newest real user request.
    // Long-running harnesses commonly append a terse "continue/finalize the
    // original task" turn.  Keeping only that newest instruction loses the
    // exact acceptance criteria needed to finish the task.  Both lifetimes are
    // bounded to their individual messages; a one-turn request is pinned only
    // once as CurrentQuery.
    std::optional<size_t> root_query_order;
    for (size_t order = 0; order < messages.size(); ++order) {
        const HarnessRenderedMessageSpan &span = messages[order];
        if (span.role != "user" || !valid_content_bounds(prompt, span)) {
            continue;
        }
        const std::string_view content = prompt.substr(
            span.content_begin, span.content_end - span.content_begin);
        if (looks_like_tool_response(content) ||
            harness_message_is_meta_only(kind, content)) {
            continue;
        }
        root_query_order = order;
        plan.root_task_message_index = span.message_index;
        break;
    }

    std::optional<size_t> query_order;
    for (size_t rev = 0; rev < messages.size(); ++rev) {
        const size_t order = messages.size() - 1 - rev;
        const HarnessRenderedMessageSpan &span = messages[order];
        if (span.role != "user" || !valid_content_bounds(prompt, span)) {
            continue;
        }
        const std::string_view content = prompt.substr(
            span.content_begin, span.content_end - span.content_begin);
        if (looks_like_tool_response(content) ||
            harness_message_is_meta_only(kind, content)) {
            continue;
        }
        query_order = order;
        plan.current_query_message_index = span.message_index;
        plan.spans.push_back(HarnessByteSpan{
            span.segment_begin, span.segment_end,
            HarnessSpanReason::CurrentQuery});
        break;
    }

    if (root_query_order.has_value() && query_order.has_value() &&
        *root_query_order != *query_order) {
        const HarnessRenderedMessageSpan &root = messages[*root_query_order];
        plan.spans.push_back(HarnessByteSpan{
            root.segment_begin, root.segment_end,
            HarnessSpanReason::RootTask});
    }

    if (query_order.has_value()) {
        size_t suffix_order = *query_order;
        // A tool continuation keeps only the newest unfinished tool transaction:
        // all parallel result messages plus the assistant call that produced
        // them. Completed earlier transactions remain semantic candidates.
        size_t tail = messages.size();
        while (tail > *query_order + 1 &&
               is_meta_only_message(kind, prompt, messages[tail - 1])) {
            --tail;
        }
        if (tail > *query_order + 1 &&
            is_tool_result_message(prompt, messages[tail - 1])) {
            size_t first_result = tail - 1;
            while (first_result > *query_order + 1 &&
                   (is_tool_result_message(prompt, messages[first_result - 1]) ||
                    is_meta_only_message(kind, prompt,
                                         messages[first_result - 1]))) {
                --first_result;
            }
            suffix_order = first_result;
            if (suffix_order > *query_order + 1 &&
                messages[suffix_order - 1].role == "assistant" &&
                valid_segment_bounds(prompt, messages[suffix_order - 1])) {
                --suffix_order;
            }
        }
        const HarnessRenderedMessageSpan &suffix = messages[suffix_order];
        if (valid_segment_bounds(prompt, suffix)) {
            plan.live_suffix_message_begin_index = suffix.message_index;
            plan.history_end = suffix.segment_begin;
            plan.live_suffix_span = HarnessByteSpan{
                suffix.segment_begin, prompt.size(),
                HarnessSpanReason::LiveToolTrajectory};
        }
    }

    std::sort(plan.spans.begin(), plan.spans.end(),
              [](const HarnessByteSpan &a, const HarnessByteSpan &b) {
                  if (a.begin != b.begin) return a.begin < b.begin;
                  if (a.end != b.end) return a.end < b.end;
                  return static_cast<int>(a.reason) <
                      static_cast<int>(b.reason);
              });
    plan.spans.erase(
        std::unique(plan.spans.begin(), plan.spans.end(),
                    [](const HarnessByteSpan &a, const HarnessByteSpan &b) {
                        return a.begin == b.begin && a.end == b.end &&
                            a.reason == b.reason;
                    }),
        plan.spans.end());
    return plan;
}

} // namespace detail
} // namespace qw3
