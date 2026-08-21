#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qw3 {
namespace detail {

enum class HarnessKind {
    None = 0,
    ClaudeCode,
    OpenCode,
    DeepSeekHarness,
    GenericToolClient,
};

enum class HarnessProtocol {
    OpenAIChat = 0,
    AnthropicMessages,
};

struct HarnessRequestSignals {
    HarnessProtocol protocol = HarnessProtocol::OpenAIChat;
    std::string user_agent;
    bool has_deepseek_harness_header = false;
    bool has_opencode_header = false;
    bool has_tools = false;
    bool has_leading_control_message = false;
    bool compact = false;
};

struct HarnessRequestContext {
    HarnessKind kind = HarnessKind::None;
    bool compact = false;
};

HarnessRequestContext classify_harness(
    const HarnessRequestSignals &signals);

const char *harness_kind_name(HarnessKind kind);

// One message after it has been rendered into the model prompt. Content
// coordinates identify the exact request-supplied bytes inside the surrounding
// chat-template segment.
struct HarnessRenderedMessageSpan {
    size_t message_index = 0;
    std::string role;
    size_t segment_begin = 0;
    size_t segment_end = 0;
    size_t content_begin = 0;
    size_t content_end = 0;
};

enum class HarnessSpanReason {
    SystemControl = 0,
    CurrentQuery,
    LiveToolTrajectory,
    ProjectPolicy,
    RootTask,
    VisualEvidence,
};

struct HarnessByteSpan {
    size_t begin = 0;
    size_t end = 0;
    HarnessSpanReason reason = HarnessSpanReason::CurrentQuery;
};

struct HarnessSemanticPlan {
    // The first durable user task in the agent run.  This remains stable when
    // a later user turn merely says to continue/finalize the original task.
    std::optional<size_t> root_task_message_index;
    std::optional<size_t> current_query_message_index;
    std::optional<size_t> live_suffix_message_begin_index;
    std::optional<HarnessByteSpan> live_suffix_span;
    // Historical retrieval candidates end immediately before the live suffix.
    // Zero means that no conversational boundary could be derived.
    size_t history_end = 0;
    std::vector<HarnessByteSpan> spans;
    size_t project_policy_frame_count = 0;
    size_t visual_evidence_frame_count = 0;
};

// True only when the whole message is composed of harness-owned reminder
// frames and whitespace. Generic OpenAI-compatible clients require a known
// persistent-policy signature; recognized harnesses may also send turn-local
// reminder frames.
bool harness_message_is_meta_only(HarnessKind kind,
                                  std::string_view content);

// Build byte-accurate semantic lifetimes from an already-rendered prompt.
// Stable control, the root task, the exact current instruction, and the latest
// durable policy frames become bounded mandatory spans. When root and current
// are the same message only CurrentQuery is emitted. The unfinished tool round is returned
// separately as live_suffix_span: it is replayed after semantic selection but
// is never confused with the score-query span or the completed tool history.
HarnessSemanticPlan derive_harness_semantic_plan(
    HarnessKind kind,
    std::string_view prompt,
    size_t control_prefix_end,
    const std::vector<HarnessRenderedMessageSpan> &messages);

} // namespace detail
} // namespace qw3
