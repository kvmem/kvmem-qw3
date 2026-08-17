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
};

struct HarnessByteSpan {
    size_t begin = 0;
    size_t end = 0;
    HarnessSpanReason reason = HarnessSpanReason::CurrentQuery;
};

struct HarnessSemanticPlan {
    std::optional<size_t> current_query_message_index;
    std::vector<HarnessByteSpan> spans;
    size_t project_policy_frame_count = 0;
};

// True only when the whole message is composed of harness-owned reminder
// frames and whitespace. Generic OpenAI-compatible clients require a known
// persistent-policy signature; recognized harnesses may also send turn-local
// reminder frames.
bool harness_message_is_meta_only(HarnessKind kind,
                                  std::string_view content);

// Build byte-accurate mandatory regions from an already-rendered prompt.
// control_prefix_end is the first conversational byte, so it includes the
// template control tokens, tool schemas/protocol, and system/developer text.
HarnessSemanticPlan derive_harness_semantic_plan(
    HarnessKind kind,
    std::string_view prompt,
    size_t control_prefix_end,
    const std::vector<HarnessRenderedMessageSpan> &messages);

} // namespace detail
} // namespace qw3
