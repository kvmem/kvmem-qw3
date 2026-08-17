#pragma once

#include "json.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace qw3 {
namespace detail {

using AnthropicJson = nlohmann::json;

// Convert an Anthropic Messages request into the OpenAI-shaped request used by
// qw3's common chat path. Harness identity stays in the internal route context
// instead of being exposed as a private request-body field.
bool anthropic_request_to_openai(const AnthropicJson &request,
                                 AnthropicJson &openai_request,
                                 std::string &error,
                                 bool require_max_tokens = true);

// Convert a completed OpenAI ChatCompletions response into an Anthropic
// Messages response. Throws std::invalid_argument for malformed input.
AnthropicJson anthropic_response_from_openai(
    const AnthropicJson &openai_response,
    const std::string &requested_model = {});

AnthropicJson anthropic_error_body(const std::string &message,
                                   const std::string &type =
                                       "invalid_request_error");

// Incrementally translates OpenAI data-only SSE into named Anthropic Messages
// events. feed() accepts arbitrary byte chunks; finish() flushes a stream that
// ended without an explicit OpenAI [DONE] marker.
class AnthropicSseAdapter {
public:
    using Emit = std::function<bool(const std::string &)>;

    AnthropicSseAdapter(std::string requested_model,
                        size_t input_tokens);

    bool feed(const char *data, size_t size, const Emit &emit,
              std::string &error);
    bool finish(const Emit &emit, std::string &error);

private:
    enum class BlockKind {
        None,
        Thinking,
        Text,
        Tool,
    };

    bool process_event(const std::string &event, const Emit &emit,
                       std::string &error);
    bool process_json(const AnthropicJson &chunk, const Emit &emit,
                      std::string &error);
    bool ensure_started(const AnthropicJson *chunk, const Emit &emit);
    bool open_block(BlockKind kind, const AnthropicJson *tool_delta,
                    const Emit &emit, std::string &error);
    bool close_block(const Emit &emit);
    bool complete_message(const Emit &emit);
    bool emit_event(const char *event_name, const AnthropicJson &payload,
                    const Emit &emit) const;

    std::string requested_model_;
    std::string message_id_;
    std::string pending_;
    std::string reasoning_;
    std::string finish_reason_;
    size_t input_tokens_ = 0;
    size_t output_tokens_ = 0;
    size_t next_block_index_ = 0;
    int active_tool_index_ = -1;
    BlockKind block_ = BlockKind::None;
    bool started_ = false;
    bool completed_ = false;
    bool emitted_any_block_ = false;
};

} // namespace detail
} // namespace qw3
