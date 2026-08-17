#include "anthropic_adapter.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qw3 {
namespace detail {
namespace {

using json = AnthropicJson;

std::string text_content(const json &content, std::string &error) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_null()) return {};
    if (!content.is_array()) {
        error = "content must be a string or an array of content blocks";
        return {};
    }
    std::string out;
    for (const json &block : content) {
        if (!block.is_object()) {
            error = "content blocks must be objects";
            return {};
        }
        const std::string type = block.value("type", "");
        if (type != "text" || !block.contains("text") ||
            !block["text"].is_string()) {
            error = "only text blocks are supported in this content field";
            return {};
        }
        out += block["text"].get<std::string>();
    }
    return out;
}

bool append_system_messages(const json &system, json &messages,
                            std::string &error) {
    if (system.is_null()) return true;
    const std::string text = text_content(system, error);
    if (!error.empty()) {
        error = "system: " + error;
        return false;
    }
    if (!text.empty()) {
        messages.push_back(json{{"role", "system"}, {"content", text}});
    }
    return true;
}

bool append_user_message(const json &content, json &messages,
                         std::string &error) {
    if (content.is_string()) {
        messages.push_back(
            json{{"role", "user"}, {"content", content}});
        return true;
    }
    if (!content.is_array()) {
        error = "user message content must be a string or an array";
        return false;
    }

    std::string pending_text;
    auto flush_text = [&]() {
        if (pending_text.empty()) return;
        messages.push_back(
            json{{"role", "user"}, {"content", pending_text}});
        pending_text.clear();
    };
    for (const json &block : content) {
        if (!block.is_object()) {
            error = "user content blocks must be objects";
            return false;
        }
        const std::string type = block.value("type", "");
        if (type == "text") {
            if (!block.contains("text") || !block["text"].is_string()) {
                error = "text blocks require a string text field";
                return false;
            }
            pending_text += block["text"].get<std::string>();
            continue;
        }
        if (type == "tool_result") {
            flush_text();
            if (!block.contains("tool_use_id") ||
                !block["tool_use_id"].is_string()) {
                error = "tool_result blocks require a string tool_use_id";
                return false;
            }
            std::string tool_error;
            std::string result = block.contains("content")
                ? text_content(block["content"], tool_error)
                : std::string();
            if (!tool_error.empty()) {
                error = "tool_result: " + tool_error;
                return false;
            }
            if (block.value("is_error", false)) {
                result = "[tool_error]\n" + result;
            }
            messages.push_back(json{
                {"role", "tool"},
                {"tool_call_id", block["tool_use_id"]},
                {"content", result}});
            continue;
        }
        error = "unsupported user content block type: " + type;
        return false;
    }
    flush_text();
    if (content.empty()) {
        messages.push_back(json{{"role", "user"}, {"content", ""}});
    }
    return true;
}

bool append_assistant_message(const json &content, json &messages,
                              std::string &error) {
    if (content.is_string()) {
        messages.push_back(
            json{{"role", "assistant"}, {"content", content}});
        return true;
    }
    if (!content.is_array()) {
        error = "assistant message content must be a string or an array";
        return false;
    }

    std::string text;
    std::string reasoning;
    json tool_calls = json::array();
    for (const json &block : content) {
        if (!block.is_object()) {
            error = "assistant content blocks must be objects";
            return false;
        }
        const std::string type = block.value("type", "");
        if (type == "text") {
            if (!block.contains("text") || !block["text"].is_string()) {
                error = "text blocks require a string text field";
                return false;
            }
            text += block["text"].get<std::string>();
        } else if (type == "thinking") {
            if (!block.contains("thinking") ||
                !block["thinking"].is_string()) {
                error = "thinking blocks require a string thinking field";
                return false;
            }
            reasoning += block["thinking"].get<std::string>();
        } else if (type == "tool_use") {
            if (!block.contains("id") || !block["id"].is_string() ||
                !block.contains("name") || !block["name"].is_string() ||
                !block.contains("input") || !block["input"].is_object()) {
                error = "tool_use blocks require string id/name and object input";
                return false;
            }
            tool_calls.push_back(json{
                {"id", block["id"]},
                {"type", "function"},
                {"function", json{{"name", block["name"]},
                                   {"arguments", block["input"].dump()}}}});
        } else {
            error = "unsupported assistant content block type: " + type;
            return false;
        }
    }

    json message{{"role", "assistant"}, {"content", text}};
    if (!reasoning.empty()) message["reasoning_content"] = reasoning;
    if (!tool_calls.empty()) message["tool_calls"] = std::move(tool_calls);
    messages.push_back(std::move(message));
    return true;
}

bool convert_messages(const json &input, json &output, std::string &error) {
    if (!input.is_array()) {
        error = "messages must be an array";
        return false;
    }
    for (const json &message : input) {
        if (!message.is_object() || !message.contains("role") ||
            !message["role"].is_string() || !message.contains("content")) {
            error = "each message requires string role and content";
            return false;
        }
        const std::string role = message["role"].get<std::string>();
        if (role == "user") {
            if (!append_user_message(message["content"], output, error)) {
                return false;
            }
        } else if (role == "assistant") {
            if (!append_assistant_message(message["content"], output,
                                          error)) {
                return false;
            }
        } else {
            error = "message role must be user or assistant";
            return false;
        }
    }
    return true;
}

bool convert_tools(const json &input, json &output, std::string &error) {
    if (!input.is_array()) {
        error = "tools must be an array";
        return false;
    }
    output = json::array();
    for (const json &tool : input) {
        if (!tool.is_object() || !tool.contains("name") ||
            !tool["name"].is_string() || !tool.contains("input_schema") ||
            !tool["input_schema"].is_object()) {
            error = "each tool requires string name and object input_schema";
            return false;
        }
        json fn{{"name", tool["name"]},
                {"parameters", tool["input_schema"]}};
        if (tool.contains("description")) {
            if (!tool["description"].is_string()) {
                error = "tool description must be a string";
                return false;
            }
            fn["description"] = tool["description"];
        }
        output.push_back(
            json{{"type", "function"}, {"function", std::move(fn)}});
    }
    return true;
}

bool convert_tool_choice(const json &input, json &output,
                         std::string &error) {
    if (!input.is_object()) {
        error = "tool_choice must be an object";
        return false;
    }
    const std::string type = input.value("type", "");
    if (type == "auto") {
        output = "auto";
    } else if (type == "any") {
        output = "required";
    } else if (type == "none") {
        output = "none";
    } else if (type == "tool") {
        if (!input.contains("name") || !input["name"].is_string() ||
            input["name"].get<std::string>().empty()) {
            error = "tool_choice type=tool requires a non-empty name";
            return false;
        }
        output = json{{"type", "function"},
                      {"function", json{{"name", input["name"]}}}};
    } else {
        error = "tool_choice.type must be auto, any, none, or tool";
        return false;
    }
    return true;
}

bool valid_positive_int(const json &value, int64_t &parsed) {
    if (!value.is_number_integer()) return false;
    parsed = value.get<int64_t>();
    return parsed > 0 && parsed <= 2147483647;
}

std::string opaque_signature(const std::string &thinking) {
    // This is a local compatibility token, not an Anthropic cryptographic
    // signature. It only needs to remain opaque and stable when Claude Code
    // returns the block in the next request.
    uint64_t a = 1469598103934665603ull;
    uint64_t b = 1099511628211ull;
    for (unsigned char c : thinking) {
        a = (a ^ c) * 1099511628211ull;
        b ^= static_cast<uint64_t>(c) + 0x9e3779b97f4a7c15ull +
             (b << 6) + (b >> 2);
    }
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << a
        << std::setw(16) << b;
    return out.str();
}

std::string anthropic_stop_reason(const std::string &finish_reason) {
    if (finish_reason == "tool_calls") return "tool_use";
    if (finish_reason == "length") return "max_tokens";
    if (finish_reason == "stop_sequence") return "stop_sequence";
    if (finish_reason == "pause_turn") return "pause_turn";
    if (finish_reason == "refusal") return "refusal";
    return "end_turn";
}

std::string message_id_from_openai(const std::string &id) {
    if (id.rfind("msg_", 0) == 0) return id;
    const size_t dash = id.find('-');
    return "msg_" + (dash == std::string::npos ? id : id.substr(dash + 1));
}

} // namespace

static bool anthropic_request_to_openai_impl(
        const json &request, json &openai_request,
        std::string &error, bool require_max_tokens) {
    error.clear();
    if (!request.is_object()) {
        error = "request body must be a JSON object";
        return false;
    }
    if (!request.contains("model") || !request["model"].is_string() ||
        request["model"].get<std::string>().empty()) {
        error = "model must be a non-empty string";
        return false;
    }
    if (!request.contains("messages")) {
        error = "messages is required";
        return false;
    }
    if (require_max_tokens && !request.contains("max_tokens")) {
        error = "max_tokens is required";
        return false;
    }

    openai_request = json::object();
    openai_request["model"] = request["model"];
    json messages = json::array();
    if (request.contains("system") &&
        !append_system_messages(request["system"], messages, error)) {
        return false;
    }
    if (!convert_messages(request["messages"], messages, error)) return false;
    openai_request["messages"] = std::move(messages);

    if (request.contains("max_tokens")) {
        int64_t max_tokens = 0;
        if (!valid_positive_int(request["max_tokens"], max_tokens)) {
            error = "max_tokens must be a positive integer";
            return false;
        }
        openai_request["max_tokens"] = max_tokens;
    }
    for (const char *key : {"temperature", "top_p", "top_k"}) {
        if (request.contains(key)) openai_request[key] = request[key];
    }
    if (request.contains("stop_sequences")) {
        if (!request["stop_sequences"].is_array()) {
            error = "stop_sequences must be an array";
            return false;
        }
        for (const json &stop : request["stop_sequences"]) {
            if (!stop.is_string()) {
                error = "stop_sequences entries must be strings";
                return false;
            }
        }
        openai_request["stop"] = request["stop_sequences"];
    }
    if (request.contains("stream")) {
        if (!request["stream"].is_boolean()) {
            error = "stream must be a boolean";
            return false;
        }
        openai_request["stream"] = request["stream"];
    }
    if (request.contains("tools")) {
        json tools;
        if (!convert_tools(request["tools"], tools, error)) return false;
        openai_request["tools"] = std::move(tools);
    }
    if (request.contains("tool_choice")) {
        json choice;
        if (!convert_tool_choice(request["tool_choice"], choice, error)) {
            return false;
        }
        openai_request["tool_choice"] = std::move(choice);
    }
    if (request.contains("thinking")) {
        const json &thinking = request["thinking"];
        if (!thinking.is_object() || !thinking.contains("type") ||
            !thinking["type"].is_string()) {
            error = "thinking requires a string type";
            return false;
        }
        const std::string type = thinking["type"].get<std::string>();
        if (type == "disabled") {
            openai_request["enable_thinking"] = false;
        } else if (type == "enabled" || type == "adaptive") {
            openai_request["enable_thinking"] = true;
            if (thinking.contains("budget_tokens")) {
                int64_t budget = 0;
                if (!valid_positive_int(thinking["budget_tokens"], budget)) {
                    error = "thinking.budget_tokens must be a positive integer";
                    return false;
                }
                openai_request["thinking_budget"] = budget;
            }
        } else {
            error = "thinking.type must be enabled, disabled, or adaptive";
            return false;
        }
    }

    // Always request the final OpenAI usage chunk so the streaming adapter can
    // populate Anthropic's message_delta usage.
    if (openai_request.value("stream", false)) {
        openai_request["stream_options"] =
            json{{"include_usage", true}};
    }
    return true;
}

bool anthropic_request_to_openai(const json &request, json &openai_request,
                                 std::string &error,
                                 bool require_max_tokens) {
    try {
        return anthropic_request_to_openai_impl(
            request, openai_request, error, require_max_tokens);
    } catch (const std::exception &e) {
        openai_request = json::object();
        error = std::string("invalid Anthropic request field: ") + e.what();
        return false;
    }
}

json anthropic_response_from_openai(const json &openai_response,
                                    const std::string &requested_model) {
    if (!openai_response.is_object() ||
        !openai_response.contains("choices") ||
        !openai_response["choices"].is_array() ||
        openai_response["choices"].empty() ||
        !openai_response["choices"][0].is_object() ||
        !openai_response["choices"][0].contains("message") ||
        !openai_response["choices"][0]["message"].is_object()) {
        throw std::invalid_argument("malformed OpenAI chat response");
    }
    const json &choice = openai_response["choices"][0];
    const json &message = choice["message"];
    json content = json::array();
    if (message.contains("reasoning_content") &&
        message["reasoning_content"].is_string() &&
        !message["reasoning_content"].get<std::string>().empty()) {
        const std::string thinking =
            message["reasoning_content"].get<std::string>();
        content.push_back(json{{"type", "thinking"},
                               {"thinking", thinking},
                               {"signature", opaque_signature(thinking)}});
    }
    if (message.contains("content") && message["content"].is_string() &&
        !message["content"].get<std::string>().empty()) {
        content.push_back(json{{"type", "text"},
                               {"text", message["content"]}});
    }
    if (message.contains("tool_calls") &&
        message["tool_calls"].is_array()) {
        for (const json &call : message["tool_calls"]) {
            if (!call.is_object() || !call.contains("function") ||
                !call["function"].is_object()) {
                continue;
            }
            const json &fn = call["function"];
            json input = json::object();
            try {
                input = json::parse(fn.value("arguments", std::string("{}")));
                if (!input.is_object()) input = json::object();
            } catch (...) {
                input = json::object();
            }
            content.push_back(json{{"type", "tool_use"},
                                   {"id", call.value("id", "")},
                                   {"name", fn.value("name", "")},
                                   {"input", std::move(input)}});
        }
    }
    if (content.empty()) {
        content.push_back(json{{"type", "text"}, {"text", ""}});
    }

    const json usage = openai_response.value("usage", json::object());
    const std::string openai_id = openai_response.value("id", "message");
    const std::string model = openai_response.value(
        "model", requested_model.empty() ? std::string("qw3")
                                          : requested_model);
    return json{
        {"id", message_id_from_openai(openai_id)},
        {"type", "message"},
        {"role", "assistant"},
        {"content", std::move(content)},
        {"model", model},
        {"stop_reason", anthropic_stop_reason(
             choice.value("finish_reason", std::string("stop")))},
        {"stop_sequence", nullptr},
        {"usage", json{{"input_tokens", usage.value("prompt_tokens", 0)},
                        {"output_tokens",
                         usage.value("completion_tokens", 0)}}}};
}

json anthropic_error_body(const std::string &message,
                          const std::string &type) {
    return json{{"type", "error"},
                {"error", json{{"type", type}, {"message", message}}}};
}

AnthropicSseAdapter::AnthropicSseAdapter(std::string requested_model,
                                         size_t input_tokens)
    : requested_model_(std::move(requested_model)),
      input_tokens_(input_tokens) {}

bool AnthropicSseAdapter::emit_event(const char *event_name,
                                     const json &payload,
                                     const Emit &emit) const {
    const std::string wire = std::string("event: ") + event_name +
        "\ndata: " + payload.dump() + "\n\n";
    return emit(wire);
}

bool AnthropicSseAdapter::ensure_started(const json *chunk,
                                         const Emit &emit) {
    if (started_) return true;
    if (chunk) {
        message_id_ = message_id_from_openai(
            chunk->value("id", std::string("message")));
        if (requested_model_.empty()) {
            requested_model_ = chunk->value("model", std::string("qw3"));
        }
    }
    if (message_id_.empty()) message_id_ = "msg_message";
    if (requested_model_.empty()) requested_model_ = "qw3";
    started_ = true;
    return emit_event(
        "message_start",
        json{{"type", "message_start"},
             {"message", json{{"id", message_id_},
                              {"type", "message"},
                              {"role", "assistant"},
                              {"content", json::array()},
                              {"model", requested_model_},
                              {"stop_reason", nullptr},
                              {"stop_sequence", nullptr},
                              {"usage", json{{"input_tokens", input_tokens_},
                                             {"output_tokens", 0}}}}}},
        emit);
}

bool AnthropicSseAdapter::close_block(const Emit &emit) {
    if (block_ == BlockKind::None) return true;
    if (block_ == BlockKind::Thinking) {
        if (!emit_event(
                "content_block_delta",
                json{{"type", "content_block_delta"},
                     {"index", next_block_index_ - 1},
                     {"delta", json{{"type", "signature_delta"},
                                    {"signature",
                                     opaque_signature(reasoning_)}}}},
                emit)) {
            return false;
        }
    }
    if (!emit_event("content_block_stop",
                    json{{"type", "content_block_stop"},
                         {"index", next_block_index_ - 1}},
                    emit)) {
        return false;
    }
    block_ = BlockKind::None;
    active_tool_index_ = -1;
    return true;
}

bool AnthropicSseAdapter::open_block(BlockKind kind,
                                     const json *tool_delta,
                                     const Emit &emit,
                                     std::string &error) {
    if (block_ == kind && kind != BlockKind::Tool) return true;
    if (!close_block(emit)) return false;
    json content_block;
    if (kind == BlockKind::Thinking) {
        content_block = json{{"type", "thinking"}, {"thinking", ""}};
    } else if (kind == BlockKind::Text) {
        content_block = json{{"type", "text"}, {"text", ""}};
    } else if (kind == BlockKind::Tool) {
        if (!tool_delta || !tool_delta->is_object() ||
            !tool_delta->contains("function") ||
            !(*tool_delta)["function"].is_object()) {
            error = "tool-call delta is missing function metadata";
            return false;
        }
        const json &fn = (*tool_delta)["function"];
        const int index = tool_delta->value("index", 0);
        active_tool_index_ = index;
        content_block = json{{"type", "tool_use"},
                             {"id", tool_delta->value("id", "")},
                             {"name", fn.value("name", "")},
                             {"input", json::object()}};
    } else {
        error = "invalid Anthropic content block state";
        return false;
    }
    const size_t index = next_block_index_++;
    block_ = kind;
    emitted_any_block_ = true;
    return emit_event("content_block_start",
                      json{{"type", "content_block_start"},
                           {"index", index},
                           {"content_block", std::move(content_block)}},
                      emit);
}

bool AnthropicSseAdapter::process_json(const json &chunk,
                                       const Emit &emit,
                                       std::string &error) {
    if (completed_) return true;
    if (chunk.contains("error")) {
        std::string message = "upstream generation error";
        if (chunk["error"].is_string()) {
            message = chunk["error"].get<std::string>();
        } else if (chunk["error"].is_object()) {
            message = chunk["error"].value("message", message);
        }
        const bool emitted = emit_event(
            "error", anthropic_error_body(message, "api_error"), emit);
        // An Anthropic stream-level error is terminal. The wrapped OpenAI
        // provider still writes [DONE] for its own protocol, but translating
        // that marker into message_delta/message_stop would falsely turn the
        // failed request into a successful response.
        completed_ = true;
        return emitted;
    }
    if (!ensure_started(&chunk, emit)) return false;
    if (chunk.contains("usage") && chunk["usage"].is_object()) {
        input_tokens_ = chunk["usage"].value("prompt_tokens", input_tokens_);
        output_tokens_ =
            chunk["usage"].value("completion_tokens", output_tokens_);
    }
    if (!chunk.contains("choices") || !chunk["choices"].is_array() ||
        chunk["choices"].empty()) {
        return true;
    }
    const json &choice = chunk["choices"][0];
    if (!choice.is_object()) return true;
    const json delta = choice.value("delta", json::object());
    if (delta.contains("reasoning_content") &&
        delta["reasoning_content"].is_string() &&
        !delta["reasoning_content"].get<std::string>().empty()) {
        if (!open_block(BlockKind::Thinking, nullptr, emit, error)) {
            return false;
        }
        const std::string value =
            delta["reasoning_content"].get<std::string>();
        reasoning_ += value;
        if (!emit_event(
                "content_block_delta",
                json{{"type", "content_block_delta"},
                     {"index", next_block_index_ - 1},
                     {"delta", json{{"type", "thinking_delta"},
                                    {"thinking", value}}}},
                emit)) {
            return false;
        }
    }
    if (delta.contains("content") && delta["content"].is_string() &&
        !delta["content"].get<std::string>().empty()) {
        if (!open_block(BlockKind::Text, nullptr, emit, error)) return false;
        if (!emit_event(
                "content_block_delta",
                json{{"type", "content_block_delta"},
                     {"index", next_block_index_ - 1},
                     {"delta", json{{"type", "text_delta"},
                                    {"text", delta["content"]}}}},
                emit)) {
            return false;
        }
    }
    if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
        for (const json &tool_delta : delta["tool_calls"]) {
            if (!tool_delta.is_object()) continue;
            const int index = tool_delta.value("index", 0);
            const bool has_metadata = tool_delta.contains("id") ||
                (tool_delta.contains("function") &&
                 tool_delta["function"].is_object() &&
                 tool_delta["function"].contains("name"));
            if (block_ != BlockKind::Tool ||
                active_tool_index_ != index || has_metadata) {
                if (!open_block(BlockKind::Tool, &tool_delta, emit, error)) {
                    return false;
                }
            }
            if (tool_delta.contains("function") &&
                tool_delta["function"].is_object()) {
                const json &fn = tool_delta["function"];
                if (fn.contains("arguments") &&
                    fn["arguments"].is_string() &&
                    !fn["arguments"].get<std::string>().empty()) {
                    if (!emit_event(
                            "content_block_delta",
                            json{{"type", "content_block_delta"},
                                 {"index", next_block_index_ - 1},
                                 {"delta", json{
                                      {"type", "input_json_delta"},
                                      {"partial_json", fn["arguments"]}}}},
                            emit)) {
                        return false;
                    }
                }
            }
        }
    }
    if (choice.contains("finish_reason") &&
        !choice["finish_reason"].is_null()) {
        if (!choice["finish_reason"].is_string()) {
            error = "OpenAI finish_reason must be a string or null";
            return false;
        }
        finish_reason_ = choice["finish_reason"].get<std::string>();
        if (!close_block(emit)) return false;
    }
    return true;
}

bool AnthropicSseAdapter::complete_message(const Emit &emit) {
    if (completed_) return true;
    if (!ensure_started(nullptr, emit)) return false;
    if (!close_block(emit)) return false;
    if (!emitted_any_block_) {
        std::string unused_error;
        if (!open_block(BlockKind::Text, nullptr, emit, unused_error) ||
            !close_block(emit)) {
            return false;
        }
    }
    const std::string reason = anthropic_stop_reason(
        finish_reason_.empty() ? std::string("stop") : finish_reason_);
    if (!emit_event(
            "message_delta",
            json{{"type", "message_delta"},
                 {"delta", json{{"stop_reason", reason},
                                {"stop_sequence", nullptr}}},
                 {"usage", json{{"output_tokens", output_tokens_}}}},
            emit)) {
        return false;
    }
    if (!emit_event("message_stop", json{{"type", "message_stop"}}, emit)) {
        return false;
    }
    completed_ = true;
    return true;
}

bool AnthropicSseAdapter::process_event(const std::string &event,
                                        const Emit &emit,
                                        std::string &error) {
    std::istringstream lines(event);
    std::string line;
    std::string data;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data:", 0) != 0) continue;
        std::string value = line.substr(5);
        if (!value.empty() && value.front() == ' ') value.erase(value.begin());
        if (!data.empty()) data += "\n";
        data += value;
    }
    if (data.empty()) return true;
    if (data == "[DONE]") return complete_message(emit);
    try {
        return process_json(json::parse(data), emit, error);
    } catch (const std::exception &e) {
        error = std::string("invalid OpenAI SSE JSON: ") + e.what();
        return false;
    }
}

bool AnthropicSseAdapter::feed(const char *data, size_t size,
                               const Emit &emit, std::string &error) {
    if (completed_) return true;
    pending_.append(data, size);
    for (;;) {
        const size_t split = pending_.find("\n\n");
        if (split == std::string::npos) break;
        std::string event = pending_.substr(0, split);
        pending_.erase(0, split + 2);
        if (!process_event(event, emit, error)) return false;
    }
    return true;
}

bool AnthropicSseAdapter::finish(const Emit &emit, std::string &error) {
    if (!pending_.empty()) {
        std::string event;
        event.swap(pending_);
        if (!process_event(event, emit, error)) return false;
    }
    return complete_message(emit);
}

} // namespace detail
} // namespace qw3
