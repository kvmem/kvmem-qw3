#include "anthropic_adapter.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using qw3::detail::AnthropicSseAdapter;
using qw3::detail::anthropic_request_to_openai;
using qw3::detail::anthropic_response_from_openai;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "anthropic_adapter_test: " << message << "\n";
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

std::vector<json> parse_named_sse(const std::string &wire) {
    std::vector<json> events;
    size_t cursor = 0;
    while (cursor < wire.size()) {
        const size_t end = wire.find("\n\n", cursor);
        const std::string record = wire.substr(
            cursor, end == std::string::npos ? std::string::npos
                                             : end - cursor);
        const size_t data = record.find("data: ");
        if (data != std::string::npos) {
            events.push_back(json::parse(record.substr(data + 6)));
        }
        if (end == std::string::npos) break;
        cursor = end + 2;
    }
    return events;
}

void test_request_conversion() {
    const json request = {
        {"model", "Qwen3.8-27B"},
        {"max_tokens", 1024},
        {"stream", true},
        {"system", json::array({
             json{{"type", "text"}, {"text", "base policy"},
                  {"cache_control", json{{"type", "ephemeral"}}}}})},
        {"thinking", json{{"type", "enabled"},
                           {"budget_tokens", 256}}},
        {"tools", json::array({json{
             {"name", "Bash"},
             {"description", "Run a command"},
             {"input_schema", json{{"type", "object"},
                                    {"properties", json{{"command", json{{"type", "string"}}}}},
                                    {"required", json::array({"command"})}}}}})},
        {"tool_choice", json{{"type", "auto"}}},
        {"messages", json::array({
             json{{"role", "user"}, {"content", "inspect the repo"}},
             json{{"role", "assistant"},
                  {"content", json::array({
                       json{{"type", "thinking"},
                            {"thinking", "I should inspect it"},
                            {"signature", "opaque"}},
                       json{{"type", "tool_use"}, {"id", "toolu_1"},
                            {"name", "Bash"},
                            {"input", json{{"command", "pwd"}}}}})}},
             json{{"role", "user"},
                  {"content", json::array({
                       json{{"type", "tool_result"},
                            {"tool_use_id", "toolu_1"},
                            {"content", "/tmp/repo"},
                            {"is_error", false}}})}}})}
    };

    json converted;
    std::string error;
    require(anthropic_request_to_openai(request, converted, error), error);
    require(converted["messages"].size() == 4,
            "system/user/assistant/tool messages were not preserved");
    require(converted["messages"][0]["role"] == "system",
            "system message missing");
    require(converted["messages"][2]["reasoning_content"] ==
                "I should inspect it",
            "thinking did not map to reasoning_content");
    require(converted["messages"][2]["tool_calls"][0]["id"] == "toolu_1",
            "tool_use id was not preserved");
    require(converted["messages"][3]["role"] == "tool",
            "tool_result did not map to a tool message");
    require(converted["tools"][0]["function"]["parameters"]["required"][0] ==
                "command",
            "input_schema did not map to function parameters");
    require(converted["enable_thinking"] == true &&
                converted["thinking_budget"] == 256,
            "thinking configuration did not map");
    require(converted["stream_options"]["include_usage"] == true,
            "streaming conversion did not request usage");
    for (auto it = converted.begin(); it != converted.end(); ++it) {
        require(it.key().rfind("_qw3_", 0) != 0,
                "private server metadata leaked into converted request");
    }
}

void test_billing_marker_does_not_invalidate_prompt_prefix() {
    auto request = [](const std::string &cch, bool cache_control) {
        json user = json{{"type", "text"}, {"text", "do the task"}};
        if (cache_control) {
            user["cache_control"] = json{{"type", "ephemeral"}};
        }
        return json{
            {"model", "qwen"},
            {"max_tokens", 64},
            {"system", json::array({
                 json{{"type", "text"},
                      {"text", "x-anthropic-billing-header: cc_version=2.1.148; cch=" + cch + ";"}},
                 json{{"type", "text"}, {"text", "stable policy"}}})},
            {"messages", json::array({json{
                 {"role", "user"},
                 {"content", json::array({user})}}})}
        };
    };

    json first;
    json second;
    std::string error;
    require(anthropic_request_to_openai(
                request("first", true), first, error), error);
    require(anthropic_request_to_openai(
                request("second", false), second, error), error);
    require(first["messages"] == second["messages"],
            "Claude transport-only marker changed the model prompt prefix");
    require(first["messages"].size() == 2 &&
                first["messages"][0]["role"] == "system" &&
                first["messages"][0]["content"] == "stable policy",
            "billing marker was not removed without disturbing real policy");
}

void test_rejects_unsupported_image() {
    const json request = {
        {"model", "qwen"},
        {"max_tokens", 64},
        {"messages", json::array({json{
             {"role", "user"},
             {"content", json::array({json{{"type", "image"},
                                             {"source", json::object()}}})}}})}
    };
    json converted;
    std::string error;
    require(!anthropic_request_to_openai(request, converted, error),
            "image input was silently accepted");
    require(error.find("unsupported user content block type") !=
                std::string::npos,
            "image rejection was not explicit");
}

void test_tool_result_followed_by_reminder_preserves_order() {
    const json request = {
        {"model", "qwen"},
        {"max_tokens", 64},
        {"messages", json::array({json{
             {"role", "user"},
             {"content", json::array({
                  json{{"type", "tool_result"},
                       {"tool_use_id", "toolu_read"},
                       {"content", "file contents"}},
                  json{{"type", "text"},
                       {"text",
                        "<system-reminder>\nInstructions from: CLAUDE.md\n"
                        "Run focused tests.\n</system-reminder>"}}})}}})}
    };

    json converted;
    std::string error;
    require(anthropic_request_to_openai(request, converted, error), error);
    require(converted["messages"].size() == 2,
            "tool_result plus reminder did not become two ordered messages");
    require(converted["messages"][0]["role"] == "tool" &&
                converted["messages"][0]["tool_call_id"] == "toolu_read",
            "tool_result did not remain first in the converted trajectory");
    require(converted["messages"][1]["role"] == "user" &&
                converted["messages"][1]["content"]
                    .get<std::string>()
                    .find("Instructions from: CLAUDE.md") !=
                    std::string::npos,
            "post-tool Claude reminder was lost or reordered");
}

void test_malformed_field_type_returns_error() {
    const json request = {
        {"model", "qwen"},
        {"max_tokens", 64},
        {"messages", json::array({json{
             {"role", "user"},
             {"content", json::array({json{{"type", 7},
                                             {"text", "bad"}}})}}})}
    };
    json converted;
    std::string error;
    require(!anthropic_request_to_openai(request, converted, error),
            "malformed content-block type was accepted");
    require(error.find("invalid Anthropic request field") !=
                std::string::npos,
            "malformed field did not produce a structured conversion error");
}

void test_nonstream_response_conversion() {
    const json response = {
        {"id", "chatcmpl-abc"},
        {"model", "Qwen3.8-27B"},
        {"choices", json::array({json{
             {"finish_reason", "tool_calls"},
             {"message", json{
                  {"role", "assistant"},
                  {"content", ""},
                  {"reasoning_content", "use Bash"},
                  {"tool_calls", json::array({json{
                       {"id", "call_1"}, {"type", "function"},
                       {"function", json{{"name", "Bash"},
                                          {"arguments", "{\"command\":\"pwd\"}"}}}}})}}}}})},
        {"usage", json{{"prompt_tokens", 25000},
                        {"completion_tokens", 32}}}
    };
    const json converted = anthropic_response_from_openai(response);
    require(converted["type"] == "message" &&
                converted["stop_reason"] == "tool_use",
            "nonstream response envelope mismatch");
    require(converted["content"][0]["type"] == "thinking" &&
                converted["content"][0]["signature"].is_string(),
            "thinking block/signature missing");
    require(converted["content"][1]["type"] == "tool_use" &&
                converted["content"][1]["input"]["command"] == "pwd",
            "tool call did not map to tool_use");
    require(converted["usage"]["input_tokens"] == 25000 &&
                converted["usage"]["output_tokens"] == 32,
            "usage mapping mismatch");
}

void test_streaming_translation_one_byte_chunks() {
    const std::vector<json> chunks = {
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array({json{{"delta", json{{"role", "assistant"}}},
                                           {"finish_reason", nullptr}}})}},
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array({json{{"delta", json{{"reasoning_content", "inspect"}}},
                                           {"finish_reason", nullptr}}})}},
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array({json{{"delta", json{{"content", "ready"}}},
                                           {"finish_reason", nullptr}}})}},
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array({json{{"delta", json{{"tool_calls", json::array({json{
                                                {"index", 0}, {"id", "call_1"},
                                                {"type", "function"},
                                                {"function", json{{"name", "Bash"},
                                                                   {"arguments", "{"}}}}})}}},
                                           {"finish_reason", nullptr}}})}},
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array({json{{"delta", json{{"tool_calls", json::array({json{
                                                {"index", 0},
                                                {"function", json{{"arguments", "\"command\":\"pwd\"}"}}}}})}}},
                                           {"finish_reason", nullptr}}})}},
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array({json{{"delta", json::object()},
                                           {"finish_reason", "tool_calls"}}})}},
        json{{"id", "chatcmpl-stream"}, {"model", "Qwen3.8-27B"},
             {"choices", json::array()},
             {"usage", json{{"prompt_tokens", 24668},
                              {"completion_tokens", 83}}}}
    };
    std::string openai_wire;
    for (const json &chunk : chunks) {
        openai_wire += "data: " + chunk.dump() + "\n\n";
    }
    openai_wire += "data: [DONE]\n\n";

    AnthropicSseAdapter adapter("Qwen3.8-27B", 24668);
    std::string anthropic_wire;
    std::string error;
    auto emit = [&](const std::string &data) {
        anthropic_wire += data;
        return true;
    };
    for (char c : openai_wire) {
        if (!adapter.feed(&c, 1, emit, error)) fail(error);
    }
    if (!adapter.finish(emit, error)) fail(error);

    const std::vector<json> events = parse_named_sse(anthropic_wire);
    require(!events.empty() && events.front()["type"] == "message_start",
            "message_start missing");
    bool thinking_start = false;
    bool signature = false;
    bool text_delta = false;
    bool tool_start = false;
    bool partial_json = false;
    bool message_delta = false;
    for (const json &event : events) {
        if (event.value("type", "") == "content_block_start" &&
            event["content_block"].value("type", "") == "thinking") {
            thinking_start = true;
        }
        if (event.value("type", "") == "content_block_delta" &&
            event["delta"].value("type", "") == "signature_delta") {
            signature = true;
        }
        if (event.value("type", "") == "content_block_delta" &&
            event["delta"].value("type", "") == "text_delta" &&
            event["delta"].value("text", "") == "ready") {
            text_delta = true;
        }
        if (event.value("type", "") == "content_block_start" &&
            event["content_block"].value("type", "") == "tool_use") {
            tool_start = event["content_block"].value("name", "") == "Bash";
        }
        if (event.value("type", "") == "content_block_delta" &&
            event["delta"].value("type", "") == "input_json_delta") {
            partial_json = true;
        }
        if (event.value("type", "") == "message_delta") {
            message_delta = event["delta"]["stop_reason"] == "tool_use" &&
                event["usage"]["output_tokens"] == 83;
        }
    }
    require(thinking_start && signature, "thinking stream was incomplete");
    require(text_delta, "text stream was incomplete");
    require(tool_start && partial_json, "tool_use stream was incomplete");
    require(message_delta && events.back()["type"] == "message_stop",
            "stream termination mismatch");
}

void test_streaming_error_is_terminal() {
    const json role = {
        {"id", "chatcmpl-error"}, {"model", "qwen"},
        {"choices", json::array({json{
             {"delta", json{{"role", "assistant"}}},
             {"finish_reason", nullptr}}})}
    };
    const json error_chunk = {
        {"id", "chatcmpl-error"}, {"model", "qwen"},
        {"choices", json::array({json{
             {"delta", json::object()}, {"finish_reason", "error"}}})},
        {"error", "backend failed"}
    };
    const std::string wire =
        "data: " + role.dump() + "\n\n" +
        "data: " + error_chunk.dump() + "\n\n" +
        "data: [DONE]\n\n";
    AnthropicSseAdapter adapter("qwen", 7);
    std::string translated;
    std::string error;
    auto emit = [&](const std::string &data) {
        translated += data;
        return true;
    };
    require(adapter.feed(wire.data(), wire.size(), emit, error), error);
    require(adapter.finish(emit, error), error);
    const std::vector<json> events = parse_named_sse(translated);
    require(events.size() == 2 && events[0]["type"] == "message_start" &&
                events[1]["type"] == "error",
            "stream error was followed by a false successful completion");
}

} // namespace

int main() {
    test_request_conversion();
    test_billing_marker_does_not_invalidate_prompt_prefix();
    test_rejects_unsupported_image();
    test_tool_result_followed_by_reminder_preserves_order();
    test_malformed_field_type_returns_error();
    test_nonstream_response_conversion();
    test_streaming_translation_one_byte_chunks();
    test_streaming_error_is_terminal();
    std::cout << "anthropic_adapter_test: PASS\n";
    return 0;
}
