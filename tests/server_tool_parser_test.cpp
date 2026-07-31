#include "../src/qw3_server.cpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using json = nlohmann::json;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "server_tool_parser_test: " << message << "\n";
    std::exit(1);
}

json string_tool(const std::string &name,
                 const json &properties,
                 const json &required) {
    return json{
        {"type", "function"},
        {"function",
         json{{"name", name},
              {"parameters",
               json{{"type", "object"},
                    {"properties", properties},
                    {"required", required},
                    {"additionalProperties", false}}}}}};
}

json write_tool(const std::string &name = "write") {
    return string_tool(
        name,
        json{{"path", json{{"type", "string"}}},
             {"content", json{{"type", "string"}}}},
        json::array({"path", "content"}));
}

void test_valid_and_no_intent() {
    const json tools = json::array({write_tool()});
    const auto none = qw3::parse_tool_calls_xml("ordinary answer", &tools);
    if (!none.valid || none.intent_detected || !none.calls.empty()) {
        fail("ordinary text was classified as malformed tool intent");
    }

    const auto parsed = qw3::parse_tool_calls_xml(
        "<tool_call><function=write>"
        "<parameter=path>/tmp/a</parameter>"
        "<parameter=content>hello</parameter>"
        "</function></tool_call>",
        &tools);
    if (!parsed.valid || !parsed.intent_detected ||
        parsed.calls.size() != 1) {
        fail("valid canonical tool call was rejected");
    }
}

void test_missing_function_inference_and_retry_case() {
    const std::string missing_function =
        "<tool_call>"
        "<parameter=path>/tmp/a</parameter>"
        "<parameter=content>hello</parameter>"
        "</tool_call>";

    const json unique_tools = json::array({write_tool()});
    const auto inferred =
        qw3::parse_tool_calls_xml(missing_function, &unique_tools);
    if (!inferred.valid || inferred.calls.size() != 1 ||
        inferred.calls.front()["function"].value("name", "") != "write") {
        fail("unambiguous missing function name was not recovered");
    }

    const json ambiguous_tools =
        json::array({write_tool("write"), write_tool("replace")});
    const auto ambiguous =
        qw3::parse_tool_calls_xml(missing_function, &ambiguous_tools);
    if (!ambiguous.intent_detected || ambiguous.valid ||
        !ambiguous.calls.empty()) {
        fail("ambiguous missing function name did not request retry");
    }
}

void test_schema_validation() {
    const json tools = json::array({write_tool()});
    const auto missing = qw3::parse_tool_calls_xml(
        "<tool_call><function=write>"
        "<parameter=path>/tmp/a</parameter>"
        "</function></tool_call>",
        &tools);
    if (missing.valid ||
        missing.error.find("missing required property") ==
            std::string::npos) {
        fail("missing required argument was accepted");
    }

    const json count_tools = json::array({string_tool(
        "count",
        json{{"value", json{{"type", "integer"}}}},
        json::array({"value"}))});
    const auto wrong_type = qw3::parse_tool_calls_xml(
        "<tool_call><function=count>"
        "<parameter=value>not-an-integer</parameter>"
        "</function></tool_call>",
        &count_tools);
    if (wrong_type.valid ||
        wrong_type.error.find("wrong JSON type") == std::string::npos) {
        fail("wrong schema type was accepted");
    }

    const auto unknown = qw3::parse_tool_calls_xml(
        "<tool_call><function=write>"
        "<parameter=path>/tmp/a</parameter>"
        "<parameter=content>hello</parameter>"
        "<parameter=extra>bad</parameter>"
        "</function></tool_call>",
        &tools);
    if (unknown.valid ||
        unknown.error.find("unknown property") == std::string::npos) {
        fail("additionalProperties=false was ignored");
    }
}

void test_incomplete_block_and_retry_prompt() {
    const json tools = json::array({write_tool()});
    const auto incomplete = qw3::parse_tool_calls_xml(
        "<tool_call><parameter=path>/tmp/a</parameter>", &tools);
    if (!incomplete.intent_detected || incomplete.valid) {
        fail("incomplete tool block did not request retry");
    }

    const json messages =
        json::array({json{{"role", "user"}, {"content", "write a file"}}});
    const std::string prompt = qw3::render_tool_retry_prompt(
        messages, &tools, false, false, {});
    if (prompt.find("previous response attempted a tool call") ==
            std::string::npos ||
        prompt.find("<function=...>") == std::string::npos) {
        fail("retry prompt is missing the generic repair instruction");
    }
}

void test_incremental_commit_gate() {
    const json tools = json::array({write_tool()});
    qw3::IncrementalToolCallStream stream(&tools);
    stream.feed(
        "<tool_call><function=write>"
        "<parameter=path>/tmp/a</parameter>");
    if (stream.ready_to_commit()) {
        fail("stream committed before every required parameter appeared");
    }
    stream.feed("<parameter=content>hello");
    if (!stream.ready_to_commit()) {
        fail("stream did not become committable after required parameters");
    }

    std::string suffix;
    stream.finish(true, suffix);
    if (stream.fatal() || suffix.empty()) {
        fail("committed stream closure repair failed");
    }
    const json call = stream.finalized_call();
    std::string validation_error;
    if (!qw3::validate_tool_call(call, &tools, validation_error)) {
        fail("repaired incremental call failed schema validation: " +
             validation_error);
    }
    const json arguments =
        json::parse(call["function"].value("arguments", ""));
    if (arguments.value("content", "") != "hello") {
        fail("repaired incremental call lost its content");
    }
}

void expect_query_range(const json &messages, size_t begin, size_t end,
                        bool tool_result) {
    const qw3::RetrievalQueryMessageRange range =
        qw3::latest_retrieval_query_message_range(messages);
    if (!range.valid() || range.begin != begin || range.end != end ||
        range.tool_result != tool_result) {
        fail("latest retrieval query message range was incorrect");
    }
}

void test_latest_external_input_query() {
    expect_query_range(
        json::array({
            json{{"role", "user"}, {"content", "initial request"}},
            json{{"role", "assistant"}, {"content", "answer"}},
            json{{"role", "user"}, {"content", "follow-up"}},
        }),
        2, 3, false);

    const std::string large_result(1024 * 1024, 'x');
    const json tool_messages = json::array({
        json{{"role", "user"}, {"content", "inspect the project"}},
        json{{"role", "assistant"},
             {"tool_calls",
              json::array({
                  json{{"function",
                        json{{"name", "read"},
                             {"arguments", R"({"path":"/tmp/a"})"}}}},
                  json{{"function",
                        json{{"name", "read"},
                             {"arguments", R"({"path":"/tmp/b"})"}}}},
              })}},
        json{{"role", "tool"}, {"content", large_result}},
        json{{"role", "tool"}, {"content", "second result"}},
    });
    expect_query_range(tool_messages, 2, 4, true);
    if (tool_messages[2]["content"].get<std::string>().size() !=
        large_result.size()) {
        fail("large tool result was truncated");
    }

    expect_query_range(
        json::array({
            json{{"role", "user"}, {"content", "initial request"}},
            json{{"role", "assistant"}, {"content", "tool call"}},
            json{{"role", "user"},
                 {"content",
                  "<tool_response>\ncomplete output\n</tool_response>"}},
        }),
        2, 3, true);
}

} // namespace

int main() {
    test_valid_and_no_intent();
    test_missing_function_inference_and_retry_case();
    test_schema_validation();
    test_incomplete_block_and_retry_prompt();
    test_incremental_commit_gate();
    test_latest_external_input_query();
    std::cout << "server_tool_parser_test: ok\n";
    return 0;
}
