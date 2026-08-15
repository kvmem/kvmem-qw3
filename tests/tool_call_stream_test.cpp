#include "tool_call_stream.hpp"

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using qw3::detail::CanonicalToolCallStreamParser;
using qw3::detail::ToolCallStreamEvent;
using qw3::detail::ToolCallStreamEventKind;
using qw3::detail::json_string_fragment;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "tool_call_stream_test: " << message << "\n";
    std::exit(1);
}

struct ParsedCall {
    std::string name;
    std::map<std::string, std::string> args;
};

ParsedCall collect(const std::vector<ToolCallStreamEvent> &events) {
    ParsedCall out;
    std::string parameter;
    bool complete = false;
    for (const ToolCallStreamEvent &event : events) {
        switch (event.kind) {
            case ToolCallStreamEventKind::ToolStart:
                out.name = event.value;
                break;
            case ToolCallStreamEventKind::ParameterStart:
                parameter = event.value;
                out.args[parameter] = {};
                break;
            case ToolCallStreamEventKind::ParameterData:
                out.args[parameter] += event.value;
                break;
            case ToolCallStreamEventKind::ParameterEnd:
                parameter.clear();
                break;
            case ToolCallStreamEventKind::ToolEnd:
                complete = true;
                break;
        }
    }
    if (!complete) fail("missing ToolEnd event");
    return out;
}

void test_one_byte_chunks() {
    const std::string content =
        "line 1\nquote=\"x\" slash=\\\\\nUTF-8: \xE4\xB8\xAD\xE6\x96\x87\n";
    const std::string text =
        "<tool_call>\n<function=Write>\n"
        "<parameter=file_path>\n/tmp/a.txt\n</parameter>\n"
        "<parameter=content>\n" + content +
        "\n</parameter>\n</function>\n</tool_call>";

    CanonicalToolCallStreamParser parser;
    std::vector<ToolCallStreamEvent> events;
    for (char c : text) {
        if (!parser.feed(std::string(1, c), events)) {
            fail("one-byte parse failed: " + parser.error());
        }
    }
    if (!parser.finish(events)) fail("one-byte finish failed: " + parser.error());
    const ParsedCall call = collect(events);
    if (call.name != "Write") fail("wrong function name");
    if (call.args.at("file_path") != "/tmp/a.txt") {
        fail("file_path newline trimming mismatch");
    }
    if (call.args.at("content") != content) {
        fail("streamed content mismatch");
    }
}

void test_near_close_marker_and_crlf_trim() {
    const std::string value = "before </parameteX> after";
    const std::string text =
        "<tool_call><function=Write>"
        "<parameter=content>\n\r" + value +
        "\r\n</parameter></function></tool_call>";
    CanonicalToolCallStreamParser parser;
    std::vector<ToolCallStreamEvent> events;
    for (size_t i = 0; i < text.size(); i += 3) {
        if (!parser.feed(text.substr(i, 3), events)) {
            fail("three-byte parse failed: " + parser.error());
        }
    }
    if (!parser.finish(events)) fail("three-byte finish failed: " + parser.error());
    const ParsedCall call = collect(events);
    if (call.args.at("content") != value) {
        fail("partial close marker or CRLF trimming mismatch");
    }
}

void test_malformed_fails_closed() {
    CanonicalToolCallStreamParser parser;
    std::vector<ToolCallStreamEvent> events;
    const std::string malformed =
        "<tool_call><function=Write><arg_key>content</arg_key></tool_call>";
    if (parser.feed(malformed, events)) {
        fail("malformed canonical call was accepted");
    }
    if (!parser.failed()) fail("malformed parser did not enter failed state");
}

void test_closure_repair() {
    for (const std::string &text : {
             "<tool_call><function=Write><parameter=content>hello",
             "<tool_call><function=Write><parameter=content>hello</parameter>",
             "<tool_call><function=Write><parameter=content>hello</parameter>"
             "</function>",
             "<tool_call><function=Write><parameter=content>hello</parameter>"
             "</function></tool_"}) {
        CanonicalToolCallStreamParser parser;
        std::vector<ToolCallStreamEvent> events;
        if (!parser.feed(text, events)) {
            fail("repairable prefix failed before EOF: " + parser.error());
        }
        std::string suffix;
        if (!parser.finish_with_closure_repair(events, suffix)) {
            fail("closure repair failed: " + parser.error());
        }
        if (suffix.empty()) fail("closure repair did not report a suffix");
        const ParsedCall call = collect(events);
        if (call.name != "Write" || call.args.at("content") != "hello") {
            fail("closure repair changed the parsed call");
        }
    }
}

void test_closure_repair_rejects_missing_structure() {
    CanonicalToolCallStreamParser parser;
    std::vector<ToolCallStreamEvent> events;
    if (!parser.feed("<tool_call><fun", events)) {
        fail("partial function prefix failed before EOF");
    }
    std::string suffix;
    if (parser.finish_with_closure_repair(events, suffix)) {
        fail("missing function structure was repaired");
    }
}

void test_json_fragment_escaping() {
    const std::string first = "quote=\" slash=\\ newline=\n tab=\t ";
    const std::string second = "control=\x01 UTF-8=\xE4\xB8\xAD\xE6\x96\x87";
    const std::string escaped =
        json_string_fragment(first) + json_string_fragment(second);
    const std::string expected =
        "quote=\\\" slash=\\\\ newline=\\n tab=\\t "
        "control=\\u0001 UTF-8=\xE4\xB8\xAD\xE6\x96\x87";
    if (escaped != expected) fail("JSON fragment escaping mismatch");
}

} // namespace

int main() {
    test_one_byte_chunks();
    test_near_close_marker_and_crlf_trim();
    test_malformed_fails_closed();
    test_closure_repair();
    test_closure_repair_rejects_missing_structure();
    test_json_fragment_escaping();
    std::cout << "tool_call_stream_test: ok\n";
    return 0;
}
