// Reuse the frozen native request/template/tool implementations verbatim.
// This CPU-only JSON-lines process does not initialize a model or CUDA context.
// Including the implementation keeps the private rendering functions identical
// to the control server without duplicating a hand-written Python template.
#include "qw3_server.cpp"

namespace {
using json = nlohmann::json;

struct StreamState {
    json tools;
    qw3::ReasoningStreamSplitter reasoning;
    qw3::IncrementalToolCallStream tool;
    qw3::detail::AnthropicSseAdapter anthropic;
    std::string utf8_pending, content_pending, raw;
    std::string message_id;
    bool in_tool = false;
    bool enabled_thinking;
    size_t output_tokens = 0;

    explicit StreamState(const json &input)
        : tools(input.value("tools", json::array())),
          reasoning(input.value("enable_thinking", true)), tool(&tools),
          anthropic(input.value("model", "baseline"), input.at("input_tokens").get<size_t>()),
          enabled_thinking(input.value("enable_thinking", true)) {
        message_id = input.value("message_id", "chatcmpl-baseline");
        if (enabled_thinking) raw = "<think>\n";
    }

    void delta(const json &value, std::string &wire, const std::string &finish = {}) {
        json choice = {{"index", 0}, {"delta", value}};
        choice["finish_reason"] = finish.empty() ? json(nullptr) : json(finish);
        json chunk = {{"id", message_id}, {"object", "chat.completion.chunk"},
                      {"choices", json::array({choice})}};
        if (!finish.empty()) chunk["usage"] = {{"completion_tokens", output_tokens}};
        const std::string data = "data: " + chunk.dump() + "\n\n";
        std::string error;
        if (!anthropic.feed(data.data(), data.size(), [&](const std::string &s) {
            wire += s; return true;
        }, error)) throw std::runtime_error(error);
    }

    void flush_tool(std::string &wire, bool validated = false) {
        if (tool.fatal()) throw std::runtime_error(tool.error());
        if (tool.complete() && !validated) return;
        if (tool.start_pending()) delta(tool.take_start_delta(), wire);
        if (tool.pending_size()) delta(tool.take_arguments_delta(), wire);
    }

    void content(const std::string &text, std::string &wire, bool finish = false) {
        if (in_tool) {
            tool.feed(text);
            flush_tool(wire);
            return;
        }
        content_pending += text;
        const size_t pos = content_pending.find("<tool_call>");
        if (pos != std::string::npos) {
            if (pos) delta({{"content", content_pending.substr(0, pos)}}, wire);
            in_tool = true;
            tool.feed(content_pending.substr(pos));
            content_pending.clear();
            flush_tool(wire);
        } else {
            const std::string tag = "<tool_call>";
            size_t keep = finish ? 0 : std::min(content_pending.size(), tag.size() - 1);
            while (keep && content_pending.compare(content_pending.size() - keep, keep, tag, 0, keep) != 0)
                --keep;
            const size_t emit = qw3::utf8_safe_prefix_len(content_pending, content_pending.size() - keep);
            if (emit) {
                delta({{"content", content_pending.substr(0, emit)}}, wire);
                content_pending.erase(0, emit);
            }
        }
    }

    std::string consume(const std::string &bytes) {
        ++output_tokens;
        const std::string text = qw3::take_complete_utf8(utf8_pending, bytes);
        raw += text;
        std::string wire;
        for (const auto &part : reasoning.push(text)) {
            if (part.first == qw3::StreamPart::Reasoning) delta({{"reasoning_content", part.second}}, wire);
            else content(part.second, wire);
        }
        return wire;
    }

    std::string finish(const std::string &reason) {
        std::string wire;
        const std::string tail = qw3::flush_utf8_pending(utf8_pending, false);
        raw += tail;
        for (const auto &part : reasoning.push(tail)) {
            if (part.first == qw3::StreamPart::Reasoning) delta({{"reasoning_content", part.second}}, wire);
            else content(part.second, wire);
        }
        for (const auto &part : reasoning.finish()) {
            if (part.first == qw3::StreamPart::Reasoning) delta({{"reasoning_content", part.second}}, wire);
            else content(part.second, wire);
        }
        content({}, wire, true);
        auto calls = qw3::parse_tool_calls_xml(raw, &tools);
        if (in_tool) {
            tool.finish();
            flush_tool(wire);
            if (tool.streaming()) {
                std::string error;
                if (!tool.validate(calls, error)) throw std::runtime_error(error);
                flush_tool(wire, true);
                if (calls.size() > 1) {
                    json indexed = json::array();
                    for (size_t i = 1; i < calls.size(); ++i) {
                        auto call = calls[i]; call["index"] = i; indexed.push_back(call);
                    }
                    delta({{"tool_calls", indexed}}, wire);
                }
            } else if (!calls.empty()) {
                json indexed = json::array();
                for (size_t i = 0; i < calls.size(); ++i) {
                    auto call = calls[i]; call["index"] = i; indexed.push_back(call);
                }
                delta({{"tool_calls", indexed}}, wire);
            } else {
                const auto split = qw3::split_reasoning(raw);
                const size_t start = split.content.find("<tool_call>");
                if (start != std::string::npos) delta({{"content", split.content.substr(start)}}, wire);
            }
        }
        delta(json::object(), wire, calls.empty() ? reason : "tool_calls");
        std::string error;
        if (!anthropic.finish([&](const std::string &s) { wire += s; return true; }, error))
            throw std::runtime_error(error);
        return wire;
    }
};
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    using json = nlohmann::json;
    qw3::QwenTokenizer tokenizer(argv[1]);
    std::unique_ptr<StreamState> stream;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto input = json::parse(line);
            const std::string op = input.at("op");
            json output;
            if (op == "render") {
                json request;
                std::string error;
                if (!qw3::detail::anthropic_request_to_openai(input.at("request"), request, error,
                        input.value("require_max_tokens", true)))
                    throw std::runtime_error(error);
                const json *tools = request.contains("tools") ? &request["tools"] : nullptr;
                std::string forced;
                bool required = false;
                if (request.contains("tool_choice")) {
                    const auto &choice = request["tool_choice"];
                    if (choice.is_string()) required = choice == "required";
                    else if (choice.is_object() && choice.contains("function"))
                        forced = choice["function"].value("name", std::string());
                }
                const bool thinking = input.value("enable_thinking", request.value("enable_thinking", true));
                const std::string prompt = qw3::render_messages(request.at("messages"), tools,
                    thinking, forced, nullptr, true, required,
                    qw3::detail::HarnessKind::ClaudeCode,
                    input.value("stable_reasoning", true));
                output = {{"prompt", prompt}, {"input_ids", tokenizer.encode(prompt)},
                          {"openai_request", request}};
            } else if (op == "tokenize") {
                output = {{"input_ids", tokenizer.encode(input.at("text").get<std::string>())}};
            } else if (op == "parse") {
                const std::string text = input.at("text");
                const json *tools = input.contains("tools") ? &input["tools"] : nullptr;
                const auto calls = qw3::parse_tool_calls_xml(text, tools);
                const auto split = qw3::split_reasoning(text);
                std::string content = split.content;
                if (!calls.empty()) {
                    const size_t begin = content.find("<tool_call>");
                    if (begin != std::string::npos) content = qw3::trim_ascii_ws(content.substr(0, begin));
                }
                output = {{"reasoning_content", split.reasoning}, {"content", content}, {"tool_calls", calls}};
            } else if (op == "anthropic_response") {
                output = qw3::detail::anthropic_response_from_openai(input.at("response"), input.value("model", "baseline"));
            } else if (op == "stream_start") {
                stream = std::make_unique<StreamState>(input);
                output = {{"wire", ""}};
            } else if (op == "stream_token") {
                if (!stream) throw std::runtime_error("no active stream");
                output = {{"wire", stream->consume(tokenizer.decode_one(input.at("token_id").get<int32_t>()))},
                          {"tool_complete", stream->tool.complete()}};
            } else if (op == "stream_text") {
                if (!stream) throw std::runtime_error("no active stream");
                output = {{"wire", stream->consume(input.at("text").get<std::string>())},
                          {"tool_complete", stream->tool.complete()}};
            } else if (op == "stream_finish") {
                if (!stream) throw std::runtime_error("no active stream");
                const auto reason = input.value("reason", "stop");
                output = {{"wire", stream->finish(reason)}, {"raw_completion", stream->raw},
                    {"finish_reason", qw3::parse_tool_calls_xml(stream->raw, &stream->tools).empty() ? reason : "tool_calls"}};
                stream.reset();
            } else {
                throw std::runtime_error("unknown bridge operation");
            }
            std::cout << json{{"ok", true}, {"result", output}}.dump() << std::endl;
        } catch (const std::exception &error) {
            std::cout << json{{"ok", false}, {"error", error.what()}}.dump() << std::endl;
        }
    }
}
