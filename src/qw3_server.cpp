#include "server.hpp"

#include "env_flags.hpp"
#include "qw3/qw3.hpp"
#include "qw3/gguf.hpp"
#include "qw3/tokenizer.hpp"

// Vendored single-header deps (included as SYSTEM headers via CMake so their
// warnings don't trip -Wall -Wextra -Wpedantic).
#include "httplib.h"
#include "json.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qw3 {

namespace {

using json = nlohmann::json;

bool serve_continuous_batching_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING");
}

bool serve_continuous_batch_request_supported(const GenerationOptions &g) {
    return g.max_tokens >= 0;
}

json usage_json(size_t prompt_tokens, size_t completion_tokens) {
    return json{{"prompt_tokens", prompt_tokens},
                {"completion_tokens", completion_tokens},
                {"total_tokens", prompt_tokens + completion_tokens}};
}

std::string basename_of(const std::string &path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Random-ish id for the OpenAI `id` field. Not security-sensitive.
std::string gen_id(const char *prefix) {
    static std::mt19937_64 rng(std::random_device{}());
    static const char *hex = "0123456789abcdef";
    std::string s = prefix;
    for (int i = 0; i < 24; ++i) s += hex[rng() & 0xF];
    return s;
}

std::string dump_json(const json &value) {
    return value.dump(-1, ' ', false, json::error_handler_t::replace);
}

void set_error_response(httplib::Response &res,
                        int status,
                        const std::string &message) {
    res.status = status;
    res.set_content(dump_json(json{{"error", message}}), "application/json");
}

int status_for_exception(const std::exception &e) {
    const std::string msg = e.what();
    if (msg.find("admission rejected") != std::string::npos) return 429;
    if (msg.find("global KV page pool exhausted") != std::string::npos) return 429;
    if (msg.find("prompt exceeds KV context") != std::string::npos) return 413;
    return 500;
}

std::string replacement_char() {
    return "\xEF\xBF\xBD";
}

size_t utf8_expected_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 0;
}

bool utf8_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

std::string take_complete_utf8(std::string &pending,
                               const std::string &piece,
                               size_t holdback = 0) {
    pending += piece;
    std::string out;
    size_t i = 0;
    const size_t limit = pending.size() > holdback ? pending.size() - holdback : 0;
    while (i < pending.size()) {
        if (i >= limit) break;
        const unsigned char c0 = static_cast<unsigned char>(pending[i]);
        const size_t len = utf8_expected_len(c0);
        if (len == 0) {
            out += replacement_char();
            ++i;
            continue;
        }
        if (i + len > pending.size() || i + len > limit) break;
        bool ok = true;
        for (size_t j = 1; j < len; ++j) {
            if (!utf8_cont(static_cast<unsigned char>(pending[i + j]))) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            out += replacement_char();
            ++i;
            continue;
        }
        out.append(pending, i, len);
        i += len;
    }
    pending.erase(0, i);
    return out;
}

std::string flush_utf8_pending(std::string &pending, bool replace_incomplete = true) {
    if (pending.empty()) return {};
    std::string out = take_complete_utf8(pending, {}, 0);
    if (!pending.empty()) {
        pending.clear();
        if (replace_incomplete) out += replacement_char();
    }
    return out;
}

size_t utf8_safe_prefix_len(const std::string &text, size_t desired) {
    size_t cut = std::min(desired, text.size());
    while (cut > 0 && cut < text.size() &&
           utf8_cont(static_cast<unsigned char>(text[cut]))) {
        --cut;
    }
    return cut;
}

struct ReasoningSplit {
    std::string reasoning;
    std::string content;
};

std::string render_content(const json &content);
std::string trim_ascii_ws(std::string s);

ReasoningSplit split_reasoning(const std::string &text) {
    const std::string open = "<think>";
    const std::string close = "</think>";
    const size_t start = text.find(open);
    if (start == std::string::npos) return ReasoningSplit{{}, text};
    const size_t reasoning_start = start + open.size();
    const size_t end = text.find(close, reasoning_start);
    if (end == std::string::npos) {
        std::string reasoning = text.substr(reasoning_start);
        if (!reasoning.empty() && reasoning.front() == '\n') reasoning.erase(reasoning.begin());
        return ReasoningSplit{reasoning, {}};
    }
    std::string reasoning = text.substr(reasoning_start, end - reasoning_start);
    std::string content = text.substr(end + close.size());
    if (!reasoning.empty() && reasoning.front() == '\n') reasoning.erase(reasoning.begin());
    if (!reasoning.empty() && reasoning.back() == '\n') reasoning.pop_back();
    while (!content.empty() && (content.front() == '\n' || content.front() == '\r')) {
        content.erase(content.begin());
    }
    return ReasoningSplit{reasoning, content};
}

bool is_tool_response_content(const std::string &content) {
    const std::string trimmed = trim_ascii_ws(content);
    const std::string open = "<tool_response>";
    const std::string close = "</tool_response>";
    return trimmed.rfind(open, 0) == 0 &&
           trimmed.size() >= close.size() &&
           trimmed.compare(trimmed.size() - close.size(), close.size(), close) == 0;
}

size_t last_query_index_for_template(const json &messages) {
    if (!messages.is_array() || messages.empty()) return 0;
    bool multi_step_tool = true;
    size_t last_query = messages.size() - 1;
    for (size_t rev = 0; rev < messages.size(); ++rev) {
        const size_t i = messages.size() - 1 - rev;
        const auto &m = messages[i];
        if (!m.is_object()) continue;
        if (multi_step_tool && m.value("role", "") == "user") {
            const std::string content =
                trim_ascii_ws(m.contains("content") ? render_content(m["content"]) : "");
            if (!is_tool_response_content(content)) {
                multi_step_tool = false;
                last_query = i;
            }
        }
    }
    return last_query;
}

enum class StreamPart {
    Reasoning,
    Content,
};

class ReasoningStreamSplitter {
public:
    explicit ReasoningStreamSplitter(bool enabled)
        : enabled_(enabled), part_(enabled ? StreamPart::Reasoning : StreamPart::Content) {}

    std::vector<std::pair<StreamPart, std::string>> push(const std::string &text) {
        if (!enabled_) return {{StreamPart::Content, text}};
        pending_ += text;
        std::vector<std::pair<StreamPart, std::string>> out;
        while (!pending_.empty()) {
            if (part_ == StreamPart::Reasoning) {
                const size_t close = pending_.find("</think>");
                if (close == std::string::npos) {
                    const size_t keep = std::min<size_t>(pending_.size(), 7);
                    const size_t emit_len =
                        utf8_safe_prefix_len(pending_, pending_.size() - keep);
                    if (emit_len > 0) {
                        out.push_back({StreamPart::Reasoning, pending_.substr(0, emit_len)});
                        pending_.erase(0, emit_len);
                    }
                    break;
                }
                if (close > 0) out.push_back({StreamPart::Reasoning, pending_.substr(0, close)});
                pending_.erase(0, close + std::string("</think>").size());
                while (!pending_.empty() && (pending_.front() == '\n' || pending_.front() == '\r')) {
                    pending_.erase(pending_.begin());
                }
                part_ = StreamPart::Content;
            } else {
                out.push_back({StreamPart::Content, pending_});
                pending_.clear();
            }
        }
        return out;
    }

    std::vector<std::pair<StreamPart, std::string>> finish() {
        if (pending_.empty()) return {};
        std::vector<std::pair<StreamPart, std::string>> out{{part_, pending_}};
        pending_.clear();
        return out;
    }

private:
    bool enabled_ = false;
    StreamPart part_ = StreamPart::Content;
    std::string pending_;
};

std::string render_content(const json &content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_null()) return {};
    if (content.is_array()) {
        std::string out;
        for (const auto &item : content) {
            if (item.is_string()) {
                out += item.get<std::string>();
            } else if (item.is_object() && item.contains("text") && item["text"].is_string()) {
                out += item["text"].get<std::string>();
            }
        }
        return out;
    }
    return content.dump();
}

std::string trim_ascii_ws(std::string s) {
    size_t b = 0;
    while (b < s.size() &&
           (s[b] == ' ' || s[b] == '\n' || s[b] == '\r' || s[b] == '\t')) {
        ++b;
    }
    size_t e = s.size();
    while (e > b &&
           (s[e - 1] == ' ' || s[e - 1] == '\n' ||
            s[e - 1] == '\r' || s[e - 1] == '\t')) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string render_tool_call(const json &call) {
    const json *fn = &call;
    if (call.is_object() && call.contains("function") && call["function"].is_object()) {
        fn = &call["function"];
    }
    if (!fn->is_object()) return {};
    const std::string name = fn->value("name", "");
    if (name.empty()) return {};

    json args = json::object();
    if (fn->contains("arguments")) {
        const json &raw = (*fn)["arguments"];
        if (raw.is_object()) {
            args = raw;
        } else if (raw.is_string()) {
            try {
                args = json::parse(raw.get<std::string>());
            } catch (...) {
                args = json{{"arguments", raw.get<std::string>()}};
            }
        }
    }

    std::string out = "<tool_call>\n<function=" + name + ">\n";
    if (args.is_object()) {
        for (auto it = args.begin(); it != args.end(); ++it) {
            out += "<parameter=" + it.key() + ">\n";
            if (it.value().is_string()) {
                out += it.value().get<std::string>();
            } else {
                out += dump_json(it.value());
            }
            out += "\n</parameter>\n";
        }
    }
    out += "</function>\n</tool_call>";
    return out;
}

std::vector<json> parse_tool_calls_xml(const std::string &text) {
    std::vector<json> calls;
    size_t pos = 0;
    while (true) {
        const size_t tc0 = text.find("<tool_call>", pos);
        if (tc0 == std::string::npos) break;
        const size_t tc1 = text.find("</tool_call>", tc0);
        if (tc1 == std::string::npos) break;
        const std::string block = text.substr(tc0, tc1 - tc0);
        const size_t fn0 = block.find("<function=");
        if (fn0 == std::string::npos) {
            pos = tc1 + 12;
            continue;
        }
        const size_t name0 = fn0 + std::string("<function=").size();
        const size_t name1 = block.find(">", name0);
        if (name1 == std::string::npos) {
            pos = tc1 + 12;
            continue;
        }
        const std::string name = block.substr(name0, name1 - name0);
        json args = json::object();
        size_t pp = name1 + 1;
        while (true) {
            const size_t p0 = block.find("<parameter=", pp);
            if (p0 == std::string::npos) break;
            const size_t key0 = p0 + std::string("<parameter=").size();
            const size_t key1 = block.find(">", key0);
            if (key1 == std::string::npos) break;
            const size_t v0 = key1 + 1;
            const size_t v1 = block.find("</parameter>", v0);
            if (v1 == std::string::npos) break;
            std::string value = block.substr(v0, v1 - v0);
            if (!value.empty() && value.front() == '\n') value.erase(value.begin());
            if (!value.empty() && value.back() == '\n') value.pop_back();
            args[block.substr(key0, key1 - key0)] = value;
            pp = v1 + std::string("</parameter>").size();
        }
        calls.push_back(json{
            {"id", gen_id("call_")},
            {"type", "function"},
            {"function", json{{"name", name}, {"arguments", dump_json(args)}}}
        });
        pos = tc1 + std::string("</tool_call>").size();
    }
    return calls;
}

json tool_call_delta(const json &calls) {
    json deltas = json::array();
    for (size_t i = 0; i < calls.size(); ++i) {
        const json &call = calls[i];
        json d = {
            {"index", static_cast<int>(i)},
            {"id", call.value("id", "")},
            {"type", call.value("type", "function")},
            {"function", json::object()}
        };
        if (call.contains("function") && call["function"].is_object()) {
            d["function"]["name"] = call["function"].value("name", "");
            d["function"]["arguments"] = call["function"].value("arguments", "{}");
        }
        deltas.push_back(d);
    }
    return json{{"tool_calls", deltas}};
}

bool could_be_tool_call_prefix(const std::string &text) {
    static const std::string target = "<tool_call>";
    size_t i = 0;
    while (i < text.size() &&
           (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) {
        ++i;
    }
    const std::string s = text.substr(i);
    if (s.empty()) return true;
    if (s.size() <= target.size()) return target.compare(0, s.size(), s) == 0;
    return s.compare(0, target.size(), target) == 0;
}

bool starts_with_tool_call(const std::string &text) {
    size_t i = 0;
    while (i < text.size() &&
           (text[i] == ' ' || text[i] == '\n' || text[i] == '\r' || text[i] == '\t')) {
        ++i;
    }
    return text.compare(i, std::string("<tool_call>").size(), "<tool_call>") == 0;
}

// Render an OpenAI messages[] array into a Qwen3.6 chat transcript. This mirrors
// the GGUF chat_template's text/tool subset closely enough for tool calling:
// tools are emitted in a system block, assistant tool_calls are serialized as
// Qwen XML tool calls, and tool results become user-side <tool_response> blocks.
// The final assistant header (+ thinking prefill or empty-think block) is
// appended for generation.
std::string render_messages(const json &messages, const json *tools,
                            bool enable_thinking,
                            const std::string &forced_tool_name = {}) {
    size_t num_sys = 0;
    std::string merged_system;
    if (messages.is_array() && !messages.empty() && messages[0].is_object()) {
        const std::string first_role = messages[0].value("role", "");
        if (first_role == "system" || first_role == "developer") {
            merged_system = trim_ascii_ws(
                messages[0].contains("content") ? render_content(messages[0]["content"]) : "");
            num_sys = 1;
            if (messages.size() > 1 && messages[1].is_object()) {
                const std::string second_role = messages[1].value("role", "");
                if (second_role == "system" || second_role == "developer") {
                    const std::string second = trim_ascii_ws(
                        messages[1].contains("content") ? render_content(messages[1]["content"]) : "");
                    merged_system += "\n" + second;
                    num_sys = 2;
                }
            }
        }
    }

    std::string prompt;
    if (tools && tools->is_array() && !tools->empty()) {
        prompt += "<|im_start|>system\n";
        prompt += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto &tool : *tools) {
            prompt += "\n" + dump_json(tool);
        }
        prompt += "\n</tools>";
        prompt += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n";
        prompt += "<tool_call>\n<function=example_function_name>\n";
        prompt += "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n";
        prompt += "<parameter=example_parameter_2>\n";
        prompt += "This is the value for the second parameter\nthat can span\nmultiple lines\n";
        prompt += "</parameter>\n</function>\n</tool_call>\n\n";
        prompt += "<IMPORTANT>\n";
        prompt += "Reminder:\n";
        prompt += "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n";
        prompt += "- Required parameters MUST be specified\n";
        prompt += "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n";
        if (!forced_tool_name.empty()) {
            prompt += "- You MUST call the function named `" + forced_tool_name + "`\n";
        }
        prompt += "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n";
        prompt += "</IMPORTANT>";
        if (!merged_system.empty()) prompt += "\n\n" + merged_system;
        prompt += "<|im_end|>\n";
    } else if (!merged_system.empty()) {
        prompt += "<|im_start|>system\n" + merged_system + "<|im_end|>\n";
    }

    const size_t last_query_index = last_query_index_for_template(messages);
    for (size_t i = 0; messages.is_array() && i < messages.size(); ++i) {
        const auto &m = messages[i];
        if (!m.is_object() || i < num_sys) continue;
        const std::string role = m.value("role", "");
        if (role == "system" || role == "developer") continue;
        const std::string content =
            trim_ascii_ws(m.contains("content") ? render_content(m["content"]) : "");
        if (role == "user") {
            prompt += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (role == "assistant") {
            std::string assistant_content = content;
            std::string reasoning_content;
            if (m.contains("reasoning_content") && m["reasoning_content"].is_string()) {
                reasoning_content =
                    trim_ascii_ws(m["reasoning_content"].get<std::string>());
            } else {
                const ReasoningSplit split = split_reasoning(assistant_content);
                if (!split.reasoning.empty() || split.content != assistant_content) {
                    reasoning_content = trim_ascii_ws(split.reasoning);
                    assistant_content = split.content;
                }
            }
            prompt += "<|im_start|>assistant\n";
            if (i > last_query_index) {
                prompt += "<think>\n" + reasoning_content + "\n</think>\n\n";
            }
            prompt += assistant_content;
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                bool first_tool_call = true;
                for (const auto &call : m["tool_calls"]) {
                    const std::string rendered = render_tool_call(call);
                    if (rendered.empty()) continue;
                    if (first_tool_call) {
                        if (!assistant_content.empty()) prompt += "\n\n";
                    } else {
                        prompt += "\n";
                    }
                    prompt += rendered;
                    first_tool_call = false;
                }
            }
            prompt += "<|im_end|>\n";
        } else if (role == "tool") {
            const bool prev_tool =
                i > 0 && messages[i - 1].is_object() &&
                messages[i - 1].value("role", "") == "tool";
            const bool next_tool =
                i + 1 < messages.size() && messages[i + 1].is_object() &&
                messages[i + 1].value("role", "") == "tool";
            if (!prev_tool) prompt += "<|im_start|>user";
            prompt += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (!next_tool) prompt += "<|im_end|>\n";
        }
    }

    prompt += "<|im_start|>assistant\n";
    if (enable_thinking) {
        prompt += "<think>\n";
    } else {
        prompt += "<think>\n\n</think>\n\n";
    }
    return prompt;
}

// Apply stop sequences: truncate `text` at the earliest occurrence of any stop
// string. Returns true if a stop was hit.
bool apply_stops(std::string &text, const std::vector<std::string> &stops) {
    size_t cut = std::string::npos;
    for (const std::string &s : stops) {
        if (s.empty()) continue;
        const size_t pos = text.find(s);
        if (pos != std::string::npos && pos < cut) cut = pos;
    }
    if (cut != std::string::npos) {
        text.erase(cut);
        return true;
    }
    return false;
}

std::vector<std::string> parse_stops(const json &req) {
    std::vector<std::string> stops;
    if (!req.contains("stop") || req["stop"].is_null()) return stops;
    const json &s = req["stop"];
    if (s.is_string()) {
        stops.push_back(s.get<std::string>());
    } else if (s.is_array()) {
        for (const auto &e : s) if (e.is_string()) stops.push_back(e.get<std::string>());
    }
    return stops;
}

} // namespace

int run_server(EngineOptions engine, ServerConfig cfg) {
    // Force the working native path for serving regardless of caller defaults.
    engine.backend = BackendKind::QwenNative;
    engine.native_heavy = true;
    if (engine.native_kernels.empty()) engine.native_kernels = "cuda";
    if (serve_continuous_batching_enabled()) {
        if (std::getenv("QW3_MATMUL") == nullptr) {
            setenv("QW3_MATMUL", "mmq", 1);
        }
        setenv("QW3_DISABLE_HGEMM", "1", 1);
        std::cerr << "[qw3-serve] continuous batching matmul guard: "
                  << "QW3_MATMUL=" << std::getenv("QW3_MATMUL")
                  << " QW3_DISABLE_HGEMM=" << std::getenv("QW3_DISABLE_HGEMM")
                  << "\n";
    }

    std::cerr << "[qw3-serve] loading model: " << engine.model_path << "\n";
    Engine eng(engine);
    GgufFile usage_gguf(engine.model_path);
    QwenTokenizer usage_tokenizer(usage_gguf);
    const std::string model_id = basename_of(engine.model_path);
    std::cerr << "[qw3-serve] model loaded; id=" << model_id << "\n";

    // Single shared KV cache + scratch in the executor => serialize generation.
    std::mutex gen_mu;
    std::atomic<uint64_t> req_counter{0};

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response &res) {
        json out = {
            {"object", "list"},
            {"data", json::array({json{{"id", model_id},
                                       {"object", "model"},
                                       {"created", unix_now()},
                                       {"owned_by", "qw3"}}})}};
        res.set_content(dump_json(out), "application/json");
    });

    // Build GenerationOptions from common OpenAI fields. Default temperature=0
    // (greedy) for reproducible evals when the client omits it; default top_p=1.
    auto make_gen = [&](const json &req) -> GenerationOptions {
        GenerationOptions g = cfg.default_generation;
        g.max_tokens = req.value("max_tokens",
                                 req.value("max_completion_tokens", g.max_tokens));
        if (g.max_tokens <= 0) g.max_tokens = 256;
        g.temperature = req.value("temperature", g.temperature);
        g.top_p = req.value("top_p", g.top_p);
        g.top_k = req.value("top_k", g.top_k);
        g.min_p = req.value("min_p", g.min_p);
        g.presence_penalty = req.value("presence_penalty", g.presence_penalty);
        g.repetition_penalty = req.value("repetition_penalty", g.repetition_penalty);
        g.seed = req.value("seed", g.seed);
        return g;
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request &hreq,
                                         httplib::Response &res) {
        json req;
        try {
            req = json::parse(hreq.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", std::string("invalid JSON: ") + e.what()}}),
                            "application/json");
            return;
        }
        if (!req.contains("messages") || !req["messages"].is_array()) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", "missing messages[]"}}),
                            "application/json");
            return;
        }
        const bool enable_thinking =
            req.value("enable_thinking", cfg.enable_thinking_default);
        const json *raw_tools = req.contains("tools") ? &req["tools"] : nullptr;
        const bool tool_choice_none =
            req.contains("tool_choice") && req["tool_choice"].is_string() &&
            req["tool_choice"].get<std::string>() == "none";
        std::string forced_tool_name;
        if (req.contains("tool_choice") && req["tool_choice"].is_object()) {
            const json &tc = req["tool_choice"];
            if (tc.contains("function") && tc["function"].is_object()) {
                forced_tool_name = tc["function"].value("name", "");
            }
        }
        const json *tools = tool_choice_none ? nullptr : raw_tools;
        const bool tool_request = tools && tools->is_array() && !tools->empty();
        const std::string prompt =
            render_messages(req["messages"], tools, enable_thinking, forced_tool_name);
        const size_t prompt_token_count = usage_tokenizer.encode(prompt).size();
        GenerationOptions g = make_gen(req);
        g.raw_prompt = true; // prompt is already chat-framed
        g.continuous_batching =
            serve_continuous_batching_enabled() &&
            serve_continuous_batch_request_supported(g);
        const std::string route = g.continuous_batching ? "continuous" : "plain";
        const std::string fallback_reason =
            g.continuous_batching ? "" :
            (serve_continuous_batching_enabled() ? "request_unsupported" : "disabled");
        const std::vector<std::string> stops = parse_stops(req);
        const bool stream = req.value("stream", false);
        const bool stream_include_usage =
            req.contains("stream_options") && req["stream_options"].is_object() &&
            req["stream_options"].value("include_usage", false);
        const bool forced_tool_request = tool_request && !forced_tool_name.empty();
        const std::string id = gen_id("chatcmpl-");
        const int64_t created = unix_now();
        const uint64_t rid = ++req_counter;
        const auto t0 = std::chrono::steady_clock::now();

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, g, stops, id, created, rid, enable_thinking,
                 tool_request, forced_tool_request,
                 stream_include_usage, prompt_token_count, route,
                 fallback_reason](size_t, httplib::DataSink &sink) {
                    std::unique_lock<std::mutex> gen_lk(gen_mu, std::defer_lock);
                    if (!g.continuous_batching) gen_lk.lock();
                    std::string acc;
                    std::string utf8_pending;
                    ReasoningStreamSplitter reasoning_splitter(enable_thinking);
                    size_t completion_tokens = 0;
                    bool stopped = false;
                    bool client_closed = false;
                    auto send_raw = [&](const std::string &s) {
                        if (client_closed) return false;
                        if (!sink.write(s.data(), s.size())) {
                            client_closed = true;
                            stopped = true;
                            return false;
                        }
                        return true;
                    };
                    auto send_delta = [&](const json &delta) {
                        json chunk = {
                            {"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_id},
                            {"choices", json::array({json{
                                {"index", 0},
                                {"delta", delta},
                                {"finish_reason", nullptr}}})}};
                        const std::string s = "data: " + dump_json(chunk) + "\n\n";
                        send_raw(s);
                    };
                    auto send_role = [&]() {
                        json chunk = {
                            {"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_id},
                            {"choices", json::array({json{
                                {"index", 0},
                                {"delta", json{{"role", "assistant"}}},
                                {"finish_reason", nullptr}}})}};
                        const std::string s = "data: " + dump_json(chunk) + "\n\n";
                        send_raw(s);
                    };
                    auto send_done = [&](const std::string &finish_reason) {
                        if (client_closed) return;
                        json done = {
                            {"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_id},
                            {"choices", json::array({json{
                                {"index", 0}, {"delta", json::object()},
                                {"finish_reason", finish_reason}}})}};
                        const std::string ds = "data: " + dump_json(done) + "\n\n";
                        send_raw(ds);
                        if (stream_include_usage) {
                            json usage = {
                                {"id", id},
                                {"object", "chat.completion.chunk"},
                                {"created", created},
                                {"model", model_id},
                                {"choices", json::array()},
                                {"usage", usage_json(prompt_token_count,
                                                     completion_tokens)}};
                            const std::string us =
                                "data: " + dump_json(usage) + "\n\n";
                            send_raw(us);
                        }
                        const std::string fin = "data: [DONE]\n\n";
                        send_raw(fin);
                        sink.done();
                    };
                    try {
                        send_role();
                        if (enable_thinking) {
                            send_delta(json{{"reasoning_content", ""}});
                        }
                        auto emit_text = [&](const std::string &text) {
                            if (text.empty()) return;
                            for (const auto &part : reasoning_splitter.push(text)) {
                                if (part.second.empty()) continue;
                                if (part.first == StreamPart::Reasoning) {
                                    send_delta(json{{"reasoning_content", part.second}});
                                } else {
                                    send_delta(json{{"content", part.second}});
                                }
                            }
                        };
                        auto finish_text_stream = [&]() {
                            const std::string tail = flush_utf8_pending(utf8_pending, false);
                            if (!tail.empty()) emit_text(tail);
                            for (const auto &part : reasoning_splitter.finish()) {
                                if (part.second.empty()) continue;
                                if (part.first == StreamPart::Reasoning) {
                                    send_delta(json{{"reasoning_content", part.second}});
                                } else {
                                    send_delta(json{{"content", part.second}});
                                }
                            }
                        };
                        if (tool_request) {
                            bool classified = forced_tool_request;
                            bool buffering_tool = forced_tool_request;
                            std::string pending_prefix;
                            eng.generate_stream(prompt, g, [&](const std::string &piece) {
                                if (stopped) return;
                                ++completion_tokens;
                                acc += piece;
                                std::string emit = take_complete_utf8(utf8_pending, piece);
                                if (!stops.empty()) {
                                    std::string probe = acc;
                                    if (apply_stops(probe, stops)) {
                                        stopped = true;
                                        utf8_pending.clear();
                                        const size_t previous_size = acc.size() - piece.size();
                                        emit = probe.size() > previous_size
                                                   ? probe.substr(previous_size)
                                                   : "";
                                        acc = std::move(probe);
                                        emit = take_complete_utf8(utf8_pending, emit);
                                    }
                                }
                                if (emit.empty() || buffering_tool) return;
                                if (!classified) {
                                    pending_prefix += emit;
                                    if (starts_with_tool_call(pending_prefix)) {
                                        buffering_tool = true;
                                        classified = true;
                                        return;
                                    }
                                    if (could_be_tool_call_prefix(pending_prefix)) {
                                        return;
                                    }
                                    classified = true;
                                    emit = pending_prefix;
                                    pending_prefix.clear();
                                }
                                emit_text(emit);
                            });
                            if (buffering_tool) {
                                std::string text = take_complete_utf8(utf8_pending, acc);
                                text += flush_utf8_pending(utf8_pending, false);
                                const std::string framed =
                                    enable_thinking ? ("<think>\n" + text) : text;
                                const std::vector<json> tool_calls =
                                    parse_tool_calls_xml(framed);
                                const ReasoningSplit split = split_reasoning(framed);
                                if (!split.reasoning.empty()) {
                                    send_delta(json{{"reasoning_content", split.reasoning}});
                                }
                                if (!tool_calls.empty()) {
                                    send_delta(tool_call_delta(tool_calls));
                                    send_done("tool_calls");
                                } else {
                                    if (!split.content.empty()) {
                                        send_delta(json{{"content", split.content}});
                                    }
                                    send_done(stopped ? "stop" : "stop");
                                }
                            } else {
                                if (!pending_prefix.empty()) emit_text(pending_prefix);
                                finish_text_stream();
                                send_done(stopped ? "stop" : "stop");
                            }
                            std::cerr << "[qw3-serve] #" << rid
                                      << " chat(stream tools) chars=" << acc.size()
                                      << " completion_tokens=" << completion_tokens
                                      << " prompt_tokens=" << prompt_token_count
                                      << " route=" << route
                                      << " streamed=" << (buffering_tool ? "false" : "true")
                                      << (client_closed ? " client_closed=true" : "")
                                      << "\n";
                            return true;
                        }
                        eng.generate_stream(prompt, g, [&](const std::string &piece) {
                            if (stopped) return;
                            ++completion_tokens;
                            acc += piece;
                            std::string emit = take_complete_utf8(utf8_pending, piece);
                            if (!stops.empty()) {
                                std::string probe = acc;
                                if (apply_stops(probe, stops)) {
                                    stopped = true;
                                    utf8_pending.clear();
                                    const size_t previous_size = acc.size() - piece.size();
                                    emit = probe.size() > previous_size
                                               ? probe.substr(previous_size)
                                               : "";
                                    emit = take_complete_utf8(utf8_pending, emit);
                                }
                            }
                            if (!emit.empty()) {
                                emit_text(emit);
                            }
                        });
                        finish_text_stream();
                        send_done(stopped ? "stop" : "stop");
                        std::cerr << "[qw3-serve] #" << rid
                                  << " chat(stream) completion_tokens="
                                  << completion_tokens
                                  << " prompt_tokens=" << prompt_token_count
                                  << " route=" << route
                                  << (client_closed ? " client_closed=true" : "")
                                  << "\n";
                        return true;
                    } catch (const std::exception &e) {
                        json chunk = {
                            {"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_id},
                            {"choices", json::array({json{
                                {"index", 0},
                                {"delta", json::object()},
                                {"finish_reason", "error"}}})},
                            {"error", e.what()}};
                        const std::string s = "data: " + dump_json(chunk) + "\n\n";
                        send_raw(s);
                        const std::string fin = "data: [DONE]\n\n";
                        send_raw(fin);
                        sink.done();
                        std::cerr << "[qw3-serve] #" << rid
                                  << " chat(stream) error=" << e.what() << "\n";
                        return false;
                    }
                });
            return;
        }

        std::string text;
        size_t completion_tokens = 0;
        try {
            if (g.continuous_batching) {
                eng.generate_stream(prompt, g, [&](const std::string &piece) {
                    ++completion_tokens;
                    text += piece;
                });
            } else {
                std::lock_guard<std::mutex> lk(gen_mu);
                eng.generate_stream(prompt, g, [&](const std::string &piece) {
                    ++completion_tokens;
                    text += piece;
                });
            }
        } catch (const std::exception &e) {
            std::cerr << "[qw3-serve] #" << rid << " chat error="
                      << e.what() << "\n";
            set_error_response(res, status_for_exception(e), e.what());
            return;
        }
        if (enable_thinking) text = "<think>\n" + text;
        std::string utf8_pending;
        text = take_complete_utf8(utf8_pending, text);
        text += flush_utf8_pending(utf8_pending, false);
        const bool stopped = apply_stops(text, stops);
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();
        std::cerr << "[qw3-serve] #" << rid << " chat chars=" << text.size()
                  << " completion_tokens=" << completion_tokens
                  << " prompt_tokens=" << prompt_token_count
                  << " route=" << route;
        if (!fallback_reason.empty()) {
            std::cerr << " fallback_reason=" << fallback_reason;
        }
        std::cerr << " " << ms << "ms\n";
        const std::vector<json> tool_calls = parse_tool_calls_xml(text);
        const ReasoningSplit split = split_reasoning(text);
        json message = json{{"role", "assistant"},
                            {"content", tool_calls.empty() ? split.content : ""}};
        if (!split.reasoning.empty()) {
            message["reasoning_content"] = split.reasoning;
        }
        std::string finish = stopped ? "stop" : "stop";
        if (!tool_calls.empty()) {
            message["tool_calls"] = tool_calls;
            finish = "tool_calls";
        }
        json out = {
            {"id", id}, {"object", "chat.completion"}, {"created", created},
            {"model", model_id},
            {"choices", json::array({json{
                {"index", 0},
                {"message", message},
                {"finish_reason", finish}}})},
            {"usage", usage_json(prompt_token_count, completion_tokens)}};
        res.set_content(dump_json(out), "application/json");
    });

    svr.Post("/v1/completions", [&](const httplib::Request &hreq,
                                    httplib::Response &res) {
        json req;
        try {
            req = json::parse(hreq.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", std::string("invalid JSON: ") + e.what()}}),
                            "application/json");
            return;
        }
        std::string prompt;
        if (req.contains("prompt") && req["prompt"].is_string()) {
            prompt = req["prompt"].get<std::string>();
        } else if (req.contains("prompt") && req["prompt"].is_array() &&
                   !req["prompt"].empty() && req["prompt"][0].is_string()) {
            prompt = req["prompt"][0].get<std::string>();
        } else {
            res.status = 400;
            res.set_content(dump_json(json{{"error", "missing prompt"}}),
                            "application/json");
            return;
        }
        GenerationOptions g = make_gen(req);
        g.raw_prompt = true; // /v1/completions sends raw text, no chat template
        g.continuous_batching =
            serve_continuous_batching_enabled() &&
            serve_continuous_batch_request_supported(g);
        const size_t prompt_token_count = usage_tokenizer.encode(prompt).size();
        const std::string route = g.continuous_batching ? "continuous" : "plain";
        const std::string fallback_reason =
            g.continuous_batching ? "" :
            (serve_continuous_batching_enabled() ? "request_unsupported" : "disabled");
        const std::vector<std::string> stops = parse_stops(req);
        const std::string id = gen_id("cmpl-");
        const int64_t created = unix_now();
        const uint64_t rid = ++req_counter;

        std::string text;
        size_t completion_tokens = 0;
        try {
            if (g.continuous_batching) {
                eng.generate_stream(prompt, g, [&](const std::string &piece) {
                    ++completion_tokens;
                    text += piece;
                });
            } else {
                std::lock_guard<std::mutex> lk(gen_mu);
                eng.generate_stream(prompt, g, [&](const std::string &piece) {
                    ++completion_tokens;
                    text += piece;
                });
            }
        } catch (const std::exception &e) {
            std::cerr << "[qw3-serve] #" << rid << " completion error="
                      << e.what() << "\n";
            set_error_response(res, status_for_exception(e), e.what());
            return;
        }
        std::string utf8_pending;
        text = take_complete_utf8(utf8_pending, text);
        text += flush_utf8_pending(utf8_pending, false);
        const bool stopped = apply_stops(text, stops);
        std::cerr << "[qw3-serve] #" << rid << " completion chars="
                  << text.size()
                  << " completion_tokens=" << completion_tokens
                  << " prompt_tokens=" << prompt_token_count
                  << " route=" << route;
        if (!fallback_reason.empty()) {
            std::cerr << " fallback_reason=" << fallback_reason;
        }
        std::cerr << "\n";
        json out = {
            {"id", id}, {"object", "text_completion"}, {"created", created},
            {"model", model_id},
            {"choices", json::array({json{
                {"index", 0}, {"text", text}, {"logprobs", nullptr},
                {"finish_reason", stopped ? "stop" : "length"}}})},
            {"usage", usage_json(prompt_token_count, completion_tokens)}};
        res.set_content(dump_json(out), "application/json");
    });

    svr.set_logger([](const httplib::Request &req, const httplib::Response &res) {
        if (req.path == "/health") return; // quiet the poll loop
        std::cerr << "[qw3-serve] " << req.method << " " << req.path
                  << " -> " << res.status << "\n";
    });

    std::cerr << "[qw3-serve] listening on http://" << cfg.host << ":"
              << cfg.port << "  (model loaded once, continuous_batching="
              << (serve_continuous_batching_enabled() ? "on" : "off") << ")\n";
    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << "[qw3-serve] failed to bind " << cfg.host << ":"
                  << cfg.port << "\n";
        return 1;
    }
    return 0;
}

} // namespace qw3
