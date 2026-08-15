#include "server.hpp"

#include "env_flags.hpp"
#include "tool_call_stream.hpp"
#include "qw3/qw3.hpp"
#include "qw3/gguf.hpp"
#include "qw3/kvmem_archive.hpp"
#include "qw3/kvmem_store.hpp"
#include "qw3/tokenizer.hpp"

// Vendored single-header deps (included as SYSTEM headers via CMake so their
// warnings don't trip -Wall -Wextra -Wpedantic).
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace qw3 {

namespace {

using json = nlohmann::json;

using ServerClock = std::chrono::steady_clock;
using ServerTimePoint = ServerClock::time_point;

double server_elapsed_seconds(ServerTimePoint begin, ServerTimePoint end) {
    return std::chrono::duration<double>(end - begin).count();
}

bool visible_stream_delta(const json &delta) {
    for (const char *key : {"content", "reasoning_content"}) {
        if (delta.contains(key) && delta[key].is_string() &&
            !delta[key].get_ref<const std::string &>().empty()) {
            return true;
        }
    }
    return delta.contains("tool_calls") && delta["tool_calls"].is_array() &&
           !delta["tool_calls"].empty();
}

// Server-side TTFT uses the request-arrival boundary, so JSON parsing,
// rendering/tokenization, queueing, prefix restoration, KVMem reselection,
// query replay, and the first decode step are all represented.  Keep both the
// first non-empty model piece and the first visible streamed payload: tool and
// reasoning framing may delay the latter even after the model has produced a
// token.
struct ServerTtftTracker {
    explicit ServerTtftTracker(ServerTimePoint request_start_in)
        : request_start(request_start_in) {}

    void start_engine(ServerTimePoint when) {
        engine_start = when;
        engine_started = true;
    }

    void observe_model_piece(const std::string &piece) {
        if (!first_model_seen && !piece.empty()) {
            first_model_seen = true;
            first_model_at = ServerClock::now();
        }
    }

    void observe_visible_output() {
        if (!first_output_seen) {
            first_output_seen = true;
            first_output_at = ServerClock::now();
        }
    }

    json timing_json(ServerTimePoint request_end) const {
        return json{
            {"server_ttft_sec", first_model_seen
                 ? json(server_elapsed_seconds(request_start, first_model_at))
                 : json(nullptr)},
            {"engine_ttft_sec", first_model_seen && engine_started
                 ? json(server_elapsed_seconds(engine_start, first_model_at))
                 : json(nullptr)},
            {"response_ttft_sec", first_output_seen
                 ? json(server_elapsed_seconds(request_start, first_output_at))
                 : json(nullptr)},
            {"request_total_sec",
             server_elapsed_seconds(request_start, request_end)}};
    }

    void log(uint64_t rid, const std::string &route, bool streaming,
             ServerTimePoint request_end) const {
        const double server_ttft_ms = first_model_seen
            ? server_elapsed_seconds(request_start, first_model_at) * 1000.0
            : -1.0;
        const double engine_ttft_ms = first_model_seen && engine_started
            ? server_elapsed_seconds(engine_start, first_model_at) * 1000.0
            : -1.0;
        const double response_ttft_ms = first_output_seen
            ? server_elapsed_seconds(request_start, first_output_at) * 1000.0
            : -1.0;
        std::cerr << std::fixed << std::setprecision(6)
                  << "[qw3-server-ttft]"
                  << " rid=" << rid
                  << " route=" << route
                  << " stream=" << (streaming ? 1 : 0)
                  << " model_token_seen=" << (first_model_seen ? 1 : 0)
                  << " server_ttft_ms=" << server_ttft_ms
                  << " engine_ttft_ms=" << engine_ttft_ms
                  << " visible_output_seen=" << (first_output_seen ? 1 : 0)
                  << " response_ttft_ms=" << response_ttft_ms
                  << " request_total_ms="
                  << server_elapsed_seconds(request_start, request_end) * 1000.0
                  << "\n";
    }

    ServerTimePoint request_start{};
    ServerTimePoint engine_start{};
    ServerTimePoint first_model_at{};
    ServerTimePoint first_output_at{};
    bool engine_started = false;
    bool first_model_seen = false;
    bool first_output_seen = false;
};

bool serve_continuous_batching_enabled() {
    return env_flag_enabled("QW3_CONTINUOUS_BATCHING");
}

void setenv_value(const char *key, const std::string &value) {
    setenv(key, value.c_str(), 1);
}

void setenv_value(const char *key, const char *value) {
    setenv(key, value, 1);
}

void setenv_value(const char *key, int value) {
    setenv_value(key, std::to_string(value));
}

void setenv_value(const char *key, uint64_t value) {
    setenv_value(key, std::to_string(value));
}

void setenv_bool(const char *key, bool value) {
    setenv_value(key, value ? "1" : "0");
}

const char *yesno(bool v) {
    return v ? "1" : "0";
}

std::string bytes_gib_label(uint64_t bytes) {
    char buf[64];
    const double gib = static_cast<double>(bytes) /
        (1024.0 * 1024.0 * 1024.0);
    std::snprintf(buf, sizeof(buf), "%.3f GiB", gib);
    return std::string(buf);
}

bool serve_continuous_batch_request_supported(const GenerationOptions &g) {
    // Request-local budgets mutate the single executor's selection policy for
    // the duration of a request. Keep them on the serialized plain/frozen path
    // until continuous batching has a per-row budget field.
    return g.max_tokens >= 0 && g.kvmem_replay_query_spans.empty() &&
           g.kvmem_semantic_budget == 0 &&
           g.kvmem_query_attention_probe_tokens == 0 &&
           g.kvmem_query_attention_score_tokens == 0 &&
           g.kvmem_query_guided_thinking_max_tokens == 0 &&
           g.kvmem_query_guided_query_max_tokens == 0 &&
           !g.kvmem_query_guided_direct;
}

json usage_json(size_t prompt_tokens, size_t completion_tokens) {
    return json{{"prompt_tokens", prompt_tokens},
                {"completion_tokens", completion_tokens},
                {"total_tokens", prompt_tokens + completion_tokens}};
}

bool valid_kvmem_cache_id(const std::string &id) {
    if (id.empty() || id.size() > 128) return false;
    return std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == ':';
    });
}

bool parse_bounded_json_u64(const json &value, uint64_t min_value,
                            uint64_t max_value, uint64_t &out) {
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed < min_value || parsed > max_value) return false;
        out = parsed;
        return true;
    }
    if (value.is_number_integer()) {
        const int64_t parsed = value.get<int64_t>();
        if (parsed < 0) return false;
        const uint64_t converted = static_cast<uint64_t>(parsed);
        if (converted < min_value || converted > max_value) return false;
        out = converted;
        return true;
    }
    return false;
}

json kvmem_cache_info_json(const KvMemLocalCacheInfo &info) {
    json out = {
        {"id", info.id},
        {"version", info.version},
        {"status", info.status},
        {"position", info.position},
        {"fingerprint", info.fingerprint},
        {"scope", info.scope},
        {"created_at", info.created_at},
        {"last_access_at", info.last_access_at},
        {"expires_at", info.expires_at == 0 ? json(nullptr)
                                             : json(info.expires_at)},
        {"selected_blocks", info.selected_blocks},
        {"total_blocks", info.total_blocks},
        {"residency", json{{"gpu_bytes", info.gpu_bytes},
                            {"cpu_bytes", info.cpu_bytes},
                            {"nvme_bytes", info.nvme_bytes}}}
    };
    return out;
}

bool parse_explicit_max_tokens(const json &req, bool &present, int &value,
                               std::string &error) {
    const char *key = nullptr;
    if (req.contains("max_tokens")) {
        key = "max_tokens";
    } else if (req.contains("max_completion_tokens")) {
        key = "max_completion_tokens";
    }
    present = key != nullptr;
    if (!present) return true;
    const json &field = req[key];
    if (!field.is_number_integer()) {
        error = std::string(key) + " must be an integer";
        return false;
    }
    const int64_t raw = field.get<int64_t>();
    if (raw < 0) {
        error = std::string(key) + " must be >= 0";
        return false;
    }
    if (raw > std::numeric_limits<int>::max()) {
        error = std::string(key) + " is too large";
        return false;
    }
    value = static_cast<int>(raw);
    return true;
}

bool parse_preserve_thinking(const json &req, bool default_value, bool &value,
                             std::string &error) {
    value = default_value;
    if (req.contains("chat_template_kwargs") &&
        !req["chat_template_kwargs"].is_null()) {
        const json &kwargs = req["chat_template_kwargs"];
        if (!kwargs.is_object()) {
            error = "chat_template_kwargs must be an object";
            return false;
        }
        if (kwargs.contains("preserve_thinking")) {
            if (!kwargs["preserve_thinking"].is_boolean()) {
                error =
                    "chat_template_kwargs.preserve_thinking must be a boolean";
                return false;
            }
            value = kwargs["preserve_thinking"].get<bool>();
        }
    }
    // A top-level value is the direct qw3 API and takes precedence over the
    // Hugging Face/vLLM-compatible chat_template_kwargs form.
    if (req.contains("preserve_thinking")) {
        if (!req["preserve_thinking"].is_boolean()) {
            error = "preserve_thinking must be a boolean";
            return false;
        }
        value = req["preserve_thinking"].get<bool>();
    }
    return true;
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
    if (dynamic_cast<const std::invalid_argument *>(&e)) return 400;
    if (msg.find("KVMem local cache not found") != std::string::npos) return 404;
    if (msg.find("KVMem local cache expired") != std::string::npos ||
        msg.find("KVMem local cache evicted") != std::string::npos) return 410;
    if (msg.find("KVMem local cache version conflict") != std::string::npos)
        return 409;
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

json make_tool_call_json(const std::string &name, const json &args) {
    if (name.empty()) return json();
    json normalized_args = args.is_object() ? args : json::object();
    return json{
        {"id", gen_id("call_")},
        {"type", "function"},
        {"function", json{{"name", name},
                          {"arguments", dump_json(normalized_args)}}}
    };
}

bool parse_json_tool_call_value(const json &value, std::vector<json> &calls) {
    if (value.is_array()) {
        bool any = false;
        for (const auto &item : value) {
            any = parse_json_tool_call_value(item, calls) || any;
        }
        return any;
    }
    if (!value.is_object()) return false;

    const json *fn = &value;
    if (value.contains("function") && value["function"].is_object()) {
        fn = &value["function"];
    }
    if (!fn->is_object()) return false;

    std::string name;
    if (fn->contains("name") && (*fn)["name"].is_string()) {
        name = (*fn)["name"].get<std::string>();
    } else if (fn->contains("tool") && (*fn)["tool"].is_string()) {
        name = (*fn)["tool"].get<std::string>();
    }
    if (name.empty()) return false;

    json args = json::object();
    if (fn->contains("arguments")) {
        const json &raw = (*fn)["arguments"];
        if (raw.is_object()) {
            args = raw;
        } else if (raw.is_string()) {
            try {
                json parsed = json::parse(raw.get<std::string>());
                if (parsed.is_object()) args = parsed;
            } catch (...) {
                args = json{{"arguments", raw.get<std::string>()}};
            }
        }
    } else if (fn->contains("parameters") && (*fn)["parameters"].is_object()) {
        args = (*fn)["parameters"];
    }
    calls.push_back(make_tool_call_json(name, args));
    return true;
}

bool parse_json_tool_call_text(const std::string &text, std::vector<json> &calls) {
    const std::string trimmed = trim_ascii_ws(text);
    if (trimmed.empty()) return false;
    try {
        const json value = json::parse(trimmed);
        return parse_json_tool_call_value(value, calls);
    } catch (...) {
        return false;
    }
}

// XML <parameter=k>v</parameter> values are always textual. Coerce a single
// scalar string into the JSON type the tool schema declares. On any failure to
// represent the value as the requested type, fall back to the original string
// so a malformed value never drops the parameter.
json coerce_scalar_string(const std::string &raw, const std::string &type) {
    const std::string value = trim_ascii_ws(raw);
    if (type == "integer") {
        try {
            size_t idx = 0;
            const long long v = std::stoll(value, &idx);
            if (!value.empty() && idx == value.size()) return json(v);
        } catch (...) {}
        return json(raw);
    }
    if (type == "number") {
        try {
            size_t idx = 0;
            const double v = std::stod(value, &idx);
            if (!value.empty() && idx == value.size()) return json(v);
        } catch (...) {}
        return json(raw);
    }
    if (type == "boolean") {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (lower == "true") return json(true);
        if (lower == "false") return json(false);
        return json(raw);
    }
    if (type == "null") {
        if (value == "null" || value.empty()) return json(nullptr);
        return json(raw);
    }
    if (type == "array" || type == "object") {
        try {
            const json parsed = json::parse(value);
            if (type == "array" && parsed.is_array()) return parsed;
            if (type == "object" && parsed.is_object()) return parsed;
        } catch (...) {}
        return json(raw);
    }
    return json(raw);  // "string" or unknown type
}

// Coerce a textual parameter value using a JSON-schema property node. Supports
// a scalar "type", a "type" array (union), and simple anyOf/oneOf. For unions
// the first sub-schema that yields a non-string (a real coercion) wins, falling
// back to the raw string when nothing matches.
json coerce_value_by_schema(const std::string &value, const json &schema) {
    if (!schema.is_object()) return json(value);
    for (const char *combo : {"anyOf", "oneOf"}) {
        if (schema.contains(combo) && schema[combo].is_array()) {
            for (const auto &sub : schema[combo]) {
                const json coerced = coerce_value_by_schema(value, sub);
                if (!coerced.is_string()) return coerced;
            }
            return json(value);
        }
    }
    if (schema.contains("type")) {
        const json &t = schema["type"];
        if (t.is_string()) {
            return coerce_scalar_string(value, t.get<std::string>());
        }
        if (t.is_array()) {
            for (const auto &tt : t) {
                if (!tt.is_string()) continue;
                const std::string ty = tt.get<std::string>();
                if (ty == "string") return json(value);
                const json coerced = coerce_scalar_string(value, ty);
                if (!coerced.is_string()) return coerced;
            }
            return json(value);
        }
    }
    return json(value);
}

// Find the JSON-schema "properties" map for a named function in an OpenAI
// tools array. Returns nullptr when the tool, its parameters, or its properties
// are absent, in which case parameters are left as raw strings.
const json *find_tool_definition(const json *tools, const std::string &name) {
    if (!tools || !tools->is_array()) return nullptr;
    for (const auto &t : *tools) {
        if (!t.is_object()) continue;
        const json *fn = (t.contains("function") && t["function"].is_object())
                             ? &t["function"]
                             : &t;
        if (!fn->is_object()) continue;
        if (fn->value("name", std::string()) != name) continue;
        return fn;
    }
    return nullptr;
}

const json *find_tool_properties(const json *tools, const std::string &name) {
    const json *fn = find_tool_definition(tools, name);
    if (!fn || !fn->contains("parameters") || !(*fn)["parameters"].is_object()) {
        return nullptr;
    }
    const json &params = (*fn)["parameters"];
    if (params.contains("properties") && params["properties"].is_object()) {
        return &params["properties"];
    }
    return nullptr;
}

bool tool_name_allowed(const json *tools, const std::string &name) {
    return !tools || !tools->is_array() || find_tool_definition(tools, name) != nullptr;
}

std::string strip_tool_control_tokens(std::string text) {
    // Some Qwen generations leak chat-template control tokens inside the
    // tool block. They are framing noise, not part of the tool name/arguments.
    for (const char *token : {"<|im_start|>", "<|im_end|>", "<|assistant|>",
                              "<|tool|>"}) {
        size_t pos = 0;
        while ((pos = text.find(token, pos)) != std::string::npos) {
            text.erase(pos, std::string(token).size());
        }
    }
    return text;
}

std::string trim_single_newlines(std::string value) {
    if (!value.empty() && value.front() == '\n') value.erase(value.begin());
    if (!value.empty() && value.front() == '\r') value.erase(value.begin());
    if (!value.empty() && value.back() == '\n') value.pop_back();
    if (!value.empty() && value.back() == '\r') value.pop_back();
    return value;
}

std::string normalized_identifier(std::string value) {
    std::string out;
    for (const unsigned char c : value) {
        if (std::isalnum(c) || c == '_' || c == '-') {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

std::string snake_case_identifier(const std::string &value) {
    std::string out;
    for (size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (std::isupper(c) && i > 0) out.push_back('_');
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

std::string schema_property_name(const json *props, const std::string &raw_key) {
    if (!props || !props->is_object()) return trim_ascii_ws(raw_key);
    const std::string wanted = normalized_identifier(raw_key);
    for (auto it = props->begin(); it != props->end(); ++it) {
        if (normalized_identifier(it.key()) == wanted) return it.key();
    }
    return trim_ascii_ws(raw_key);
}

void add_tool_argument(json &args, const json *props,
                       const std::string &raw_key, std::string value) {
    const std::string key = schema_property_name(props, raw_key);
    if (key.empty()) return;
    value = trim_single_newlines(std::move(value));
    if (props && props->contains(key) && (*props)[key].is_object()) {
        args[key] = coerce_value_by_schema(value, (*props)[key]);
    } else {
        args[key] = value;
    }
}

void parse_parameter_tags(const std::string &block, const json *props, json &args) {
    size_t pos = 0;
    while ((pos = block.find("<parameter=", pos)) != std::string::npos) {
        const size_t key0 = pos + std::string("<parameter=").size();
        const size_t key1 = block.find('>', key0);
        if (key1 == std::string::npos) break;
        const size_t value0 = key1 + 1;
        const size_t value1 = block.find("</parameter>", value0);
        if (value1 == std::string::npos) break;
        add_tool_argument(args, props, block.substr(key0, key1 - key0),
                          block.substr(value0, value1 - value0));
        pos = value1 + std::string("</parameter>").size();
    }
}

void parse_arg_key_value_tags(const std::string &block, const json *props, json &args) {
    size_t pos = 0;
    while ((pos = block.find("<arg_key>", pos)) != std::string::npos) {
        const size_t key0 = pos + std::string("<arg_key>").size();
        const size_t key1 = block.find("</arg_key>", key0);
        if (key1 == std::string::npos) break;
        const size_t value_tag = block.find("<arg_value>", key1);
        if (value_tag == std::string::npos) break;
        const size_t value0 = value_tag + std::string("<arg_value>").size();
        size_t value1 = block.find("</arg_value>", value0);
        if (value1 == std::string::npos) value1 = block.find("</parameter>", value0);
        if (value1 == std::string::npos) value1 = block.find("</tool_call>", value0);
        if (value1 == std::string::npos) value1 = block.size();
        if (value1 == std::string::npos) break;
        add_tool_argument(args, props, block.substr(key0, key1 - key0),
                          block.substr(value0, value1 - value0));
        pos = value1;
    }
}

void parse_loose_parameter_tags(const std::string &block, const json *props, json &args) {
    // Handles variants such as <parameter>lines>[680, 720] where the model
    // omitted the '=' and used the next '>' as the key/value separator.
    size_t pos = 0;
    while ((pos = block.find("<parameter>", pos)) != std::string::npos) {
        const size_t key0 = pos + std::string("<parameter>").size();
        const size_t key1 = block.find('>', key0);
        if (key1 == std::string::npos) break;
        const size_t value0 = key1 + 1;
        size_t value1 = block.find("</parameter>", value0);
        if (value1 == std::string::npos) value1 = block.find("</tool_call>", value0);
        if (value1 == std::string::npos) value1 = block.size();
        if (value1 == std::string::npos) break;
        add_tool_argument(args, props, block.substr(key0, key1 - key0),
                          block.substr(value0, value1 - value0));
        pos = value1;
    }
}

void parse_schema_named_tags(const std::string &block, const json *props, json &args) {
    if (!props || !props->is_object()) return;
    for (auto it = props->begin(); it != props->end(); ++it) {
        const std::string key = it.key();
        const std::string snake_key = snake_case_identifier(key);
        const std::vector<std::string> tag_names =
            snake_key == key ? std::vector<std::string>{key}
                              : std::vector<std::string>{key, snake_key};
        for (const std::string &tag_name : tag_names) {
            const std::string open = "<" + tag_name + ">";
            size_t pos = 0;
            while ((pos = block.find(open, pos)) != std::string::npos) {
                const size_t value0 = pos + open.size();
                const std::string close = "</" + tag_name + ">";
                size_t value1 = block.find(close, value0);
                if (value1 == std::string::npos) {
                    // A seen malformed form uses <file_path>value<parameter>...
                    // rather than a matching closing tag. Limit recovery to
                    // the next tool/parameter delimiter so code text cannot swallow
                    // the rest of the request.
                    value1 = block.find("<parameter", value0);
                    const size_t arg_key = block.find("<arg_key>", value0);
                    if (value1 == std::string::npos ||
                        (arg_key != std::string::npos && arg_key < value1)) {
                        value1 = arg_key;
                    }
                    const size_t end = block.find("</tool_call>", value0);
                    if (value1 == std::string::npos ||
                        (end != std::string::npos && end < value1)) {
                        value1 = end;
                    }
                    if (value1 == std::string::npos) value1 = block.size();
                }
                if (value1 == std::string::npos) break;
                add_tool_argument(args, props, key, block.substr(value0, value1 - value0));
                pos = value1 + (block.compare(value1, close.size(), close) == 0
                                    ? close.size() : 1);
            }
        }
    }
}

std::string tool_name_from_prefix(const std::string &inner, const json *tools) {
    const std::string text = trim_ascii_ws(inner);
    size_t pos = 0;
    while (pos < text.size() &&
           !(std::isalnum(static_cast<unsigned char>(text[pos])) ||
             text[pos] == '_' || text[pos] == '-')) {
        ++pos;
    }
    if (pos == text.size()) return {};
    size_t end = pos;
    while (end < text.size() &&
           (std::isalnum(static_cast<unsigned char>(text[end])) ||
            text[end] == '_' || text[end] == '-')) {
        ++end;
    }
    std::string candidate = text.substr(pos, end - pos);
    if (candidate == "function") {
        while (end < text.size() && (text[end] == ' ' || text[end] == '=' ||
                                     text[end] == '<' || text[end] == '>')) {
            ++end;
        }
        const size_t name0 = end;
        while (end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[end])) ||
                text[end] == '_' || text[end] == '-')) {
            ++end;
        }
        candidate = text.substr(name0, end - name0);
    }
    return tool_name_allowed(tools, candidate) ? candidate : std::string();
}

std::string tool_name_from_marker(const std::string &inner, const json *tools) {
    for (const std::string &marker : {"<function=", "function=", "function_"}) {
        size_t pos = inner.find(marker);
        if (pos == std::string::npos) continue;
        pos += marker.size();
        size_t end = pos;
        while (end < inner.size() &&
               (std::isalnum(static_cast<unsigned char>(inner[end])) ||
                inner[end] == '_' || inner[end] == '-')) {
            ++end;
        }
        const std::string candidate = inner.substr(pos, end - pos);
        if (tool_name_allowed(tools, candidate)) return candidate;
    }
    return tool_name_from_prefix(inner, tools);
}

std::string infer_tool_name_from_arguments(const json *tools, const json &args) {
    if (!tools || !tools->is_array() || !args.is_object() || args.empty()) return {};
    std::string match;
    size_t best_required = 0;
    bool tied = false;
    for (const auto &tool : *tools) {
        if (!tool.is_object()) continue;
        const json *fn = (tool.contains("function") && tool["function"].is_object())
                             ? &tool["function"] : &tool;
        if (!fn->is_object() || !fn->contains("name") ||
            !fn->contains("parameters") || !(*fn)["parameters"].is_object()) {
            continue;
        }
        const json &required = (*fn)["parameters"].value("required", json::array());
        if (!required.is_array() || required.empty()) continue;
        bool all_present = true;
        for (const auto &key : required) {
            if (!key.is_string() || !args.contains(key.get<std::string>())) {
                all_present = false;
                break;
            }
        }
        if (all_present) {
            const size_t required_count = required.size();
            if (required_count > best_required) {
                best_required = required_count;
                match = fn->value("name", std::string());
                tied = false;
            } else if (required_count == best_required) {
                tied = true;
            }
        }
    }
    return tied ? std::string() : match;
}

bool parse_tool_call_block(const std::string &inner, const json *tools,
                           std::vector<json> &calls) {
    if (parse_json_tool_call_text(inner, calls)) return true;

    const std::string normalized = strip_tool_control_tokens(inner);
    std::string name = tool_name_from_marker(normalized, tools);
    const json *props = find_tool_properties(tools, name);
    json args = json::object();
    parse_parameter_tags(normalized, props, args);
    parse_arg_key_value_tags(normalized, props, args);
    parse_loose_parameter_tags(normalized, props, args);
    parse_schema_named_tags(normalized, props, args);

    // A few generations place a JSON object directly after function_edit>.
    // Parse it only when it starts at an object boundary; arbitrary code text
    // must never be interpreted as tool arguments.
    if (name.empty() || args.empty()) {
        const size_t object0 = normalized.find('{');
        const size_t object1 = normalized.rfind('}');
        if (object0 != std::string::npos && object1 > object0) {
            std::vector<json> parsed;
            if (parse_json_tool_call_text(
                    "{\"name\":" + dump_json(name) +
                    ",\"arguments\":" +
                        normalized.substr(object0, object1 - object0 + 1) + "}",
                    parsed) && !parsed.empty()) {
                if (!name.empty() || parsed.front().contains("function")) {
                    calls.push_back(parsed.front());
                    return true;
                }
            }
        }
    }

    if (name.empty()) name = infer_tool_name_from_arguments(tools, args);
    if (name.empty() || !tool_name_allowed(tools, name)) return false;
    calls.push_back(make_tool_call_json(name, args));
    return true;
}

bool json_schema_type_matches(const json &value, const std::string &type) {
    if (type == "null") return value.is_null();
    if (type == "boolean") return value.is_boolean();
    if (type == "integer") return value.is_number_integer();
    if (type == "number") return value.is_number();
    if (type == "string") return value.is_string();
    if (type == "array") return value.is_array();
    if (type == "object") return value.is_object();
    return true;
}

bool validate_json_schema_value(const json &value, const json &schema,
                                const std::string &path,
                                std::string &error) {
    if (!schema.is_object()) return true;
    if (schema.contains("const") && value != schema["const"]) {
        error = path + " does not match const";
        return false;
    }
    if (schema.contains("enum") && schema["enum"].is_array()) {
        bool matched = false;
        for (const auto &candidate : schema["enum"]) {
            if (value == candidate) {
                matched = true;
                break;
            }
        }
        if (!matched) {
            error = path + " is not an allowed enum value";
            return false;
        }
    }
    for (const char *key : {"anyOf", "oneOf"}) {
        if (!schema.contains(key) || !schema[key].is_array()) continue;
        size_t matches = 0;
        for (const auto &sub : schema[key]) {
            std::string ignored;
            if (validate_json_schema_value(value, sub, path, ignored)) {
                ++matches;
            }
        }
        const bool ok = std::string(key) == "oneOf" ? matches == 1
                                                     : matches > 0;
        if (!ok) {
            error = path + " does not satisfy " + key;
            return false;
        }
    }
    if (schema.contains("type")) {
        const json &type = schema["type"];
        bool matched = false;
        if (type.is_string()) {
            matched = json_schema_type_matches(
                value, type.get<std::string>());
        } else if (type.is_array()) {
            for (const auto &candidate : type) {
                if (candidate.is_string() &&
                    json_schema_type_matches(
                        value, candidate.get<std::string>())) {
                    matched = true;
                    break;
                }
            }
        } else {
            matched = true;
        }
        if (!matched) {
            error = path + " has the wrong JSON type";
            return false;
        }
    }
    if (value.is_object()) {
        const json &required = schema.value("required", json::array());
        if (required.is_array()) {
            for (const auto &key : required) {
                if (key.is_string() &&
                    !value.contains(key.get<std::string>())) {
                    error = path + " is missing required property " +
                            key.get<std::string>();
                    return false;
                }
            }
        }
        const json *properties =
            schema.contains("properties") && schema["properties"].is_object()
                ? &schema["properties"] : nullptr;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (properties && properties->contains(it.key())) {
                if (!validate_json_schema_value(
                        it.value(), (*properties)[it.key()],
                        path + "." + it.key(), error)) {
                    return false;
                }
            } else if (schema.contains("additionalProperties") &&
                       schema["additionalProperties"].is_boolean() &&
                       !schema["additionalProperties"].get<bool>()) {
                error = path + " contains unknown property " + it.key();
                return false;
            }
        }
    }
    if (value.is_array() && schema.contains("items") &&
        schema["items"].is_object()) {
        for (size_t i = 0; i < value.size(); ++i) {
            if (!validate_json_schema_value(
                    value[i], schema["items"],
                    path + "[" + std::to_string(i) + "]", error)) {
                return false;
            }
        }
    }
    return true;
}

bool validate_tool_call(const json &call, const json *tools,
                        std::string &error) {
    if (!call.is_object() || !call.contains("function") ||
        !call["function"].is_object()) {
        error = "tool call is missing a function object";
        return false;
    }
    const json &fn = call["function"];
    const std::string name = fn.value("name", std::string());
    if (name.empty() || !tool_name_allowed(tools, name)) {
        error = "tool call has an unknown function name";
        return false;
    }
    json args;
    try {
        args = json::parse(fn.value("arguments", std::string()));
    } catch (const std::exception &e) {
        error = std::string("tool arguments are not valid JSON: ") + e.what();
        return false;
    }
    if (!args.is_object()) {
        error = "tool arguments must be a JSON object";
        return false;
    }
    const json *definition = find_tool_definition(tools, name);
    if (definition && definition->contains("parameters")) {
        return validate_json_schema_value(
            args, (*definition)["parameters"], "arguments", error);
    }
    return true;
}

struct ToolParseResult {
    std::vector<json> calls;
    bool intent_detected = false;
    bool valid = true;
    std::string error;
};

ToolParseResult parse_tool_calls_xml(const std::string &text,
                                     const json *tools = nullptr) {
    ToolParseResult result;
    size_t pos = 0;
    while (true) {
        const size_t tc0 = text.find("<tool_call>", pos);
        if (tc0 == std::string::npos) break;
        result.intent_detected = true;
        const size_t tc1 = text.find("</tool_call>", tc0);
        if (tc1 == std::string::npos) {
            result.valid = false;
            result.error = "incomplete <tool_call> block";
            break;
        }
        const size_t inner0 = tc0 + std::string("<tool_call>").size();
        const std::string inner = text.substr(inner0, tc1 - inner0);
        std::vector<json> block_calls;
        if (!parse_tool_call_block(inner, tools, block_calls) ||
            block_calls.empty()) {
            result.valid = false;
            result.error = "tool block could not be parsed";
            break;
        }
        for (const auto &call : block_calls) {
            std::string validation_error;
            if (!validate_tool_call(call, tools, validation_error)) {
                result.valid = false;
                result.error = std::move(validation_error);
                break;
            }
            result.calls.push_back(call);
        }
        if (!result.valid) break;
        pos = tc1 + std::string("</tool_call>").size();
    }
    if (result.intent_detected && result.calls.empty() && result.valid) {
        result.valid = false;
        result.error = "tool intent produced no tool calls";
    }
    return result;
}

std::string tool_call_parse_error_message(const std::string &reason) {
    return "tool_call_parse_error: " +
           (reason.empty() ? "tool call could not be parsed" : reason);
}

json tool_call_delta(const json &calls, size_t begin = 0) {
    json deltas = json::array();
    for (size_t i = begin; i < calls.size(); ++i) {
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

bool schema_is_plain_string(const json &schema) {
    return schema.is_object() && schema.contains("type") &&
           schema["type"].is_string() &&
           schema["type"].get<std::string>() == "string";
}

bool tool_has_only_string_properties(const json *tools,
                                     const std::string &name) {
    const json *props = find_tool_properties(tools, name);
    if (!props || !props->is_object() || props->empty()) return false;
    for (auto it = props->begin(); it != props->end(); ++it) {
        if (!schema_is_plain_string(it.value())) return false;
    }
    return true;
}

bool tool_allows_incremental_additional_property(
        const json *tools, const std::string &name) {
    const json *definition = find_tool_definition(tools, name);
    if (!definition || !definition->contains("parameters") ||
        !(*definition)["parameters"].is_object()) {
        return true;
    }
    const json &parameters = (*definition)["parameters"];
    if (!parameters.contains("additionalProperties")) return true;
    const json &additional = parameters["additionalProperties"];
    if (additional.is_boolean()) return additional.get<bool>();
    return schema_is_plain_string(additional);
}

json incremental_tool_start_delta(size_t index,
                                  const std::string &id,
                                  const std::string &name,
                                  const std::string &arguments) {
    return json{{"tool_calls", json::array({json{
        {"index", static_cast<int>(index)},
        {"id", id},
        {"type", "function"},
        {"function", json{{"name", name},
                          {"arguments", arguments}}}
    }})}};
}

json incremental_tool_arguments_delta(size_t index,
                                      const std::string &arguments) {
    return json{{"tool_calls", json::array({json{
        {"index", static_cast<int>(index)},
        {"function", json{{"arguments", arguments}}}
    }})}};
}

class IncrementalToolCallStream {
public:
    explicit IncrementalToolCallStream(const json *tools) : tools_(tools) {}

    void feed(const std::string &text) {
        if (abandoned_ || fatal_ || parser_.complete()) return;
        std::vector<detail::ToolCallStreamEvent> events;
        if (!parser_.feed(text, events)) {
            parser_failed();
            return;
        }
        process(events);
    }

    void finish(bool allow_closure_repair,
                std::string &synthesized_suffix) {
        synthesized_suffix.clear();
        if (abandoned_ || fatal_) return;
        std::vector<detail::ToolCallStreamEvent> events;
        const bool ok = allow_closure_repair
            ? parser_.finish_with_closure_repair(events, synthesized_suffix)
            : parser_.finish(events);
        if (!ok) {
            parser_failed();
            return;
        }
        process(events);
    }

    bool streaming() const { return streaming_; }
    bool abandoned() const { return abandoned_; }
    bool fatal() const { return fatal_; }
    bool complete() const { return complete_; }
    bool start_pending() const { return start_pending_; }
    size_t pending_size() const { return pending_arguments_.size(); }
    size_t argument_size() const { return arguments_all_.size(); }
    const std::string &error() const { return error_; }

    json finalized_call() const {
        if (!complete_ || name_.empty() || call_id_.empty()) return json();
        return json{
            {"id", call_id_},
            {"type", "function"},
            {"function",
             json{{"name", name_},
                  {"arguments", arguments_all_}}}};
    }

    bool ready_to_commit() const {
        if (!streaming_ || fatal_ || abandoned_) return false;
        const json *definition = find_tool_definition(tools_, name_);
        if (!definition || !definition->contains("parameters") ||
            !(*definition)["parameters"].is_object()) {
            return true;
        }
        const json &required =
            (*definition)["parameters"].value("required", json::array());
        if (!required.is_array()) return true;
        for (const auto &key : required) {
            if (key.is_string() &&
                seen_parameters_.count(key.get<std::string>()) == 0) {
                return false;
            }
        }
        return true;
    }

    json take_start_delta() {
        start_pending_ = false;
        const std::string fragment = take_pending_arguments();
        return incremental_tool_start_delta(0, call_id_, name_, fragment);
    }

    json take_arguments_delta() {
        return incremental_tool_arguments_delta(0, take_pending_arguments());
    }

    bool validate(const std::vector<json> &calls, std::string &reason) const {
        if (!streaming_ || fatal_ || !complete_) {
            reason = fatal_ ? error_ : "canonical stream did not complete";
            return false;
        }
        if (calls.empty() || !calls.front().is_object() ||
            !calls.front().contains("function") ||
            !calls.front()["function"].is_object()) {
            reason = "full parser did not produce the streamed tool call";
            return false;
        }
        const json &fn = calls.front()["function"];
        if (fn.value("name", std::string()) != name_) {
            reason = "streamed and fully parsed function names differ";
            return false;
        }
        try {
            const json streamed = json::parse(arguments_all_);
            const json parsed =
                json::parse(fn.value("arguments", std::string("{}")));
            if (streamed != parsed) {
                reason = "streamed and fully parsed arguments differ";
                return false;
            }
        } catch (const std::exception &e) {
            reason = std::string("argument validation failed: ") + e.what();
            return false;
        }
        return true;
    }

private:
    void parser_failed() {
        if (streaming_) {
            fatal_ = true;
            error_ = parser_.error();
        } else {
            abandoned_ = true;
        }
    }

    void append_arguments(const std::string &text) {
        arguments_all_ += text;
        pending_arguments_ += text;
    }

    std::string take_pending_arguments() {
        std::string out;
        out.swap(pending_arguments_);
        return out;
    }

    void fail(std::string message) {
        fatal_ = true;
        error_ = std::move(message);
    }

    void process(const std::vector<detail::ToolCallStreamEvent> &events) {
        for (const detail::ToolCallStreamEvent &event : events) {
            if (abandoned_ || fatal_) return;
            switch (event.kind) {
                case detail::ToolCallStreamEventKind::ToolStart:
                    name_ = event.value;
                    if (!tool_name_allowed(tools_, name_) ||
                        !tool_has_only_string_properties(tools_, name_)) {
                        abandoned_ = true;
                        return;
                    }
                    call_id_ = gen_id("call_");
                    streaming_ = true;
                    start_pending_ = true;
                    append_arguments("{");
                    break;
                case detail::ToolCallStreamEventKind::ParameterStart: {
                    if (!streaming_ || parameter_open_) {
                        fail("invalid incremental parameter start");
                        return;
                    }
                    const json *props = find_tool_properties(tools_, name_);
                    parameter_name_ =
                        schema_property_name(props, event.value);
                    const bool known_parameter =
                        props && props->contains(parameter_name_);
                    if (parameter_name_.empty() ||
                        (!known_parameter &&
                         !tool_allows_incremental_additional_property(
                             tools_, name_)) ||
                        !seen_parameters_.insert(parameter_name_).second) {
                        fail("unknown, empty, or duplicate incremental parameter");
                        return;
                    }
                    if (!first_parameter_) append_arguments(",");
                    first_parameter_ = false;
                    append_arguments(dump_json(json(parameter_name_)) + ":\"");
                    parameter_open_ = true;
                    break;
                }
                case detail::ToolCallStreamEventKind::ParameterData:
                    if (!parameter_open_) {
                        fail("incremental parameter data outside a parameter");
                        return;
                    }
                    append_arguments(
                        detail::json_string_fragment(event.value));
                    break;
                case detail::ToolCallStreamEventKind::ParameterEnd:
                    if (!parameter_open_) {
                        fail("incremental parameter end without a parameter");
                        return;
                    }
                    append_arguments("\"");
                    parameter_name_.clear();
                    parameter_open_ = false;
                    break;
                case detail::ToolCallStreamEventKind::ToolEnd:
                    if (!streaming_ || parameter_open_) {
                        fail("incremental tool ended inside a parameter");
                        return;
                    }
                    append_arguments("}");
                    complete_ = true;
                    break;
            }
        }
    }

    const json *tools_ = nullptr;
    detail::CanonicalToolCallStreamParser parser_;
    std::string name_;
    std::string call_id_;
    std::string parameter_name_;
    std::string arguments_all_;
    std::string pending_arguments_;
    std::string error_;
    std::unordered_set<std::string> seen_parameters_;
    bool streaming_ = false;
    bool abandoned_ = false;
    bool fatal_ = false;
    bool complete_ = false;
    bool start_pending_ = false;
    bool parameter_open_ = false;
    bool first_parameter_ = true;
};

std::string tool_calls_debug_summary(const std::vector<json> &calls) {
    json summary = json::array();
    for (const auto &call : calls) {
        json item = json::object();
        if (call.contains("function") && call["function"].is_object()) {
            const json &fn = call["function"];
            item["name"] = fn.value("name", "");
            json keys = json::array();
            try {
                const json args =
                    json::parse(fn.value("arguments", "{}"));
                if (args.is_object()) {
                    for (auto it = args.begin(); it != args.end(); ++it) {
                        keys.push_back(it.key());
                    }
                }
            } catch (...) {
                keys.push_back("<invalid-json-arguments>");
            }
            item["argument_keys"] = keys;
        }
        summary.push_back(item);
    }
    return dump_json(summary);
}

std::string tools_debug_summary(const json &tools) {
    json summary = json::array();
    if (!tools.is_array()) return "[]";
    for (const auto &tool : tools) {
        if (!tool.is_object()) continue;
        const json *fn = &tool;
        if (tool.contains("function") && tool["function"].is_object()) {
            fn = &tool["function"];
        }
        json item = json::object();
        item["name"] = fn->value("name", "");
        json required = json::array();
        if (fn->contains("parameters") && (*fn)["parameters"].is_object()) {
            const json &params = (*fn)["parameters"];
            if (params.contains("required") && params["required"].is_array()) {
                required = params["required"];
            }
        }
        item["required"] = required;
        summary.push_back(item);
    }
    return dump_json(summary);
}

// Streaming tool-call detection. The model may emit natural-language reasoning
// before a <tool_call> block (the Hermes prompt at render_messages explicitly
// allows this), so the stream cannot be classified once on its first token.
// Content streams until a marker appears; canonical string arguments then stream
// as OpenAI deltas, while recovery formats remain buffered for the full parser.
//
// Returns how many leading bytes of `text` are safe to emit as content right
// now. If a complete "<tool_call>" marker is present, returns its byte offset
// and sets marker_found. Otherwise holds back the longest tail of `text` that
// could be the start of a "<tool_call>" marker still being streamed.
size_t tool_call_safe_emit_len(const std::string &text, bool &marker_found) {
    static const std::string marker = "<tool_call>";
    marker_found = false;
    const size_t pos = text.find(marker);
    if (pos != std::string::npos) {
        marker_found = true;
        return pos;
    }
    const size_t max_partial = std::min(text.size(), marker.size() - 1);
    for (size_t k = max_partial; k > 0; --k) {
        if (text.compare(text.size() - k, k, marker, 0, k) == 0) {
            return text.size() - k;
        }
    }
    return text.size();
}

// Render an OpenAI messages[] array into a Qwen3.6 chat transcript. This mirrors
// the GGUF chat_template's text/tool subset closely enough for tool calling:
// tools are emitted in a system block, assistant tool_calls are serialized as
// Qwen XML tool calls, and tool results become user-side <tool_response> blocks.
// The final assistant header (+ thinking prefill or empty-think block) is
// appended for generation.
struct RenderedMessageSpan {
    size_t message_index = 0;
    std::string role;
    size_t segment_begin = 0;
    size_t segment_end = 0;
    size_t content_begin = 0;
    size_t content_end = 0;
};

std::string render_messages(
        const json &messages, const json *tools, bool enable_thinking,
        bool preserve_thinking,
        const std::string &forced_tool_name = {},
        std::vector<RenderedMessageSpan> *message_spans = nullptr,
        bool add_generation_prompt = true) {
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
        const std::string rendered_content =
            m.contains("content") ? render_content(m["content"]) : "";
        // Tool results can contain byte-sensitive content such as source code.
        const std::string content =
            role == "tool" ? rendered_content : trim_ascii_ws(rendered_content);
        if (role == "user") {
            const size_t segment_begin = prompt.size();
            prompt += "<|im_start|>user\n";
            const size_t content_begin = prompt.size();
            prompt += content;
            const size_t content_end = prompt.size();
            prompt += "<|im_end|>\n";
            if (message_spans) {
                message_spans->push_back(RenderedMessageSpan{
                    i, role, segment_begin, prompt.size(), content_begin,
                    content_end});
            }
        } else if (role == "assistant") {
            const size_t segment_begin = prompt.size();
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
            detail::append_historical_thinking(
                prompt, reasoning_content, preserve_thinking, i,
                last_query_index);
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
            if (message_spans) {
                message_spans->push_back(RenderedMessageSpan{
                    i, role, segment_begin, prompt.size(), 0, 0});
            }
        } else if (role == "tool") {
            const size_t segment_begin = prompt.size();
            const bool prev_tool =
                i > 0 && messages[i - 1].is_object() &&
                messages[i - 1].value("role", "") == "tool";
            const bool next_tool =
                i + 1 < messages.size() && messages[i + 1].is_object() &&
                messages[i + 1].value("role", "") == "tool";
            if (!prev_tool) prompt += "<|im_start|>user";
            prompt += "\n<tool_response>\n";
            const size_t content_begin = prompt.size();
            prompt += content;
            const size_t content_end = prompt.size();
            prompt += "\n</tool_response>";
            if (!next_tool) prompt += "<|im_end|>\n";
            if (message_spans) {
                message_spans->push_back(RenderedMessageSpan{
                    i, role, segment_begin, prompt.size(), content_begin,
                    content_end});
            }
        }
    }

    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
        if (enable_thinking) {
            prompt += "<think>\n";
        } else {
            prompt += "<think>\n\n</think>\n\n";
        }
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

const char *generation_finish_reason(bool stop_matched,
                                     size_t completion_tokens,
                                     int max_tokens) {
    // The engine returns normally both when it emits EOS and when it exhausts
    // max_tokens. A client stop string takes precedence; otherwise reaching
    // the configured token ceiling is OpenAI's "length", while an earlier
    // normal return is EOS and therefore "stop".
    if (max_tokens == 0) return "prefill_only";
    if (stop_matched) return "stop";
    if (max_tokens > 0 &&
        completion_tokens >= static_cast<size_t>(max_tokens)) {
        return "length";
    }
    return "stop";
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
    if (cfg.max_active <= 0) throw std::runtime_error("--max-active must be > 0");
    if (cfg.max_pending <= 0) throw std::runtime_error("--max-pending must be > 0");
    if (cfg.prefill_burst < 0) throw std::runtime_error("--prefill-burst must be >= 0");
    if (cfg.kv_page_size <= 0) throw std::runtime_error("--kv-page-size must be > 0");
    if (cfg.kv_pool_pages < 0) throw std::runtime_error("--kv-pool-pages must be >= 0");
    if (cfg.mtp_kv_pool_pages < 0) throw std::runtime_error("--mtp-kv-pool-pages must be >= 0");
    if (cfg.kv_dtype != "fp16" && cfg.kv_dtype != "fp32" &&
        cfg.kv_dtype != "q8" && cfg.kv_dtype != "fp8") {
        throw std::runtime_error("invalid --kv-dtype (want fp16|fp32|q8|fp8): " + cfg.kv_dtype);
    }
    uint64_t archive_prefix_tokens = 0;
    if (!engine.kvmem_archive_dir.empty()) {
        const KvMemArchiveManifest manifest =
            KvMemArchive::read_manifest(engine.kvmem_archive_dir);
        if (!manifest.sealed) {
            throw std::runtime_error(
                "archive-backed serving requires a sealed archive");
        }
        archive_prefix_tokens = engine.kvmem_archive_tokens == 0
            ? manifest.total_tokens
            : std::min<uint64_t>(engine.kvmem_archive_tokens,
                                 manifest.total_tokens);
        const uint32_t bt = std::max<uint32_t>(1, manifest.layout.block_tokens);
        archive_prefix_tokens = (archive_prefix_tokens / bt) * bt;
        engine.kvmem_archive_tokens = archive_prefix_tokens;
        if (archive_prefix_tokens == 0 ||
            archive_prefix_tokens >=
                static_cast<uint64_t>(std::max(1, engine.ctx_size))) {
            throw std::runtime_error(
                "archive-backed serving needs --ctx larger than its non-empty "
                "attached prefix");
        }
        if (engine.kvmem_semantic_expansion != "none") {
            throw std::runtime_error(
                "archive-backed serving v1 requires "
                "--kvmem-semantic-expansion none because the archive does not "
                "persist message/round group metadata");
        }
    }
    if (engine.prefill_chunk < 0) {
        engine.prefill_chunk = 2048;
    }
    if (!engine.native_mtp_chain_set) {
        engine.native_mtp_chain = 0;
    }
    if (engine.native_mtp_chain < 0) {
        throw std::runtime_error("--mtp-chain must be >= 0");
    }
    if (engine.mtp_policy != "fixed" && engine.mtp_policy != "adaptive") {
        throw std::runtime_error("invalid --mtp-policy (want fixed|adaptive): " +
                                 engine.mtp_policy);
    }
    if (engine.mtp_adaptive_min_chain < 0) {
        throw std::runtime_error("--mtp-adaptive-min-chain must be >= 0");
    }
    if (engine.mtp_adaptive_max_chain < 0) {
        throw std::runtime_error("--mtp-adaptive-max-chain must be >= 0");
    }
    if (cfg.continuous_batching) {
        if (!cfg.paged_kv_set) cfg.paged_kv = true;
        if (!cfg.body_batch_set) cfg.body_batch = true;
    }
    if (cfg.continuous_batching && !cfg.paged_kv) {
        throw std::runtime_error(
            "--continuous-batching requires paged KV; remove --no-paged-kv");
    }
    if (!cfg.paged_kv) {
        cfg.continuous_batching = false;
        cfg.mtp_paged_prefix = false;
    }
    const bool mtp_enabled = !engine.native_mtp_trace && engine.native_mtp_chain > 0;
    engine.native_mtp_speculate = mtp_enabled;
    if (mtp_enabled && cfg.continuous_batching && !cfg.mtp_batched_draft_set) {
        cfg.mtp_batched_draft = true;
    }
    if (mtp_enabled && cfg.paged_kv && !cfg.mtp_paged_prefix_set) {
        cfg.mtp_paged_prefix = true;
    }
    if (!mtp_enabled) {
        cfg.mtp_batched_draft = false;
        cfg.mtp_paged_prefix = false;
    }
    if (cfg.prefix_cache && !cfg.continuous_batching) {
        std::cerr << "[qw3-serve] --prefix-cache requires --continuous-batching; "
                     "prefix caching disabled\n";
        cfg.prefix_cache = false;
    }
    if (cfg.prefix_cache && mtp_enabled) {
        std::cerr << "[qw3-serve] --prefix-cache with MTP: caching paired "
                     "main/draft KV pages and MTP prefix state\n";
    }
    if (cfg.kvmem_prefix_cache && !engine.kvmem_enabled) {
        std::cerr << "[qw3-serve] --kvmem-prefix-cache requires --kvmem; "
                     "kvmem prefix caching disabled\n";
        cfg.kvmem_prefix_cache = false;
    }
    if (cfg.kvmem_prefix_cache && cfg.continuous_batching) {
        std::cerr << "[qw3-serve] note: --kvmem-prefix-cache applies to the plain "
                     "(non-continuous-batching) serve route; it is inert while "
                     "--continuous-batching is active\n";
    }
    if (cfg.kvmem_query_replay &&
        (!engine.kvmem_enabled || !engine.kvmem_query_conditioned)) {
        throw std::runtime_error(
            "--kvmem-query-replay requires --kvmem and "
            "--kvmem-query-conditioned");
    }
    if (cfg.kvmem_query_replay && cfg.continuous_batching) {
        std::cerr << "[qw3-serve] note: --kvmem-query-replay is currently "
                     "single-request only; it is inert while "
                     "--continuous-batching is active\n";
    }

    // The backend still reads several low-level toggles from process config.
    // Keep that as an internal bridge; the user-facing API is the explicit CLI
    // surface above, and this happens before model load.
    setenv_bool("QW3_CONTINUOUS_BATCHING", cfg.continuous_batching);
    setenv_bool("QW3_CONTINUOUS_BATCHING_BODY_BATCH", cfg.body_batch);
    setenv_bool("QW3_CONTINUOUS_MTP_BATCHED_DRAFT", cfg.mtp_batched_draft);
    setenv_bool("QW3_MTP_PAGED_PREFIX", cfg.mtp_paged_prefix);
    // Prefix caching is only meaningful on the continuous-batching path; force
    // it off otherwise. Page budget unlimited (0) remains the default, but do
    // not clobber an explicit internal limit. Long-lived frozen-branch servers
    // need that limit because every distinct prompt prefix also owns a hybrid
    // recurrent-state snapshot, whose memory is not bounded by KV-pool pressure.
    // Tracing is likewise an internal diagnostic and is intentionally left
    // untouched.
    {
        const bool prefix_cache_on = cfg.prefix_cache && cfg.continuous_batching;
        setenv_bool("QW3_PREFIX_CACHE", prefix_cache_on);
        if (std::getenv("QW3_PREFIX_CACHE_MAX_PAGES") == nullptr) {
            setenv_value("QW3_PREFIX_CACHE_MAX_PAGES", 0);
        }
    }
    // kvmem single-request prefix cache: plain-route warm reuse. Requires kvmem;
    // inert on the CB path (only generate_plain / generate_mtp on the shared
    // executor read it). Tracing left to QW3_KVMEM_PREFIX_CACHE_TRACE opt-in.
    setenv_bool("QW3_KVMEM_PREFIX_CACHE",
                cfg.kvmem_prefix_cache && engine.kvmem_enabled);
    setenv_bool("QW3_KVMEM_QUERY_REPLAY",
                cfg.kvmem_query_replay && engine.kvmem_enabled &&
                    engine.kvmem_query_conditioned && !cfg.continuous_batching);
    setenv_bool("QW3_MTP_SPECULATE", engine.native_mtp_speculate);
    setenv_bool("QW3_KVMEM_ARCHIVE_SERVE",
                !engine.kvmem_archive_dir.empty());
    setenv_value("QW3_MTP_POLICY", engine.mtp_policy);
    if (engine.mtp_adaptive_min_chain > 0) {
        setenv_value("QW3_MTP_ADAPTIVE_MIN_CHAIN",
                     engine.mtp_adaptive_min_chain);
    }
    if (engine.mtp_adaptive_max_chain > 0) {
        setenv_value("QW3_MTP_ADAPTIVE_MAX_CHAIN",
                     engine.mtp_adaptive_max_chain);
    }
    setenv_value("QW3_KV_DTYPE", cfg.kv_dtype);
    setenv_value("QW3_MATMUL", "mmq");
    setenv_bool("QW3_DISABLE_HGEMM", true);
    setenv_value("QW3_PAGED_KV_PAGE_SIZE", cfg.kv_page_size);
    setenv_value("QW3_CONTINUOUS_BATCHING_MAX_ACTIVE", cfg.max_active);
    setenv_value("QW3_CONTINUOUS_BATCHING_MAX_PENDING", cfg.max_pending);
    if (cfg.prefill_burst > 0) {
        setenv_value("QW3_CONTINUOUS_BATCHING_PREFILL_BURST", cfg.prefill_burst);
        setenv_value("QW3_CONTINUOUS_BATCHING_ACTIVE_PREFILL_BURST",
                     cfg.prefill_burst);
    }
    if (cfg.max_total_tokens_set) {
        setenv_value("QW3_CONTINUOUS_BATCHING_MAX_TOTAL_TOKENS", cfg.max_total_tokens);
    }
    if (cfg.kv_pool_pages > 0) {
        setenv_value("QW3_CONTINUOUS_BATCHING_KV_POOL_PAGES", cfg.kv_pool_pages);
    }
    if (cfg.mtp_kv_pool_pages > 0) {
        setenv_value("QW3_CONTINUOUS_BATCHING_MTP_KV_POOL_PAGES", cfg.mtp_kv_pool_pages);
    }

    const bool kvmem_all_optimizations =
        engine.kvmem_opt_stage_out &&
        engine.kvmem_opt_stage_in &&
        engine.kvmem_opt_pack;
    const KvMemKeepAllocation kvmem_keep =
        resolve_kvmem_keep_allocation(
            static_cast<uint32_t>(std::max(1, engine.kvmem_block_tokens)),
            static_cast<uint32_t>(std::max(1, engine.kvmem_budget)),
            engine.kvmem_sink_blocks,
            engine.kvmem_recent_blocks,
            engine.kvmem_sink_tokens,
            engine.kvmem_recent_tokens);
    auto keep_source_name = [](KvMemKeepSource source) {
        switch (source) {
            case KvMemKeepSource::Tokens: return "tokens";
            case KvMemKeepSource::Blocks: return "blocks";
            case KvMemKeepSource::Auto: return "auto";
        }
        return "unknown";
    };
    std::cerr << "[qw3-serve] effective serving parameters:\n"
              << "  host=" << cfg.host << "\n"
              << "  port=" << cfg.port << "\n"
              << "  model=" << engine.model_path << "\n"
              << "  backend=" << backend_kind_name(engine.backend) << "\n"
              << "  native_kernels=" << engine.native_kernels << "\n"
              << "  cpu_embedding=" << yesno(engine.cpu_embedding) << "\n"
              << "  ctx=" << engine.ctx_size << "\n"
              << "  batch=" << engine.batch_size << "\n"
              << "  prefill_chunk=" << engine.prefill_chunk << "\n"
              << "  kv_dtype=" << cfg.kv_dtype << "\n"
              << "  paged_kv=" << yesno(cfg.paged_kv) << "\n"
              << "  kv_page_size=" << cfg.kv_page_size << "\n"
              << "  kv_pool_pages=" << cfg.kv_pool_pages << " (0=auto)\n"
              << "  mtp_kv_pool_pages=" << cfg.mtp_kv_pool_pages << " (0=auto)\n"
              << "  continuous_batching=" << yesno(cfg.continuous_batching) << "\n"
              << "  body_batch=" << yesno(cfg.body_batch) << "\n"
              << "  max_active=" << cfg.max_active << "\n"
              << "  max_pending=" << cfg.max_pending << "\n"
              << "  prefill_burst="
              << (cfg.prefill_burst > 0 ? std::to_string(cfg.prefill_burst)
                                         : std::string("max-active"))
              << "\n"
              << "  max_total_tokens="
              << (cfg.max_total_tokens_set ? std::to_string(cfg.max_total_tokens)
                                           : std::string("auto(ctx)"))
              << "\n"
              << "  mtp_chain=" << engine.native_mtp_chain << "\n"
              << "  mtp_policy=" << engine.mtp_policy << "\n"
              << "  mtp_adaptive_min_chain="
              << (engine.mtp_adaptive_min_chain > 0
                      ? std::to_string(engine.mtp_adaptive_min_chain)
                      : std::string("auto"))
              << "\n"
              << "  mtp_adaptive_max_chain="
              << (engine.mtp_adaptive_max_chain > 0
                      ? std::to_string(engine.mtp_adaptive_max_chain)
                      : std::string("auto"))
              << "\n"
              << "  mtp_speculate=" << yesno(engine.native_mtp_speculate) << "\n"
              << "  mtp_batched_draft=" << yesno(cfg.mtp_batched_draft) << "\n"
              << "  mtp_paged_prefix=" << yesno(cfg.mtp_paged_prefix) << "\n"
              << "  prefix_cache="
              << yesno(cfg.prefix_cache && cfg.continuous_batching) << "\n"
              << "  prefix_cache_max_pages="
              << (std::getenv("QW3_PREFIX_CACHE_MAX_PAGES")
                      ? std::getenv("QW3_PREFIX_CACHE_MAX_PAGES")
                      : "0(unlimited)")
              << "\n"
              << "  prefix_cache_max_entries="
              << (std::getenv("QW3_PREFIX_CACHE_MAX_ENTRIES")
                      ? std::getenv("QW3_PREFIX_CACHE_MAX_ENTRIES")
                      : "64(default)")
              << "\n"
              << "  tool_argument_streaming=canonical-string\n"
              << "  matmul=mmq\n"
              << "  disable_hgemm=1\n"
              << "  default_max_tokens="
              << (cfg.default_max_tokens_set
                      ? std::to_string(cfg.default_generation.max_tokens)
                      : std::string("remaining_context"))
              << "\n"
              << "  enable_thinking_default="
              << yesno(cfg.enable_thinking_default) << "\n"
              << "  preserve_thinking_default="
              << yesno(cfg.preserve_thinking_default) << "\n"
              << "  thinking_budget_default="
              << (cfg.thinking_budget_default > 0
                      ? std::to_string(cfg.thinking_budget_default)
                      : std::string("0(disabled)"))
              << "\n"
              << "  kvmem=" << yesno(engine.kvmem_enabled) << "\n"
              << "  kvmem_block_tokens=" << engine.kvmem_block_tokens << "\n"
              << "  kvmem_budget=" << engine.kvmem_budget << "\n"
              << "  kvmem_prefill_budget="
              << (engine.kvmem_prefill_budget > 0
                      ? engine.kvmem_prefill_budget
                      : engine.kvmem_budget)
              << "\n"
              << "  kvmem_update_mode=" << engine.kvmem_update_mode << "\n"
              << "  kvmem_performance_mode="
              << (kvmem_all_optimizations
                      ? "all-on" : "factorial-ablation")
              << "\n"
              << "  kvmem_opt_stage_out="
              << yesno(engine.kvmem_opt_stage_out) << "\n"
              << "  kvmem_opt_stage_in="
              << yesno(engine.kvmem_opt_stage_in) << "\n"
              << "  kvmem_opt_pack="
              << yesno(engine.kvmem_opt_pack) << "\n"
              << "  kvmem_query_conditioned="
              << yesno(engine.kvmem_query_conditioned) << "\n"
              << "  kvmem_recompute_query="
              << yesno(engine.kvmem_recompute_query) << "\n"
              << "  kvmem_immutable_k="
              << yesno(engine.kvmem_immutable_source_k) << "\n"
              << "  kvmem_raw_k_nvme="
              << yesno(engine.kvmem_raw_k_nvme) << "\n"
              << "  kvmem_query_replay="
              << yesno(cfg.kvmem_query_replay && !cfg.continuous_batching) << "\n"
              << "  kvmem_method=" << engine.kvmem_method << "\n"
              << "  kvmem_retrieval_method=" << engine.kvmem_retrieval_method << "\n"
              << "  kvmem_adaptive_gain_1to2="
              << engine.kvmem_adaptive_gain_1to2 << "\n"
              << "  kvmem_adaptive_gain_2to4="
              << engine.kvmem_adaptive_gain_2to4 << "\n"
              << "  kvmem_index_placement=" << engine.kvmem_index_placement << "\n"
              << "  kvmem_numa_policy=" << engine.kvmem_numa_policy << "\n"
              << "  kvmem_index_staging_mb="
              << engine.kvmem_index_staging_mb << "\n"
              << "  kvmem_adaptive_score_mode="
              << engine.kvmem_adaptive_score_mode << "\n"
              << "  kvmem_semantic_expansion="
              << engine.kvmem_semantic_expansion << "\n"
              << "  kvmem_round_retrieval="
              << yesno(engine.kvmem_semantic_expansion == "round") << "\n"
              << "  kvmem_group_score_reduce="
              << engine.kvmem_group_score_reduce << "\n"
              << "  kvmem_group_length_alpha="
              << engine.kvmem_group_length_alpha << "\n"
              << "  kvmem_sink="
              << kvmem_keep.sink_effective_tokens << " tokens / "
              << kvmem_keep.sink_blocks << " blocks"
              << " (target=" << kvmem_keep.sink_target_tokens
              << ", source=" << keep_source_name(kvmem_keep.sink_source)
              << ")\n"
              << "  kvmem_recent="
              << kvmem_keep.recent_effective_tokens << " tokens / "
              << kvmem_keep.recent_blocks << " blocks"
              << " (target=" << kvmem_keep.recent_target_tokens
              << ", source=" << keep_source_name(kvmem_keep.recent_source)
              << ")\n"
              << "  kvmem_gpu_memory_ratio=" << engine.kvmem_gpu_memory_ratio << "\n"
              << "  kvmem_cpu_tier=" << engine.kvmem_cpu_bytes
              << " bytes (" << bytes_gib_label(engine.kvmem_cpu_bytes) << ")\n"
              << "  kvmem_nvme_tier=" << engine.kvmem_nvme_bytes
              << " bytes (" << bytes_gib_label(engine.kvmem_nvme_bytes) << ")\n"
              << "  kvmem_nvme_dir="
              << (engine.kvmem_nvme_dir.empty()
                      ? std::string("(unset)")
                      : engine.kvmem_nvme_dir)
              << "\n"
              << "  kvmem_prefix_cache="
              << yesno(cfg.kvmem_prefix_cache && engine.kvmem_enabled &&
                       !cfg.continuous_batching)
              << "\n"
              << "  kvmem_archive="
              << (engine.kvmem_archive_dir.empty()
                      ? std::string("(disabled)")
                      : engine.kvmem_archive_dir)
              << "\n"
              << "  kvmem_archive_tokens=" << archive_prefix_tokens << "\n";

    std::cerr << "[qw3-serve] loading model: " << engine.model_path << "\n";
    Engine eng(engine);
    std::unique_ptr<GgufFile> usage_gguf;
    std::unique_ptr<QwenTokenizer> usage_tokenizer_owner;
    if (std::filesystem::is_directory(engine.model_path)) {
        usage_tokenizer_owner = std::make_unique<QwenTokenizer>(engine.model_path);
    } else {
        usage_gguf = std::make_unique<GgufFile>(engine.model_path);
        usage_tokenizer_owner = std::make_unique<QwenTokenizer>(*usage_gguf);
    }
    QwenTokenizer &usage_tokenizer = *usage_tokenizer_owner;
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

    svr.Get(R"(/v1/kvmem/caches/([A-Za-z0-9_.:-]+))",
            [&](const httplib::Request &req, httplib::Response &res) {
        const std::string id = req.matches[1].str();
        std::lock_guard<std::mutex> lk(gen_mu);
        const KvMemLocalCacheInfo info = eng.kvmem_local_cache_info(id);
        if (!info.found) {
            set_error_response(res, 404,
                               "KVMem local cache not found: " + id);
            return;
        }
        res.set_content(
            dump_json(json{{"object", "kvmem.cache"},
                           {"kvmem_cache", kvmem_cache_info_json(info)}}),
            "application/json");
    });

    svr.Delete(R"(/v1/kvmem/caches/([A-Za-z0-9_.:-]+))",
               [&](const httplib::Request &req, httplib::Response &res) {
        const std::string id = req.matches[1].str();
        std::lock_guard<std::mutex> lk(gen_mu);
        if (!eng.erase_kvmem_local_cache(id)) {
            set_error_response(res, 404,
                               "KVMem local cache not found: " + id);
            return;
        }
        res.set_content(
            dump_json(json{{"id", id}, {"status", "evicted"}}),
            "application/json");
    });

    // llama.cpp-compatible tokenizer-count endpoint.  AgentLongBench's
    // canonical worker uses this to size the generation request before it
    // calls /v1/chat/completions.  Returning only count by default avoids
    // serializing a 100K-250K element token-id array for long prompts; the
    // opt-in return_tokens mode exists for tokenizer-parity validation.
    auto handle_tokenize = [&](const httplib::Request &hreq,
                               httplib::Response &res) {
        json req;
        try {
            req = json::parse(hreq.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(
                dump_json(json{{"error", std::string("invalid JSON: ") + e.what()}}),
                "application/json");
            return;
        }
        if (!req.contains("content") || !req["content"].is_string()) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", "missing string content"}}),
                            "application/json");
            return;
        }
        const std::string content = req["content"].get<std::string>();
        const std::vector<int32_t> tokens = usage_tokenizer.encode(content);
        json out{{"count", tokens.size()}};
        if (req.value("return_tokens", false)) out["tokens"] = tokens;
        res.set_content(dump_json(out), "application/json");
    };
    svr.Post("/tokenize", handle_tokenize);
    svr.Post("/v1/tokenize", handle_tokenize);

    // Build GenerationOptions from common OpenAI fields. Sampling defaults to
    // the Qwen3-recommended preset for the request's thinking mode (see below);
    // any field the client sends overrides it.
    auto make_gen = [&](const json &req, size_t prompt_token_count,
                        bool enable_thinking = true) -> GenerationOptions {
        GenerationOptions g = cfg.default_generation;
        // Qwen3-recommended sampling preset per mode, applied only where the
        // user did not pin the value on the CLI or in the request. Thinking:
        // temp 0.6 / top_p 0.95 (the struct + CLI default). Non-thinking:
        // temp 0.7 / top_p 0.8. top_k=20 / min_p=0 are shared, so untouched.
        if (!enable_thinking) {
            if (!cfg.temperature_set) g.temperature = 0.7f;
            if (!cfg.top_p_set) g.top_p = 0.8f;
        }
        const uint64_t occupied = archive_prefix_tokens +
            static_cast<uint64_t>(prompt_token_count);
        const int remaining_ctx = occupied <
                static_cast<uint64_t>(std::max(1, engine.ctx_size))
            ? static_cast<int>(
                  static_cast<uint64_t>(engine.ctx_size) - occupied)
            : 0;
        bool has_max_tokens = false;
        int requested_max_tokens = 0;
        std::string max_tokens_error;
        if (!parse_explicit_max_tokens(req, has_max_tokens,
                                       requested_max_tokens,
                                       max_tokens_error)) {
            throw std::invalid_argument(max_tokens_error);
        }
        if (has_max_tokens) {
            // 0 has an intentional, first-class meaning: execute the complete
            // prefill/state-update path without entering sampling or decode.
            g.max_tokens = requested_max_tokens;
        } else {
            g.max_tokens = cfg.default_max_tokens_set
                ? cfg.default_generation.max_tokens
                : remaining_ctx;
        }
        if (cfg.default_max_tokens_set &&
            cfg.default_generation.max_tokens > 0 &&
            g.max_tokens > cfg.default_generation.max_tokens) {
            std::cerr << "[qw3-serve] capping request max_tokens from "
                      << g.max_tokens << " to "
                      << cfg.default_generation.max_tokens
                      << " (server limit)\n";
            g.max_tokens = cfg.default_generation.max_tokens;
        }
        if (g.max_tokens > remaining_ctx) {
            std::cerr << "[qw3-serve] capping request max_tokens from "
                      << g.max_tokens << " to " << remaining_ctx
                      << " (remaining context)\n";
            g.max_tokens = remaining_ctx;
        }
        // kvmem: a single turn's generation must not exceed the GPU pool's
        // generation reserve (--kvmem-gen-budget). The bounded pool is sized for
        // select_budget + gen_budget, so capping max_tokens at gen_budget keeps
        // step-mode decode (which never stages out) from exhausting the pool.
        if (engine.kvmem_enabled && engine.kvmem_gen_budget > 0 &&
            g.max_tokens > engine.kvmem_gen_budget) {
            std::cerr << "[qw3-serve] capping request max_tokens from "
                      << g.max_tokens << " to " << engine.kvmem_gen_budget
                      << " (kvmem gen budget)\n";
            g.max_tokens = engine.kvmem_gen_budget;
        }
        g.temperature = req.value("temperature", g.temperature);
        g.top_p = req.value("top_p", g.top_p);
        g.top_k = req.value("top_k", g.top_k);
        g.min_p = req.value("min_p", g.min_p);
        g.presence_penalty = req.value("presence_penalty", g.presence_penalty);
        g.repetition_penalty = req.value("repetition_penalty", g.repetition_penalty);
        g.seed = req.value("seed", g.seed);
        g.ignore_eos = req.value("ignore_eos",
                                 req.value("ignore_eos_token", g.ignore_eos));
        g.recover_thinking_eos = req.value("recover_thinking_eos", enable_thinking);
        g.thinking_budget = req.value("thinking_budget", cfg.thinking_budget_default);
        if (g.thinking_budget < 0) g.thinking_budget = 0;

        const bool has_guided_thinking =
            req.contains("kvmem_query_guided_thinking_max_tokens");
        const bool has_guided_query =
            req.contains("kvmem_query_guided_query_max_tokens");
        bool guided_direct = false;
        if (req.contains("kvmem_query_guided_direct")) {
            if (!req["kvmem_query_guided_direct"].is_boolean()) {
                throw std::invalid_argument(
                    "kvmem_query_guided_direct must be a boolean");
            }
            guided_direct =
                req["kvmem_query_guided_direct"].get<bool>();
        }
        if (has_guided_thinking != has_guided_query) {
            throw std::invalid_argument(
                "guided query requires both "
                "kvmem_query_guided_thinking_max_tokens and "
                "kvmem_query_guided_query_max_tokens");
        }
        if (has_guided_thinking) {
            if (!engine.kvmem_enabled || !engine.kvmem_query_conditioned) {
                throw std::invalid_argument(
                    "guided query requires --kvmem and "
                    "--kvmem-query-conditioned");
            }
            uint64_t thinking_max = 0;
            uint64_t query_max = 0;
            if (!parse_bounded_json_u64(
                    req["kvmem_query_guided_thinking_max_tokens"],
                    guided_direct ? 0 : 1,
                    std::numeric_limits<uint32_t>::max(), thinking_max) ||
                !parse_bounded_json_u64(
                    req["kvmem_query_guided_query_max_tokens"], 1,
                    std::numeric_limits<uint32_t>::max(), query_max)) {
                throw std::invalid_argument(
                    "guided-query thinking/query limits must be positive "
                    "uint32 integers");
            }
            g.kvmem_query_guided_thinking_max_tokens =
                static_cast<uint32_t>(thinking_max);
            g.kvmem_query_guided_query_max_tokens =
                static_cast<uint32_t>(query_max);
            g.kvmem_query_guided_direct = guided_direct;
            if (guided_direct && thinking_max != 0) {
                throw std::invalid_argument(
                    "direct guided query requires a zero private-thinking "
                    "limit");
            }
        } else if (guided_direct) {
            throw std::invalid_argument(
                "kvmem_query_guided_direct requires guided-query token limits");
        }

        if (req.contains("kvmem_semantic_budget")) {
            if (!engine.kvmem_enabled) {
                throw std::invalid_argument(
                    "kvmem_semantic_budget requires --kvmem");
            }
            const uint64_t configured_max = static_cast<uint64_t>(
                std::max(0, engine.kvmem_budget));
            uint64_t requested = 0;
            if (!parse_bounded_json_u64(
                    req["kvmem_semantic_budget"], 1, configured_max,
                    requested)) {
                throw std::invalid_argument(
                    "kvmem_semantic_budget must be a positive integer no "
                    "larger than --kvmem-budget (" +
                    std::to_string(configured_max) + ")");
            }
            const uint64_t block_tokens = static_cast<uint64_t>(
                std::max(1, engine.kvmem_block_tokens));
            if (requested % block_tokens != 0) {
                throw std::invalid_argument(
                    "kvmem_semantic_budget must be divisible by "
                    "--kvmem-block-tokens (" +
                    std::to_string(block_tokens) + ")");
            }
            const KvMemKeepAllocation keep =
                resolve_kvmem_keep_allocation(
                    static_cast<uint32_t>(block_tokens),
                    static_cast<uint32_t>(configured_max),
                    engine.kvmem_sink_blocks,
                    engine.kvmem_recent_blocks,
                    engine.kvmem_sink_tokens,
                    engine.kvmem_recent_tokens);
            const uint64_t keep_blocks =
                static_cast<uint64_t>(keep.sink_blocks) +
                keep.recent_blocks;
            if (requested / block_tokens < keep_blocks) {
                throw std::invalid_argument(
                    "kvmem_semantic_budget is smaller than the configured "
                    "sink + recent allocation (" +
                    std::to_string(keep_blocks * block_tokens) +
                    " tokens)");
            }
            g.kvmem_semantic_budget =
                static_cast<uint32_t>(requested);
        }

        // ARCHIVED DeltaNet-state debug entry (2026-07-23).
        //
        // The frozen LongMemEval-M error-10 experiments did not show a stable,
        // attributable gain:
        //   * replace accumulated recurrent state with a state rebuilt from the
        //     selected 224K source tokens: 1/10 with the inline grader, but 6/10
        //     when the same outputs were rejudged by DeepSeek V4 Pro;
        //   * retain the accumulated state and replay the same selected tokens as
        //     additional DeltaNet updates: 5/10 with the inline grader.
        // Exporting one ~229K-token state also cost about 91 seconds and wrote a
        // ~150.5 MiB artifact per sample. Because the score depended strongly on
        // judge route and neither construction isolated a reliable improvement,
        // export/import/capture/seed are no longer supported request controls.
        // Keep rejecting the retired names instead of silently ignoring an old
        // experiment script and accidentally reporting a normal KVMem result.
        static constexpr const char *kArchivedRebuiltStateFields[] = {
            "kvmem_rebuilt_state_export",
            "kvmem_rebuilt_state_import",
            "kvmem_rebuilt_state_capture",
            "kvmem_rebuilt_state_seed",
        };
        for (const char *field : kArchivedRebuiltStateFields) {
            if (req.contains(field)) {
                throw std::invalid_argument(
                    std::string(field) +
                    " is archived and disabled; see KVMI-012");
            }
        }

#if 0  // Archived DeltaNet recurrent-state debug parser; see note above.
        auto parse_rebuilt_state_key = [&](const char *field,
                                           std::string &out) {
            if (!req.contains(field)) return;
            if (!req[field].is_string()) {
                throw std::invalid_argument(std::string(field) +
                                            " must be a string key");
            }
            out = req[field].get<std::string>();
            const bool valid = !out.empty() && out.size() <= 128 &&
                std::all_of(out.begin(), out.end(), [](unsigned char c) {
                    return std::isalnum(c) || c == '-' || c == '_' || c == '.';
                });
            if (!valid) {
                throw std::invalid_argument(
                    std::string(field) +
                    " must contain 1..128 characters from [A-Za-z0-9_.-]");
            }
        };
        parse_rebuilt_state_key("kvmem_rebuilt_state_export",
                                g.kvmem_rebuilt_state_export_key);
        parse_rebuilt_state_key("kvmem_rebuilt_state_import",
                                g.kvmem_rebuilt_state_import_key);
        parse_rebuilt_state_key("kvmem_rebuilt_state_capture",
                                g.kvmem_rebuilt_state_capture_key);
        parse_rebuilt_state_key("kvmem_rebuilt_state_seed",
                                g.kvmem_rebuilt_state_seed_key);
        if (!g.kvmem_rebuilt_state_export_key.empty() &&
            !g.kvmem_rebuilt_state_import_key.empty()) {
            throw std::invalid_argument(
                "kvmem_rebuilt_state_export and kvmem_rebuilt_state_import "
                "are mutually exclusive");
        }
        if (!g.kvmem_rebuilt_state_capture_key.empty() &&
            (!g.kvmem_rebuilt_state_export_key.empty() ||
             !g.kvmem_rebuilt_state_import_key.empty() ||
             !g.kvmem_rebuilt_state_seed_key.empty())) {
            throw std::invalid_argument(
                "kvmem_rebuilt_state_capture cannot be combined with other "
                "rebuilt-state operations");
        }
        if (!g.kvmem_rebuilt_state_seed_key.empty() &&
            g.kvmem_rebuilt_state_export_key.empty()) {
            throw std::invalid_argument(
                "kvmem_rebuilt_state_seed requires kvmem_rebuilt_state_export");
        }
        if (!g.kvmem_rebuilt_state_export_key.empty() && g.max_tokens != 0) {
            throw std::invalid_argument(
                "kvmem_rebuilt_state_export requires max_tokens=0");
        }
#endif
        return g;
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request &hreq,
                                         httplib::Response &res) {
        const auto server_request_start = std::chrono::steady_clock::now();
        json req;
        try {
            req = json::parse(hreq.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", std::string("invalid JSON: ") + e.what()}}),
                            "application/json");
            return;
        }
        const auto server_json_end = std::chrono::steady_clock::now();
        if (!req.contains("messages") || !req["messages"].is_array()) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", "missing messages[]"}}),
                            "application/json");
            return;
        }
        bool explicit_max_tokens = false;
        int requested_max_tokens = 0;
        std::string max_tokens_error;
        if (!parse_explicit_max_tokens(req, explicit_max_tokens,
                                       requested_max_tokens,
                                       max_tokens_error)) {
            set_error_response(res, 400, max_tokens_error);
            return;
        }
        const bool prefill_only = explicit_max_tokens
            ? requested_max_tokens == 0
            : (cfg.default_max_tokens_set &&
               cfg.default_generation.max_tokens == 0);
        KvMemReselectMode kvmem_reselect_mode = KvMemReselectMode::Auto;
        if (req.contains("kvmem_reselect")) {
            if (!req["kvmem_reselect"].is_string()) {
                set_error_response(res, 400,
                                   "kvmem_reselect must be auto|force|off");
                return;
            }
            const std::string mode = req["kvmem_reselect"].get<std::string>();
            if (mode == "auto") {
                kvmem_reselect_mode = KvMemReselectMode::Auto;
            } else if (mode == "force") {
                kvmem_reselect_mode = KvMemReselectMode::Force;
            } else if (mode == "off") {
                kvmem_reselect_mode = KvMemReselectMode::Off;
            } else {
                set_error_response(res, 400,
                                   "kvmem_reselect must be auto|force|off");
                return;
            }
        }
        KvMemPrefillWindowMode kvmem_prefill_window_mode =
            KvMemPrefillWindowMode::Pressure;
        if (req.contains("kvmem_prefill_window")) {
            if (!req["kvmem_prefill_window"].is_string()) {
                set_error_response(
                    res, 400,
                    "kvmem_prefill_window must be "
                    "pressure|keep_selected|semantic_chunk");
                return;
            }
            const std::string mode =
                req["kvmem_prefill_window"].get<std::string>();
            if (mode == "pressure") {
                kvmem_prefill_window_mode =
                    KvMemPrefillWindowMode::Pressure;
            } else if (mode == "keep_selected") {
                kvmem_prefill_window_mode =
                    KvMemPrefillWindowMode::KeepSelected;
            } else if (mode == "semantic_chunk") {
                kvmem_prefill_window_mode =
                    KvMemPrefillWindowMode::SemanticChunk;
            } else {
                set_error_response(
                    res, 400,
                    "kvmem_prefill_window must be "
                    "pressure|keep_selected|semantic_chunk");
                return;
            }
        }
        uint32_t kvmem_prefill_semantic_start_tokens = 0;
        uint32_t kvmem_prefill_semantic_query_tokens = 0;
        auto parse_nonnegative_u32 = [&](const char *name,
                                         uint32_t &value) -> bool {
            if (!req.contains(name)) return true;
            if (!req[name].is_number_integer()) {
                set_error_response(
                    res, 400, std::string(name) +
                                  " must be a non-negative integer");
                return false;
            }
            const int64_t parsed = req[name].get<int64_t>();
            if (parsed < 0 ||
                static_cast<uint64_t>(parsed) >
                    std::numeric_limits<uint32_t>::max()) {
                set_error_response(
                    res, 400, std::string(name) +
                                  " must fit in uint32");
                return false;
            }
            value = static_cast<uint32_t>(parsed);
            return true;
        };
        if (!parse_nonnegative_u32(
                "kvmem_prefill_semantic_start_tokens",
                kvmem_prefill_semantic_start_tokens) ||
            !parse_nonnegative_u32(
                "kvmem_prefill_semantic_query_tokens",
                kvmem_prefill_semantic_query_tokens)) {
            return;
        }
        if (kvmem_prefill_window_mode !=
                KvMemPrefillWindowMode::SemanticChunk &&
            (kvmem_prefill_semantic_start_tokens != 0 ||
             kvmem_prefill_semantic_query_tokens != 0)) {
            set_error_response(
                res, 400,
                "kvmem_prefill_semantic_* requires "
                "kvmem_prefill_window=semantic_chunk");
            return;
        }
        bool kvmem_session_request = false;
        bool kvmem_session_reset = false;
        std::string kvmem_session_id;
        std::string kvmem_session_op;
        if (req.contains("kvmem_session_id") ||
            req.contains("kvmem_session_op")) {
            if (!req.contains("kvmem_session_id") ||
                !req["kvmem_session_id"].is_string() ||
                req["kvmem_session_id"].get<std::string>().empty()) {
                set_error_response(
                    res, 400,
                    "kvmem_session_id must be a non-empty string");
                return;
            }
            if (!req.contains("kvmem_session_op") ||
                !req["kvmem_session_op"].is_string()) {
                set_error_response(
                    res, 400,
                    "kvmem_session_op must be start|append|finish");
                return;
            }
            kvmem_session_id = req["kvmem_session_id"].get<std::string>();
            kvmem_session_op = req["kvmem_session_op"].get<std::string>();
            if (kvmem_session_op != "start" &&
                kvmem_session_op != "append" &&
                kvmem_session_op != "finish") {
                set_error_response(
                    res, 400,
                    "kvmem_session_op must be start|append|finish");
                return;
            }
            if (!engine.kvmem_enabled) {
                set_error_response(
                    res, 400,
                    "kvmem_session_* requires --kvmem");
                return;
            }
            kvmem_session_request = true;
            kvmem_session_reset = kvmem_session_op == "start";
            if (kvmem_session_op != "finish" && !prefill_only) {
                set_error_response(
                    res, 400,
                    "kvmem session start/append requires max_tokens=0");
                return;
            }
            if (kvmem_session_reset &&
                kvmem_prefill_window_mode ==
                    KvMemPrefillWindowMode::KeepSelected) {
                set_error_response(
                    res, 400,
                    "kvmem_prefill_window=keep_selected requires an active "
                    "session selection");
                return;
            }
        }
        bool kvmem_cache_request = false;
        std::string kvmem_cache_id;
        std::string kvmem_cache_operation;
        KvMemLocalCacheMode kvmem_cache_mode =
            KvMemLocalCacheMode::None;
        uint64_t kvmem_cache_expected_version = 0;
        bool kvmem_cache_expected_version_set = false;
        uint64_t kvmem_cache_ttl_seconds = 0;
        if (req.contains("kvmem_cache")) {
            if (!engine.kvmem_enabled) {
                set_error_response(res, 400,
                                   "kvmem_cache requires --kvmem");
                return;
            }
            if (kvmem_session_request) {
                set_error_response(
                    res, 400,
                    "kvmem_cache and kvmem_session_* are mutually exclusive");
                return;
            }
            const json &cache = req["kvmem_cache"];
            if (!cache.is_object()) {
                set_error_response(res, 400,
                                   "kvmem_cache must be an object");
                return;
            }
            const bool save = cache.contains("save");
            const bool load = cache.contains("load");
            if (save == load) {
                set_error_response(
                    res, 400,
                    "kvmem_cache requires exactly one save or load object");
                return;
            }
            const json &operation = cache[save ? "save" : "load"];
            if (!operation.is_object() || !operation.contains("id") ||
                !operation["id"].is_string()) {
                set_error_response(
                    res, 400,
                    "kvmem_cache save/load requires a string id");
                return;
            }
            kvmem_cache_id = operation["id"].get<std::string>();
            if (!valid_kvmem_cache_id(kvmem_cache_id)) {
                set_error_response(
                    res, 400,
                    "kvmem_cache id must be 1..128 characters using only "
                    "letters, digits, '.', '_', '-', or ':'");
                return;
            }
            if (save) {
                kvmem_cache_operation = "save";
                if (!prefill_only) {
                    set_error_response(
                        res, 400,
                        "kvmem_cache save currently requires max_tokens=0");
                    return;
                }
                const std::string scope = operation.value("scope", "local");
                const std::string when =
                    operation.value("when", "after_request");
                if (scope != "local" || when != "after_request") {
                    set_error_response(
                        res, 400,
                        "kvmem_cache save supports only scope=local and "
                        "when=after_request");
                    return;
                }
                if (operation.contains("ttl_seconds")) {
                    if (!parse_bounded_json_u64(
                            operation["ttl_seconds"], 0, 31536000,
                            kvmem_cache_ttl_seconds)) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache ttl_seconds must be in [0,31536000]");
                        return;
                    }
                }
            } else {
                kvmem_cache_operation = "load";
                const std::string mode = operation.value("mode", "frozen");
                if (mode == "frozen") {
                    kvmem_cache_mode = KvMemLocalCacheMode::Frozen;
                } else if (mode == "append") {
                    kvmem_cache_mode = KvMemLocalCacheMode::Append;
                } else {
                    set_error_response(
                        res, 400,
                        "kvmem_cache load mode must be frozen|append");
                    return;
                }
                if (operation.contains("required") &&
                    (!operation["required"].is_boolean() ||
                     !operation["required"].get<bool>())) {
                    set_error_response(
                        res, 400,
                        "process-local cache loads require required=true; "
                        "missing caches never silently trigger full prefill");
                    return;
                }
                if (operation.contains("expected_version")) {
                    if (!parse_bounded_json_u64(
                            operation["expected_version"], 1,
                            std::numeric_limits<uint64_t>::max(),
                            kvmem_cache_expected_version)) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache expected_version must be a positive "
                            "integer");
                        return;
                    }
                    kvmem_cache_expected_version_set = true;
                }
                if (kvmem_cache_mode == KvMemLocalCacheMode::Append) {
                    if (!prefill_only) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache append currently requires "
                            "max_tokens=0");
                        return;
                    }
                    if (!kvmem_cache_expected_version_set) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache append requires expected_version");
                        return;
                    }
                }
            }
            kvmem_cache_request = true;
        }
        const bool has_kvmem_query =
            req.contains("kvmem_query_span") ||
            req.contains("kvmem_query_message_range");
        if (kvmem_reselect_mode == KvMemReselectMode::Force &&
            !has_kvmem_query) {
            set_error_response(
                res, 400,
                "kvmem_reselect=force requires kvmem_query_span or "
                "kvmem_query_message_range");
            return;
        }
        if (kvmem_reselect_mode == KvMemReselectMode::Force &&
            !engine.kvmem_query_conditioned) {
            set_error_response(
                res, 400,
                "kvmem_reselect=force requires --kvmem-query-conditioned");
            return;
        }
        if (kvmem_reselect_mode == KvMemReselectMode::Off &&
            has_kvmem_query) {
            set_error_response(
                res, 400,
                "KVMem query metadata cannot be used with "
                "kvmem_reselect=off");
            return;
        }
        if (req.contains("kvmem_query_span") &&
            req.contains("kvmem_query_message_range")) {
            set_error_response(
                res, 400,
                "kvmem_query_span and kvmem_query_message_range are "
                "mutually exclusive");
            return;
        }
        const bool enable_thinking =
            req.value("enable_thinking", cfg.enable_thinking_default);
        bool preserve_thinking = cfg.preserve_thinking_default;
        std::string preserve_thinking_error;
        if (!parse_preserve_thinking(
                req, cfg.preserve_thinking_default, preserve_thinking,
                preserve_thinking_error)) {
            set_error_response(res, 400, preserve_thinking_error);
            return;
        }
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
        if (tool_request) {
            std::cerr << "[qw3-serve] incoming tools="
                      << tools_debug_summary(*tools);
            if (!forced_tool_name.empty()) {
                std::cerr << " forced=" << forced_tool_name;
            }
            std::cerr << "\n";
        }
        bool transcript_replay = false;
        if (req.contains("kvmem_transcript_replay")) {
            if (!req["kvmem_transcript_replay"].is_boolean()) {
                set_error_response(res, 400,
                                   "kvmem_transcript_replay must be a boolean");
                return;
            }
            transcript_replay = req["kvmem_transcript_replay"].get<bool>();
        }
        if (transcript_replay &&
            (!engine.kvmem_enabled || !engine.kvmem_query_conditioned)) {
            set_error_response(
                res, 400,
                "kvmem_transcript_replay requires --kvmem and "
                "--kvmem-query-conditioned");
            return;
        }
        if (transcript_replay && engine.kvmem_retrieval_method != "mean-k") {
            set_error_response(
                res, 400,
                "kvmem_transcript_replay currently requires mean-k retrieval");
            return;
        }
        if (kvmem_cache_request && transcript_replay) {
            set_error_response(
                res, 400,
                "kvmem_cache cannot be combined with "
                "kvmem_transcript_replay");
            return;
        }
        std::vector<RenderedMessageSpan> rendered_message_spans;
        const bool explicit_retrieval_groups =
            req.contains("kvmem_retrieval_group_spans");
        const bool auto_message_groups =
            engine.kvmem_semantic_expansion == "message" &&
            !explicit_retrieval_groups;
        const bool map_retrieval_groups =
            explicit_retrieval_groups || auto_message_groups;
        const auto server_render_start = std::chrono::steady_clock::now();
        const std::string prompt = render_messages(
            req["messages"], tools, enable_thinking, preserve_thinking,
            forced_tool_name,
            (transcript_replay || map_retrieval_groups)
                ? &rendered_message_spans : nullptr,
            /*add_generation_prompt=*/!prefill_only);
        const auto server_render_end = std::chrono::steady_clock::now();
        std::vector<int32_t> prompt_token_ids =
            usage_tokenizer.encode(prompt);
        const auto server_tokenize_end = std::chrono::steady_clock::now();
        size_t prompt_token_count = prompt_token_ids.size();
        if (archive_prefix_tokens + prompt_token_count >=
            static_cast<uint64_t>(std::max(1, engine.ctx_size))) {
            set_error_response(
                res,
                413,
                "archive prefix plus prompt exceeds KV context: archive_tokens=" +
                    std::to_string(archive_prefix_tokens) +
                    " prompt_tokens=" +
                    std::to_string(prompt_token_count) +
                    " ctx=" + std::to_string(engine.ctx_size));
            return;
        }
        GenerationOptions g;
        try {
            g = make_gen(req, prompt_token_count, enable_thinking);
        } catch (const std::invalid_argument &e) {
            set_error_response(res, 400, e.what());
            return;
        }
        g.raw_prompt = true; // prompt is already chat-framed
        g.thinking_open = enable_thinking; // budget only runs while <think> is open
        g.kvmem_reselect_mode = kvmem_reselect_mode;
        g.kvmem_prefill_window_mode = kvmem_prefill_window_mode;
        g.kvmem_prefill_semantic_start_tokens =
            kvmem_prefill_semantic_start_tokens;
        g.kvmem_prefill_semantic_query_tokens =
            kvmem_prefill_semantic_query_tokens;
        g.kvmem_session_id = kvmem_session_id;
        if (kvmem_cache_request) {
            if (kvmem_cache_operation == "save") {
                g.kvmem_cache_save_id = kvmem_cache_id;
                g.kvmem_cache_ttl_seconds = kvmem_cache_ttl_seconds;
            } else {
                g.kvmem_cache_load_id = kvmem_cache_id;
                g.kvmem_cache_load_mode = kvmem_cache_mode;
                g.kvmem_cache_expected_version =
                    kvmem_cache_expected_version;
                g.kvmem_cache_expected_version_set =
                    kvmem_cache_expected_version_set;
            }
        }

        // Optional semantic retrieval groups. Flattened benchmarks supply byte
        // spans inside one user message; ordinary Chat requests in message mode
        // derive complete rendered-message spans automatically. All boundaries
        // are mapped to prompt tokens in one O(prompt_tokens) decode pass.
        if (map_retrieval_groups) {
            if (!engine.kvmem_enabled ||
                engine.kvmem_semantic_expansion == "none") {
                set_error_response(
                    res, 400,
                    "kvmem_retrieval_group_spans requires --kvmem and "
                    "--kvmem-semantic-expansion round|message");
                return;
            }

            struct RequestedGroup {
                size_t begin = 0;
                size_t end = 0;
            };
            std::vector<RequestedGroup> requested;
            if (explicit_retrieval_groups) {
                const json &spans =
                    req["kvmem_retrieval_group_spans"];
                if (!spans.is_array() || spans.empty() ||
                    spans.size() > 65535) {
                    set_error_response(
                        res, 400,
                        "kvmem_retrieval_group_spans must be an array of "
                        "1..65535 spans");
                    return;
                }
                requested.reserve(spans.size());
                size_t previous_abs_end = 0;
                for (const json &span : spans) {
                    if (!span.is_object() ||
                        !span.contains("message_index") ||
                        !span.contains("content_start") ||
                        !span.contains("content_end") ||
                        !span["message_index"].is_number_integer() ||
                        !span["content_start"].is_number_integer() ||
                        !span["content_end"].is_number_integer()) {
                        set_error_response(
                            res, 400,
                            "each kvmem_retrieval_group_spans entry requires "
                            "integer message_index, content_start, and "
                            "content_end");
                        return;
                    }
                    const int64_t message_index =
                        span["message_index"].get<int64_t>();
                    const int64_t content_start =
                        span["content_start"].get<int64_t>();
                    const int64_t content_end =
                        span["content_end"].get<int64_t>();
                    if (message_index < 0 ||
                        message_index >=
                            static_cast<int64_t>(req["messages"].size()) ||
                        content_start < 0 || content_end <= content_start) {
                        set_error_response(
                            res, 400,
                            "kvmem retrieval group content span is invalid");
                        return;
                    }
                    const RenderedMessageSpan *rendered = nullptr;
                    for (const RenderedMessageSpan &candidate :
                         rendered_message_spans) {
                        if (candidate.message_index ==
                            static_cast<size_t>(message_index)) {
                            rendered = &candidate;
                            break;
                        }
                    }
                    const json &message =
                        req["messages"][static_cast<size_t>(message_index)];
                    if (!rendered || rendered->role != "user" ||
                        !message.is_object() ||
                        !message.contains("content") ||
                        !message["content"].is_string()) {
                        set_error_response(
                            res, 400,
                            "explicit retrieval groups currently require "
                            "spans inside string-content user messages");
                        return;
                    }
                    const std::string raw_content =
                        message["content"].get<std::string>();
                    size_t trim_begin = 0;
                    while (trim_begin < raw_content.size() &&
                           (raw_content[trim_begin] == ' ' ||
                            raw_content[trim_begin] == '\n' ||
                            raw_content[trim_begin] == '\r' ||
                            raw_content[trim_begin] == '\t')) {
                        ++trim_begin;
                    }
                    size_t trim_end = raw_content.size();
                    while (trim_end > trim_begin &&
                           (raw_content[trim_end - 1] == ' ' ||
                            raw_content[trim_end - 1] == '\n' ||
                            raw_content[trim_end - 1] == '\r' ||
                            raw_content[trim_end - 1] == '\t')) {
                        --trim_end;
                    }
                    if (content_start <
                            static_cast<int64_t>(trim_begin) ||
                        content_end >
                            static_cast<int64_t>(trim_end)) {
                        set_error_response(
                            res, 400,
                            "kvmem retrieval group span falls in whitespace "
                            "removed by the chat renderer");
                        return;
                    }
                    const size_t abs_begin =
                        rendered->content_begin +
                        static_cast<size_t>(content_start) - trim_begin;
                    const size_t abs_end =
                        rendered->content_begin +
                        static_cast<size_t>(content_end) - trim_begin;
                    if (abs_end > rendered->content_end ||
                        (!requested.empty() &&
                         abs_begin < previous_abs_end)) {
                        set_error_response(
                            res, 400,
                            "kvmem retrieval group spans must be sorted and "
                            "non-overlapping in rendered prompt order");
                        return;
                    }
                    requested.push_back({abs_begin, abs_end});
                    previous_abs_end = abs_end;
                }
            } else {
                const size_t final_query =
                    last_query_index_for_template(req["messages"]);
                requested.reserve(rendered_message_spans.size());
                for (const RenderedMessageSpan &span :
                     rendered_message_spans) {
                    // The final user query is pinned/replayed independently and
                    // must not become a historical retrieval candidate.
                    if (span.message_index >= final_query ||
                        span.segment_end <= span.segment_begin) {
                        continue;
                    }
                    requested.push_back(
                        {span.segment_begin, span.segment_end});
                }
                if (requested.empty()) {
                    set_error_response(
                        res, 400,
                        "message semantic expansion found no historical "
                        "messages; flattened prompts must provide "
                        "kvmem_retrieval_group_spans");
                    return;
                }
            }

            std::vector<size_t> token_bytes;
            token_bytes.reserve(prompt_token_ids.size() + 1);
            token_bytes.push_back(0);
            size_t decoded_bytes = 0;
            bool decode_matches = true;
            for (int32_t token : prompt_token_ids) {
                const std::string piece =
                    usage_tokenizer.decode_one(token);
                if (decoded_bytes + piece.size() > prompt.size() ||
                    prompt.compare(decoded_bytes, piece.size(), piece) != 0) {
                    decode_matches = false;
                    break;
                }
                decoded_bytes += piece.size();
                token_bytes.push_back(decoded_bytes);
            }
            if (!decode_matches || decoded_bytes != prompt.size()) {
                set_error_response(
                    res, 500,
                    "could not map semantic group byte spans through tokenizer "
                    "pieces exactly");
                return;
            }

            uint32_t rounded_boundaries = 0;
            for (const RequestedGroup &group : requested) {
                const auto begin_it = std::upper_bound(
                    token_bytes.begin(), token_bytes.end(),
                    group.begin);
                const size_t token_begin =
                    begin_it == token_bytes.begin()
                        ? 0
                        : static_cast<size_t>(
                              begin_it - token_bytes.begin() - 1);
                const auto end_it = std::lower_bound(
                    token_bytes.begin(), token_bytes.end(),
                    group.end);
                const size_t token_end = static_cast<size_t>(
                    end_it - token_bytes.begin());
                if (token_begin >= token_end ||
                    token_end > prompt_token_ids.size()) {
                    set_error_response(
                        res, 400,
                        "kvmem retrieval group maps to an empty token span");
                    return;
                }
                rounded_boundaries +=
                    token_bytes[token_begin] == group.begin ? 0u : 1u;
                rounded_boundaries +=
                    token_bytes[token_end] == group.end ? 0u : 1u;
                const uint32_t begin_u32 =
                    static_cast<uint32_t>(token_begin);
                const uint32_t end_u32 =
                    static_cast<uint32_t>(token_end);
                // Outward rounding can make two adjacent byte groups overlap
                // by one BPE token. Preserve that bounded overlap for scoring;
                // only the original byte spans are required to be disjoint.
                g.kvmem_retrieval_group_spans.push_back(
                    GenerationOptions::KvMemRetrievalGroupSpan{
                        begin_u32, end_u32});
            }
            std::cerr
                << "[qw3-serve] KVMem semantic retrieval mode="
                << engine.kvmem_semantic_expansion << " groups="
                << g.kvmem_retrieval_group_spans.size()
                << " rounded_boundaries=" << rounded_boundaries
                << " token_span=["
                << g.kvmem_retrieval_group_spans.front().begin
                << ","
                << g.kvmem_retrieval_group_spans.back().end
                << ") prompt_tokens=" << prompt_token_count << "\n";
        }

        // One-shot selected-context cache refresh ablation. This never uses
        // trace dumps or cross-request artifacts: the native backend performs
        // the long prefill, freezes the final selection, and rebuilds that
        // compact context inside the same request. Keep it explicitly gated so
        // production clients cannot opt into an expensive representation test.
        if (req.contains("kvmem_inline_refresh")) {
            if (!engine.kvmem_enabled) {
                set_error_response(
                    res, 400, "kvmem_inline_refresh requires --kvmem");
                return;
            }
            if (!env_flag_enabled("QW3_KVMEM_ENABLE_INLINE_REFRESH")) {
                set_error_response(
                    res, 403,
                    "kvmem_inline_refresh is disabled; set "
                    "QW3_KVMEM_ENABLE_INLINE_REFRESH=1 for controlled "
                    "diagnostics");
                return;
            }
            if (!req["kvmem_inline_refresh"].is_string()) {
                set_error_response(
                    res, 400,
                    "kvmem_inline_refresh must be \"kv_only\" or "
                    "\"kv_and_state\"");
                return;
            }
            const std::string mode =
                req["kvmem_inline_refresh"].get<std::string>();
            if (mode == "kv_only") {
                g.kvmem_inline_refresh = KvMemInlineRefreshMode::KvOnly;
            } else if (mode == "kv_and_state") {
                g.kvmem_inline_refresh =
                    KvMemInlineRefreshMode::KvAndState;
            } else {
                set_error_response(
                    res, 400,
                    "kvmem_inline_refresh must be \"kv_only\" or "
                    "\"kv_and_state\"");
                return;
            }
            if (transcript_replay || !kvmem_session_id.empty()) {
                set_error_response(
                    res, 400,
                    "kvmem_inline_refresh requires a standalone one-shot "
                    "request");
                return;
            }
            if (kvmem_cache_request) {
                set_error_response(
                    res, 400,
                    "kvmem_inline_refresh cannot be combined with "
                    "kvmem_cache");
                return;
            }
            std::cerr << "[qw3-serve] KVMem INLINE REFRESH enabled mode="
                      << mode << " prompt_tokens=" << prompt_token_count
                      << "\n";
        }

        // Diagnostics-only oracle selection. The benchmark caller supplies
        // exact rendered-prompt token spans after verifying tokenizer parity.
        // Keeping this behind an explicit environment gate prevents a normal
        // API client from accidentally turning gold provenance into a product
        // feature or contaminating production evaluations.
        if (req.contains("kvmem_oracle_token_spans")) {
            if (!engine.kvmem_enabled) {
                set_error_response(
                    res, 400,
                    "kvmem_oracle_token_spans requires --kvmem");
                return;
            }
            if (transcript_replay) {
                set_error_response(
                    res, 400,
                    "kvmem_oracle_token_spans is a final-query-only control "
                    "and cannot be combined with kvmem_transcript_replay");
                return;
            }
            if (kvmem_cache_request) {
                set_error_response(
                    res, 400,
                    "kvmem_oracle_token_spans cannot be combined with "
                    "kvmem_cache");
                return;
            }
            if (!env_flag_enabled("QW3_KVMEM_ENABLE_ORACLE")) {
                set_error_response(
                    res, 403,
                    "kvmem_oracle_token_spans is disabled; set "
                    "QW3_KVMEM_ENABLE_ORACLE=1 for controlled diagnostics");
                return;
            }
            const json &spans = req["kvmem_oracle_token_spans"];
            if (!spans.is_array() || spans.empty() || spans.size() > 64) {
                set_error_response(
                    res, 400,
                    "kvmem_oracle_token_spans must be an array of 1..64 spans");
                return;
            }
            for (const json &span : spans) {
                if (!span.is_object() || !span.contains("begin") ||
                    !span.contains("end") ||
                    !span["begin"].is_number_integer() ||
                    !span["end"].is_number_integer()) {
                    set_error_response(
                        res, 400,
                        "each kvmem_oracle_token_spans entry requires integer "
                        "begin and end");
                    return;
                }
                const int64_t begin = span["begin"].get<int64_t>();
                const int64_t end = span["end"].get<int64_t>();
                if (begin < 0 || end <= begin ||
                    end > static_cast<int64_t>(prompt_token_count)) {
                    set_error_response(
                        res, 400,
                        "kvmem_oracle_token_spans entry is outside the "
                        "rendered prompt");
                    return;
                }
                g.kvmem_oracle_token_spans.push_back(
                    GenerationOptions::KvMemOracleTokenSpan{
                        static_cast<uint32_t>(begin),
                        static_cast<uint32_t>(end)});
            }
            if (req.contains("kvmem_oracle_only")) {
                if (!req["kvmem_oracle_only"].is_boolean()) {
                    set_error_response(
                        res, 400, "kvmem_oracle_only must be a boolean");
                    return;
                }
                g.kvmem_oracle_only =
                    req["kvmem_oracle_only"].get<bool>();
            }
            std::cerr << "[qw3-serve] KVMem ORACLE enabled spans="
                      << g.kvmem_oracle_token_spans.size()
                      << " only=" << (g.kvmem_oracle_only ? 1 : 0)
                      << " prompt_tokens=" << prompt_token_count << "\n";
        } else if (req.contains("kvmem_oracle_only")) {
            set_error_response(
                res, 400,
                "kvmem_oracle_only requires kvmem_oracle_token_spans");
            return;
        }

        // Query-conditioned KVMem: mark the final user message's token span so
        // the executor selects the decode window by multi-token mean relevance
        // to the question instead of recency. Computed by render-twice-and-diff
        // (robust to chat template + BPE): re-render with the final user
        // message's content emptied; the common-prefix-len + length-delta then
        // brackets exactly the question content tokens (role markers / assistant
        // suffix fall in the shared prefix/suffix). Only when the server was
        // launched with --kvmem-query-conditioned; otherwise the span stays empty
        // and selection is byte-identical to the single-token / recency path.
        if (engine.kvmem_query_conditioned &&
            kvmem_reselect_mode != KvMemReselectMode::Off) {
            const json &msgs = req["messages"];
            bool explicit_span = false;

            // Experimental whole-round query. Unlike kvmem_query_span, which
            // marks bytes inside one message, this half-open message range can
            // cover a role-preserving user/assistant/tool round. Re-rendering
            // with those messages removed and taking the token LCP/LCS keeps
            // the mapping exact across chat-template control tokens. The
            // ordinary API path never sends this field.
            if (req.contains("kvmem_query_message_range")) {
                const json &range = req["kvmem_query_message_range"];
                if (!range.is_object() ||
                    !range.contains("message_begin") ||
                    !range.contains("message_end") ||
                    !range["message_begin"].is_number_integer() ||
                    !range["message_end"].is_number_integer()) {
                    set_error_response(
                        res, 400,
                        "kvmem_query_message_range requires integer "
                        "message_begin and message_end");
                    return;
                }
                const int64_t message_begin =
                    range["message_begin"].get<int64_t>();
                const int64_t message_end =
                    range["message_end"].get<int64_t>();
                if (message_begin < 0 || message_end <= message_begin ||
                    message_end > static_cast<int64_t>(msgs.size())) {
                    set_error_response(
                        res, 400,
                        "kvmem_query_message_range is outside messages[]");
                    return;
                }
                if (transcript_replay) {
                    set_error_response(
                        res, 400,
                        "kvmem_query_message_range cannot be combined with "
                        "kvmem_transcript_replay");
                    return;
                }

                json msgs_empty = json::array();
                for (size_t i = 0; i < msgs.size(); ++i) {
                    if (i < static_cast<size_t>(message_begin) ||
                        i >= static_cast<size_t>(message_end)) {
                        msgs_empty.push_back(msgs[i]);
                    }
                }
                const std::string empty_prompt = render_messages(
                    msgs_empty, tools, enable_thinking, preserve_thinking,
                    forced_tool_name,
                    /*message_spans=*/nullptr,
                    /*add_generation_prompt=*/!prefill_only);
                const std::vector<int32_t> tok_empty =
                    usage_tokenizer.encode(empty_prompt);
                size_t qb = 0;
                const size_t prefix_max =
                    std::min(prompt_token_ids.size(), tok_empty.size());
                while (qb < prefix_max &&
                       prompt_token_ids[qb] == tok_empty[qb]) {
                    ++qb;
                }
                size_t suffix = 0;
                while (suffix < prompt_token_ids.size() - qb &&
                       suffix < tok_empty.size() - qb &&
                       prompt_token_ids[prompt_token_ids.size() - 1 - suffix] ==
                           tok_empty[tok_empty.size() - 1 - suffix]) {
                    ++suffix;
                }
                const size_t qe = prompt_token_ids.size() - suffix;
                if (qe <= qb) {
                    set_error_response(
                        res, 400,
                        "kvmem_query_message_range maps to an empty token "
                        "span");
                    return;
                }
                g.kvmem_query_begin = static_cast<uint32_t>(qb);
                g.kvmem_query_end = static_cast<uint32_t>(qe);
                explicit_span = true;
                std::cerr
                    << "[qw3-serve] kvmem explicit query message range ["
                    << message_begin << "," << message_end << ") -> tokens ["
                    << qb << "," << qe << ") of "
                    << prompt_token_ids.size() << "\n";
            }

            // Experimental role-preserving transcript replay. Render-time byte
            // spans are converted to exact token spans in one linear pass over
            // message segments. Segment boundaries are Qwen special-token
            // boundaries; verify compositional tokenization against the full
            // prompt before accepting the mapping. Only user messages that
            // ARRIVE after select_budget + gen_budget is already full become
            // replay/reselection events.
            if (transcript_replay) {
                std::vector<int32_t> rebuilt;
                rebuilt.reserve(prompt_token_ids.size());
                size_t byte_cursor = 0;
                size_t token_cursor = 0;
                const uint64_t pressure_threshold =
                    static_cast<uint64_t>(std::max(
                        0, engine.kvmem_prefill_budget > 0
                               ? engine.kvmem_prefill_budget
                               : engine.kvmem_budget)) +
                    static_cast<uint64_t>(std::max(0, engine.kvmem_gen_budget));
                for (const RenderedMessageSpan &span : rendered_message_spans) {
                    const std::string gap = prompt.substr(
                        byte_cursor, span.segment_begin - byte_cursor);
                    const std::vector<int32_t> gap_tokens =
                        usage_tokenizer.encode(gap);
                    rebuilt.insert(rebuilt.end(), gap_tokens.begin(), gap_tokens.end());
                    token_cursor += gap_tokens.size();

                    const std::string segment = prompt.substr(
                        span.segment_begin, span.segment_end - span.segment_begin);
                    const std::vector<int32_t> segment_tokens =
                        usage_tokenizer.encode(segment);
                    const size_t segment_token_begin = token_cursor;
                    if (span.message_index < msgs.size() &&
                        msgs[span.message_index].is_object() &&
                        msgs[span.message_index].contains(
                            "kvmem_session_start")) {
                        const json &marker =
                            msgs[span.message_index]["kvmem_session_start"];
                        if (!marker.is_boolean()) {
                            set_error_response(
                                res, 400,
                                "message kvmem_session_start must be a boolean");
                            return;
                        }
                        if (marker.get<bool>()) {
                            g.kvmem_replay_session_starts.push_back(
                                static_cast<uint32_t>(segment_token_begin));
                        }
                    }
                    if (span.role == "user") {
                        std::string empty_segment = segment;
                        const size_t local_content_begin =
                            span.content_begin - span.segment_begin;
                        const size_t local_content_end =
                            span.content_end - span.segment_begin;
                        empty_segment.erase(
                            local_content_begin,
                            local_content_end - local_content_begin);
                        const std::vector<int32_t> empty_tokens =
                            usage_tokenizer.encode(empty_segment);
                        size_t local_qb = 0;
                        const size_t prefix_max =
                            std::min(segment_tokens.size(), empty_tokens.size());
                        while (local_qb < prefix_max &&
                               segment_tokens[local_qb] == empty_tokens[local_qb]) {
                            ++local_qb;
                        }
                        size_t suffix = 0;
                        while (suffix < segment_tokens.size() - local_qb &&
                               suffix < empty_tokens.size() - local_qb &&
                               segment_tokens[segment_tokens.size() - 1 - suffix] ==
                                   empty_tokens[empty_tokens.size() - 1 - suffix]) {
                            ++suffix;
                        }
                        const size_t local_qe = segment_tokens.size() - suffix;
                        if (local_qe > local_qb &&
                            segment_token_begin >= pressure_threshold) {
                            g.kvmem_replay_query_spans.push_back(
                                GenerationOptions::KvMemReplayQuerySpan{
                                    static_cast<uint32_t>(segment_token_begin +
                                                          local_qb),
                                    static_cast<uint32_t>(segment_token_begin +
                                                          local_qe)});
                        }
                    }
                    rebuilt.insert(rebuilt.end(), segment_tokens.begin(),
                                   segment_tokens.end());
                    token_cursor += segment_tokens.size();
                    byte_cursor = span.segment_end;
                }
                const std::vector<int32_t> tail_tokens =
                    usage_tokenizer.encode(prompt.substr(byte_cursor));
                rebuilt.insert(rebuilt.end(), tail_tokens.begin(), tail_tokens.end());
                if (rebuilt != prompt_token_ids) {
                    set_error_response(
                        res, 500,
                        "kvmem_transcript_replay token-span mapping was not "
                        "compositional at message boundaries");
                    return;
                }
                if (g.kvmem_replay_query_spans.empty()) {
                    set_error_response(
                        res, 400,
                        "kvmem_transcript_replay found no user query arriving "
                        "after the KVMem pressure threshold");
                    return;
                }
                const auto &last = g.kvmem_replay_query_spans.back();
                g.kvmem_query_begin = last.begin;
                g.kvmem_query_end = last.end;
                explicit_span = true;
                std::cerr << "[qw3-serve] kvmem transcript replay: events="
                          << g.kvmem_replay_query_spans.size()
                          << " sessions="
                          << g.kvmem_replay_session_starts.size()
                          << " threshold=" << pressure_threshold
                          << " first=[" << g.kvmem_replay_query_spans.front().begin
                          << "," << g.kvmem_replay_query_spans.front().end << ")"
                          << " last=[" << last.begin << "," << last.end << ")"
                          << " prompt_tokens=" << prompt_token_count << "\n";
            }

            // Optional diagnostics-only sample key. Restrict it to a compact,
            // JSON-safe alphabet because the executor's score dump is written
            // directly with fprintf. It is never consumed by retrieval logic.
            if (req.contains("kvmem_trace_tag")) {
                if (!req["kvmem_trace_tag"].is_string()) {
                    set_error_response(res, 400,
                                       "kvmem_trace_tag must be a string");
                    return;
                }
                const std::string tag = req["kvmem_trace_tag"].get<std::string>();
                const bool tag_ok = !tag.empty() && tag.size() <= 128 &&
                    std::all_of(tag.begin(), tag.end(), [](unsigned char c) {
                        return std::isalnum(c) || c == '-' || c == '_' ||
                               c == '.' || c == ':';
                    });
                if (!tag_ok) {
                    set_error_response(
                        res, 400,
                        "kvmem_trace_tag must be 1..128 characters from "
                        "[A-Za-z0-9_.:-]");
                    return;
                }
                g.kvmem_trace_tag = tag;
            }

            // Optional diagnostics-only span for the benchmark history. It is
            // mapped from UTF-8 content bytes to exact rendered-prompt tokens by
            // the same remove-and-diff procedure used for the query span. The
            // resulting bounds are exported with selected KVMem blocks, enabling
            // offline projection into the RAG history coordinate system.
            if (req.contains("kvmem_context_span")) {
                const json &span = req["kvmem_context_span"];
                if (!span.is_object() || !span.contains("message_index") ||
                    !span.contains("content_start") ||
                    !span.contains("content_end") ||
                    !span["message_index"].is_number_integer() ||
                    !span["content_start"].is_number_integer() ||
                    !span["content_end"].is_number_integer()) {
                    set_error_response(
                        res, 400,
                        "kvmem_context_span requires integer message_index, "
                        "content_start, and content_end");
                    return;
                }
                const int64_t message_index = span["message_index"].get<int64_t>();
                const int64_t content_start = span["content_start"].get<int64_t>();
                const int64_t content_end = span["content_end"].get<int64_t>();
                if (message_index < 0 ||
                    message_index >= static_cast<int64_t>(msgs.size()) ||
                    !msgs[static_cast<size_t>(message_index)].is_object() ||
                    !msgs[static_cast<size_t>(message_index)].contains("content") ||
                    !msgs[static_cast<size_t>(message_index)]["content"].is_string()) {
                    set_error_response(res, 400,
                                       "kvmem_context_span message_index does not "
                                       "reference a string-content message");
                    return;
                }
                const std::string content =
                    msgs[static_cast<size_t>(message_index)]["content"]
                        .get<std::string>();
                if (content_start < 0 || content_end <= content_start ||
                    content_end > static_cast<int64_t>(content.size())) {
                    set_error_response(res, 400,
                                       "kvmem_context_span content offsets are "
                                       "outside the message content");
                    return;
                }
                json msgs_empty = msgs;
                std::string content_empty = content;
                content_empty.erase(static_cast<size_t>(content_start),
                                    static_cast<size_t>(content_end - content_start));
                msgs_empty[static_cast<size_t>(message_index)]["content"] =
                    std::move(content_empty);
                const std::string empty_prompt = render_messages(
                    msgs_empty, tools, enable_thinking, preserve_thinking,
                    forced_tool_name,
                    /*message_spans=*/nullptr,
                    /*add_generation_prompt=*/!prefill_only);
                const std::vector<int32_t> tok_full = usage_tokenizer.encode(prompt);
                const std::vector<int32_t> tok_empty =
                    usage_tokenizer.encode(empty_prompt);
                size_t cb = 0;
                const size_t prefix_max = std::min(tok_full.size(), tok_empty.size());
                while (cb < prefix_max && tok_full[cb] == tok_empty[cb]) ++cb;
                size_t suffix = 0;
                while (suffix < tok_full.size() - cb &&
                       suffix < tok_empty.size() - cb &&
                       tok_full[tok_full.size() - 1 - suffix] ==
                           tok_empty[tok_empty.size() - 1 - suffix]) {
                    ++suffix;
                }
                const size_t ce = tok_full.size() - suffix;
                if (ce <= cb) {
                    set_error_response(res, 400,
                                       "kvmem_context_span maps to an empty token span");
                    return;
                }
                g.kvmem_context_begin = static_cast<uint32_t>(cb);
                g.kvmem_context_end = static_cast<uint32_t>(ce);
                if (std::getenv("QW3_KVMEM_TRACE")) {
                    std::cerr << "[qw3-serve] kvmem context span [" << cb
                              << "," << ce << ") of " << tok_full.size()
                              << " prompt tokens, tag="
                              << (g.kvmem_trace_tag.empty() ? "(none)"
                                                           : g.kvmem_trace_tag)
                              << "\n";
                }
            }

            // Optional API extension for benchmarks whose exact canonical
            // prompt is one long user message containing both history and the
            // final question.  Offsets are UTF-8 byte offsets into that
            // message's content.  Removing only the marked substring and
            // comparing the two chat renderings preserves the exact model
            // prompt while still identifying the true retrieval query.
            if (req.contains("kvmem_query_span")) {
                const json &span = req["kvmem_query_span"];
                if (!span.is_object() || !span.contains("message_index") ||
                    !span.contains("content_start") ||
                    !span.contains("content_end") ||
                    !span["message_index"].is_number_integer() ||
                    !span["content_start"].is_number_integer() ||
                    !span["content_end"].is_number_integer()) {
                    set_error_response(
                        res, 400,
                        "kvmem_query_span requires integer message_index, "
                        "content_start, and content_end");
                    return;
                }
                const int64_t message_index = span["message_index"].get<int64_t>();
                const int64_t content_start = span["content_start"].get<int64_t>();
                const int64_t content_end = span["content_end"].get<int64_t>();
                if (message_index < 0 ||
                    message_index >= static_cast<int64_t>(msgs.size()) ||
                    !msgs[static_cast<size_t>(message_index)].is_object() ||
                    !msgs[static_cast<size_t>(message_index)].contains("content") ||
                    !msgs[static_cast<size_t>(message_index)]["content"].is_string()) {
                    set_error_response(res, 400,
                                       "kvmem_query_span message_index does not "
                                       "reference a string-content message");
                    return;
                }
                const std::string content =
                    msgs[static_cast<size_t>(message_index)]["content"]
                        .get<std::string>();
                if (content_start < 0 || content_end <= content_start ||
                    content_end > static_cast<int64_t>(content.size())) {
                    set_error_response(res, 400,
                                       "kvmem_query_span content offsets are "
                                       "outside the message content");
                    return;
                }

                json msgs_empty = msgs;
                std::string content_empty = content;
                content_empty.erase(static_cast<size_t>(content_start),
                                    static_cast<size_t>(content_end - content_start));
                msgs_empty[static_cast<size_t>(message_index)]["content"] =
                    std::move(content_empty);
                const std::string empty_prompt = render_messages(
                    msgs_empty, tools, enable_thinking, preserve_thinking,
                    forced_tool_name,
                    /*message_spans=*/nullptr,
                    /*add_generation_prompt=*/!prefill_only);
                const std::vector<int32_t> tok_full = usage_tokenizer.encode(prompt);
                const std::vector<int32_t> tok_empty =
                    usage_tokenizer.encode(empty_prompt);
                size_t qb = 0;
                const size_t prefix_max = std::min(tok_full.size(), tok_empty.size());
                while (qb < prefix_max && tok_full[qb] == tok_empty[qb]) ++qb;
                size_t suffix = 0;
                while (suffix < tok_full.size() - qb &&
                       suffix < tok_empty.size() - qb &&
                       tok_full[tok_full.size() - 1 - suffix] ==
                           tok_empty[tok_empty.size() - 1 - suffix]) {
                    ++suffix;
                }
                const size_t qe = tok_full.size() - suffix;
                if (qe > qb) {
                    g.kvmem_query_begin = static_cast<uint32_t>(qb);
                    g.kvmem_query_end = static_cast<uint32_t>(qe);
                    if (transcript_replay) {
                        // The transcript mapper initially records the complete
                        // arriving user-message content. An explicit subspan on
                        // that final message is the actual retrieval query, so
                        // keep the answer-producing replay event and the global
                        // query coordinates identical. Intermediate historical
                        // user events retain their complete-message spans.
                        if (g.kvmem_replay_query_spans.empty()) {
                            set_error_response(
                                res, 400,
                                "explicit transcript query span has no final "
                                "replay event");
                            return;
                        }
                        g.kvmem_replay_query_spans.back().begin =
                            static_cast<uint32_t>(qb);
                        g.kvmem_replay_query_spans.back().end =
                            static_cast<uint32_t>(qe);
                    }
                    explicit_span = true;
                    std::cerr << "[qw3-serve] kvmem explicit query span ["
                              << qb << "," << qe << ") of " << tok_full.size()
                              << " prompt tokens, message=" << message_index
                              << " content_bytes=[" << content_start << ","
                              << content_end << ")\n";
                    if (std::getenv("QW3_KVMEM_TRACE")) {
                        const std::vector<int32_t> slice(
                            tok_full.begin() + static_cast<long>(qb),
                            tok_full.begin() + static_cast<long>(qe));
                        std::string txt = usage_tokenizer.decode(slice);
                        if (txt.size() > 200) txt = txt.substr(0, 200) + "...";
                        std::cerr << "[qw3-serve] kvmem query span text: \""
                                  << txt << "\"\n";
                    }
                }
            }

            // Existing default behavior is deliberately unchanged when the
            // explicit field is absent: use the complete final user message.
            if (!explicit_span && !req.contains("kvmem_query_span")) {
                const size_t lqi = last_query_index_for_template(msgs);
                if (lqi < msgs.size() && msgs[lqi].is_object() &&
                    msgs[lqi].value("role", "") == "user") {
                    json msgs_empty = msgs;
                    msgs_empty[lqi]["content"] = "";
                    const std::string empty_prompt = render_messages(
                        msgs_empty, tools, enable_thinking,
                        preserve_thinking, forced_tool_name,
                        /*message_spans=*/nullptr,
                        /*add_generation_prompt=*/!prefill_only);
                    const std::vector<int32_t> tok_full =
                        usage_tokenizer.encode(prompt);
                    const std::vector<int32_t> tok_empty =
                        usage_tokenizer.encode(empty_prompt);
                    if (tok_full.size() > tok_empty.size()) {
                        size_t qb = 0;
                        const size_t maxn = tok_empty.size();
                        while (qb < maxn && tok_full[qb] == tok_empty[qb]) ++qb;
                        const size_t qe =
                            qb + (tok_full.size() - tok_empty.size());
                        g.kvmem_query_begin = static_cast<uint32_t>(qb);
                        g.kvmem_query_end = static_cast<uint32_t>(qe);
                        std::cerr << "[qw3-serve] kvmem query span [" << qb
                                  << "," << qe << ") of " << tok_full.size()
                                  << " prompt tokens\n";
                        if (std::getenv("QW3_KVMEM_TRACE")) {
                            const std::vector<int32_t> slice(
                                tok_full.begin() + static_cast<long>(qb),
                                tok_full.begin() + static_cast<long>(qe));
                            std::string txt = usage_tokenizer.decode(slice);
                            if (txt.size() > 200)
                                txt = txt.substr(0, 200) + "...";
                            std::cerr
                                << "[qw3-serve] kvmem query span text: \""
                                << txt << "\"\n";
                        }
                    }
                }
            }
        }

        // Controlled round-alignment experiment. The request's byte spans are
        // first mapped through the canonical, unmodified prompt above so query
        // and group coordinates remain auditable. We then insert an ordinary
        // newline token before the first round (to align its absolute start)
        // and after every real round (to align the next start). Group score
        // spans continue to cover only real source tokens; fixed-block
        // materialization naturally includes the trailing newline fillers but
        // can no longer include tokens from the adjacent round.
        //
        // This is intentionally an exact-token override rather than appending
        // newline text and re-tokenizing: BPE could merge textual whitespace
        // and silently produce a different filler count. It is standalone and
        // opt-in because the filler tokens are visible to the model (there is
        // no internal causal-padding mask) and therefore form an accuracy
        // ablation, not an API formatting default.
        if (req.contains("kvmem_round_padding")) {
            if (!req["kvmem_round_padding"].is_number_integer()) {
                set_error_response(
                    res, 400, "kvmem_round_padding must be an integer");
                return;
            }
            const int64_t requested_alignment =
                req["kvmem_round_padding"].get<int64_t>();
            if (requested_alignment <= 0 ||
                requested_alignment >
                    static_cast<int64_t>(
                        std::numeric_limits<uint32_t>::max())) {
                set_error_response(
                    res, 400, "kvmem_round_padding must be positive");
                return;
            }
            if (!map_retrieval_groups ||
                g.kvmem_retrieval_group_spans.empty()) {
                set_error_response(
                    res, 400,
                    "kvmem_round_padding requires "
                    "kvmem_retrieval_group_spans");
                return;
            }
            if (requested_alignment != engine.kvmem_block_tokens) {
                set_error_response(
                    res, 400,
                    "kvmem_round_padding must equal --kvmem-block-tokens "
                    "for exclusive round blocks");
                return;
            }
            if (transcript_replay || kvmem_session_request ||
                kvmem_cache_request ||
                req.contains("kvmem_oracle_token_spans") ||
                req.contains("kvmem_inline_refresh")) {
                set_error_response(
                    res, 400,
                    "kvmem_round_padding currently supports only standalone "
                    "round-retrieval requests without replay/oracle/refresh");
                return;
            }

            const std::vector<int32_t> newline_tokens =
                usage_tokenizer.encode("\n");
            if (newline_tokens.size() != 1 ||
                usage_tokenizer.decode_one(newline_tokens.front()) != "\n") {
                set_error_response(
                    res, 500,
                    "tokenizer has no exact one-token newline for "
                    "kvmem_round_padding");
                return;
            }
            const int32_t newline_token = newline_tokens.front();
            const uint32_t alignment =
                static_cast<uint32_t>(requested_alignment);
            const auto original_groups =
                g.kvmem_retrieval_group_spans;
            for (size_t i = 1; i < original_groups.size(); ++i) {
                if (original_groups[i - 1].end !=
                    original_groups[i].begin) {
                    set_error_response(
                        res, 400,
                        "kvmem_round_padding requires exact, gap-free token "
                        "round boundaries (no BPE overlap/rounding)");
                    return;
                }
            }

            struct PaddingInsertion {
                uint32_t original_boundary = 0;
                uint32_t count = 0;
            };
            std::vector<PaddingInsertion> insertions;
            insertions.reserve(original_groups.size() + 1);
            std::vector<int32_t> aligned_tokens;
            const uint64_t worst_extra =
                static_cast<uint64_t>(original_groups.size() + 1) *
                (alignment - 1u);
            if (static_cast<uint64_t>(prompt_token_ids.size()) +
                    worst_extra >
                std::numeric_limits<size_t>::max()) {
                set_error_response(
                    res, 413, "round padding size overflows host indexing");
                return;
            }
            aligned_tokens.reserve(
                prompt_token_ids.size() +
                static_cast<size_t>(worst_extra));

            auto append_source = [&](uint32_t begin, uint32_t end) {
                aligned_tokens.insert(
                    aligned_tokens.end(),
                    prompt_token_ids.begin() +
                        static_cast<std::ptrdiff_t>(begin),
                    prompt_token_ids.begin() +
                        static_cast<std::ptrdiff_t>(end));
            };
            auto append_padding = [&](uint32_t original_boundary,
                                      uint32_t count) {
                if (count == 0) return;
                aligned_tokens.insert(
                    aligned_tokens.end(), count, newline_token);
                insertions.push_back(
                    PaddingInsertion{original_boundary, count});
            };

            const uint32_t first_begin = original_groups.front().begin;
            const uint32_t last_end = original_groups.back().end;
            if (last_end > prompt_token_ids.size()) {
                set_error_response(
                    res, 500,
                    "round padding group exceeds the tokenized prompt");
                return;
            }
            append_source(0, first_begin);
            const uint32_t prefix_padding = static_cast<uint32_t>(
                (alignment - aligned_tokens.size() % alignment) %
                alignment);
            append_padding(first_begin, prefix_padding);

            std::vector<GenerationOptions::KvMemRetrievalGroupSpan>
                aligned_groups;
            aligned_groups.reserve(original_groups.size());
            uint32_t cursor = first_begin;
            uint64_t round_padding = 0;
            uint32_t min_padding = alignment;
            uint32_t max_padding = 0;
            for (const auto &group : original_groups) {
                if (group.begin != cursor || group.end <= group.begin ||
                    group.end > prompt_token_ids.size()) {
                    set_error_response(
                        res, 500,
                        "round padding received inconsistent token groups");
                    return;
                }
                const uint32_t aligned_begin =
                    static_cast<uint32_t>(aligned_tokens.size());
                if (aligned_begin % alignment != 0) {
                    set_error_response(
                        res, 500,
                        "round padding failed to align a group start");
                    return;
                }
                append_source(group.begin, group.end);
                const uint32_t aligned_end =
                    static_cast<uint32_t>(aligned_tokens.size());
                aligned_groups.push_back(
                    GenerationOptions::KvMemRetrievalGroupSpan{
                        aligned_begin, aligned_end});
                const uint32_t pad = static_cast<uint32_t>(
                    (alignment - aligned_tokens.size() % alignment) %
                    alignment);
                append_padding(group.end, pad);
                round_padding += pad;
                min_padding = std::min(min_padding, pad);
                max_padding = std::max(max_padding, pad);
                cursor = group.end;
            }
            append_source(last_end,
                          static_cast<uint32_t>(prompt_token_ids.size()));

            auto map_boundary_after_padding =
                [&](uint32_t original) -> uint32_t {
                uint64_t mapped = original;
                for (const PaddingInsertion &insertion : insertions) {
                    if (insertion.original_boundary > original) break;
                    mapped += insertion.count;
                }
                if (mapped > std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error(
                        "round padding mapped token position overflows u32");
                }
                return static_cast<uint32_t>(mapped);
            };
            if (g.kvmem_query_end > g.kvmem_query_begin) {
                g.kvmem_query_begin =
                    map_boundary_after_padding(g.kvmem_query_begin);
                g.kvmem_query_end =
                    map_boundary_after_padding(g.kvmem_query_end);
            }
            if (g.kvmem_context_end > g.kvmem_context_begin) {
                g.kvmem_context_begin =
                    map_boundary_after_padding(g.kvmem_context_begin);
                g.kvmem_context_end =
                    map_boundary_after_padding(g.kvmem_context_end);
            }
            g.kvmem_retrieval_group_spans =
                std::move(aligned_groups);
            prompt_token_ids = std::move(aligned_tokens);
            prompt_token_count = prompt_token_ids.size();
            if (archive_prefix_tokens + prompt_token_count >=
                static_cast<uint64_t>(std::max(1, engine.ctx_size))) {
                set_error_response(
                    res, 413,
                    "archive prefix plus round-padded prompt exceeds KV context: "
                    "archive_tokens=" +
                        std::to_string(archive_prefix_tokens) +
                        " "
                    "prompt_tokens=" +
                        std::to_string(prompt_token_count) +
                        " ctx=" + std::to_string(engine.ctx_size));
                return;
            }
            const int remaining_ctx = std::max(
                1, engine.ctx_size -
                       static_cast<int>(archive_prefix_tokens) -
                       static_cast<int>(prompt_token_count));
            if (g.max_tokens > remaining_ctx) {
                std::cerr
                    << "[qw3-serve] capping request max_tokens from "
                    << g.max_tokens << " to " << remaining_ctx
                    << " after round padding\n";
                g.max_tokens = remaining_ctx;
            }
            g.prompt_token_ids_override.assign(
                prompt_token_ids.begin(), prompt_token_ids.end());
            std::cerr
                << "[qw3-serve] KVMem round padding alignment="
                << alignment << " newline_token=" << newline_token
                << " groups=" << original_groups.size()
                << " prefix_padding=" << prefix_padding
                << " round_padding=" << round_padding
                << " per_round_min=" << min_padding
                << " per_round_max=" << max_padding
                << " total_added="
                << (prefix_padding + round_padding)
                << " prompt_tokens=" << prompt_token_count
                << " query_span=[" << g.kvmem_query_begin << ","
                << g.kvmem_query_end << ")\n";
        }

        g.continuous_batching =
            !kvmem_session_request &&
            !kvmem_cache_request &&
            serve_continuous_batching_enabled() &&
            serve_continuous_batch_request_supported(g);
        const std::string route = kvmem_cache_request
            ? ("cache-" + kvmem_cache_operation)
            : (kvmem_session_request
                   ? ("session-" + kvmem_session_op)
                   : (g.continuous_batching ? "continuous" : "plain"));
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
        const auto server_preprocess_end = std::chrono::steady_clock::now();
        const auto t0 = server_preprocess_end;
        auto log_server_accounting =
            [rid, server_request_start, server_json_end,
             server_render_start, server_render_end, server_tokenize_end,
             server_preprocess_end](std::chrono::steady_clock::time_point
                                       engine_start,
                                   std::chrono::steady_clock::time_point
                                       engine_end,
                                   std::chrono::steady_clock::time_point
                                       response_end,
                                   bool streaming) {
                auto ms = [](auto begin, auto end) {
                    return std::chrono::duration<double, std::milli>(
                               end - begin)
                        .count();
                };
                const double total_ms =
                    ms(server_request_start, response_end);
                const double preprocess_ms =
                    ms(server_request_start, server_preprocess_end);
                const double queue_ms =
                    ms(server_preprocess_end, engine_start);
                const double engine_ms = ms(engine_start, engine_end);
                const double response_ms = ms(engine_end, response_end);
                const double sum_ms =
                    preprocess_ms + queue_ms + engine_ms + response_ms;
                const double json_ms =
                    ms(server_request_start, server_json_end);
                const double validate_ms =
                    ms(server_json_end, server_render_start);
                const double render_ms =
                    ms(server_render_start, server_render_end);
                const double tokenize_ms =
                    ms(server_render_end, server_tokenize_end);
                const double span_setup_ms =
                    ms(server_tokenize_end, server_preprocess_end);
                const double preprocess_sum_ms =
                    json_ms + validate_ms + render_ms + tokenize_ms +
                    span_setup_ms;
                // Preserve additive request accounting in serialized logs;
                // millisecond precision alone can make the displayed
                // components differ by a rounding microsecond.
                std::cerr << std::fixed << std::setprecision(6)
                          << "[qw3-server-accounting]"
                          << " rid=" << rid
                          << " stream=" << (streaming ? 1 : 0)
                          << " total_ms=" << total_ms
                          << " preprocess_ms=" << preprocess_ms
                          << " queue_ms=" << queue_ms
                          << " engine_ms=" << engine_ms
                          << " response_ms=" << response_ms
                          << " sum_ms=" << sum_ms
                          << " error_ms=" << (total_ms - sum_ms)
                          << " pre_json_ms=" << json_ms
                          << " pre_validate_ms=" << validate_ms
                          << " pre_render_ms=" << render_ms
                          << " pre_tokenize_ms=" << tokenize_ms
                          << " pre_span_setup_ms=" << span_setup_ms
                          << " pre_sum_ms=" << preprocess_sum_ms
                          << " pre_error_ms="
                          << (preprocess_ms - preprocess_sum_ms)
                          << "\n";
            };

        // The streaming content provider outlives this handler scope, so the
        // raw `tools` pointer into `req` would dangle. Capture a by-value copy
        // for schema-driven argument coercion inside the stream callback.
        const json tools_schema = tools ? *tools : json();

        if (stream) {
            res.set_header("Cache-Control", "no-cache");
            res.set_header("X-Accel-Buffering", "no");
            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, g, stops, id, created, rid, t0,
                 server_request_start, enable_thinking,
                 tool_request, forced_tool_request, tools_schema,
                 stream_include_usage, prompt_token_count, route,
                 fallback_reason, kvmem_session_request,
                 kvmem_session_reset,
                 kvmem_cache_request, kvmem_cache_id,
                 log_server_accounting](size_t, httplib::DataSink &sink) {
                    std::unique_lock<std::mutex> gen_lk(gen_mu, std::defer_lock);
                    if (!g.continuous_batching) gen_lk.lock();
                    std::string acc;
                    std::string utf8_pending;
                    ReasoningStreamSplitter reasoning_splitter(enable_thinking);
                    size_t completion_tokens = 0;
                    bool stopped = false;
                    bool client_closed = false;
                    auto engine_start = std::chrono::steady_clock::now();
                    auto engine_end = engine_start;
                    ServerTtftTracker ttft(server_request_start);
                    auto last_stream_write = std::chrono::steady_clock::now();
                    auto send_raw = [&](const std::string &s) {
                        if (client_closed) return false;
                        if (sink.is_writable && !sink.is_writable()) {
                            client_closed = true;
                            stopped = true;
                            return false;
                        }
                        if (!sink.write(s.data(), s.size())) {
                            client_closed = true;
                            stopped = true;
                            return false;
                        }
                        last_stream_write = std::chrono::steady_clock::now();
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
                        const bool sent = send_raw(s);
                        if (sent && visible_stream_delta(delta)) {
                            ttft.observe_visible_output();
                        }
                        return sent;
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
                        if (kvmem_cache_request) {
                            const KvMemLocalCacheInfo info =
                                eng.kvmem_local_cache_info(kvmem_cache_id);
                            if (info.found) {
                                done["kvmem_cache"] =
                                    kvmem_cache_info_json(info);
                            }
                        }
                        done["timing"] =
                            ttft.timing_json(std::chrono::steady_clock::now());
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
                        auto generate_request =
                            [&](const CancellableTokenCallback &callback) {
                            engine_start = std::chrono::steady_clock::now();
                            ttft.start_engine(engine_start);
                            auto tracked_callback =
                                [&](const std::string &piece) {
                                    ttft.observe_model_piece(piece);
                                    return callback(piece);
                                };
                            if (kvmem_session_request) {
                                eng.generate_session_stream(
                                    prompt, g,
                                    [&](const std::string &piece) {
                                        (void)tracked_callback(piece);
                                    },
                                    kvmem_session_reset);
                            } else {
                                eng.generate_stream_cancellable(
                                    prompt, g, tracked_callback);
                            }
                            engine_end = std::chrono::steady_clock::now();
                        };
                        send_role();
                        if (enable_thinking && g.max_tokens > 0) {
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
                            // The model may stream natural-language reasoning
                            // before a <tool_call> block (the Hermes prompt
                            // explicitly allows this). Once a canonical call for
                            // a string-only schema appears, transcode its XML
                            // parameters into standard OpenAI argument deltas.
                            // Recovery syntaxes stay on the full-buffer parser.
                            bool buffering_tool = false;
                            bool streamed_content = false;
                            bool tool_delta_committed = false;
                            std::string content_pending;
                            std::string forced_prefix;
                            const json *stream_tools =
                                tools_schema.is_array() ? &tools_schema : nullptr;
                            IncrementalToolCallStream incremental(stream_tools);
                            auto last_tool_delta =
                                std::chrono::steady_clock::now();
                            size_t next_progress_tokens = 1024;
                            constexpr size_t kArgumentFlushBytes = 1024;
                            constexpr size_t kToolCommitBytes = 8192;
                            constexpr auto kArgumentFlushInterval =
                                std::chrono::milliseconds(500);
                            constexpr auto kToolHeartbeatInterval =
                                std::chrono::seconds(2);

                            auto flush_incremental = [&](bool force) {
                                if (!tool_delta_committed ||
                                    !incremental.streaming() || client_closed) {
                                    return;
                                }
                                if (incremental.start_pending()) {
                                    send_delta(incremental.take_start_delta());
                                    last_tool_delta =
                                        std::chrono::steady_clock::now();
                                }
                                const auto now = std::chrono::steady_clock::now();
                                if (incremental.pending_size() > 0 &&
                                    (force ||
                                     incremental.pending_size() >=
                                         kArgumentFlushBytes ||
                                     now - last_tool_delta >=
                                         kArgumentFlushInterval)) {
                                    send_delta(incremental.take_arguments_delta());
                                    last_tool_delta = now;
                                }
                            };
                            auto service_tool_stream = [&]() {
                                if (!tool_delta_committed &&
                                    incremental.argument_size() >=
                                        kToolCommitBytes &&
                                    incremental.ready_to_commit()) {
                                    tool_delta_committed = true;
                                    std::cerr << "[qw3-serve] #" << rid
                                              << " tool_stream_commit bytes="
                                              << incremental.argument_size()
                                              << "\n";
                                }
                                flush_incremental(false);
                                const auto now = std::chrono::steady_clock::now();
                                if (!client_closed &&
                                    now - last_stream_write >=
                                        kToolHeartbeatInterval) {
                                    // Empty OpenAI deltas are protocol-safe and
                                    // keep read-idle timers alive even when the
                                    // full tool call must remain buffered.
                                    send_delta(json::object());
                                }
                                if (completion_tokens >= next_progress_tokens) {
                                    const double elapsed =
                                        std::chrono::duration<double>(
                                            now - t0).count();
                                    std::cerr << "[qw3-serve] #" << rid
                                              << " tool_buffer_progress tokens="
                                              << completion_tokens
                                              << " chars=" << acc.size()
                                              << " elapsed=" << elapsed << "s"
                                              << " incremental="
                                              << (incremental.streaming()
                                                      ? "true" : "false")
                                              << "\n";
                                    do {
                                        next_progress_tokens += 1024;
                                    } while (completion_tokens >=
                                             next_progress_tokens);
                                }
                            };

                            generate_request([&](const std::string &piece) {
                                if (stopped || client_closed) return false;
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
                                if (buffering_tool) {
                                    if (!emit.empty()) incremental.feed(emit);
                                } else if (!emit.empty()) {
                                    content_pending += emit;
                                    bool marker_found = false;
                                    const size_t safe =
                                        tool_call_safe_emit_len(
                                            content_pending, marker_found);
                                    if (safe > 0) {
                                        const std::string prefix =
                                            content_pending.substr(0, safe);
                                        if (forced_tool_request) {
                                            forced_prefix += prefix;
                                        } else {
                                            emit_text(prefix);
                                            streamed_content = true;
                                        }
                                        content_pending.erase(0, safe);
                                    }
                                    if (marker_found) {
                                        if (forced_tool_request &&
                                            !forced_prefix.empty()) {
                                            emit_text(forced_prefix);
                                            forced_prefix.clear();
                                            streamed_content = true;
                                        }
                                        buffering_tool = true;
                                        // content_pending begins with the marker
                                        // and may already include part of the
                                        // function/first parameter.
                                        incremental.feed(content_pending);
                                        content_pending.clear();
                                    }
                                }
                                if (incremental.fatal()) {
                                    stopped = true;
                                }
                                if (buffering_tool &&
                                    !incremental.fatal()) {
                                    service_tool_stream();
                                }
                                return !stopped && !client_closed;
                            });

                            if (client_closed) {
                                const auto request_end =
                                    std::chrono::steady_clock::now();
                                ttft.log(rid, route, true, request_end);
                                log_server_accounting(
                                    engine_start, engine_end,
                                    request_end, true);
                                std::cerr << "[qw3-serve] #" << rid
                                          << " chat(stream tools) chars="
                                          << acc.size()
                                          << " completion_tokens="
                                          << completion_tokens
                                          << " prompt_tokens="
                                          << prompt_token_count
                                          << " route=" << route
                                          << " buffered_tool="
                                          << (buffering_tool ? "true" : "false")
                                          << " client_closed=true\n";
                                return true;
                            }

                            if (buffering_tool) {
                                const bool hit_max_tokens =
                                    g.max_tokens > 0 &&
                                    completion_tokens >=
                                        static_cast<size_t>(g.max_tokens);
                                std::string closure_suffix;
                                if (!incremental.fatal()) {
                                    incremental.finish(
                                        tool_delta_committed &&
                                            !hit_max_tokens,
                                        closure_suffix);
                                }
                                if (incremental.fatal()) {
                                    if (!tool_delta_committed) {
                                        if (hit_max_tokens) {
                                            throw std::runtime_error(
                                                "tool_call_parse_error: tool "
                                                "call was truncated at "
                                                "max_tokens");
                                        }
                                        throw std::runtime_error(
                                            tool_call_parse_error_message(
                                                incremental.error()));
                                    }
                                    throw std::runtime_error(
                                        "tool_call_parse_error: committed "
                                        "incremental stream failed: " +
                                        incremental.error());
                                }

                                std::string final_utf8_pending;
                                std::string text = take_complete_utf8(
                                    final_utf8_pending, acc);
                                text += flush_utf8_pending(
                                    final_utf8_pending, false);
                                const std::string framed =
                                    (enable_thinking
                                         ? ("<think>\n" + text)
                                         : text) +
                                    closure_suffix;
                                ToolParseResult parsed;
                                if (!closure_suffix.empty()) {
                                    parsed.intent_detected = true;
                                    const json repaired_call =
                                        incremental.finalized_call();
                                    std::string validation_error;
                                    if (repaired_call.is_null() ||
                                        !validate_tool_call(
                                            repaired_call,
                                            tools_schema.is_array()
                                                ? &tools_schema : nullptr,
                                            validation_error)) {
                                        parsed.valid = false;
                                        parsed.error = validation_error.empty()
                                            ? "closure repair produced no "
                                              "complete tool call"
                                            : validation_error;
                                    } else {
                                        parsed.calls.push_back(repaired_call);
                                    }
                                } else {
                                    parsed = parse_tool_calls_xml(
                                        framed,
                                        tools_schema.is_array() ? &tools_schema
                                                                : nullptr);
                                }
                                if (!parsed.valid || parsed.calls.empty()) {
                                    const std::string parse_error =
                                        parsed.error.empty()
                                            ? "tool intent produced no call"
                                            : parsed.error;
                                    if (!tool_delta_committed) {
                                        if (hit_max_tokens) {
                                            throw std::runtime_error(
                                                "tool_call_parse_error: tool "
                                                "call was truncated at "
                                                "max_tokens");
                                        }
                                        throw std::runtime_error(
                                            tool_call_parse_error_message(
                                                parse_error));
                                    }
                                    throw std::runtime_error(
                                        "tool_call_parse_error: committed "
                                        "tool call failed validation: " +
                                        parse_error);
                                }

                                const ReasoningSplit split =
                                    split_reasoning(framed);
                                if (!streamed_content &&
                                    !split.reasoning.empty()) {
                                    send_delta(json{
                                        {"reasoning_content", split.reasoning}});
                                }
                                if (incremental.streaming()) {
                                    std::string validation_error;
                                    if (!incremental.validate(
                                            parsed.calls,
                                            validation_error)) {
                                        if (!tool_delta_committed) {
                                            if (hit_max_tokens) {
                                                throw std::runtime_error(
                                                    "tool_call_parse_error: "
                                                    "tool call was truncated "
                                                    "at max_tokens");
                                            }
                                            throw std::runtime_error(
                                                tool_call_parse_error_message(
                                                    validation_error));
                                        }
                                        throw std::runtime_error(
                                            "tool_call_parse_error: "
                                            "incremental/full parse mismatch: " +
                                            validation_error);
                                    }
                                    if (!tool_delta_committed) {
                                        tool_delta_committed = true;
                                    }
                                    flush_incremental(true);
                                    if (parsed.calls.size() > 1) {
                                        send_delta(tool_call_delta(
                                            parsed.calls, 1));
                                    }
                                } else {
                                    send_delta(tool_call_delta(parsed.calls));
                                }
                                std::cerr << "[qw3-serve] #" << rid
                                          << " tool_calls="
                                          << tool_calls_debug_summary(
                                                 parsed.calls)
                                          << " incremental="
                                          << (incremental.streaming()
                                                  ? "true" : "false")
                                          << " closure_repaired="
                                          << (!closure_suffix.empty()
                                                  ? "true" : "false")
                                          << "\n";
                                send_done("tool_calls");
                            } else if (forced_tool_request) {
                                if (g.max_tokens > 0 &&
                                    completion_tokens >=
                                        static_cast<size_t>(g.max_tokens)) {
                                    throw std::runtime_error(
                                        "tool_call_parse_error: forced tool "
                                        "call was truncated at max_tokens");
                                }
                                throw std::runtime_error(
                                    tool_call_parse_error_message(
                                        "forced tool choice produced no "
                                        "<tool_call> block"));
                            } else {
                                if (!content_pending.empty()) {
                                    emit_text(content_pending);
                                }
                                finish_text_stream();
                                send_done(generation_finish_reason(
                                    stopped, completion_tokens, g.max_tokens));
                            }
                            std::cerr << "[qw3-serve] #" << rid
                                      << " chat(stream tools) chars=" << acc.size()
                                      << " completion_tokens=" << completion_tokens
                                      << " prompt_tokens=" << prompt_token_count
                                      << " route=" << route
                                      << " buffered_tool=" << (buffering_tool ? "true" : "false")
                                      << " incremental_tool="
                                      << (incremental.streaming() ? "true" : "false")
                                      << "\n";
                            const auto request_end =
                                std::chrono::steady_clock::now();
                            ttft.log(rid, route, true, request_end);
                            log_server_accounting(
                                engine_start, engine_end,
                                request_end, true);
                            return true;
                        }
                        generate_request([&](const std::string &piece) {
                            if (stopped || client_closed) return false;
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
                            return !stopped && !client_closed;
                        });
                        finish_text_stream();
                        send_done(generation_finish_reason(
                            stopped, completion_tokens, g.max_tokens));
                        std::cerr << "[qw3-serve] #" << rid
                                  << " chat(stream) completion_tokens="
                                  << completion_tokens
                                  << " prompt_tokens=" << prompt_token_count
                                  << " route=" << route
                                  << (client_closed ? " client_closed=true" : "")
                                  << "\n";
                        const auto request_end =
                            std::chrono::steady_clock::now();
                        ttft.log(rid, route, true, request_end);
                        log_server_accounting(
                            engine_start, engine_end,
                            request_end, true);
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
                        ttft.log(rid, route, true,
                                 std::chrono::steady_clock::now());
                        return false;
                    }
                });
            return;
        }

        std::string text;
        size_t completion_tokens = 0;
        auto engine_start = std::chrono::steady_clock::now();
        auto engine_end = engine_start;
        ServerTtftTracker ttft(server_request_start);
        auto consume_piece = [&](const std::string &piece) {
            ttft.observe_model_piece(piece);
            ++completion_tokens;
            text += piece;
        };
        try {
            if (kvmem_session_request) {
                std::lock_guard<std::mutex> lk(gen_mu);
                engine_start = std::chrono::steady_clock::now();
                ttft.start_engine(engine_start);
                eng.generate_session_stream(
                    prompt, g, [&](const std::string &piece) {
                        consume_piece(piece);
                    }, kvmem_session_reset);
            } else if (g.continuous_batching) {
                engine_start = std::chrono::steady_clock::now();
                ttft.start_engine(engine_start);
                eng.generate_stream_cancellable(prompt, g, [&](const std::string &piece) {
                    consume_piece(piece);
                    return true;
                });
            } else {
                std::lock_guard<std::mutex> lk(gen_mu);
                engine_start = std::chrono::steady_clock::now();
                ttft.start_engine(engine_start);
                eng.generate_stream_cancellable(prompt, g, [&](const std::string &piece) {
                    consume_piece(piece);
                    return true;
                });
            }
            engine_end = std::chrono::steady_clock::now();
        } catch (const std::exception &e) {
            std::cerr << "[qw3-serve] #" << rid << " chat error="
                      << e.what() << "\n";
            set_error_response(res, status_for_exception(e), e.what());
            return;
        }
        if (enable_thinking && g.max_tokens > 0) text = "<think>\n" + text;
        std::string utf8_pending;
        text = take_complete_utf8(utf8_pending, text);
        text += flush_utf8_pending(utf8_pending, false);
        const bool stopped = apply_stops(text, stops);
        ToolParseResult parsed = parse_tool_calls_xml(text, tools);
        const bool tool_parse_failed =
            tool_request &&
            ((parsed.intent_detected &&
              (!parsed.valid || parsed.calls.empty())) ||
             (forced_tool_request && parsed.calls.empty()));
        if (tool_parse_failed) {
            if (g.max_tokens > 0 &&
                completion_tokens >= static_cast<size_t>(g.max_tokens)) {
                set_error_response(
                    res, 500,
                    "tool_call_parse_error: tool call was truncated at "
                    "max_tokens");
                return;
            }
            const std::string error = tool_call_parse_error_message(
                parsed.error.empty()
                    ? "forced tool choice produced no call"
                    : parsed.error);
            std::cerr << "[qw3-serve] #" << rid << " " << error << "\n";
            set_error_response(res, 500, error);
            return;
        }
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
        const ReasoningSplit split = split_reasoning(text);
        json message = json{{"role", "assistant"},
                            {"content", parsed.calls.empty()
                                            ? split.content : ""}};
        if (!split.reasoning.empty()) {
            message["reasoning_content"] = split.reasoning;
        }
        std::string finish = generation_finish_reason(
            stopped, completion_tokens, g.max_tokens);
        if (!parsed.calls.empty()) {
            std::cerr << "[qw3-serve] #" << rid
                      << " tool_calls="
                      << tool_calls_debug_summary(parsed.calls)
                      << "\n";
            message["tool_calls"] = parsed.calls;
            finish = "tool_calls";
        }
        json out = {
            {"id", id}, {"object", "chat.completion"}, {"created", created},
            {"model", model_id},
            {"choices", json::array({json{
                {"index", 0},
                {"message", message},
                {"finish_reason", finish}}})},
            {"usage", usage_json(prompt_token_count, completion_tokens)},
            {"timing", ttft.timing_json(std::chrono::steady_clock::now())}};
        if (kvmem_cache_request) {
            const KvMemLocalCacheInfo info =
                eng.kvmem_local_cache_info(kvmem_cache_id);
            if (!info.found) {
                set_error_response(
                    res, 500,
                    "KVMem local cache operation completed without metadata");
                return;
            }
            out["kvmem_cache"] = kvmem_cache_info_json(info);
        }
        res.set_content(dump_json(out), "application/json");
        const auto request_end = std::chrono::steady_clock::now();
        ttft.log(rid, route, false, request_end);
        log_server_accounting(
            engine_start, engine_end, request_end, false);
    });

    svr.Post("/v1/completions", [&](const httplib::Request &hreq,
                                    httplib::Response &res) {
        const auto server_request_start = std::chrono::steady_clock::now();
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
        std::vector<uint32_t> exact_prompt_tokens;
        if (req.contains("prompt") && req["prompt"].is_string()) {
            prompt = req["prompt"].get<std::string>();
        } else if (req.contains("prompt") && req["prompt"].is_array() &&
                   !req["prompt"].empty() && req["prompt"][0].is_string()) {
            prompt = req["prompt"][0].get<std::string>();
        } else if (req.contains("prompt") && req["prompt"].is_array() &&
                   !req["prompt"].empty()) {
            exact_prompt_tokens.reserve(req["prompt"].size());
            for (const json &value : req["prompt"]) {
                if (!value.is_number_integer()) {
                    set_error_response(
                        res, 400,
                        "integer-array prompt must contain only token IDs");
                    return;
                }
                const int64_t token = value.get<int64_t>();
                if (token < 0 ||
                    token > static_cast<int64_t>(
                                std::numeric_limits<uint32_t>::max())) {
                    set_error_response(
                        res, 400,
                        "integer-array prompt token ID is out of range");
                    return;
                }
                exact_prompt_tokens.push_back(static_cast<uint32_t>(token));
            }
        } else {
            res.status = 400;
            res.set_content(dump_json(json{{"error", "missing prompt"}}),
                            "application/json");
            return;
        }
        bool explicit_max_tokens = false;
        int requested_max_tokens = 0;
        std::string max_tokens_error;
        if (!parse_explicit_max_tokens(req, explicit_max_tokens,
                                       requested_max_tokens,
                                       max_tokens_error)) {
            set_error_response(res, 400, max_tokens_error);
            return;
        }
        (void)explicit_max_tokens;
        (void)requested_max_tokens;
        const size_t prompt_token_count = exact_prompt_tokens.empty()
            ? usage_tokenizer.encode(prompt).size()
            : exact_prompt_tokens.size();
        if (archive_prefix_tokens + prompt_token_count >=
            static_cast<uint64_t>(std::max(1, engine.ctx_size))) {
            set_error_response(
                res,
                413,
                "archive prefix plus prompt exceeds KV context: archive_tokens=" +
                    std::to_string(archive_prefix_tokens) +
                    " prompt_tokens=" +
                    std::to_string(prompt_token_count) +
                    " ctx=" + std::to_string(engine.ctx_size));
            return;
        }
        const bool enable_thinking =
            req.value("enable_thinking", cfg.enable_thinking_default);
        GenerationOptions g;
        try {
            g = make_gen(req, prompt_token_count, enable_thinking);
        } catch (const std::invalid_argument &e) {
            set_error_response(res, 400, e.what());
            return;
        }
        g.raw_prompt = true; // /v1/completions sends raw text, no chat template
        // Raw completions receive an already-rendered prompt. The caller is
        // responsible for including the assistant/<think> prefix; this flag
        // only makes the ordinary thinking-token budget track that open span.
        g.thinking_open = enable_thinking;
        g.prompt_token_ids_override = std::move(exact_prompt_tokens);

        // Raw-token persistent sessions are useful when a benchmark must freeze
        // a byte/token-identical history and time only a later query fragment.
        // Chat sessions cannot provide that guarantee when the split falls
        // inside one message because rendering the two requests would insert an
        // extra role boundary.  Keep the controls explicit and token based:
        // callers may pass an integer-array prompt plus a fragment-local query
        // span.  The native persistent-session layer translates that span to
        // logical sequence coordinates using the current append base.
        bool kvmem_session_request = false;
        bool kvmem_session_reset = false;
        std::string kvmem_session_op;
        if (req.contains("kvmem_session_id") ||
            req.contains("kvmem_session_op")) {
            if (!engine.kvmem_enabled) {
                set_error_response(
                    res, 400, "kvmem_session_* requires --kvmem");
                return;
            }
            if (!req.contains("kvmem_session_id") ||
                !req["kvmem_session_id"].is_string() ||
                req["kvmem_session_id"].get<std::string>().empty() ||
                !req.contains("kvmem_session_op") ||
                !req["kvmem_session_op"].is_string()) {
                set_error_response(
                    res, 400,
                    "raw completion sessions require a non-empty "
                    "kvmem_session_id and kvmem_session_op=start|append|finish");
                return;
            }
            g.kvmem_session_id =
                req["kvmem_session_id"].get<std::string>();
            kvmem_session_op =
                req["kvmem_session_op"].get<std::string>();
            if (kvmem_session_op != "start" &&
                kvmem_session_op != "append" &&
                kvmem_session_op != "finish") {
                set_error_response(
                    res, 400,
                    "kvmem_session_op must be start|append|finish");
                return;
            }
            if (kvmem_session_op != "finish" && g.max_tokens != 0) {
                set_error_response(
                    res, 400,
                    "kvmem session start/append requires max_tokens=0");
                return;
            }
            kvmem_session_request = true;
            kvmem_session_reset = kvmem_session_op == "start";
        }

        bool kvmem_cache_request = false;
        std::string kvmem_cache_id;
        std::string kvmem_cache_operation;
        KvMemLocalCacheMode kvmem_cache_mode =
            KvMemLocalCacheMode::None;
        uint64_t kvmem_cache_expected_version = 0;
        bool kvmem_cache_expected_version_set = false;
        uint64_t kvmem_cache_ttl_seconds = 0;
        if (req.contains("kvmem_cache")) {
            if (!engine.kvmem_enabled) {
                set_error_response(res, 400,
                                   "kvmem_cache requires --kvmem");
                return;
            }
            if (kvmem_session_request) {
                set_error_response(
                    res, 400,
                    "kvmem_cache and kvmem_session_* are mutually exclusive");
                return;
            }
            const json &cache = req["kvmem_cache"];
            if (!cache.is_object()) {
                set_error_response(res, 400,
                                   "kvmem_cache must be an object");
                return;
            }
            const bool save = cache.contains("save");
            const bool load = cache.contains("load");
            if (save == load) {
                set_error_response(
                    res, 400,
                    "kvmem_cache requires exactly one save or load object");
                return;
            }
            const json &operation = cache[save ? "save" : "load"];
            if (!operation.is_object() || !operation.contains("id") ||
                !operation["id"].is_string()) {
                set_error_response(
                    res, 400,
                    "kvmem_cache save/load requires a string id");
                return;
            }
            kvmem_cache_id = operation["id"].get<std::string>();
            if (!valid_kvmem_cache_id(kvmem_cache_id)) {
                set_error_response(
                    res, 400,
                    "kvmem_cache id must be 1..128 characters using only "
                    "letters, digits, '.', '_', '-', or ':'");
                return;
            }
            if (save) {
                kvmem_cache_operation = "save";
                if (g.max_tokens != 0) {
                    set_error_response(
                        res, 400,
                        "kvmem_cache save currently requires max_tokens=0");
                    return;
                }
                const std::string scope = operation.value("scope", "local");
                const std::string when =
                    operation.value("when", "after_request");
                if (scope != "local" || when != "after_request") {
                    set_error_response(
                        res, 400,
                        "kvmem_cache save supports only scope=local and "
                        "when=after_request");
                    return;
                }
                if (operation.contains("ttl_seconds") &&
                    !parse_bounded_json_u64(
                        operation["ttl_seconds"], 0, 31536000,
                        kvmem_cache_ttl_seconds)) {
                    set_error_response(
                        res, 400,
                        "kvmem_cache ttl_seconds must be in [0,31536000]");
                    return;
                }
                g.kvmem_cache_save_id = kvmem_cache_id;
                g.kvmem_cache_ttl_seconds = kvmem_cache_ttl_seconds;
            } else {
                kvmem_cache_operation = "load";
                const std::string mode = operation.value("mode", "frozen");
                if (mode == "frozen") {
                    kvmem_cache_mode = KvMemLocalCacheMode::Frozen;
                } else if (mode == "append") {
                    kvmem_cache_mode = KvMemLocalCacheMode::Append;
                } else {
                    set_error_response(
                        res, 400,
                        "kvmem_cache load mode must be frozen|append");
                    return;
                }
                if (operation.contains("required") &&
                    (!operation["required"].is_boolean() ||
                     !operation["required"].get<bool>())) {
                    set_error_response(
                        res, 400,
                        "process-local cache loads require required=true; "
                        "missing caches never silently trigger full prefill");
                    return;
                }
                if (operation.contains("expected_version")) {
                    if (!parse_bounded_json_u64(
                            operation["expected_version"], 1,
                            std::numeric_limits<uint64_t>::max(),
                            kvmem_cache_expected_version)) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache expected_version must be a positive "
                            "integer");
                        return;
                    }
                    kvmem_cache_expected_version_set = true;
                }
                if (kvmem_cache_mode == KvMemLocalCacheMode::Append) {
                    if (g.max_tokens != 0) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache append currently requires "
                            "max_tokens=0");
                        return;
                    }
                    if (!kvmem_cache_expected_version_set) {
                        set_error_response(
                            res, 400,
                            "kvmem_cache append requires expected_version");
                        return;
                    }
                }
                g.kvmem_cache_load_id = kvmem_cache_id;
                g.kvmem_cache_load_mode = kvmem_cache_mode;
                g.kvmem_cache_expected_version =
                    kvmem_cache_expected_version;
                g.kvmem_cache_expected_version_set =
                    kvmem_cache_expected_version_set;
            }
            kvmem_cache_request = true;
        }

        if (req.contains("kvmem_reselect")) {
            if (!req["kvmem_reselect"].is_string()) {
                set_error_response(
                    res, 400, "kvmem_reselect must be auto|force|off");
                return;
            }
            const std::string mode =
                req["kvmem_reselect"].get<std::string>();
            if (mode == "auto") {
                g.kvmem_reselect_mode = KvMemReselectMode::Auto;
            } else if (mode == "force") {
                g.kvmem_reselect_mode = KvMemReselectMode::Force;
            } else if (mode == "off") {
                g.kvmem_reselect_mode = KvMemReselectMode::Off;
            } else {
                set_error_response(
                    res, 400, "kvmem_reselect must be auto|force|off");
                return;
            }
        }
        if (req.contains("kvmem_query_token_span")) {
            const json &span = req["kvmem_query_token_span"];
            uint64_t begin = 0;
            uint64_t end = 0;
            if (!span.is_object() || !span.contains("begin") ||
                !span.contains("end") ||
                !parse_bounded_json_u64(
                    span["begin"], 0,
                    std::numeric_limits<uint32_t>::max(), begin) ||
                !parse_bounded_json_u64(
                    span["end"], 1,
                    std::numeric_limits<uint32_t>::max(), end) ||
                end <= begin) {
                set_error_response(
                    res, 400,
                    "kvmem_query_token_span requires uint32 begin < end");
                return;
            }
            if (!engine.kvmem_enabled ||
                !engine.kvmem_query_conditioned) {
                set_error_response(
                    res, 400,
                    "kvmem_query_token_span requires --kvmem and "
                    "--kvmem-query-conditioned");
                return;
            }
            g.kvmem_query_begin = static_cast<uint32_t>(begin);
            g.kvmem_query_end = static_cast<uint32_t>(end);
        }
        if (g.kvmem_reselect_mode == KvMemReselectMode::Force &&
            g.kvmem_query_end <= g.kvmem_query_begin) {
            set_error_response(
                res, 400,
                "kvmem_reselect=force requires kvmem_query_token_span");
            return;
        }
        if (req.contains("kvmem_trace_tag")) {
            if (!req["kvmem_trace_tag"].is_string()) {
                set_error_response(res, 400,
                                   "kvmem_trace_tag must be a string");
                return;
            }
            const std::string tag =
                req["kvmem_trace_tag"].get<std::string>();
            if (tag.empty() || tag.size() > 128 ||
                !std::all_of(tag.begin(), tag.end(), [](unsigned char c) {
                    return std::isalnum(c) || c == '.' || c == '_' ||
                           c == '-' || c == ':';
                })) {
                set_error_response(
                    res, 400,
                    "kvmem_trace_tag must be 1..128 characters from "
                    "[A-Za-z0-9_.:-]");
                return;
            }
            g.kvmem_trace_tag = tag;
        }
        g.continuous_batching =
            !kvmem_session_request && !kvmem_cache_request &&
            serve_continuous_batching_enabled() &&
            serve_continuous_batch_request_supported(g);
        const std::string route = kvmem_cache_request
            ? ("cache-" + kvmem_cache_operation)
            : (kvmem_session_request
                   ? ("session-" + kvmem_session_op)
                   : (g.continuous_batching ? "continuous" : "plain"));
        const std::string fallback_reason =
            g.continuous_batching ? "" :
            (serve_continuous_batching_enabled() ? "request_unsupported" : "disabled");
        const std::vector<std::string> stops = parse_stops(req);
        const std::string id = gen_id("cmpl-");
        const int64_t created = unix_now();
        const uint64_t rid = ++req_counter;

        std::string text;
        size_t completion_tokens = 0;
        auto generation_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point first_token_at{};
        bool first_token_seen = false;
        ServerTtftTracker ttft(server_request_start);
        auto consume_piece = [&](const std::string &piece) {
            ++completion_tokens;
            ttft.observe_model_piece(piece);
            if (!first_token_seen && !piece.empty()) {
                first_token_seen = true;
                first_token_at = std::chrono::steady_clock::now();
            }
            text += piece;
        };
        try {
            if (kvmem_session_request) {
                std::lock_guard<std::mutex> lk(gen_mu);
                generation_start = std::chrono::steady_clock::now();
                ttft.start_engine(generation_start);
                eng.generate_session_stream(
                    prompt, g, consume_piece, kvmem_session_reset);
            } else if (g.continuous_batching) {
                generation_start = std::chrono::steady_clock::now();
                ttft.start_engine(generation_start);
                eng.generate_stream_cancellable(prompt, g, [&](const std::string &piece) {
                    consume_piece(piece);
                    return true;
                });
            } else {
                std::lock_guard<std::mutex> lk(gen_mu);
                generation_start = std::chrono::steady_clock::now();
                ttft.start_engine(generation_start);
                eng.generate_stream_cancellable(prompt, g, [&](const std::string &piece) {
                    consume_piece(piece);
                    return true;
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
        const auto response_build_at = std::chrono::steady_clock::now();
        json timing = ttft.timing_json(response_build_at);
        // Preserve the original engine-relative field for existing clients.
        timing["first_token_sec"] = first_token_seen
            ? json(std::chrono::duration<double>(
                  first_token_at - generation_start).count())
            : json(nullptr);
        json out = {
            {"id", id}, {"object", "text_completion"}, {"created", created},
            {"model", model_id},
            {"choices", json::array({json{
                {"index", 0}, {"text", text}, {"logprobs", nullptr},
                {"finish_reason", generation_finish_reason(
                    stopped, completion_tokens, g.max_tokens)}}})},
            {"usage", usage_json(prompt_token_count, completion_tokens)},
            {"timing", std::move(timing)}};
        if (kvmem_cache_request) {
            std::lock_guard<std::mutex> lk(gen_mu);
            const KvMemLocalCacheInfo info =
                eng.kvmem_local_cache_info(kvmem_cache_id);
            if (!info.found) {
                set_error_response(
                    res, 500,
                    "KVMem local cache operation completed without metadata");
                return;
            }
            out["kvmem_cache"] = kvmem_cache_info_json(info);
        }
        res.set_content(dump_json(out), "application/json");
        ttft.log(rid, route, false, std::chrono::steady_clock::now());
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
