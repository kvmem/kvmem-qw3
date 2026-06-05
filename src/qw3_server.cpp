#include "server.hpp"

#include "qw3/qw3.hpp"

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

// Render an OpenAI messages[] array into a Qwen3 chat transcript. System
// messages are concatenated into the system block; user/assistant turns are
// emitted in order with <|im_start|>role ... <|im_end|> framing. The final
// assistant header (+ optional empty-think block) is appended for generation.
std::string render_messages(const json &messages, bool enable_thinking) {
    std::string system;
    std::string body;
    for (const auto &m : messages) {
        const std::string role = m.value("role", "");
        const std::string content = m.value("content", "");
        if (role == "system") {
            if (!system.empty()) system += "\n";
            system += content;
        } else if (role == "user" || role == "assistant" || role == "tool") {
            const std::string r = (role == "tool") ? "user" : role;
            body += "<|im_start|>" + r + "\n" + content + "<|im_end|>\n";
        }
    }
    std::string prompt;
    if (!system.empty()) {
        prompt += "<|im_start|>system\n" + system + "<|im_end|>\n";
    }
    prompt += body;
    prompt += "<|im_start|>assistant\n";
    if (!enable_thinking) prompt += "<think>\n\n</think>\n\n";
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

    std::cerr << "[qw3-serve] loading model: " << engine.model_path << "\n";
    Engine eng(engine);
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
        res.set_content(out.dump(), "application/json");
    });

    // Build GenerationOptions from common OpenAI fields. Default temperature=0
    // (greedy) for reproducible evals when the client omits it; default top_p=1.
    auto make_gen = [](const json &req) -> GenerationOptions {
        GenerationOptions g;
        g.max_tokens = req.value("max_tokens",
                                 req.value("max_completion_tokens", 256));
        if (g.max_tokens <= 0) g.max_tokens = 256;
        g.temperature = req.value("temperature", 0.0f);
        g.top_p = req.value("top_p", 1.0f);
        g.seed = req.value("seed", static_cast<uint64_t>(0));
        return g;
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request &hreq,
                                         httplib::Response &res) {
        json req;
        try {
            req = json::parse(hreq.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(),
                            "application/json");
            return;
        }
        if (!req.contains("messages") || !req["messages"].is_array()) {
            res.status = 400;
            res.set_content(json{{"error", "missing messages[]"}}.dump(),
                            "application/json");
            return;
        }
        const bool enable_thinking =
            req.value("enable_thinking", cfg.enable_thinking_default);
        const std::string prompt = render_messages(req["messages"], enable_thinking);
        GenerationOptions g = make_gen(req);
        g.raw_prompt = true; // prompt is already chat-framed
        const std::vector<std::string> stops = parse_stops(req);
        const bool stream = req.value("stream", false);
        const std::string id = gen_id("chatcmpl-");
        const int64_t created = unix_now();
        const uint64_t rid = ++req_counter;
        const auto t0 = std::chrono::steady_clock::now();

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [&, prompt, g, stops, id, created, rid](size_t,
                                                        httplib::DataSink &sink) {
                    std::lock_guard<std::mutex> lk(gen_mu);
                    std::string acc;
                    int n_tok = 0;
                    bool stopped = false;
                    auto send_role = [&]() {
                        json chunk = {
                            {"id", id}, {"object", "chat.completion.chunk"},
                            {"created", created}, {"model", model_id},
                            {"choices", json::array({json{
                                {"index", 0},
                                {"delta", json{{"role", "assistant"}}},
                                {"finish_reason", nullptr}}})}};
                        const std::string s = "data: " + chunk.dump() + "\n\n";
                        sink.write(s.data(), s.size());
                    };
                    send_role();
                    eng.generate_stream(prompt, g, [&](const std::string &piece) {
                        if (stopped) return;
                        acc += piece;
                        std::string emit = piece;
                        if (!stops.empty()) {
                            std::string probe = acc;
                            if (apply_stops(probe, stops)) {
                                stopped = true;
                                // emit only the part before the stop that we
                                // haven't already sent
                                emit = probe.size() > (acc.size() - piece.size())
                                           ? probe.substr(acc.size() - piece.size())
                                           : "";
                            }
                        }
                        if (!emit.empty()) {
                            ++n_tok;
                            json chunk = {
                                {"id", id}, {"object", "chat.completion.chunk"},
                                {"created", created}, {"model", model_id},
                                {"choices", json::array({json{
                                    {"index", 0},
                                    {"delta", json{{"content", emit}}},
                                    {"finish_reason", nullptr}}})}};
                            const std::string s = "data: " + chunk.dump() + "\n\n";
                            sink.write(s.data(), s.size());
                        }
                    });
                    json done = {
                        {"id", id}, {"object", "chat.completion.chunk"},
                        {"created", created}, {"model", model_id},
                        {"choices", json::array({json{
                            {"index", 0}, {"delta", json::object()},
                            {"finish_reason", stopped ? "stop" : "length"}}})}};
                    const std::string ds = "data: " + done.dump() + "\n\n";
                    sink.write(ds.data(), ds.size());
                    const std::string fin = "data: [DONE]\n\n";
                    sink.write(fin.data(), fin.size());
                    sink.done();
                    std::cerr << "[qw3-serve] #" << rid
                              << " chat(stream) tokens=" << n_tok << "\n";
                    return true;
                });
            return;
        }

        std::string text;
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            text = eng.generate(prompt, g);
        }
        const bool stopped = apply_stops(text, stops);
        const double ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();
        std::cerr << "[qw3-serve] #" << rid << " chat chars=" << text.size()
                  << " " << ms << "ms\n";
        json out = {
            {"id", id}, {"object", "chat.completion"}, {"created", created},
            {"model", model_id},
            {"choices", json::array({json{
                {"index", 0},
                {"message", json{{"role", "assistant"}, {"content", text}}},
                {"finish_reason", stopped ? "stop" : "stop"}}})},
            {"usage", json{{"prompt_tokens", 0},
                           {"completion_tokens", 0},
                           {"total_tokens", 0}}}};
        res.set_content(out.dump(), "application/json");
    });

    svr.Post("/v1/completions", [&](const httplib::Request &hreq,
                                    httplib::Response &res) {
        json req;
        try {
            req = json::parse(hreq.body);
        } catch (const std::exception &e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("invalid JSON: ") + e.what()}}.dump(),
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
            res.set_content(json{{"error", "missing prompt"}}.dump(),
                            "application/json");
            return;
        }
        GenerationOptions g = make_gen(req);
        g.raw_prompt = true; // /v1/completions sends raw text, no chat template
        const std::vector<std::string> stops = parse_stops(req);
        const std::string id = gen_id("cmpl-");
        const int64_t created = unix_now();
        const uint64_t rid = ++req_counter;

        std::string text;
        {
            std::lock_guard<std::mutex> lk(gen_mu);
            text = eng.generate(prompt, g);
        }
        const bool stopped = apply_stops(text, stops);
        std::cerr << "[qw3-serve] #" << rid << " completion chars="
                  << text.size() << "\n";
        json out = {
            {"id", id}, {"object", "text_completion"}, {"created", created},
            {"model", model_id},
            {"choices", json::array({json{
                {"index", 0}, {"text", text}, {"logprobs", nullptr},
                {"finish_reason", stopped ? "stop" : "length"}}})},
            {"usage", json{{"prompt_tokens", 0},
                           {"completion_tokens", 0},
                           {"total_tokens", 0}}}};
        res.set_content(out.dump(), "application/json");
    });

    svr.set_logger([](const httplib::Request &req, const httplib::Response &res) {
        if (req.path == "/health") return; // quiet the poll loop
        std::cerr << "[qw3-serve] " << req.method << " " << req.path
                  << " -> " << res.status << "\n";
    });

    std::cerr << "[qw3-serve] listening on http://" << cfg.host << ":"
              << cfg.port << "  (model loaded once, requests serialized)\n";
    if (!svr.listen(cfg.host, cfg.port)) {
        std::cerr << "[qw3-serve] failed to bind " << cfg.host << ":"
                  << cfg.port << "\n";
        return 1;
    }
    return 0;
}

} // namespace qw3

