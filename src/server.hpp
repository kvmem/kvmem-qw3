#pragma once

#include "qw3/qw3.hpp"

#include <cstdint>
#include <string>

namespace qw3 {

// Configuration for the OpenAI-compatible HTTP server. The model itself is
// loaded once (from EngineOptions) and reused across all requests. Defaults are
// the conservative single-request serving baseline; explicit CLI switches opt
// into paged KV, continuous batching, and MTP.
struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    GenerationOptions default_generation;
    bool default_max_tokens_set = false;
    // Track whether the user pinned sampling on the CLI. When unset, the server
    // applies the Qwen3-recommended preset for the request's thinking mode
    // (thinking: temp 0.6/top_p 0.95; non-thinking: temp 0.7/top_p 0.8).
    bool temperature_set = false;
    bool top_p_set = false;
    bool continuous_batching = false;
    bool paged_kv = false;
    bool paged_kv_set = false;
    bool body_batch = false;
    bool body_batch_set = false;
    bool mtp_paged_prefix = false;
    bool mtp_paged_prefix_set = false;
    bool mtp_batched_draft = false;
    bool mtp_batched_draft_set = false;
    // Lossless page-aligned prefix KV caching on the continuous-batching path.
    // Off by default; --prefix-cache opts in. Page budget is unlimited (bounded
    // by the KV pool) and tracing is off — these are the tuned values, so they
    // are not exposed as separate switches.
    bool prefix_cache = false;
    int max_active = 2;
    int max_pending = 128;
    int prefill_burst = 0;
    uint64_t max_total_tokens = 0;
    bool max_total_tokens_set = false;
    int kv_page_size = 16;
    int kv_pool_pages = 0;
    int mtp_kv_pool_pages = 0;
    std::string kv_dtype = "fp16";
    // Default think-suppression for /v1/chat/completions when the request does
    // not carry an explicit `enable_thinking`. false => inject the empty
    // <think>\n\n</think> block (brief working, no long CoT), matching the eval
    // harness expectation.
    bool enable_thinking_default = false;
    // Default thinking budget (max tokens inside <think>) when a chat request
    // omits `thinking_budget`. 0 disables the cap.
    int thinking_budget_default = 0;
};

// Loads the model (via Engine(engine)) and serves until the process is killed.
// Returns a process exit code (0 on clean shutdown). Blocking.
int run_server(EngineOptions engine, ServerConfig cfg);

} // namespace qw3
