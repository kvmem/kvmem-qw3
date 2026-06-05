#pragma once

#include "qw3/qw3.hpp"

#include <string>

namespace qw3 {

// Configuration for the OpenAI-compatible HTTP server. The model itself is
// loaded once (from EngineOptions) and reused across all requests; requests are
// serialized behind a mutex because the native executor holds a single shared
// KV cache + scratch and is not safe for concurrent generate() calls.
struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080;
    GenerationOptions default_generation;
    // Default think-suppression for /v1/chat/completions when the request does
    // not carry an explicit `enable_thinking`. false => inject the empty
    // <think>\n\n</think> block (brief working, no long CoT), matching the eval
    // harness expectation.
    bool enable_thinking_default = false;
};

// Loads the model (via Engine(engine)) and serves until the process is killed.
// Returns a process exit code (0 on clean shutdown). Blocking.
int run_server(EngineOptions engine, ServerConfig cfg);

} // namespace qw3
