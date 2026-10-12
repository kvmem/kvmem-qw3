#pragma once
#include "harness_semantics.hpp"
#include "json.hpp"
#include <string>
#include <vector>

namespace qw3::detail {
struct ServerRenderTestResult {
    std::size_t query_index = 0;
    std::string prompt;
    std::vector<HarnessRenderedMessageSpan> messages;
    std::size_t control_end = 0;
};
// Implemented only in the host renderer-test target; no HTTP server is started.
ServerRenderTestResult inspect_server_render(const nlohmann::json& messages,
    HarnessKind kind, bool stable_reasoning, bool add_generation_prompt);
} // namespace qw3::detail
