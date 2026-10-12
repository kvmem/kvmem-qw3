#include "qw3/qw3.hpp"

#include "server_render_test_access.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require_equal(const std::string &actual,
                   const std::string &expected,
                   const char *message) {
    if (actual != expected) {
        throw std::runtime_error(message);
    }
}

void test_non_thinking_prompt_with_system_message() {
    const std::string rendered = qw3::render_qwen3_chat_prompt(
        "system instructions",
        "hello",
        false);
    require_equal(
        rendered,
        "<|im_start|>system\n"
        "system instructions"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "hello"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n",
        "non-thinking chat prompt mismatch");
}

void test_thinking_prompt_does_not_inject_empty_think_block() {
    const std::string rendered = qw3::render_qwen3_chat_prompt(
        "system instructions",
        "hello",
        true);
    require_equal(
        rendered,
        "<|im_start|>system\n"
        "system instructions"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "hello"
        "<|im_end|>\n"
        "<|im_start|>assistant\n",
        "thinking chat prompt mismatch");
}

void test_empty_system_message_is_omitted() {
    const std::string rendered = qw3::render_qwen3_chat_prompt(
        "",
        "hello",
        false);
    require_equal(
        rendered,
        "<|im_start|>user\n"
        "hello"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n",
        "empty system message was not omitted");
}


void test_server_clock_reasoning_policy() {
    using namespace qw3::detail;
    const std::string clock =
        "Time sampled while preparing turn 8, step 2: 2026-10-12T12:00:00+08:00[Asia/Shanghai]\n"
        "Browser time zone for this request: Asia/Shanghai. Interpret otherwise-unqualified dates and times in this zone.\n"
        "Elapsed since the preceding model-visible message: 1s.";
    nlohmann::json messages = nlohmann::json::array({
        {{"role","user"},{"content","CURRENT_TASK"}},
        {{"role","assistant"},{"content",""},{"reasoning_content","PLAN"},
         {"tool_calls",nlohmann::json::array({{{"id","call"},{"type","function"},
            {"function",{{"name","lookup"},{"arguments","{}"}}}}})}},
        {{"role","tool"},{"content","RESULT"},{"tool_call_id","call"}},
        {{"role","user"},{"content",clock}}});
    for (const auto kind : {HarnessKind::DeepSeekHarness, HarnessKind::GenericToolClient}) {
        const auto ordinary = inspect_server_render(messages,HarnessKind::None,false,true);
        const auto active = inspect_server_render(messages,kind,false,true);
        if (ordinary.query_index != 3 || active.query_index != 0 ||
            ordinary.prompt.find("PLAN") != std::string::npos ||
            active.prompt.find("<think>\nPLAN\n</think>\n\n") == std::string::npos ||
            ordinary.prompt.find(clock) == std::string::npos || active.prompt.find(clock) == std::string::npos ||
            active.prompt.find("RESULT") == std::string::npos)
            throw std::runtime_error("server clock/current-task reasoning policy mismatch");
        require_equal(inspect_server_render(messages,kind,true,true).prompt,
            inspect_server_render(messages,HarnessKind::None,true,true).prompt,
            "stable reasoning prompt changed with clock task attribution");
        require_equal(inspect_server_render(messages,kind,false,false).prompt,
            inspect_server_render(messages,HarnessKind::None,false,false).prompt,
            "prefill-only prompt changed with clock task attribution");
        const auto plan = derive_harness_semantic_plan(kind,active.prompt,active.control_end,active.messages);
        if (plan.current_query_message_index != 0 || plan.live_suffix_message_begin_index != 1)
            throw std::runtime_error("real server renderer lost the current task/live tool boundary");
        auto next = messages; next.push_back({{"role","user"},{"content","NEXT_TASK"}});
        const auto next_render = inspect_server_render(next,kind,false,true);
        if (next_render.query_index != 4 || next_render.prompt.find("PLAN") != std::string::npos)
            throw std::runtime_error("completed trajectory reasoning leaked past a new user task");
    }
}

} // namespace

int main() {
    test_non_thinking_prompt_with_system_message();
    test_thinking_prompt_does_not_inject_empty_think_block();
    test_empty_system_message_is_omitted();
    test_server_clock_reasoning_policy();
    std::cout << "qw3 prompt render tests ok\n";
    return 0;
}
