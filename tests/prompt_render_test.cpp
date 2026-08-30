#include "qw3/qw3.hpp"

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

} // namespace

int main() {
    test_non_thinking_prompt_with_system_message();
    test_thinking_prompt_does_not_inject_empty_think_block();
    test_empty_system_message_is_omitted();
    std::cout << "qw3 prompt render tests ok\n";
    return 0;
}
