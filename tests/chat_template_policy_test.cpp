#include "server.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "chat_template_policy_test: " << message << "\n";
    std::exit(1);
}

void expect_block(bool preserve_thinking, size_t message_index,
                  size_t last_query_index, const std::string &expected) {
    std::string prompt;
    qw3::detail::append_historical_thinking(
        prompt, "reasoning", preserve_thinking, message_index,
        last_query_index);
    if (prompt != expected) {
        fail("historical thinking policy mismatch");
    }
}

} // namespace

int main() {
    const std::string block = "<think>\nreasoning\n</think>\n\n";

    // A prior assistant turn is omitted under the default Qwen3.6 policy.
    expect_block(false, 1, 2, "");

    // preserve_thinking retains that same turn after a newer user message.
    expect_block(true, 1, 2, block);

    // Thinking inside the current user/tool loop remains preserved by default.
    expect_block(false, 3, 2, block);

    std::cout << "chat_template_policy_test: ok\n";
    return 0;
}
