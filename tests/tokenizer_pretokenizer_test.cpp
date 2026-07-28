#include "tokenizer_internal.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect_pieces(const std::string &text,
                   const std::vector<std::string> &expected) {
    const std::vector<std::string> actual =
        qw3::detail::qwen_pre_tokenize(text);
    if (actual != expected) {
        throw std::runtime_error("pre-tokenizer piece mismatch");
    }
}

} // namespace

int main() {
    expect_pieces(" alpha", {" alpha"});
    expect_pieces("  alpha", {" ", " alpha"});
    expect_pieces("    alpha", {"   ", " alpha"});
    expect_pieces("    20", {"   ", " ", "2", "0"});
    expect_pieces(" (alpha)", {" (", "alpha", ")"});
    expect_pieces("a  +  b", {"a", " ", " +", " ", " b"});
    expect_pieces("\t\talpha", {"\t", "\talpha"});
    expect_pieces("alpha\n    beta", {"alpha", "\n", "   ", " beta"});
    expect_pieces("123", {"1", "2", "3"});
    expect_pieces("'RE", {"'RE"});

    std::cout << "qw3 tokenizer pre-tokenizer ok\n";
    return 0;
}
