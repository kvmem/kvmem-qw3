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
    // Expected pieces from the model's serialized Split regex, including its
    // combining-mark rule. These cover failures hidden by block/locale tests.
    expect_pieces("但丁的《神曲》中，《地狱》是第一部分。",
        {"但丁的", "《神曲", "》中", "，《", "地狱", "》是第一部分", "。"});
    expect_pieces("ＡＢ，ＣＤ！", {"ＡＢ", "，ＣＤ", "！"});
    expect_pieces("カタカナ・テスト", {"カタカナ", "・テスト"});
    expect_pieces("١٢३４²Ⅳ½", {"١", "٢", "३", "４", "²", "Ⅳ", "½"});
    expect_pieces("किताब किताब", {"किताब", " किताब"});
    expect_pieces("שָׁלוֹם", {"שָׁלוֹם"});
    expect_pieces("a\u0085\u0085b", {"a", "\u0085", "\u0085b"});
    expect_pieces("a\u202f\u202fb", {"a", "\u202f", "\u202fb"});
    expect_pieces("a\u3000\u3000b", {"a", "\u3000", "\u3000b"});
    expect_pieces("\u0301a!", {"\u0301a", "!"});
    expect_pieces("a\ufe0fb", {"a\ufe0fb"});
    expect_pieces("𝒜𝟜𝟝!", {"𝒜", "𝟜", "𝟝", "!"});
    expect_pieces("'ſuffix", {"'ſ", "uffix"});

    using qw3::detail::qwen_normalize_utf8;
    for (const auto &item : std::vector<std::pair<std::string, std::string>>{
            {"cafe\u0301", "café"}, {"A\u030a", "Å"}, {"\u212b", "Å"},
            {"\u1100\u1161\u11a8", "각"}, {"q\u0315\u0300", "q\u0300\u0315"},
            {"\ufb01", "\ufb01"}, {"Ａ", "Ａ"},
            {std::string("e\u0301\0x", 5), std::string("é\0x", 4)}}) {
        if (qwen_normalize_utf8(item.first) != item.second)
            throw std::runtime_error("NFC normalization mismatch");
    }
    bool invalid = false;
    try { (void)qwen_normalize_utf8("\xc0\xaf"); }
    catch (const std::runtime_error &) { invalid = true; }
    if (!invalid) throw std::runtime_error("invalid UTF-8 accepted by NFC");

    std::cout << "qw3 tokenizer pre-tokenizer ok\n";
    return 0;
}
