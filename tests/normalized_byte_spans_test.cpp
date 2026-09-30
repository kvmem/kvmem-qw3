#include "normalized_byte_spans.hpp"
#include <iostream>

using qw3::detail::TextByteSpan;
using qw3::detail::normalize_byte_spans;

int main() {
    const auto unchanged = normalize_byte_spans("one\ntwo", {{0, 3}, {4, 7}});
    if (unchanged.text != "one\ntwo" || unchanged.spans[1].begin != 4 || unchanged.rounded_boundaries)
        throw std::runtime_error("unchanged text mapping failed");
    const auto words = normalize_byte_spans("cafe\u0301 next", {{0, 6}, {7, 11}});
    if (words.text != "café next" || words.spans[0].end != 5 ||
        words.spans[1].begin != 6 || words.spans[1].end != 10 || words.rounded_boundaries)
        throw std::runtime_error("normalized word boundaries failed");
    const auto split = normalize_byte_spans("e\u0301!", {{0, 1}, {1, 3}, {3, 4}});
    if (split.spans[0].begin != 0 || split.spans[0].end != 2 ||
        split.spans[1].begin != 0 || split.spans[1].end != 2 ||
        split.spans[2].begin != 2 || split.spans[2].end != 3 || split.rounded_boundaries != 2)
        throw std::runtime_error("composed grapheme outward rounding failed");
    const auto empty = normalize_byte_spans("e\u0301!", {{1, 1}, {3, 3}});
    if (empty.spans[0].begin != empty.spans[0].end ||
        empty.spans[1].begin != 2 || empty.spans[1].end != 2)
        throw std::runtime_error("empty span acquired normalized content");
    const auto hangul = normalize_byte_spans("\u1100\u1161\u11a8!", {{0, 3}, {3, 6}, {6, 9}, {9, 10}});
    for (size_t i = 0; i < 3; ++i)
        if (hangul.spans[i].begin != 0 || hangul.spans[i].end != 3)
            throw std::runtime_error("Hangul composition mapping failed");
    if (hangul.spans[3].begin != 3 || hangul.spans[3].end != 4)
        throw std::runtime_error("Hangul following boundary failed");
    const auto reordered = normalize_byte_spans("q\u0315\u0300z", {{1, 3}, {3, 5}, {5, 6}});
    if (reordered.text != "q\u0300\u0315z" || reordered.spans[0].begin != 0 ||
        reordered.spans[0].end != 5 || reordered.spans[2].begin != 5)
        throw std::runtime_error("combining mark reorder mapping failed");
    const std::string long_text = std::string(100000, 'a') + "e\u0301\n" + std::string(100000, 'b');
    const auto long_result = normalize_byte_spans(long_text, {{0, 100004}, {100004, 200004}});
    if (long_result.spans[0].end != 100003 || long_result.spans[1].begin != 100003 ||
        long_result.spans[1].end != 200003)
        throw std::runtime_error("long history offset mapping failed");
    bool invalid = false;
    try { (void)normalize_byte_spans("x", {{0, 2}}); }
    catch (const std::invalid_argument &) { invalid = true; }
    if (!invalid) throw std::runtime_error("out-of-range span was accepted");
    std::cout << "normalized byte spans PASS\n";
}
