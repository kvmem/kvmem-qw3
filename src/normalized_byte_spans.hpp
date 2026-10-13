#pragma once

#include "tokenizer_internal.hpp"
#ifndef UTF8PROC_STATIC
#define UTF8PROC_STATIC
#endif
#include "utf8proc/utf8proc.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace qw3::detail {

struct TextByteSpan { size_t begin, end; };
struct NormalizedByteSpans {
    std::string text;
    std::vector<TextByteSpan> spans;
    size_t rounded_boundaries = 0;
};

// Map original UTF-8 byte ranges into NFC text. Most requests need no mapping.
// If normalization changes a grapheme containing a boundary, include that
// whole grapheme; splitting an e+accent into disjoint normalized bytes would
// otherwise lose or assign its composed character to the wrong semantic group.
// Grapheme boundaries are normalization-safe, including Hangul L/V/T sequences.
inline NormalizedByteSpans normalize_byte_spans(
        const std::string &text, const std::vector<TextByteSpan> &spans) {
    NormalizedByteSpans out{qwen_normalize_utf8(text), spans, 0};
    for (const auto &span : spans)
        if (span.begin > span.end || span.end > text.size())
            throw std::invalid_argument("semantic byte span is outside its text");
    if (out.text == text || spans.empty()) return out;
    struct Boundary { size_t byte, span; bool end; };
    std::vector<Boundary> boundaries;
    boundaries.reserve(2 * spans.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        boundaries.push_back({spans[i].begin, i, false});
        boundaries.push_back({spans[i].end, i, true});
    }
    std::stable_sort(boundaries.begin(), boundaries.end(), [](const auto &a, const auto &b) { return a.byte < b.byte; });
    size_t next = 0;
    std::string joined;
    joined.reserve(out.text.size());
    auto append_cluster = [&](size_t begin, size_t end) {
        const auto original = text.substr(begin, end - begin);
        const auto normalized = qwen_normalize_utf8(original);
        const size_t mapped_begin = joined.size();
        joined += normalized;
        const size_t mapped_end = joined.size();
        const bool unchanged = original == normalized;
        while (next < boundaries.size() && boundaries[next].byte <= end) {
            const auto &boundary = boundaries[next++];
            size_t value;
            if (boundary.byte == begin) value = mapped_begin;
            else if (boundary.byte == end) value = mapped_end;
            else if (unchanged) value = mapped_begin + boundary.byte - begin;
            else {
                value = boundary.end ? mapped_end : mapped_begin;
                ++out.rounded_boundaries;
            }
            auto &span = out.spans[boundary.span];
            (boundary.end ? span.end : span.begin) = value;
        }
    };
    size_t cluster = 0, cursor = 0;
    utf8proc_int32_t previous = -1, state = 0;
    while (cursor < text.size()) {
        utf8proc_int32_t current;
        const auto bytes = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t *>(text.data() + cursor),
            static_cast<utf8proc_ssize_t>(text.size() - cursor), &current);
        if (bytes <= 0) throw std::invalid_argument("invalid UTF-8 in semantic text");
        if (previous >= 0 && utf8proc_grapheme_break_stateful(previous, current, &state)) {
            append_cluster(cluster, cursor);
            cluster = cursor;
        }
        previous = current;
        cursor += static_cast<size_t>(bytes);
    }
    append_cluster(cluster, cursor);
    if (joined != out.text || next != boundaries.size())
        throw std::runtime_error("NFC semantic boundary mapping did not cover the normalized text");
    // Empty regions must not acquire a character through outward rounding.
    for (size_t i = 0; i < spans.size(); ++i)
        if (spans[i].begin == spans[i].end) out.spans[i].end = out.spans[i].begin;
    return out;
}

} // namespace qw3::detail
