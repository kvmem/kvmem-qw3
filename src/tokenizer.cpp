#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <cwctype>
#include <limits>
#include <locale>
#include <stdexcept>
#include <utility>

namespace qw3 {
namespace {

// UTF-8 helpers operating on byte strings.

uint32_t utf8_decode(const std::string &s, size_t &pos) {
    if (pos >= s.size()) return 0;
    const uint8_t c0 = static_cast<uint8_t>(s[pos]);
    if (c0 < 0x80) { ++pos; return c0; }
    if ((c0 & 0xE0) == 0xC0 && pos + 1 < s.size()) {
        const uint32_t v = ((c0 & 0x1F) << 6) | (static_cast<uint8_t>(s[pos + 1]) & 0x3F);
        pos += 2;
        return v;
    }
    if ((c0 & 0xF0) == 0xE0 && pos + 2 < s.size()) {
        const uint32_t v = ((c0 & 0x0F) << 12)
                         | ((static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 6)
                         |  (static_cast<uint8_t>(s[pos + 2]) & 0x3F);
        pos += 3;
        return v;
    }
    if ((c0 & 0xF8) == 0xF0 && pos + 3 < s.size()) {
        const uint32_t v = ((c0 & 0x07) << 18)
                         | ((static_cast<uint8_t>(s[pos + 1]) & 0x3F) << 12)
                         | ((static_cast<uint8_t>(s[pos + 2]) & 0x3F) << 6)
                         |  (static_cast<uint8_t>(s[pos + 3]) & 0x3F);
        pos += 4;
        return v;
    }
    ++pos;
    return c0;
}

std::string utf8_encode(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// GPT-2 bytes_to_unicode table. Reversible bijection: every byte 0..255 maps to
// a single Unicode codepoint that displays as a normal character (so that the
// vocab strings can be stored as plain UTF-8).
std::vector<uint32_t> gpt2_bytes_to_unicode_table() {
    std::vector<uint32_t> result(256, 0);
    std::vector<uint8_t> bs;
    for (int b = 0x21; b <= 0x7E; ++b) bs.push_back(static_cast<uint8_t>(b));
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(static_cast<uint8_t>(b));
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(static_cast<uint8_t>(b));
    std::vector<uint32_t> cs(bs.begin(), bs.end());
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), static_cast<uint8_t>(b)) == bs.end()) {
            bs.push_back(static_cast<uint8_t>(b));
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) result[bs[i]] = cs[i];
    return result;
}

bool ascii_digit(uint8_t c) { return c >= '0' && c <= '9'; }
bool ascii_alpha(uint8_t c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool ascii_space(uint8_t c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; }
bool cp_is_cjk(uint32_t cp) {
    // CJK Unified, CJK Ext A, Hiragana, Katakana, Hangul, fullwidth forms.
    return (cp >= 0x3040 && cp <= 0x30FF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0xAC00 && cp <= 0xD7AF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xFF00 && cp <= 0xFFEF);
}

bool cp_is_letter(uint32_t cp) {
    if (cp < 0x80) return ascii_alpha(static_cast<uint8_t>(cp));
    if (cp_is_cjk(cp)) return true;
    // C wide-character functions otherwise start in the "C" locale even when
    // LANG/LC_ALL is UTF-8.  Constructing the environment locale explicitly
    // makes accented Latin and other scripts follow Unicode-like alpha
    // classification without mutating the process-global locale.
    static const std::locale utf8_locale("");
    return cp <= static_cast<uint32_t>(WCHAR_MAX) &&
           std::use_facet<std::ctype<wchar_t>>(utf8_locale).is(
               std::ctype_base::alpha, static_cast<wchar_t>(cp));
}

bool cp_is_number(uint32_t cp) {
    if (cp < 0x80) return ascii_digit(static_cast<uint8_t>(cp));
    static const std::locale utf8_locale("");
    return cp <= static_cast<uint32_t>(WCHAR_MAX) &&
           std::use_facet<std::ctype<wchar_t>>(utf8_locale).is(
               std::ctype_base::digit, static_cast<wchar_t>(cp));
}

bool cp_is_mark(uint32_t cp) {
    // The common Unicode combining-mark blocks.  Qwen3.5's regex joins marks
    // to letter runs (\p{L}\p{M}) rather than treating them as punctuation.
    return (cp >= 0x0300 && cp <= 0x036F) ||
           (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F);
}

bool cp_is_space(uint32_t cp) {
    if (cp < 0x80) return ascii_space(static_cast<uint8_t>(cp));
    return cp == 0xA0 || cp == 0x2028 || cp == 0x2029 || (cp >= 0x2000 && cp <= 0x200A);
}

} // namespace

QwenTokenizer::QwenTokenizer(const GgufFile &gguf) {
    const auto &meta = gguf.metadata();
    auto it = meta.find("tokenizer.ggml.tokens");
    if (it == meta.end() || it->second.string_array.empty()) {
        throw std::runtime_error("missing tokenizer.ggml.tokens in GGUF");
    }
    tokens_ = it->second.string_array;
    token_to_id_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        token_to_id_.emplace(tokens_[i], static_cast<int32_t>(i));
    }

    if (const auto m = meta.find("tokenizer.ggml.merges"); m != meta.end()) {
        merge_rank_.reserve(m->second.string_array.size() * 2);
        for (size_t i = 0; i < m->second.string_array.size(); ++i) {
            const std::string &line = m->second.string_array[i];
            const size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            merge_rank_.emplace(std::make_pair(line.substr(0, sp), line.substr(sp + 1)),
                                static_cast<int32_t>(i));
        }
    } else {
        throw std::runtime_error("missing tokenizer.ggml.merges in GGUF (Qwen needs merge ranks)");
    }

    if (const auto b = meta.find("tokenizer.ggml.bos_token_id"); b != meta.end()) {
        bos_id_ = static_cast<int32_t>(b->second.unsigned_value);
    }
    if (const auto e = meta.find("tokenizer.ggml.eos_token_id"); e != meta.end()) {
        eos_id_ = static_cast<int32_t>(e->second.unsigned_value);
    }
    if (const auto ab = meta.find("tokenizer.ggml.add_bos_token"); ab != meta.end()) {
        add_bos_ = ab->second.bool_value;
    }

    build_byte_maps();

    // Pull every Qwen-style special token directly from the vocab.
    // These appear in the chat template; matching them atomically before
    // BPE keeps the rendered prompt aligned with what the model expects.
    static const char *kCandidates[] = {
        "<|im_start|>", "<|im_end|>",
        "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|image_pad|>", "<|video_pad|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<think>", "</think>",
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
    };
    for (const char *s : kCandidates) {
        const auto t = token_to_id_.find(s);
        if (t != token_to_id_.end()) special_.emplace_back(s, t->second);
    }
    // Longest first so the matcher prefers e.g. "<|im_start|>" over "<".
    std::sort(special_.begin(), special_.end(),
              [](const auto &a, const auto &b) { return a.first.size() > b.first.size(); });
}

void QwenTokenizer::build_byte_maps() {
    const std::vector<uint32_t> table = gpt2_bytes_to_unicode_table();
    for (int b = 0; b < 256; ++b) {
        const std::string ch = utf8_encode(table[static_cast<uint8_t>(b)]);
        byte_to_char_[static_cast<uint8_t>(b)] = ch;
        char_to_byte_[ch] = static_cast<uint8_t>(b);
    }
}

std::string QwenTokenizer::bytes_to_chars(const std::string &bytes) const {
    std::string out;
    out.reserve(bytes.size());
    for (unsigned char b : bytes) out += byte_to_char_.at(b);
    return out;
}

std::vector<std::string> QwenTokenizer::pre_tokenize(const std::string &text) const {
    // Exact Qwen3.5 pre-tokenizer control flow, matching llama.cpp's optimized
    // implementation of:
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)
    //   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    //   | \p{N}
    //   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    //   | \s*[\r\n]+ | \s+(?!\S) | \s+
    //
    // Two details are important for long JSON/tool traces: Qwen3.5 isolates
    // every numeric codepoint (not 1-3 digit groups), and a whitespace run
    // before more text leaves its final space for the next token.  The old
    // approximation got both wrong and drifted by 10-16% on AgentLongBench.
    struct Codepoint {
        uint32_t value;
        size_t byte_begin;
        size_t byte_end;
    };
    std::vector<Codepoint> cpts;
    cpts.reserve(text.size());
    for (size_t byte_pos = 0; byte_pos < text.size();) {
        const size_t begin = byte_pos;
        const uint32_t cp = utf8_decode(text, byte_pos);
        cpts.push_back({cp, begin, byte_pos});
    }

    std::vector<std::string> pieces;
    pieces.reserve(cpts.size());
    const size_t len = cpts.size();
    size_t pos = 0;
    auto cp = [&](size_t i) -> uint32_t {
        return i < len ? cpts[i].value : 0xFFFFFFFFu;
    };
    auto is_letter_or_mark = [&](size_t i) -> bool {
        return i < len && (cp_is_letter(cp(i)) || cp_is_mark(cp(i)));
    };
    auto is_number = [&](size_t i) -> bool {
        return i < len && cp_is_number(cp(i));
    };
    auto is_space = [&](size_t i) -> bool {
        return i < len && cp_is_space(cp(i));
    };
    auto add_piece = [&](size_t begin, size_t end) {
        if (end <= begin) return;
        const size_t byte_begin = cpts[begin].byte_begin;
        const size_t byte_end = cpts[end - 1].byte_end;
        pieces.push_back(text.substr(byte_begin, byte_end - byte_begin));
    };

    while (pos < len) {
        const size_t start = pos;
        const uint32_t c0 = cp(pos);

        // English contractions, case-insensitive.
        if (c0 == '\'' && pos + 1 < len && cp(pos + 1) < 0x80) {
            const char a = static_cast<char>(
                std::tolower(static_cast<unsigned char>(cp(pos + 1))));
            const char b = pos + 2 < len && cp(pos + 2) < 0x80
                ? static_cast<char>(
                      std::tolower(static_cast<unsigned char>(cp(pos + 2))))
                : '\0';
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                pos += 2;
                add_piece(start, pos);
                continue;
            }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                pos += 3;
                add_piece(start, pos);
                continue;
            }
        }

        // Optional non-newline/non-letter/non-number joiner plus letters/marks.
        if (c0 != '\r' && c0 != '\n' && !is_number(pos) &&
            (is_letter_or_mark(pos) || is_letter_or_mark(pos + 1))) {
            ++pos;
            while (is_letter_or_mark(pos)) ++pos;
            add_piece(start, pos);
            continue;
        }

        // Qwen3.5 isolates every numeric codepoint.
        if (is_number(pos)) {
            ++pos;
            add_piece(start, pos);
            continue;
        }

        // Optional literal space plus a punctuation/symbol run and trailing
        // CR/LF.  Out-of-range is excluded explicitly.
        size_t punct = c0 == ' ' ? pos + 1 : pos;
        if (punct < len && !is_space(punct) && !is_letter_or_mark(punct) &&
            !is_number(punct)) {
            pos = punct;
            while (pos < len && !is_space(pos) && !is_letter_or_mark(pos) &&
                   !is_number(pos)) {
                ++pos;
            }
            while (pos < len && (cp(pos) == '\r' || cp(pos) == '\n')) ++pos;
            add_piece(start, pos);
            continue;
        }

        // Whitespace alternatives, in regex order.
        size_t whitespace = 0;
        size_t last_newline_end = 0;
        while (is_space(pos + whitespace)) {
            const uint32_t w = cp(pos + whitespace);
            if (w == '\r' || w == '\n') {
                last_newline_end = pos + whitespace + 1;
            }
            ++whitespace;
        }
        if (last_newline_end > 0) {
            pos = last_newline_end;
            add_piece(start, pos);
            continue;
        }
        // \s+(?!\S): before more text, emit all but the final whitespace so
        // that the final character can become the next token's joiner.
        if (whitespace > 1 && pos + whitespace < len) {
            pos += whitespace - 1;
            add_piece(start, pos);
            continue;
        }
        if (whitespace > 0) {
            pos += whitespace;
            add_piece(start, pos);
            continue;
        }

        // Defensive fallback for an unclassified codepoint.
        ++pos;
        add_piece(start, pos);
    }
    return pieces;
}

std::vector<int32_t> QwenTokenizer::bpe_piece(const std::string &piece) const {
    if (piece.empty()) return {};

    // Step 1: byte-encode the piece into the GPT-2 visible alphabet, then
    // split into 1-codepoint symbols. Each symbol is the UTF-8 form of one
    // remapped byte.
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char b : piece) symbols.push_back(byte_to_char_.at(b));

    // Step 2: iteratively apply the lowest-rank merge until no more merges
    // apply (standard BPE algorithm).
    while (symbols.size() > 1) {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_idx = std::string::npos;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto it = merge_rank_.find({symbols[i], symbols[i + 1]});
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }
        if (best_idx == std::string::npos) break;
        symbols[best_idx] += symbols[best_idx + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_idx) + 1);
    }

    // Step 3: look up the resulting symbols in the vocab.
    std::vector<int32_t> out;
    out.reserve(symbols.size());
    for (const std::string &sym : symbols) {
        const auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            out.push_back(it->second);
        } else {
            // Should be unreachable for byte-level BPE: every single-byte
            // symbol must be in the vocab. If it isn't, emit each byte
            // individually as a safety net.
            for (char c : sym) {
                std::string single(1, c);
                const auto it1 = token_to_id_.find(single);
                if (it1 != token_to_id_.end()) out.push_back(it1->second);
            }
        }
    }
    return out;
}

std::vector<int32_t> QwenTokenizer::encode(const std::string &text, bool add_bos_override) const {
    std::vector<int32_t> out;
    if ((add_bos_override || add_bos_) && bos_id_ < vocab_size()) out.push_back(bos_id_);

    size_t i = 0;
    while (i < text.size()) {
        // Try to match a special token first (longest match).
        bool matched = false;
        for (const auto &sp : special_) {
            if (text.compare(i, sp.first.size(), sp.first) == 0) {
                out.push_back(sp.second);
                i += sp.first.size();
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Find the next special token (or end of string), tokenize the
        // intervening text normally.
        size_t end = text.size();
        for (const auto &sp : special_) {
            const size_t pos = text.find(sp.first, i);
            if (pos != std::string::npos && pos < end) end = pos;
        }
        const std::string chunk = text.substr(i, end - i);
        for (const std::string &piece : pre_tokenize(chunk)) {
            const std::vector<int32_t> ids = bpe_piece(piece);
            out.insert(out.end(), ids.begin(), ids.end());
        }
        i = end;
    }
    return out;
}

std::string QwenTokenizer::decode_one(int32_t id) const {
    if (id < 0 || id >= vocab_size()) return {};
    const std::string &s = tokens_[id];
    // Byte-decode: walk UTF-8 codepoints and reverse the byte<->char map.
    std::string out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t end = pos;
        utf8_decode(s, end);
        const std::string ch = s.substr(pos, end - pos);
        const auto it = char_to_byte_.find(ch);
        if (it != char_to_byte_.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            // Non-byte token (e.g. a special token printed by name).
            out.append(s.begin() + pos, s.begin() + end);
        }
        pos = end;
    }
    return out;
}

std::string QwenTokenizer::decode(const std::vector<int32_t> &ids) const {
    std::string out;
    for (int32_t id : ids) out += decode_one(id);
    return out;
}

} // namespace qw3
