#include "qw3/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
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
bool ascii_newline(uint8_t c) { return c == '\r' || c == '\n'; }

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
    if (ascii_alpha(static_cast<uint8_t>(cp))) return true;
    if (cp < 0x80) return false;
    // Treat any non-ASCII non-CJK printable codepoint as a letter for the
    // letter-run rule; this is intentionally loose.
    if (cp_is_cjk(cp)) return false;
    if (cp >= 0x80 && cp <= 0xA0) return false;
    return true;
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
    // Qwen / GPT-style pre-tokenizer (simplified port). The exact upstream
    // regex is:
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3}
    //   | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    // We mirror the high-level shape: digits (max 3), CJK runs, letter runs
    // (with optional single leading-space "joiner"), punctuation runs, and
    // whitespace handling, working on UTF-8 codepoints.
    std::vector<std::string> pieces;
    const size_t len = text.size();
    size_t pos = 0;
    while (pos < len) {
        const size_t start = pos;
        const uint8_t c0 = static_cast<uint8_t>(text[pos]);

        // English contractions ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
        if (c0 == '\'' && pos + 1 < len) {
            const char a = text[pos + 1];
            const char b = pos + 2 < len ? text[pos + 2] : '\0';
            if (a == 's' || a == 't' || a == 'm' || a == 'd') {
                pos += 2; pieces.push_back(text.substr(start, pos - start)); continue;
            }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                pos += 3; pieces.push_back(text.substr(start, pos - start)); continue;
            }
        }

        // Digits: up to 3 contiguous decimal digits at a time.
        if (ascii_digit(c0)) {
            int n = 0;
            while (pos < len && ascii_digit(static_cast<uint8_t>(text[pos])) && n < 3) {
                ++pos; ++n;
            }
            pieces.push_back(text.substr(start, pos - start)); continue;
        }

        // Optional joiner + letter run: matches "[^\r\n\p{L}\p{N}]?\p{L}+".
        // This is the rule that turns " world" into a single piece. It must
        // run before any whitespace collapsing.
        {
            size_t cp_end_lk = pos;
            const uint32_t cp_lk = utf8_decode(text, cp_end_lk);
            // Try with the current codepoint as the leading joiner.
            const bool joinerable = !ascii_newline(static_cast<uint8_t>(c0)) &&
                                    !cp_is_letter(cp_lk) &&
                                    !(cp_lk < 0x80 && ascii_digit(static_cast<uint8_t>(cp_lk)));
            if (joinerable && cp_end_lk < len) {
                size_t after = cp_end_lk;
                const uint32_t next_cp = utf8_decode(text, after);
                if (cp_is_letter(next_cp)) {
                    pos = cp_end_lk;
                    while (pos < len) {
                        size_t e2 = pos;
                        const uint32_t cp2 = utf8_decode(text, e2);
                        if (!cp_is_letter(cp2)) break;
                        pos = e2;
                    }
                    pieces.push_back(text.substr(start, pos - start));
                    continue;
                }
            }
            if (cp_is_letter(cp_lk)) {
                pos = cp_end_lk;
                while (pos < len) {
                    size_t e2 = pos;
                    const uint32_t cp2 = utf8_decode(text, e2);
                    if (!cp_is_letter(cp2)) break;
                    pos = e2;
                }
                pieces.push_back(text.substr(start, pos - start));
                continue;
            }
        }

        // CJK runs.
        {
            size_t cp_end = pos;
            const uint32_t cp = utf8_decode(text, cp_end);
            if (cp_is_cjk(cp)) {
                do {
                    pos = cp_end;
                    if (pos >= len) break;
                    size_t next_end = pos;
                    const uint32_t next_cp = utf8_decode(text, next_end);
                    if (!cp_is_cjk(next_cp)) break;
                    cp_end = next_end;
                } while (pos < len);
                pieces.push_back(text.substr(start, pos - start));
                continue;
            }
        }

        // Whitespace handling: pull newlines as their own run; otherwise
        // emit the whole whitespace block (after consuming the optional
        // leading-space joiner above didn't apply, this is a pure-space
        // tail that does not precede a letter, so it stays as-is).
        if (ascii_space(c0)) {
            size_t p = pos;
            size_t last_nl_end = 0;
            while (p < len && ascii_space(static_cast<uint8_t>(text[p]))) {
                if (ascii_newline(static_cast<uint8_t>(text[p]))) last_nl_end = p + 1;
                ++p;
            }
            if (last_nl_end > 0) {
                pos = last_nl_end;
            } else {
                pos = p;
            }
            pieces.push_back(text.substr(start, pos - start));
            continue;
        }

        // Punctuation / symbol run (optional single leading space).
        {
            size_t punct_pos = pos;
            if (c0 == ' ') {
                size_t e = punct_pos; utf8_decode(text, e); punct_pos = e;
            }
            const size_t punct_run_start = punct_pos;
            while (punct_pos < len) {
                size_t e2 = punct_pos;
                const uint32_t cp2 = utf8_decode(text, e2);
                if (cp_is_letter(cp2) || cp_is_space(cp2) || cp_is_cjk(cp2) ||
                    (cp2 < 0x80 && ascii_digit(static_cast<uint8_t>(cp2)))) break;
                punct_pos = e2;
            }
            if (punct_pos > punct_run_start) {
                while (punct_pos < len && ascii_newline(static_cast<uint8_t>(text[punct_pos]))) ++punct_pos;
                pos = punct_pos;
                pieces.push_back(text.substr(start, pos - start));
                continue;
            }
        }

        // Fallback: consume one codepoint.
        size_t cp_end = pos;
        utf8_decode(text, cp_end);
        pos = cp_end > pos ? cp_end : pos + 1;
        pieces.push_back(text.substr(start, pos - start));
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
