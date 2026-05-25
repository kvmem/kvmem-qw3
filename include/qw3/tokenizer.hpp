#pragma once

#include "qw3/gguf.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qw3 {

/* Qwen35 / GPT-2 style byte-level BPE tokenizer.
 *
 * Built directly from GGUF metadata: `tokenizer.ggml.tokens`,
 * `tokenizer.ggml.merges`, plus the bos/eos/add_bos hints. Special tokens
 * such as <|im_start|> / <|im_end|> / <think> / </think> are matched as
 * atomic units before BPE runs.
 *
 * Implementation notes:
 *  - GPT-2's byte->unicode mapping is applied so that raw UTF-8 bytes are
 *    representable as printable characters that appear in the vocab.
 *  - The pre-tokenizer is the Qwen3 / GPT-4-ish split: digit runs, CJK,
 *    letters with optional leading space, punctuation runs, whitespace.
 *    Not a perfect port of the upstream Rust tokenizers regex; it is the
 *    minimum that matches single chat templates well enough to compare
 *    logits against llama.cpp on short prompts. This will be tightened in
 *    follow-up work if/when a long-prompt token-stream diff misses. */
class QwenTokenizer {
public:
    explicit QwenTokenizer(const GgufFile &gguf);

    std::vector<int32_t> encode(const std::string &text, bool add_bos = false) const;
    std::string decode(const std::vector<int32_t> &ids) const;
    std::string decode_one(int32_t id) const;

    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    bool add_bos_default() const { return add_bos_; }
    int32_t vocab_size() const { return static_cast<int32_t>(tokens_.size()); }

    // Translate text bytes into the BPE-encoded character form (visible vocab
    // alphabet). Useful for debugging and for the dump-logits diff.
    std::string bytes_to_chars(const std::string &bytes) const;

private:
    void build_byte_maps();
    std::vector<std::string> pre_tokenize(const std::string &text) const;
    std::vector<int32_t> bpe_piece(const std::string &piece) const;

    // size_t pair hasher
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string> &p) const noexcept {
            return std::hash<std::string>()(p.first) * 1000003ull
                 ^ std::hash<std::string>()(p.second);
        }
    };

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> merge_rank_;

    // GPT-2 byte<->unicode maps. char_to_byte_ inverts byte_to_char_.
    std::unordered_map<uint8_t, std::string> byte_to_char_;
    std::unordered_map<std::string, uint8_t> char_to_byte_;

    std::vector<std::pair<std::string, int32_t>> special_; // matched longest-first

    int32_t bos_id_ = 0;
    int32_t eos_id_ = 0;
    bool add_bos_ = false;
};

} // namespace qw3
