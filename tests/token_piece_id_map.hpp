#pragma once

#include "qw3/tokenizer.hpp"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace qw3::test {
// The native generation callback delivers decode_one(id) once per emitted
// token. If that decoder is injective over the entire vocabulary, its inverse
// identifies the original id exactly. Keep raw bytes, including partial UTF-8;
// never concatenate, normalize, or re-tokenize the callback pieces.
// This helper is unsuitable for a transport that coalesces text callbacks.
class TokenPieceIdMap {
public:
    explicit TokenPieceIdMap(const QwenTokenizer &tokenizer) {
        ids_.reserve(static_cast<size_t>(tokenizer.vocab_size()));
        for (int32_t id = 0; id < tokenizer.vocab_size(); ++id) {
            const auto piece = tokenizer.decode_one(id);
            if (piece.empty() || !ids_.emplace(piece, static_cast<uint32_t>(id)).second)
                throw std::runtime_error("per-token decoder is empty or non-injective; direct token-ID observation required");
        }
    }
    uint32_t recover(const std::string &piece) const {
        const auto found = ids_.find(piece);
        if (found == ids_.end())
            throw std::runtime_error("callback is not one native token piece; cannot infer its token ID");
        return found->second;
    }
    size_t size() const { return ids_.size(); }
private:
    std::unordered_map<std::string, uint32_t> ids_;
};
} // namespace qw3::test
