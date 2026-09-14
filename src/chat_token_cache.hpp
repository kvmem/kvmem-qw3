#pragma once

#include "qw3/tokenizer.hpp"
#include <cstddef>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace qw3::detail {

// Cache only independently tokenizable spans separated by a recognized atomic
// special token. Splitting arbitrary UTF-8/BPE byte prefixes is NOT safe.
class ChatTokenCache {
public:
    explicit ChatTokenCache(const QwenTokenizer &tokenizer,
                            size_t bytes = 64u * 1024u * 1024u)
        : tokenizer_(tokenizer), limit_(bytes) {
        const auto marker = tokenizer_.encode(boundary_);
        enabled_ = !tokenizer_.add_bos_default() && marker.size() == 1 &&
            marker[0] == tokenizer_.token_id(boundary_);
    }

    std::vector<int32_t> encode(const std::string &text) {
        if (!enabled_ || !limit_ || text.size() < 1024)
            return tokenizer_.encode(text);
        std::vector<int32_t> result;
        size_t begin = 0;
        while (begin < text.size()) {
            size_t end = text.find(boundary_, begin + 1);
            if (end == std::string::npos) end = text.size();
            append(text.substr(begin, end - begin), result);
            begin = end;
        }
        return result;
    }

    void prepare_decoded_pieces() {
        std::call_once(decoded_once_, [&]() {
            decoded_pieces_.reserve(tokenizer_.vocab_size());
            for (int32_t id = 0; id < tokenizer_.vocab_size(); ++id)
                decoded_pieces_.push_back(tokenizer_.decode_one(id));
        });
    }

    const std::string &decode_one(int32_t id) {
        prepare_decoded_pieces();
        static const std::string empty;
        return id >= 0 && static_cast<size_t>(id) < decoded_pieces_.size()
            ? decoded_pieces_[id] : empty;
    }

private:
    void append(const std::string &text, std::vector<int32_t> &output) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            const auto found = index_.find(text);
            if (found != index_.end()) {
                const auto entry = found->second;
                output.insert(output.end(), entry->tokens.begin(), entry->tokens.end());
                lru_.splice(lru_.begin(), lru_, entry);
                return;
            }
        }
        auto tokens = tokenizer_.encode(text);
        output.insert(output.end(), tokens.begin(), tokens.end());
        const size_t cost = text.size() * 2 + tokens.size() * sizeof(int32_t) + 256;
        if (cost > limit_) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (index_.count(text)) return;
        while (!lru_.empty() && bytes_ > limit_ - cost) {
            bytes_ -= lru_.back().bytes;
            index_.erase(lru_.back().text);
            lru_.pop_back();
        }
        lru_.push_front({text, std::move(tokens), cost});
        index_.emplace(lru_.front().text, lru_.begin());
        bytes_ += cost;
    }
    struct Entry { std::string text; std::vector<int32_t> tokens; size_t bytes; };
    const QwenTokenizer &tokenizer_;
    const std::string boundary_ = "<|im_start|>";
    size_t limit_, bytes_ = 0;
    bool enabled_ = false;
    std::mutex mu_;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    // Immutable vocabulary-sized table. Harness span mapping repeatedly walks
    // every historical token; byte-decoding it again can dominate CPU time.
    std::once_flag decoded_once_;
    std::vector<std::string> decoded_pieces_;
};

} // namespace qw3::detail
