#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace qw3::detail {

// Visual preparation assigns consecutive IDs beginning at 0x80000001. Keep
// that common mapping implicit, while retaining arbitrary-ID and duplicate
// handling for library callers, edited histories and restored sessions.
class InputEmbeddingOverrideIndex {
public:
    static constexpr size_t missing = std::numeric_limits<size_t>::max();

    void clear() {
        sparse_.clear();
        size_ = 0;
        dense_ = true;
    }

    bool append(uint32_t token_id) {
        if (dense_ && uint64_t(token_id) == first_id + size_) {
            ++size_;
            return true;
        }
        if (dense_) {
            sparse_.reserve(size_ + 1);
            for (size_t i = 0; i < size_; ++i) {
                sparse_.emplace(static_cast<uint32_t>(first_id + i), i);
            }
            dense_ = false;
        }
        const bool inserted = sparse_.emplace(token_id, size_).second;
        if (inserted) ++size_;
        return inserted;
    }

    size_t find(uint32_t token_id) const {
        if (dense_) {
            const uint64_t id = token_id;
            return id >= first_id && id - first_id < size_
                ? static_cast<size_t>(id - first_id) : missing;
        }
        const auto it = sparse_.find(token_id);
        return it == sparse_.end() ? missing : it->second;
    }

private:
    static constexpr uint64_t first_id = 0x80000001ULL;
    size_t size_ = 0;
    bool dense_ = true;
    std::unordered_map<uint32_t, size_t> sparse_;
};

} // namespace qw3::detail
