#pragma once

#include "json.hpp"
#include <cstddef>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace qw3::detail {

// Exact request-message cache, independent of KV state. Only raw messages are
// cached; system/tools/sampling fields on the current request remain current.
// Missing or evicted handles must fail BEFORE inference so the caller can
// resend its full messages without duplicating a generation.
class MessagePrefixCache {
public:
    using Json = nlohmann::json;
    explicit MessagePrefixCache(size_t bytes = 512u * 1024u * 1024u,
                                size_t entries = 2)
        : byte_limit_(bytes), entry_limit_(entries) {}

    enum class Restore { Ok, Missing, Invalid };

    Restore restore(const std::string &id, size_t count,
                    const Json &suffix, Json &messages) {
        if (!suffix.is_array()) return Restore::Invalid;
        std::lock_guard<std::mutex> lock(mu_);
        const auto found = index_.find(id);
        if (found == index_.end()) return Restore::Missing;
        auto entry = found->second;
        if (count > entry->messages.size()) return Restore::Invalid;
        Json restored = Json::array();
        auto &array = restored.get_ref<Json::array_t &>();
        array.reserve(count + suffix.size());
        array.insert(array.end(), entry->messages.begin(),
                     entry->messages.begin() + count);
        array.insert(array.end(), suffix.begin(), suffix.end());
        lru_.splice(lru_.begin(), lru_, entry);
        messages = std::move(restored);
        return Restore::Ok;
    }

    // Returns false instead of acknowledging a handle that cannot be retained.
    bool store(std::string id, Json messages) {
        if (!messages.is_array()) return false;
        const size_t cost = accounted_bytes(messages) + id.size() * 2 + 256;
        if (!entry_limit_ || cost > byte_limit_) return false;
        std::lock_guard<std::mutex> lock(mu_);
        // A handle is immutable; accidental reuse must never rewrite history.
        if (index_.count(id)) return false;
        while (!lru_.empty() &&
               (lru_.size() >= entry_limit_ || bytes_ > byte_limit_ - cost)) {
            bytes_ -= lru_.back().bytes;
            index_.erase(lru_.back().id);
            lru_.pop_back();
        }
        lru_.push_front({std::move(id), std::move(messages), cost});
        index_.emplace(lru_.front().id, lru_.begin());
        bytes_ += cost;
        return true;
    }

    size_t accounted_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return bytes_;
    }

private:
    static size_t accounted_bytes(const Json &value) {
        // Conservative node allowance plus actual payload lengths; no giant
        // JSON reserialization solely for size accounting.
        size_t result = 256;
        if (value.is_string()) result += value.get_ref<const std::string &>().size();
        if (value.is_array() || value.is_object()) {
            for (auto it = value.begin(); it != value.end(); ++it) {
                if (value.is_object()) result += it.key().size() + 128;
                result += accounted_bytes(it.value());
            }
        }
        return result;
    }
    struct Entry { std::string id; Json messages; size_t bytes; };
    size_t byte_limit_, entry_limit_, bytes_ = 0;
    mutable std::mutex mu_;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

} // namespace qw3::detail
