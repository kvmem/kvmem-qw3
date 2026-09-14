#pragma once

#include "json.hpp"
#include "qw3/qw3.hpp"
#include "native_request_json.hpp"
#include <list>
#include <mutex>
#include <unordered_map>

namespace qw3::detail {

inline size_t prepared_json_bytes(const nlohmann::json &value) {
    size_t bytes = 256;
    if (value.is_string()) bytes += value.get_ref<const std::string &>().size();
    if (value.is_array() || value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (value.is_object()) bytes += it.key().size() + 128;
            bytes += prepared_json_bytes(it.value());
        }
    }
    return bytes;
}

// These records retain CPU metadata only. Feature tensors stay in the bounded
// vision cache and are resolved afresh for each in-flight request.
struct PreparedMessage {
    nlohmann::json raw;
    nlohmann::json rendered;
    std::vector<VisionImageSourcePtr> images;
    std::vector<VisionEncoding::Grid> grids;
    uint64_t image_hash_state = VisionImageSource::fingerprint_seed;
    size_t bytes = 0;
    void account() {
        bytes = sizeof(*this) + prepared_json_bytes(raw) + prepared_json_bytes(rendered);
        for (const auto &source : images) {
            bytes += sizeof(VisionImageSource) + source->image().media_type.size() +
                     source->image().base64_data.size() + sizeof(VisionEncoding::Grid) + 64;
        }
    }
};

struct VisionTokenMapping {
    std::vector<int32_t> raw_tokens, mapped_tokens;
    std::vector<std::array<uint32_t, 3>> positions;
    std::vector<uint64_t> row_identities;
    std::vector<std::pair<uint32_t, uint32_t>> spans;
    std::vector<VisionImageSourcePtr> images;
    std::vector<VisionEncoding::Grid> grids;
    int32_t image_pad = -1, vision_start = -1, vision_end = -1;
    uint32_t dim = 0;
    size_t bytes() const {
        return sizeof(*this) + (raw_tokens.capacity() + mapped_tokens.capacity()) * sizeof(int32_t) +
            positions.capacity() * sizeof(positions[0]) + row_identities.capacity() * sizeof(uint64_t) +
            spans.capacity() * sizeof(spans[0]) + images.capacity() * sizeof(VisionImageSourcePtr) +
            grids.capacity() * sizeof(VisionEncoding::Grid);
    }
};

class PreparedMessageCache {
public:
    struct Entry {
        std::vector<std::shared_ptr<const PreparedMessage>> messages;
        std::shared_ptr<const VisionTokenMapping> mapping;
        std::shared_ptr<const NativeMessageSnapshot> native;
    };
    enum class Restore { Ok, Missing, Invalid };
    explicit PreparedMessageCache(size_t bytes = 512u * 1024u * 1024u, size_t entries = 2)
        : byte_limit_(bytes), entry_limit_(entries) {}
    // Native clients send complete histories. Match exact raw records without
    // requiring a client extension; compare outside the cache lock.
    std::shared_ptr<const Entry> match_prefix(const nlohmann::json &messages,
                                               size_t &count) const {
        std::vector<std::shared_ptr<const Entry>> snapshots;
        {
            std::lock_guard<std::mutex> lock(mu_);
            for (const auto &node : lru_) snapshots.push_back(node.value);
        }
        count = 0;
        std::shared_ptr<const Entry> best;
        for (const auto &entry : snapshots) {
            if (entry->messages.size() <= count) continue;
            size_t matched = 0;
            while (matched < std::min(messages.size(), entry->messages.size()) &&
                   messages[matched] == entry->messages[matched]->raw) ++matched;
            if (matched > count) { count = matched; best = entry; }
        }
        return best;
    }
    std::shared_ptr<const Entry> latest_native() const {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto &node : lru_) if (node.value->native) return node.value;
        return {};
    }
    Restore restore(const std::string &id, size_t count, std::shared_ptr<const Entry> &result) {
        std::lock_guard<std::mutex> lock(mu_);
        auto found = index_.find(id);
        if (found == index_.end()) return Restore::Missing;
        if (count > found->second->value->messages.size()) return Restore::Invalid;
        lru_.splice(lru_.begin(), lru_, found->second);
        result = found->second->value;
        return Restore::Ok;
    }
    bool store(std::string id, std::shared_ptr<const Entry> entry) {
        size_t cost = sizeof(Entry) + id.size() * 2 + 256 +
                      entry->messages.capacity() * sizeof(entry->messages[0]);
        // Deliberately charge shared records to each entry. This conservative
        // bound avoids a global reference-accounting/locking hot path.
        for (const auto &message : entry->messages) cost += message->bytes;
        if (entry->mapping) cost += entry->mapping->bytes();
        if (entry->native) cost += entry->native->bytes();
        if (!entry_limit_ || cost > byte_limit_) return false;
        std::lock_guard<std::mutex> lock(mu_);
        if (index_.count(id)) return false;
        while (!lru_.empty() && (lru_.size() >= entry_limit_ || bytes_ > byte_limit_ - cost)) {
            bytes_ -= lru_.back().bytes;
            index_.erase(lru_.back().id);
            lru_.pop_back();
        }
        lru_.push_front({std::move(id), std::move(entry), cost});
        index_.emplace(lru_.front().id, lru_.begin());
        bytes_ += cost;
        return true;
    }
    size_t accounted_bytes() const {
        std::lock_guard<std::mutex> lock(mu_);
        return bytes_;
    }
private:
    struct Node { std::string id; std::shared_ptr<const Entry> value; size_t bytes; };
    size_t byte_limit_, entry_limit_, bytes_ = 0;
    mutable std::mutex mu_;
    std::list<Node> lru_;
    std::unordered_map<std::string, std::list<Node>::iterator> index_;
};

} // namespace qw3::detail
