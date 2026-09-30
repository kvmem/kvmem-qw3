#pragma once

#include "multimodal_prefix_cache.hpp"
#include "sha256.hpp"

#include <stdexcept>

namespace qw3::detail {

// Persistent full-transcript identity with a bounded exact rollback suffix.
// The client still sends the full request. Its discarded prefix is verified
// with SHA-256; only the recent suffix can return an exact divergence offset.
// A mismatch/shortening before that suffix safely returns no reusable prefix.
// This is token/visual identity, NOT a substitute for checkpoint/KV validity.
class KvMemPrefixHistory {
public:
    struct Match {
        uint64_t common_prefix = 0;
        bool outside_retained_tail = false;
    private:
        friend class KvMemPrefixHistory;
        const KvMemPrefixHistory *owner = nullptr;
        uint64_t revision = 0;
        const uint32_t *tokens = nullptr;
        size_t token_count = 0;
        const MultimodalTokenIdentity *identities = nullptr;
        size_t identity_count = 0;
        uint64_t fingerprint = 0;
        bool prefix_verified = false;
    };

    uint64_t size() const { return end_; }
    uint64_t tail_begin() const { return end_ - tail_.size(); }
    size_t retained_tokens() const { return tail_.size(); }
    size_t capacity_bytes() const { return tail_.capacity() * sizeof(Row); }
    bool empty() const { return end_ == 0; }

    // `tokens` and `identities` must remain unchanged until assign consumes
    // this proof. They are immutable request-local vectors in the backend.
    Match match(const std::vector<uint32_t> &tokens,
                const std::vector<MultimodalTokenIdentity> &identities,
                uint64_t fingerprint) const {
        Match result;
        result.owner = this;
        result.revision = revision_;
        result.tokens = tokens.data();
        result.token_count = tokens.size();
        result.identities = identities.data();
        result.identity_count = identities.size();
        result.fingerprint = fingerprint;
        const uint64_t begin = tail_begin();
        if (tokens.size() < begin) {
            result.outside_retained_tail = true;
            return result;
        }
        Sha256 prefix;
        for (uint64_t i = 0; i < begin; ++i) {
            const Row row = make_row(tokens[i], identities, fingerprint);
            if (!row.valid) {
                result.outside_retained_tail = true;
                return result;
            }
            hash_row(prefix, row);
        }
        if (prefix.finish_hex() != prefix_digest_) {
            result.outside_retained_tail = true;
            return result;
        }
        result.prefix_verified = true;
        result.common_prefix = begin;
        const uint64_t limit = std::min<uint64_t>(tokens.size(), end_);
        while (result.common_prefix < limit) {
            const uint64_t i = result.common_prefix;
            if (!equal(tail_[i - begin], make_row(tokens[i], identities, fingerprint))) break;
            ++result.common_prefix;
        }
        return result;
    }

    // Capture only model-committed rows. An emitted but unforwarded last token
    // is deliberately excluded. Construction is transactional on allocation
    // failure. A valid match proof avoids hashing the long prefix twice.
    void assign(const std::vector<uint32_t> &prompt,
                const std::vector<uint32_t> &emitted, uint64_t committed,
                const std::vector<MultimodalTokenIdentity> &identities,
                uint64_t fingerprint, size_t keep_tokens,
                const Match *proof = nullptr) {
        if (!keep_tokens || committed > prompt.size() + uint64_t(emitted.size()))
            throw std::invalid_argument("KVMem prefix capture exceeds committed request or has zero tail capacity");
        const uint64_t begin = committed > keep_tokens ? committed - keep_tokens : 0;
        Sha256 prefix;
        uint64_t hashed = 0;
        if (proof && proof->owner == this && proof->revision == revision_ &&
            proof->prefix_verified && proof->tokens == prompt.data() &&
            proof->token_count == prompt.size() &&
            proof->identities == identities.data() &&
            proof->identity_count == identities.size() &&
            proof->fingerprint == fingerprint && begin >= tail_begin()) {
            prefix = prefix_state_;
            hashed = tail_begin();
        }
        auto row_at = [&](uint64_t i) {
            return make_row(i < prompt.size() ? prompt[i] : emitted[i - prompt.size()],
                            identities, fingerprint);
        };
        for (uint64_t i = hashed; i < begin; ++i) hash_row(prefix, row_at(i));
        auto digest_state = prefix;
        auto digest = digest_state.finish_hex();
        std::vector<Row> tail;
        tail.reserve(static_cast<size_t>(committed - begin));
        for (uint64_t i = begin; i < committed; ++i) tail.push_back(row_at(i));
        tail_.swap(tail);
        prefix_state_ = prefix;
        prefix_digest_ = std::move(digest);
        end_ = committed;
        ++revision_;
    }

    void clear() {
        std::vector<Row>().swap(tail_);
        prefix_state_ = Sha256{};
        auto empty = prefix_state_;
        prefix_digest_ = empty.finish_hex();
        end_ = 0;
        ++revision_;
    }

    KvMemPrefixHistory() { clear(); }

private:
    struct Row {
        uint32_t token = 0;
        uint32_t source = 0;
        std::array<uint32_t, 3> position{};
        uint64_t identity = 0;
        bool valid = true;
        bool fallback = false;
    };

    static Row make_row(uint32_t token,
                        const std::vector<MultimodalTokenIdentity> &identities,
                        uint64_t fingerprint) {
        Row row;
        row.token = token;
        if (!(token & 0x80000000u)) return row;
        const auto *visual = multimodal_token_identity(identities, token);
        if (!visual) { row.valid = false; return row; }
        row.source = visual->source_token_id;
        row.position = visual->position;
        row.fallback = visual->cache_identity == 0;
        row.identity = row.fallback ? fingerprint : visual->cache_identity;
        return row;
    }

    static bool equal(const Row &a, const Row &b) {
        return a.valid && b.valid && a.token == b.token && a.source == b.source &&
            a.position == b.position && a.identity == b.identity && a.fallback == b.fallback;
    }

    static void hash_row(Sha256 &hash, const Row &row) {
        // Explicit little-endian serialization: no structure padding, native
        // endianness or embedding allocation address enters cache authority.
        uint8_t bytes[31]{};
        size_t n = 0;
        auto add = [&](uint64_t value, unsigned count) {
            for (unsigned i = 0; i < count; ++i) bytes[n++] = uint8_t(value >> (8 * i));
        };
        add(row.token, 4);
        if (row.token & 0x80000000u) {
            add(row.source, 4);
            for (auto value : row.position) add(value, 4);
            add(row.identity, 8);
            add(row.valid, 1);
            add(row.fallback, 1);
        }
        hash.update(bytes, n);
    }

    Sha256 prefix_state_;
    std::string prefix_digest_;
    std::vector<Row> tail_;
    uint64_t end_ = 0;
    uint64_t revision_ = 0;
};

} // namespace qw3::detail
