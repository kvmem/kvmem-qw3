#pragma once

#include "kvmem_prefix_history.hpp"
#include "kvmem_query_history.hpp"
#include "qwen_executor.hpp"
#include <array>
#include <vector>

namespace qw3::detail {

// Token identity follows accepted target-model transitions. Publication can
// stop in the middle of a verified batch, so emitted text is not its authority.
class KvmemAcceptedOutput {
public:
    explicit KvmemAcceptedOutput(uint64_t input_end) : input_end_(input_end) {}
    void record(uint64_t begin, const uint32_t *tokens, size_t count) {
        if (begin < input_end_ || begin - input_end_ != tokens_.size())
            throw std::logic_error("query history output has a noncontiguous accepted prefix");
        if (count) tokens_.insert(tokens_.end(), tokens, tokens + count);
    }
    const std::vector<uint32_t> &tokens() const { return tokens_; }
private:
    uint64_t input_end_;
    std::vector<uint32_t> tokens_;
};

// Logical query evidence survives a physical KV rebuild. The identity belongs
// to the last accepted input, independently of reusable GPU checkpoints.
struct KvmemQueryHistoryState {
    using Positions = std::vector<std::array<uint32_t, 3>>;
    struct PreparedInput {
        KvMemPrefixHistory identity;
        Positions positions;
        uint64_t input_end = 0;
        uint64_t verified_prefix = 0;
    };

    KvMemQueryHistory scores;
    KvMemPrefixHistory identity;
    Positions positions;

    PreparedInput prepare_input(const std::vector<uint32_t> &input,
            const std::vector<MultimodalTokenIdentity> &images,
            const Positions &input_positions, uint64_t fingerprint) const {
        if (input.empty())
            throw std::invalid_argument("query history requires a nonempty logical input");
        if (!input_positions.empty() && input_positions.size() != input.size())
            throw std::invalid_argument("query history requires complete input position identity");
        PreparedInput next;
        next.input_end = input.size();
        next.verified_prefix = identity.match(input, images, fingerprint).common_prefix;
        if (input_positions.empty() != positions.empty()) {
            next.verified_prefix = 0;
        } else if (!input_positions.empty()) {
            next.verified_prefix = std::min<uint64_t>(next.verified_prefix, positions.size());
            for (uint64_t i = 0; i < next.verified_prefix; ++i) {
                if (input_positions[i] != positions[i]) { next.verified_prefix = i; break; }
            }
        }
        // Prepare allocation before score commit, so identity publication is a
        // move and an allocation failure cannot separate scores from proof.
        next.identity.assign(input, {}, input.size(), images, fingerprint, input.size());
        next.positions = input_positions;
        return next;
    }

    void accept_input(PreparedInput &&next) {
        identity = std::move(next.identity);
        positions = std::move(next.positions);
    }
};

// Metadata and checkpoint leases for ONE executor lineage. This must travel
// with that executor when a completed conversation is parked between turns.
// It is never selected through a process-global "current session" pointer:
// suspended controllers retain explicit references to their own state.
struct KvmemPrefixState {
    struct ReplayCheckpoint {
        QwenExecutor::StateSnapshot state;
        bool resumable = false;
        bool source_index_ready = false;
    };

    std::vector<uint32_t> log;
    KvMemPrefixHistory history;
    KvmemQueryHistoryState query_history;
    KvMemPrefixHistory::Match match;
    uint64_t input_embedding_fingerprint = 0;
    std::vector<MultimodalTokenIdentity> input_embedding_identities;
    QwenExecutor::StateSnapshot end_checkpoint;
    QwenExecutor::StateSnapshot prompt_checkpoint;
    std::array<ReplayCheckpoint, 3> replay_checkpoints;
    size_t replay_checkpoint_next = 0;
    uint64_t prompt_position = 0;
    uint64_t end_position = 0;
    bool prompt_resumable = false;
    bool source_index_ready = false;
    bool end_resumable = false;
    bool end_source_index_ready = false;
    bool query_stashed = false;
    uint64_t query_begin = 0;
    uint64_t query_end = 0;
    uint64_t query_source_end = 0;
    bool valid = false;

    KvmemPrefixState() = default;
    KvmemPrefixState(const KvmemPrefixState &) = delete;
    KvmemPrefixState &operator=(const KvmemPrefixState &) = delete;
    KvmemPrefixState(KvmemPrefixState &&) = default;
    KvmemPrefixState &operator=(KvmemPrefixState &&) = default;
};

} // namespace qw3::detail
