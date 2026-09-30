#pragma once

#include "kvmem_prefix_history.hpp"
#include "qwen_executor.hpp"
#include <array>
#include <vector>

namespace qw3::detail {

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
