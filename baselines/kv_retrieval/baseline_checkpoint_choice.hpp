#pragma once
#include <cstdint>

namespace qw3::baseline {
struct CheckpointCandidate {
    uint32_t position;
    uint64_t generation;
    bool ready;
};
struct CheckpointChoice {
    uint32_t position = 0;
    bool postselection = false;
};

// Choose among actually captured states. Index/MTP completeness must still be
// finalized and checked by the caller; this never synthesizes or rewinds state.
constexpr CheckpointChoice choose_latest_aligned_checkpoint(
        CheckpointCandidate prompt, CheckpointCandidate postselection,
        uint32_t committed_end, uint64_t generation, uint32_t block_tokens) {
    const uint32_t bt = block_tokens ? block_tokens : 1;
    auto eligible = [&](CheckpointCandidate c) {
        return c.ready && c.position > 0 && c.position <= committed_end &&
               c.position % bt == 0 && c.generation == generation;
    };
    CheckpointChoice choice;
    if (eligible(prompt)) choice.position = prompt.position;
    if (eligible(postselection) && postselection.position > choice.position)
        choice = {postselection.position, true};
    return choice;
}
} // namespace qw3::baseline
