#pragma once

// Preserve the serial refresh boundary while speculating inside an open tool
// call or code fence. Boundary candidates still participate in p/q acceptance.
#include "kvmem_refresh_policy.hpp"

#include <string>

namespace qw3::detail {

struct DFlashDeferredRefreshLimit {
    uint32_t drafts;
    bool needs_text_boundary_check;
};

inline DFlashDeferredRefreshLimit dflash_deferred_refresh_limit(
        uint32_t requested, uint32_t decoded, uint32_t epoch_begin,
        uint32_t trigger, uint32_t refreshes, uint32_t max_refreshes,
        uint32_t epoch_limit, bool text_safe, uint32_t guard = 512) {
    const uint32_t ordinary = kvmem_middecode_draft_limit(
        requested, decoded, epoch_begin, trigger, refreshes, max_refreshes);
    if (!kvmem_middecode_refresh_due(decoded, epoch_begin, trigger, refreshes, max_refreshes) || text_safe)
        return {ordinary, false};
    const uint32_t elapsed = decoded - epoch_begin;
    if (!epoch_limit || kvmem_middecode_emergency_refresh_due(elapsed, epoch_limit, guard))
        return {0, false};
    // Include the verifier's extra output token. An emergency refresh must
    // still observe exactly the same first eligible token as serial decode.
    const uint32_t before_emergency = epoch_limit - guard - elapsed;
    return {std::min(requested, before_emergency - 1), true};
}

struct DFlashRefreshBoundary {
    uint32_t candidates;
    bool includes_boundary;
};

// Keep the first closing candidate INSIDE p/q verification. Dropping it
// because of its content and drawing afresh would condition the proposal.
// On full acceptance the caller must leave this candidate pending and omit
// the usual bonus token, restoring the target/draft state before that token.
// Rejection keeps the ordinary residual token and discards later candidates.
template <typename Piece>
DFlashRefreshBoundary dflash_refresh_boundary(
        std::string_view generated, uint32_t candidates, Piece piece) {
    std::string preview(generated);
    for (uint32_t index = 0; index < candidates; ++index) {
        preview += piece(index);
        if (kvmem_middecode_refresh_text_safe(preview)) return {index + 1, true};
    }
    return {candidates, false};
}

}  // namespace qw3::detail
