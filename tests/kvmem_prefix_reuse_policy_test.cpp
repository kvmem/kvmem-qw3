#include "kvmem_prefix_reuse_policy.hpp"

// Native Anthropic tool traffic replays from 2432 while its scoring query is
// [2415,2439). Keeping that replay origin as P caused a next-turn cold prefill.
static_assert(!qw3::detail::kvmem_checkpoint_outside_query(2432, 2415, 2439, 32));
static_assert(qw3::detail::kvmem_checkpoint_outside_query(2400, 2415, 2439, 32));
static_assert(qw3::detail::kvmem_checkpoint_outside_query(2464, 2415, 2439, 32));
#include "multimodal_prefix_cache.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

// Cumulative history positions are independent of bounded physical slots.
static_assert(qw3::detail::kvmem_branch_safe_prompt_checkpoint_split(
    (1ull << 40) + 253053, (1ull << 40) + 253005, 32) == (1ull << 40) + 253024);
static_assert(qw3::detail::kvmem_query_safe_prompt_checkpoint_split(
    (1ull << 40) + 6792, 0, 32, (1ull << 40) + 6550, (1ull << 40) + 6783)
        == (1ull << 40) + 6528);
static_assert(qw3::detail::kvmem_branch_safe_prompt_checkpoint_split(
    UINT64_MAX, UINT64_MAX - 1, 32) == 0);
static_assert(qw3::detail::kvmem_warm_query_snapshot_reusable(
    true, (1ull << 40) + 1, (1ull << 40) + 3, (1ull << 40) + 5,
    (1ull << 40) + 1, (1ull << 40) + 3, (1ull << 40) + 8));

namespace {

using qw3::detail::KvmemPrefixReuseSafety;
using qw3::detail::MultimodalTokenIdentity;
using qw3::detail::kvmem_branch_safe_prompt_checkpoint_split;
using qw3::detail::kvmem_checkpoint_selection_compatible;
using qw3::detail::kvmem_prefix_reuse_safety;
using qw3::detail::kvmem_preserve_warm_history_index;
using qw3::detail::kvmem_query_safe_prompt_checkpoint_split;
using qw3::detail::kvmem_query_replay_capture_ready;
using qw3::detail::kvmem_query_storage_rows;
using qw3::detail::kvmem_post_query_replay_checkpoint_aligned;
using qw3::detail::kvmem_warm_query_snapshot_reusable;
using qw3::detail::kvmem_warm_query_snapshot_coordinates_match;
using qw3::detail::kvmem_warm_query_snapshot_capture_ready;
using qw3::detail::kvmem_warm_query_snapshot_source_end;
using qw3::detail::multimodal_prefix_lcp;
using qw3::detail::multimodal_row_cache_identity;

[[noreturn]] void fail(const std::string &message) {
    std::cerr << "kvmem_prefix_reuse_policy_test: " << message << "\n";
    std::exit(1);
}

void require(bool condition, const std::string &message) {
    if (!condition) fail(message);
}

void test_tool_continuation_falls_back_from_m_to_p() {
    // Regression for an above-budget tool continuation with reselection off:
    // M is unaligned and has no resumable source-index cursor, while aligned P
    // is ready.  M must be rejected and P left admissible.
    constexpr KvmemPrefixReuseSafety safety = kvmem_prefix_reuse_safety(
        /*logical_prompt_tokens=*/235968,
        /*selection_budget=*/229376,
        /*query_capture_active=*/false,
        /*per_token_source_index=*/false,
        /*end_resumable=*/false,
        /*end_source_index_ready=*/false,
        /*prompt_resumable=*/true,
        /*prompt_source_index_ready=*/true);
    static_assert(safety.above_selection_budget);
    static_assert(safety.source_index_required);
    static_assert(safety.block_end_checkpoint);
    static_assert(!safety.block_prompt_checkpoint);
    require(safety.block_end_checkpoint && !safety.block_prompt_checkpoint,
            "unsafe M was not rejected in favor of the resumable P checkpoint");
    require(kvmem_preserve_warm_history_index(
                /*warm_reuse=*/true, /*history_index_capture=*/true),
            "tool continuation did not preserve P's source-index cursor");
}

void test_branch_safe_prompt_checkpoint() {
    // The sampled warm request ended at 235076, but OpenCode's next render
    // shared only 235070 tokens.  Its query replay ceiling is therefore 235040.
    // The old last-boundary policy chose 235072 and guaranteed a MISS.
    constexpr uint32_t split =
        kvmem_branch_safe_prompt_checkpoint_split(235076, 0, 32);
    constexpr uint32_t next_common_prefix = 235070;
    constexpr uint32_t next_query_ceiling =
        (next_common_prefix / 32) * 32;
    static_assert(split == 235040);
    static_assert(split <= next_query_ceiling);
    require(split == 235040 && split <= next_query_ceiling,
            "P checkpoint was not moved before the rewritten branch tail");

    // A warm continuation may itself start near the end.  Keep the same split
    // when it still advances beyond the restored prefix; otherwise return zero
    // and let the caller use its established fallback.
    static_assert(kvmem_branch_safe_prompt_checkpoint_split(
                      235076, 235000, 32) == 235040);
    static_assert(kvmem_branch_safe_prompt_checkpoint_split(
                      235076, 235040, 32) == 235040);
    // DeepSWE Scriggo request #171 resumed at M=253005 and ended at 253053.
    // The conservative split (252992) is behind M, but the appended suffix
    // crosses 253024. Preserve that aligned P; exact LCP still protects a
    // rewritten tail, while the next semantic replay gets a legal boundary.
    static_assert(kvmem_branch_safe_prompt_checkpoint_split(
                      253053, 253005, 32) == 253024);
    require(kvmem_branch_safe_prompt_checkpoint_split(
                253053, 253005, 32) == 253024,
            "short warm append discarded its reachable aligned P checkpoint");
    // No block boundary is reachable in this shorter suffix, so prompt-end
    // fallback remains necessary; semantic checkpoint admission rejects it.
    static_assert(kvmem_branch_safe_prompt_checkpoint_split(
                      253020, 253005, 32) == 0);
    static_assert(kvmem_branch_safe_prompt_checkpoint_split(16, 0, 32) == 12);
}

void test_query_safe_prompt_checkpoint() {
    // OSWorld's first request is below budget: preparation still needs the
    // complete Q or a checkpoint before Q. A 16-token response then tail
    // rewrite leaves D=6803 < M=6807, so P must be 6528, not 6752 inside Q.
    static_assert(kvmem_query_safe_prompt_checkpoint_split(
                      6792, 0, 32, 6550, 6783) == 6528);

    // Real OpenCode tool continuation from the 64K guided DeepSWE pilot.  The
    // ordinary branch-tail split selected 240224, which is inside the current
    // user query [240171,240279).  That overwrote the reusable P=240160 and
    // forced the very next tool-result turn into a cold 64K prefill.
    constexpr uint32_t cold = kvmem_query_safe_prompt_checkpoint_split(
        240279, 0, 32, 240171, 240279);
    constexpr uint32_t warm = kvmem_query_safe_prompt_checkpoint_split(
        240279, 240160, 32, 240171, 240279);
    static_assert(cold == 240160);
    static_assert(warm == 240160);
    require(cold == 240160 && warm == 240160,
            "query-interior P was not clamped to the replay boundary");

    // A checkpoint at or after the complete query is eligible through the
    // durable query snapshot and must not be moved backwards.  No-query turns
    // retain the existing branch-tail policy byte-for-byte.
    static_assert(kvmem_query_safe_prompt_checkpoint_split(
                      240400, 0, 32, 240171, 240200) == 240352);
    static_assert(kvmem_query_safe_prompt_checkpoint_split(
                      235076, 0, 32, 0, 0) == 235040);
}

void test_guided_query_snapshot_lifetime() {
    // The private query was generated from the prompt ending at 292979.  The
    // following tool continuation shares through 293060, so the detached Q is
    // reusable even though its row count differs from the original public
    // query [235296,235441).
    static_assert(kvmem_warm_query_snapshot_reusable(
        true, 235296, 235441, 292979,
        235296, 235441, 293060));
    require(kvmem_warm_query_snapshot_reusable(
                true, 235296, 235441, 292979,
                235296, 235441, 293060),
            "guided query snapshot was not reusable on a stable tool turn");

    // Divergence before the prompt that produced the private query, or a new
    // public task span, must invalidate it.
    static_assert(!kvmem_warm_query_snapshot_reusable(
        true, 235296, 235441, 292979,
        235296, 235441, 292900));
    static_assert(kvmem_warm_query_snapshot_coordinates_match(
        true, 235296, 235441, 292979,
        235296, 235441));
    static_assert(!kvmem_warm_query_snapshot_coordinates_match(
        true, 235296, 235441, 292979,
        235297, 235441));
    static_assert(!kvmem_warm_query_snapshot_reusable(
        true, 235296, 235441, 292979,
        293100, 293200, 293300));

    // The next tool request restores a 135-row private Q while the public
    // query span is 145 rows. It remains a complete scorer input and must be
    // re-stashed even though it can never satisfy the public row-count test.
    static_assert(kvmem_warm_query_snapshot_capture_ready(
        false, true, true, false));
    static_assert(!kvmem_warm_query_snapshot_capture_ready(
        false, false, true, false));
    static_assert(kvmem_warm_query_snapshot_source_end(
        false, true, 296701, 293409, 235441) == 293409);
    require(kvmem_warm_query_snapshot_capture_ready(
                false, true, true, false),
            "restored guided Q was dropped after one tool continuation");

    static_assert(kvmem_query_replay_capture_ready(
        true, true, false));
    static_assert(!kvmem_query_replay_capture_ready(
        true, false, false));
    static_assert(!kvmem_query_replay_capture_ready(
        false, true, false));
    require(kvmem_query_replay_capture_ready(true, true, false),
            "ready inherited guided Q was rejected by public row-count check");

    // Both directions occur in real harness traces. The first probe generated
    // a 37-row private query from a 591-row public task; Termenv generated a
    // 190-row private query from a 145-row public task. Capacity follows the
    // larger representation while scoring still uses the actual captured rows.
    static_assert(kvmem_query_storage_rows(591, 37) == 591);
    static_assert(kvmem_query_storage_rows(145, 190) == 190);
    static_assert(kvmem_query_storage_rows(0, 0) == 1);
    require(kvmem_query_storage_rows(145, 190) == 190,
            "longer private query did not enlarge the restore layout");

    // The real probe produced M=130340 and P=130304 with block32. M is a
    // strict prefix but not a legal replay boundary; P must win.
    static_assert(!kvmem_post_query_replay_checkpoint_aligned(
        true, 130340, 120512, 32));
    static_assert(kvmem_post_query_replay_checkpoint_aligned(
        true, 130304, 120512, 32));
    static_assert(kvmem_post_query_replay_checkpoint_aligned(
        true, 120512, 120512, 32));
}

void test_cold_and_dense_controls() {
    constexpr KvmemPrefixReuseSafety dense = kvmem_prefix_reuse_safety(
        220000, 229376, false, false,
        /*end_resumable=*/false, /*end_source_index_ready=*/false,
        /*prompt_resumable=*/false, /*prompt_source_index_ready=*/false);
    static_assert(!dense.block_end_checkpoint);
    static_assert(!dense.block_prompt_checkpoint);

    constexpr KvmemPrefixReuseSafety both_unsafe =
        kvmem_prefix_reuse_safety(
            235968, 229376, false, false,
            false, false, false, false);
    static_assert(both_unsafe.block_end_checkpoint);
    static_assert(both_unsafe.block_prompt_checkpoint);

    require(!kvmem_preserve_warm_history_index(false, true),
            "cold history capture tried to preserve a stale index");
    require(!kvmem_preserve_warm_history_index(true, false),
            "non-capturing dense continuation needlessly preserved an index");
}

void test_query_prepare_and_per_token_guards() {
    // Query preparation below budget retains the pre-existing strict guard.
    constexpr KvmemPrefixReuseSafety prepare = kvmem_prefix_reuse_safety(
        220000, 229376, true, false,
        false, false, true, true);
    static_assert(prepare.block_end_checkpoint);
    static_assert(!prepare.block_prompt_checkpoint);

    // Raw per-token layouts cannot be extended above budget, even if stale
    // checkpoint metadata happens to say resumable/ready.
    constexpr KvmemPrefixReuseSafety per_token = kvmem_prefix_reuse_safety(
        235968, 229376, false, true,
        true, true, true, true);
    static_assert(per_token.source_index_layout_incompatible);
    static_assert(per_token.block_end_checkpoint);
    static_assert(per_token.block_prompt_checkpoint);
}

void test_middecode_selection_invalidates_old_prompt_checkpoint() {
    // Happy DOM regression: P was captured while generation 7 was still dense;
    // a guided mid-decode selection assembled generation 8.  Falling back to P
    // on the next tool turn mixed generation-7 pages with generation-8 window
    // metadata and eventually hit "page table too small".
    static_assert(!kvmem_checkpoint_selection_compatible(7, 8));
    static_assert(kvmem_checkpoint_selection_compatible(8, 8));
    require(!kvmem_checkpoint_selection_compatible(7, 8),
            "pre-selection P remained admissible after guided reselection");
    require(kvmem_checkpoint_selection_compatible(8, 8),
            "turn-end M from the current selection was rejected");

    // Happy DOM ended its first guided refresh at M=193242 (block32), only six
    // tokens before the next aligned boundary.  The following tool callback is
    // prepare-only: it must resume this exact current-generation M and append
    // its suffix, not reject M and cold-prefill the complete 193K history.
    // A real semantic replay still requires an aligned checkpoint.
    static_assert(kvmem_post_query_replay_checkpoint_aligned(
        /*semantic_replay=*/false, 193242, 22848, 32));
    static_assert(!kvmem_post_query_replay_checkpoint_aligned(
        /*semantic_replay=*/true, 193242, 22848, 32));
    require(kvmem_post_query_replay_checkpoint_aligned(
                /*semantic_replay=*/false, 193242, 22848, 32),
            "prepare-only tool continuation rejected exact unaligned M");
    require(!kvmem_post_query_replay_checkpoint_aligned(
                /*semantic_replay=*/true, 193242, 22848, 32),
            "semantic query replay accepted an unaligned M boundary");
}

MultimodalTokenIdentity visual_identity(uint32_t row, uint64_t image,
                                        uint32_t position) {
    return MultimodalTokenIdentity{
        0x80000000u | (row + 1), 151655,
        {position, position, position},
        multimodal_row_cache_identity(image, row)};
}

void test_multimodal_prefix_identity() {
    const std::vector<uint32_t> tokens = {
        10, 0x80000001u, 0x80000002u, 11,
        0x80000003u, 0x80000004u, 12};
    const std::vector<MultimodalTokenIdentity> warm = {
        visual_identity(0, 101, 1), visual_identity(1, 101, 2),
        visual_identity(2, 202, 4), visual_identity(3, 202, 5)};

    // Appending a new image changes the whole-request fingerprint but not the
    // identities of visual rows already present in the reusable prefix.
    std::vector<MultimodalTokenIdentity> appended = warm;
    appended.push_back(visual_identity(4, 303, 7));
    require(multimodal_prefix_lcp(tokens, 7, warm, appended, 1001, 1002) == 7,
            "appended image invalidated an unchanged visual prefix");

    // A changed source image with the same synthetic token layout must shorten
    // the LCP at the first affected visual row.
    std::vector<MultimodalTokenIdentity> changed = warm;
    changed[2] = visual_identity(2, 999, 4);
    require(multimodal_prefix_lcp(tokens, 7, warm, changed, 1001, 1003) == 4,
            "changed image payload reused stale language-model KV");

    // M-RoPE coordinates are part of the cache identity even when pixels match.
    std::vector<MultimodalTokenIdentity> moved = warm;
    moved[1].position[2] += 1;
    require(multimodal_prefix_lcp(tokens, 7, warm, moved, 1001, 1001) == 2,
            "changed visual position reused stale language-model KV");

    // Legacy callers without row identities keep the prior whole-fingerprint
    // behavior, while text before the first image remains reusable.
    std::vector<MultimodalTokenIdentity> legacy = warm;
    for (auto &item : legacy) item.cache_identity = 0;
    require(multimodal_prefix_lcp(tokens, 7, legacy, legacy, 7, 8) == 1,
            "legacy multimodal caller bypassed whole-request fingerprint");
    require(multimodal_prefix_lcp(tokens, 1, legacy, legacy, 7, 8) == 1,
            "image change invalidated an earlier text-only prefix");
}

}  // namespace

void test_warm_periodic_selection_preserves_replay_origin() {
    using qw3::detail::kvmem_periodic_reselect_boundary_due;
    using qw3::detail::kvmem_periodic_reselect_align_drafts;
    // Retained live trace: prompt158994, periodic selections159062/159128,
    // P159104 at generation57 invalidated by generation58, then EOS159158.
    // The candidate instead selects at159104 and retains that aligned P.
    require(kvmem_periodic_reselect_boundary_due(159104, 158994, 64, 128),
            "retained trace did not reach its first due aligned selection");
    require(!kvmem_periodic_reselect_boundary_due(159128, 159104, 64, 128),
            "late partial-block selection invalidated the replay origin");
    require(!kvmem_periodic_reselect_boundary_due(159158, 159104, 64, 128),
            "EOS triggered an unaligned selection");
    for (uint32_t bt : {1u, 32u, 128u}) {
        for (uint32_t interval : {1u, 64u, 255u, 256u}) {
            for (uint32_t offset = 0; offset < bt; ++offset) {
                for (uint32_t acceptance = 1; acceptance <= 8; ++acceptance) {
                    uint64_t pos = (1ull << 40) + offset;
                    uint64_t last = pos;
                    const uint64_t end = pos + 1024;
                    while (pos < end) {
                        const uint32_t drafts = qw3::detail::kvmem_checkpoint_draft_limit(
                            pos, bt, 7, kvmem_periodic_reselect_align_drafts(pos, last, interval, bt, 7));
                        const uint32_t count = acceptance < drafts + 1 ? acceptance : drafts + 1;
                        pos += count;
                        if (kvmem_periodic_reselect_boundary_due(pos, last, interval, bt)) {
                            require(pos % bt == 0, "periodic selection is not aligned");
                            require(pos - last >= interval, "periodic selection happened early");
                            require(pos - last < uint64_t(interval) + bt,
                                    "speculation skipped a due selection boundary");
                            last = pos;
                        }
                    }
                    require(pos - last < uint64_t(interval) + bt,
                            "periodic selection was postponed indefinitely");
                }
            }
        }
    }
    require(!kvmem_periodic_reselect_align_drafts(10, 11, 64, 128, 7),
            "invalid reversed progress enabled a boundary clamp");
}

int main() {
    test_warm_periodic_selection_preserves_replay_origin();
    // Every possible MTP acceptance length reaches an aligned state before
    // crossing it, including the final current-token-only transition.
    for (uint64_t start = 1071316; start < 1071348; ++start) {
        uint64_t pos = start;
        const uint64_t boundary = (start / 32 + 1) * 32;
        while (pos < boundary) {
            const auto drafts = qw3::detail::kvmem_checkpoint_draft_limit(pos, 32, 4, true);
            require(pos + drafts + 1 <= boundary, "MTP skipped the replay checkpoint boundary");
            pos += drafts + 1;
        }
        require(pos == boundary, "post-selection checkpoint boundary was not reached");
    }
    require(qw3::detail::kvmem_checkpoint_draft_limit(31, 32, 4, false) == 4,
            "ordinary MTP decode was changed without a due checkpoint");
    require(qw3::detail::kvmem_checkpoint_draft_limit((1ull << 40) + 31, 32, 4, true) == 0,
            "checkpoint scheduling truncated the logical history position");
    test_tool_continuation_falls_back_from_m_to_p();
    test_branch_safe_prompt_checkpoint();
    test_query_safe_prompt_checkpoint();
    test_guided_query_snapshot_lifetime();
    test_cold_and_dense_controls();
    test_query_prepare_and_per_token_guards();
    test_middecode_selection_invalidates_old_prompt_checkpoint();
    test_multimodal_prefix_identity();
    std::cout << "kvmem_prefix_reuse_policy_test: PASS\n";
    return 0;
}

// Native Anthropic text observation: live message header begins at 1022,
// scoring body at 1025, M=1016. Replay needs 992, not the body's 1024.
static_assert(qw3::detail::kvmem_replay_capture_begin(1025, 1022, 1125) / 32 * 32 == 992);
static_assert(qw3::detail::kvmem_replay_capture_begin(1025, 0, 0) == 1025);
static_assert(qw3::detail::kvmem_replay_capture_begin(1025, 2048, 2100) == 1025);

static_assert(qw3::detail::kvmem_query_safe_prompt_checkpoint_split(582, 0, 32, 31, 573) == 576);
static_assert(qw3::detail::kvmem_query_safe_prompt_checkpoint_split(582, 0, 32, 31, 580) == 0);
static_assert(qw3::detail::kvmem_complete_query_replay_boundary(2441, 2415, 2439, 128) == 2304);
static_assert(qw3::detail::kvmem_complete_query_replay_boundary(2574, 2415, 2439, 128) == 2560);
static_assert(qw3::detail::kvmem_complete_query_replay_boundary(1019, 1025, 1080, 32) == 992);

// The Windows long-history failure restored P=127744 but the current tool
// result begins around 253K. Replaying everything after P incorrectly pins
// about 189K tokens against a 128K window. Advance only the replay checkpoint,
// without truncating any ingested source or relaxing the complete-Q rule.
static_assert(qw3::detail::kvmem_cached_query_replay_boundary(
    253799, 1882, 1972, 128, 127744) == 253696);
static_assert(316760 - qw3::detail::kvmem_cached_query_replay_boundary(
    253799, 1882, 1972, 128, 127744) < 65536);
static_assert(qw3::detail::kvmem_cached_query_replay_boundary(
    2441, 2415, 2439, 128, 2048) == 2304);
static_assert(qw3::detail::kvmem_cached_query_replay_boundary(
    1882, 1882, 1972, 128, 127744) == 127744);
static_assert(qw3::detail::kvmem_cached_query_replay_boundary(
    1019, 1025, 1080, 32, 1024) == 1024);
static_assert(qw3::detail::kvmem_cached_query_replay_boundary(
    (1ull << 40) + 500, 1882, 1972, 128, (1ull << 40)) ==
    (1ull << 40) + 384);

// DSH evidence request 44 planned 1006 mandatory blocks against A=1152.
// Reusing old P=139744 as the replay origin pinned 1171 blocks and overflowed
// A, even though the current live suffix starts at 145088. Cached Q must let
// prefill advance the replay snapshot to that current boundary.
static_assert(qw3::detail::kvmem_cached_query_replay_boundary(
    145088, 41769, 41784, 32, 139744) == 145088);
static_assert((145814 - qw3::detail::kvmem_cached_query_replay_boundary(
    145088, 41769, 41784, 32, 139744) + 31) / 32 == 23);
static_assert((145814 - 139744 + 31) / 32 == 190);
