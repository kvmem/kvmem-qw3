#include "kvmem_prefix_reuse_policy.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using qw3::detail::KvmemPrefixReuseSafety;
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

}  // namespace

int main() {
    test_tool_continuation_falls_back_from_m_to_p();
    test_branch_safe_prompt_checkpoint();
    test_query_safe_prompt_checkpoint();
    test_guided_query_snapshot_lifetime();
    test_cold_and_dense_controls();
    test_query_prepare_and_per_token_guards();
    test_middecode_selection_invalidates_old_prompt_checkpoint();
    std::cout << "kvmem_prefix_reuse_policy_test: PASS\n";
    return 0;
}
