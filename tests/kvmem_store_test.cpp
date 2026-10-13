// KvMemStore host-logic unit test (block-sparse KV, Task #37).
//
// Pure host logic — no GPU. Covers: block registration (partial-block growth),
// the working-set diff (stage-in/out + window remap plan), and the built-in
// cumulative-attention top-k selection (sink/recent preservation + budget).

#include "qw3/kvmem_store.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace qw3;

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static void test_register_append() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    KvMemStore s(cfg);

    // 100 tokens -> 1 partial block of 100.
    s.register_append(100);
    CHECK(s.block_count() == 1);
    CHECK(s.blocks()[0].n_tokens == 100);
    CHECK(s.blocks()[0].orig_pos_start == 0);

    // +50 -> first block fills to 128, remainder 22 starts a new block.
    s.register_append(50);
    CHECK(s.block_count() == 2);
    CHECK(s.blocks()[0].n_tokens == 128);
    CHECK(s.blocks()[1].n_tokens == 22);
    CHECK(s.blocks()[1].orig_pos_start == 128);

    // +384 (3 full blocks) -> fills block1 to 128 (106 added), then more.
    s.register_append(384);
    // Total = 100+50+384 = 534 = 4 full blocks (512) + 22.
    CHECK(s.blocks()[1].n_tokens == 128);
    CHECK(s.block_count() == 5);
    CHECK(s.blocks()[2].n_tokens == 128);
    CHECK(s.blocks()[3].n_tokens == 128);
    CHECK(s.blocks()[4].n_tokens == 22);
    CHECK(s.blocks()[4].orig_pos_start == 128 * 4);
}

static void test_selection_diff_and_remap() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    KvMemStore s(cfg);
    s.register_append(128 * 5);  // 5 full blocks, ids 0..4
    CHECK(s.block_count() == 5);

    // Select blocks {0, 2, 4} (out of order on purpose).
    auto plan = s.set_selection({4, 0, 2});
    CHECK(plan.stage_in.size() == 3);       // all three newly resident
    CHECK(plan.stage_out.size() == 2);      // cold GPU blocks 1 and 3 leave
    CHECK(plan.stage_out[0] == 1);
    CHECK(plan.stage_out[1] == 3);
    CHECK(plan.remaps.size() == 3);
    CHECK(plan.total_window_tokens == 128 * 3);
    // Window order is ascending block id; packed contiguously from 0.
    // Each block's K is currently baked at its true position (orig_pos_start),
    // so from_base == orig position, to_base == window slot.
    CHECK(plan.remaps[0].block_id == 0);
    CHECK(plan.remaps[0].from_base == 0);
    CHECK(plan.remaps[0].to_base == 0);
    CHECK(plan.remaps[0].skip == true);     // block 0 already at window slot 0
    CHECK(plan.remaps[1].block_id == 2);
    CHECK(plan.remaps[1].from_base == 128 * 2);   // baked at true position
    CHECK(plan.remaps[1].to_base == 128);         // remapped into window slot 1
    CHECK(plan.remaps[1].skip == false);
    CHECK(plan.remaps[2].block_id == 4);
    CHECK(plan.remaps[2].from_base == 128 * 4);
    CHECK(plan.remaps[2].to_base == 128 * 2);
    CHECK(s.blocks()[0].remap_count == 0);
    CHECK(s.blocks()[2].remap_count == 1);
    CHECK(s.blocks()[4].remap_count == 1);
    for (uint32_t id : plan.stage_out) {
        s.set_block_tier(id, KvTier::CPU, static_cast<int32_t>(id), -1);
    }

    // Reselect {0, 2, 3}: block 4 leaves, block 3 enters, 0 and 2 stay.
    // Blocks 0 and 2 are now baked at their window slots (0 and 128), so on
    // reselection they map to the SAME slots and skip; block 3 enters from its
    // true position 128*3 -> window slot 2.
    auto plan2 = s.set_selection({0, 2, 3});
    CHECK(plan2.stage_in.size() == 1 && plan2.stage_in[0] == 3);
    CHECK(plan2.stage_out.size() == 1 && plan2.stage_out[0] == 4);
    CHECK(plan2.remaps.size() == 3);
    CHECK(plan2.remaps[0].block_id == 0 && plan2.remaps[0].skip == true);
    CHECK(plan2.remaps[1].block_id == 2 && plan2.remaps[1].skip == true);
    CHECK(plan2.remaps[2].block_id == 3);
    CHECK(plan2.remaps[2].from_base == 128 * 3);  // baked at true pos, first move
    CHECK(plan2.remaps[2].to_base == 128 * 2);
    CHECK(plan2.remaps[2].skip == false);
    CHECK(plan2.total_window_tokens == 128 * 3);
    CHECK(s.blocks()[2].remap_count == 1);  // stable slot: no new write
    CHECK(s.blocks()[3].remap_count == 1);

    // Tier canonicalization is another fp16 re-RoPE write and must contribute
    // to the diagnostic count even though it is not part of set_selection().
    s.record_block_rerope(3, s.blocks()[3].orig_pos_start);
    CHECK(s.blocks()[3].remap_count == 2);
}

static void test_immutable_source_selection_uses_bounded_delta_remaps() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    cfg.immutable_refresh_remaps = 2;
    cfg.immutable_refresh_abs_delta_tokens = 0;
    KvMemStore s(cfg);
    s.register_append(32 * 6);

    // A resident block uses one cheap in-place delta remap and records its new
    // active GPU frame.
    auto p1 = s.set_selection({0, 4});
    CHECK(p1.remaps.size() == 2);
    CHECK(p1.remaps[1].block_id == 4);
    CHECK(p1.remaps[1].from_base == 128);
    CHECK(p1.remaps[1].to_base == 32);
    CHECK(!p1.remaps[1].skip);
    CHECK(!p1.remaps[1].raw_refresh);
    CHECK(s.blocks()[4].baked_pos == 32);
    CHECK(s.blocks()[4].remap_count == 1);

    // A different selection starts at the current active frame.
    auto p2 = s.set_selection({0, 2, 4});
    CHECK(p2.remaps.size() == 3);
    CHECK(p2.remaps[2].block_id == 4);
    CHECK(p2.remaps[2].from_base == 32);
    CHECK(p2.remaps[2].to_base == 64);
    CHECK(s.blocks()[4].baked_pos == 64);
    CHECK(s.blocks()[4].remap_count == 2);

    // The next move crosses the configured two-write count limit and is
    // rebuilt from raw K, resetting the drift counters.
    auto p3 = s.set_selection({0, 4});
    CHECK(p3.remaps[1].from_base == 64);
    CHECK(p3.remaps[1].to_base == 32);
    CHECK(p3.remaps[1].raw_refresh);
    CHECK(s.blocks()[4].baked_pos == 32);
    CHECK(s.blocks()[4].remap_count == 0);

    // A block loaded from CPU/NVMe is always rebuilt because its GPU K slot is
    // newly allocated and has no valid active bake.
    s.set_block_tier(3, KvTier::CPU, 0, -1);
    auto p4 = s.set_selection({0, 3});
    CHECK(p4.remaps[1].raw_refresh);
    CHECK(!p4.remaps[1].skip);
}

static void test_cold_immutable_same_position_rebuilds_k() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    KvMemStore s(cfg);
    s.register_append(32 * 4);

    // Block 2 was previously compacted to slot 1, then evicted. Its recorded
    // bake equals the next target slot, but its working K no longer resides on
    // GPU. The plan must rebuild K from raw authority instead of skipping.
    (void)s.set_selection({0, 1});
    s.set_block_baked_pos(2, 32);
    s.set_block_tier(2, KvTier::CPU, 0, -1);
    const auto plan = s.set_selection({0, 2});
    CHECK(plan.stage_in.size() == 1);
    CHECK(plan.stage_in[0] == 2);
    CHECK(plan.remaps.size() == 2);
    CHECK(plan.remaps[1].from_base == 32);
    CHECK(plan.remaps[1].to_base == 32);
    CHECK(!plan.remaps[1].working_k_resident);
    CHECK(plan.remaps[1].raw_refresh);
    CHECK(!plan.remaps[1].skip);
}

static void test_fp8_every_move_uses_raw_authority() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32; cfg.immutable_source_k = true;
    cfg.immutable_refresh_abs_delta_tokens = 1;
    KvMemStore store(cfg); store.register_append(32 * 6);
    // Block 4 repeatedly moves between compact slots 1 and 2. No intermediate
    // quantized working K may become the next rotation's numerical authority.
    for (unsigned i = 0; i < 100; ++i) {
        const auto ids = i % 2 ? std::vector<uint32_t>{0, 2, 4} : std::vector<uint32_t>{0, 4};
        const auto moved = store.set_selection(ids);
        const auto &last = moved.remaps.back();
        CHECK(last.block_id == 4 && last.working_k_resident);
        CHECK(!last.skip && last.raw_refresh);
        CHECK(store.blocks()[4].remap_count == 0);
        CHECK(store.blocks()[4].remap_abs_delta == 0);
        const auto same = store.set_selection(ids);
        for (const auto &entry : same.remaps) CHECK(entry.skip && !entry.raw_refresh);
    }
}

static void test_default_profile_reuses_moved_resident_k() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    CHECK(cfg.optimize_stage_out);
    CHECK(cfg.optimize_stage_in);
    CHECK(cfg.optimize_pack);
    KvMemStore s(cfg);
    s.register_append(32 * 4);

    // Block 2 remains physically resident but moves from original position 64
    // to compact position 32. Default profiles use the bounded in-place
    // re-RoPE path and retain raw K as the periodic reset authority.
    const auto plan = s.set_selection({0, 2});
    CHECK(plan.remaps.size() == 2);
    CHECK(plan.remaps[1].working_k_resident);
    CHECK(plan.remaps[1].from_base == 64);
    CHECK(plan.remaps[1].to_base == 32);
    CHECK(!plan.remaps[1].raw_refresh);
    CHECK(!plan.remaps[1].skip);
    CHECK(s.blocks()[2].remap_count == 1);
}

static void test_explicit_force_raw_refresh_rebuilds_stable_selection() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    KvMemStore s(cfg);
    s.register_append(32 * 4);

    (void)s.set_selection({0, 2});
    const auto plan = s.set_selection({0, 2}, /*force_raw_refresh=*/true);
    CHECK(plan.remaps.size() == 2);
    CHECK(plan.stage_in.empty());
    for (const KvMemRemap &rm : plan.remaps) {
        CHECK(rm.working_k_resident);
        CHECK(rm.raw_refresh);
        CHECK(!rm.skip);
    }
}

static void test_stage_in_uses_tier_residency() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;
    KvMemStore s(cfg);
    s.register_append(128 * 4);

    auto plan = s.set_selection({0, 3});
    CHECK(plan.stage_out.size() == 2);
    s.set_block_tier(1, KvTier::CPU, 1, -1);
    s.set_block_tier(2, KvTier::CPU, 2, -1);

    auto plan2 = s.set_selection({0, 1, 3});
    CHECK(plan2.stage_in.size() == 1);
    CHECK(plan2.stage_in[0] == 1);
    CHECK(plan2.stage_out.empty());
}

static void test_stage_in_ablation_forces_full_reload() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.immutable_source_k = true;
    cfg.optimize_stage_in = false;
    KvMemStore s(cfg);
    s.register_append(32 * 5);

    // Even though blocks 0 and 2 remain selected, the ablation deliberately
    // stages every GPU block out and reloads the complete selected set.
    auto first = s.set_selection({0, 2, 4});
    CHECK(first.selection_overlap_blocks == 0);
    CHECK(first.gpu_reused_blocks == 0);
    CHECK(first.stage_out.size() == 5);
    CHECK(first.stage_in.size() == 3);
    for (const KvMemRemap &rm : first.remaps) {
        CHECK(!rm.working_k_resident);
        CHECK(rm.raw_refresh);
        CHECK(!rm.skip);
    }

    // Simulate the executor publishing the selected blocks back on GPU, then
    // reselect the same logical set. Natural overlap is observable, but actual
    // GPU reuse remains zero and all three blocks travel through the tiers.
    for (uint32_t id : first.stage_out) {
        s.set_block_tier(id, KvTier::CPU, static_cast<int32_t>(id), -1);
    }
    for (uint32_t id : std::vector<uint32_t>{0, 2, 4}) {
        s.set_block_tier(id, KvTier::GPU);
    }
    auto second = s.set_selection({0, 2, 4});
    CHECK(second.selection_overlap_blocks == 3);
    CHECK(second.gpu_reused_blocks == 0);
    CHECK(second.retained_position_stable == 3);
    CHECK(second.retained_position_moved == 0);
    CHECK(second.stage_out.size() == 3);
    CHECK(second.stage_in.size() == 3);
    for (const KvMemRemap &rm : second.remaps) {
        CHECK(!rm.working_k_resident);
        CHECK(rm.raw_refresh);
        CHECK(!rm.skip);
    }
}

static void test_topk_budget_sink_recent() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 4;   // budget = 4 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    KvMemStore s(cfg);
    s.register_append(128 * 10);   // 10 blocks, ids 0..9
    CHECK(s.budget_blocks() == 4);

    // Give middle blocks distinct attention scores; make block 5 and 6 hottest.
    std::vector<double> scores(10, 0.0);
    scores[5] = 100.0;
    scores[6] = 90.0;
    scores[3] = 10.0;
    s.accumulate_attn(scores);

    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 4);
    // Must contain sink (0) and recent (9), plus the two hottest middle (5,6).
    bool has0 = false, has9 = false, has5 = false, has6 = false;
    for (uint32_t id : sel) {
        if (id == 0) has0 = true;
        if (id == 9) has9 = true;
        if (id == 5) has5 = true;
        if (id == 6) has6 = true;
    }
    CHECK(has0 && has9 && has5 && has6);
    // Returned ascending.
    for (size_t i = 1; i < sel.size(); ++i) CHECK(sel[i] > sel[i - 1]);
}

static void test_topk_zero_recent_keeps_no_suffix() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 4;   // budget = 4 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;         // literal: no always-kept suffix
    KvMemStore s(cfg);
    s.register_append(128 * 10);   // ids 0..9

    std::vector<double> scores(10, 0.0);
    scores[3] = 80.0;
    scores[5] = 100.0;
    scores[6] = 90.0;
    scores[9] = -100.0;            // tail must not be pinned
    s.set_retrieval_scores(scores);

    const auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 4);
    CHECK(std::find(sel.begin(), sel.end(), 0) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 3) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 5) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 6) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 9) == sel.end());
}

static void test_budget_scaled_keep_allocation() {
    {
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/32,
            /*select_budget=*/32 * 1024,
            /*sink_blocks=*/-1,
            /*recent_blocks=*/-1,
            /*sink_tokens=*/-1,
            /*recent_tokens=*/-1);
        CHECK(keep.sink_target_tokens == 1024);
        CHECK(keep.recent_target_tokens == 4096);
        CHECK(keep.sink_blocks == 32);
        CHECK(keep.recent_blocks == 128);
        CHECK(keep.sink_effective_tokens == 1024);
        CHECK(keep.recent_effective_tokens == 4096);
        CHECK(keep.sink_source == KvMemKeepSource::Auto);
        CHECK(keep.recent_source == KvMemKeepSource::Auto);
    }
    {
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/512,
            /*select_budget=*/224 * 1024,
            -1, -1, -1, -1);
        CHECK(keep.sink_target_tokens == 2048);
        CHECK(keep.recent_target_tokens == 16384);
        CHECK(keep.sink_blocks == 4);
        CHECK(keep.recent_blocks == 32);
    }
    {
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/1024,
            /*select_budget=*/224 * 1024,
            -1, -1, -1, -1);
        CHECK(keep.sink_blocks == 2);
        CHECK(keep.recent_blocks == 16);
    }
    {
        // Explicit block values preserve historical experiments byte-for-byte.
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/512,
            /*select_budget=*/224 * 1024,
            /*sink_blocks=*/8,
            /*recent_blocks=*/0,
            /*sink_tokens=*/-1,
            /*recent_tokens=*/-1);
        CHECK(keep.sink_blocks == 8);
        CHECK(keep.recent_blocks == 0);
        CHECK(keep.sink_effective_tokens == 4096);
        CHECK(keep.recent_effective_tokens == 0);
        CHECK(keep.sink_source == KvMemKeepSource::Blocks);
        CHECK(keep.recent_source == KvMemKeepSource::Blocks);
    }
    {
        // Explicit token values are rounded up independently of block size.
        const auto keep = resolve_kvmem_keep_allocation(
            /*block_tokens=*/512,
            /*select_budget=*/32 * 1024,
            /*sink_blocks=*/-1,
            /*recent_blocks=*/-1,
            /*sink_tokens=*/1025,
            /*recent_tokens=*/4097);
        CHECK(keep.sink_blocks == 3);
        CHECK(keep.recent_blocks == 9);
        CHECK(keep.sink_effective_tokens == 1536);
        CHECK(keep.recent_effective_tokens == 4608);
        CHECK(keep.sink_source == KvMemKeepSource::Tokens);
        CHECK(keep.recent_source == KvMemKeepSource::Tokens);
    }

    bool conflict_threw = false;
    try {
        (void)resolve_kvmem_keep_allocation(
            32, 32 * 1024,
            /*sink_blocks=*/1, /*recent_blocks=*/-1,
            /*sink_tokens=*/1024, /*recent_tokens=*/-1);
    } catch (const std::runtime_error &) {
        conflict_threw = true;
    }
    CHECK(conflict_threw);

    bool budget_threw = false;
    try {
        (void)resolve_kvmem_keep_allocation(
            1024, 32 * 1024,
            /*sink_blocks=*/20, /*recent_blocks=*/20,
            /*sink_tokens=*/-1, /*recent_tokens=*/-1);
    } catch (const std::runtime_error &) {
        budget_threw = true;
    }
    CHECK(budget_threw);
}

static void test_topk_mandatory_blocks_stay_inside_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;  // four blocks including mandatory suffix
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    std::vector<double> scores(10, 0.0);
    scores[2] = 100.0;
    scores[3] = 90.0;
    scores[4] = 80.0;
    // The low-scoring query suffix must replace ordinary top-k candidates,
    // never extend the compact window to six blocks.
    scores[8] = -10.0;
    scores[9] = -20.0;
    s.set_retrieval_scores(scores);
    const auto sel = s.pick_topk_blocks({8, 9});
    CHECK(sel.size() == 4);
    CHECK(std::find(sel.begin(), sel.end(), 0) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 2) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 8) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 9) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 3) == sel.end());
}

static void test_mandatory_overlap_deduplicates_and_recent_is_best_effort() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.sink_blocks = 2;
    cfg.recent_blocks = 3;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    // Mandatory block 0 overlaps the sink and is repeated. The hard union is
    // {0,1,5}, leaving exactly one best-effort recent slot for block 9.
    const auto sel = s.pick_topk_blocks({0, 0, 5});
    CHECK(sel.size() == 4);
    for (uint32_t id : {0u, 1u, 5u, 9u}) {
        CHECK(std::find(sel.begin(), sel.end(), id) != sel.end());
    }
    CHECK(std::find(sel.begin(), sel.end(), 8) == sel.end());
}

static void test_mandatory_hard_union_overflow_throws() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 2;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    bool topk_threw = false;
    try {
        (void)s.pick_topk_blocks({1, 2, 3, 4});
    } catch (const std::runtime_error &) {
        topk_threw = true;
    }
    CHECK(topk_threw);

    bool semantic_threw = false;
    try {
        (void)s.pick_semantic_groups(
            {{32 * 5, 32 * 7}}, {1.0}, {1, 2, 3, 4});
    } catch (const std::runtime_error &) {
        semantic_threw = true;
    }
    CHECK(semantic_threw);
}

static void test_long_agent_query_is_not_an_unbounded_suffix_pin() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 512;  // production 64K selection budget
    cfg.sink_blocks = 0;            // SystemControl below already covers sink
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(128 * 547);   // observed 69,982-token Claude trace

    // Historical behavior: SystemControl occupies [0,172], while treating the
    // score query at block 172 as pin_from_block materializes [172,546]. The
    // deduplicated union is all 547 blocks and must reproduce the overflow.
    std::vector<uint32_t> legacy;
    for (uint32_t id = 0; id < 547; ++id) legacy.push_back(id);
    bool legacy_threw = false;
    try {
        (void)s.pick_topk_blocks(legacy);
    } catch (const std::runtime_error &) {
        legacy_threw = true;
    }
    CHECK(legacy_threw);

    // Fixed behavior: retain exact control [0,172], exact task [172,179], and
    // only the newest eight-block live tool suffix [539,546]. Completed tool
    // history remains a scored candidate and the result stays within 512.
    std::vector<uint32_t> bounded;
    for (uint32_t id = 0; id <= 179; ++id) bounded.push_back(id);
    for (uint32_t id = 539; id <= 546; ++id) bounded.push_back(id);
    std::vector<double> scores(547, 0.0);
    scores[300] = 1000.0;  // completed historical tool evidence
    s.set_retrieval_scores(scores);
    const auto selected = s.pick_topk_blocks(bounded);
    CHECK(selected.size() == 512);
    CHECK(std::find(selected.begin(), selected.end(), 300) != selected.end());
    for (uint32_t id = 539; id <= 546; ++id) {
        CHECK(std::find(selected.begin(), selected.end(), id) != selected.end());
    }
}

static void test_prefill_pressure_mandatory_blocks_stay_inside_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.prefill_budget = 32 * 5;
    cfg.sink_blocks = 1;
    KvMemStore s(cfg);
    s.register_append(32 * 10);

    const auto sel = s.pick_prefill_pressure_blocks({3, 4});
    CHECK(sel.size() == 5);
    CHECK(std::find(sel.begin(), sel.end(), 0) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 3) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 4) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 8) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 9) != sel.end());
    CHECK(std::find(sel.begin(), sel.end(), 7) == sel.end());

    bool budget_threw = false;
    try {
        (void)s.pick_prefill_pressure_blocks({1, 2, 3, 4, 5});
    } catch (const std::runtime_error &) {
        budget_threw = true;
    }
    CHECK(budget_threw);
}

static void test_prefill_pressure_sink_full_recent_tail() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 5;  // budget = 5 blocks
    cfg.sink_blocks = 2;
    cfg.recent_blocks = 0;        // pressure policy deliberately ignores this
    KvMemStore s(cfg);
    s.register_append(128 * 10);  // ids 0..9

    // Make old middle blocks arbitrarily hot. Pressure selection must remain a
    // pure positional sink + newest-tail policy: [0,1] + [7,8,9].
    std::vector<double> scores(10, 0.0);
    scores[2] = 1000.0;
    scores[3] = 999.0;
    scores[4] = 998.0;
    s.set_retrieval_scores(scores);
    const std::vector<uint32_t> expected{0, 1, 7, 8, 9};
    CHECK(s.pick_prefill_pressure_blocks() == expected);

    // A prior semantic working set and a different score vector cannot affect
    // the next pressure choice.
    (void)s.set_selection({0, 1, 2, 3, 4});
    scores.assign(10, 0.0);
    scores[5] = 5000.0;
    scores[6] = 4000.0;
    s.set_retrieval_scores(scores);
    CHECK(s.pick_prefill_pressure_blocks() == expected);

    // `recent_blocks` is a semantic-selector pin, not the pressure tail size.
    cfg.recent_blocks = 4;
    KvMemStore with_semantic_recent(cfg);
    with_semantic_recent.register_append(128 * 10);
    CHECK(with_semantic_recent.pick_prefill_pressure_blocks() == expected);
}

static void test_prefill_pressure_edges() {
    // With no sink, the whole pressure budget is the newest tail.
    {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = 32 * 3;
        cfg.sink_blocks = 0;
        KvMemStore s(cfg);
        s.register_append(32 * 6);
        const std::vector<uint32_t> expected{3, 4, 5};
        CHECK(s.pick_prefill_pressure_blocks() == expected);
    }

    // Sink is clamped by the budget, leaving no tail when it consumes all slots.
    {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = 32 * 3;
        cfg.sink_blocks = 8;
        cfg.recent_blocks = 8;
        KvMemStore s(cfg);
        s.register_append(32 * 7 + 11);  // partial block id 7 included in n
        const std::vector<uint32_t> expected{0, 1, 2};
        CHECK(s.pick_prefill_pressure_blocks() == expected);
    }

    // Below budget remains identity, including the trailing partial block.
    {
        KvMemStoreConfig cfg;
        cfg.block_tokens = 32;
        cfg.select_budget = 32 * 8;
        cfg.sink_blocks = 2;
        KvMemStore s(cfg);
        s.register_append(32 * 3 + 7);
        const std::vector<uint32_t> expected{0, 1, 2, 3};
        CHECK(s.pick_prefill_pressure_blocks() == expected);
    }

    KvMemStoreConfig cfg;
    KvMemStore empty(cfg);
    CHECK(empty.pick_prefill_pressure_blocks().empty());
}

static void test_prefill_pressure_budget_is_independent_from_semantic_budget() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 4;
    cfg.prefill_budget = 32 * 8;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(32 * 12);

    // Pressure construction keeps a wide sink+tail context.
    const std::vector<uint32_t> pressure{0, 5, 6, 7, 8, 9, 10, 11};
    CHECK(s.prefill_budget_blocks() == 8);
    CHECK(s.pick_prefill_pressure_blocks() == pressure);

    // The final semantic selector still contracts to four blocks. The hotter
    // middle blocks win independently of the wider prefill construction frame.
    std::vector<double> scores(12, 0.0);
    scores[2] = 4.0;
    scores[3] = 3.0;
    scores[4] = 2.0;
    s.set_retrieval_scores(scores);
    const std::vector<uint32_t> semantic{0, 2, 3, 4};
    CHECK(s.budget_blocks() == 4);
    CHECK(s.pick_topk_blocks() == semantic);
}

static void test_request_semantic_budget_override_is_scoped_and_bounded() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 8;   // CLI/server maximum.
    cfg.prefill_budget = 32 * 12;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    KvMemStore s(cfg);
    s.register_append(32 * 16);

    std::vector<double> scores(16, 0.0);
    for (uint32_t id = 0; id < scores.size(); ++id) {
        scores[id] = static_cast<double>(id);
    }
    s.set_retrieval_scores(scores);

    CHECK(s.select_budget_tokens() == 32 * 8);
    CHECK(s.runtime_select_budget_tokens() == 0);
    CHECK(s.pick_topk_blocks().size() == 8);
    CHECK(s.pick_prefill_pressure_blocks().size() == 12);

    // A request may narrow semantic selection while pressure prefill remains
    // at the configured 12-block construction window.
    s.set_runtime_select_budget(32 * 4);
    CHECK(s.select_budget_tokens() == 32 * 4);
    CHECK(s.runtime_select_budget_tokens() == 32 * 4);
    CHECK(s.pick_topk_blocks().size() == 4);
    CHECK(s.pick_prefill_pressure_blocks().size() == 12);

    // Zero restores the CLI maximum for the next request.
    s.set_runtime_select_budget(0);
    CHECK(s.select_budget_tokens() == 32 * 8);
    CHECK(s.pick_topk_blocks().size() == 8);

    bool over_max_threw = false;
    try {
        s.set_runtime_select_budget(32 * 9);
    } catch (const std::invalid_argument &) {
        over_max_threw = true;
    }
    CHECK(over_max_threw);

    bool unaligned_threw = false;
    try {
        s.set_runtime_select_budget(32 * 4 + 1);
    } catch (const std::invalid_argument &) {
        unaligned_threw = true;
    }
    CHECK(unaligned_threw);

    bool keep_allocation_threw = false;
    try {
        s.set_runtime_select_budget(32);  // sink + recent need two blocks.
    } catch (const std::invalid_argument &) {
        keep_allocation_threw = true;
    }
    CHECK(keep_allocation_threw);
    CHECK(s.runtime_select_budget_tokens() == 0);
}

static void test_quota_policy_sink_recent_retrieval_profile() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 6;   // budget = 6 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.select_policy = KvMemSelectPolicy::Quota;
    cfg.retrieval_blocks = 2;
    cfg.profile_blocks = 2;
    KvMemStore s(cfg);
    s.register_append(128 * 12);

    std::vector<double> profile(12, 0.0);
    profile[3] = 80.0;
    profile[4] = 70.0;
    s.accumulate_attn(profile);

    std::vector<double> retrieval(12, 0.0);
    retrieval[7] = 100.0;
    retrieval[8] = 90.0;
    s.set_retrieval_scores(retrieval);

    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 6);
    bool has0 = false, has11 = false, has3 = false, has4 = false;
    bool has7 = false, has8 = false;
    for (uint32_t id : sel) {
        if (id == 0) has0 = true;
        if (id == 11) has11 = true;
        if (id == 3) has3 = true;
        if (id == 4) has4 = true;
        if (id == 7) has7 = true;
        if (id == 8) has8 = true;
    }
    CHECK(has0 && has11 && has3 && has4 && has7 && has8);
    for (size_t i = 1; i < sel.size(); ++i) CHECK(sel[i] > sel[i - 1]);
}

static void test_topk_all_fit() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 100;  // huge budget
    KvMemStore s(cfg);
    s.register_append(128 * 3);
    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 3);
    CHECK(sel[0] == 0 && sel[1] == 1 && sel[2] == 2);
}

static void test_topk_empty() {
    KvMemStoreConfig cfg;
    KvMemStore s(cfg);
    CHECK(s.pick_topk_blocks().empty());
    auto plan = s.set_selection({0, 1});  // stale ids on empty store
    CHECK(plan.remaps.empty());
    CHECK(plan.total_window_tokens == 0);
}

static void test_tier_metadata() {
    KvMemStoreConfig cfg;
    KvMemStore s(cfg);
    s.register_append(128 * 2);
    s.set_block_tier(1, KvTier::CPU, 7, -1);
    CHECK(s.blocks()[1].tier == KvTier::CPU);
    CHECK(s.blocks()[1].cpu_slot == 7);
    CHECK(s.blocks()[1].nvme_slot == -1);
    s.set_block_tier(1, KvTier::SSD, -1, 3);
    CHECK(s.blocks()[1].tier == KvTier::SSD);
    CHECK(s.blocks()[1].cpu_slot == -1);
    CHECK(s.blocks()[1].nvme_slot == 3);
    s.set_block_baked_pos(1, 128);
    CHECK(s.blocks()[1].baked_pos == 128);
    s.set_block_baked_pos(1, 0);
    CHECK(s.blocks()[1].baked_pos == 0);
}

static void test_inclusive_tier_metadata() {
    KvMemStoreConfig cfg;
    cfg.optimize_stage_out = true;
    KvMemStore s(cfg);
    s.register_append(128 * 2);

    // First spill leaves both a CPU cache copy and clean SSD backing.
    s.set_block_ssd_backing(1, 3, true);
    s.set_block_tier(1, KvTier::CPU, 7, 3);
    CHECK(s.blocks()[1].tier == KvTier::CPU);
    CHECK(s.blocks()[1].cpu_slot == 7);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);

    // Promotion makes GPU active without discarding either clean copy.
    s.set_block_tier(1, KvTier::GPU);
    CHECK(s.blocks()[1].tier == KvTier::GPU);
    CHECK(s.blocks()[1].cpu_slot == 7);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);

    // A CPU-cache eviction must not change active GPU residency.
    s.set_block_cpu_copy(1, -1);
    CHECK(s.blocks()[1].tier == KvTier::GPU);
    CHECK(s.blocks()[1].cpu_slot == -1);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);
}

static void test_stage_out_off_exclusive_tier_metadata() {
    KvMemStoreConfig cfg;
    cfg.optimize_stage_out = false;
    KvMemStore s(cfg);
    s.register_append(128 * 2);

    // Stage-out-off uses exclusive move semantics. Promotion must drop every
    // lower-tier handle.
    s.set_block_tier(1, KvTier::SSD, -1, 3);
    CHECK(s.blocks()[1].tier == KvTier::SSD);
    CHECK(s.blocks()[1].nvme_slot == 3);
    CHECK(s.blocks()[1].ssd_clean);
    s.set_block_tier(1, KvTier::GPU);
    CHECK(s.blocks()[1].tier == KvTier::GPU);
    CHECK(s.blocks()[1].cpu_slot == -1);
    CHECK(s.blocks()[1].nvme_slot == -1);
    CHECK(!s.blocks()[1].ssd_clean);
}

static void test_factorial_optimization_controls_are_orthogonal() {
    for (bool stage_out : {false, true}) {
        for (bool stage_in : {false, true}) {
            for (bool pack : {false, true}) {
                KvMemStoreConfig cfg;
                cfg.block_tokens = 32;
                cfg.immutable_source_k = true;
                cfg.optimize_stage_out = stage_out;
                cfg.optimize_stage_in = stage_in;
                cfg.optimize_pack = pack;

                KvMemStore selection_store(cfg);
                selection_store.register_append(32 * 3);
                (void) selection_store.set_selection({0, 2});
                const KvMemPlan plan =
                    selection_store.set_selection({0, 2});
                CHECK(plan.gpu_reused_blocks ==
                      (stage_in ? 2u : 0u));
                CHECK(plan.stage_in.size() ==
                      (stage_in ? 0u : 2u));
                CHECK(plan.stage_out.size() ==
                      (stage_in ? 1u : 3u));

                KvMemStore tier_store(cfg);
                tier_store.register_append(32);
                tier_store.set_block_ssd_backing(0, 9, true);
                tier_store.set_block_tier(
                    0, KvTier::SSD, -1, 9);
                tier_store.set_block_tier(
                    0, KvTier::GPU);
                CHECK((tier_store.blocks()[0].nvme_slot >= 0) ==
                      stage_out);
                CHECK(tier_store.blocks()[0].ssd_clean ==
                      stage_out);
            }
        }
    }
}

static void test_truncate_to() {
    KvMemStoreConfig cfg; cfg.block_tokens = 128;

    // (1) truncate on a block boundary: 5 full blocks -> keep exactly 3.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 5);           // ids 0..4, 640 tokens
        auto dropped = s.truncate_to(128 * 3);
        CHECK(s.block_count() == 3);
        CHECK(dropped.size() == 2);           // blocks 4 then 3 popped (back-first)
        CHECK(dropped[0].block_id == 4);
        CHECK(dropped[1].block_id == 3);
        CHECK(s.blocks()[2].n_tokens == 128);
        // Re-registering the suffix restores the original layout.
        s.register_append(128 * 2);
        CHECK(s.block_count() == 5);
        CHECK(s.blocks()[3].orig_pos_start == 128 * 3);
        CHECK(s.blocks()[4].n_tokens == 128);
    }

    // (2) truncate mid-block: shrink the trailing partial block in place.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 3);           // ids 0..2
        auto dropped = s.truncate_to(128 * 2 + 40);
        CHECK(s.block_count() == 3);          // block 2 shrunk, not popped
        CHECK(dropped.empty());
        CHECK(s.blocks()[2].n_tokens == 40);
        CHECK(s.blocks()[2].orig_pos_start == 128 * 2);
        // Growth continues from the shrunk boundary, not the old end.
        s.register_append(10);
        CHECK(s.block_count() == 3);
        CHECK(s.blocks()[2].n_tokens == 50);
    }

    // (3) truncate straddling a partial trailing block: pop the partial, shrink
    // the now-trailing full block.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 2 + 30);      // ids 0,1 full; id 2 partial (30)
        CHECK(s.block_count() == 3);
        auto dropped = s.truncate_to(128 + 50); // keep block0 full, block1 -> 50
        CHECK(s.block_count() == 2);
        CHECK(dropped.size() == 1);
        CHECK(dropped[0].block_id == 2);
        CHECK(s.blocks()[1].n_tokens == 50);
    }

    // (4) no-op cases: token_pos == total (the ckpt_M case) and > total.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 2 + 10);      // 266 tokens
        CHECK(s.truncate_to(266).empty());
        CHECK(s.block_count() == 3);
        CHECK(s.truncate_to(9999).empty());
        CHECK(s.block_count() == 3);
    }

    // (5) truncate to 0: everything dropped, dense indices intact on regrow.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 2);
        auto dropped = s.truncate_to(0);
        CHECK(s.block_count() == 0);
        CHECK(dropped.size() == 2);
        s.register_append(128);
        CHECK(s.block_count() == 1);
        CHECK(s.blocks()[0].block_id == 0);
        CHECK(s.blocks()[0].orig_pos_start == 0);
    }

    // (6) dropped list carries tier slots so the executor can release them.
    {
        KvMemStore s(cfg);
        s.register_append(128 * 3);
        s.set_block_tier(2, KvTier::CPU, 7, -1);
        auto dropped = s.truncate_to(128 * 2);
        CHECK(dropped.size() == 1);
        CHECK(dropped[0].block_id == 2);
        CHECK(dropped[0].tier == KvTier::CPU);
        CHECK(dropped[0].cpu_slot == 7);
    }
}

// DeltaNet-retrieval config surfaces through KvMemStoreConfig with sane defaults
// and that retrieval scores (which the DeltaNet scorer writes via
// set_retrieval_scores) rank blocks exactly like the mean-k path — the store is
// method-agnostic, so a DeltaNet score set feeds pick_topk identically.
static void test_deltanet_config_and_scores() {
    KvMemStoreConfig cfg;
    // Defaults: DeltaNet retrieval off by default; decay on; topk = 4.
    CHECK(cfg.retrieval_method == KvMemRetrievalMethod::MeanK);
    CHECK(cfg.deltanet_decay == true);
    CHECK(cfg.deltanet_topk_q == 4);
    CHECK(cfg.deltanet_topk_h == 4);
    CHECK(cfg.deltanet_layers == 0);
    CHECK(cfg.deltanet_layer_policy == KvMemDeltaNetLayerPolicy::Even);
    CHECK(cfg.deltanet_mem_budget_bytes == 0);

    cfg.block_tokens = 128;
    cfg.select_budget = 128 * 4;   // budget = 4 blocks
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.retrieval_method = KvMemRetrievalMethod::DeltaNet;
    KvMemStore s(cfg);
    s.register_append(128 * 10);   // 10 blocks, ids 0..9

    // A DeltaNet score vector (what kvmem_retrieval_score_deltanet produces):
    // blocks 4 and 7 rank highest in the middle.
    std::vector<double> dn_scores(10, 0.0);
    dn_scores[4] = 5.0;
    dn_scores[7] = 4.0;
    dn_scores[2] = 1.0;
    s.set_retrieval_scores(dn_scores);

    auto sel = s.pick_topk_blocks();
    CHECK(sel.size() == 4);
    bool has0 = false, has9 = false, has4 = false, has7 = false;
    for (uint32_t id : sel) {
        if (id == 0) has0 = true;
        if (id == 9) has9 = true;
        if (id == 4) has4 = true;
        if (id == 7) has7 = true;
    }
    // Sink (0) + recent (9) kept unconditionally; the two hottest middle (4,7) win.
    CHECK(has0 && has9 && has4 && has7);
    for (size_t i = 1; i < sel.size(); ++i) CHECK(sel[i] > sel[i - 1]);
}

static void test_round_groups_are_selected_whole() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32;
    cfg.select_budget = 32 * 8;
    cfg.sink_blocks = 1;
    cfg.recent_blocks = 1;
    cfg.semantic_expansion = KvMemSemanticExpansion::Round;
    cfg.group_score_reduce =
        KvMemGroupScoreReduce::LengthNormalizedMass;
    cfg.group_length_norm_alpha = 0.5;
    KvMemStore s(cfg);
    s.register_append(32 * 20);

    // Four variable-length groups. Group 1 is hottest and overlaps blocks
    // [4,7]; group 2 is next and overlaps [8,10]. Sink block 0 and recent block
    // 19 leave six slots, so group 1 must be admitted whole and group 2 skipped
    // rather than partially selected.
    const std::vector<std::pair<uint32_t, uint32_t>> groups{
        {32, 32 * 4},
        {32 * 4 + 3, 32 * 8 - 5},
        {32 * 8, 32 * 11},
        {32 * 12, 32 * 14},
    };
    const std::vector<double> scores{1.0, 10.0, 9.0, 2.0};
    const auto selected = s.pick_semantic_groups(groups, scores);
    CHECK(selected.size() <= 8);
    CHECK(std::find(selected.begin(), selected.end(), 0) != selected.end());
    CHECK(std::find(selected.begin(), selected.end(), 19) != selected.end());
    for (uint32_t id = 4; id <= 7; ++id) {
        CHECK(std::find(selected.begin(), selected.end(), id) != selected.end());
    }
    // The second-ranked group cannot fit after the first: none of its unique
    // blocks may leak in through a partial admission.
    for (uint32_t id = 8; id <= 10; ++id) {
        CHECK(std::find(selected.begin(), selected.end(), id) == selected.end());
    }
}

static void test_round_groups_charge_shared_boundary_once() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 64;
    cfg.select_budget = 64 * 4;
    cfg.sink_blocks = 0;
    cfg.recent_blocks = 0;
    KvMemStore s(cfg);
    s.register_append(64 * 8);

    // Both groups overlap block 2. Their union is blocks [1,3], not four
    // blocks, so both fit and the shared physical transfer is charged once.
    const std::vector<std::pair<uint32_t, uint32_t>> groups{
        {64, 64 * 2 + 10},
        {64 * 2 + 10, 64 * 4},
    };
    const auto selected =
        s.pick_semantic_groups(groups, std::vector<double>{2.0, 1.0}, {7});
    CHECK(selected.size() == 4);
    for (uint32_t id : {1u, 2u, 3u, 7u}) {
        CHECK(std::find(selected.begin(), selected.end(), id) != selected.end());
    }
}

static void test_repository_reuse_order_and_generations() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 4; cfg.select_budget = 16; cfg.prefill_budget = 8;
    cfg.sink_blocks = 1; cfg.recent_blocks = 1; cfg.immutable_source_k = true;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 4});
    KvMemStore s(cfg, repo);
    std::vector<KvMemBlockHandle> hs;
    for (uint32_t i = 0; i < 4; ++i) {
        auto l = repo->try_reserve(i * 4, 4).commit();
        hs.push_back(l.handle());
        s.register_repository_block(std::move(l), i * 4);
        s.set_block_tier(hs.back().slot, KvTier::CPU, i, -1, hs.back().generation);
    }
    const auto old = hs[1];
    CHECK(repo->retire_if_unpinned(old));
    CHECK(repo->try_reclaim(old, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    auto newest = repo->try_reserve(16, 4).commit();
    hs[1] = newest.handle();
    CHECK(hs[1].slot == old.slot);
    s.register_repository_block(std::move(newest), 12);
    CHECK(s.block_count() == 4 && s.history_end() == 20);
    CHECK(s.blocks()[1].orig_pos_start == 4 && s.blocks()[1].history_pos_start == 16);
    CHECK(s.history_span_blocks(8, 20) == std::vector<uint32_t>({2, 3, 1}));
    CHECK(s.history_span_blocks(4, 12, false) == std::vector<uint32_t>({2}));
    bool missing_history = false;
    try { s.history_span_blocks(4, 12); }
    catch (const std::runtime_error &) { missing_history = true; }
    CHECK(missing_history); // slot 1 now stores [16,20), never the missing [4,8)
    s.set_repository_retrieval_scores(hs, {0, 0, 0, 0});
    s.set_runtime_select_budget(8);
    CHECK(s.pick_topk_blocks() == std::vector<uint32_t>({0, 1}));
    CHECK(s.pick_prefill_pressure_blocks() == std::vector<uint32_t>({0, 1}));
    s.set_runtime_select_budget(0);
    CHECK(s.pick_topk_blocks() == std::vector<uint32_t>({0, 2, 3, 1}));
    s.set_runtime_select_budget(12);
    CHECK(s.pick_repository_semantic_groups({{4, 12}, {8, 12}}, {100, 1}) ==
          std::vector<uint32_t>({0, 2, 1})); // skip incomplete high-scoring group
    s.set_runtime_select_budget(0);
    auto frame = s.repository_read_leases();
    CHECK(frame.size() == 4 && frame[3].record().token_begin == 16);
    CHECK(!repo->retire_if_unpinned(hs[2])); // in-flight scorer owns its frame
    frame.clear();

    s.set_repository_retrieval_scores(hs, {1, 2, 3, 4});
    bool stale = false;
    try { s.set_repository_retrieval_scores({old}, {100}); }
    catch (const std::runtime_error &) { stale = true; }
    CHECK(stale && s.blocks()[1].retrieval_score == 2); // no partial score update
    s.accumulate_repository_attn({hs[0], hs[1]}, {3, 7});
    s.accumulate_repository_attn({hs[1]}, {2});
    CHECK(s.blocks()[0].profile_score == 3 && s.blocks()[1].profile_score == 9);
    stale = false;
    try { s.accumulate_repository_attn({hs[0], old}, {100, 100}); }
    catch (const std::runtime_error &) { stale = true; }
    CHECK(stale && s.blocks()[0].profile_score == 3 && s.blocks()[1].profile_score == 9);
    bool duplicate = false;
    try { s.accumulate_repository_attn({hs[0], hs[0]}, {100, 100}); }
    catch (const std::invalid_argument &) { duplicate = true; }
    CHECK(duplicate && s.blocks()[0].profile_score == 3);
    stale = false;
    try { s.set_block_ssd_backing(old.slot, 42, true, old.generation); }
    catch (const std::runtime_error &) { stale = true; }
    CHECK(stale && !s.blocks()[1].ssd_clean);
    stale = false;
    try { s.set_repository_selection({old}); }
    catch (const std::runtime_error &) { stale = true; }
    CHECK(stale);

    auto plan = s.set_repository_selection({hs[1], hs[3], hs[0]});
    CHECK(plan.remaps.size() == 3 && plan.total_window_tokens == 12);
    CHECK(plan.remaps[0].block_id == 0 && plan.remaps[0].to_base == 0);
    CHECK(plan.remaps[1].block_id == 3 && plan.remaps[1].to_base == 4);
    CHECK(plan.remaps[2].block_id == 1 && plan.remaps[2].to_base == 8);
    auto checkpoint = s.repository_checkpoint_leases();
    CHECK(checkpoint.size() == 3);
    s.clear_working_set();
    s.set_block_tier(hs[1].slot, KvTier::CPU, 1, -1, hs[1].generation);
    CHECK(!repo->retire_if_unpinned(hs[1]));
    checkpoint.clear();
    CHECK(!repo->retire_if_unpinned(hs[1])); // pending transition still owns old window
    plan.repository_previous_window.reset();
    CHECK(repo->retire_if_unpinned(hs[1]));
    CHECK(repo->try_reclaim(hs[1], [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    CHECK(s.pick_topk_blocks() == std::vector<uint32_t>({0, 2, 3}));
}

static void test_repository_high_positions_and_holes() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32; cfg.select_budget = 64; cfg.prefill_budget = 64;
    cfg.sink_blocks = 0; cfg.recent_blocks = 0;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    KvMemStore s(cfg, repo);
    const uint64_t base = uint64_t{1} << 34;
    std::vector<KvMemBlockHandle> hs;
    for (uint32_t i = 0; i < 3; ++i) {
        auto l = repo->try_reserve(base + i * 32, 32).commit();
        hs.push_back(l.handle());
        s.register_repository_block(std::move(l), i * 32);
    }
    CHECK(s.history_end() == base + 96);
    CHECK(s.pick_topk_blocks() == std::vector<uint32_t>({1, 2}));
    CHECK(s.pick_prefill_pressure_blocks() == std::vector<uint32_t>({1, 2}));
    auto plan = s.set_repository_selection({hs[2], hs[1]});
    CHECK(plan.stage_out == std::vector<uint32_t>({0})); // never unallocated slots
    CHECK(plan.remaps.size() == 2 && plan.total_window_tokens == 64);
    CHECK(plan.remaps[0].to_base == 0 && plan.remaps[1].to_base == 32);
    CHECK(s.blocks()[1].history_pos_start == base + 32);
    CHECK(s.history_span_blocks(base + 32, base + 96) == std::vector<uint32_t>({1, 2}));
    CHECK(s.pick_repository_semantic_groups({{base + 32, base + 96}}, {1}) ==
          std::vector<uint32_t>({1, 2}));
}

static void test_repository_store_long_continuation() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32; cfg.select_budget = 96; cfg.prefill_budget = 96;
    cfg.sink_blocks = 1; cfg.recent_blocks = 1; cfg.immutable_source_k = true;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    KvMemStore s(cfg, repo);
    const size_t metadata_slots = s.blocks().capacity();
    for (uint32_t step = 0; step < 2048; ++step) {
        if (repo->stats().ready == 0) {
            // The CPU resource owner reclaims a cold non-working block. The
            // selection below is the production KvMemStore implementation.
            uint32_t victim = UINT32_MAX;
            for (const auto &b : s.blocks()) {
                if (!s.is_live(b.block_id) || b.tier == KvTier::GPU || b.in_working_set ||
                    b.history_pos_start == 0) continue;
                if (victim == UINT32_MAX || b.history_pos_start < s.blocks()[victim].history_pos_start)
                    victim = b.block_id;
            }
            CHECK(victim != UINT32_MAX);
            if (victim == UINT32_MAX) return;
            const auto h = s.repository_handle(victim);
            CHECK(repo->retire_if_unpinned(h));
            CHECK(repo->try_reclaim(h, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
        }
        auto fresh = repo->try_reserve(uint64_t(step) * 32, 32).commit();
        s.register_repository_block(std::move(fresh), 96);
        auto scoring = s.repository_read_leases();
        std::vector<KvMemBlockHandle> handles;
        std::vector<double> scores;
        for (const auto &l : scoring) {
            handles.push_back(l.handle());
            scores.push_back(double((l.record().logical_id * 17 + step) % 23));
        }
        s.set_repository_retrieval_scores(handles, scores);
        handles.clear();
        for (auto id : s.pick_topk_blocks()) handles.push_back(s.repository_handle(id));
        auto plan = s.set_repository_selection(handles);
        CHECK(plan.total_window_tokens <= cfg.select_budget);
        uint64_t prior = 0;
        for (size_t i = 0; i < plan.remaps.size(); ++i) {
            const auto &b = s.blocks()[plan.remaps[i].block_id];
            CHECK(i == 0 || b.history_pos_start > prior);
            CHECK(plan.remaps[i].to_base == int32_t(i * 32));
            prior = b.history_pos_start;
        }
        for (auto id : plan.stage_out) {
            const auto h = s.repository_handle(id);
            s.set_block_tier(id, KvTier::CPU, int32_t(id), -1, h.generation);
        }
        for (auto id : plan.stage_in) {
            const auto h = s.repository_handle(id);
            s.set_block_tier(id, KvTier::GPU, -1, -1, h.generation);
        }
        CHECK(s.blocks().capacity() == metadata_slots);
        CHECK(s.history_end() == uint64_t(step + 1) * 32);
    }
    CHECK(repo->stats().reclaimed == 2040 && s.block_count() == 8);
}

static void test_repository_batched_append_and_partial_commit() {
    KvMemStoreConfig cfg;
    cfg.block_tokens = 32; cfg.select_budget = 64;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    KvMemStore s(cfg, repo);
    auto a = s.prepare_repository_append(73, 0);
    CHECK(a && a->spans().size() == 3 && repo->ready_slots() == 1);
    CHECK(repo->stats().writing == 3 && repo->stats().live == 0);
    CHECK(s.history_end() == 0 && s.repository_read_leases().empty());
    bool rejected = false;
    try { s.repository_storage_read(0, 1); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected); // no unwritten payload is published by allocation
    rejected = false;
    try { s.prepare_repository_append(1, 0); } catch (const std::logic_error &) { rejected = true; }
    CHECK(rejected);
    auto dma = a->io_leases();
    const auto cancelled = a->spans()[2].handle;
    a->commit(37); // accept a full block and five tokens, reject the third
    CHECK(s.history_end() == 37 && s.repository_directory_entries() == 2);
    CHECK(s.blocks()[1].n_tokens == 5 && repo->stats().cancelled == 1);
    CHECK(repo->try_reclaim(cancelled, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::DeferredReaders);
    dma.clear();
    CHECK(repo->try_reclaim(cancelled, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::Reclaimed);
    auto read = s.repository_storage_read(30, 7);
    CHECK(read.spans.size() == 2 && read.spans[0].storage_begin == 30 &&
          read.spans[0].token_count == 2 && read.spans[1].storage_begin == 32 &&
          read.spans[1].token_count == 5);
    rejected = false;
    try { s.repository_storage_read(37, 1); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected); // rejected draft bytes do not extend valid KV

    // Crossing a partial tail and two fresh blocks keeps one ordinary batch.
    a = s.prepare_repository_append(60, 37);
    CHECK(a && a->spans().size() == 3);
    CHECK(a->spans()[0].token_count == 27 && a->spans()[0].storage_begin == 37);
    CHECK(a->spans()[1].history_begin == 64 && a->spans()[1].storage_begin == 96);
    CHECK(a->spans()[2].history_begin == 96 && a->spans()[2].storage_begin == 64);
    const auto pages = kvmem_storage_page_spans(a->spans(), 16, 37);
    CHECK(pages.size() == 3);
    CHECK(pages[0].storage_page == 2 && pages[0].window_page == 2 && pages[0].page_count == 2);
    CHECK(pages[1].storage_page == 6 && pages[1].window_page == 4 && pages[1].page_count == 2);
    CHECK(pages[2].storage_page == 4 && pages[2].window_page == 6 && pages[2].page_count == 1);
    a->commit(60);
    CHECK(s.history_end() == 97 && repo->ready_slots() == 0);
    CHECK(s.history_span_blocks(0, 97) == std::vector<uint32_t>({0, 1, 3, 2}));
    // Reserving a further whole block fails atomically, leaving the partial
    // tail available for a smaller append; no foreground GC/reclaim happens.
    const auto before = repo->stats().reservations;
    CHECK(!s.prepare_repository_append(32, 97));
    CHECK(s.history_end() == 97 && repo->stats().reservations == before);
    a = s.prepare_repository_append(31, 97);
    CHECK(a && a->spans().size() == 1);
    a->commit(31);
    CHECK(s.history_end() == 128 && s.blocks()[2].n_tokens == 32);
}

static void test_repository_append_extension_and_write_views() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{3, 32});
    KvMemStore s(cfg, repo);
    auto a = s.prepare_repository_append(29, 0);
    a->commit(29);
    a = s.prepare_repository_append(1, 29);
    CHECK(a->extend(3) && a->spans().size() == 1);
    CHECK(a->extend(7) && a->spans().size() == 2);
    CHECK(a->begin() == 29 && a->end() == 36 && s.history_end() == 29);
    CHECK(s.blocks()[0].n_tokens == 29); // speculative extension is not committed
    auto write = a->storage_write(31, 4);
    CHECK(write.spans.size() == 2 && write.leases.size() == 2);
    CHECK(write.spans[0].token_count == 1 && write.spans[0].storage_begin == 31);
    CHECK(write.spans[1].token_count == 3 && write.spans[1].history_begin == 32);
    const auto pending = write.spans[1].handle;
    CHECK(!repo->try_pin(pending)); // Writing is accessible only through its reservation
    CHECK(a->extend(40) && a->spans().size() == 3 && a->end() == 69);
    CHECK(a->spans()[1].token_count == 32 && a->spans()[2].token_count == 5);
    const auto before = repo->stats().reservations;
    CHECK(!a->extend(68)); // one additional block is unavailable
    CHECK(a->end() == 69 && a->spans().size() == 3 && repo->stats().reservations == before);
    CHECK(a->extend(39) && a->end() == 69); // never shrink outstanding writes
    bool rejected = false;
    try { a->storage_write(68, 2); } catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected);
    const auto cancelled = a->spans()[2].handle;
    auto late_io = a->storage_write(64, 5);
    CHECK(a->discarded_page_ranges(40, 16).empty());
    CHECK(a->discarded_page_ranges(4, 16) ==
          (std::vector<std::pair<uint32_t, uint32_t>>{{3, 2}}));
    CHECK(a->discarded_page_ranges(3, 16) ==
          (std::vector<std::pair<uint32_t, uint32_t>>{{2, 3}}));
    CHECK(a->discarded_page_ranges(0, 16) ==
          (std::vector<std::pair<uint32_t, uint32_t>>{{2, 3}})); // preserve committed page [16,32)
    a->commit(4); // accept 3 tail tokens and just 1 new-block token
    CHECK(s.history_end() == 33 && s.blocks()[pending.slot].n_tokens == 1);
    CHECK(repo->try_reclaim(cancelled, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::DeferredReaders);
    late_io = {};
    CHECK(repo->try_reclaim(cancelled, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::Reclaimed);
    write = {};
    // The following batch must still use the correct partial-tail reservation.
    a = s.prepare_repository_append(1, 33);
    CHECK(a->extend(31) && a->spans().size() == 1);
    CHECK(a->extend(32) && a->spans().size() == 2);
    auto tail = a->storage_write(63, 2);
    CHECK(tail.spans.size() == 2 && tail.spans[0].token_count == 1);
    a->commit(32);
    CHECK(s.history_end() == 65);
}

static void test_repository_score_frame_masks_and_versions() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.select_budget = 64;
    cfg.sink_blocks = 1; cfg.recent_blocks = 1;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{5, 32});
    KvMemStore s(cfg, repo);
    std::vector<KvMemIndexStamp> index(5);
    for (uint32_t i = 0; i < 5; ++i) {
        auto l = repo->try_reserve(i * 32, 32).commit();
        const auto h = l.handle();
        index[h.slot] = {h, 32};
        s.register_repository_block(std::move(l), 0);
        s.set_block_tier(h.slot, KvTier::CPU, 0, -1, h.generation);
    }
    auto frame = s.repository_score_frame(index, 160, true);
    CHECK(frame.enabled == std::vector<uint8_t>({0, 1, 1, 1, 0}));
    const auto old = index[1].handle;
    CHECK(!repo->retire_if_unpinned(old)); // scoring pins survive through selection
    frame = {};
    CHECK(repo->retire_if_unpinned(old));
    CHECK(repo->try_reclaim(old, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::Reclaimed);
    auto next = repo->try_reserve(160, 32).commit();
    const auto fresh = next.handle();
    CHECK(fresh.slot == old.slot && fresh.generation != old.generation);
    s.register_repository_block(std::move(next), 0);
    s.set_block_tier(fresh.slot, KvTier::CPU, 0, -1, fresh.generation);
    bool rejected = false;
    try { s.repository_score_frame(index, 192, true); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected); // valid address, stale canonical Mean-K generation
    index[fresh.slot] = {fresh, 32};
    frame = s.repository_score_frame(index, 192, true);
    CHECK(frame.enabled == std::vector<uint8_t>({0, 0, 1, 1, 1}));
    frame = {};
    CHECK(repo->retire_if_unpinned(index[3].handle));
    frame = s.repository_score_frame(index, 192, false);
    CHECK(frame.enabled == std::vector<uint8_t>({1, 1, 1, 0, 1})); // deleted block excluded before softmax
    frame = s.repository_score_frame(index, 128, true);
    CHECK(frame.enabled == std::vector<uint8_t>({1, 0, 1, 0, 0})); // exact query boundary, all-fit
    frame = {};
    rejected = false;
    try { s.repository_score_frame(index, 80, false); }
    catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected); // full-block mean is not a mean of its first 16 tokens
    index[2].tokens = 16;
    frame = s.repository_score_frame(index, 80, false);
    CHECK(frame.enabled == std::vector<uint8_t>({1, 0, 1, 0, 0}));
}

static void test_repository_append_abort_and_read_lifetime() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{3, 32});
    KvMemStore s(cfg, repo);
    auto a = s.prepare_repository_append(5, 0);
    a->commit(5);
    a = s.prepare_repository_append(50, 5);
    const auto cancelled = a->spans()[1].handle;
    a.reset();
    CHECK(s.history_end() == 5 && s.blocks()[0].n_tokens == 5);
    CHECK(s.repository_read_leases().size() == 1);
    CHECK(repo->try_reclaim(cancelled, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::Reclaimed);
    a = s.prepare_repository_append(27, 5);
    a->commit(27);
    const auto h = s.repository_handle(0);
    auto read = s.repository_storage_read(4, 6);
    s.set_block_tier(0, KvTier::CPU, 0, -1, h.generation);
    CHECK(!repo->retire_if_unpinned(h));
    read = {};
    CHECK(repo->retire_if_unpinned(h));
    CHECK(repo->try_reclaim(h, [](const auto &) { return true; }) ==
          KvMemRepository::ReclaimResult::Reclaimed);
    bool rejected = false;
    try { s.repository_storage_read(0, 1); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    // Reusing a slot never causes its old logical interval to reappear.
    for (unsigned i = 0; i < 3; ++i) {
        a = s.prepare_repository_append(32, 0);
        a->commit(32);
    }
    CHECK(s.repository_directory_entries() == 3);
    CHECK(s.blocks()[0].history_pos_start == 96);
    CHECK(s.repository_storage_read(96, 1).spans[0].storage_begin == 0);
    rejected = false;
    try { s.repository_storage_read(0, 1); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
}

static void test_repository_chunked_payload_continuation() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    KvMemStore s(cfg, repo);
    std::vector<uint64_t> payload(8 * 32, UINT64_MAX);
    const uint64_t high = uint64_t{1} << 34;
    s.register_repository_block(repo->try_reserve(high, 32).commit(), 0);
    for (uint32_t i = 0; i < 32; ++i) payload[i] = high + i;
    bool recycled = false;
    for (uint32_t step = 0; step < 4096; ++step) {
        const uint32_t count = step % 113 + 1;
        // Simulate a CPU owner maintaining ready capacity in advance. This
        // tests addressing, not GC victim policy or GPU performance.
        for (const auto &b : s.blocks()) {
            if (!s.is_live(b.block_id) || b.history_pos_start + b.n_tokens >= s.history_end()) continue;
            const auto h = s.repository_handle(b.block_id);
            s.set_block_tier(h.slot, KvTier::CPU, h.slot, -1, h.generation);
            CHECK(repo->retire_if_unpinned(h));
            CHECK(repo->try_reclaim(h, [](const auto &) { return true; }) ==
                  KvMemRepository::ReclaimResult::Reclaimed);
            recycled = true;
        }
        const uint64_t begin = s.history_end();
        const uint32_t tail_offset = static_cast<uint32_t>(begin % 32);
        uint32_t local = 0;
        if (tail_offset) {
            auto read = s.repository_storage_read(begin - 1, 1);
            const auto &b = s.blocks()[read.spans[0].handle.slot];
            local = static_cast<uint32_t>(b.baked_pos) + tail_offset;
        }
        auto a = s.prepare_repository_append(count, local);
        CHECK(bool(a));
        if (!a) return;
        for (const auto &span : a->spans()) {
            CHECK(span.storage_begin + span.token_count <= payload.size());
            for (uint32_t i = 0; i < span.token_count; ++i)
                payload[span.storage_begin + i] = span.history_begin + i;
        }
        const auto page_spans = kvmem_storage_page_spans(a->spans(), 16, local);
        std::vector<uint32_t> page_map((local + count + 15) / 16, UINT32_MAX);
        for (const auto &p : page_spans)
            for (uint32_t i = 0; i < p.page_count; ++i) page_map[p.window_page + i] = p.storage_page + i;
        // This is the same compact-page-table addressing the attention kernel
        // uses for both main and MTP KV append: only the page aliases differ.
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t win = local + i;
            CHECK(payload[page_map[win / 16] * 16 + win % 16] == begin + i);
        }
        a->commit(count);
        const auto read = s.repository_storage_read(begin, count);
        for (const auto &span : read.spans)
            for (uint32_t i = 0; i < span.token_count; ++i)
                CHECK(payload[span.storage_begin + i] == span.history_begin + i);
        CHECK(s.repository_directory_entries() <= 8 && s.block_count() == 8);
    }
    CHECK(recycled && s.history_end() > high + 200000);
}

static void test_repository_page_map_invalid_inputs() {
    const std::vector<std::vector<KvMemStorageSpan>> cases{
        {{0, 0, 0, {}}},                        // empty slice
        {{0, 1, 32, {}}},                       // different page offset
        {{0, 0, 31, {}}, {31, 64, 5, {}}},      // split in the middle of a page
        {{0, 0, 32, {}}, {64, 64, 5, {}}},      // missing history
        {{UINT64_MAX - 1, 0, 32, {}}},          // history overflow
        {{0, UINT32_MAX - 15, 32, {}}},         // storage overflow
    };
    for (const auto &spans : cases) {
        bool rejected = false;
        try { kvmem_storage_page_spans(spans, 16, 0); }
        catch (const std::invalid_argument &) { rejected = true; }
        CHECK(rejected);
    }
    const auto compact = kvmem_storage_page_spans({{1024, 32, 32, {}}, {1056, 64, 32, {}}}, 16, 0);
    CHECK(compact.size() == 1 && compact[0].storage_page == 2 &&
          compact[0].window_page == 0 && compact[0].page_count == 4);
}

static void test_repository_io_completion_during_append() {
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.sink_blocks = 0;
    KvMemStore store(cfg, repo);
    auto a = store.prepare_repository_append(32, 0);
    const auto h = a->spans().front().handle;
    a->commit(32); a.reset();
    auto io = store.repository_storage_read(0, 32);
    store.set_block_io_in_flight(h.slot, true, h.generation);
    a = store.prepare_repository_append(32, 32);
    store.set_block_cpu_copy(h.slot, 0, h.generation);
    store.set_block_ssd_backing(h.slot, 0, true, h.generation);
    store.set_block_io_in_flight(h.slot, false, h.generation);
    CHECK(store.blocks()[h.slot].cpu_slot == 0 && store.blocks()[h.slot].ssd_clean);
    CHECK(store.history_end() == 32); // completion did not publish prepared tokens
    bool rejected = false;
    try { store.set_repository_selection({h}); } catch (const std::logic_error &) { rejected = true; }
    CHECK(rejected); // structural changes remain forbidden
    a->commit(32); a.reset();
    store.set_block_tier(h.slot, KvTier::CPU, 0, 0, h.generation);
    CHECK(!repo->retire_if_unpinned(h)); // actual reader still owns payload
    io = {};
    CHECK(repo->retire_if_unpinned(h));
    CHECK(repo->try_reclaim(h, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    rejected = false;
    try { store.set_block_cpu_copy(h.slot, 0, h.generation); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected); // a late publication cannot target a retired generation
}

static void test_repository_physical_snapshot() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.sink_blocks = 0;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    KvMemStore store(cfg, repo);
    auto a = store.prepare_repository_append(37, 0);
    auto first = a->spans()[0].handle;
    a->commit(37); a.reset();
    store.set_block_tier(first.slot, KvTier::CPU, first.slot, -1, first.generation);
    CHECK(repo->retire_if_unpinned(first));
    auto holes = store.repository_slot_read(0, 4);
    CHECK(holes.spans.size() == 1 && holes.spans[0].token_count == 5);
    a = store.prepare_repository_append(50, 37);
    auto writing = a->spans().back().handle;
    auto capture = store.repository_slot_read(0, 4, a.get());
    CHECK(capture.spans.size() == 2 && capture.spans[0].token_count == 32 && capture.spans[1].token_count == 23);
    CHECK(store.repository_slot_read(writing.slot, 1).spans.empty());
    CHECK(!repo->try_pin(writing)); // only the reservation supplied this lease
    a.reset();
    CHECK(repo->try_reclaim(writing, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::DeferredReaders);
    capture = {};
    CHECK(repo->try_reclaim(writing, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    CHECK(store.repository_slot_read(0, 4).spans.size() == 1);
    CHECK(repo->try_reclaim(first, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
}

static void test_repository_checkpoint_rewind() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.sink_blocks = 0;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{8, 32});
    KvMemStore store(cfg, repo);
    auto append = store.prepare_repository_append(37, 0);
    const auto first = append->spans()[0].handle, tail = append->spans()[1].handle;
    append->commit(37); append.reset();
    auto before = store.repository_checkpoint(37);
    append = store.prepare_repository_append(50, 37);
    const auto last = append->spans().back().handle;
    append->commit(50); append.reset();
    auto after = store.repository_checkpoint(87);
    store.set_block_io_in_flight(last.slot, true, last.generation);
    bool rejected = false;
    try { store.prepare_repository_rewind(37); } catch (const std::logic_error &) { rejected = true; }
    CHECK(rejected && store.history_end() == 87 && after->valid());
    store.set_block_io_in_flight(last.slot, false, last.generation);
    auto rewind = store.prepare_repository_rewind(37);
    CHECK(rewind->discarded_spans().size() == 2);
    CHECK(rewind->discarded_spans()[0].storage_begin == tail.slot * 32 + 5);
    CHECK(rewind->discarded_spans()[0].token_count == 27);
    CHECK(rewind->discarded_spans()[1].handle == last);
    rewind.reset(); // preflight cancellation leaves the live branch unchanged
    CHECK(store.history_end() == 87 && after->valid());
    rewind = store.prepare_repository_rewind(37);
    rewind->commit();
    CHECK(store.history_end() == 37 && store.blocks()[tail.slot].n_tokens == 5);
    CHECK(store.repository_checkpoint_valid(before) && !store.repository_checkpoint_valid(after));
    CHECK(store.repository_directory_entries() == 2);
    CHECK(repo->try_reclaim(last, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::DeferredReaders);
    rewind.reset();
    CHECK(repo->try_reclaim(last, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    append = store.prepare_repository_append(50, 37);
    CHECK(append->spans().back().handle != last);
    append->commit(50); append.reset();
    CHECK(!after->valid() && store.repository_checkpoint_valid(before)); // same length, new branch
    auto read = store.repository_storage_read(32, 55);
    CHECK(read.spans.size() == 2 && read.spans[1].handle != last);
    read = {}; after.reset();
    std::vector<std::shared_ptr<KvMemStore::RepositoryCheckpoint>> checkpoints;
    for (size_t i = 1; i < KvMemStore::repository_checkpoint_capacity; ++i)
        checkpoints.push_back(store.repository_checkpoint(87));
    rejected = false;
    try { store.repository_checkpoint(87); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected);
    checkpoints.clear();
    for (size_t i = 0; i < 1000; ++i) CHECK(store.repository_checkpoint(87)->valid());
    // A checkpoint pins its selected prefix even after foreground stage-out.
    store.set_block_tier(first.slot, KvTier::CPU, first.slot, -1, first.generation);
    CHECK(!repo->retire_if_unpinned(first));
    before.reset();
    CHECK(repo->retire_if_unpinned(first));
    CHECK(repo->try_reclaim(first, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    // Rewind may preserve older GC holes, but cannot invent a partial tail in one.
    rejected = false;
    try { store.prepare_repository_rewind(17); } catch (const std::runtime_error &) { rejected = true; }
    CHECK(rejected && store.history_end() == 87);
    rewind = store.prepare_repository_rewind(32); rewind->commit(); rewind.reset();
    CHECK(store.history_end() == 32 && store.repository_directory_entries() == 1);
    for (uint32_t slot = 0; slot < repo->capacity(); ++slot) {
        const auto h = repo->retired_at(slot);
        if (h.generation) CHECK(repo->try_reclaim(h, [](const auto &) { return true; }) == KvMemRepository::ReclaimResult::Reclaimed);
    }
    CHECK(repo->ready_slots() == 8);
}

static void test_repository_pressure_selection_lifetime() {
    KvMemStoreConfig cfg; cfg.block_tokens = 32; cfg.prefill_budget = 96;
    cfg.sink_blocks = 1;
    auto repo = std::make_shared<KvMemRepository>(KvMemRepository::Config{4, 32});
    KvMemStore store(cfg, repo);
    std::vector<KvMemBlockHandle> handles;
    for (uint32_t i = 0; i < 4; ++i) {
        auto lease = repo->try_reserve(i * 32, 32).commit();
        handles.push_back(lease.handle());
        store.register_repository_block(std::move(lease), 0);
        store.set_block_tier(handles.back().slot, KvTier::CPU, 0, -1, handles.back().generation);
    }
    auto selection = store.pin_repository_pressure_selection({});
    CHECK(selection.blocks == std::vector<uint32_t>({0, 2, 3}));
    // Deterministically put a CPU reclaimer BETWEEN candidate selection and
    // installation, the gap that previously had no repository lease.
    std::vector<uint8_t> retired(4);
    std::thread worker([&] {
        for (uint32_t i = 0; i < 4; ++i) retired[i] = repo->retire_if_unpinned(handles[i]);
    });
    worker.join();
    CHECK(retired == std::vector<uint8_t>({0, 1, 0, 0}));
    std::vector<KvMemBlockHandle> chosen;
    for (const auto &lease : selection.leases) chosen.push_back(lease.handle());
    auto plan = store.set_repository_selection(chosen);
    selection.leases.clear();
    CHECK(plan.total_window_tokens == 96);
    for (auto id : selection.blocks) CHECK(!repo->retire_if_unpinned(handles[id]));
    store.clear_working_set();
    for (auto id : selection.blocks) CHECK(repo->retire_if_unpinned(handles[id]));
}

int main() {
    test_repository_pressure_selection_lifetime();
    test_repository_checkpoint_rewind();
    test_repository_physical_snapshot();
    test_repository_batched_append_and_partial_commit();
    test_repository_io_completion_during_append();
    test_repository_append_extension_and_write_views();
    test_repository_score_frame_masks_and_versions();
    test_repository_append_abort_and_read_lifetime();
    test_repository_chunked_payload_continuation();
    test_repository_page_map_invalid_inputs();
    test_repository_reuse_order_and_generations();
    test_repository_high_positions_and_holes();
    test_repository_store_long_continuation();
    test_register_append();
    test_selection_diff_and_remap();
    test_immutable_source_selection_uses_bounded_delta_remaps();
    test_fp8_every_move_uses_raw_authority();
    test_cold_immutable_same_position_rebuilds_k();
    test_default_profile_reuses_moved_resident_k();
    test_explicit_force_raw_refresh_rebuilds_stable_selection();
    test_stage_in_uses_tier_residency();
    test_stage_in_ablation_forces_full_reload();
    test_topk_budget_sink_recent();
    test_topk_zero_recent_keeps_no_suffix();
    test_budget_scaled_keep_allocation();
    test_topk_mandatory_blocks_stay_inside_budget();
    test_mandatory_overlap_deduplicates_and_recent_is_best_effort();
    test_mandatory_hard_union_overflow_throws();
    test_long_agent_query_is_not_an_unbounded_suffix_pin();
    test_prefill_pressure_mandatory_blocks_stay_inside_budget();
    test_prefill_pressure_sink_full_recent_tail();
    test_prefill_pressure_edges();
    test_prefill_pressure_budget_is_independent_from_semantic_budget();
    test_request_semantic_budget_override_is_scoped_and_bounded();
    test_quota_policy_sink_recent_retrieval_profile();
    test_topk_all_fit();
    test_topk_empty();
    test_tier_metadata();
    test_inclusive_tier_metadata();
    test_stage_out_off_exclusive_tier_metadata();
    test_factorial_optimization_controls_are_orthogonal();
    test_truncate_to();
    test_deltanet_config_and_scores();
    test_round_groups_are_selected_whole();
    test_round_groups_charge_shared_boundary_once();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
