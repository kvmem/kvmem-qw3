// KvMemStore host-logic unit test (block-sparse KV, Task #37).
//
// Pure host logic — no GPU. Covers: block registration (partial-block growth),
// the working-set diff (stage-in/out + window remap plan), and the built-in
// cumulative-attention top-k selection (sink/recent preservation + budget).

#include "qw3/kvmem_store.hpp"

#include <cstdio>
#include <cstdlib>
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
}

int main() {
    test_register_append();
    test_selection_diff_and_remap();
    test_stage_in_uses_tier_residency();
    test_topk_budget_sink_recent();
    test_quota_policy_sink_recent_retrieval_profile();
    test_topk_all_fit();
    test_topk_empty();
    test_tier_metadata();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
