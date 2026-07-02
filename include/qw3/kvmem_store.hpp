#pragma once

// Block-sparse KV attention — block repository + working-set selection.
//
// A single long context (agent multi-turn, cumulative up to ~1M tokens) is
// split into fixed-size blocks (default 128 tokens, independent of the 16-token
// physical KV page). Each selection round an external algorithm (v1: built-in
// cumulative-attention top-k) picks a set of block IDs within the model's
// context budget (e.g. 128k). Only selected blocks are used for attention;
// they are remapped into a contiguous in-window position range via re-RoPE
// (see rope_block_remap_kernel) so RoPE stays in-distribution.
//
// This module is PURE HOST LOGIC: block metadata, selection, the diff that
// drives stage-in / stage-out, and the position remap plan. It owns no GPU
// memory and calls no kernels — the executor consumes its output (a per-block
// remap plan + the set of blocks to stage) to assemble the live page-index
// list. Keeping it host-only makes the selection/diff math unit-testable
// without a GPU (see tests/kv_block_store_test.cpp).
//
// Scope note: blocks/selection/tiering/remap apply ONLY to standard attention
// layers (1/4 of layers in Qwen3.6). DeltaNet recurrent layers store O(1)
// state and are untouched. Selection is GLOBAL (all attention layers share the
// same selected block IDs); the per-layer dimension is reserved for later.

#include <cstdint>
#include <string>
#include <vector>

namespace qw3 {

enum class KvTier : uint8_t { GPU = 0, CPU = 1, SSD = 2 };

// Metadata for one context block, spanning all attention layers (a token range
// has KV in every attention layer). Physical storage handles per tier are
// added in the tiering task (#41); v1 tracks logical placement + remap state.
struct KvMemBlock {
    uint32_t block_id = 0;          // dense index in append order (0,1,2,...)
    uint32_t orig_pos_start = 0;    // first original token position in block
    uint32_t n_tokens = 0;          // tokens in this block (<= block_tokens)
    KvTier   tier = KvTier::GPU;    // where the block's KV currently lives
    int32_t  gpu_slot = -1;         // future physical GPU block/page slot
    int32_t  cpu_slot = -1;         // CPU pinned tier slot, -1 if absent
    int32_t  nvme_slot = -1;        // NVMe tier slot, -1 if absent
    bool     dirty_gpu = false;     // GPU copy newer than backing copy
    bool     in_flight = false;     // async copy/verification owns this block

    // The window position this block's K is CURRENTLY baked to, in place, in
    // the GPU cache (no-copy design: the stored cache IS the repository). After
    // prefill a block is baked at its true sequential position == orig_pos_start.
    // Each assembly remaps IN PLACE from baked_pos to the new window slot via a
    // de-rotate(baked_pos)+re-rotate(new) pass (rope_block_remap_paged), reusing
    // the same __sincosf so the prior bake's error cancels. All window slots are
    // <=128k (the regime production already runs in), so repeated in-place
    // remaps stay clean; fp16 storage drift is ~1 ULP/remap, bounded by skipping
    // no-op remaps (baked_pos already == new slot).
    int64_t  baked_pos = -1;        // -1 until first registered (then orig_pos_start)
    bool     in_working_set = false;

    // Cumulative attention quality (built-in top-k selection signal, #40).
    double   attn_score = 0.0;
    double   profile_score = 0.0;    // window-local cumulative attention heat
    double   retrieval_score = 0.0;  // global retrieval score

    uint32_t orig_pos_end() const { return orig_pos_start + n_tokens; }
};

// One block's remap instruction for the executor: the block's K is currently
// baked in place at from_base; re-bake it to to_base (its new window slot).
// Fed to rope_block_remap_paged per attention layer during assembly. `skip` is
// set when from_base == to_base (the block already sits in the right window
// slot) so the executor issues no kernel — the dominant case across rounds with
// stable selection.
struct KvMemRemap {
    uint32_t block_id = 0;
    uint32_t n_tokens = 0;
    int32_t  from_base = 0;   // current bake position (de-rotate source)
    int32_t  to_base = 0;     // assigned in-window first-token position
    bool     skip = false;    // from_base == to_base: no kernel needed
};

// One block removed by truncate_to, carrying the tier slots the executor must
// release. GPU pages are NOT freed here (restore_state already truncated the KV
// page table); only CPU/NVMe tier slots are the executor's to release.
struct KvMemDroppedBlock {
    uint32_t block_id = 0;
    KvTier   tier = KvTier::GPU;
    int32_t  gpu_slot = -1;
    int32_t  cpu_slot = -1;
    int32_t  nvme_slot = -1;
};

// Result of diffing a new selection against the current working set.
struct KvMemPlan {
    std::vector<uint32_t> stage_in;   // selected blocks not currently resident
    std::vector<uint32_t> stage_out;  // resident blocks no longer selected
    // Full remap plan for the new working set, in window order. The executor
    // assembles the kernel page-index list from these blocks in this order and
    // sets the query position to total_window_tokens.
    std::vector<KvMemRemap> remaps;
    uint32_t total_window_tokens = 0;  // sum of n_tokens over selected blocks
};

// Which signal ranks the middle (non-sink/recent) blocks each reselection.
// All three feed the SAME pick_topk_blocks (sink + recent windows are always
// kept); they differ only in what writes attn_score:
//   Retrieval — global content-frame similarity to the current query; re-ranks
//               ALL historical blocks each interval and CAN resurrect a block
//               already dropped from the window (Quest/InfLLM-style). Default.
//   H2O       — window-local cumulative attention heat; a heavy-hitter
//               RETENTION signal that keeps attended blocks resident but cannot
//               resurrect a dropped one (H2O-style).
//   Recency   — no learned signal; pick_topk keeps only sink + recent blocks.
// Retrieval auto-falls-back to H2O when its index/query is not live (q8/fp8
// cache, or the first post-prefill reselect before any query exists), and H2O
// in turn degenerates to Recency when no heat has accumulated. Selecting H2O or
// Recency explicitly disables the more expensive upstream signal.
enum class KvMemMethod : uint8_t { Retrieval = 0, H2O = 1, Recency = 2 };
enum class KvMemSelectPolicy : uint8_t { TopK = 0, Quota = 1 };
// Query-conditioned scorer (--kvmem-retrieval-method): MeanK = softmax-over-pages
// on the cheap per-block mean key; PerToken = ExactMass over raw per-token keys.
enum class KvMemRetrievalMethod : uint8_t { MeanK = 0, PerToken = 1 };
enum class KvMemUpdateMode : uint8_t { Interval = 0, Step = 1 };

struct KvMemStoreConfig {
    uint32_t block_tokens = 128;     // --kvmem-block-tokens
    uint32_t select_budget = 131072; // --kvmem-budget (max window tokens)
    uint32_t sink_blocks = 1;        // --kvmem-sink-blocks (always-kept prefix)
    uint32_t recent_blocks = 0;      // --kvmem-recent-blocks (always-kept suffix);
                                     // 0 => derived from budget at use time
    KvMemMethod select_method = KvMemMethod::Retrieval; // --kvmem-method
    KvMemSelectPolicy select_policy = KvMemSelectPolicy::TopK;
    KvMemRetrievalMethod retrieval_method = KvMemRetrievalMethod::MeanK;
    KvMemUpdateMode update_mode = KvMemUpdateMode::Interval;

    // Quota policy. Zero means derive from the remaining budget at selection
    // time. Quotas are measured in blocks and apply after sink/recent are kept.
    uint32_t retrieval_blocks = 0;
    uint32_t profile_blocks = 0;

    // GPU KV residency budget. The current implementation uses this to cap the
    // active working-set budget; CPU/NVMe tiering will use the same estimated
    // block capacity as the global GPU repository high-watermark.
    double gpu_memory_ratio = 0.50;
    double gpu_high_watermark = 0.95;
    double gpu_low_watermark = 0.85;
    uint32_t estimated_gpu_block_capacity = 0;
    uint64_t estimated_block_bytes = 0;
    uint64_t cpu_tier_bytes = 0;
    uint64_t nvme_tier_bytes = 0;
    std::string nvme_tier_dir;
};

class KvMemStore {
public:
    explicit KvMemStore(KvMemStoreConfig cfg) : cfg_(cfg) {}

    const KvMemStoreConfig &config() const { return cfg_; }
    uint32_t block_count() const { return static_cast<uint32_t>(blocks_.size()); }
    const std::vector<KvMemBlock> &blocks() const { return blocks_; }

    // Register newly-appended context tokens as blocks. Called as the context
    // grows. Partial trailing blocks are extended in place until full, then a
    // new block starts. Returns the number of blocks that became newly full
    // (the trailing partial block is always present once any token exists).
    void register_append(uint32_t n_new_tokens);

    // Drop trailing blocks so the store holds exactly `token_pos` tokens (the
    // inverse of register_append). Used by the kvmem prefix cache to rewind the
    // block table when resuming at a prompt-end checkpoint whose position is
    // BELOW the live end. A partially-covered trailing block is shrunk in place;
    // fully-past blocks are popped from the back (block_id == vector index, so
    // popping keeps indices dense). Returns the popped blocks + their tier slots
    // so the caller can release CPU/NVMe storage. `token_pos >= total_tokens_`
    // is a no-op (returns empty).
    std::vector<KvMemDroppedBlock> truncate_to(uint32_t token_pos);

    // Capacity of the selection budget measured in blocks.
    uint32_t budget_blocks() const { return cfg_.select_budget / cfg_.block_tokens; }

    // Built-in v1 selection: top-k blocks by cumulative attention score, always
    // keeping the first `sink_blocks` and last `recent_blocks` blocks. Returns
    // selected block IDs in ascending (window) order. Honors the budget.
    std::vector<uint32_t> pick_topk_blocks() const;

    // Accumulate per-block attention quality (decode side-channel, #40). `scores`
    // is indexed by block_id; entries beyond block_count() are ignored.
    void accumulate_attn(const std::vector<double> &scores);

    // Overwrite per-block selection score (global KV retrieval). Unlike
    // accumulate_attn (a window-local running sum), retrieval re-scores every
    // historical block fresh each interval by query/mean-key similarity, so the
    // score is REPLACED, not added. `scores` is indexed by block_id; entries
    // beyond block_count() are ignored, blocks beyond `scores.size()` are reset
    // to 0 so a shrinking index never leaves stale heat behind.
    void set_attn_scores(const std::vector<double> &scores);

    // Explicit retrieval scores for selector plugins. This currently aliases
    // set_attn_scores for backwards compatibility, but also preserves the
    // retrieval/profile split used by the quota selector.
    void set_retrieval_scores(const std::vector<double> &scores);

    void set_block_tier(uint32_t block_id, KvTier tier,
                        int32_t cpu_slot = -1,
                        int32_t nvme_slot = -1);
    void set_block_baked_pos(uint32_t block_id, int64_t baked_pos);

    // Diff `selected_ids` against current GPU residency / working-set state and produce the
    // stage-in/out lists + the window remap plan. Each remap de-rotates from the
    // block's CURRENT bake position and re-rotates to its new window slot (in
    // place, no copy); blocks already in the right slot get skip=true. Updates
    // each block's baked_pos / in_working_set. `selected_ids` need not be sorted;
    // it is sorted ascending internally so window order is deterministic (sink
    // first ... recent last). Blocks are packed contiguously from window pos 0.
    KvMemPlan set_selection(std::vector<uint32_t> selected_ids);

    // Reset working-set membership (e.g. new session). Block table + baked_pos
    // are kept (the cache still holds K baked at whatever position it was left).
    void clear_working_set();

private:
    KvMemStoreConfig cfg_;
    std::vector<KvMemBlock> blocks_;
    uint32_t total_tokens_ = 0;  // total appended context length
};

} // namespace qw3
