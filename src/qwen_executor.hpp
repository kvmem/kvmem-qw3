#pragma once

#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include "qw3/pinned_kv_tier.hpp"

#include <array>
#include <cstdio>
#include <deque>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

class KvmemCpuWorkerPool;

struct NativeExecutorReport {
    bool ok = false;
    uint64_t ops_executed = 0;
    int argmax_token = -1;
    float argmax_logit = 0.0f;
    std::string argmax_text;
    std::vector<std::string> executed;
    std::vector<double> elapsed_us;
    std::vector<std::string> missing_kernels;
};

class KvPhysicalPageAllocator {
public:
    virtual ~KvPhysicalPageAllocator() = default;
    virtual int32_t allocate_physical_page() = 0;
    virtual void release_physical_pages(const std::vector<int32_t> &pages) = 0;
    virtual uint32_t free_pages() const = 0;
    virtual uint32_t used_pages() const = 0;
    virtual uint32_t total_pages() const = 0;
};

/* Per-session executor.
 *
 * Owns transient device-resident scratch buffers (h, norm, attn_out, ffn_*,
 * recurrent_state, etc.) and the logits buffer; reuses them across every
 * forward step. All weight tensors live in QwenWeights and are NOT touched by
 * this class except by pointer. */
class QwenExecutor {
public:
    struct StateSnapshot {
        bool ready = false;
        uint32_t position = 0;
        uint32_t kv_logical_pages = 0;
        uint32_t mtp_prefix_len = 0;
        uint32_t kvmem_registered_pos = 0;
        // kvmem window state: when kvmem is active the assembled window advances
        // in lockstep with position_ as decode/verify tokens append at the
        // window tail. A snapshot/restore (MTP rollback) must round-trip the
        // window query position and page-table length so a rejected verify
        // batch leaves the window exactly where it was before the batch. Under
        // identity (all-fit) selection window_query_pos == position and these
        // are consistent with the position restore; under non-kvmem decode they
        // are zero/false and restore is a no-op.
        bool kvmem_active = false;
        uint32_t window_query_pos = 0;
        uint32_t window_page_count = 0;
        std::unique_ptr<DeviceTensor> h;
        std::vector<std::unique_ptr<DeviceTensor>> recurrent_states;
        std::vector<std::unique_ptr<DeviceTensor>> conv_states;
    };
    struct StateCheckpointSet {
        bool ready = false;
        uint32_t base_position = 0;
        uint32_t count = 0;
        uint32_t h_stride = 0;
        uint32_t checkpoint_stride = 0;
        uint32_t checkpoint_row = 0;
        uint32_t h_checkpoint_row = 0;
        // kvmem window state at the verify batch's base. When the batch ran in a
        // window frame (kvmem_active_), restore_state_checkpoint(index) must roll
        // the window tail back to base + (index+1), exactly mirroring how it
        // rolls position_ back. The verify batch only ever GREW the window at the
        // tail (no mid-batch re-RoPE), so truncating window_pages_host_/_count_
        // and resetting window_query_pos_ is sufficient — no device re-upload.
        bool kvmem_active = false;
        uint32_t window_base_query_pos = 0;
        uint32_t window_base_page_count = 0;
        std::unique_ptr<DeviceTensor> h;
        std::shared_ptr<DeviceTensor> h_shared;
        std::vector<std::unique_ptr<DeviceTensor>> recurrent_states;
        std::vector<std::unique_ptr<DeviceTensor>> conv_states;
        std::vector<std::shared_ptr<DeviceTensor>> recurrent_states_shared;
        std::vector<std::shared_ptr<DeviceTensor>> conv_states_shared;
    };
    struct DecodeStateView {
        uint32_t position = 0;
        uint32_t kv_ctx_size = 0;
        uint32_t kv_page_size = 0;
        uint32_t kv_page_count = 0;
        const int32_t *kv_page_indices_host = nullptr;
        const DeviceTensor *kv_page_indices_device = nullptr;
        const std::vector<std::unique_ptr<DeviceTensor>> *k_cache = nullptr;
        const std::vector<std::unique_ptr<DeviceTensor>> *v_cache = nullptr;
        const std::vector<DeviceTensor *> *k_cache_external = nullptr;
        const std::vector<DeviceTensor *> *v_cache_external = nullptr;
        const std::vector<std::unique_ptr<DeviceTensor>> *recurrent_states = nullptr;
        const std::vector<std::unique_ptr<DeviceTensor>> *conv_states = nullptr;
        const DeviceTensor *hidden = nullptr;
    };
    struct MutableDecodeStateView {
        uint32_t position = 0;
        uint32_t kv_ctx_size = 0;
        uint32_t kv_page_size = 0;
        uint32_t kv_page_count = 0;
        const int32_t *kv_page_indices_host = nullptr;
        const DeviceTensor *kv_page_indices_device = nullptr;
        std::vector<DeviceTensor *> *k_cache_external = nullptr;
        std::vector<DeviceTensor *> *v_cache_external = nullptr;
        std::vector<std::unique_ptr<DeviceTensor>> *k_cache = nullptr;
        std::vector<std::unique_ptr<DeviceTensor>> *v_cache = nullptr;
        std::vector<std::unique_ptr<DeviceTensor>> *recurrent_states = nullptr;
        std::vector<std::unique_ptr<DeviceTensor>> *conv_states = nullptr;
        DeviceTensor *hidden = nullptr;
    };
    struct KvStateSnapshot {
        uint32_t seq_len = 0;
        uint32_t ctx_size = 0;
        uint32_t page_size = 0;
        uint32_t logical_pages = 0;
        std::vector<int32_t> physical_pages;
    };
    struct KvCacheStorage {
        uint64_t physical_slots = 0;
        std::vector<DeviceTensor *> k_cache;
        std::vector<DeviceTensor *> v_cache;
    };
    struct MtpPrefixStateView {
        bool ready = false;
        uint32_t prefix_len = 0;
        uint32_t ctx_size = 0;
        uint32_t page_size = 0;
        uint32_t page_count = 0;
        const int32_t *page_indices_host = nullptr;
        const DeviceTensor *page_indices_device = nullptr;
        DeviceTensor *k_cache = nullptr;
        DeviceTensor *v_cache = nullptr;
        DeviceTensor *prefix_hidden = nullptr;
        DeviceTensor *current_hidden = nullptr;
        DeviceTensor *draft_hidden = nullptr;
    };

    QwenExecutor(const QwenNativeModel &model,
                 const QwenWeights &weights,
                 DeviceBackend &backend,
                 uint32_t kv_ctx_size,
                 KvPhysicalPageAllocator *kv_page_allocator = nullptr,
                 KvCacheStorage *external_kv_cache = nullptr,
                 KvPhysicalPageAllocator *mtp_kv_page_allocator = nullptr,
                 KvCacheStorage *external_mtp_kv_cache = nullptr);
    ~QwenExecutor();

    void reset_state();
    uint32_t position() const { return position_; }
    uint32_t kv_ctx_size() const { return kv_ctx_size_; }
    DecodeStateView decode_state_view() const;
    MutableDecodeStateView mutable_decode_state_view();
    MtpPrefixStateView mtp_prefix_state_view();
    KvStateSnapshot kv_state_snapshot() const;
    void prepare_runtime_state();
    void prepare_kv_pages(uint32_t logical_pos, uint32_t count);
    void prepare_mtp_prefix_pages(uint32_t logical_pos, uint32_t count);
    void set_mtp_prefix_len(uint32_t prefix_len);
    void prepare_decode_token_pages(uint32_t count = 1);
    void advance_position(uint32_t count = 1) { position_ += count; }

    NativeExecutorReport dry_run_token(uint32_t token_id, bool execute_heavy);
    NativeExecutorReport forward_one_token(uint32_t token_id,
                                           bool compute_logits = true);
    // Execute one recurrent layer plus its FFN using the executor's current
    // hidden state. This does not advance position and is used by the
    // continuous-batching body path while standard-attention layers are
    // processed as a cross-request batch.
    NativeExecutorReport forward_recurrent_layer_from_current_hidden(uint32_t layer_index);
    // Batched prefill. Processes `tokens` consecutively as a single forward
    // pass using batched matmuls for the linear projections + FFN. Per-token
    // ops (attention, recurrent state) still iterate sequentially inside.
    // After return, position_ has advanced by tokens.size(). When
    // compute_logits is true, the LM-head logits + argmax correspond to the
    // LAST token in the batch. Chunked prefill can set compute_logits=false
    // for intermediate chunks because only the final prompt token seeds decode.
    // When `row_logits_host` is non-null (verify path only, i.e. row_argmaxes
    // set), each row's full fp32 LM-head logits are copied to host alongside
    // the per-row argmaxes. Used by the MTP speculative-sampling accept test,
    // which needs the target distribution per row (not just its argmax).
    NativeExecutorReport forward_n_tokens(const std::vector<uint32_t> &tokens,
                                          bool compute_logits = true,
                                          std::vector<DeviceArgmax> *row_argmaxes = nullptr,
                                          StateCheckpointSet *state_checkpoints = nullptr,
                                          uint32_t state_checkpoint_count = 0,
                                          bool copy_last_logits = true,
                                          std::vector<std::vector<float>> *row_logits_host = nullptr);

    // The prefill sub-chunk width forward_n_tokens will use for a batch of
    // `total` tokens (the non-verify path). This is the single source of truth
    // for the chunk cap: CLI/env override, the free-memory safety floor, and the
    // bounded KVMem GPU pool headroom cap. Callers that must keep the whole batch
    // resident (e.g. MTP prefix priming reads the last batch's hidden rows) drive
    // their outer loop by this width so forward_n_tokens never re-splits a chunk.
    uint32_t effective_prefill_chunk_size(uint32_t total) const;

    // Diagnostic single-step MTP draft head. Uses the current target
    // pre-output hidden state (`h_`) plus `token_id` and writes MTP logits to
    // the normal logits buffer. This does not perform speculative acceptance.
    NativeExecutorReport forward_mtp_draft(uint32_t token_id);
    std::vector<NativeExecutorReport> forward_mtp_draft_chain(uint32_t token_id,
                                                              uint32_t max_tokens);
    std::vector<NativeExecutorReport> forward_mtp_draft_chain_with_prefix(uint32_t token_id,
                                                                          uint32_t max_tokens);
    std::vector<NativeExecutorReport> forward_mtp_draft_chain_with_prefix_device(uint32_t token_id,
                                                                                 uint32_t max_tokens);
    NativeExecutorReport prime_mtp_prefix_from_last_batch(const std::vector<uint32_t> &tokens,
                                                          uint32_t base_position,
                                                          uint32_t batch_min_override = 0);
    // KVMem long-context variant: logical_base_position identifies the
    // append/page-table location, while rope_base_position is the compact
    // selected-window coordinate used by MTP Q/K.  Keeping these separate is
    // what prevents a million-token session from ever entering MTP RoPE.
    NativeExecutorReport prime_mtp_prefix_from_last_batch_at(
        const std::vector<uint32_t> &tokens,
        uint32_t logical_base_position,
        uint32_t rope_base_position,
        uint32_t batch_min_override = 0);
    NativeExecutorReport prime_mtp_prefix_from_current(uint32_t token,
                                                       uint32_t base_position);
    NativeExecutorReport prime_mtp_prefix_from_current_at(
        uint32_t token,
        uint32_t logical_position,
        uint32_t rope_position);
    NativeExecutorReport replay_tokens_with_mtp_prefix(const std::vector<uint32_t> &tokens,
                                                       uint32_t base_position,
                                                       bool rebuild_prefix,
                                                       double *prefix_seconds = nullptr,
                                                       uint64_t *prefix_ops = nullptr);
    void commit_mtp_prefix(uint32_t prefix_len);
    void commit_mtp_prefix_from_current_hidden(uint32_t prefix_len);
    // MTP prefix construction happens immediately after the target-model
    // chunk and needs that chunk's exact compact RoPE frame.
    uint32_t last_forward_logical_base() const {
        return last_forward_logical_base_;
    }
    uint32_t last_forward_rope_base() const {
        return last_forward_rope_base_;
    }
    uint32_t last_forward_rows() const { return last_forward_rows_; }
    bool kvmem_mtp_local_positions() const {
        return kvmem_mtp_local_positions_;
    }
    // RoPE coordinate used by the most recently committed target token. The
    // logical position may be far beyond the native context, while this value
    // stays inside the current selected window.
    uint32_t last_committed_rope_position() const {
        if (kvmem_active_) {
            return window_query_pos_ > 0 ? window_query_pos_ - 1 : 0;
        }
        return position_ > 0 ? position_ - 1 : 0;
    }
    // Pressure stage-out normally runs at the end of forward_n_tokens.  MTP
    // prefix capture must run first so an evicted block never records an empty
    // MTP tail.  These calls are inert outside bounded KVMem.
    void kvmem_set_defer_prefill_pressure(bool enabled);
    void kvmem_finish_deferred_prefill_pressure();
    StateSnapshot snapshot_state();
    void capture_state(StateSnapshot &snapshot);
    void restore_state(const StateSnapshot &snapshot);
    // Diagnostic query replay: after the ordinary post-prefill semantic
    // selection has been computed, keep that exact context selection, rewind
    // only the suffix beginning at a block-aligned query boundary, restore the
    // recurrent/conv state captured at that boundary, and let the backend
    // prefill the suffix again against the final selected window.  Unlike
    // restore_state(), this deliberately does NOT restore the old pressure
    // window.  Default inference never calls this path.
    void kvmem_begin_query_replay(const StateSnapshot &boundary,
                                  const std::vector<uint32_t> &context_block_ids,
                                  bool reset_recurrent_state = false);
    void kvmem_end_query_replay();
    // ARCHIVED (2026-07-23): diagnostic DeltaNet state interchange. Retained
    // below only as source history; it is not part of the compiled executor API.
#if 0
    void kvmem_export_recurrent_state(
        const std::string &path,
        const std::vector<uint32_t> &source_tokens) const;
    void kvmem_import_recurrent_state(
        const std::string &path,
        const std::vector<uint32_t> &expected_source_tokens);
#endif
    // Start a new local recurrent segment while retaining the assembled
    // normal-attention KVMem window. Used only by controlled transcript-memory
    // construction experiments; ordinary inference never calls it.
    void kvmem_reset_recurrent_state();
    void restore_state_checkpoint(const StateCheckpointSet &checkpoints,
                                  uint32_t index);

    // ---- Prefix-cache reuse (serve / continuous-batching) -----------------
    // Seed this (freshly-reset) executor from a cached, page-aligned prefix.
    // Installs `shared_pages` as logical KV pages [0..n) marked non-owned (the
    // pool pins them; this executor never frees them), restores recurrent +
    // conv state from `recur`, and sets position_ = aligned_len. Subsequent
    // prefill must start at aligned_len (use prefill_offset). aligned_len must
    // equal shared_pages.size() * kv page_size.
    void seed_from_shared_prefix(const std::vector<int32_t> &shared_pages,
                                 const StateSnapshot &recur,
                                 uint32_t aligned_len);
    // Hand the physical KV pages for logical range [logical_start_page..end)
    // to the caller WITHOUT freeing them, and mark the retained pages
    // [0..logical_start_page) as non-owned so this executor's dtor won't free
    // them either (they are now pinned by the prefix cache). Returns the
    // detached physical pages. Used at commit time when the executor keeps
    // reading the shared prefix it just promoted.
    std::vector<int32_t> mark_kv_prefix_shared(uint32_t logical_start_page);
    // Snapshot of the current physical KV pages (logical order). Used to
    // record a freshly-computed prefix into the cache.
    std::vector<int32_t> kv_physical_pages() const;
    uint32_t kv_page_size_public() const { return kv_pages_.page_size; }

    // Per-token batch-scratch footprint in bytes (sum of all *_batch_ tensors
    // at batch=1). Used to size prefill chunks against free device memory.
    uint64_t per_token_scratch_bytes() const;

    // Prefill chunk override: -1 = use env / built-in default (512), 0 =
    // whole-prompt (no chunking), >0 = chunk to this many tokens. Set by
    // the CLI flag `--prefill-chunk N`. When set, takes precedence over
    // QW3_PREFILL_CHUNK; the safety floor based on free device memory still
    // applies.
    void set_prefill_chunk_override(int v) { prefill_chunk_override_ = v; }
    int  prefill_chunk_override() const { return prefill_chunk_override_; }

    // Copy the most recent logits tensor back to host. Returns false if
    // forward_one_token has not been called yet.
    bool copy_last_logits(std::vector<float> &out) const;

    // ---- Block-sparse KV attention (single-session, opt-in) ---------------
    // Master switch. When false (default) the forward path is byte-identical
    // to the pre-block-sparse code: no KvMemStore, no window page table, no
    // re-RoPE. Set once at session start from the CLI flag.
    void set_kvmem_enabled(bool on) { kvmem_enabled_ = on; }
    bool kvmem_enabled() const { return kvmem_enabled_; }
    // True when a CPU or NVMe tier is configured (blocks can be offloaded off
    // GPU). Stable for the session. The kvmem prefix cache uses !kvmem_has_tiers()
    // to gate prompt-end (partial) resume to the dense/untiered case where
    // [0,P) blocks are permanently GPU-resident.
    bool kvmem_has_tiers() const {
        return kvmem_cpu_tier_ != nullptr || kvmem_nvme_tier_ != nullptr;
    }
    // Mark the final user message's token span [begin,end) for query-conditioned
    // multi-token selection. Called by the backend BEFORE prefill. begin==end -> no
    // span -> byte-identical single-token/recency path. prompt_tokens is the full
    // prompt length: it fixes the final block count up front so the incremental
    // per-layer content index (#91) can size its per-layer stride before the first
    // K chunk is captured. (Public: backend-invoked.)
    void kvmem_set_query_span(uint32_t begin, uint32_t end,
                              uint32_t prompt_tokens,
                              bool preserve_content_index = false,
                              bool capture_content_without_query = false); // before prefill
    uint32_t kvmem_query_expected_tokens() const {
        return kvmem_query_end_ > kvmem_query_begin_
            ? kvmem_query_end_ - kvmem_query_begin_ : 0;
    }
    uint32_t kvmem_query_captured_tokens() const {
        return g_query_multi_count_;
    }
    bool kvmem_query_capture_complete() const {
        const uint32_t expected = kvmem_query_expected_tokens();
        return expected > 0 && g_query_multi_ready_ &&
            g_query_multi_count_ >= expected;
    }
    // Publish the incrementally captured mean-K index for the CURRENT logical
    // prefix, even though the full teacher-forced transcript has not yet been
    // consumed. Used only by experimental transcript replay before an
    // intermediate semantic re-selection.
    bool kvmem_publish_captured_prefix(uint32_t scoreable_tokens);
    // Attach diagnostics-only request metadata to score dumps. The context span
    // is prompt-absolute and trace_tag is a stable sample id. Neither value is
    // read by retrieval/scoring code.
    void kvmem_set_trace_metadata(const std::string &trace_tag,
                                  uint32_t context_begin,
                                  uint32_t context_end,
                                  const std::vector<uint32_t> &prompt_tokens);
    void kvmem_set_trace_reselect_event(uint32_t index, uint32_t count) {
        kvmem_trace_event_index_ = index;
        kvmem_trace_event_count_ = count;
    }
    // Clean-query prefill (task #50, QW3_KVMEM_CLEAN_QUERY). Backend-invoked.
    // stash: after a PASS-A isolated question prefill, copy the captured de-RoPE'd
    // query into a persistent buffer that survives reset_state. restore: copy it
    // back into g_query_multi_ and flip ready, so context selection (and decode-time
    // reselects) rank by the recency-free query. set_pin_from_block: force
    // pick_topk to always keep blocks with id >= b (the question + generated live
    // tail); 0xffffffff (default, set by reset_state) => no pin => byte-identical.
    void kvmem_stash_clean_query();
    void kvmem_restore_clean_query();
    void kvmem_set_pin_from_block(uint32_t b) { kvmem_qc_pin_from_block_ = b; }
    // Borrow the pinned CPU-tier buffer from a shared pool instead of allocating
    // it per executor. Set before configure_kvmem(); the pool must outlive this
    // executor. No-op effect when kvmem or the CPU tier is off.
    void set_host_tier_pool(HostTierBufferPool *pool) { host_tier_pool_ = pool; }
    void configure_kvmem(const KvMemStoreConfig &cfg);

    // ---- Window-aware batched verify support (CB-MTP ragged path) ----------
    // When kvmem is active the assembled window advances in lockstep with
    // position_ as decode/verify tokens append at the window tail. The CB-MTP
    // ragged verify executor builds its page metadata from the WINDOW (not the
    // full cache); unlike forward_n_tokens it does NOT self-advance the window,
    // so it must call kvmem_advance_window(chunk) after a successful batched
    // verify. These accessors expose the window frame the metadata builder reads.
    bool kvmem_active() const { return kvmem_active_; }
    // True when query-conditioned selection uses the raw per-token index
    // (g_kraw_multi_, --kvmem-retrieval-method per-token). That index is strided by
    // the per-turn total token count and cannot be fixed-stride at large --ctx, so
    // the backend refuses above-budget session reuse (M-reuse) in this mode and
    // falls back to a full cold prefill (correct, just unoptimized).
    bool kvmem_qc_pertoken() const { return kvmem_qc_pertoken_; }
    uint32_t window_query_pos() const { return window_query_pos_; }
    uint32_t window_page_count() const { return window_page_count_; }
    const std::vector<int32_t> &window_pages_host() const {
        return window_pages_host_;
    }

    // ---- Tier residency reporting (diagnostics) ----------------------------
    // Where the block-sparse KV currently lives across the GPU bounded pool /
    // CPU pinned tier / NVMe tier. Counts come from the live allocators; bytes
    // are derived from the per-block estimate. All-zero when kvmem is disabled.
    // Used to report the GPU/CPU/NVMe KV footprint during long-context sweeps.
    struct KvMemTierUsage {
        bool     enabled = false;
        bool     active = false;
        uint64_t total_blocks = 0;
        uint64_t block_bytes = 0;
        bool     gpu_pool = false;          // bounded pool present (i.e. spilling)
        uint64_t gpu_used_bytes = 0;
        uint64_t gpu_capacity_bytes = 0;
        bool     cpu_tier = false;
        uint64_t cpu_used_bytes = 0;
        uint64_t cpu_capacity_bytes = 0;
        uint64_t cpu_raw_k_bytes = 0;
        uint64_t cpu_spill_bytes = 0;
        bool     nvme_tier = false;
        uint64_t nvme_used_bytes = 0;
        uint64_t nvme_capacity_bytes = 0;
    };
    KvMemTierUsage kvmem_tier_usage() const;
    // Extend the assembled window so `n` verify tokens can append at the window
    // tail. True KV for [true_base_pos, true_base_pos+n) must already be
    // allocated (caller runs prepare_kv_pages first). No-op unless kvmem is
    // active. Mirrors the chunk_bs extend in forward_n_tokens.
    void kvmem_extend_window_for_verify(uint32_t n, uint32_t true_base_pos) {
        if (!kvmem_active_) return;
        kvmem_extend_window_for_decode_n(n, true_base_pos);
    }
    // Advance the window tail by `n` after a BATCHED verify kernel appended `n`
    // tokens at window slot window_query_pos_. forward_one_token /
    // forward_n_tokens self-advance the window; the ragged batched-prefill path
    // bypasses them and must call this on the success path only. No-op unless
    // kvmem is active.
    void kvmem_advance_window(uint32_t n) {
        if (!kvmem_active_) return;
        window_query_pos_ += n;
    }

    // Register newly-appended context tokens with the block store (called after
    // prefill / each committed decode token grows the context). No-op when
    // block-sparse is disabled.
    void kvmem_register_append(uint32_t n_new_tokens);
    // Rewind the block store to exactly `token_pos` tokens after restore_state()
    // has already rewound position/KV-pages/recurrent/window to a prompt-end
    // checkpoint (kvmem prefix cache). Drops trailing blocks, releases their
    // CPU/NVMe tier slots (GPU pages were freed by restore_state's KV-page
    // truncate, so they are NOT released here), and invalidates the per-session
    // selection indices so the pre-suffix pressure-window rebuild and the final
    // semantic reselect both operate over the correct block count. No-op when
    // block-sparse is disabled or token_pos is already the live end.
    void kvmem_truncate_to(uint32_t token_pos);
    // Spill cold blocks to the tier mid-prefill if the bounded GPU page pool is
    // about to run short. `next_chunk_tokens` is the size of the upcoming prefill
    // append, so the offload fires while there is still room for it (a full chunk
    // can grab >100 pages at once, so a "only when nearly empty" trigger would
    // let the pool exhaust mid-chunk and throw). No-op when not bounded/tiered.
    void kvmem_maybe_prefill_offload(uint32_t next_chunk_tokens);
    // SSD Opt2+ write-through producer. Call only after every KV source for
    // [0, completed_pos) is durable (for local-position MTP this means after
    // prime_mtp_prefix_from_last_batch). Full newly-completed blocks are packed
    // to pinned slabs on the copy stream without changing GPU residency; their
    // D2H overlaps the following prefill chunk and SSD persistence follows in
    // the existing bounded background writer.
    void kvmem_prefill_writeback(uint32_t completed_pos);

    // Assemble the deterministic long-prefill working set: configured sink
    // prefix plus the newest blocks filling the rest of the selection budget.
    // Unlike kvmem_reselect(), this never runs or consumes a semantic scorer.
    uint32_t kvmem_reselect_prefill_pressure();

    // Normalize an already-sparse window before appending a new prefill suffix.
    // This is what prevents a previous turn's semantic working set from shaping
    // the next above-budget prefill before physical page pressure first fires.
    // Cold/all-fit paths are no-ops and transition through the normal pressure
    // trigger only after they actually outgrow the resident pool.
    void kvmem_prepare_prefill_window(uint32_t upcoming_tokens);
    // Keep the currently assembled semantic selection while appending a
    // teacher-forced suffix. If the suffix would exhaust the reserved GPU
    // headroom, the executor fails explicitly instead of silently replacing
    // the semantic context with sink+recent.
    void kvmem_set_keep_selected_prefill(bool keep) {
        kvmem_keep_selected_prefill_ = keep;
    }

    // Re-select the working set from the built-in cumulative-attention top-k
    // and assemble it: re-RoPE each moved block in place (per attention layer)
    // and install the window page table + window query position used by the
    // next decode steps. No-op when disabled. Returns the number of blocks in
    // the assembled window (0 when disabled).
    uint32_t kvmem_reselect();
    // Split form of kvmem_reselect() for overlapping tier prefetch with
    // independent compute. prepare computes the selection plan and starts
    // CPU/NVMe -> GPU prefetch; finish waits, assembles the window, and spills
    // deselected blocks. No attention over KVMem may run between prepare and
    // finish.
    uint32_t kvmem_prepare_reselect();
    uint32_t kvmem_finish_reselect();

    // Install an explicit block-ID selection instead of the built-in top-k
    // (the external-selector hook; also used by tests to force a fixed set,
    // e.g. the identity all-blocks selection that must reproduce the plain
    // path byte-for-byte). Assembles the same way as kvmem_reselect.
    uint32_t kvmem_set_selection(const std::vector<uint32_t> &block_ids);

    const KvMemStore *block_store() const { return block_store_.get(); }

    // ---- KVMem component timing (env QW3_KVMEM_TIMING; OFF by default) ------
    // Process-global wall-clock breakdown of the tier/selection components used
    // by the latency-breakdown harness. Snapshot at request admit, emit the
    // delta at finish. When enabled the GPU-async regions add a device sync so
    // their kernel time is captured -- this perturbs throughput, so measure
    // throughput WITHOUT the flag and the breakdown WITH it.
    struct KvMemTimingSnapshot {
        uint64_t retrieval_ns = 0;
        uint64_t stage_in_ns = 0;
        uint64_t stage_out_ns = 0;
        uint64_t assemble_ns = 0;
        // assemble_ns is split into its three GPU substeps: window page-table
        // construction (the "virtual page" reordering + sync_window_pages_device
        // H2D), re-RoPE of moved blocks (rope_block_remap), and k̄ recompute
        // (block_kmean). Captured only under the timing flag (extra device syncs).
        uint64_t assemble_pages_ns = 0;
        uint64_t assemble_rerope_ns = 0;
        uint64_t assemble_kbar_ns = 0;
        uint32_t retrieval_calls = 0;
        uint32_t stage_in_calls = 0;
        uint32_t stage_out_calls = 0;
        uint32_t assemble_calls = 0;
        uint32_t stage_in_blocks = 0;
        uint32_t stage_out_blocks = 0;
    };
    static bool kvmem_timing_enabled();
    static KvMemTimingSnapshot kvmem_timing_snapshot();
    static void kvmem_timing_emit_delta(const char *tag,
                                        const KvMemTimingSnapshot &baseline);

private:
    struct KvPageTable {
        uint32_t page_size = 16;
        uint32_t max_pages = 0;
        std::string alloc_mode = "identity";
        std::vector<int32_t> pages;
        // Per-logical-page ownership. true = this table allocated the physical
        // page and must release it on reset/truncate/dtor. false = the page is
        // borrowed from a prefix-cache entry (pinned elsewhere); never release.
        // Always the same length as `pages`.
        std::vector<bool> owned;
        std::unique_ptr<DeviceTensor> device_pages;
        KvPhysicalPageAllocator *allocator = nullptr;
        uint32_t device_synced = 0;

        void configure(uint32_t ctx_size, KvPhysicalPageAllocator *page_allocator);
        void set_allocator(KvPhysicalPageAllocator *page_allocator);
        void reset();
        void ensure_pages(DeviceBackend &backend, uint32_t ctx_size,
                          uint32_t logical_pos, uint32_t count);
        void validate_physical_capacity(uint64_t physical_slots,
                                        const char *label) const;
        void truncate_to_logical_pages(uint32_t logical_pages);
        bool logical_page_resident(uint32_t logical_page) const;
        int32_t ensure_logical_page_resident(DeviceBackend &backend,
                                             uint32_t logical_page);
        // Allocate many incoming ranges and publish all page-table changes
        // with one compact H2D upload.
        void ensure_logical_page_ranges_resident(
            DeviceBackend &backend,
            const std::vector<std::pair<uint32_t, uint32_t>> &ranges);
        void release_logical_pages(DeviceBackend &backend,
                                   uint32_t logical_start,
                                   uint32_t count);
        // Release many disjoint logical ranges with one allocator update and
        // one compact page-table upload spanning the touched interval. This
        // avoids one tiny pageable H2D operation per KVMem block during a
        // batched stage-out.
        void release_logical_page_ranges(
            DeviceBackend &backend,
            const std::vector<std::pair<uint32_t, uint32_t>> &ranges);
        int32_t allocate_physical_page(uint32_t logical_page) const;
        // Install pre-existing (pinned, cache-owned) physical pages as logical
        // pages [0..shared.size()) without allocating. Must be called on a
        // freshly-reset table (pages empty). The pages are marked non-owned so
        // they are never released by this table.
        void adopt_shared_pages(DeviceBackend &backend,
                                const std::vector<int32_t> &shared);
        // Hand the physical pages for logical range [logical_start..end) to the
        // caller WITHOUT releasing them, and drop them from this table. Used to
        // transfer freshly-computed pages into a prefix-cache entry so the
        // executor dtor won't free them. Returns the detached physical pages.
        std::vector<int32_t> detach_pages_from(uint32_t logical_start);
        const int32_t *host_indices() const { return pages.data(); }
        const DeviceTensor &device_indices() const { return *device_pages; }
        uint32_t count() const { return static_cast<uint32_t>(pages.size()); }
        uint64_t physical_slots() const;
    };

    void begin_record_timing(bool enabled) const;
    void record(NativeExecutorReport &report, const std::string &op) const;
    void ensure_scratch();
    void allocate_kvmem_gpu_cache(uint64_t physical_slots);
    void allocate_kvmem_mtp_gpu_cache(uint64_t physical_slots);
    void ensure_mtp_scratch();
    void ensure_mtp_batch_scratch(uint32_t batch);
    void ensure_logits_batch_scratch(uint32_t batch);
    void ensure_kv_pages(uint32_t logical_pos, uint32_t count);
    const int32_t *kv_page_indices() const { return kv_pages_.host_indices(); }
    const DeviceTensor &kv_page_indices_device() const { return kv_pages_.device_indices(); }
    uint32_t kv_page_count() const { return kv_pages_.count(); }
    uint32_t kv_page_size() const { return kv_pages_.page_size; }
    DeviceTensor &k_cache(uint32_t layer);
    DeviceTensor &attention_k_cache(uint32_t layer);
    DeviceTensor &v_cache(uint32_t layer);
    DeviceTensor &mtp_k_cache();
    DeviceTensor &mtp_v_cache();
    bool has_external_kv_cache() const { return external_kv_cache_ != nullptr; }
    NativeExecutorReport forward_mtp_draft_from(uint32_t token_id,
                                                const DeviceTensor &h_input,
                                                uint32_t rope_pos,
                                                uint32_t cache_pos,
                                                uint32_t seq_len,
                                                bool compute_logits = true,
                                                DeviceArgmaxBuffer *argmax_out = nullptr,
                                                uint32_t argmax_out_index = 0,
                                                const DeviceArgmaxBuffer *token_source = nullptr,
                                                uint32_t token_source_index = 0,
                                                bool window_frame = false,
                                                bool kv_only = false);

    const QwenNativeModel &model_;
    const QwenWeights &weights_;
    DeviceBackend &backend_;

    void ensure_batch_scratch(uint32_t batch);

    bool scratch_ready_ = false;
    std::unique_ptr<DeviceTensor> h_;
    std::unique_ptr<DeviceTensor> norm_;
    std::unique_ptr<DeviceTensor> attn_out_;
    std::unique_ptr<DeviceTensor> ffn_gate_;
    std::unique_ptr<DeviceTensor> ffn_up_;
    std::unique_ptr<DeviceTensor> ffn_mid_;
    std::unique_ptr<DeviceTensor> ffn_out_;
    std::unique_ptr<DeviceTensor> proj_;
    std::unique_ptr<DeviceTensor> gate_proj_;
    std::unique_ptr<DeviceTensor> alpha_;
    std::unique_ptr<DeviceTensor> beta_;
    std::unique_ptr<DeviceTensor> core_;

    // Batched scratch for forward_n_tokens. Sized to `batch_capacity_` rows
    // each. Allocated on demand (and grown lazily) by ensure_batch_scratch.
    uint32_t batch_capacity_ = 0;
    std::unique_ptr<DeviceTensor> h_batch_;
    std::unique_ptr<DeviceTensor> norm_batch_;
    std::unique_ptr<DeviceTensor> attn_out_batch_;
    std::unique_ptr<DeviceTensor> ffn_gate_batch_;
    std::unique_ptr<DeviceTensor> ffn_up_batch_;
    std::unique_ptr<DeviceTensor> ffn_mid_batch_;
    std::unique_ptr<DeviceTensor> ffn_out_batch_;
    std::unique_ptr<DeviceTensor> proj_batch_;
    std::unique_ptr<DeviceTensor> gate_proj_batch_;
    std::unique_ptr<DeviceTensor> alpha_batch_;
    std::unique_ptr<DeviceTensor> beta_batch_;
    std::unique_ptr<DeviceTensor> core_batch_;
    std::unique_ptr<DeviceTensor> q_batch_;
    std::unique_ptr<DeviceTensor> k_batch_;
    std::unique_ptr<DeviceTensor> v_batch_;
    std::unique_ptr<DeviceTensor> mid_batch_;
    // Preallocated scratch for the per-token conv output (size = conv_dim).
    // Was a cudaMalloc/cudaFree per call inside recurrent_single_token.
    std::unique_ptr<DeviceTensor> conv_out_;
    // Batched scratch for the recurrent conv output during prefill: sized
    // [batch_capacity_, max_recurrent_qkv]. Used by recurrent_batch as the
    // intermediate buffer between conv -> l2_norm -> deltanet.
    std::unique_ptr<DeviceTensor> conv_out_batch_;
    // Per-layer DeltaNet state and conv1d ring buffer. Indexed by absolute
    // layer index; entries for non-recurrent (full attention) layers stay
    // null. This is essential for correctness: each recurrent layer keeps
    // its own [num_v_heads * head_v_dim * head_k_dim] state and
    // [(conv_k - 1) * conv_dim] conv buffer that persist across tokens.
    std::vector<std::unique_ptr<DeviceTensor>> recurrent_states_;
    std::vector<std::unique_ptr<DeviceTensor>> conv_states_;
    std::unique_ptr<DeviceTensor> q_;
    std::unique_ptr<DeviceTensor> k_;
    std::unique_ptr<DeviceTensor> v_;
    std::unique_ptr<DeviceTensor> mid_;
    std::unique_ptr<DeviceTensor> logits_;
    std::unique_ptr<DeviceTensor> scores_;
    // One [ctx_size * n_kv_heads * head_dim] tensor per standard attention layer.
    std::vector<std::unique_ptr<DeviceTensor>> k_cache_;
    std::vector<std::unique_ptr<DeviceTensor>> v_cache_;
    KvCacheStorage *external_kv_cache_ = nullptr;
    KvCacheStorage *external_mtp_kv_cache_ = nullptr;
    std::unique_ptr<KvPhysicalPageAllocator> kvmem_gpu_page_pool_;
    std::vector<std::unique_ptr<DeviceTensor>> kvmem_k_cache_storage_;
    std::vector<std::unique_ptr<DeviceTensor>> kvmem_v_cache_storage_;
    KvCacheStorage kvmem_kv_cache_view_;
    // Bounded GPU page pool for the MTP layer's KV, sibling of
    // kvmem_gpu_page_pool_. Engaged only on the single-request internal-MTP
    // path when the main kvmem pool is engaged; keeps MTP GPU residency to one
    // layer's window instead of the full-context dense cache.
    std::unique_ptr<KvPhysicalPageAllocator> kvmem_mtp_gpu_page_pool_;
    bool kvmem_mtp_tiered_ = false;

    bool mtp_scratch_ready_ = false;
    std::unique_ptr<DeviceTensor> mtp_h_;
    std::unique_ptr<DeviceTensor> mtp_embd_;
    std::unique_ptr<DeviceTensor> mtp_enorm_;
    std::unique_ptr<DeviceTensor> mtp_hnorm_;
    std::unique_ptr<DeviceTensor> mtp_concat_;
    std::unique_ptr<DeviceTensor> mtp_k_cache_;
    std::unique_ptr<DeviceTensor> mtp_v_cache_;
    std::unique_ptr<DeviceTensor> mtp_zero_h_;
    std::unique_ptr<DeviceTensor> mtp_prefix_h_;
    KvPageTable mtp_kv_pages_;
    uint32_t mtp_batch_capacity_ = 0;
    std::unique_ptr<DeviceTensor> mtp_h_input_batch_;
    std::unique_ptr<DeviceTensor> mtp_h_batch_;
    std::unique_ptr<DeviceTensor> mtp_norm_batch_;
    std::unique_ptr<DeviceTensor> mtp_concat_batch_;
    std::unique_ptr<DeviceTensor> mtp_q_batch_;
    std::unique_ptr<DeviceTensor> mtp_k_batch_;
    std::unique_ptr<DeviceTensor> mtp_v_batch_;
    std::unique_ptr<DeviceTensor> mtp_mid_batch_;
    std::unique_ptr<DeviceTensor> mtp_ffn_gate_batch_;
    std::unique_ptr<DeviceTensor> mtp_ffn_up_batch_;
    std::unique_ptr<DeviceTensor> mtp_ffn_mid_batch_;
    std::unique_ptr<DeviceTensor> mtp_ffn_out_batch_;
    std::unique_ptr<DeviceArgmaxBuffer> mtp_draft_argmaxes_;
    uint32_t mtp_draft_argmax_capacity_ = 0;
    uint32_t mtp_prefix_len_ = 0;
    uint32_t logits_batch_capacity_ = 0;
    std::unique_ptr<DeviceTensor> logits_batch_;

    uint32_t kv_ctx_size_ = 0;
    uint32_t position_ = 0;
    uint32_t last_forward_logical_base_ = 0;
    uint32_t last_forward_rope_base_ = 0;
    uint32_t last_forward_rows_ = 0;
    int      prefill_chunk_override_ = -1;
    KvPageTable kv_pages_;

    // ---- Block-sparse KV attention state (inert unless enabled) -----------
    // All zero/null when kvmem_enabled_ is false, so the forward path
    // takes the identical pre-block-sparse branches.
    bool kvmem_enabled_ = false;
    uint32_t kvmem_registered_pos_ = 0;
    // Suppress the automatic sink+recent pressure reselect while the short
    // query suffix is being replayed into an already-final semantic window.
    // The selected suffix blocks were removed first, so replay consumes the
    // same GPU-pool capacity they occupied before the rewind.
    bool kvmem_query_replay_active_ = false;
    bool kvmem_keep_selected_prefill_ = false;
    // True once a selection has been assembled this session; gates the decode
    // window substitution. Cleared by reset_state().
    bool kvmem_active_ = false;
    bool kvmem_immutable_source_k_ = false;
    bool kvmem_mtp_local_positions_ = false;
    bool kvmem_defer_prefill_pressure_ = false;
    uint32_t kvmem_deferred_prefill_tokens_ = 0;
    std::unique_ptr<KvMemStore> block_store_;
    // Window page table: selected blocks' physical pages in ascending order.
    // There is exactly one GPU K/V copy; immutable mode bounds its re-RoPE
    // drift with an authoritative unrotated CPU raw-K mirror.
    std::vector<int32_t> window_pages_host_;
    std::unique_ptr<DeviceTensor> window_pages_device_;
    uint32_t window_page_count_ = 0;
    // MTP-draft mirror of the main window page table. The MTP draft head has its
    // own KV cache (mtp_kv_pages_), so the window is a separate page-pointer
    // reordering over the SAME selected blocks (lockstep with the main window;
    // its length tracks window_query_pos_). Built + re-RoPE'd in kvmem_assemble.
    std::vector<int32_t> mtp_window_pages_host_;
    std::unique_ptr<DeviceTensor> mtp_window_pages_device_;
    uint32_t mtp_window_page_count_ = 0;
    std::unique_ptr<PinnedKvTier> kvmem_cpu_tier_;
    std::unique_ptr<NvmeKvTier> kvmem_nvme_tier_;
    std::unique_ptr<HostBuffer> kvmem_cpu_bytes_;
    // Optional per-executor persistent workers for the many small parallel
    // gather/scatter jobs issued by long-context reselection. Keeping the
    // implementation opaque here avoids exposing synchronization details in
    // the executor interface.
    std::unique_ptr<KvmemCpuWorkerPool> kvmem_cpu_worker_pool_;
    // When CPU V stage-in overlaps immutable raw-K materialization, each side
    // gets an independent queue. This avoids serializing on the primary pool's
    // run mutex and lets sparse V gathering use more workers than block-major
    // raw K when V is the measured critical branch.
    std::unique_ptr<KvmemCpuWorkerPool> kvmem_stagein_worker_pool_;
    bool kvmem_stagein_assembly_overlap_enabled_ = false;
    bool kvmem_stagein_assembly_overlap_active_ = false;
    // CPU-only opt_2/3 may retain a clean spill record after CPU->GPU stage-in.
    // This turns a later eviction of the same block into metadata/page release
    // instead of another GPU->CPU copy. It is enabled only when the CPU budget,
    // after reserving worst-case immutable raw-K storage, can back every context
    // block; smaller-memory configurations retain exclusive CPU semantics.
    bool kvmem_inclusive_cpu_backing_ = false;
    // Shared recycler for the pinned CPU-tier buffer. When set (continuous-
    // batching path), configure_kvmem borrows the buffer from here instead of
    // cudaHostAlloc-ing per request; the destructor returns it. Borrowed, owned
    // by the backend and must outlive this executor. Null => own per-executor.
    HostTierBufferPool *host_tier_pool_ = nullptr;
    // Pinned single-block staging buffer for stage-out. D2H into pinned host
    // memory lets a block's K/V page copies queue asynchronously on the copy
    // stream; a pageable destination forces the driver to serialize each copy
    // through an internal bounce buffer, which was the dominant stage-out cost.
    // Lazily grown to one block.
    std::unique_ptr<HostBuffer> kvmem_stage_pinned_;
    // Immutable-K v3: unrotated K is authoritative in demand-allocated,
    // pageable CPU chunks. The compatibility layout is
    // [standard_layer, token-within-chunk, per_pos]. The optimized layout is
    // [block-within-chunk, standard_layer, token-within-block, per_pos], so
    // assembly gathers one ~MiB block instead of 16 independent 64 KiB layer
    // slices. GPU stores only the current active/window-baked K.
    // Capture/materialization still use modest pinned bounce buffers.
    uint32_t kvmem_raw_k_chunk_tokens_ = 2048;
    bool kvmem_raw_k_block_major_ = false;
    std::vector<std::unique_ptr<uint8_t[]>> kvmem_raw_k_chunks_;
    uint64_t kvmem_raw_k_mirror_bytes_ = 0;
    uint64_t kvmem_raw_k_row_bytes_ = 0;
    std::vector<uint8_t> kvmem_raw_k_valid_tokens_;
    std::vector<uint32_t> kvmem_raw_layers_;
    std::vector<int32_t> kvmem_raw_layer_slot_;
    std::vector<std::unique_ptr<DeviceTensor>> kvmem_raw_capture_dev_;
    uint32_t kvmem_raw_capture_rows_ = 0;
    std::unique_ptr<HostBuffer> kvmem_raw_capture_host_;
    std::unique_ptr<DeviceTensor> kvmem_raw_transfer_dev_;
    std::unique_ptr<HostBuffer> kvmem_raw_transfer_host_;
    uint32_t kvmem_raw_transfer_blocks_ = 0;
    uint32_t kvmem_raw_transfer_block_cap_ = 128;
    struct RawMaterializeSlot {
        std::unique_ptr<DeviceTensor> device;
        std::unique_ptr<HostBuffer> host;
        std::unique_ptr<DeviceTransferFence> h2d_done;
        std::unique_ptr<DeviceTransferFence> compute_done;
        uint64_t device_elements = 0;
        uint64_t host_bytes = 0;
    };
    std::array<RawMaterializeSlot, 2> kvmem_raw_pipeline_slots_;
    bool kvmem_raw_pipeline_enabled_ = false;
    uint64_t kvmem_assembly_raw_gather_ns_ = 0;
    uint64_t kvmem_assembly_raw_h2d_submit_ns_ = 0;
    uint64_t kvmem_assembly_raw_h2d_wait_ns_ = 0;
    uint64_t kvmem_assembly_raw_bytes_ = 0;
    uint32_t kvmem_assembly_raw_batches_ = 0;
    // Assembly optimization: a persistent FP32 [position, RoPE pair, sin/cos]
    // table preserves the legacy de-rotate/re-rotate arithmetic while sharing
    // transcendental results across KV heads and attention layers.
    bool kvmem_rope_table_enabled_ = false;
    bool kvmem_rope_table_explicit_ = false;
    uint32_t kvmem_rope_table_positions_ = 0;
    std::unique_ptr<DeviceTensor> kvmem_rope_sincos_;
    int64_t kvmem_raw_decode_block_start_ = -1;
    uint32_t kvmem_raw_decode_first_row_ = 0;
    uint32_t kvmem_raw_decode_rows_ = 0;
    // The MTP head is one additional standard-attention layer.  In the
    // long-context local-position path its unrotated K is authoritative here;
    // the GPU MTP K cache is only a materialized selected-window view.  MTP V
    // remains in the ordinary CPU/NVMe spill record because it has no RoPE.
    std::vector<std::unique_ptr<uint8_t[]>> kvmem_raw_mtp_k_chunks_;
    uint64_t kvmem_raw_mtp_k_mirror_bytes_ = 0;
    std::vector<uint8_t> kvmem_raw_mtp_k_valid_tokens_;
    std::unique_ptr<DeviceTensor> kvmem_raw_mtp_capture_dev_;
    uint32_t kvmem_raw_mtp_capture_rows_ = 0;
    std::unique_ptr<HostBuffer> kvmem_raw_mtp_capture_host_;
    std::unique_ptr<DeviceTensor> kvmem_raw_mtp_transfer_dev_;
    std::unique_ptr<HostBuffer> kvmem_raw_mtp_transfer_host_;
    // Immutable mode shares one strict host-memory budget between the
    // demand-allocated raw-K authority and a pageable, sparse CPU V cache.
    // Sparse slots are backed by lazy ~64 MiB pageable slabs: this avoids one
    // mmap-sized allocation per block without pinning tens of GiB. Empty slabs
    // are released, and their full allocation (not just live slots) is charged
    // to the shared budget. Bounded pinned staging is handled independently.
    bool kvmem_sparse_cpu_tier_ = false;
    uint64_t kvmem_cpu_budget_bytes_ = 0;
    uint64_t kvmem_cpu_sparse_bytes_ = 0;
    struct KvMemCpuSparseSlab {
        std::unique_ptr<uint8_t[]> data;
        uint32_t live_slots = 0;
        uint32_t capacity_slots = 0;
    };
    std::vector<KvMemCpuSparseSlab> kvmem_cpu_sparse_slabs_;
    std::vector<uint8_t> kvmem_cpu_sparse_slot_live_;
    uint32_t kvmem_cpu_sparse_slots_per_slab_ = 1;
    uint64_t kvmem_cpu_sparse_slot_bytes_ = 0;
    struct KvMemPrefetchBlock {
        uint32_t block_id = 0;
        KvTier from = KvTier::GPU;
    };
    struct KvMemPrefetchNvmeRead {
        uint32_t block_id = 0;
        uint64_t bytes = 0;
        int32_t slot = -1;
        uint64_t batch_offset = 0;
        std::vector<uint8_t> buffer;
    };
    struct KvMemPrefetchCpuRead {
        uint32_t block_id = 0;
        uint64_t bytes = 0;
        int32_t slot = -1;
        uint64_t batch_offset = 0;
    };
    struct KvMemPrefetchCpuBatch {
        size_t read_begin = 0;
        size_t read_end = 0;
        std::unique_ptr<HostBuffer> buffer;
        std::vector<int32_t> src_page_indices;
        std::vector<int32_t> dst_page_indices;
        std::unique_ptr<DeviceTransferFence> fence;
    };
    struct KvMemPrefetchNvmeBatch {
        size_t read_begin = 0;
        size_t read_end = 0;
        std::unique_ptr<HostBuffer> buffer;
        std::vector<NvmeIoSpan> spans;
        std::future<NvmeBatchIoStats> future;
    };
    struct KvMemPrefetchPerf {
        uint64_t start_enter_ns = 0;
        uint64_t start_exit_ns = 0;
        uint64_t finish_enter_ns = 0;
        uint64_t finish_exit_ns = 0;
        uint64_t cpu_gather_ns = 0;
        uint64_t cpu_h2d_enqueue_ns = 0;
        uint64_t cpu_h2d_wait_ns = 0;
        uint64_t nvme_read_ns = 0;
        uint64_t nvme_wait_ns = 0;
        uint64_t pending_write_wait_ns = 0;
        uint64_t nvme_h2d_enqueue_ns = 0;
        uint64_t h2d_wait_ns = 0;
        uint64_t cpu_bytes = 0;
        uint64_t nvme_bytes = 0;
        uint64_t nvme_read_syscalls = 0;
        uint32_t cpu_h2d_batches = 0;
        uint32_t nvme_read_batches = 0;
        uint32_t cpu_blocks = 0;
        uint32_t nvme_blocks = 0;
    };
    struct KvMemPrefetchState {
        bool active = false;
        bool queued_h2d = false;
        // Frozen when prefetch starts.  In local-position MTP mode historical
        // V is an independently tiered source and must be restored even while
        // the newest accepted-token MTP prefix is temporarily catching up.
        bool stage_mtp_pages = false;
        std::vector<KvMemPrefetchBlock> blocks;
        std::vector<KvMemPrefetchCpuRead> cpu_reads;
        bool bulk_cpu = false;
        size_t next_cpu_read = 0;
        std::deque<KvMemPrefetchCpuBatch> cpu_batches;
        std::vector<KvMemPrefetchNvmeRead> nvme_reads;
        std::future<void> nvme_future;
        bool bulk_nvme = false;
        size_t next_nvme_read = 0;
        std::deque<KvMemPrefetchNvmeBatch> nvme_batches;
        KvMemPrefetchPerf perf;
    };
    struct KvMemStageOutPerf {
        uint64_t total_ns = 0;
        uint64_t canonicalize_and_d2h_ns = 0;
        uint64_t cpu_copy_ns = 0;
        uint64_t nvme_write_ns = 0;
        uint64_t bytes = 0;
        uint32_t blocks = 0;
        uint32_t cpu_blocks = 0;
        uint32_t nvme_blocks = 0;
        uint32_t clean_blocks = 0;
        uint64_t clean_bytes_avoided = 0;
        uint64_t async_submit_ns = 0;
        uint64_t async_gather_ns = 0;
        uint64_t async_backpressure_ns = 0;
        uint64_t async_submitted_bytes = 0;
        uint32_t async_batches = 0;
    };
    struct KvMemPendingWriteBatch {
        struct CpuCopy {
            uint32_t block_id = 0;
            int32_t slot = -1;
            uint8_t *dst = nullptr;
            uint64_t buffer_offset = 0;
            uint64_t bytes = 0;
        };
        std::shared_ptr<std::vector<uint8_t>> buffer;
        std::shared_ptr<HostBuffer> pinned_buffer;
        uint64_t buffer_bytes = 0;
        bool proactive = false;
        std::vector<NvmeIoSpan> spans;
        std::vector<uint32_t> block_ids;
        std::vector<CpuCopy> cpu_copies;
        std::future<NvmeBatchIoStats> future;
        uint64_t submit_ns = 0;
    };
    struct KvMemProactiveD2hBatch {
        std::shared_ptr<HostBuffer> buffer;
        uint64_t bytes = 0;
        uint32_t completed_pos = 0;
        std::vector<NvmeIoSpan> spans;
        std::vector<uint32_t> block_ids;
        std::vector<KvMemPendingWriteBatch::CpuCopy> cpu_copies;
        std::vector<int32_t> src_page_indices;
        std::vector<int32_t> dst_page_indices;
        std::unique_ptr<DeviceTransferFence> fence;
        uint64_t submit_ns = 0;
    };
    struct KvMemReselectPerf {
        bool active = false;
        uint64_t sequence = 0;
        uint64_t start_ns = 0;
        uint64_t selection_ns = 0;
        uint64_t stage_out_ns = 0;
    };
    KvMemPrefetchState kvmem_prefetch_;
    KvMemPrefetchPerf kvmem_last_prefetch_perf_;
    KvMemStageOutPerf kvmem_last_stage_out_perf_;
    KvMemReselectPerf kvmem_reselect_perf_;
    std::deque<KvMemPendingWriteBatch> kvmem_pending_writes_;
    // Completed write slabs are recycled instead of repeatedly allocating and
    // faulting 64 MiB pageable buffers on the stage-out critical path.
    std::deque<std::shared_ptr<std::vector<uint8_t>>>
        kvmem_free_write_slabs_;
    // SSD prefill write-through owns two chunk-sized slab sets by default.
    // With 64 MiB I/O slabs a 2048-token chunk of the current model occupies
    // two slabs, so four slabs are needed to avoid recycling the previous
    // chunk's SSD-owned storage before the next D2H can start. A slab remains
    // alive through D2H and its background SSD write, so GPU residency can be
    // retained while persistence progresses independently.
    std::deque<std::shared_ptr<HostBuffer>>
        kvmem_free_writeback_slabs_;
    size_t kvmem_writeback_slab_count_ = 0;
    std::deque<KvMemProactiveD2hBatch>
        kvmem_proactive_d2h_batches_;
    bool kvmem_prefill_writeback_enabled_ = false;
    uint32_t kvmem_writeback_next_block_ = 0;
    uint64_t kvmem_writeback_d2h_bytes_ = 0;
    uint64_t kvmem_writeback_d2h_wait_ns_ = 0;
    uint32_t kvmem_writeback_d2h_batches_ = 0;
    uint32_t kvmem_writeback_blocks_ = 0;
    // Two pinned D2H slabs are enough to overlap CPU scatter of batch N with
    // the copy-stream transfer of batch N+1. They are allocated only for
    // CPU-only opt_2/3 profiles; SSD profiles keep their existing writer path.
    std::deque<std::unique_ptr<HostBuffer>> kvmem_free_stageout_slabs_;
    std::deque<std::unique_ptr<HostBuffer>> kvmem_free_read_slabs_;
    size_t kvmem_read_slab_count_ = 0;
    uint64_t kvmem_async_write_completed_bytes_ = 0;
    uint64_t kvmem_async_write_completed_syscalls_ = 0;
    uint64_t kvmem_async_write_completed_ns_ = 0;
    void kvmem_submit_write_batch(
        std::shared_ptr<std::vector<uint8_t>> buffer,
        std::vector<NvmeIoSpan> spans,
        std::vector<uint32_t> block_ids);
    void kvmem_submit_pinned_write_batch(
        std::shared_ptr<HostBuffer> buffer, uint64_t bytes,
        std::vector<NvmeIoSpan> spans,
        std::vector<uint32_t> block_ids,
        std::vector<KvMemPendingWriteBatch::CpuCopy> cpu_copies);
    void kvmem_submit_pinned_cpu_copy_batch(
        std::shared_ptr<HostBuffer> buffer, uint64_t bytes,
        std::vector<uint32_t> block_ids,
        std::vector<KvMemPendingWriteBatch::CpuCopy> cpu_copies);
    void kvmem_finish_proactive_d2h(bool wait_all);
    void kvmem_reap_pending_writes(bool wait_all);
    bool kvmem_pending_reselect_ = false;
    KvMemPlan kvmem_pending_plan_;
    // Attention query position within the assembled window (== sum of selected
    // block token counts at assembly; grows by 1 per decoded token appended at
    // the window tail). Equals position_ under identity (all-block) selection.
    uint32_t window_query_pos_ = 0;
    // Assemble window_pages_* + per-layer re-RoPE from a finished selection plan.
    void kvmem_assemble(const KvMemPlan &plan);
    void kvmem_stage_in(const KvMemPlan &plan);
    void kvmem_start_prefetch(const KvMemPlan &plan);
    void kvmem_finish_prefetch();
    void kvmem_submit_prefetch_cpu_batches();
    void kvmem_submit_prefetch_nvme_batches();
    void kvmem_stage_out_cpu_batched(
        const std::vector<uint32_t> &block_ids);
    void kvmem_stage_out(const std::vector<uint32_t> &block_ids);
    bool kvmem_block_pages_resident(const KvMemBlock &block) const;
    bool kvmem_block_mtp_pages_resident(const KvMemBlock &block) const;
    uint64_t kvmem_kv_page_bytes() const;
    bool kvmem_mtp_prefix_covers_registered() const;
    uint64_t kvmem_block_spill_bytes(const KvMemBlock &block) const;
    uint8_t *kvmem_cpu_data();
    const uint8_t *kvmem_cpu_data() const;
    uint64_t kvmem_cpu_bytes() const;
    uint8_t *kvmem_cpu_slot_data(int32_t slot);
    const uint8_t *kvmem_cpu_slot_data(int32_t slot) const;
    bool kvmem_cpu_budget_has(uint64_t bytes) const;
    bool kvmem_reserve_cpu_slot(int32_t slot);
    void kvmem_release_cpu_slot(int32_t slot);
    void kvmem_evict_cpu_for_raw(uint64_t bytes);
    void kvmem_ensure_raw_k_chunks(uint32_t base, uint32_t rows,
                                   bool mtp);
    void kvmem_write_raw_k(uint32_t layer_slot, uint32_t base,
                           const uint8_t *src, uint32_t rows, bool mtp);
    void kvmem_read_raw_k(uint32_t layer_slot, uint32_t base,
                          uint8_t *dst, uint32_t rows, bool mtp) const;
    void kvmem_truncate_raw_k(uint32_t token_pos);
    void kvmem_canonicalize_block_for_tier(uint32_t block_id);
    // Grow mtp_baked_pos_ to cover every registered block, initializing new
    // entries to their true (canonical) first position. No-op unless the MTP
    // head is tiered/windowed.
    void kvmem_sync_mtp_baked_pos();
    void kvmem_copy_block_to_host(const KvMemBlock &block,
                                  std::vector<uint8_t> &dst);
    // Issue the block's K/V page D2H copies into a caller-owned buffer (must
    // hold kvmem_block_spill_bytes(block) bytes). Pinned buffers let the copies
    // queue asynchronously on the copy stream; the caller drains with
    // wait_kv_transfer() before reading.
    void kvmem_copy_block_to_host_ptr(const KvMemBlock &block, uint8_t *out);
    // Ensure the pinned stage-out staging buffer holds at least `bytes`
    // (grows if needed). Returns the buffer base pointer.
    uint8_t *kvmem_ensure_stage_pinned(uint64_t bytes);
    void kvmem_copy_block_from_host(const KvMemBlock &block,
                                    const std::vector<uint8_t> &src,
                                    bool stage_mtp_pages);
    void kvmem_copy_block_from_host(const KvMemBlock &block,
                                    const void *src,
                                    uint64_t bytes,
                                    bool stage_mtp_pages);
    void sync_window_pages_device(uint32_t have_pages);
    void sync_mtp_window_pages_device(uint32_t have_pages);
    // Grow the window page table by the trailing physical page so a decode
    // token can be appended at window slot window_query_pos_.
    void kvmem_extend_window_for_decode();
    // Batched analogue: grow the window so `n` tokens can be appended starting
    // at window slot window_query_pos_ (used by the window-aware batched verify
    // in forward_n_tokens).
    void kvmem_extend_window_for_decode_n(uint32_t n,
                                          uint32_t true_base_pos);
    // MTP-draft mirror: grow the MTP window so `n` speculative draft tokens can
    // be appended at window slot window_query_pos_ aliasing the MTP cache's
    // true-tail pages. Trimmed back to the pre-chain length after the draft.
    void kvmem_extend_mtp_window_for_decode_n(uint32_t n,
                                              uint32_t true_base_pos);
    void kvmem_register_until(uint32_t target_pos);
    std::unique_ptr<DeviceTensor> kvmem_alloc_raw_k_tensor(
        uint64_t count, const char *label);
    // Mean-K is a persistent derivative of the KV cache, so store it at the
    // same precision as K (fp16/fp8/fp32) instead of unconditionally expanding
    // it to fp32. Accumulation in the builders/scorers remains fp32.
    std::unique_ptr<DeviceTensor> kvmem_alloc_mean_index_tensor(
        uint64_t count, const char *label);
    void kvmem_ensure_raw_capture_capacity(uint32_t rows);
    void kvmem_capture_raw_k_batch(uint32_t layer, const DeviceTensor &raw_k,
                                   uint32_t batch);
    void kvmem_capture_raw_k_decode(uint32_t layer,
                                    const DeviceTensor &raw_k,
                                    uint32_t row);
    void kvmem_flush_raw_k_capture(uint32_t true_base, uint32_t first_row,
                                   uint32_t rows);
    void kvmem_flush_raw_k_decode();
    void kvmem_materialize_raw_k(
        const std::vector<const KvMemRemap *> &refreshes);
    void kvmem_ensure_rope_sincos_table();
    void kvmem_ensure_raw_mtp_capture_capacity(uint32_t rows);
    void kvmem_capture_raw_mtp_k(const DeviceTensor &raw_k,
                                 uint32_t logical_base,
                                 uint32_t rows,
                                 uint32_t src_row = 0);
    void kvmem_materialize_raw_mtp_k(const KvMemPlan &plan);

    // ---- Cumulative-attention selection signal (#40, low-intrusion) -------
    // Per-window-block representative K (mean baked K) + a GPU-resident
    // per-block score accumulator. Each decode step a single kernel scores the
    // current Q against every window block's k̄ and atomic-adds into the
    // accumulator (no D2H). At the reselect boundary the accumulator is drained
    // to host and folded into KvMemStore::attn_score so pick_topk_blocks
    // ranks by attention heat instead of recency alone. All inert (and no extra
    // kernels) unless kvmem_active_ and a representative layer exists.
    void kvmem_recompute_kbar();        // after assembly: rebuild k̄ + reset accum
    void kvmem_score_current_step(uint32_t layer_index, float scale);
    void kvmem_drain_scores();           // accum -> KvMemStore::accumulate_attn
    int32_t bs_score_layer_ = -1;               // representative standard-attn layer (-1 none)
    uint32_t bs_window_blocks_ = 0;             // blocks in the current window
    bool bs_score_ready_ = false;               // accumulator holds a live interval
    std::vector<uint32_t> bs_window_block_ids_; // window slot w -> block_id (for drain)
    std::vector<int32_t> bs_win_base_host_;     // window slot w -> first window pos
    std::vector<int32_t> bs_blk_tokens_host_;   // window slot w -> token count
    std::unique_ptr<DeviceTensor> bs_kbar_;            // [blocks, n_kv_heads, head_dim] fp32
    std::unique_ptr<DeviceTensor> bs_score_accum_;     // [blocks] fp32
    std::unique_ptr<DeviceTensor> bs_win_base_dev_;    // [blocks] int32
    std::unique_ptr<DeviceTensor> bs_blk_tokens_dev_;  // [blocks] int32
    uint32_t bs_kbar_capacity_ = 0;             // allocated block capacity
    // Batched re-RoPE inputs (moved blocks only): window slot / original bake /
    // token count per moved block, uploaded once per reselect, then reused across
    // all standard-attention layers (one batched launch per layer).
    std::vector<int32_t> bs_remap_to_host_;     // moved block i -> window slot (to_base)
    std::vector<int32_t> bs_remap_from_host_;   // moved block i -> original bake (from_base)
    std::vector<int32_t> bs_remap_ntok_host_;   // moved block i -> token count
    std::unique_ptr<DeviceTensor> bs_remap_to_dev_;    // [moved] int32
    std::unique_ptr<DeviceTensor> bs_remap_from_dev_;  // [moved] int32
    std::unique_ptr<DeviceTensor> bs_remap_ntok_dev_;  // [moved] int32
    uint32_t bs_remap_capacity_ = 0;            // allocated moved-block capacity
    // MTP-head re-RoPE lockstep (Design B). The MTP draft head's KV rotation does
    // NOT follow main's block.baked_pos: during prefill a mid-prefill offload
    // re-RoPEs main K to packed window slots while MTP K stays canonical (its
    // window is not built until decode priming catches up). So MTP carries its
    // own per-block baked position and its own moved-block remap arrays, de-rotated
    // in lockstep with the MTP window rather than the main remaps. mtp_baked_pos_
    // is the single source of truth for where MTP K currently sits (canonical on
    // registration; a packed slot after a decode assemble; back to canonical after
    // a stage-out canonicalize). Inert (stays canonical, zero moved blocks -> no
    // launch) unless the MTP window is built, so identity-budget runs are
    // byte-identical to a non-MTP-tiered build.
    std::vector<int64_t> mtp_baked_pos_;        // block_id -> MTP K current bake
    std::vector<int32_t> bs_mtp_remap_to_host_;
    std::vector<int32_t> bs_mtp_remap_from_host_;
    std::vector<int32_t> bs_mtp_remap_ntok_host_;
    std::unique_ptr<DeviceTensor> bs_mtp_remap_to_dev_;
    std::unique_ptr<DeviceTensor> bs_mtp_remap_from_dev_;
    std::unique_ptr<DeviceTensor> bs_mtp_remap_ntok_dev_;
    uint32_t bs_mtp_remap_capacity_ = 0;

    // ---- Global content-frame KV retrieval (#48/#49) ----------------------
    // The window-local signal above can only RETAIN blocks already inside the
    // active window. This maintains a position-invariant CONTENT mean-Key for
    // every historical (prefill) block and, each retrieval interval, scores ALL
    // of them by similarity to the de-RoPE'd current query — so a block dropped
    // from the window can be RESURRECTED when it matches the query. The index is
    // built ONCE from the pristine post-prefill cache (blocks baked at true
    // positions) and is immutable thereafter (content is position-invariant).
    // When the index is live, global retrieval scores OVERWRITE the window-local
    // heat via set_attn_scores; a quantized (q8/fp8) cache that can't be
    // de-RoPE'd falls back to the window-local signal.
    void kvmem_build_content_index();         // once, from pristine cache
    // Preserve the non-query-conditioned retrieval index before the first
    // pressure eviction, without coupling that bookkeeping to the selection
    // policy used for the eviction itself.
    void kvmem_prepare_content_index_before_first_selection();
    void kvmem_snapshot_content_query(uint32_t layer_index);  // per step
    bool kvmem_retrieval_score(
        std::string *failure_reason = nullptr);  // interval -> set_attn_scores
    bool g_content_ready_ = false;                    // g_kbar_ holds the index
    bool g_query_ready_ = false;                       // g_query_content_ is live
    uint32_t g_indexed_blocks_ = 0;                    // blocks covered by the index
    uint32_t g_kbar_global_capacity_ = 0;              // allocated block capacity
    std::vector<int32_t> g_orig_base_host_;            // block_id -> true first pos
    std::vector<int32_t> g_blk_tokens_host_;           // block_id -> token count
    std::unique_ptr<DeviceTensor> g_kbar_;             // [blocks, n_kv_heads, head_dim], KV dtype, by block_id
    std::unique_ptr<DeviceTensor> g_score_dev_;        // [blocks] fp32
    std::unique_ptr<DeviceTensor> g_query_content_;    // [n_heads, head_dim] fp32 (content frame)
    std::unique_ptr<DeviceTensor> g_orig_base_dev_;    // [blocks] int32
    std::unique_ptr<DeviceTensor> g_blk_tokens_dev_;   // [blocks] int32

    // ---- Query-conditioned multi-token selection (#77-#82) -----------------
    // When the serve layer marks the final user message's token span [qb,qe),
    // the executor captures the de-RoPE'd (content-frame) query rows for those
    // tokens DURING prefill at bs_score_layer_ (kvmem_capture_query_multi), then
    // at the prefill->decode boundary scores every block by the mean over the M
    // question tokens (rewards broadly-relevant blocks) instead of the single
    // last-token query. Default OFF (span empty) -> the single-token retrieval /
    // recency path is byte-identical.
    void kvmem_capture_query_multi(uint32_t slot, uint32_t chunk_off,
                                   uint32_t batch, uint32_t base_pos,
                                   uint32_t rope_base_pos,
                                   uint32_t q_token_stride);
    // Long retrieval queries are captured through two bounded GPU/pinned
    // bounce slots into pageable host memory, then streamed back in token
    // chunks for scoring. This keeps GPU use O(chunk) instead of O(L*S).
    struct QueryCaptureSlot {
        std::unique_ptr<DeviceTensor> device;
        std::unique_ptr<HostBuffer> pinned;
        std::unique_ptr<DeviceTransferFence> copy_done;
        uint64_t capacity_elems = 0;
        uint64_t dst_elem_offset = 0;
        uint64_t elem_count = 0;
        bool pending = false;
    };
    void kvmem_drain_query_capture_slot(uint32_t slot);
    void kvmem_drain_query_capture();
    bool kvmem_score_host_query_chunks(
        uint32_t n_blocks, uint32_t kbar_stride, float scale,
        uint32_t excl_lo_end, uint32_t excl_hi_begin,
        std::string *failure_reason);
    uint32_t kvmem_query_begin_ = 0;
    uint32_t kvmem_query_end_ = 0;                             // begin==end -> no span
    uint32_t g_query_multi_count_ = 0;                         // rows captured so far (per slot)
    uint64_t g_query_multi_capacity_ = 0;                      // GPU rows: full query (legacy) or score chunk
    bool g_query_multi_ready_ = false;                         // all span rows captured
    // Persistent API sessions build the mean-K content index while ingesting
    // history, before a retrieval query exists. This flag keeps K capture live
    // when query_begin==query_end; query capture/scoring remain disabled.
    bool kvmem_qc_capture_active_ = false;
    std::unique_ptr<DeviceTensor> g_query_multi_;              // [L,S,H,D], FP16 mean-K or legacy FP32
    bool kvmem_query_host_capture_ = false;
    std::unique_ptr<uint16_t[]> g_query_multi_host_;           // pageable [L,S,H,D] fp16 bits
    uint64_t g_query_multi_host_capacity_ = 0;                 // elements, not rows
    std::array<QueryCaptureSlot, 2> g_query_capture_slots_;
    uint32_t g_query_capture_next_slot_ = 0;
    std::unique_ptr<HostBuffer> g_query_score_pinned_;         // packed [L,C,H,D]
    uint64_t g_query_score_pinned_capacity_ = 0;               // bytes
    // Clean-query prefill (task #50). The PASS-A isolated-question query is stashed
    // here; it is NOT touched by reset_state so it survives the pass boundary.
    std::unique_ptr<DeviceTensor> g_query_multi_clean_;        // [L, S, n_heads, head_dim] fp32
    uint32_t g_query_multi_clean_count_ = 0;                   // rows stashed (== S when ready)
    // Force pick_topk to always keep blocks with id >= this (question + live tail);
    // 0xffffffff => no pin. pick_topk result is unioned with [pin,block_count) in
    // kvmem_selection_with_pin() before set_selection at every reselect site.
    uint32_t kvmem_qc_pin_from_block_ = 0xffffffffu;
    std::vector<uint32_t> kvmem_selection_with_pin();

    // ---- Per-normal-attention-layer multi-layer selection (#85-#90) --------
    // Qwen3.6-27B is hybrid: full_attention_interval=4 over 64 layers => 16
    // normal-attention layers. The single-layer proxy above (bs_score_layer_)
    // ranks blocks by ONE early std layer's content mean-key, a poor proxy for
    // "which old session holds the answer". This generalizes the query-
    // conditioned path to score by ALL L normal layers at once: g_kbar_multi_
    // holds a per-layer content index, g_query_multi_ holds per-layer query rows
    // ([L,S,...]), and one fused kernel scores every block by the query rows in a
    // single launch + single D2H (the step-mode reselect fires once at the prefill
    // ->decode boundary, so this is a one-time per-request cost). Multi-layer
    // capture is always on for --kvmem-query-conditioned. QW3_KVMEM_QC_LAYERS=N
    // caps L (debug).
    void kvmem_resolve_std_layers();        // populate std_layers_/std_layer_slot_
    // mean-k scorer (--kvmem-retrieval-method mean-k, default): per-(layer, query
    // token, head) softmax of q·k̄ OVER PAGES on the cheap per-block mean key.
    // mask_mode: -1 = honor env QW3_KVMEM_MASK_KEPT (default), 0 = force no mask,
    // 1 = force mask of the always-kept sink/recent bands (used by the dump sweep).
    bool kvmem_retrieval_score_mean_softmax(
        int mask_mode = -1,
        std::string *failure_reason = nullptr);  // boundary, softmax-over-pages
    bool kvmem_no_rerope_ = false;                             // env: skip re-RoPE collapse (true-pos test)
    bool kvmem_fix_bakedpos_ = true;                           // record window bake pos for window-baked chunks (env off-switch QW3_KVMEM_FIX_BAKEDPOS=0)

    // ---- Raw-key ExactMass selection (--kvmem-retrieval-method per-token) ---
    // ExactMass softmaxes the per-token logit scale·(q·k_raw) over ALL key tokens
    // then sums each block's mass (Σexp under a global denominator), so a needle
    // token's mass survives AND blocks compete globally — distinct from softmax-
    // over-pages (exp of the diluted per-block mean key). Keeps the RAW per-token
    // de-RoPE'd keys in g_kraw_multi_ (~7.4 GB fp32), allocated/captured ONLY when
    // per-token is selected. kvmem_qc_pertoken_ mirrors the retrieval_method enum.
    bool kvmem_retrieval_score_exactmass(
        std::string *failure_reason = nullptr);  // boundary, raw-key ExactMass
    bool kvmem_qc_pertoken_ = false;                           // retrieval_method == PerToken
    uint32_t kvmem_qc_total_tokens_ = 0;                       // total prompt tokens (kraw stride)
    std::unique_ptr<DeviceTensor> g_kraw_multi_;              // [L, total_tokens, n_kv_heads, head_dim] fp32
    bool g_kraw_multi_ready_ = false;                          // g_kraw_multi_ holds raw keys
    int32_t kvmem_qc_layer_cap_ = -1;                          // env: cap L (-1 = all std layers)
    uint32_t kvmem_qc_num_layers_ = 0;                         // L (resolved std-layer count)
    uint32_t kvmem_query_span_ = 0;                            // S (span length, == capacity)
    std::vector<int32_t> std_layer_slot_;                      // il -> slot 0..L-1, or -1
    std::vector<uint32_t> std_layers_;                         // slot -> il
    std::unique_ptr<DeviceTensor> g_kbar_multi_;              // [L, blocks, n_subblocks, n_kv_heads, head_dim], KV dtype
    bool g_kbar_multi_ready_ = false;                          // g_kbar_multi_ holds the index
    uint32_t g_kbar_multi_blocks_ = 0;                         // blocks covered (per layer)
    uint32_t g_kbar_multi_capacity_ = 0;                       // allocated block capacity (per layer)
    uint32_t kvmem_qc_n_subblocks_ = 1;                        // sub-block means per block (SubBlockMeanK; 1 = plain mean-k)
    bool kvmem_qc_subblock_max_ = true;                        // sub-block reduction: true=max over sub-blocks (MaxSim), false=sum (mass). No-op at n_subblocks==1.

    // ---- DeltaNet-state retrieval (--kvmem-retrieval-method deltanet) --------
    // Scores each historical block by the net EDIT it made to the DeltaNet
    // recurrent state (E_j = S_j - a_j S_{j-1}) read by the current DeltaNet query
    // (see deltanet_retrieval.md). Full per-block state edits are large (d_v*d_k
    // fp32 per head per layer), so only a memory-budget-capped subset of the 48
    // DeltaNet layers feeds the score (dn_layers_/dn_layer_slot_, evenly spaced).
    // During prefill the qw3 delta kernel snapshots S_j per block into
    // g_deltanet_snap_ and a companion scan accumulates each block's in-block
    // log-decay into g_deltanet_decaysum_; the DeltaNet queries are captured
    // (conv + L2-norm, exactly as the read-out uses them) into g_deltanet_q_. At
    // the reselect boundary kvmem_retrieval_score_deltanet folds these into a
    // per-block score. Allocated only when the deltanet method is selected.
    bool kvmem_qc_deltanet_ = false;                           // retrieval_method == DeltaNet
    void kvmem_resolve_deltanet_layers();                      // populate dn_layers_/dn_layer_slot_ (mem-budget capped)
    void kvmem_capture_deltanet_query(uint32_t dn_slot, uint32_t chunk_off,
                                      uint32_t batch, uint32_t base_pos,
                                      const DeviceTensor &conv_out, uint32_t conv_stride);
    bool kvmem_retrieval_score_deltanet(
        std::string *failure_reason = nullptr);  // boundary DeltaNet-state scorer
    std::vector<uint32_t> dn_layers_;                          // dn_slot -> layer id (recurrent)
    std::vector<int32_t> dn_layer_slot_;                       // il -> dn_slot 0..L_dn-1, or -1
    uint32_t kvmem_dn_num_layers_ = 0;                         // L_dn (selected DeltaNet-layer count)
    uint32_t kvmem_dn_qcount_ = 0;                             // captured DeltaNet query rows (per slot)
    bool kvmem_dn_ready_ = false;                              // snapshots + decay + query all captured
    std::unique_ptr<DeviceTensor> g_deltanet_snap_;            // [L_dn, blocks, num_v_heads, d_v, d_k] fp32 (S_j per block)
    std::unique_ptr<DeviceTensor> g_deltanet_decaysum_;        // [L_dn, blocks, num_v_heads] fp32 (Σ in-block log-decay)
    std::unique_ptr<DeviceTensor> g_deltanet_decay_d_;         // [L_dn, blocks, num_v_heads] fp32 (post-block decay exp(G_M-G_j) or 1)
    std::unique_ptr<DeviceTensor> g_deltanet_q_;               // [L_dn, S, num_k_heads, d_k] fp32 (L2-normed query)
    std::unique_ptr<DeviceTensor> g_deltanet_r_;               // [L_dn, blocks, num_v_heads] fp32 (per-layer block/head scores)
    uint32_t g_deltanet_capacity_blocks_ = 0;                  // allocated block capacity (per layer)

    // Incremental content-index build from the prefill K batch (#91). The paged
    // builder (kvmem_build_content_index) can only run once from the pristine
    // cache, so for histories larger than the GPU page pool it covers only the
    // blocks resident at the first mid-prefill offload (the tail is unindexed and
    // unselectable). When a query span is active we instead capture each block's
    // per-layer content mean-key the moment its K is RoPE'd during prefill
    // (kvmem_capture_kbar_multi), covering EVERY block. kvmem_qc_total_blocks_ is
    // the final block count (from the prompt length at span-set), which fixes the
    // per-layer stride in g_kbar_multi_; kvmem_qc_captured_blocks_ tracks progress.
    void kvmem_capture_kbar_multi(uint32_t slot, uint32_t batch, uint32_t base_pos,
                                  uint32_t rope_base_pos, uint32_t k_token_stride);
    uint32_t kvmem_qc_total_blocks_ = 0;                       // final block count (index stride)
    uint32_t kvmem_qc_prompt_tokens_ = 0;                      // exact final logical prompt length
    uint32_t kvmem_qc_captured_blocks_ = 0;                    // blocks captured so far (slot 0)
    uint32_t kvmem_qc_captured_tokens_ = 0;                    // exact contiguous prefix represented by the index
    // Fixed per-layer stride (in blocks) of g_kbar_multi_, pinned at ctx_blocks
    // (ceil(kv_ctx_size_/block_tokens)) for the whole session so preserved [0,D)
    // index slices stay valid as the block count grows across resumed turns
    // (server-side session continuation). 0 -> not yet allocated; when nonzero the
    // scorer passes it as kbar_layer_stride while g_kbar_multi_blocks_ stays the
    // scored count. Legacy (non-resumable) turns leave it == the turn's block count.
    uint32_t kvmem_qc_layer_stride_blocks_ = 0;
    // Set by kvmem_truncate_to when a truncate drops ZERO blocks (pure append /
    // session resume): the resident token length that the preserved [0,D) index
    // already covers. Consumed + cleared at the top of kvmem_set_query_span, where
    // it seeds kvmem_qc_captured_blocks_ = resume/block_tokens and suppresses the
    // full-buffer zero so only the new suffix's blocks are (re)captured. 0 on a cold
    // turn (reset_state / a dropping truncate), which zeroes the whole index.
    uint32_t kvmem_qc_resume_base_tokens_ = 0;
    // Diagnostics-only metadata copied from GenerationOptions once per request.
    std::string kvmem_trace_tag_;
    uint32_t kvmem_context_begin_ = 0;
    uint32_t kvmem_context_end_ = 0;
    uint32_t kvmem_trace_event_index_ = 0;
    uint32_t kvmem_trace_event_count_ = 0;
    std::vector<uint32_t> kvmem_trace_prompt_tokens_;
public:
    // Enable/disable decode-time content capture around the (plain) decode loop.
    // The server-side session-continuation path enables it before decoding an
    // above-budget query-conditioned turn and disables (+ finalizes the trailing
    // partial block) after, so the turn's generated blocks are indexed for the
    // NEXT turn's reuse. No-op unless kvmem + a QC span + the fixed-stride index
    // are live; safe to call unconditionally.
    void kvmem_decode_capture_begin();
    void kvmem_decode_capture_finalize();
    // Per-std-layer decode hook: stage the just-RoPE'd K row (k_) for this token,
    // de-RoPE'd at its bake position rope_pos into the position-invariant content
    // frame. Called once per normal-attention layer inside forward_one_token.
    void kvmem_decode_capture_stage(uint32_t layer_index, uint32_t rope_pos);

private:
    // ---- Decode-time content capture (server-side session continuation) -------
    // The incremental prefill capture (kvmem_capture_kbar_multi) only runs during
    // prefill, so a turn's generated tokens are never indexed. Above budget they
    // land in the preserved [0,D) region on the next resume turn but would stay
    // zero-ranked (unselectable), breaking reuse==fresh equivalence. So above budget
    // we index each generated block DURING decode as it completes. Decode K is
    // window-baked at a position that RESELECT re-bakes mid-block (interval < block
    // size), so contiguous-position batch de-RoPE is unsafe; instead each token's K
    // is de-RoPE'd at its OWN bake position the moment it is produced (reselect-
    // immune, since the content frame is position-invariant) and the resulting
    // content-frame rows are staged here [L, block_tokens, n_kv_heads*head_dim] fp32.
    // On block completion the staged rows are meaned (rope_dim==0, no further rotate)
    // into g_kbar_multi_[slot*stride + true_block_index]. Allocated lazily on the
    // first above-budget decode; unused (and un-allocated) below budget / off / MTP.
    void kvmem_capture_decode_block(uint32_t true_block_index, uint32_t rows);
    std::unique_ptr<DeviceTensor> g_kbar_decode_stage_;        // [L, block_tokens, n_kv_heads*head_dim] fp32 content-frame rows
    bool kvmem_decode_capture_on_ = false;                     // enabled by kvmem_decode_capture_begin (plain above-budget QC turn)
    bool decode_stage_active_ = false;                         // currently staging a block started at its first (offset-0) token
    uint32_t decode_stage_rows_ = 0;                           // content-frame rows staged in the current block
    uint32_t decode_stage_block_ = 0;                          // TRUE block index currently being staged

    // ---- KVMem attention-distribution diagnostics -------------------------
    // Enabled only when QW3_KVMEM_ATTN_TRACE points at a JSONL output path.
    // Recomputes true decode softmax attention at sampled token positions and
    // aggregates mass by selected KVMem block for every standard-attention layer.
    bool kvmem_attn_trace_enabled() const;
    bool kvmem_attn_trace_sample_now() const;
    void kvmem_trace_attention_layer(uint32_t layer_index,
                                     const DeviceTensor &k_cache,
                                     const DeviceTensor &q,
                                     uint32_t q_stride,
                                     const DeviceTensor &page_indices,
                                     uint32_t n_pages,
                                     uint32_t seq_len,
                                     float scale);
    std::FILE *kvmem_attn_trace_file_ = nullptr;
    uint64_t kvmem_attn_trace_seen_tokens_ = 0;
    uint64_t kvmem_attn_trace_sample_ = 0;
    uint32_t kvmem_attn_trace_mass_capacity_ = 0;
    std::unique_ptr<DeviceTensor> kvmem_attn_trace_mass_;

    // ---- Global attention-distribution diagnostics ------------------------
    // Enabled when QW3_ATTN_TRACE points at a JSONL output path. Unlike the
    // kvmem trace above, this does not require kvmem and groups the currently
    // visible logical KV sequence into fixed-size blocks.
    bool global_attn_trace_enabled() const;
    bool global_attn_trace_sample_now() const;
    void global_trace_attention_layer(uint32_t layer_index,
                                      const DeviceTensor &k_cache,
                                      const DeviceTensor &q,
                                      uint32_t q_stride,
                                      const DeviceTensor &page_indices,
                                      uint32_t n_pages,
                                      uint32_t seq_len,
                                      float scale);
    std::FILE *global_attn_trace_file_ = nullptr;
    uint64_t global_attn_trace_seen_tokens_ = 0;
    uint64_t global_attn_trace_sample_ = 0;
    uint32_t global_attn_trace_block_tokens_ = 0;
    uint32_t global_attn_trace_block_capacity_ = 0;
    std::vector<int32_t> global_attn_trace_base_host_;
    std::vector<int32_t> global_attn_trace_tokens_host_;
    std::unique_ptr<DeviceTensor> global_attn_trace_base_dev_;
    std::unique_ptr<DeviceTensor> global_attn_trace_tokens_dev_;
    std::unique_ptr<DeviceTensor> global_attn_trace_mass_;

    // Set by reset_state() and cleared after the first eager forward_one_token
    // call of a generate() session. Suppresses CUDA-graph capture on token 0
    // so every backend-side scratch buffer (q8_1, fattn, argmax_dev, ...) is
    // allocated and primed before we record pointers into a graph.
    bool decode_graph_warmup_pending_ = true;

    mutable double trace_last_seconds_ = 0.0;
};

} // namespace qw3
