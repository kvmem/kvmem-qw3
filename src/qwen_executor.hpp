#pragma once

#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"
#include "qw3/kvmem_store.hpp"
#include "qw3/nvme_kv_tier.hpp"
#include "qw3/pinned_kv_tier.hpp"

#include <cstdio>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

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
    NativeExecutorReport prime_mtp_prefix_from_current(uint32_t token,
                                                       uint32_t base_position);
    NativeExecutorReport replay_tokens_with_mtp_prefix(const std::vector<uint32_t> &tokens,
                                                       uint32_t base_position,
                                                       bool rebuild_prefix,
                                                       double *prefix_seconds = nullptr,
                                                       uint64_t *prefix_ops = nullptr);
    void commit_mtp_prefix(uint32_t prefix_len);
    void commit_mtp_prefix_from_current_hidden(uint32_t prefix_len);
    StateSnapshot snapshot_state();
    void capture_state(StateSnapshot &snapshot);
    void restore_state(const StateSnapshot &snapshot);
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
    // Mark the final user message's token span [begin,end) for query-conditioned
    // multi-token selection. Called by the backend BEFORE prefill. begin==end -> no
    // span -> byte-identical single-token/recency path. prompt_tokens is the full
    // prompt length: it fixes the final block count up front so the incremental
    // per-layer content index (#91) can size its per-layer stride before the first
    // K chunk is captured. (Public: backend-invoked.)
    void kvmem_set_query_span(uint32_t begin, uint32_t end,
                              uint32_t prompt_tokens);          // before prefill
    // Context-free query embedding (AgentKV run_segment isolation, opt-in via
    // QW3_KVMEM_QC_QUERY_CONTEXTFREE). Call right AFTER kvmem_set_query_span and
    // BEFORE the main prefill (KV empty, position 0): runs an isolated forward over
    // the question tokens alone (no history) to capture a context-free retrieval Q,
    // then rolls the executor back so the full prefill rebuilds the real KV cache.
    // No-op (returns false) when the flag is off — byte-identical. (Public:
    // backend-invoked.)
    bool kvmem_capture_query_contextfree(const std::vector<uint32_t> &question_tokens);
    bool kvmem_query_contextfree_enabled() const { return kvmem_qc_query_contextfree_; }
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
    // Spill cold blocks to the tier mid-prefill if the bounded GPU page pool is
    // about to run short. `next_chunk_tokens` is the size of the upcoming prefill
    // append, so the offload fires while there is still room for it (a full chunk
    // can grab >100 pages at once, so a "only when nearly empty" trigger would
    // let the pool exhaust mid-chunk and throw). No-op when not bounded/tiered.
    void kvmem_maybe_prefill_offload(uint32_t next_chunk_tokens);

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
        void release_logical_pages(DeviceBackend &backend,
                                   uint32_t logical_start,
                                   uint32_t count);
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
    void ensure_mtp_scratch();
    void ensure_mtp_batch_scratch(uint32_t batch);
    void ensure_logits_batch_scratch(uint32_t batch);
    void ensure_kv_pages(uint32_t logical_pos, uint32_t count);
    const int32_t *kv_page_indices() const { return kv_pages_.host_indices(); }
    const DeviceTensor &kv_page_indices_device() const { return kv_pages_.device_indices(); }
    uint32_t kv_page_count() const { return kv_pages_.count(); }
    uint32_t kv_page_size() const { return kv_pages_.page_size; }
    DeviceTensor &k_cache(uint32_t layer);
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
    int      prefill_chunk_override_ = -1;
    KvPageTable kv_pages_;

    // ---- Block-sparse KV attention state (inert unless enabled) -----------
    // All zero/null when kvmem_enabled_ is false, so the forward path
    // takes the identical pre-block-sparse branches.
    bool kvmem_enabled_ = false;
    uint32_t kvmem_registered_pos_ = 0;
    // True once a selection has been assembled this session; gates the decode
    // window substitution. Cleared by reset_state().
    bool kvmem_active_ = false;
    std::unique_ptr<KvMemStore> block_store_;
    // Window page table: the selected blocks' ORIGINAL physical pages, in
    // ascending (window) order. No-copy — these alias kv_pages_ slots; the
    // window is a reordering of pointers, re-RoPE rebakes K in place. Borrowed,
    // never released by this table.
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
    struct KvMemPrefetchBlock {
        uint32_t block_id = 0;
        KvTier from = KvTier::GPU;
    };
    struct KvMemPrefetchNvmeRead {
        uint32_t block_id = 0;
        uint64_t bytes = 0;
        std::vector<uint8_t> buffer;
    };
    struct KvMemPrefetchState {
        bool active = false;
        bool queued_h2d = false;
        std::vector<KvMemPrefetchBlock> blocks;
        std::vector<KvMemPrefetchNvmeRead> nvme_reads;
        std::future<void> nvme_future;
    };
    KvMemPrefetchState kvmem_prefetch_;
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
    void kvmem_stage_out(const std::vector<uint32_t> &block_ids);
    bool kvmem_block_pages_resident(const KvMemBlock &block) const;
    uint64_t kvmem_kv_page_bytes() const;
    uint64_t kvmem_block_spill_bytes(const KvMemBlock &block) const;
    uint8_t *kvmem_cpu_data();
    const uint8_t *kvmem_cpu_data() const;
    uint64_t kvmem_cpu_bytes() const;
    void kvmem_canonicalize_block_for_tier(uint32_t block_id);
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
                                    const std::vector<uint8_t> &src);
    void kvmem_copy_block_from_host(const KvMemBlock &block,
                                    const void *src,
                                    uint64_t bytes);
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
    void kvmem_snapshot_content_query(uint32_t layer_index);  // per step
    bool kvmem_retrieval_score();              // interval -> set_attn_scores
    bool g_content_ready_ = false;                    // g_kbar_ holds the index
    bool g_query_ready_ = false;                       // g_query_content_ is live
    uint32_t g_indexed_blocks_ = 0;                    // blocks covered by the index
    uint32_t g_kbar_global_capacity_ = 0;              // allocated block capacity
    std::vector<int32_t> g_orig_base_host_;            // block_id -> true first pos
    std::vector<int32_t> g_blk_tokens_host_;           // block_id -> token count
    std::unique_ptr<DeviceTensor> g_kbar_;             // [blocks, n_kv_heads, head_dim] fp32, by block_id
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
    bool kvmem_retrieval_score_multitoken();                   // boundary, multi-token
    uint32_t kvmem_query_begin_ = 0;
    uint32_t kvmem_query_end_ = 0;                             // begin==end -> no span
    uint32_t kvmem_query_base_pos_ = 0;                        // position() at span-set; anchors
                                                               // prompt-token indices vs the
                                                               // possibly-chunked prefill calls
    uint32_t g_query_multi_count_ = 0;                         // rows captured so far (per slot)
    uint32_t g_query_multi_capacity_ = 0;                      // allocated rows (cap, == S)
    bool g_query_multi_ready_ = false;                         // all span rows captured
    std::unique_ptr<DeviceTensor> g_query_multi_;              // [L, S, n_heads, head_dim] fp32

    // ---- Per-normal-attention-layer multi-layer selection (#85-#90) --------
    // Qwen3.6-27B is hybrid: full_attention_interval=4 over 64 layers => 16
    // normal-attention layers. The single-layer proxy above (bs_score_layer_)
    // ranks blocks by ONE early std layer's content mean-key, a poor proxy for
    // "which old session holds the answer". This generalizes the query-
    // conditioned path to score by ALL L normal layers at once: g_kbar_multi_
    // holds a per-layer content index, g_query_multi_ holds per-layer query rows
    // ([L,S,...]), and one fused kernel sums ReLU(q·k̄) over (layers,tokens) per
    // block in a single launch + single D2H (the step-mode reselect fires once
    // at the prefill->decode boundary, so this is a one-time per-request cost).
    // This is the DEFAULT for --kvmem-query-conditioned; QW3_KVMEM_QC_SINGLE_LAYER=1
    // forces the old single-layer multitoken path (L=1, layer=bs_score_layer_)
    // for A/B comparison. QW3_KVMEM_QC_LAYERS=N caps L (debug).
    void kvmem_resolve_std_layers();        // populate std_layers_/std_layer_slot_
    bool kvmem_retrieval_score_multilayer();                   // boundary, fused multi-layer
    bool kvmem_qc_single_layer_ = false;                       // env: force L=1 old path
    bool kvmem_qc_softmax_ = false;                            // env: softmax-over-pages scorer
    bool kvmem_no_rerope_ = false;                             // env: skip re-RoPE collapse (true-pos test)

    // ---- Raw-key ColBERT MaxSim selection (#104) ---------------------------
    // The mean-k scorers above collapse a block's ~1024 tokens to one mean key,
    // so q·k̄ ≡ mean_t(q·k) DILUTES a single answer-bearing token. QW3_KVMEM_QC_MAXSIM
    // (default OFF, byte-identical when unset) instead keeps the RAW per-token
    // de-RoPE'd keys (captured into g_kraw_multi_ during prefill, like the mean) and
    // scores each block by a per-(query token, head) MAX over its tokens — a needle
    // token spikes the block score instead of being averaged away. Only allocated /
    // run when the flag is set; the ~7.4 GB fp32 buffer is the cost (efficiency
    // deferred, utility test first). Respects QW3_KVMEM_QC_LAYERS for cheap L=1 runs.
    bool kvmem_retrieval_score_maxsim();                       // boundary, raw-key MaxSim
    bool kvmem_qc_maxsim_ = false;                             // env: raw-key MaxSim scorer
    // ---- Raw-key ExactMass selection (AgentKV port) ------------------------
    // ExactMass softmaxes the per-token logit scale·(q·k_raw) over ALL key tokens
    // then sums each block's mass (Σexp under a global denominator), so a needle
    // token's mass survives AND blocks compete globally — distinct from softmax-
    // over-pages (exp of the diluted per-block mean key). Reuses g_kraw_multi_ +
    // g_query_multi_ (no new buffer/capture). QW3_KVMEM_QC_EXACTMASS, default OFF.
    bool kvmem_retrieval_score_exactmass();                    // boundary, raw-key ExactMass
    bool kvmem_qc_exactmass_ = false;                          // env: raw-key ExactMass scorer
    bool kvmem_qc_exactmass_strict_ = false;                   // env: strict-AgentKV group-mean query
    // ---- Context-free query embedding (AgentKV run_segment isolation) -------
    // AgentKV computes the retrieval query by prefilling the QUESTION ALONE (no
    // history, causal self-attention only) so the query embedding is not pulled
    // toward the history by full-prompt cross-attention. qw3's default capture
    // grabs the CONTEXTUALIZED question Q during the full prefill. When this flag
    // is set, after set_query_span (KV empty, pos 0) we run an isolated forward
    // over just the question token ids, capture the context-free de-RoPE'd Q into
    // g_query_multi_, then roll the executor back to a clean pre-prefill state so
    // the main prefill rebuilds the real KV cache. Retrieval-only: g_query_multi_
    // feeds ONLY the block scorer; generation still uses the question's KV in the
    // re-RoPE'd window (qw3 never re-prefills the question post-selection).
    // QW3_KVMEM_QC_QUERY_CONTEXTFREE, default OFF, byte-identical when unset. The
    // capture method is public (backend-invoked, declared near kvmem_set_query_span).
    bool kvmem_qc_query_contextfree_ = false;                  // env: context-free query embed
    bool kvmem_capture_query_only_ = false;                    // transient: suppress kbar/kraw in isolated pass
    uint32_t kvmem_qc_total_tokens_ = 0;                       // total prompt tokens (kraw stride)
    std::unique_ptr<DeviceTensor> g_kraw_multi_;              // [L, total_tokens, n_kv_heads, head_dim] fp32
    bool g_kraw_multi_ready_ = false;                          // g_kraw_multi_ holds raw keys
    int32_t kvmem_qc_layer_cap_ = -1;                          // env: cap L (-1 = all std layers)
    uint32_t kvmem_qc_num_layers_ = 0;                         // L (resolved std-layer count)
    uint32_t kvmem_query_span_ = 0;                            // S (span length, == capacity)
    std::vector<int32_t> std_layer_slot_;                      // il -> slot 0..L-1, or -1
    std::vector<uint32_t> std_layers_;                         // slot -> il
    std::unique_ptr<DeviceTensor> g_kbar_multi_;              // [L, blocks, n_kv_heads, head_dim] fp32
    bool g_kbar_multi_ready_ = false;                          // g_kbar_multi_ holds the index
    uint32_t g_kbar_multi_blocks_ = 0;                         // blocks covered (per layer)
    uint32_t g_kbar_multi_capacity_ = 0;                       // allocated block capacity (per layer)

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
    uint32_t kvmem_qc_captured_blocks_ = 0;                    // blocks captured so far (slot 0)

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
