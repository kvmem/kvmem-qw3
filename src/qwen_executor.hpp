#pragma once

#include "qwen_native.hpp"
#include "qwen_weights.hpp"
#include "qw3/device_backend.hpp"

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
        std::unique_ptr<DeviceTensor> h;
        std::vector<std::unique_ptr<DeviceTensor>> recurrent_states;
        std::vector<std::unique_ptr<DeviceTensor>> conv_states;
    };
    struct StateCheckpointSet {
        bool ready = false;
        uint32_t base_position = 0;
        uint32_t count = 0;
        uint32_t h_stride = 0;
        std::vector<std::unique_ptr<DeviceTensor>> recurrent_states;
        std::vector<std::unique_ptr<DeviceTensor>> conv_states;
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

    QwenExecutor(const QwenNativeModel &model,
                 const QwenWeights &weights,
                 DeviceBackend &backend,
                 uint32_t kv_ctx_size,
                 KvPhysicalPageAllocator *kv_page_allocator = nullptr,
                 KvCacheStorage *external_kv_cache = nullptr);
    ~QwenExecutor();

    void reset_state();
    uint32_t position() const { return position_; }
    uint32_t kv_ctx_size() const { return kv_ctx_size_; }
    DecodeStateView decode_state_view() const;
    MutableDecodeStateView mutable_decode_state_view();
    KvStateSnapshot kv_state_snapshot() const;
    void prepare_runtime_state();
    void prepare_kv_pages(uint32_t logical_pos, uint32_t count);
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
    NativeExecutorReport forward_n_tokens(const std::vector<uint32_t> &tokens,
                                          bool compute_logits = true,
                                          std::vector<DeviceArgmax> *row_argmaxes = nullptr,
                                          StateCheckpointSet *state_checkpoints = nullptr,
                                          uint32_t state_checkpoint_count = 0,
                                          bool copy_last_logits = true);

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

private:
    struct KvPageTable {
        uint32_t page_size = 16;
        uint32_t max_pages = 0;
        std::string alloc_mode = "identity";
        std::vector<int32_t> pages;
        std::unique_ptr<DeviceTensor> device_pages;
        KvPhysicalPageAllocator *allocator = nullptr;
        uint32_t device_synced = 0;

        void configure(uint32_t ctx_size, KvPhysicalPageAllocator *page_allocator);
        void reset();
        void ensure_pages(DeviceBackend &backend, uint32_t ctx_size,
                          uint32_t logical_pos, uint32_t count);
        int32_t allocate_physical_page(uint32_t logical_page) const;
        const int32_t *host_indices() const { return pages.data(); }
        const DeviceTensor &device_indices() const { return *device_pages; }
        uint32_t count() const { return static_cast<uint32_t>(pages.size()); }
        uint64_t physical_slots() const;
    };

    void begin_record_timing(bool enabled) const;
    void record(NativeExecutorReport &report, const std::string &op) const;
    void ensure_scratch();
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
                                                uint32_t token_source_index = 0);

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

    // Set by reset_state() and cleared after the first eager forward_one_token
    // call of a generate() session. Suppresses CUDA-graph capture on token 0
    // so every backend-side scratch buffer (q8_1, fattn, argmax_dev, ...) is
    // allocated and primed before we record pointers into a graph.
    bool decode_graph_warmup_pending_ = true;

    mutable double trace_last_seconds_ = 0.0;
};

} // namespace qw3
