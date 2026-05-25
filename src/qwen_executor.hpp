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
    std::vector<std::string> missing_kernels;
};

/* Per-session executor.
 *
 * Owns transient device-resident scratch buffers (h, norm, attn_out, ffn_*,
 * recurrent_state, etc.) and the logits buffer; reuses them across every
 * forward step. All weight tensors live in QwenWeights and are NOT touched by
 * this class except by pointer. */
class QwenExecutor {
public:
    QwenExecutor(const QwenNativeModel &model,
                 const QwenWeights &weights,
                 DeviceBackend &backend,
                 uint32_t kv_ctx_size);
    ~QwenExecutor();

    void reset_state();
    uint32_t position() const { return position_; }
    uint32_t kv_ctx_size() const { return kv_ctx_size_; }

    NativeExecutorReport dry_run_token(uint32_t token_id, bool execute_heavy);
    NativeExecutorReport forward_one_token(uint32_t token_id);
    // Batched prefill. Processes `tokens` consecutively as a single forward
    // pass using batched matmuls for the linear projections + FFN. Per-token
    // ops (attention, recurrent state) still iterate sequentially inside.
    // After return, position_ has advanced by tokens.size() and the LM-head
    // logits + argmax correspond to the LAST token in the batch.
    NativeExecutorReport forward_n_tokens(const std::vector<uint32_t> &tokens);

    // Copy the most recent logits tensor back to host. Returns false if
    // forward_one_token has not been called yet.
    bool copy_last_logits(std::vector<float> &out) const;

private:
    void record(NativeExecutorReport &report, const std::string &op) const;
    void ensure_scratch();

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

    uint32_t kv_ctx_size_ = 0;
    uint32_t position_ = 0;
};

} // namespace qw3
