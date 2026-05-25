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

    // Copy the most recent logits tensor back to host. Returns false if
    // forward_one_token has not been called yet.
    bool copy_last_logits(std::vector<float> &out) const;

private:
    void record(NativeExecutorReport &report, const std::string &op) const;
    void ensure_scratch();

    const QwenNativeModel &model_;
    const QwenWeights &weights_;
    DeviceBackend &backend_;

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
