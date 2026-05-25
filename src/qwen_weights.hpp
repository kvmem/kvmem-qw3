#pragma once

#include "qwen_native.hpp"
#include "qw3/device_backend.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace qw3 {

/* Device-resident Qwen weights uploaded once at engine startup.
 *
 * The model is described by QwenNativeModel (CPU-side: mmap pointers, dims,
 * dtypes). QwenWeights walks that description, uploads every required tensor
 * onto the DeviceBackend exactly once, and exposes stable DeviceWeight*
 * handles. Repeated forward passes then reuse the same device buffers without
 * triggering any further H2D copies.
 *
 * Earlier code allocated and uploaded weights inside forward_one_token(),
 * which made each decode step pay a multi-GB cudaMemcpy. */
struct QwenLayerWeights {
    bool recurrent = false;
    DeviceWeight *attn_norm = nullptr;
    DeviceWeight *ffn_norm = nullptr;
    DeviceWeight *ffn_gate = nullptr;
    DeviceWeight *ffn_up = nullptr;
    DeviceWeight *ffn_down = nullptr;

    // standard attention layers
    DeviceWeight *attn_q = nullptr;
    DeviceWeight *attn_k = nullptr;
    DeviceWeight *attn_v = nullptr;
    DeviceWeight *attn_q_norm = nullptr;
    DeviceWeight *attn_k_norm = nullptr;
    DeviceWeight *attn_output = nullptr;

    // recurrent / DeltaNet-style layers
    DeviceWeight *attn_qkv = nullptr;
    DeviceWeight *attn_gate = nullptr;
    DeviceWeight *ssm_a = nullptr;
    DeviceWeight *ssm_alpha = nullptr;
    DeviceWeight *ssm_beta = nullptr;
    DeviceWeight *ssm_conv1d = nullptr;
    DeviceWeight *ssm_dt_bias = nullptr;
    DeviceWeight *ssm_norm = nullptr;
    DeviceWeight *ssm_out = nullptr;

    // cached layer-output dimensions
    uint64_t q_rows = 0;
    uint64_t k_rows = 0;
    uint64_t v_rows = 0;
    uint64_t ffn_dim = 0;
    uint64_t recurrent_qkv_dim = 0;
    uint64_t recurrent_value_dim = 0;
    uint64_t ssm_out_rows = 0;
};

class QwenWeights {
public:
    QwenWeights(const QwenNativeModel &model, DeviceBackend &backend);
    ~QwenWeights();

    QwenWeights(const QwenWeights &) = delete;
    QwenWeights &operator=(const QwenWeights &) = delete;

    DeviceWeight &token_embd() const { return *token_embd_; }
    DeviceWeight &output_norm() const { return *output_norm_; }
    DeviceWeight &output() const { return *output_; }

    const QwenLayerWeights &layer(uint32_t i) const { return layers_[i]; }
    uint32_t n_layers() const { return static_cast<uint32_t>(layers_.size()); }

    uint64_t total_bytes_uploaded() const { return uploaded_bytes_; }
    uint64_t tensor_count() const { return owned_.size(); }

private:
    DeviceWeight *bind(const GgufTensorInfo *tensor);

    const QwenNativeModel &model_;
    DeviceBackend &backend_;
    std::vector<std::unique_ptr<DeviceWeight>> owned_;
    std::unordered_map<const GgufTensorInfo *, DeviceWeight *> by_tensor_;
    DeviceWeight *token_embd_ = nullptr;
    DeviceWeight *output_norm_ = nullptr;
    DeviceWeight *output_ = nullptr;
    std::vector<QwenLayerWeights> layers_;
    uint64_t uploaded_bytes_ = 0;
};

} // namespace qw3
