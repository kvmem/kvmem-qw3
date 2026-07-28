#pragma once

#include "qw3/model_source.hpp"
#include "qw3/qw3.hpp"
#include "qw3/qwen_config.hpp"

#include <memory>
#include <string>
#include <vector>

namespace qw3 {

struct QwenLayerTensors {
    bool recurrent = false;
    const ModelTensorInfo *attn_norm = nullptr;
    const ModelTensorInfo *attn_q = nullptr;
    const ModelTensorInfo *attn_k = nullptr;
    const ModelTensorInfo *attn_v = nullptr;
    const ModelTensorInfo *attn_q_norm = nullptr;
    const ModelTensorInfo *attn_k_norm = nullptr;
    const ModelTensorInfo *attn_qkv = nullptr;
    const ModelTensorInfo *attn_gate = nullptr;
    const ModelTensorInfo *attn_output = nullptr;
    const ModelTensorInfo *ffn_norm = nullptr;
    const ModelTensorInfo *ffn_gate = nullptr;
    const ModelTensorInfo *ffn_up = nullptr;
    const ModelTensorInfo *ffn_down = nullptr;
    const ModelTensorInfo *ssm_a = nullptr;
    const ModelTensorInfo *ssm_alpha = nullptr;
    const ModelTensorInfo *ssm_beta = nullptr;
    const ModelTensorInfo *ssm_conv1d = nullptr;
    const ModelTensorInfo *ssm_dt_bias = nullptr;
    const ModelTensorInfo *ssm_norm = nullptr;
    const ModelTensorInfo *ssm_out = nullptr;
};

struct QwenMtpTensors {
    bool present = false;
    uint32_t layer_index = 0;
    QwenLayerTensors layer;
    const ModelTensorInfo *eh_proj = nullptr;
    const ModelTensorInfo *embed_tokens = nullptr;      // optional; falls back to token_embd.weight
    const ModelTensorInfo *enorm = nullptr;
    const ModelTensorInfo *hnorm = nullptr;
    const ModelTensorInfo *shared_head_head = nullptr;  // optional; falls back to output.weight
    const ModelTensorInfo *shared_head_norm = nullptr;  // optional; falls back to output_norm.weight
};

class QwenNativeModel {
public:
    explicit QwenNativeModel(std::unique_ptr<GgufFile> gguf);
    explicit QwenNativeModel(std::unique_ptr<ModelSource> source);

    const GgufFile &gguf() const;
    const ModelSource &source() const;
    const NativePlanInfo &plan() const;
    const QwenConfig &config() const;
    const std::vector<QwenLayerTensors> &layers() const;
    const QwenMtpTensors *mtp() const;
    const ModelTensorInfo *token_embedding() const;
    const ModelTensorInfo *output_norm() const;
    const ModelTensorInfo *output() const;
    std::string token_text(uint32_t token) const;
    std::string describe_plan() const;

private:
    void bind();
    void bind_mtp();
    const ModelTensorInfo *require_tensor(const std::string &name);
    const ModelTensorInfo *require_any_tensor(const std::vector<std::string> &names);
    const ModelTensorInfo *optional_tensor(const std::string &name);
    void add_missing(const std::string &name);
    void add_mtp_missing(const std::string &name);
    void count_bound(const ModelTensorInfo *tensor);
    void count_mtp_bound(const ModelTensorInfo *tensor);

    std::unique_ptr<ModelSource> source_;
    std::unique_ptr<QwenConfig> config_;
    NativePlanInfo plan_;
    std::vector<QwenLayerTensors> layers_;
    QwenMtpTensors mtp_;
    const ModelTensorInfo *token_embd_ = nullptr;
    const ModelTensorInfo *output_norm_ = nullptr;
    const ModelTensorInfo *output_ = nullptr;
};

} // namespace qw3
