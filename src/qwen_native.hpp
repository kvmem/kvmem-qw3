#pragma once

#include "qw3/gguf.hpp"
#include "qw3/qw3.hpp"
#include "qw3/qwen_config.hpp"

#include <memory>
#include <string>
#include <vector>

namespace qw3 {

struct QwenLayerTensors {
    bool recurrent = false;
    const GgufTensorInfo *attn_norm = nullptr;
    const GgufTensorInfo *attn_q = nullptr;
    const GgufTensorInfo *attn_k = nullptr;
    const GgufTensorInfo *attn_v = nullptr;
    const GgufTensorInfo *attn_q_norm = nullptr;
    const GgufTensorInfo *attn_k_norm = nullptr;
    const GgufTensorInfo *attn_qkv = nullptr;
    const GgufTensorInfo *attn_gate = nullptr;
    const GgufTensorInfo *attn_output = nullptr;
    const GgufTensorInfo *ffn_norm = nullptr;
    const GgufTensorInfo *ffn_gate = nullptr;
    const GgufTensorInfo *ffn_up = nullptr;
    const GgufTensorInfo *ffn_down = nullptr;
    const GgufTensorInfo *ssm_a = nullptr;
    const GgufTensorInfo *ssm_alpha = nullptr;
    const GgufTensorInfo *ssm_beta = nullptr;
    const GgufTensorInfo *ssm_conv1d = nullptr;
    const GgufTensorInfo *ssm_dt_bias = nullptr;
    const GgufTensorInfo *ssm_norm = nullptr;
    const GgufTensorInfo *ssm_out = nullptr;
};

class QwenNativeModel {
public:
    explicit QwenNativeModel(std::unique_ptr<GgufFile> gguf);

    const GgufFile &gguf() const;
    const NativePlanInfo &plan() const;
    const QwenConfig &config() const;
    const std::vector<QwenLayerTensors> &layers() const;
    const GgufTensorInfo *token_embedding() const;
    const GgufTensorInfo *output_norm() const;
    const GgufTensorInfo *output() const;
    std::string describe_plan() const;

private:
    void bind();
    const GgufTensorInfo *require_tensor(const std::string &name);
    const GgufTensorInfo *require_any_tensor(const std::vector<std::string> &names);
    const GgufTensorInfo *optional_tensor(const std::string &name);
    void add_missing(const std::string &name);
    void count_bound(const GgufTensorInfo *tensor);

    std::unique_ptr<GgufFile> gguf_;
    std::unique_ptr<QwenConfig> config_;
    NativePlanInfo plan_;
    std::vector<QwenLayerTensors> layers_;
    const GgufTensorInfo *token_embd_ = nullptr;
    const GgufTensorInfo *output_norm_ = nullptr;
    const GgufTensorInfo *output_ = nullptr;
};

} // namespace qw3
