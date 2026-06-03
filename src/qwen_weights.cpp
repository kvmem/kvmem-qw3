#include "qwen_weights.hpp"

#include <stdexcept>

namespace qw3 {
namespace {

uint64_t tensor_rows(const GgufTensorInfo &tensor) {
    if (tensor.dims.size() < 2) return 1;
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    return rows;
}

uint64_t tensor_cols(const GgufTensorInfo &tensor) {
    if (tensor.dims.empty()) return 1;
    return tensor.dims[0];
}

const void *tensor_payload(const QwenNativeModel &model, const GgufTensorInfo &tensor) {
    return model.gguf().data() + tensor.abs_offset;
}

const char *tensor_label(const GgufTensorInfo &tensor) {
    return tensor.name.empty() ? "tensor" : tensor.name.c_str();
}

} // namespace

QwenWeights::QwenWeights(const QwenNativeModel &model, DeviceBackend &backend)
    : model_(model), backend_(backend) {
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        throw std::runtime_error("native model plan is incomplete; cannot upload weights");
    }

    token_embd_ = bind(model_.token_embedding());
    output_norm_ = bind(model_.output_norm());
    output_ = bind(model_.output());
    if (!token_embd_ || !output_norm_ || !output_) {
        throw std::runtime_error("required root tensors are missing");
    }

    layers_.reserve(model_.layers().size());
    for (const QwenLayerTensors &src : model_.layers()) {
        layers_.push_back(bind_layer(src));
    }

    if (const QwenMtpTensors *src = model_.mtp()) {
        mtp_.present = true;
        mtp_.layer = bind_layer(src->layer);
        mtp_.eh_proj = bind(src->eh_proj);
        mtp_.embed_tokens = bind(src->embed_tokens);
        mtp_.enorm = bind(src->enorm);
        mtp_.hnorm = bind(src->hnorm);
        mtp_.shared_head_head = bind(src->shared_head_head);
        mtp_.shared_head_norm = bind(src->shared_head_norm);
    }
}

QwenWeights::~QwenWeights() = default;

QwenLayerWeights QwenWeights::bind_layer(const QwenLayerTensors &src) {
    QwenLayerWeights dst;
    dst.recurrent = src.recurrent;
    dst.attn_norm = bind(src.attn_norm);
    dst.ffn_norm = bind(src.ffn_norm);
    dst.ffn_gate = bind(src.ffn_gate);
    dst.ffn_up = bind(src.ffn_up);
    dst.ffn_down = bind(src.ffn_down);
    dst.ffn_dim = src.ffn_gate ? tensor_rows(*src.ffn_gate) : 0;

    if (src.recurrent) {
        dst.attn_qkv = bind(src.attn_qkv);
        dst.attn_gate = bind(src.attn_gate);
        dst.ssm_a = bind(src.ssm_a);
        dst.ssm_alpha = bind(src.ssm_alpha);
        dst.ssm_beta = bind(src.ssm_beta);
        dst.ssm_conv1d = bind(src.ssm_conv1d);
        dst.ssm_dt_bias = bind(src.ssm_dt_bias);
        dst.ssm_norm = bind(src.ssm_norm);
        dst.ssm_out = bind(src.ssm_out);
        dst.recurrent_qkv_dim = src.attn_qkv ? tensor_rows(*src.attn_qkv) : 0;
        dst.recurrent_value_dim = src.attn_gate ? tensor_rows(*src.attn_gate) : 0;
        dst.ssm_out_rows = src.ssm_out ? tensor_rows(*src.ssm_out) : 0;
    } else {
        dst.attn_q = bind(src.attn_q);
        dst.attn_k = bind(src.attn_k);
        dst.attn_v = bind(src.attn_v);
        dst.attn_q_norm = bind(src.attn_q_norm);
        dst.attn_k_norm = bind(src.attn_k_norm);
        dst.attn_output = bind(src.attn_output);
        dst.q_rows = src.attn_q ? tensor_rows(*src.attn_q) : 0;
        dst.k_rows = src.attn_k ? tensor_rows(*src.attn_k) : 0;
        dst.v_rows = src.attn_v ? tensor_rows(*src.attn_v) : 0;
    }
    return dst;
}

DeviceWeight *QwenWeights::bind(const GgufTensorInfo *tensor) {
    if (!tensor) return nullptr;
    auto it = by_tensor_.find(tensor);
    if (it != by_tensor_.end()) return it->second;

    std::unique_ptr<DeviceWeight> weight;
    if (tensor->type == 8) {
        weight = backend_.weight_q8_0(tensor_payload(model_, *tensor),
                                      tensor_rows(*tensor),
                                      tensor_cols(*tensor),
                                      tensor_label(*tensor));
    } else if (tensor->type == 0) {
        weight = backend_.weight_f32(
            reinterpret_cast<const float *>(tensor_payload(model_, *tensor)),
            tensor_rows(*tensor) * tensor_cols(*tensor),
            tensor_label(*tensor));
    } else {
        throw std::runtime_error("unsupported tensor type for device backend: " + tensor->name);
    }
    uploaded_bytes_ += tensor->bytes;
    DeviceWeight *raw = weight.get();
    owned_.push_back(std::move(weight));
    by_tensor_.emplace(tensor, raw);
    return raw;
}

} // namespace qw3
