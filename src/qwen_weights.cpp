#include "qwen_weights.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qw3 {
namespace {

uint64_t tensor_rows(const ModelTensorInfo &tensor) {
    if (tensor.dims.size() < 2) return 1;
    uint64_t rows = 1;
    for (size_t i = 1; i < tensor.dims.size(); ++i) rows *= tensor.dims[i];
    return rows;
}

uint64_t tensor_cols(const ModelTensorInfo &tensor) {
    if (tensor.dims.empty()) return 1;
    return tensor.dims[0];
}

const char *tensor_label(const ModelTensorInfo &tensor) {
    return tensor.name.empty() ? "tensor" : tensor.name.c_str();
}

uint32_t tensor_item_bytes(ModelTensorType type) {
    switch (type) {
        case ModelTensorType::F32: return 4;
        case ModelTensorType::BF16:
        case ModelTensorType::F16: return 2;
        case ModelTensorType::FP8_E4M3: return 1;
        default: return 0;
    }
}

float tensor_value_f32(const void *data, ModelTensorType type, uint64_t index) {
    if (type == ModelTensorType::F32) {
        return static_cast<const float *>(data)[index];
    }
    if (type == ModelTensorType::BF16) {
        const uint32_t bits =
            static_cast<uint32_t>(static_cast<const uint16_t *>(data)[index]) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    throw std::runtime_error("HF scalar transform requires F32 or BF16 input");
}

uint64_t reordered_old_index(uint64_t new_index,
                             uint32_t head_dim,
                             uint32_t k_heads,
                             uint32_t v_heads) {
    if (head_dim == 0 || k_heads == 0 || v_heads == 0 ||
        v_heads % k_heads != 0) {
        throw std::runtime_error("invalid Qwen DeltaNet reorder metadata");
    }
    const uint64_t values_per_k = v_heads / k_heads;
    const uint64_t head_slot = new_index / head_dim;
    const uint64_t component = new_index % head_dim;
    const uint64_t value_in_group = head_slot / k_heads;
    const uint64_t key_head = head_slot % k_heads;
    return (key_head * values_per_k + value_in_group) * head_dim + component;
}

struct PreparedTensor {
    ModelTensorType type = ModelTensorType::Unknown;
    const void *data = nullptr;
    const void *scale_data = nullptr;
    uint64_t scale_count = 0;
    std::vector<float> f32_data;
    std::vector<uint8_t> reordered_data;
    std::vector<float> reordered_scales;
};

PreparedTensor prepare_tensor(const ModelTensorInfo &tensor) {
    PreparedTensor prepared;
    prepared.type = tensor.type;
    prepared.data = tensor.data;
    prepared.scale_data = tensor.scale_data;
    prepared.scale_count = tensor.scale_bytes / sizeof(float);
    const uint64_t rows = tensor_rows(tensor);
    const uint64_t cols = tensor_cols(tensor);
    const uint64_t elements = rows * cols;

    if (tensor.add_one || tensor.negative_exp) {
        prepared.f32_data.resize(static_cast<size_t>(elements));
        for (uint64_t i = 0; i < elements; ++i) {
            float value = tensor_value_f32(tensor.data, tensor.type, i);
            if (tensor.add_one) value += 1.0f;
            if (tensor.negative_exp) value = -std::exp(value);
            prepared.f32_data[static_cast<size_t>(i)] = value;
        }
        prepared.type = ModelTensorType::F32;
        prepared.data = prepared.f32_data.data();
    }

    if (tensor.reorder == ModelTensorReorder::None) return prepared;
    if (tensor.reorder_count !=
        static_cast<uint64_t>(tensor.reorder_v_heads) * tensor.reorder_head_dim) {
        throw std::runtime_error("Qwen DeltaNet reorder count mismatch: " + tensor.name);
    }
    const uint32_t item_bytes = tensor_item_bytes(prepared.type);
    if (item_bytes == 0 || elements * item_bytes > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("unsupported Qwen reorder tensor type: " + tensor.name);
    }
    const size_t bytes = static_cast<size_t>(elements * item_bytes);
    const auto *source = static_cast<const uint8_t *>(prepared.data);
    prepared.reordered_data.assign(source, source + bytes);

    if (tensor.reorder == ModelTensorReorder::Rows) {
        if (tensor.reorder_offset + tensor.reorder_count > rows) {
            throw std::runtime_error("Qwen row reorder exceeds tensor: " + tensor.name);
        }
        const size_t row_bytes = static_cast<size_t>(cols * item_bytes);
        for (uint64_t new_row = 0; new_row < tensor.reorder_count; ++new_row) {
            const uint64_t old_row = reordered_old_index(
                new_row, tensor.reorder_head_dim,
                tensor.reorder_k_heads, tensor.reorder_v_heads);
            std::memcpy(
                prepared.reordered_data.data() +
                    static_cast<size_t>(tensor.reorder_offset + new_row) * row_bytes,
                source + static_cast<size_t>(tensor.reorder_offset + old_row) * row_bytes,
                row_bytes);
        }
        if (prepared.type == ModelTensorType::FP8_E4M3) {
            if (!prepared.scale_data || prepared.scale_count != rows) {
                throw std::runtime_error("FP8 scale cannot follow Qwen row reorder: " +
                                         tensor.name);
            }
            const auto *source_scales = static_cast<const float *>(prepared.scale_data);
            prepared.reordered_scales.assign(source_scales, source_scales + rows);
            for (uint64_t new_row = 0; new_row < tensor.reorder_count; ++new_row) {
                const uint64_t old_row = reordered_old_index(
                    new_row, tensor.reorder_head_dim,
                    tensor.reorder_k_heads, tensor.reorder_v_heads);
                prepared.reordered_scales[static_cast<size_t>(tensor.reorder_offset + new_row)] =
                    source_scales[tensor.reorder_offset + old_row];
            }
            prepared.scale_data = prepared.reordered_scales.data();
        }
    } else {
        if (tensor.reorder_offset + tensor.reorder_count > cols) {
            throw std::runtime_error("Qwen column reorder exceeds tensor: " + tensor.name);
        }
        for (uint64_t row = 0; row < rows; ++row) {
            for (uint64_t new_col = 0; new_col < tensor.reorder_count; ++new_col) {
                const uint64_t old_col = reordered_old_index(
                    new_col, tensor.reorder_head_dim,
                    tensor.reorder_k_heads, tensor.reorder_v_heads);
                std::memcpy(
                    prepared.reordered_data.data() +
                        static_cast<size_t>(row * cols + tensor.reorder_offset + new_col) *
                            item_bytes,
                    source +
                        static_cast<size_t>(row * cols + tensor.reorder_offset + old_col) *
                            item_bytes,
                    item_bytes);
            }
        }
    }
    prepared.data = prepared.reordered_data.data();
    return prepared;
}

} // namespace

QwenWeights::QwenWeights(const QwenNativeModel &model, DeviceBackend &backend,
                         bool cpu_embedding)
    : model_(model), backend_(backend) {
    const NativePlanInfo &plan = model_.plan();
    if (!plan.supported) {
        throw std::runtime_error("native model plan is incomplete; cannot upload weights");
    }

    if (cpu_embedding && model_.output() == model_.token_embedding()) {
        throw std::runtime_error(
            "--cpu-embedding requires a separate output/LM-head weight");
    }
    token_embd_ = cpu_embedding
        ? bind_host_bf16(model_.token_embedding())
        : bind(model_.token_embedding());
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
    auto prepare_fp8_pair = [&](DeviceWeight *first,
                                DeviceWeight *second,
                                const char *label) {
        if (!first || !second ||
            first->format != DeviceWeightFormat::FP8_E4M3 ||
            second->format != DeviceWeightFormat::FP8_E4M3) {
            return;
        }
        if (auto st = backend_.prepare_fp8_fanout_pair(*first, *second);
            !st.ok) {
            throw std::runtime_error(
                std::string("failed to prepare ") + label + ": " +
                st.message);
        }
    };
    auto prepare_nvfp4_pair = [&](DeviceWeight *first,
                                  DeviceWeight *second,
                                  const char *label) {
        if (!first || !second ||
            first->format != DeviceWeightFormat::NVFP4_E2M1 ||
            second->format != DeviceWeightFormat::NVFP4_E2M1) {
            return;
        }
        if (auto st = backend_.prepare_nvfp4_fanout_pair(*first, *second);
            !st.ok) {
            throw std::runtime_error(
                std::string("failed to prepare ") + label + ": " +
                st.message);
        }
    };
    dst.recurrent = src.recurrent;
    dst.attn_norm = bind(src.attn_norm);
    dst.ffn_norm = bind(src.ffn_norm);
    dst.ffn_gate = bind(src.ffn_gate);
    dst.ffn_up = bind(src.ffn_up);
    dst.ffn_down = bind(src.ffn_down);
    dst.ffn_dim = src.ffn_gate ? tensor_rows(*src.ffn_gate) : 0;
    prepare_fp8_pair(dst.ffn_gate, dst.ffn_up, "FP8 FFN gate/up pair");
    prepare_nvfp4_pair(dst.ffn_gate, dst.ffn_up, "NVFP4 FFN gate/up pair");

    if (src.recurrent) {
        dst.attn_qkv = bind(src.attn_qkv);
        dst.attn_gate = bind(src.attn_gate);
        prepare_fp8_pair(
            dst.attn_qkv, dst.attn_gate,
            "FP8 recurrent QKV/gate pair");
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
        prepare_fp8_pair(
            dst.attn_k, dst.attn_v,
            "FP8 attention K/V pair");
        dst.attn_q_norm = bind(src.attn_q_norm);
        dst.attn_k_norm = bind(src.attn_k_norm);
        dst.attn_output = bind(src.attn_output);
        dst.q_rows = src.attn_q ? tensor_rows(*src.attn_q) : 0;
        dst.k_rows = src.attn_k ? tensor_rows(*src.attn_k) : 0;
        dst.v_rows = src.attn_v ? tensor_rows(*src.attn_v) : 0;
    }
    return dst;
}

DeviceWeight *QwenWeights::bind(const ModelTensorInfo *tensor) {
    if (!tensor) return nullptr;
    auto it = by_tensor_.find(tensor);
    if (it != by_tensor_.end()) return it->second;

    PreparedTensor prepared = prepare_tensor(*tensor);
    std::unique_ptr<DeviceWeight> weight;
    if (prepared.type == ModelTensorType::Q8_0) {
        weight = backend_.weight_q8_0(prepared.data,
                                      tensor_rows(*tensor),
                                      tensor_cols(*tensor),
                                      tensor_label(*tensor));
    } else if (prepared.type == ModelTensorType::BF16) {
        weight = backend_.weight_bf16(prepared.data,
                                      tensor_rows(*tensor),
                                      tensor_cols(*tensor),
                                      tensor_label(*tensor));
    } else if (prepared.type == ModelTensorType::FP8_E4M3) {
        weight = backend_.weight_fp8_e4m3(prepared.data,
                                          prepared.scale_data,
                                          prepared.scale_count,
                                          tensor_rows(*tensor),
                                          tensor_cols(*tensor),
                                          tensor_label(*tensor));
    } else if (prepared.type == ModelTensorType::NVFP4_E2M1) {
        if (tensor->scale_dims.empty()) {
            throw std::runtime_error("NVFP4 tensor has no scale dimensions: " +
                                     tensor->name);
        }
        uint64_t scale_rows = 1;
        for (size_t i = 1; i < tensor->scale_dims.size(); ++i) {
            scale_rows *= tensor->scale_dims[i];
        }
        weight = backend_.weight_nvfp4_e2m1(
            prepared.data, tensor->scale_data,
            scale_rows, tensor->scale_dims[0],
            tensor->input_global_scale_divisor,
            tensor->weight_global_scale_divisor,
            tensor_rows(*tensor), tensor_cols(*tensor),
            tensor_label(*tensor));
    } else if (prepared.type == ModelTensorType::F32) {
        weight = backend_.weight_f32(
            reinterpret_cast<const float *>(prepared.data),
            tensor_rows(*tensor) * tensor_cols(*tensor),
            tensor_label(*tensor));
    } else {
        throw std::runtime_error("unsupported tensor type " +
                                 std::string(model_tensor_type_name(prepared.type)) +
                                 " for device backend: " + tensor->name);
    }
    uploaded_bytes_ += tensor->bytes;
    DeviceWeight *raw = weight.get();
    if (raw->format == DeviceWeightFormat::NVFP4_E2M1) {
        uses_nvfp4_ = true;
    } else if (raw->format == DeviceWeightFormat::Q8_0) {
        uses_q8_ = true;
    }
    owned_.push_back(std::move(weight));
    by_tensor_.emplace(tensor, raw);
    return raw;
}

DeviceWeight *QwenWeights::bind_host_bf16(const ModelTensorInfo *tensor) {
    if (!tensor) return nullptr;
    auto it = by_tensor_.find(tensor);
    if (it != by_tensor_.end()) return it->second;
    if (tensor->type != ModelTensorType::BF16 || tensor->add_one ||
        tensor->negative_exp || tensor->reorder != ModelTensorReorder::None) {
        throw std::runtime_error(
            "--cpu-embedding requires an untransformed BF16 token embedding: " +
            tensor->name);
    }
    const uint64_t rows = tensor_rows(*tensor);
    const uint64_t cols = tensor_cols(*tensor);
    if (!tensor->data || cols == 0 ||
        rows > std::numeric_limits<uint64_t>::max() / cols ||
        rows * cols > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t) ||
        tensor->bytes != rows * cols * sizeof(uint16_t)) {
        throw std::runtime_error(
            "invalid BF16 token embedding storage: " + tensor->name);
    }
    std::unique_ptr<DeviceWeight> weight = backend_.weight_bf16_host(
        tensor->data, rows, cols, tensor_label(*tensor));
    host_resident_bytes_ += tensor->bytes;
    DeviceWeight *raw = weight.get();
    owned_.push_back(std::move(weight));
    by_tensor_.emplace(tensor, raw);
    return raw;
}

} // namespace qw3
