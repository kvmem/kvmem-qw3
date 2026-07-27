#pragma once

#include "qw3/gguf.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

enum class ModelTensorType {
    F32,
    F16,
    BF16,
    FP8_E4M3,
    Q8_0,
    NVFP4_E2M1,
    U8,
    I8,
    I32,
    Unknown,
};

enum class ModelTensorReorder {
    None,
    Rows,
    Columns,
};

const char *model_tensor_type_name(ModelTensorType type);

struct ModelTensorInfo {
    std::string name;
    std::string source_name;
    // Internal tensor order follows GGUF: input/contiguous dimension first.
    std::vector<uint64_t> dims;
    ModelTensorType type = ModelTensorType::Unknown;
    const void *data = nullptr;
    uint64_t bytes = 0;

    // Quantization auxiliaries. NVFP4 uses all four fields; FP8 uses scale.
    const void *scale_data = nullptr;
    uint64_t scale_bytes = 0;
    std::vector<uint64_t> scale_dims;
    // Normalized FP32 storage for scale tensors persisted in BF16.
    std::vector<float> owned_scale_f32;
    float input_global_scale_divisor = 1.0f;
    float weight_global_scale_divisor = 1.0f;

    // HF Qwen3.5 stores several tensors in a parameterization/layout that the
    // GGUF-oriented executor does not consume directly. These tags request a
    // one-shot staging transform during device upload; GGUF tensors leave them
    // at their defaults.
    bool add_one = false;
    bool negative_exp = false;
    ModelTensorReorder reorder = ModelTensorReorder::None;
    uint64_t reorder_offset = 0;
    uint64_t reorder_count = 0;
    uint32_t reorder_head_dim = 0;
    uint32_t reorder_k_heads = 0;
    uint32_t reorder_v_heads = 0;
};

struct ModelSourceInfo {
    std::string format;
    std::string architecture;
    uint64_t tensor_count = 0;
    uint64_t tensor_bytes = 0;
};

class ModelSource {
public:
    virtual ~ModelSource() = default;

    virtual const ModelTensorInfo *find_tensor(const std::string &internal_name) const = 0;
    virtual const ModelSourceInfo &info() const = 0;
    virtual const GgufFile *gguf() const { return nullptr; }
    virtual const std::string &model_directory() const = 0;
};

std::unique_ptr<ModelSource> make_gguf_model_source(std::unique_ptr<GgufFile> gguf);
std::unique_ptr<ModelSource> open_model_source(const std::string &path);

} // namespace qw3
