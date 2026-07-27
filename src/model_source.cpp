#include "qw3/model_source.hpp"

#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace qw3 {
namespace {

using json = nlohmann::json;

json read_json(const std::filesystem::path &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open JSON file: " + path.string());
    json value;
    in >> value;
    return value;
}

uint64_t checked_product(const std::vector<uint64_t> &dims) {
    uint64_t value = 1;
    for (uint64_t dim : dims) {
        if (dim != 0 && value > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error("tensor shape overflows uint64");
        }
        value *= dim;
    }
    return value;
}

uint32_t dtype_bytes(ModelTensorType type) {
    switch (type) {
        case ModelTensorType::F32:
        case ModelTensorType::I32: return 4;
        case ModelTensorType::F16:
        case ModelTensorType::BF16: return 2;
        case ModelTensorType::FP8_E4M3:
        case ModelTensorType::U8:
        case ModelTensorType::I8: return 1;
        default: return 0;
    }
}

ModelTensorType safetensors_dtype(const std::string &name) {
    if (name == "F32") return ModelTensorType::F32;
    if (name == "F16") return ModelTensorType::F16;
    if (name == "BF16") return ModelTensorType::BF16;
    if (name == "F8_E4M3" || name == "F8_E4M3FN") return ModelTensorType::FP8_E4M3;
    if (name == "U8") return ModelTensorType::U8;
    if (name == "I8") return ModelTensorType::I8;
    if (name == "I32") return ModelTensorType::I32;
    return ModelTensorType::Unknown;
}

std::vector<uint64_t> reversed_shape(const std::vector<uint64_t> &shape) {
    return std::vector<uint64_t>(shape.rbegin(), shape.rend());
}

float read_scalar_f32(const ModelTensorInfo &tensor) {
    if (!tensor.data || checked_product(tensor.dims) != 1) {
        throw std::runtime_error("expected scalar tensor: " + tensor.source_name);
    }
    if (tensor.type == ModelTensorType::F32) {
        float value = 0.0f;
        std::memcpy(&value, tensor.data, sizeof(value));
        return value;
    }
    throw std::runtime_error("expected F32 scalar tensor: " + tensor.source_name);
}

void set_fp8_scale(ModelTensorInfo &weight, const ModelTensorInfo &scale) {
    const uint64_t count = checked_product(scale.dims);
    weight.scale_dims = scale.dims;
    if (scale.type == ModelTensorType::F32) {
        weight.scale_data = scale.data;
        weight.scale_bytes = scale.bytes;
        return;
    }
    if (scale.type != ModelTensorType::BF16) {
        throw std::runtime_error("unsupported FP8 scale type " +
                                 std::string(model_tensor_type_name(scale.type)) +
                                 ": " + scale.source_name);
    }
    const auto *source = static_cast<const uint16_t *>(scale.data);
    weight.owned_scale_f32.resize(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) {
        const uint32_t bits = static_cast<uint32_t>(source[i]) << 16;
        std::memcpy(&weight.owned_scale_f32[static_cast<size_t>(i)],
                    &bits, sizeof(bits));
    }
    weight.scale_data = weight.owned_scale_f32.data();
    weight.scale_bytes = count * sizeof(float);
}

class GgufModelSource final : public ModelSource {
public:
    explicit GgufModelSource(std::unique_ptr<GgufFile> gguf) : gguf_(std::move(gguf)) {
        if (!gguf_) throw std::invalid_argument("GGUF model source requires a file");
        const ModelInfo model = gguf_->model_info();
        info_.format = "gguf";
        info_.architecture = model.architecture;
        info_.tensor_count = model.tensor_count;
        for (const GgufTensorInfo &src : gguf_->tensors()) {
            auto tensor = std::make_unique<ModelTensorInfo>();
            tensor->name = src.name;
            tensor->source_name = src.name;
            tensor->dims = src.dims;
            tensor->type = src.type == 8 ? ModelTensorType::Q8_0
                                         : src.type == 0 ? ModelTensorType::F32
                                                         : ModelTensorType::Unknown;
            tensor->data = gguf_->data() + src.abs_offset;
            tensor->bytes = src.bytes;
            info_.tensor_bytes += src.bytes;
            const ModelTensorInfo *raw = tensor.get();
            tensors_.emplace(src.name, std::move(tensor));
            by_name_.emplace(src.name, raw);
        }
    }

    const ModelTensorInfo *find_tensor(const std::string &name) const override {
        const auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : it->second;
    }
    const ModelSourceInfo &info() const override { return info_; }
    const GgufFile *gguf() const override { return gguf_.get(); }
    const std::string &model_directory() const override { return empty_directory_; }

private:
    std::unique_ptr<GgufFile> gguf_;
    ModelSourceInfo info_;
    std::string empty_directory_;
    std::unordered_map<std::string, std::unique_ptr<ModelTensorInfo>> tensors_;
    std::unordered_map<std::string, const ModelTensorInfo *> by_name_;
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path &path) : path_(path.string()) {
        fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open " + path_ + ": " + std::strerror(errno));
        }
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
            const std::string error = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot stat " + path_ + ": " + error);
        }
        size_ = static_cast<uint64_t>(st.st_size);
        data_ = static_cast<const uint8_t *>(
            ::mmap(nullptr, static_cast<size_t>(size_), PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            const std::string error = std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot mmap " + path_ + ": " + error);
        }
    }

    ~MappedFile() {
        if (data_) ::munmap(const_cast<uint8_t *>(data_), static_cast<size_t>(size_));
        if (fd_ >= 0) ::close(fd_);
    }

    const uint8_t *data() const { return data_; }
    uint64_t size() const { return size_; }

private:
    std::string path_;
    int fd_ = -1;
    const uint8_t *data_ = nullptr;
    uint64_t size_ = 0;
};

struct RawSafetensor {
    ModelTensorType type = ModelTensorType::Unknown;
    std::vector<uint64_t> shape;
    const void *data = nullptr;
    uint64_t bytes = 0;
};

class SafetensorsShard {
public:
    explicit SafetensorsShard(const std::filesystem::path &path) : file_(path) {
        if (file_.size() < sizeof(uint64_t)) {
            throw std::runtime_error("truncated safetensors file: " + path.string());
        }
        uint64_t header_bytes = 0;
        std::memcpy(&header_bytes, file_.data(), sizeof(header_bytes));
        if (header_bytes > file_.size() - sizeof(uint64_t)) {
            throw std::runtime_error("invalid safetensors header length: " + path.string());
        }
        const char *header_begin = reinterpret_cast<const char *>(file_.data() + sizeof(uint64_t));
        const json header = json::parse(header_begin, header_begin + header_bytes);
        const uint64_t payload_offset = sizeof(uint64_t) + header_bytes;
        for (const auto &entry : header.items()) {
            if (entry.key() == "__metadata__") continue;
            const json &desc = entry.value();
            if (!desc.contains("dtype") || !desc.contains("shape") ||
                !desc.contains("data_offsets")) {
                throw std::runtime_error("malformed safetensors entry: " + entry.key());
            }
            RawSafetensor tensor;
            tensor.type = safetensors_dtype(desc.at("dtype").get<std::string>());
            tensor.shape = desc.at("shape").get<std::vector<uint64_t>>();
            const std::vector<uint64_t> offsets =
                desc.at("data_offsets").get<std::vector<uint64_t>>();
            if (offsets.size() != 2 || offsets[1] < offsets[0] ||
                payload_offset + offsets[1] > file_.size()) {
                throw std::runtime_error("invalid safetensors data offsets: " + entry.key());
            }
            tensor.data = file_.data() + payload_offset + offsets[0];
            tensor.bytes = offsets[1] - offsets[0];
            const uint32_t item_bytes = dtype_bytes(tensor.type);
            if (item_bytes != 0 && checked_product(tensor.shape) * item_bytes != tensor.bytes) {
                throw std::runtime_error("safetensors shape/byte mismatch: " + entry.key());
            }
            tensors_.emplace(entry.key(), std::move(tensor));
        }
    }

    const RawSafetensor *find(const std::string &name) const {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

private:
    MappedFile file_;
    std::unordered_map<std::string, RawSafetensor> tensors_;
};

std::string replace_suffix(const std::string &value, const std::string &old_suffix,
                           const std::string &new_suffix) {
    if (value.size() < old_suffix.size() ||
        value.compare(value.size() - old_suffix.size(), old_suffix.size(), old_suffix) != 0) {
        return {};
    }
    return value.substr(0, value.size() - old_suffix.size()) + new_suffix;
}

bool ends_with(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string hf_name_for_internal(const std::string &name) {
    if (name == "token_embd.weight") return "model.language_model.embed_tokens.weight";
    if (name == "output_norm.weight") return "model.language_model.norm.weight";
    if (name == "output.weight") return "lm_head.weight";

    if (name.rfind("blk.", 0) != 0) return {};
    const size_t index_begin = 4;
    const size_t dot = name.find('.', index_begin);
    if (dot == std::string::npos) return {};
    uint32_t layer = 0;
    try {
        layer = static_cast<uint32_t>(std::stoul(name.substr(index_begin, dot - index_begin)));
    } catch (...) {
        return {};
    }
    const std::string suffix = name.substr(dot + 1);
    if (layer == 64) {
        const std::string prefix = "mtp.layers.0.";
        if (suffix == "attn_norm.weight") return prefix + "input_layernorm.weight";
        if (suffix == "ffn_norm.weight" || suffix == "post_attention_norm.weight")
            return prefix + "post_attention_layernorm.weight";
        if (suffix == "attn_q.weight") return prefix + "self_attn.q_proj.weight";
        if (suffix == "attn_k.weight") return prefix + "self_attn.k_proj.weight";
        if (suffix == "attn_v.weight") return prefix + "self_attn.v_proj.weight";
        if (suffix == "attn_q_norm.weight") return prefix + "self_attn.q_norm.weight";
        if (suffix == "attn_k_norm.weight") return prefix + "self_attn.k_norm.weight";
        if (suffix == "attn_output.weight") return prefix + "self_attn.o_proj.weight";
        if (suffix == "ffn_gate.weight") return prefix + "mlp.gate_proj.weight";
        if (suffix == "ffn_up.weight") return prefix + "mlp.up_proj.weight";
        if (suffix == "ffn_down.weight") return prefix + "mlp.down_proj.weight";
        if (suffix == "nextn.eh_proj.weight") return "mtp.fc.weight";
        if (suffix == "nextn.enorm.weight") return "mtp.pre_fc_norm_embedding.weight";
        if (suffix == "nextn.hnorm.weight") return "mtp.pre_fc_norm_hidden.weight";
        if (suffix == "nextn.shared_head_norm.weight") return "mtp.norm.weight";
        return {};
    }

    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    if (suffix == "attn_norm.weight") return prefix + "input_layernorm.weight";
    if (suffix == "ffn_norm.weight" || suffix == "post_attention_norm.weight")
        return prefix + "post_attention_layernorm.weight";
    if (suffix == "attn_q.weight") return prefix + "self_attn.q_proj.weight";
    if (suffix == "attn_k.weight") return prefix + "self_attn.k_proj.weight";
    if (suffix == "attn_v.weight") return prefix + "self_attn.v_proj.weight";
    if (suffix == "attn_q_norm.weight") return prefix + "self_attn.q_norm.weight";
    if (suffix == "attn_k_norm.weight") return prefix + "self_attn.k_norm.weight";
    if (suffix == "attn_output.weight") return prefix + "self_attn.o_proj.weight";
    if (suffix == "attn_qkv.weight") return prefix + "linear_attn.in_proj_qkv.weight";
    if (suffix == "attn_gate.weight") return prefix + "linear_attn.in_proj_z.weight";
    if (suffix == "ssm_a") return prefix + "linear_attn.A_log";
    if (suffix == "ssm_alpha.weight") return prefix + "linear_attn.in_proj_a.weight";
    if (suffix == "ssm_beta.weight") return prefix + "linear_attn.in_proj_b.weight";
    if (suffix == "ssm_conv1d.weight") return prefix + "linear_attn.conv1d.weight";
    if (suffix == "ssm_dt.bias") return prefix + "linear_attn.dt_bias";
    if (suffix == "ssm_norm.weight") return prefix + "linear_attn.norm.weight";
    if (suffix == "ssm_out.weight") return prefix + "linear_attn.out_proj.weight";
    if (suffix == "ffn_gate.weight") return prefix + "mlp.gate_proj.weight";
    if (suffix == "ffn_up.weight") return prefix + "mlp.up_proj.weight";
    if (suffix == "ffn_down.weight") return prefix + "mlp.down_proj.weight";
    return {};
}

class HfSafetensorsModelSource final : public ModelSource {
public:
    explicit HfSafetensorsModelSource(std::filesystem::path directory)
        : directory_(std::filesystem::canonical(std::move(directory)).string()) {
        const json config = read_json(std::filesystem::path(directory_) / "config.json");
        const json index = read_json(std::filesystem::path(directory_) /
                                     "model.safetensors.index.json");
        if (!index.contains("weight_map")) {
            throw std::runtime_error("safetensors index has no weight_map");
        }
        weight_map_ = index.at("weight_map").get<std::unordered_map<std::string, std::string>>();
        info_.format = "safetensors/compressed-tensors";
        info_.architecture = "qwen35";
        info_.tensor_count = weight_map_.size();
        if (index.contains("metadata")) {
            info_.tensor_bytes = index.at("metadata").value("total_size", uint64_t{0});
        }
        const std::string method = config.value("quantization_config", json::object())
                                       .value("quant_method", std::string());
        if (method != "compressed-tensors") {
            throw std::runtime_error("HF model is not compressed-tensors: " + method);
        }
        const json &text = config.at("text_config");
        linear_k_heads_ = text.at("linear_num_key_heads").get<uint32_t>();
        linear_v_heads_ = text.at("linear_num_value_heads").get<uint32_t>();
        linear_k_head_dim_ = text.at("linear_key_head_dim").get<uint32_t>();
        linear_v_head_dim_ = text.at("linear_value_head_dim").get<uint32_t>();
        if (linear_k_heads_ == 0 || linear_v_heads_ == 0 ||
            linear_v_heads_ % linear_k_heads_ != 0) {
            throw std::runtime_error("unsupported Qwen DeltaNet head grouping");
        }
    }

    const ModelTensorInfo *find_tensor(const std::string &internal_name) const override {
        const auto cached = logical_.find(internal_name);
        if (cached != logical_.end()) return cached->second.get();
        const std::string hf_name = hf_name_for_internal(internal_name);
        if (hf_name.empty()) return nullptr;

        const RawSafetensor *raw = raw_tensor(hf_name);
        std::string packed_name;
        if (!raw) {
            packed_name = replace_suffix(hf_name, ".weight", ".weight_packed");
            if (!packed_name.empty()) raw = raw_tensor(packed_name);
        }
        if (!raw) return nullptr;

        auto tensor = std::make_unique<ModelTensorInfo>();
        tensor->name = internal_name;
        tensor->source_name = packed_name.empty() ? hf_name : packed_name;
        tensor->data = raw->data;
        tensor->bytes = raw->bytes;
        tensor->dims = reversed_shape(raw->shape);

        if (!packed_name.empty()) {
            tensor->type = ModelTensorType::NVFP4_E2M1;
            if (tensor->dims.empty()) throw std::runtime_error("NVFP4 weight has no dimensions");
            tensor->dims[0] *= 2;
            const std::string base = replace_suffix(hf_name, ".weight", "");
            const ModelTensorInfo scale = raw_info(base + ".weight_scale");
            const ModelTensorInfo input_global = raw_info(base + ".input_global_scale");
            const ModelTensorInfo weight_global = raw_info(base + ".weight_global_scale");
            tensor->scale_data = scale.data;
            tensor->scale_bytes = scale.bytes;
            tensor->scale_dims = scale.dims;
            tensor->input_global_scale_divisor = read_scalar_f32(input_global);
            tensor->weight_global_scale_divisor = read_scalar_f32(weight_global);
        } else if (raw->type == ModelTensorType::FP8_E4M3) {
            tensor->type = ModelTensorType::FP8_E4M3;
            const std::string scale_name = replace_suffix(hf_name, ".weight", ".weight_scale");
            const ModelTensorInfo scale = raw_info(scale_name);
            set_fp8_scale(*tensor, scale);
        } else {
            tensor->type = raw->type;
        }
        set_compatibility_transform(*tensor, hf_name);

        const ModelTensorInfo *result = tensor.get();
        logical_.emplace(internal_name, std::move(tensor));
        return result;
    }

    const ModelSourceInfo &info() const override { return info_; }
    const std::string &model_directory() const override { return directory_; }

private:
    void set_row_reorder(ModelTensorInfo &tensor,
                         uint64_t offset,
                         uint64_t count,
                         uint32_t head_dim) const {
        tensor.reorder = ModelTensorReorder::Rows;
        tensor.reorder_offset = offset;
        tensor.reorder_count = count;
        tensor.reorder_head_dim = head_dim;
        tensor.reorder_k_heads = linear_k_heads_;
        tensor.reorder_v_heads = linear_v_heads_;
    }

    void set_compatibility_transform(ModelTensorInfo &tensor,
                                     const std::string &hf_name) const {
        // Qwen3.5's ordinary RMSNorm stores an additive offset, while the
        // executor (and converted GGUF) stores the final multiplicative weight.
        const bool mtp_pre_fc_norm =
            hf_name == "mtp.pre_fc_norm_embedding.weight" ||
            hf_name == "mtp.pre_fc_norm_hidden.weight";
        tensor.add_one =
            (ends_with(hf_name, "norm.weight") &&
             !ends_with(hf_name, "linear_attn.norm.weight")) ||
            mtp_pre_fc_norm;

        if (hf_name.find(".linear_attn.") == std::string::npos) return;
        const uint64_t qk_rows =
            static_cast<uint64_t>(linear_k_heads_) * linear_k_head_dim_;
        const uint64_t v_rows =
            static_cast<uint64_t>(linear_v_heads_) * linear_v_head_dim_;

        if (ends_with(hf_name, ".in_proj_qkv.weight")) {
            set_row_reorder(tensor, 2 * qk_rows, v_rows, linear_v_head_dim_);
        } else if (ends_with(hf_name, ".in_proj_z.weight")) {
            set_row_reorder(tensor, 0, v_rows, linear_v_head_dim_);
        } else if (ends_with(hf_name, ".in_proj_a.weight") ||
                   ends_with(hf_name, ".in_proj_b.weight")) {
            set_row_reorder(tensor, 0, linear_v_heads_, 1);
        } else if (ends_with(hf_name, ".A_log")) {
            tensor.negative_exp = true;
            tensor.reorder = ModelTensorReorder::Columns;
            tensor.reorder_offset = 0;
            tensor.reorder_count = linear_v_heads_;
            tensor.reorder_head_dim = 1;
            tensor.reorder_k_heads = linear_k_heads_;
            tensor.reorder_v_heads = linear_v_heads_;
        } else if (ends_with(hf_name, ".dt_bias")) {
            tensor.reorder = ModelTensorReorder::Columns;
            tensor.reorder_offset = 0;
            tensor.reorder_count = linear_v_heads_;
            tensor.reorder_head_dim = 1;
            tensor.reorder_k_heads = linear_k_heads_;
            tensor.reorder_v_heads = linear_v_heads_;
        } else if (ends_with(hf_name, ".conv1d.weight")) {
            set_row_reorder(tensor, 2 * qk_rows, v_rows, linear_v_head_dim_);
        } else if (ends_with(hf_name, ".out_proj.weight")) {
            tensor.reorder = ModelTensorReorder::Columns;
            tensor.reorder_offset = 0;
            tensor.reorder_count = v_rows;
            tensor.reorder_head_dim = linear_v_head_dim_;
            tensor.reorder_k_heads = linear_k_heads_;
            tensor.reorder_v_heads = linear_v_heads_;
        }
    }

    const SafetensorsShard &shard(const std::string &filename) const {
        auto it = shards_.find(filename);
        if (it != shards_.end()) return *it->second;
        const std::filesystem::path path = std::filesystem::path(directory_) / filename;
        if (!std::filesystem::is_regular_file(path)) {
            throw std::runtime_error("model shard is missing (download incomplete): " +
                                     path.string());
        }
        auto value = std::make_unique<SafetensorsShard>(path);
        const SafetensorsShard *raw = value.get();
        shards_.emplace(filename, std::move(value));
        return *raw;
    }

    const RawSafetensor *raw_tensor(const std::string &name) const {
        const auto file = weight_map_.find(name);
        if (file == weight_map_.end()) return nullptr;
        const RawSafetensor *value = shard(file->second).find(name);
        if (!value) throw std::runtime_error("tensor missing from indexed shard: " + name);
        return value;
    }

    ModelTensorInfo raw_info(const std::string &name) const {
        const RawSafetensor *raw = raw_tensor(name);
        if (!raw) throw std::runtime_error("required quantization tensor is missing: " + name);
        ModelTensorInfo value;
        value.name = name;
        value.source_name = name;
        value.type = raw->type;
        value.dims = reversed_shape(raw->shape);
        value.data = raw->data;
        value.bytes = raw->bytes;
        return value;
    }

    std::string directory_;
    ModelSourceInfo info_;
    std::unordered_map<std::string, std::string> weight_map_;
    mutable std::unordered_map<std::string, std::unique_ptr<SafetensorsShard>> shards_;
    mutable std::unordered_map<std::string, std::unique_ptr<ModelTensorInfo>> logical_;
    uint32_t linear_k_heads_ = 0;
    uint32_t linear_v_heads_ = 0;
    uint32_t linear_k_head_dim_ = 0;
    uint32_t linear_v_head_dim_ = 0;
};

} // namespace

const char *model_tensor_type_name(ModelTensorType type) {
    switch (type) {
        case ModelTensorType::F32: return "F32";
        case ModelTensorType::F16: return "F16";
        case ModelTensorType::BF16: return "BF16";
        case ModelTensorType::FP8_E4M3: return "FP8_E4M3";
        case ModelTensorType::Q8_0: return "Q8_0";
        case ModelTensorType::NVFP4_E2M1: return "NVFP4_E2M1";
        case ModelTensorType::U8: return "U8";
        case ModelTensorType::I8: return "I8";
        case ModelTensorType::I32: return "I32";
        default: return "unknown";
    }
}

std::unique_ptr<ModelSource> make_gguf_model_source(std::unique_ptr<GgufFile> gguf) {
    return std::make_unique<GgufModelSource>(std::move(gguf));
}

std::unique_ptr<ModelSource> open_model_source(const std::string &path) {
    const std::filesystem::path model_path(path);
    if (std::filesystem::is_directory(model_path)) {
        return std::make_unique<HfSafetensorsModelSource>(model_path);
    }
    return make_gguf_model_source(std::make_unique<GgufFile>(path));
}

} // namespace qw3
