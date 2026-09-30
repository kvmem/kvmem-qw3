#include "dflash_model.hpp"
#include "qw3/kvmem_archive.hpp"
#include "sha256.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include "json.hpp"

namespace qw3 {

DFlashModel::DFlashModel(const std::string &directory)
    : config_(directory), source_(open_raw_safetensors_source(directory)) {
    const auto &c = config_;
    const uint64_t h = c.hidden_size, ff = c.intermediate_size;
    const uint64_t q = uint64_t(c.num_heads) * c.head_dim;
    const uint64_t kv = uint64_t(c.num_kv_heads) * c.head_dim;
    const uint64_t groups = h / c.conv_group_size;
    bind("fc.weight", {h, h * c.target_layer_ids.size()});
    bind("hidden_norm.weight", {h});
    bind("norm.weight", {h});
    bind("candidate_selector.hidden_projection.weight", {c.selector_rank, h});
    bind("candidate_selector.predecessor_codebook", {c.vocab_size, c.selector_rank});
    bind("candidate_selector.successor_codebook", {c.vocab_size, c.selector_rank});
    for (uint32_t i = 0; i < c.num_layers; ++i) {
        const auto prefix = "layers." + std::to_string(i) + ".";
        bind(prefix + "input_layernorm.weight", {h});
        bind(prefix + "post_attention_layernorm.weight", {h});
        bind(prefix + "self_attn.q_proj.weight", {q, h});
        bind(prefix + "self_attn.k_proj.weight", {kv, h});
        bind(prefix + "self_attn.v_proj.weight", {kv, h});
        bind(prefix + "self_attn.o_proj.weight", {h, q});
        bind(prefix + "self_attn.q_norm.weight", {c.head_dim});
        bind(prefix + "self_attn.k_norm.weight", {c.head_dim});
        bind(prefix + "mlp.gate_proj.weight", {ff, h});
        bind(prefix + "mlp.up_proj.weight", {ff, h});
        bind(prefix + "mlp.down_proj.weight", {h, ff});
        for (const auto *kind : {"attention_conv", "mlp_conv"}) {
            bind(prefix + kind + ".base_kernel", {2, c.conv_kernel_size, h});
            bind(prefix + kind + ".kernel_projection.weight", {2 * c.conv_kernel_size * groups, h});
        }
    }
}

void DFlashModel::bind(const std::string &name, const std::vector<uint64_t> &shape) {
    const auto *tensor = source_->find_tensor(name);
    if (!tensor) throw std::runtime_error("missing DFlash tensor: " + name);
    const std::vector<uint64_t> dims(shape.rbegin(), shape.rend());
    if (tensor->dims != dims) throw std::runtime_error("DFlash tensor shape mismatch: " + name);
    if (tensor->type != ModelTensorType::BF16 && tensor->type != ModelTensorType::F32)
        throw std::runtime_error("DFlash tensor requires BF16 or F32: " + name);
    uint64_t count = 1;
    for (const auto dim : shape) {
        if (dim == 0 || count > std::numeric_limits<uint64_t>::max() / dim)
            throw std::runtime_error("DFlash tensor element count overflow: " + name);
        count *= dim;
    }
    if (count > std::numeric_limits<uint64_t>::max() - parameters_ ||
        tensor->bytes > std::numeric_limits<uint64_t>::max() - bytes_)
        throw std::runtime_error("DFlash total tensor size overflow");
    parameters_ += count;
    bytes_ += tensor->bytes;
    tensors_.emplace(name, tensor);
}

const ModelTensorInfo &DFlashModel::tensor(const std::string &name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("unknown DFlash tensor: " + name);
    return *it->second;
}

std::string DFlashModel::identity_sha256() const {
    // Include the configuration and exactly the files loaded by the raw
    // safetensors source. File digests use the same stat-validated cache as the
    // target GGUF; directory names and unrelated tokenizer files are excluded.
    const std::filesystem::path dir(source_->model_directory());
    std::set<std::string> files{"config.json"};
    const auto index = dir / "model.safetensors.index.json";
    if (std::filesystem::exists(index)) {
        files.insert(index.filename().string());
        std::ifstream in(index);
        const auto map = nlohmann::json::parse(in).at("weight_map");
        for (const auto &value : map) files.insert(value.get<std::string>());
    } else {
        files.insert("model.safetensors");
    }
    detail::Sha256 hash;
    const std::string domain = "qw3-dflash-archive-identity-v1\n";
    hash.update(domain.data(), domain.size());
    for (const auto &name : files) {
        const auto digest = kvmem_archive_model_sha256((dir / name).string());
        const std::string record = std::to_string(name.size()) + ":" + name + ":" + digest + "\n";
        hash.update(record.data(), record.size());
    }
    return hash.finish_hex();
}

} // namespace qw3
