#include "qw3/dflash_config.hpp"

#include "json.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace qw3 {
namespace {
using json = nlohmann::json;

uint32_t integer(const json &object, const char *name, bool positive = true) {
    const auto &value = object.at(name);
    if (!value.is_number_integer() || value < 0 ||
        value > std::numeric_limits<uint32_t>::max() || (positive && value == 0))
        throw std::runtime_error(std::string("invalid DFlash integer: ") + name);
    return value.get<uint32_t>();
}

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(std::string("DFlash 2 config: ") + message);
}
} // namespace

DFlashConfig::DFlashConfig(const std::string &directory) {
    const auto path = std::filesystem::path(directory) / "config.json";
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open DFlash config: " + path.string());
    json root;
    input >> root;
    require(root.at("architectures") == json::array({"DFlash2DraftModel"}),
            "expected DFlash2DraftModel architecture");
    require(root.at("model_type") == "qwen3", "expected qwen3 draft backbone");
    require(!root.at("is_causal").get<bool>(), "draft attention must be non-causal");
    require(!root.value("attention_bias", false), "attention bias is unsupported");
    require(root.value("hidden_act", std::string("silu")) == "silu", "expected SiLU FFN");
    require(!root.value("tie_word_embeddings", false), "draft must share target embedding/head explicitly");
    hidden_size = integer(root, "hidden_size");
    intermediate_size = integer(root, "intermediate_size");
    num_layers = integer(root, "num_hidden_layers");
    num_target_layers = integer(root, "num_target_layers");
    num_heads = integer(root, "num_attention_heads");
    num_kv_heads = integer(root, "num_key_value_heads");
    head_dim = integer(root, "head_dim");
    vocab_size = integer(root, "vocab_size");
    max_position_embeddings = integer(root, "max_position_embeddings");
    sliding_window = integer(root, "sliding_window");
    require(num_heads % num_kv_heads == 0, "invalid grouped-query head counts");
    require(head_dim % 2 == 0, "RoPE head dimension must be even");
    require(sliding_window <= max_position_embeddings, "sliding window exceeds context length");
    require(root.at("use_sliding_window").get<bool>(), "sliding attention must be enabled");
    const auto types = root.at("layer_types").get<std::vector<std::string>>();
    require(types.size() == num_layers, "layer_types length mismatch");
    for (const auto &type : types) require(type == "sliding_attention", "unsupported draft layer type");
    rms_norm_eps = root.at("rms_norm_eps").get<float>();
    const auto &rope = root.at("rope_parameters");
    require(rope.at("rope_type") == "default", "unsupported draft RoPE scaling");
    rope_theta = rope.at("rope_theta").get<float>();
    require(std::isfinite(rms_norm_eps) && rms_norm_eps > 0, "invalid norm epsilon");
    require(std::isfinite(rope_theta) && rope_theta > 0, "invalid RoPE theta");
    const auto &draft = root.at("dflash_config");
    block_size = integer(draft, "block_size");
    mask_token_id = integer(draft, "mask_token_id", false);
    conv_kernel_size = integer(draft, "conv_kernel_size");
    conv_group_size = integer(draft, "conv_group_size");
    selector_rank = integer(draft, "selector_rank");
    selector_top_k = integer(draft, "selector_top_k");
    require(block_size >= 2 && block_size <= sliding_window, "invalid speculation block size");
    require(mask_token_id < vocab_size, "mask token is outside the vocabulary");
    require(hidden_size % conv_group_size == 0, "convolution groups do not divide hidden size");
    require(conv_kernel_size <= block_size, "convolution kernel exceeds block size");
    require(selector_top_k <= vocab_size, "selector top-k exceeds vocabulary");
    const auto &layers = draft.at("target_layer_ids");
    require(layers.is_array() && !layers.empty(), "missing target feature layers");
    std::set<uint32_t> unique;
    for (const auto &id : layers) {
        require(id.is_number_integer() && id >= 0 && id < num_target_layers,
                "target feature layer is outside the model");
        const auto layer = id.get<uint32_t>();
        require(unique.insert(layer).second, "duplicate target feature layer");
        target_layer_ids.push_back(layer);
    }
}

void DFlashConfig::validate_target(uint32_t layers, uint32_t hidden, uint32_t vocab) const {
    require(layers == num_target_layers, "target layer count mismatch");
    require(hidden == hidden_size, "target hidden size mismatch");
    require(vocab == vocab_size, "target vocabulary size mismatch");
}

} // namespace qw3
