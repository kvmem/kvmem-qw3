#include "dflash_model.hpp"
#include "json.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
using nlohmann::json;
namespace fs = std::filesystem;

void require(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
void rejected(const std::function<void()> &fn, const std::string &message) {
    try { fn(); } catch (const std::exception &) { return; }
    throw std::runtime_error("accepted invalid input: " + message);
}

json config() {
    return {
        {"architectures", {"DFlash2DraftModel"}}, {"model_type", "qwen3"},
        {"is_causal", false}, {"hidden_size", 16}, {"intermediate_size", 32},
        {"num_hidden_layers", 2}, {"num_target_layers", 4},
        {"num_attention_heads", 4}, {"num_key_value_heads", 2}, {"head_dim", 4},
        {"vocab_size", 8}, {"max_position_embeddings", 128}, {"sliding_window", 32},
        {"use_sliding_window", true}, {"rms_norm_eps", 1e-6},
        {"layer_types", {"sliding_attention", "sliding_attention"}},
        {"rope_parameters", {{"rope_type", "default"}, {"rope_theta", 10000000}}},
        {"dflash_config", {{"block_size", 4}, {"mask_token_id", 7},
            {"conv_kernel_size", 2}, {"conv_group_size", 4},
            {"selector_rank", 2}, {"selector_top_k", 3}, {"target_layer_ids", {0, 2}}}}
    };
}

json tensors() {
    json shape = {
        {"fc.weight", {16, 32}}, {"hidden_norm.weight", {16}}, {"norm.weight", {16}},
        {"candidate_selector.hidden_projection.weight", {2, 16}},
        {"candidate_selector.predecessor_codebook", {8, 2}},
        {"candidate_selector.successor_codebook", {8, 2}}
    };
    const json layer = {
        {"input_layernorm.weight", {16}}, {"post_attention_layernorm.weight", {16}},
        {"self_attn.q_proj.weight", {16, 16}}, {"self_attn.k_proj.weight", {8, 16}},
        {"self_attn.v_proj.weight", {8, 16}}, {"self_attn.o_proj.weight", {16, 16}},
        {"self_attn.q_norm.weight", {4}}, {"self_attn.k_norm.weight", {4}},
        {"mlp.gate_proj.weight", {32, 16}}, {"mlp.up_proj.weight", {32, 16}},
        {"mlp.down_proj.weight", {16, 32}},
        {"attention_conv.base_kernel", {2, 2, 16}},
        {"attention_conv.kernel_projection.weight", {16, 16}},
        {"mlp_conv.base_kernel", {2, 2, 16}}, {"mlp_conv.kernel_projection.weight", {16, 16}}
    };
    for (unsigned i = 0; i < 2; ++i)
        for (const auto &[name, dims] : layer.items())
            shape["layers." + std::to_string(i) + "." + name] = dims;
    return shape;
}

void write_model(const fs::path &directory, const json &cfg, const json &shapes,
                 bool sharded = false) {
    fs::remove(directory / "model.safetensors.index.json");
    std::ofstream(directory / "config.json") << cfg;
    json header;
    json weights;
    uint64_t offset = 0;
    for (const auto &[name, shape] : shapes.items()) {
        uint64_t count = 1;
        for (const auto &dim : shape) count *= dim.get<uint64_t>();
        header[name] = {{"dtype", "BF16"}, {"shape", shape},
                        {"data_offsets", {offset, offset + count * 2}}};
        offset += count * 2;
        weights[name] = sharded ? "part.safetensors" : "model.safetensors";
    }
    std::string encoded = header.dump();
    encoded.append((8 - encoded.size() % 8) % 8, ' ');
    const uint64_t bytes = encoded.size();
    std::ofstream file(directory / (sharded ? "part.safetensors" : "model.safetensors"),
                       std::ios::binary);
    file.write(reinterpret_cast<const char *>(&bytes), sizeof(bytes));
    file.write(encoded.data(), encoded.size());
    // BF16 0.25: must not inherit the Qwen3.5 norm '+1' transformation.
    const std::vector<uint16_t> payload(offset / 2, 0x3e80);
    file.write(reinterpret_cast<const char *>(payload.data()), offset);
    if (sharded)
        std::ofstream(directory / "model.safetensors.index.json") << json{{"weight_map", weights}};
}

void audit(const std::string &directory) {
    const qw3::DFlashModel model(directory);
    const auto &c = model.config();
    std::cout << json{{"directory", directory}, {"tensors", model.tensors().size()},
        {"parameters", model.parameter_count()}, {"weight_bytes", model.tensor_bytes()},
        {"block_size", c.block_size}, {"target_layer_ids", c.target_layer_ids},
        {"draft_layers", c.num_layers}, {"hidden_size", c.hidden_size}}.dump() << '\n';
}
} // namespace

int main(int argc, char **argv) {
    const auto directory = fs::temp_directory_path() / ("qw3-dflash-model-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        if (argc == 2) { audit(argv[1]); return 0; }
        require(argc == 1, "usage: qw3-dflash-model [HF model directory]");
        fs::create_directories(directory);
        for (const bool sharded : {false, true}) {
            write_model(directory, config(), tensors(), sharded);
            qw3::DFlashModel model(directory.string());
            model.config().validate_target(4, 16, 8);
            require(model.tensors().size() == 36, "not all draft tensors were bound");
            const auto identity = model.identity_sha256();
            require(identity.size() == 64 && identity == model.identity_sha256(), "unstable draft identity");
            std::ofstream(directory / "unrelated.txt") << "not a model input";
            require(identity == model.identity_sha256(), "identity includes unrelated files");
            const auto &fc = model.tensor("fc.weight");
            require(fc.dims == std::vector<uint64_t>({32, 16}), "HF matrix order was not reversed");
            const auto &norm = model.tensor("hidden_norm.weight");
            uint16_t value;
            std::memcpy(&value, norm.data, 2);
            require(value == 0x3e80 && !norm.add_one && !norm.negative_exp &&
                    norm.reorder == qw3::ModelTensorReorder::None, "draft norm was transformed");
            rejected([&] { model.config().validate_target(4, 16, 9); }, "incompatible vocabulary");
            rejected([&] { model.config().validate_target(3, 16, 8); }, "incompatible target depth");
            rejected([&] { model.config().validate_target(4, 32, 8); }, "incompatible feature width");
            auto changed = config(); changed["rms_norm_eps"] = 2e-6;
            std::ofstream(directory / "config.json") << changed;
            require(identity != qw3::DFlashModel(directory.string()).identity_sha256(), "identity missed config change");
            std::ofstream(directory / "config.json") << config();
            const auto shard = directory / (sharded ? "part.safetensors" : "model.safetensors");
            std::fstream edit(shard, std::ios::binary | std::ios::in | std::ios::out);
            edit.seekp(-2, std::ios::end); const uint16_t changed_weight = 0x3e81;
            edit.write(reinterpret_cast<const char *>(&changed_weight), sizeof(changed_weight)); edit.close();
            require(identity != qw3::DFlashModel(directory.string()).identity_sha256(), "identity missed same-size weight change");
        }
        const auto invalid = [&](json cfg, json weights) {
            write_model(directory, cfg, weights);
            rejected([&] { qw3::DFlashModel model(directory.string()); }, "model validation");
        };
        auto cfg = config(); cfg["dflash_config"]["target_layer_ids"] = {0, 0};
        invalid(cfg, tensors());
        cfg = config(); cfg["dflash_config"]["target_layer_ids"] = {0, 4};
        invalid(cfg, tensors());
        cfg = config(); cfg["dflash_config"]["conv_group_size"] = 3;
        invalid(cfg, tensors());
        cfg = config(); cfg["dflash_config"]["mask_token_id"] = 8;
        invalid(cfg, tensors());
        cfg = config(); cfg["is_causal"] = true;
        invalid(cfg, tensors());
        cfg = config(); cfg["num_attention_heads"] = -4;
        invalid(cfg, tensors());
        auto shapes = tensors(); shapes.erase("candidate_selector.predecessor_codebook");
        invalid(config(), shapes);
        shapes = tensors(); shapes["fc.weight"] = {16, 16};
        invalid(config(), shapes);
        shapes = tensors(); shapes["layers.1.mlp_conv.base_kernel"] = {2, 3, 16};
        invalid(config(), shapes);
        write_model(directory, config(), tensors());
        fs::resize_file(directory / "model.safetensors", 12);
        rejected([&] { qw3::DFlashModel model(directory.string()); }, "truncated safetensors");
        fs::remove_all(directory);
        std::cout << "DFlash config, standalone/sharded BF16 weights, and corruption checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        fs::remove_all(directory);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
