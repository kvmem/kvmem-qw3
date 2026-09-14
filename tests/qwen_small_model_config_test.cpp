#include "qw3/model_source.hpp"
#include "qw3/qwen_config.hpp"
#include "json.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using nlohmann::json;

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("qw3-small-config-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    try {
        json config = {{"quantization_config", {{"quant_method", "compressed-tensors"}}},
            {"text_config", {{"model_type", "qwen3_5_text"},
                {"num_hidden_layers", 32}, {"mtp_num_hidden_layers", 1},
                {"hidden_size", 2560}, {"intermediate_size", 9216},
                {"max_position_embeddings", 262144}, {"num_attention_heads", 16},
                {"num_key_value_heads", 4}, {"head_dim", 256},
                {"linear_conv_kernel_dim", 4}, {"linear_num_key_heads", 16},
                {"linear_num_value_heads", 32}, {"linear_key_head_dim", 128},
                {"linear_value_head_dim", 128}, {"full_attention_interval", 4},
                {"vocab_size", 248320},
                {"rope_parameters", {{"partial_rotary_factor", 0.25},
                                     {"rope_theta", 10000000.0}}}}}};
        const json header = {{"mtp.fc.weight", {{"dtype", "BF16"},
            {"shape", {1, 1}}, {"data_offsets", {0, 2}}}}};
        const std::string encoded = header.dump();
        const uint64_t size = encoded.size();
        const uint16_t value = 0;
        {
            std::ofstream tensor(directory / "model.safetensors", std::ios::binary);
            tensor.write(reinterpret_cast<const char *>(&size), sizeof(size));
            tensor.write(encoded.data(), encoded.size());
            tensor.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
        std::ofstream(directory / "model.safetensors.index.json") << json{
            {"weight_map", {{"mtp.fc.weight", "model.safetensors"}}}};
        for (uint32_t layers : {24U, 32U, 64U}) {
            config["text_config"]["num_hidden_layers"] = layers;
            std::ofstream(directory / "config.json") << config;
            qw3::QwenConfig parsed(directory.string());
            if (parsed.n_layers != layers || parsed.block_count != layers + 1 ||
                parsed.rope_dim != 64 || parsed.rope_theta != 10000000.0f ||
                parsed.ssm_inner_size != 8192 || parsed.n_heads / parsed.n_kv_heads != 4) {
                throw std::runtime_error("small-model shape or nested RoPE parsing failed");
            }
            const auto source = qw3::open_model_source(directory.string());
            const auto *mtp = source->find_tensor(
                "blk." + std::to_string(layers) + ".nextn.eh_proj.weight");
            if (!mtp || mtp->source_name != "mtp.fc.weight" ||
                source->find_tensor("blk." + std::to_string(layers - 1) +
                                    ".nextn.eh_proj.weight")) {
                throw std::runtime_error("MTP mapping ignored the configured layer count");
            }
        }
        config["text_config"].erase("rope_parameters");
        config["text_config"]["partial_rotary_factor"] = 0.25;
        config["text_config"]["rope_theta"] = 10000000.0;
        std::ofstream(directory / "config.json") << config;
        if (qw3::QwenConfig(directory.string()).rope_dim != 64) {
            throw std::runtime_error("legacy top-level RoPE parsing regressed");
        }
        std::filesystem::remove_all(directory);
        std::cout << "small-model config and MTP mapping passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
