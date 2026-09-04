#include "vision_gpu_frontend.hpp"

#include "json.hpp"
#include "vision_embedding_storage.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace qw3::detail {
namespace {

using json = nlohmann::json;

void require_ok(DeviceStatus status, const char *operation) {
    if (!status.ok) {
        throw std::runtime_error(std::string(operation) + ": " + status.message);
    }
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest, ties to even. Preserve NaNs rather than accidentally
    // rounding them to infinity.
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x7fffu + ((bits >> 16) & 1u);
    } else if ((bits & 0x007fffffu) != 0) {
        bits |= 0x00010000u;
    }
    return static_cast<uint16_t>(bits >> 16);
}

float round_to_bf16(float value) {
    const uint32_t bits = static_cast<uint32_t>(float_to_bf16(value)) << 16;
    float rounded = 0.0f;
    std::memcpy(&rounded, &bits, sizeof(rounded));
    return rounded;
}

uint64_t shape_count(const std::vector<uint64_t> &shape) {
    uint64_t count = 1;
    for (uint64_t dim : shape) {
        if (dim == 0 || count > std::numeric_limits<uint64_t>::max() / dim) {
            throw std::runtime_error("invalid or overflowing safetensors shape");
        }
        count *= dim;
    }
    return count;
}

struct HostTensor {
    std::vector<uint64_t> shape;
    std::vector<uint16_t> bf16;
};

class SafeTensorReader {
public:
    explicit SafeTensorReader(std::filesystem::path model_directory)
        : directory_(std::move(model_directory)) {
        const auto index_path = directory_ / "model.safetensors.index.json";
        std::ifstream index_in(index_path);
        if (!index_in) {
            throw std::runtime_error("missing vision checkpoint index: " +
                                     index_path.string());
        }
        json index;
        index_in >> index;
        for (auto it = index.at("weight_map").begin();
             it != index.at("weight_map").end(); ++it) {
            weight_map_.emplace(it.key(), it.value().get<std::string>());
        }
    }

    HostTensor read(const std::string &name) {
        const auto map_it = weight_map_.find(name);
        if (map_it == weight_map_.end()) {
            throw std::runtime_error("missing vision tensor: " + name);
        }
        const std::string &shard_name = map_it->second;
        json &header = headers_[shard_name];
        uint64_t data_base = 0;
        if (header.is_null()) {
            std::ifstream input(directory_ / shard_name, std::ios::binary);
            if (!input) {
                throw std::runtime_error("cannot open checkpoint shard: " +
                                         (directory_ / shard_name).string());
            }
            uint64_t header_bytes = 0;
            input.read(reinterpret_cast<char *>(&header_bytes), sizeof(header_bytes));
            if (!input || header_bytes == 0 || header_bytes > (1ULL << 30)) {
                throw std::runtime_error("invalid safetensors header in " + shard_name);
            }
            std::string encoded(static_cast<size_t>(header_bytes), '\0');
            input.read(encoded.data(), static_cast<std::streamsize>(header_bytes));
            if (!input) throw std::runtime_error("truncated safetensors header");
            header = json::parse(encoded);
            header_sizes_[shard_name] = header_bytes;
        }
        data_base = sizeof(uint64_t) + header_sizes_.at(shard_name);
        const json &entry = header.at(name);
        if (entry.at("dtype").get<std::string>() != "BF16") {
            throw std::runtime_error("native vision requires BF16 tensor: " + name);
        }
        HostTensor result;
        result.shape = entry.at("shape").get<std::vector<uint64_t>>();
        const uint64_t count = shape_count(result.shape);
        const auto offsets = entry.at("data_offsets").get<std::array<uint64_t, 2>>();
        if (offsets[1] < offsets[0] || offsets[1] - offsets[0] != count * 2) {
            throw std::runtime_error("invalid safetensors offsets for " + name);
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
            throw std::runtime_error("vision tensor is too large: " + name);
        }
        result.bf16.resize(static_cast<size_t>(count));
        std::ifstream input(directory_ / shard_name, std::ios::binary);
        input.seekg(static_cast<std::streamoff>(data_base + offsets[0]));
        input.read(reinterpret_cast<char *>(result.bf16.data()),
                   static_cast<std::streamsize>(count * 2));
        if (!input) throw std::runtime_error("truncated vision tensor: " + name);
        return result;
    }

private:
    std::filesystem::path directory_;
    std::unordered_map<std::string, std::string> weight_map_;
    std::unordered_map<std::string, json> headers_;
    std::unordered_map<std::string, uint64_t> header_sizes_;
};

struct WeightWithShape {
    std::unique_ptr<DeviceWeight> device;
    uint64_t bytes = 0;
};

WeightWithShape upload_weight(DeviceBackend &backend,
                              SafeTensorReader &reader,
                              const std::string &name,
                              uint64_t expected_rows,
                              uint64_t expected_cols,
                              const char *label) {
    HostTensor host = reader.read(name);
    const uint64_t count = shape_count(host.shape);
    if (count != expected_rows * expected_cols) {
        throw std::runtime_error("unexpected shape for " + name);
    }
    WeightWithShape result;
    result.bytes = count * sizeof(uint16_t);
    result.device = backend.weight_bf16(host.bf16.data(), expected_rows,
                                        expected_cols, label);
    return result;
}

} // namespace

struct GpuVisionFrontend::Impl {
    static constexpr uint32_t kHidden = 1152;
    static constexpr uint32_t kIntermediate = 4304;
    static constexpr uint32_t kHeads = 16;
    static constexpr uint32_t kHeadDim = 72;
    static constexpr uint32_t kPaddedHeadDim = 128;
    static constexpr uint32_t kPatchDim = 1536;
    static constexpr uint32_t kMerge = 2;
    static constexpr uint32_t kMergedHidden = 4608;
    static constexpr uint32_t kOutput = 5120;
    static constexpr uint32_t kLayers = 27;
    static constexpr uint32_t kPositionSide = 48;

    struct Layer {
        std::unique_ptr<DeviceWeight> norm1_weight;
        std::unique_ptr<DeviceWeight> norm1_bias;
        std::unique_ptr<DeviceWeight> qkv_weight;
        std::unique_ptr<DeviceWeight> qkv_bias;
        std::unique_ptr<DeviceWeight> proj_weight;
        std::unique_ptr<DeviceWeight> proj_bias;
        std::unique_ptr<DeviceWeight> norm2_weight;
        std::unique_ptr<DeviceWeight> norm2_bias;
        std::unique_ptr<DeviceWeight> fc1_weight;
        std::unique_ptr<DeviceWeight> fc1_bias;
        std::unique_ptr<DeviceWeight> fc2_weight;
        std::unique_ptr<DeviceWeight> fc2_bias;
    };

    struct CacheEntry {
        std::vector<CpuVisionImage> images;
        CpuVisionEncoding encoding;
        uint64_t bytes = 0;
    };

    DeviceBackend &backend;
    CpuVisionFrontend preprocessor;
    std::unique_ptr<DeviceWeight> patch_weight;
    std::unique_ptr<DeviceWeight> patch_bias;
    std::unique_ptr<DeviceWeight> position_table;
    std::array<Layer, kLayers> layers;
    std::unique_ptr<DeviceWeight> merger_norm_weight;
    std::unique_ptr<DeviceWeight> merger_norm_bias;
    std::unique_ptr<DeviceWeight> merger_fc1_weight;
    std::unique_ptr<DeviceWeight> merger_fc1_bias;
    std::unique_ptr<DeviceWeight> merger_fc2_weight;
    std::unique_ptr<DeviceWeight> merger_fc2_bias;
    uint64_t uploaded_bytes = 0;
    std::mutex encode_mutex;
    uint64_t cache_limit_bytes = 512ULL << 20;
    uint64_t cache_bytes = 0;
    std::vector<CacheEntry> cache;

    Impl(DeviceBackend &backend_arg,
         const std::string &model_directory,
         const std::string &python_executable,
         const std::string &worker_script,
         uint32_t preprocess_threads)
        : backend(backend_arg),
          preprocessor(model_directory, python_executable, worker_script,
                       preprocess_threads, true) {
        if (!backend.supports_bf16_activations()) {
            throw std::runtime_error("native CUDA vision requires BF16 activations");
        }
        if (const char *value = std::getenv("QW3_VISION_GPU_CACHE_MIB")) {
            char *end = nullptr;
            errno = 0;
            const unsigned long long mib = std::strtoull(value, &end, 10);
            if (errno == 0 && end != value && *end == '\0' &&
                mib <= std::numeric_limits<uint64_t>::max() / (1ULL << 20)) {
                cache_limit_bytes = static_cast<uint64_t>(mib) << 20;
            }
        }
        SafeTensorReader reader(model_directory);
        auto take = [&](const std::string &name, uint64_t rows, uint64_t cols,
                        const char *label) {
            WeightWithShape loaded = upload_weight(
                backend, reader, name, rows, cols, label);
            uploaded_bytes += loaded.bytes;
            return std::move(loaded.device);
        };
        const std::string prefix = "model.visual.";
        patch_weight = take(prefix + "patch_embed.proj.weight", kHidden,
                            kPatchDim, "vision_patch_w");
        patch_bias = take(prefix + "patch_embed.proj.bias", 1, kHidden,
                          "vision_patch_b");
        position_table = take(prefix + "pos_embed.weight", 2304, kHidden,
                              "vision_position_table");

        for (uint32_t i = 0; i < kLayers; ++i) {
            Layer &layer = layers[i];
            const std::string base = prefix + "blocks." + std::to_string(i) + ".";
            layer.norm1_weight = take(base + "norm1.weight", 1, kHidden,
                                      "vision_ln1_w");
            layer.norm1_bias = take(base + "norm1.bias", 1, kHidden,
                                    "vision_ln1_b");
            layer.qkv_weight = take(base + "attn.qkv.weight", 3 * kHidden,
                                    kHidden, "vision_qkv_w");
            layer.qkv_bias = take(base + "attn.qkv.bias", 1, 3 * kHidden,
                                  "vision_qkv_b");
            layer.proj_weight = take(base + "attn.proj.weight", kHidden,
                                     kHidden, "vision_proj_w");
            layer.proj_bias = take(base + "attn.proj.bias", 1, kHidden,
                                   "vision_proj_b");
            layer.norm2_weight = take(base + "norm2.weight", 1, kHidden,
                                      "vision_ln2_w");
            layer.norm2_bias = take(base + "norm2.bias", 1, kHidden,
                                    "vision_ln2_b");
            layer.fc1_weight = take(base + "mlp.linear_fc1.weight",
                                    kIntermediate, kHidden, "vision_fc1_w");
            layer.fc1_bias = take(base + "mlp.linear_fc1.bias", 1,
                                  kIntermediate, "vision_fc1_b");
            layer.fc2_weight = take(base + "mlp.linear_fc2.weight", kHidden,
                                    kIntermediate, "vision_fc2_w");
            layer.fc2_bias = take(base + "mlp.linear_fc2.bias", 1, kHidden,
                                  "vision_fc2_b");
        }
        merger_norm_weight = take(prefix + "merger.norm.weight", 1, kHidden,
                                  "vision_merge_ln_w");
        merger_norm_bias = take(prefix + "merger.norm.bias", 1, kHidden,
                                "vision_merge_ln_b");
        merger_fc1_weight = take(prefix + "merger.linear_fc1.weight",
                                 kMergedHidden, kMergedHidden,
                                 "vision_merge_fc1_w");
        merger_fc1_bias = take(prefix + "merger.linear_fc1.bias", 1,
                               kMergedHidden, "vision_merge_fc1_b");
        merger_fc2_weight = take(prefix + "merger.linear_fc2.weight", kOutput,
                                 kMergedHidden, "vision_merge_fc2_w");
        merger_fc2_bias = take(prefix + "merger.linear_fc2.bias", 1, kOutput,
                               "vision_merge_fc2_b");
        require_ok(backend.synchronize(), "native vision weight upload");
    }

    void build_position_metadata(
            const std::vector<CpuVisionEncoding::Grid> &grids,
            uint64_t patch_count,
            std::vector<uint32_t> &indices,
            std::vector<float> &weights) const {
        indices.resize(static_cast<size_t>(patch_count) * 4);
        weights.resize(static_cast<size_t>(patch_count) * 4);
        uint64_t token = 0;
        for (const auto &grid : grids) {
            if (grid.height % kMerge != 0 || grid.width % kMerge != 0) {
                throw std::runtime_error("vision grid is not divisible by merge size 2");
            }
            const uint32_t bh_count = grid.height / kMerge;
            const uint32_t bw_count = grid.width / kMerge;
            for (uint32_t t = 0; t < grid.temporal; ++t) {
                (void)t;
                for (uint32_t bh = 0; bh < bh_count; ++bh) {
                    for (uint32_t bw = 0; bw < bw_count; ++bw) {
                        for (uint32_t ih = 0; ih < kMerge; ++ih) {
                            const uint32_t h = bh * kMerge + ih;
                            const float hf = grid.height > 1
                                ? static_cast<float>(h) * (kPositionSide - 1) /
                                      static_cast<float>(grid.height - 1)
                                : 0.0f;
                            const uint32_t h0 = static_cast<uint32_t>(hf);
                            const uint32_t h1 = std::min(h0 + 1, kPositionSide - 1);
                            const float hw = hf - h0;
                            for (uint32_t iw = 0; iw < kMerge; ++iw) {
                                const uint32_t w = bw * kMerge + iw;
                                const float wf = grid.width > 1
                                    ? static_cast<float>(w) * (kPositionSide - 1) /
                                          static_cast<float>(grid.width - 1)
                                    : 0.0f;
                                const uint32_t w0 = static_cast<uint32_t>(wf);
                                const uint32_t w1 = std::min(w0 + 1, kPositionSide - 1);
                                const float ww = wf - w0;
                                const std::array<uint32_t, 4> index = {
                                    h0 * kPositionSide + w0,
                                    h0 * kPositionSide + w1,
                                    h1 * kPositionSide + w0,
                                    h1 * kPositionSide + w1};
                                const std::array<float, 4> weight = {
                                    (1.0f - hw) * (1.0f - ww),
                                    (1.0f - hw) * ww,
                                    hw * (1.0f - ww), hw * ww};
                                for (uint32_t c = 0; c < 4; ++c) {
                                    indices[token * 4 + c] = index[c];
                                    weights[token * 4 + c] = weight[c];
                                }
                                ++token;
                            }
                        }
                    }
                }
            }
        }
        if (token != patch_count) {
            throw std::runtime_error("vision position row count mismatch");
        }
    }

    void build_rope_and_segments(
            const std::vector<CpuVisionEncoding::Grid> &grids,
            uint64_t patch_count,
            std::vector<uint16_t> &cos_values,
            std::vector<uint16_t> &sin_values,
            std::vector<uint32_t> &cu_seqlens) const {
        cos_values.resize(static_cast<size_t>(patch_count) * kHeadDim);
        sin_values.resize(static_cast<size_t>(patch_count) * kHeadDim);
        cu_seqlens.clear();
        cu_seqlens.push_back(0);
        uint64_t token = 0;
        for (const auto &grid : grids) {
            const uint32_t bh_count = grid.height / kMerge;
            const uint32_t bw_count = grid.width / kMerge;
            for (uint32_t t = 0; t < grid.temporal; ++t) {
                for (uint32_t bh = 0; bh < bh_count; ++bh) {
                    for (uint32_t bw = 0; bw < bw_count; ++bw) {
                        for (uint32_t ih = 0; ih < kMerge; ++ih) {
                            const uint32_t h = bh * kMerge + ih;
                            for (uint32_t iw = 0; iw < kMerge; ++iw) {
                                const uint32_t w = bw * kMerge + iw;
                                for (uint32_t d = 0; d < kHeadDim; ++d) {
                                    const uint32_t base = d % (kHeadDim / 2);
                                    const uint32_t freq_index = base % 18;
                                    const uint32_t position = base < 18 ? h : w;
                                    // Transformers moves the rotary module to
                                    // BF16 before applying it: both inv_freq and
                                    // position*inv_freq are rounded to BF16
                                    // before cos/sin. Matching those two
                                    // rounding points matters for tall/large
                                    // images whose position IDs exceed 100.
                                    const float inv_freq = round_to_bf16(std::pow(
                                        10000.0f,
                                        -static_cast<float>(2 * freq_index) /
                                            36.0f));
                                    const float angle = round_to_bf16(
                                        static_cast<float>(position) * inv_freq);
                                    cos_values[token * kHeadDim + d] =
                                        float_to_bf16(std::cos(angle));
                                    sin_values[token * kHeadDim + d] =
                                        float_to_bf16(std::sin(angle));
                                }
                                ++token;
                            }
                        }
                    }
                }
                const uint64_t next = static_cast<uint64_t>(cu_seqlens.back()) +
                                      grid.height * grid.width;
                if (next > std::numeric_limits<uint32_t>::max()) {
                    throw std::runtime_error("vision sequence exceeds uint32 range");
                }
                cu_seqlens.push_back(static_cast<uint32_t>(next));
            }
        }
        if (token != patch_count || cu_seqlens.back() != patch_count) {
            throw std::runtime_error("vision RoPE/segment row count mismatch");
        }
    }

    CpuVisionEncoding encode(const std::vector<CpuVisionImage> &images) {
        std::lock_guard<std::mutex> lock(encode_mutex);
        const auto same_images = [&](const std::vector<CpuVisionImage> &cached) {
            if (cached.size() != images.size()) return false;
            for (size_t i = 0; i < images.size(); ++i) {
                if (cached[i].media_type != images[i].media_type ||
                    cached[i].base64_data != images[i].base64_data) {
                    return false;
                }
            }
            return true;
        };
        for (size_t i = 0; i < cache.size(); ++i) {
            if (!same_images(cache[i].images)) continue;
            CacheEntry hit = std::move(cache[i]);
            cache.erase(cache.begin() + static_cast<std::ptrdiff_t>(i));
            cache.push_back(std::move(hit));
            CpuVisionEncoding result = cache.back().encoding;
            result.cache_hit = true;
            return result;
        }
        CpuVisionPatches input = preprocessor.preprocess(images);
        if (input.patch_dim != kPatchDim) {
            throw std::runtime_error("native vision patch width must be 1536");
        }
        const uint64_t patches = input.patches.size() / kPatchDim;
        if (patches == 0 || patches % (kMerge * kMerge) != 0 ||
            patches > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("invalid native vision patch count");
        }
        const uint32_t tokens = static_cast<uint32_t>(patches);
        const uint32_t output_rows = tokens / 4;

        std::vector<uint32_t> position_indices;
        std::vector<float> position_weights;
        build_position_metadata(input.grids, patches, position_indices,
                                position_weights);
        std::vector<uint16_t> cos_values;
        std::vector<uint16_t> sin_values;
        std::vector<uint32_t> cu_seqlens;
        build_rope_and_segments(input.grids, patches, cos_values, sin_values,
                                cu_seqlens);

        auto patches_d = backend.tensor_bf16(patches * kPatchDim, "vision_patches");
        auto position_d = backend.tensor_bf16(patches * kHidden, "vision_position");
        auto cos_d = backend.tensor_bf16(patches * kHeadDim, "vision_cos");
        auto sin_d = backend.tensor_bf16(patches * kHeadDim, "vision_sin");
        require_ok(backend.copy_bytes_from_host(
            *patches_d, 0, input.patches.data(), input.patches.size() * 2),
            "upload vision patches");
        require_ok(backend.vision_bf16_position_interpolate(
            *position_d, *position_table, position_indices.data(),
            position_weights.data(), tokens, kHidden),
            "vision position interpolation");
        require_ok(backend.copy_bytes_from_host(
            *cos_d, 0, cos_values.data(), cos_values.size() * 2),
            "upload vision cosine");
        require_ok(backend.copy_bytes_from_host(
            *sin_d, 0, sin_values.data(), sin_values.size() * 2),
            "upload vision sine");

        auto hidden = backend.tensor_bf16(patches * kHidden, "vision_hidden");
        auto norm = backend.tensor_bf16(patches * kHidden, "vision_norm");
        auto qkv = backend.tensor_bf16(patches * 3 * kHidden, "vision_qkv");
        const uint64_t padded_elems = static_cast<uint64_t>(patches) *
                                      kHeads * kPaddedHeadDim;
        auto q = backend.tensor_bf16(padded_elems, "vision_q");
        auto k = backend.tensor_bf16(padded_elems, "vision_k");
        auto v = backend.tensor_bf16(padded_elems, "vision_v");
        auto attention_padded = backend.tensor_bf16(
            padded_elems, "vision_attention_padded");
        auto attention = backend.tensor_bf16(patches * kHidden,
                                             "vision_attention");
        auto intermediate = backend.tensor_bf16(
            patches * kIntermediate, "vision_intermediate");
        auto residual_out = backend.tensor_bf16(patches * kHidden,
                                                "vision_residual_out");

        require_ok(backend.vision_bf16_linear(
            *hidden, *patch_weight, *patches_d, patch_bias.get(), nullptr,
            tokens, kPatchDim, kHidden, false), "vision patch projection");
        require_ok(backend.add_n(*hidden, *hidden, *position_d,
                                 patches * kHidden),
                   "vision add position embeddings");

        for (const Layer &layer : layers) {
            require_ok(backend.vision_bf16_layer_norm(
                *norm, *hidden, *layer.norm1_weight, *layer.norm1_bias,
                tokens, kHidden, 1e-6f), "vision attention layer norm");
            require_ok(backend.vision_bf16_linear(
                *qkv, *layer.qkv_weight, *norm, layer.qkv_bias.get(), nullptr,
                tokens, kHidden, 3 * kHidden, false), "vision QKV projection");
            require_ok(backend.vision_bf16_qkv_rope_pad(
                *q, *k, *v, *qkv, *cos_d, *sin_d, tokens, kHeads,
                kHeadDim, kPaddedHeadDim), "vision QKV RoPE");
            require_ok(backend.vision_bf16_attention(
                *attention_padded, *q, *k, *v, cu_seqlens.data(),
                static_cast<uint32_t>(cu_seqlens.size() - 1), kHeads,
                kPaddedHeadDim, 1.0f / std::sqrt(static_cast<float>(kHeadDim))),
                "vision attention");
            require_ok(backend.vision_bf16_compact_heads(
                *attention, *attention_padded, tokens, kHeads, kHeadDim,
                kPaddedHeadDim), "vision compact attention heads");
            require_ok(backend.vision_bf16_linear(
                *residual_out, *layer.proj_weight, *attention,
                layer.proj_bias.get(), hidden.get(), tokens, kHidden,
                kHidden, false), "vision attention output projection");
            hidden.swap(residual_out);

            require_ok(backend.vision_bf16_layer_norm(
                *norm, *hidden, *layer.norm2_weight, *layer.norm2_bias,
                tokens, kHidden, 1e-6f), "vision MLP layer norm");
            require_ok(backend.vision_bf16_linear(
                *intermediate, *layer.fc1_weight, *norm,
                layer.fc1_bias.get(), nullptr, tokens, kHidden,
                kIntermediate, true), "vision MLP fc1");
            require_ok(backend.vision_bf16_linear(
                *residual_out, *layer.fc2_weight, *intermediate,
                layer.fc2_bias.get(), hidden.get(), tokens, kIntermediate,
                kHidden, false), "vision MLP fc2");
            hidden.swap(residual_out);
        }

        require_ok(backend.vision_bf16_layer_norm(
            *norm, *hidden, *merger_norm_weight, *merger_norm_bias,
            tokens, kHidden, 1e-6f), "vision merger layer norm");
        auto merged_hidden = backend.tensor_bf16(
            static_cast<uint64_t>(output_rows) * kMergedHidden,
            "vision_merged_hidden");
        require_ok(backend.vision_bf16_linear(
            *merged_hidden, *merger_fc1_weight, *norm,
            merger_fc1_bias.get(), nullptr, output_rows, kMergedHidden,
            kMergedHidden, false, true), "vision merger fc1");
        auto final_unique = backend.tensor_bf16(
            static_cast<uint64_t>(output_rows) * kOutput,
            "vision_projected_embeddings");
        require_ok(backend.vision_bf16_linear(
            *final_unique, *merger_fc2_weight, *merged_hidden,
            merger_fc2_bias.get(), nullptr, output_rows, kMergedHidden,
            kOutput, false), "vision merger fc2");
        require_ok(backend.synchronize(), "native vision forward");

        std::shared_ptr<DeviceTensor> final_device(std::move(final_unique));
        CpuVisionEncoding result;
        result.embedding_dim = kOutput;
        result.grids = std::move(input.grids);
        result.storage = std::make_shared<DeviceInputEmbeddingStorage>(
            std::move(final_device), output_rows, kOutput);
        const uint64_t result_bytes = static_cast<uint64_t>(output_rows) *
                                      kOutput * sizeof(uint16_t);
        if (cache_limit_bytes > 0 && result_bytes <= cache_limit_bytes) {
            while (!cache.empty() &&
                   cache_bytes > cache_limit_bytes - result_bytes) {
                cache_bytes -= cache.front().bytes;
                cache.erase(cache.begin());
            }
            CacheEntry entry;
            entry.images = images;
            entry.encoding = result;
            entry.bytes = result_bytes;
            cache.push_back(std::move(entry));
            cache_bytes += result_bytes;
        }
        return result;
    }
};

GpuVisionFrontend::GpuVisionFrontend(DeviceBackend &backend,
                                     std::string model_directory,
                                     std::string python_executable,
                                     std::string worker_script,
                                     uint32_t preprocess_threads)
    : impl_(std::make_unique<Impl>(backend, model_directory,
                                   python_executable, worker_script,
                                   preprocess_threads)) {}

GpuVisionFrontend::~GpuVisionFrontend() = default;

CpuVisionEncoding GpuVisionFrontend::encode(
        const std::vector<CpuVisionImage> &images) {
    return impl_->encode(images);
}

uint64_t GpuVisionFrontend::weight_bytes() const {
    return impl_->uploaded_bytes;
}

} // namespace qw3::detail
