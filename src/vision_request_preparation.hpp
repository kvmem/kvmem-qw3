#pragma once
#include "vision_cpu_frontend.hpp"
#include "prepared_message_cache.hpp"
#include "multimodal_prefix_cache.hpp"
#include "qw3/tokenizer.hpp"
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

namespace qw3::detail {
using json = nlohmann::json;
struct PreparedVisionRequest {
    detail::CpuVisionEncoding encoded;
    std::vector<std::pair<uint32_t, uint32_t>> token_spans;
    std::vector<uint64_t> image_fingerprints;
    uint64_t fingerprint = 0;
    std::vector<VisionImageSourcePtr> sources;
    size_t prefix_messages_reused = 0, prefix_images_reused = 0;
    size_t mapping_tokens_reused = 0;

    bool active() const { return !encoded.grids.empty(); }
};

using VisionEncoder = std::function<
    VisionEncoding(const std::vector<VisionImage> &)>;
using PreparedVisionEncoder = std::function<
    VisionEncoding(const std::vector<VisionImageSourcePtr> &)>;

inline bool same_vision_grid(const VisionEncoding::Grid &a, const VisionEncoding::Grid &b) {
    return a.temporal == b.temporal && a.height == b.height &&
           a.width == b.width && a.rows == b.rows;
}

inline bool parse_data_image_url(const std::string &url,
                          detail::CpuVisionImage &image,
                          std::string &error) {
    if (url.rfind("data:", 0) != 0) {
        error = "CPU vision V1 accepts base64 data: image URLs only";
        return false;
    }
    const size_t comma = url.find(',');
    const size_t base64 = url.find(";base64");
    if (comma == std::string::npos || base64 == std::string::npos ||
        base64 > comma) {
        error = "image URL must be a base64 data URI";
        return false;
    }
    image.media_type = url.substr(5, base64 - 5);
    image.base64_data = url.substr(comma + 1);
    if (image.media_type.rfind("image/", 0) != 0 ||
        image.base64_data.empty()) {
        error = "invalid base64 image data URI";
        return false;
    }
    return true;
}

inline bool openai_image_block(const json &block,
                        detail::CpuVisionImage &image,
                        std::string &error) {
    if (!block.is_object() || block.value("type", "") != "image_url") {
        return false;
    }
    if (!block.contains("image_url")) {
        error = "image_url block requires image_url";
        return false;
    }
    std::string url;
    if (block["image_url"].is_string()) {
        url = block["image_url"].get<std::string>();
    } else if (block["image_url"].is_object() &&
               block["image_url"].contains("url") &&
               block["image_url"]["url"].is_string()) {
        url = block["image_url"]["url"].get<std::string>();
    } else {
        error = "image_url must be a string or an object with string url";
        return false;
    }
    return parse_data_image_url(url, image, error);
}

inline bool prepare_vision_messages(json &messages,
                             const VisionEncoder &encoder,
                             PreparedVisionRequest &prepared,
                             std::string &error) {
    std::vector<detail::CpuVisionImage> images;
    for (const json &message : messages) {
        if (!message.is_object() || !message.contains("content") ||
            !message["content"].is_array()) {
            continue;
        }
        for (const json &block : message["content"]) {
            if (!block.is_object() || block.value("type", "") != "image_url") {
                continue;
            }
            detail::CpuVisionImage image;
            if (!openai_image_block(block, image, error)) return false;
            images.push_back(std::move(image));
        }
    }
    if (images.empty()) return true;
    if (!encoder) {
        error = "image input requires --vision-model DIR";
        return false;
    }
    try {
        prepared.encoded = encoder(images);
    } catch (const std::exception &e) {
        error = e.what();
        return false;
    }

    // Hash the ordered source payloads rather than the projected embeddings:
    // this is deterministic across worker/kernel implementations and avoids a
    // second pass over the much larger projected tensor. It is cache identity,
    // not an authentication primitive.
    auto mix = [](uint64_t &fingerprint, std::string_view value) {
        for (unsigned char byte : value) {
            fingerprint ^= static_cast<uint64_t>(byte);
            fingerprint *= 1099511628211ULL;
        }
        fingerprint ^= 0xffu;
        fingerprint *= 1099511628211ULL;
    };
    uint64_t fingerprint = 1469598103934665603ULL;
    prepared.image_fingerprints.clear();
    prepared.image_fingerprints.reserve(images.size());
    for (const auto &image : images) {
        mix(fingerprint, image.media_type);
        mix(fingerprint, image.base64_data);
        uint64_t image_fingerprint = 1469598103934665603ULL;
        mix(image_fingerprint, image.media_type);
        mix(image_fingerprint, image.base64_data);
        prepared.image_fingerprints.push_back(
            image_fingerprint != 0 ? image_fingerprint : 1);
    }
    prepared.fingerprint = fingerprint != 0 ? fingerprint : 1;

    size_t image_index = 0;
    for (json &message : messages) {
        if (!message.is_object() || !message.contains("content") ||
            !message["content"].is_array()) {
            continue;
        }
        for (json &block : message["content"]) {
            if (!block.is_object() || block.value("type", "") != "image_url") {
                continue;
            }
            if (image_index >= prepared.encoded.grids.size()) {
                error = "vision worker image count does not match the request";
                return false;
            }
            const uint32_t rows = prepared.encoded.grids[image_index].rows;
            std::string placeholder = "<|vision_start|>";
            placeholder.reserve(placeholder.size() +
                                static_cast<size_t>(rows) * 13 + 16);
            for (uint32_t row = 0; row < rows; ++row) {
                placeholder += "<|image_pad|>";
            }
            placeholder += "<|vision_end|>";
            block = json{{"type", "text"}, {"text", std::move(placeholder)}};
            ++image_index;
        }
    }
    if (image_index != prepared.encoded.grids.size()) {
        error = "vision request mutation lost an image";
        return false;
    }
    return true;
}

inline bool prepare_vision_messages_incremental(
        json &suffix, const PreparedVisionEncoder &encoder,
        const std::shared_ptr<const PreparedMessageCache::Entry> &previous,
        size_t prefix_count, PreparedMessageCache::Entry &candidate,
        PreparedVisionRequest &prepared, std::string &error) {
    if (!suffix.is_array() || (prefix_count &&
        (!previous || prefix_count > previous->messages.size()))) {
        error = "invalid prepared message prefix";
        return false;
    }
    candidate.messages.clear();
    if (previous) candidate.messages.assign(previous->messages.begin(),
                                             previous->messages.begin() + prefix_count);
    prepared.prefix_messages_reused = prefix_count;
    for (const auto &message : candidate.messages) {
        prepared.sources.insert(prepared.sources.end(), message->images.begin(), message->images.end());
    }
    prepared.prefix_images_reused = prepared.sources.size();
    uint64_t state = candidate.messages.empty() ? VisionImageSource::fingerprint_seed
                                               : candidate.messages.back()->image_hash_state;
    std::vector<std::shared_ptr<PreparedMessage>> fresh;
    fresh.reserve(suffix.size());
    for (auto &message : suffix) {
        auto record = std::make_shared<PreparedMessage>();
        record->raw = std::move(message);
        record->rendered = record->raw;
        if (record->raw.is_object() && record->raw.contains("content") &&
            record->raw["content"].is_array()) {
            for (const auto &block : record->raw["content"]) {
                if (!block.is_object() || block.value("type", "") != "image_url") continue;
                VisionImage image;
                if (!openai_image_block(block, image, error)) return false;
                auto source = std::make_shared<const VisionImageSource>(std::move(image));
                state = source->append_fingerprint(state);
                record->images.push_back(source);
                prepared.sources.push_back(std::move(source));
            }
        }
        record->image_hash_state = state;
        fresh.push_back(std::move(record));
    }
    if (!prepared.sources.empty()) {
        if (!encoder) { error = "image input requires --vision-model DIR"; return false; }
        try { prepared.encoded = encoder(prepared.sources); }
        catch (const std::exception &e) { error = e.what(); return false; }
        if (prepared.encoded.grids.size() != prepared.sources.size()) {
            error = "vision worker image count does not match prepared sources";
            return false;
        }
        for (const auto &image : prepared.sources) prepared.image_fingerprints.push_back(image->fingerprint());
        prepared.fingerprint = state ? state : 1;
    }
    size_t index = 0;
    for (const auto &message : candidate.messages) {
        for (const auto &grid : message->grids) {
            if (index >= prepared.encoded.grids.size() ||
                !same_vision_grid(grid, prepared.encoded.grids[index++])) {
                error = "cached image grid changed within the same server";
                return false;
            }
        }
    }
    for (auto &message : fresh) {
        if (message->rendered.is_object() && message->rendered.contains("content") &&
            message->rendered["content"].is_array()) {
            for (auto &block : message->rendered["content"]) {
                if (!block.is_object() || block.value("type", "") != "image_url") continue;
                const auto grid = prepared.encoded.grids.at(index++);
                std::string placeholder = "<|vision_start|>";
                placeholder.reserve(placeholder.size() + static_cast<size_t>(grid.rows) * 13 + 16);
                for (uint32_t row = 0; row < grid.rows; ++row) placeholder += "<|image_pad|>";
                placeholder += "<|vision_end|>";
                block = json{{"type", "text"}, {"text", std::move(placeholder)}};
                message->grids.push_back(grid);
            }
        }
        message->account();
        candidate.messages.push_back(std::move(message));
    }
    json rendered = json::array();
    auto &array = rendered.get_ref<json::array_t &>();
    array.reserve(candidate.messages.size());
    for (const auto &message : candidate.messages) array.push_back(message->rendered);
    suffix = std::move(rendered);
    return true;
}

inline bool finalize_vision_tokens(
        std::vector<int32_t> &tokens,
        const QwenTokenizer &tokenizer,
        PreparedVisionRequest &prepared,
        GenerationOptions &generation,
        std::string &error,
        const std::shared_ptr<const VisionTokenMapping> &previous = {},
        std::shared_ptr<const VisionTokenMapping> *saved = nullptr) {
    if (!prepared.active()) return true;
    const int32_t image_pad = tokenizer.token_id("<|image_pad|>");
    const int32_t vision_start = tokenizer.token_id("<|vision_start|>");
    const int32_t vision_end = tokenizer.token_id("<|vision_end|>");
    if (image_pad < 0 || vision_start < 0 || vision_end < 0) {
        error = "model tokenizer is missing Qwen vision special tokens";
        return false;
    }
    if (prepared.encoded.embedding_dim == 0 ||
        prepared.encoded.embedding_dim > std::numeric_limits<uint32_t>::max()) {
        error = "vision worker returned an invalid embedding width";
        return false;
    }

    std::shared_ptr<VisionTokenMapping> mapping;
    if (saved) {
        mapping = std::make_shared<VisionTokenMapping>();
        mapping->raw_tokens = tokens;
        mapping->image_pad = image_pad;
        mapping->vision_start = vision_start;
        mapping->vision_end = vision_end;
        mapping->dim = prepared.encoded.embedding_dim;
        mapping->images = prepared.sources;
        mapping->grids = prepared.encoded.grids;
    }
    size_t stable = 0, prefix_grids = 0;
    if (previous && previous->image_pad == image_pad &&
        previous->vision_start == vision_start && previous->vision_end == vision_end &&
        previous->dim == prepared.encoded.embedding_dim &&
        previous->images.size() == previous->spans.size() &&
        prepared.sources.size() == prepared.encoded.grids.size()) {
        const size_t limit = std::min(tokens.size(), previous->raw_tokens.size());
        while (stable < limit && tokens[stable] == previous->raw_tokens[stable]) ++stable;
        size_t images_equal = 0;
        for (; images_equal < std::min(prepared.sources.size(), previous->images.size()); ++images_equal) {
            const auto &a = prepared.sources[images_equal];
            const auto &b = previous->images[images_equal];
            if (!same_vision_grid(prepared.encoded.grids[images_equal], previous->grids[images_equal]) ||
                (a != b && (a->fingerprint() != b->fingerprint() ||
                 a->image().media_type != b->image().media_type ||
                 a->image().base64_data != b->image().base64_data))) break;
        }
        if (images_equal < previous->spans.size()) stable = std::min<size_t>(stable, previous->spans[images_equal].first);
        for (const auto &span : previous->spans) {
            // Restart at vision_start when a rewrite intersects an image.
            if (stable > span.first && stable < span.second) { stable = span.first; break; }
            if (span.second > stable) break;
            ++prefix_grids;
        }
    }
    prepared.mapping_tokens_reused = stable;
    std::vector<uint8_t> modality(tokens.size(), 0);
    size_t embedding_row = 0;
    size_t search = 0;
    generation.input_embedding_overrides.clear();
    generation.input_embedding_fingerprint = prepared.fingerprint;
    generation.input_mrope_positions.assign(tokens.size(), {0, 0, 0});
    prepared.token_spans.clear();

    if (prepared.image_fingerprints.size() != prepared.encoded.grids.size()) {
        error = "vision source identity count does not match encoded images";
        return false;
    }
    if (stable) {
        std::copy_n(previous->mapped_tokens.begin(), stable, tokens.begin());
        std::copy_n(previous->positions.begin(), stable, generation.input_mrope_positions.begin());
        prepared.token_spans.assign(previous->spans.begin(), previous->spans.begin() + prefix_grids);
        generation.input_embedding_overrides.reserve(prepared.encoded.storage
            ? prepared.encoded.storage->rows() : prepared.encoded.embeddings.size() / prepared.encoded.embedding_dim);
        for (size_t i = 0; i < prefix_grids; ++i) {
            const auto span = previous->spans[i];
            for (size_t pos = span.first + 1; pos + 1 < span.second; ++pos) {
                GenerationOptions::InputEmbeddingOverride row;
                row.token_id = static_cast<uint32_t>(tokens[pos]);
                row.source_token_id = static_cast<uint32_t>(image_pad);
                row.cache_identity = previous->row_identities[embedding_row];
                row.position = previous->positions[pos];
                if (prepared.encoded.storage) {
                    row.storage = prepared.encoded.storage;
                    row.embedding_row = static_cast<uint32_t>(embedding_row);
                } else {
                    const size_t offset = embedding_row * prepared.encoded.embedding_dim;
                    row.embedding.assign(prepared.encoded.embeddings.begin() + offset,
                                         prepared.encoded.embeddings.begin() + offset + prepared.encoded.embedding_dim);
                }
                generation.input_embedding_overrides.push_back(std::move(row));
                ++embedding_row;
            }
        }
        search = stable;
    }
    for (size_t grid_index = prefix_grids;
         grid_index < prepared.encoded.grids.size(); ++grid_index) {
        const auto &grid = prepared.encoded.grids[grid_index];
        while (search + 1 < tokens.size() &&
               !(tokens[search] == vision_start &&
                 tokens[search + 1] == image_pad)) {
            ++search;
        }
        if (search + 1 >= tokens.size()) {
            error = "rendered prompt has fewer vision spans than encoded images";
            return false;
        }
        const size_t span_begin = search;
        const size_t begin = ++search;
        while (search < tokens.size() && tokens[search] == image_pad) {
            ++search;
        }
        const size_t end = search;
        if (end - begin != grid.rows) {
            error = "rendered image token run does not match the processed image grid";
            return false;
        }
        if (end >= tokens.size() || tokens[end] != vision_end) {
            error = "rendered image token run is missing vision_end";
            return false;
        }
        prepared.token_spans.emplace_back(
            static_cast<uint32_t>(span_begin),
            static_cast<uint32_t>(end + 1));
        for (size_t pos = begin; pos < end; ++pos) {
            modality[pos] = 1;
            if (embedding_row >= 0x7fffffffu) {
                error = "too many visual embedding rows";
                return false;
            }
            const uint32_t virtual_id =
                0x80000000u | static_cast<uint32_t>(embedding_row + 1);
            GenerationOptions::InputEmbeddingOverride override;
            override.token_id = virtual_id;
            override.source_token_id = static_cast<uint32_t>(image_pad);
            override.cache_identity =
                detail::multimodal_row_cache_identity(
                    prepared.image_fingerprints[grid_index],
                    static_cast<uint32_t>(pos - begin));
            if (prepared.encoded.storage) {
                override.storage = prepared.encoded.storage;
                override.embedding_row = static_cast<uint32_t>(embedding_row);
            } else {
                const size_t offset =
                    embedding_row * prepared.encoded.embedding_dim;
                override.embedding.assign(
                    prepared.encoded.embeddings.begin() +
                        static_cast<std::ptrdiff_t>(offset),
                    prepared.encoded.embeddings.begin() +
                        static_cast<std::ptrdiff_t>(offset +
                                                    prepared.encoded.embedding_dim));
            }
            generation.input_embedding_overrides.push_back(std::move(override));
            tokens[pos] = static_cast<int32_t>(virtual_id);
            ++embedding_row;
        }
    }
    if (prepared.encoded.storage) {
        if (prepared.encoded.storage->rows() != embedding_row ||
            prepared.encoded.storage->dim() != prepared.encoded.embedding_dim) {
            error = "device visual embeddings do not match prompt mapping";
            return false;
        }
    } else {
        if (embedding_row * prepared.encoded.embedding_dim !=
            prepared.encoded.embeddings.size()) {
            error = "unused visual embeddings remain after prompt mapping";
            return false;
        }
    }

    uint32_t current = stable ? previous->positions[stable - 1][0] + 1 : 0;
    size_t image_index = prefix_grids;
    for (size_t pos = stable; pos < tokens.size();) {
        if (modality[pos] == 0) {
            generation.input_mrope_positions[pos] = {current, current, current};
            ++current;
            ++pos;
            continue;
        }
        if (image_index >= prepared.encoded.grids.size()) {
            error = "image modality run count exceeds image grids";
            return false;
        }
        const auto &grid = prepared.encoded.grids[image_index++];
        if (grid.height % 2 != 0 || grid.width % 2 != 0) {
            error = "Qwen vision grid is not divisible by spatial merge size 2";
            return false;
        }
        const uint32_t llm_h = grid.height / 2;
        const uint32_t llm_w = grid.width / 2;
        const uint32_t plane = llm_h * llm_w;
        for (uint32_t row = 0; row < grid.rows; ++row, ++pos) {
            const uint32_t t = plane > 0 ? row / plane : 0;
            const uint32_t rem = plane > 0 ? row % plane : 0;
            const uint32_t h = llm_w > 0 ? rem / llm_w : 0;
            const uint32_t w = llm_w > 0 ? rem % llm_w : 0;
            const std::array<uint32_t, 3> mrope = {
                current + t, current + h, current + w};
            generation.input_mrope_positions[pos] = mrope;
            generation.input_embedding_overrides[
                static_cast<size_t>(tokens[pos] & 0x7fffffffu) - 1].position =
                    mrope;
        }
        current += std::max({grid.temporal, llm_h, llm_w});
    }
    if (mapping) {
        mapping->mapped_tokens = tokens;
        mapping->positions = generation.input_mrope_positions;
        mapping->spans = prepared.token_spans;
        mapping->row_identities.reserve(generation.input_embedding_overrides.size());
        for (const auto &row : generation.input_embedding_overrides) mapping->row_identities.push_back(row.cache_identity);
        *saved = std::move(mapping);
    }
    return true;
}

} // namespace qw3::detail
