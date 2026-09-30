#pragma once

#include "qw3/kvmem_archive.hpp"
#include "json.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace qw3::detail {

// Host-only validation shared by the archive reader and GPU restore path.
// Offsets refer to the immutable ladder payload, never to process pointers.
struct DFlashArchiveView {
    std::string identity;
    uint64_t next_position = 0;
    uint32_t rows = 0, rope_next = 0, layers = 0, kv_heads = 0, head_dim = 0;
    uint32_t window = 0, max_position = 0, block_size = 0;
    std::vector<size_t> keys, values;
    uint64_t tensor_bytes() const { return uint64_t(rows) * kv_heads * head_dim * sizeof(float); }
};

inline std::optional<DFlashArchiveView> read_dflash_archive(const KvMemArchiveState &state) {
    auto require = [](bool ok, const char *why) {
        if (!ok) throw std::runtime_error(std::string("invalid DFlash archive state: ") + why);
    };
    require(kvmem_archive_state_bytes(state.entries) == state.payload.size(), "payload length");
    size_t offset = 0, metadata_offset = 0, metadata_bytes = 0;
    bool tensors = false;
    for (const auto &e : state.entries) {
        if (e.kind == KvMemArchiveStateEntry::DFlashMetadata) {
            require(metadata_bytes == 0 && e.index == 0 && e.bytes > 0 && e.bytes <= 4096,
                    "duplicate or oversized metadata");
            metadata_offset = offset; metadata_bytes = e.bytes;
        } else if (e.kind == KvMemArchiveStateEntry::DFlashRawK || e.kind == KvMemArchiveStateEntry::DFlashV) {
            tensors = true;
        }
        offset += e.bytes;
    }
    if (!metadata_bytes) {
        require(!tensors, "draft tensors have no metadata");
        return std::nullopt;
    }
    const auto j = nlohmann::json::parse(state.payload.begin() + metadata_offset,
                                       state.payload.begin() + metadata_offset + metadata_bytes);
    auto number = [&](const char *name, uint64_t limit) {
        const auto &value = j.at(name);
        require(value.is_number_unsigned(), "unsigned metadata field required");
        const auto v = value.get<uint64_t>();
        require(v <= limit, "metadata field exceeds limit");
        return v;
    };
    require(number("version", 1) == 1, "metadata version");
    DFlashArchiveView v;
    v.identity = j.at("identity_sha256").get<std::string>();
    require(v.identity.size() == 64 && std::all_of(v.identity.begin(), v.identity.end(),
        [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }), "model fingerprint");
    v.next_position = number("next_position", UINT64_MAX);
    v.rows = number("rows", 8192); v.rope_next = number("rope_next", UINT32_MAX);
    v.layers = number("layers", 128); v.kv_heads = number("kv_heads", 128);
    v.head_dim = number("head_dim", 1024); v.window = number("window", 8192);
    v.max_position = number("max_position", UINT32_MAX); v.block_size = number("block_size", 8192);
    require(v.layers && v.kv_heads && v.head_dim && v.window && v.block_size, "zero shape");
    require(v.next_position == state.position && v.next_position >= v.rows && v.rows <= v.window &&
            v.rows <= v.rope_next && uint64_t(v.rope_next) + v.block_size <= v.max_position &&
            uint64_t(v.window) + v.block_size <= v.max_position, "positions");
    const size_t count = v.rows ? v.layers : 0;
    v.keys.assign(count, SIZE_MAX); v.values.assign(count, SIZE_MAX);
    offset = 0;
    for (const auto &e : state.entries) {
        if (e.kind == KvMemArchiveStateEntry::DFlashRawK || e.kind == KvMemArchiveStateEntry::DFlashV) {
            auto &offsets = e.kind == KvMemArchiveStateEntry::DFlashRawK ? v.keys : v.values;
            require(e.index < count && offsets[e.index] == SIZE_MAX && e.bytes == v.tensor_bytes(),
                    "duplicate, out-of-range, or missized draft tensor");
            offsets[e.index] = offset;
        }
        offset += e.bytes;
    }
    require(std::find(v.keys.begin(), v.keys.end(), SIZE_MAX) == v.keys.end() &&
            std::find(v.values.begin(), v.values.end(), SIZE_MAX) == v.values.end(), "missing draft tensor");
    return v;
}

} // namespace qw3::detail
