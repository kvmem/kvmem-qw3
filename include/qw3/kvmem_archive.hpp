#pragma once

// KVMem Context Archive — a durable, immutable, attachable long context.
//
// The archive stores only bytes that cannot be reproduced without re-running
// the model: position-free (content-frame) raw K, V, and a ladder of DeltaNet
// recurrent/conv/hidden snapshots. Everything else — the block table, bake
// positions, GPU page assignment, working set, retrieval scores, and the whole
// retrieval index — is derived when the archive is attached.
//
// This class owns the archive's metadata (manifest, validity bitmaps, token
// stream, ladder files). It deliberately does NOT own `rawk.bin` or `v.bin`:
// those are ordinary `NvmeKvTier` arenas opened in durable + direct-mapped
// mode, so the executor's existing asynchronous tier I/O works unchanged.
// Direct mapping (slot == chunk id for raw K, slot == block id for V) is what
// lets the archive omit any persisted slot table.
//
// See docs/kvmem_context_archive_design.md.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qw3 {

namespace kvmem_archive_files {
inline constexpr const char *kManifest = "manifest.json";
inline constexpr const char *kManifestTmp = "manifest.json.tmp";
inline constexpr const char *kRawK = "rawk.bin";
inline constexpr const char *kV = "v.bin";
inline constexpr const char *kTokens = "tokens.bin";
inline constexpr const char *kChunkValid = "valid.chunks";
inline constexpr const char *kBlockValid = "valid.blocks";
inline constexpr const char *kStateDir = "state";
} // namespace kvmem_archive_files

// Everything that fixes the archive's byte layout or its numerics. An attach
// must match this exactly; a mismatch is an error, never a reinterpretation.
struct KvMemArchiveLayout {
    std::string architecture;
    std::string model_name;      // GGUF basename, for human diagnosis
    uint64_t model_bytes = 0;
    // Lower-case SHA-256 of the complete GGUF.  Format v1/v2 archives predate
    // this field and are accepted through an explicit legacy compatibility
    // path; every newly-built archive requires it.
    std::string model_sha256;
    uint32_t n_layers = 0;
    uint32_t full_attention_interval = 0;
    uint32_t n_standard_layers = 0;
    uint32_t n_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t head_v_dim = 0;
    uint32_t rope_dim = 0;
    double rope_theta = 0.0;
    std::string kv_dtype;        // "fp8" for every supported archive today
    uint32_t block_tokens = 0;
    uint32_t raw_chunk_tokens = 0;
    uint32_t kv_page_size = 0;
    bool immutable_source_k = true;
    bool raw_k_block_major = true;
    bool mtp_archived = false;
    uint64_t raw_k_row_bytes = 0;
    uint64_t raw_chunk_bytes = 0;  // main standard-attention layers only
    uint64_t mtp_chunk_bytes = 0;  // 0 unless mtp_archived
    // Physical direct-mapped record, including the optional MTP V segment.
    uint64_t v_block_bytes = 0;

    // One canonical line; equality of this string is the match test.
    std::string key() const;
    // Empty when compatible, otherwise a human-readable first difference.
    std::string explain_mismatch(const KvMemArchiveLayout &other) const;
};

struct KvMemArchiveManifest {
    uint32_t format_version = 3;
    KvMemArchiveLayout layout;
    // Recorded for provenance only. Every retrieval, budget, tiering, and
    // scheduling knob lives here so it is free to differ between the build and
    // any attach, which is what makes same-context A/B possible.
    std::string policy_snapshot;
    uint64_t total_tokens = 0;
    uint32_t total_blocks = 0;
    uint32_t raw_chunks = 0;
    std::vector<uint64_t> ladder;  // ascending positions that have state files
    bool sealed = false;
    int64_t created_at = 0;
    int64_t sealed_at = 0;
};

// One tensor inside a ladder state file.
struct KvMemArchiveStateEntry {
    enum Kind : uint32_t {
        Hidden = 0,
        MtpPrefixHidden = 1,
        Recurrent = 2,
        Conv = 3,
    };
    uint32_t kind = Hidden;
    uint32_t index = 0;  // layer index for Recurrent/Conv, 0 otherwise
    uint64_t bytes = 0;
};

struct KvMemArchiveState {
    uint64_t position = 0;
    uint64_t kvmem_registered_pos = 0;
    uint32_t mtp_prefix_len = 0;
    std::vector<KvMemArchiveStateEntry> entries;
    std::vector<uint8_t> payload;  // entries concatenated in order
};

class KvMemArchive {
public:
    ~KvMemArchive();
    KvMemArchive(const KvMemArchive &) = delete;
    KvMemArchive &operator=(const KvMemArchive &) = delete;

    // Create a new archive, or reopen an unsealed one for resumption. An
    // existing sealed archive, or one whose layout differs, is an error.
    static std::unique_ptr<KvMemArchive> open_for_build(
        const std::string &dir, const KvMemArchiveLayout &layout,
        const std::string &policy_snapshot);

    // Open a sealed archive read-only.
    static std::unique_ptr<KvMemArchive> attach(const std::string &dir);

    // Read a manifest without opening the payload files.
    static KvMemArchiveManifest read_manifest(const std::string &dir);

    const std::string &dir() const { return dir_; }
    const KvMemArchiveManifest &manifest() const { return manifest_; }
    bool writable() const { return writable_; }
    std::string raw_k_path() const;
    std::string v_path() const;

    // ---- build side ----
    void append_tokens(const uint32_t *tokens, size_t count);
    // Roll back token bytes appended after the last durable ladder point
    // before resuming an interrupted build.
    void truncate_tokens(uint64_t count);
    uint64_t token_count() const { return token_count_; }
    void mark_chunk_valid(uint32_t chunk);
    void mark_block_valid(uint32_t block);
    void reserve_chunks(uint32_t chunks);
    void reserve_blocks(uint32_t blocks);
    void write_state(const KvMemArchiveState &state);
    // Publish progress without sealing, so a crashed build can resume.
    void checkpoint_metadata(uint64_t total_tokens, uint32_t total_blocks,
                             uint32_t raw_chunks);
    void seal(uint64_t total_tokens, uint32_t total_blocks,
              uint32_t raw_chunks);

    // ---- attach side ----
    bool chunk_valid(uint32_t chunk) const;
    bool block_valid(uint32_t block) const;
    std::vector<uint32_t> read_tokens(uint64_t offset, uint64_t count) const;
    KvMemArchiveState read_state(uint64_t position) const;
    // Largest ladder position <= `position`, or UINT64_MAX when none exists.
    uint64_t ladder_at_or_below(uint64_t position) const;
    // Where an interrupted build should restart: the highest ladder position
    // whose raw-K chunks and V blocks are all valid. Zero means start over.
    uint64_t resume_position() const;

private:
    KvMemArchive() = default;
    void write_manifest();
    void write_bitmaps();
    void load_bitmaps();
    std::string state_path(uint64_t position) const;

    std::string dir_;
    KvMemArchiveManifest manifest_;
    bool writable_ = false;
    std::vector<uint8_t> chunk_valid_;
    std::vector<uint8_t> block_valid_;
    uint64_t token_count_ = 0;
    int tokens_fd_ = -1;
};

// Byte length a state payload needs, given per-entry sizes.
uint64_t kvmem_archive_state_bytes(
    const std::vector<KvMemArchiveStateEntry> &entries);

// Compute the complete GGUF SHA-256, using a stat-validated cache under
// $QW3_MODEL_DIGEST_CACHE_DIR, $XDG_CACHE_HOME, or ~/.cache.  The cache avoids
// rereading multi-GiB weights on every archive attach without weakening the
// on-disk archive identity.
std::string kvmem_archive_model_sha256(const std::string &model_path);

} // namespace qw3
